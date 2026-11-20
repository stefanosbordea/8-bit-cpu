"""Every primitive of the program generator, executed in the L1 simulator, against the formula of docs/SPEC.md
computed with numpy: on random data and on worst-case operands, with the cycles per row and the overflow flag."""
import numpy as np
import pytest

from lsllm.config import ModelConfig
from lsllm.ls import gen
from lsllm.ls.gen import PAGE, Emitter
from lsllm.ls.iss import L1

VALUES = np.where(np.arange(256) < 128, np.arange(256), np.arange(256) - 256)    # the int8 value at every table index
EXP = np.round(127 * np.exp((VALUES - 127) / 8)).astype(np.int64)
X, Y, W = PAGE + 16, 2 * PAGE + 40, 3 * PAGE + 5         # addresses used by the tests: input, output, weights
ONES = 4 * PAGE - 100                                    # a row of ones that crosses a page end


def rq(acc, sh, relu=False):
    acc = np.asarray(acc, dtype=np.int64)
    assert acc.min() >= -2 ** 23 and acc.max() < 2 ** 23
    t = acc >> sh
    return np.clip(np.maximum(t, 0) if relu else t, -128, 127)


def rqr(acc, sh, relu=False):
    return rq(np.asarray(acc, dtype=np.int64) + ((1 << sh) >> 1), sh, relu)


def run(em, data, size=1 << 16, **state):
    """The CPU after the program of `em` followed by HALT. data: {address: values}. No overflow allowed."""
    em.emit("HALT")
    image = np.zeros(size, dtype=np.int8)
    for address, values in data.items():
        values = np.asarray(values, dtype=np.int64).ravel()
        image[address:address + len(values)] = values.astype(np.uint8).view(np.int8)
    cpu = L1(em.words(), image)
    for name, value in state.items():
        setattr(cpu, name, value)
    cpu.run()
    assert not cpu.overflow
    return cpu


def out(cpu, address, n):
    return np.array(cpu.view[address:address + n], dtype=np.int64)


def per_pass(cpu, em):
    """Cycles of one pass of every marked loop of the program."""
    costs = set()
    for first, last, kind in em.loops:
        total = sum(4 * cpu.count[a] + cpu.elems[a] for a in range(first, last + 1))
        costs.add(total / cpu.count[last])
    return costs


def stream(n):
    """Cycles of n multiply-adds: the classic MAC for one (4), MAC #n (n + 4), two instructions above 255."""
    return 4 if n == 1 else n + 4 if n <= 255 else 259 + stream(n - 255)


@pytest.mark.parametrize("rows, n", [(1, 1), (5, 2), (64, 64), (3, 255), (2, 256), (2, 510), (300, 7), (256, 3)])
def test_linear_rows(rows, n):
    rng = np.random.default_rng(rows * 1000 + n)
    table = rng.integers(-128, 128, 256)
    for case in range(6):
        w, x = rng.integers(-127, 128, (rows, n)), rng.integers(-128, 128, n)
        sh, relu, lut = int(rng.integers(0, 12)), case % 2, 3 if case % 3 == 0 else None
        c = None if case < 3 else int(rng.integers(-32768, 32768))
        em = Emitter()
        em.rows(W, X, Y, rows, n, sh, c=c, relu=relu, lut=lut)
        cpu = run(em, {W: w, X: x, 3 * 256: table})
        expected = rqr(w @ x, sh, relu) if c is None else rq(c + w @ x, sh, relu)
        assert (out(cpu, Y, rows) == (table[expected % 256] if lut else expected)).all()
        assert per_pass(cpu, em) == {20 + stream(n) + 4 * (lut is not None)}      # n + 24 for n = 2..255
        assert out(cpu, Y + rows, 8).tolist() == [0] * 8             # nothing written behind the result
        assert cpu.p == [X + n, W + rows * n, Y + rows]


@pytest.mark.parametrize("w, x, c, sh, expected", [
    (-128, -128, 32767, 0, 127), (-128, -128, 32767, 16, 127), (-128, -128, 32767, 17, 63), (-128, -128, 32767, 23, 0),
    (127, -128, -32768, 0, -128), (127, -128, -32768, 16, -128), (127, -128, -32768, 17, -64), (127, -128, -32768, 23, -1),
    (0, 0, -1, 0, -1), (0, 0, -1, 23, -1), (0, 0, 127, 0, 127), (0, 0, 128, 0, 127), (0, 0, -129, 0, -128)])
def test_linear_rows_with_the_largest_accumulator(w, x, c, sh, expected):
    """510 products of the same sign plus the largest preload: |ACC| reaches 2^23 - 1 and does not overflow."""
    em = Emitter()
    em.rows(W, X, Y, 2, 510, sh, c=c)
    cpu = run(em, {W: np.full(1020, w), X: np.full(510, x)})
    assert out(cpu, Y, 2).tolist() == [expected, expected]
    assert cpu.acc == c + 510 * w * x and abs(cpu.acc) < 2 ** 23
    em = Emitter()
    em.rows(W, X, Y, 1, 510, sh, c=c, relu=1)
    assert out(run(em, {W: np.full(510, w), X: np.full(510, x)}), Y, 1) == max(expected, 0)


@pytest.mark.parametrize("rows, n", [(1, 1), (64, 64), (64, 172), (3, 300), (2, 510), (260, 4)])
def test_residual_rows(rows, n):
    rng = np.random.default_rng(rows + n)
    for case in range(4):
        scale = (127, 30, 8, 1)[case]                                # large weights saturate, small ones round
        w, h, x = rng.integers(-scale, scale + 1, (rows, n)), rng.integers(-128, 128, n), rng.integers(-128, 128, rows)
        em = Emitter()
        em.residual(W, X, Y, rows, n)
        cpu = run(em, {W: w, X: h, Y: x})
        assert (out(cpu, Y, rows) == rq(256 * x + 128 + w @ h, 8)).all()
        assert per_pass(cpu, em) == {24 + stream(n)}                 # n + 28 for n = 2..255
        assert (out(cpu, X, n) == h).all()


@pytest.mark.parametrize("x, w, h, expected", [(127, 127, 127, 127), (-128, 127, -128, -128), (127, -128, 127, -128),
                                              (-128, -128, -128, 127), (0, 0, 0, 0), (-1, 0, 0, -1), (127, 0, 5, 127)])
def test_residual_rows_worst_case(x, w, h, expected):
    em = Emitter()
    em.residual(W, X, Y, 3, 509)                                     # SPEC B.4: preload 256 x + 128 and n = 509
    cpu = run(em, {W: np.full(3 * 509, w), X: np.full(509, h), Y: [x] * 3})
    assert out(cpu, Y, 3).tolist() == [expected] * 3 and cpu.acc == 256 * x + 128 + 509 * w * h


@pytest.mark.parametrize("n", [1, 256, 300])
def test_table_map(n):
    rng = np.random.default_rng(n)
    table = rng.integers(-128, 128, 256)
    x = np.resize(rng.permutation(VALUES), n)                        # every int8 value when n >= 256
    em = Emitter()
    em.table_map(X, Y, n, 9)
    cpu = run(em, {X: x, 9 * 256: table})
    assert (out(cpu, Y, n) == table[x % 256]).all() and per_pass(cpu, em) == {16}


@pytest.mark.parametrize("sh", [0, 1, 7, 14, 15])
def test_elementwise_product(sh):
    rng = np.random.default_rng(sh)
    a = np.concatenate([[-128, -128, 127, 127, 0, 1, -1], rng.integers(-128, 128, 293)])
    b = np.concatenate([[-128, 127, -128, 127, 5, -1, -1], rng.integers(-128, 128, 293)])
    em = Emitter()
    em.product(X, W, Y, 300, sh)
    cpu = run(em, {X: a, W: b})
    assert (out(cpu, Y, 300) == rqr(a * b, sh)).all() and per_pass(cpu, em) == {20}


@pytest.mark.parametrize("n", [1, 2, 64, 256, 510])
def test_norm_rms(n):
    rng = np.random.default_rng(n)
    table = np.concatenate([rng.integers(1, 128, 128), np.zeros(128)]).astype(np.int64)
    cases = [(rng.integers(-128, 128, n), int(rng.integers(0, 24)), int(rng.integers(0, 15))) for case in range(8)]
    cases += [(np.full(n, -128), sa, sb) for sa in (0, 16, 23) for sb in (0, 7, 14)]             # the largest sum of squares
    cases += [(np.zeros(n), 0, 3), (rng.integers(-3, 4, n), 0, 0)]
    for case, (x, sa, sb) in enumerate(cases):
        cell = (X - 1, PAGE - 2)[case % 2]                           # in the page of x, or in the page before it
        em = Emitter()
        em.rms(X, Y, n, 2, sa, sb, cell)
        cpu = run(em, {X: x, 2 * 256: table})
        m = rq((x * x).sum(), sa)
        assert 0 <= m <= 127
        assert (out(cpu, Y, n) == rqr(x * table[m], sb)).all()
        assert per_pass(cpu, em) == {24} and cpu.d[cell] == table[m]


def test_norm_dyt_is_the_table_map_of_site_n():
    cfg = ModelConfig(vocab_size=8, dim=8, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=4, max_seq_len=4)
    from lsllm.ls.cost import dummy_model
    for norm, name in (("dyt", "LUT"), ("rms", "MAC")):
        b = gen.build(*dummy_model(cfg, {"norm": norm, "ffn": "relu"}))
        tables = [args[1] for name_, args, tag in b.code if tag == "norm" and name_ == "LUT"]
        assert tables == [0, 1, 2]                                   # sites 0, 1 and the final one, in this order
        assert any(name_ == name for name_, args, tag in b.code if tag == "norm")


@pytest.mark.parametrize("d", [2, 8, 32])
def test_rope_of_one_head(d):
    rng = np.random.default_rng(d)
    cfg = ModelConfig(dim=2 * d, n_heads=2, n_kv_heads=1, max_seq_len=40)
    theta = 10000.0 ** (-np.arange(0, d, 2) / d)
    angle = np.outer(np.arange(40), theta)
    C, S = np.round(127 * np.cos(angle)).astype(np.int64), np.round(127 * np.sin(angle)).astype(np.int64)
    rows = gen.records(cfg, 0, 0, C, S)
    heads = [rng.integers(-128, 128, d) for case in range(6)] + [np.full(d, -128), np.full(d, 127), np.resize([127, -128], d)]
    for case, head in enumerate(heads):
        p = (0, 1, 39, 17, 5, 22, 3, 9, 30)[case]
        em = Emitter()
        if case % 2:                                                 # the address of the head is read from two bytes
            at = X + 7 * case
            em.rope(X, d, X + 100, Y, fields=X - 2)
            cpu = run(em, {at: head, X + 100: rows[p, 9:], X - 2: [at >> 8 & 15, at & 255]})
        else:
            em.rope(X, d, X + 100, Y)
            cpu = run(em, {X: head, X + 100: rows[p, 9:]})
        a, b = head[0::2], head[1::2]
        expected = np.concatenate([rqr(a * C[p] - b * S[p], 7), rqr(a * S[p] + b * C[p], 7)])
        assert (out(cpu, Y, d) == expected).all() and per_pass(cpu, em) == {22}
    table = np.concatenate([np.resize([127, -127], d), np.resize([-127, 127], d)])   # not a rotation: the largest |ACC|
    for head in (np.full(d, -128), np.resize([-128, 127], d), np.resize([127, -128], d)):
        em = Emitter()
        em.rope(X, d, X + 100, Y)
        cpu = run(em, {X: head, X + 100: table})
        pairs = head.reshape(-1, 2)
        sums = np.concatenate([(pairs * [127, -127]).sum(1), (pairs * [-127, 127]).sum(1)])
        assert (out(cpu, Y, d) == rqr(sums, 7)).all() and abs(sums).max() in (0, 32385)


@pytest.mark.parametrize("d, p", [(2, 0), (8, 0), (8, 1), (8, 99), (8, 255), (32, 255), (16, 200)])
def test_scores_maximum_and_weights(d, p):
    """SPEC C.6, steps 1 to 3."""
    rng = np.random.default_rng(d + p)
    keys, count, fields, q_at = 4 * PAGE + 2304, X - 3, X + 80, X + 1
    relu = np.where(np.arange(256) < 128, np.arange(256), 0)
    for case in range(8):
        K, q = rng.integers(-128, 128, (256, d)), rng.integers(-128, 128, d)
        sh, c = int(rng.integers(0, 12)), int(rng.integers(-32768, 32768))
        table = np.where(VALUES <= 0, np.round(127 * np.exp(VALUES / 4.0)), 0).astype(np.int64)
        if case == 3:                                                # the largest sums: all products 16,384, all codes 127
            K, q, sh, c = np.full((256, d), -128), np.full(d, -128), 0, 32767
        if case == 4:                                                # every code 0: the maximum is 0, every weight 127
            sh, c = 0, -32768
            K = -np.abs(np.maximum(K, -127)) * np.sign(q)            # every product is <= 0
        if case == 5:                                                # small scores: neighbouring codes, many ties
            sh, c = 9, 20000
        if case >= 6:                                                # any table gives the same bytes
            table = rng.integers(-128, 128, 256)
        em = Emitter()
        em.scores(fields, q_at, Y, d, sh, count, 6, 7)
        cpu = run(em, {keys: K, q_at: q, q_at - 1: [c & 255], fields: [keys >> 12, keys >> 8 & 15, keys & 255, c >> 8 & 255],
                       count: [(p + 1) & 255], 6 * 256: relu, 7 * 256: table})
        code = rq(c + K[:p + 1] @ q, sh, relu=True)
        assert (out(cpu, Y, p + 1) == table[(code - code.max()) % 256]).all() and cpu.r[2] == code.max()
        assert per_pass(cpu, em) == {d + 40, 20}                     # code and maximum, then the weight
        assert out(cpu, Y + p + 1, 4).tolist() == [0] * 4
        assert [cpu.count[last] for first, last, kind in em.loops] == [p + 1, p + 1]
        assert (code.max() == 0) if case == 4 else (code.min() == 127) if case == 3 else True


def value_rows(d, block):
    """Address of V[j] for j < d in a cache block at `block`: the rule in the text of gen.kv_layout."""
    chunks, keys, stride = gen.kv_layout(d, 256)
    return [block + PAGE * (j // 16) + 256 * ((min(16, d - 16 * (j // 16)) - j % 16) % 16) for j in range(d)], chunks


@pytest.mark.parametrize("d, T", [(8, 256), (16, 256), (32, 256), (24, 256), (8, 16), (4, 40), (2, 256)])
def test_value_mixing_and_division(d, T):
    """SPEC C.6, steps 4 and 5."""
    rng = np.random.default_rng(d + T)
    block, e_at, fields, ks = 5 * PAGE, X + 300, X - 8, set()
    where, chunks = value_rows(d, block)
    assert len(set(where)) == d
    recip = np.minimum(127, np.round(8192 / np.maximum(np.arange(256), 1))).astype(np.int64)
    for p in (0, 1, T // 2, T - 1):
        for case in ("random", "random", "largest", "one", "small", "table"):
            V, e, sh, table = rng.integers(-128, 128, (d, T)), rng.integers(0, 128, T), int(rng.integers(0, 16)), recip
            e[rng.integers(0, p + 1)] = 127                          # the key with the largest code
            if case == "largest":                                    # the largest sums: p + 1 products of -16,256
                V, e = np.full((d, T), -128), np.full(T, 127)
            if case == "one":                                        # one key has all the weight: Dsum = 127, k = 7
                e = np.where(np.arange(T) == p // 2, 127, 0)
            if case == "small":                                      # the other weights are 0 or 1
                e = np.where(np.arange(T) == p, 127, rng.integers(0, 2, T))
            if case == "table":                                      # any table gives the same bytes
                table = rng.integers(-128, 128, 256)
            em = Emitter()
            em.mix(fields, e_at, Y, ONES, chunks, sh, 9, T == 256)
            data = {e_at: e, ONES: np.ones(T), fields: [block >> 12, Y >> 8 & 15, Y & 255], 9 * 256: table,
                    **{address: row for address, row in zip(where, V)}}
            cpu = run(em, data, len=p + 1)
            dsum = int(e[:p + 1].sum())
            k = dsum.bit_length()
            r = table[dsum >> (k - 7)]
            a = rq(V[:, :p + 1] @ e[:p + 1], k)
            assert 7 <= k <= 15 and 64 <= dsum >> (k - 7) <= 127 and np.abs(V[:, :p + 1] @ e[:p + 1]).max() <= 128 * dsum
            assert (out(cpu, Y, d) == rqr(a * r, sh)).all()
            assert out(cpu, e_at + p + 1, 3).tolist() == [k, r, block // PAGE + len(chunks) - 1]
            assert per_pass(cpu, em) <= {p + 57, p + 65}
            assert (p + 65 in per_pass(cpu, em)) == (T == 256 and d >= 15)
            ks.add(k)
    assert {7, 15} <= ks if T == 256 else 7 in ks                    # 15 bits: 127 * 256 = 32,512


def test_value_mixing_costs_p_plus_57_per_row_and_one_stream_for_the_sum():
    cycles = []
    for p in (9, 10):
        em = Emitter()
        em.mix(X - 8, X + 300, Y, ONES, [(0, 8)], 7, 9, False)
        cycles.append(run(em, {X + 300: [127] * 16, ONES: [1] * 16, X - 8: [5, Y >> 8 & 15, Y & 255]}, len=p + 1).cycles)
    assert cycles[1] - cycles[0] == 9                                # one more key: one more product in 9 streams
    assert cycles[0] == 8 * (9 + 57) + (9 + 5) + 4 * 26              # 8 rows, the stream of the sum, 26 instructions


def test_one_piece_of_program_runs_for_every_record_of_a_chain():
    """Emitter.for_records: three records in any order in memory, the last has page 0 as its link."""
    first, second, third, cell = 3 * PAGE + 100, 2 * PAGE + 7, 3 * PAGE + 50, X
    em = Emitter()

    def body():                                                      # send the fourth and fifth byte of the record
        em.point(0, cell + 3)
        em.emit("LD", 0, 0, 1)
        em.emit("TX", 0)
        em.emit("LD", 0, 0, 1)
        em.emit("TX", 0)
        em.at(0, cell + 5)
    em.for_records(first, cell, 5, body)
    em.emit("TX", 3)                                                 # R3 = 0 after the last copy
    link = lambda address: [address >> 8 & 15, address & 255, address >> 12]
    cpu = run(em, {first: link(second) + [1, 2], second: link(third) + [3, 4], third: [0, 0, 0, 5, 6]})
    assert cpu.tx == [1, 2, 3, 4, 5, 6, 0]
    jump = next(a for a, (name, args, tag) in enumerate(em.code) if name == "JMP")
    inside = range(em.code[jump][1][0], jump + 1)
    # per record: 12 cycles per byte, 14 instructions and the 5 of the body; the last one skips MVP, MVH and JMP
    assert sum(4 * cpu.count[a] for a in inside) == 3 * (5 * 12 + 4 * (14 + 5)) - 4 * 3


@pytest.mark.parametrize("d", [2, 8, 16, 24, 40])
def test_cache_write_of_the_values(d):
    rng = np.random.default_rng(d)
    block, pos = 5 * PAGE, X - 2
    where, chunks = value_rows(d, block)
    for p in (0, 7, 255):
        v = rng.integers(-128, 128, d)
        em = Emitter()
        em.scatter(X, [(block // PAGE + page, rows) for page, rows in chunks], pos)
        cpu = run(em, {X: v, pos: [p]})
        memory = out(cpu, block, len(chunks) * PAGE)
        assert [memory[address - block + p] for address in where] == v.tolist()
        assert np.count_nonzero(memory) == np.count_nonzero(v)       # nothing else was written


@pytest.mark.parametrize("d, T", [(8, 256), (16, 256), (32, 256), (24, 100), (2, 256)])
def test_key_pointer_from_the_position_record(d, T):
    cfg = ModelConfig(dim=2 * d, n_heads=2, n_kv_heads=1, max_seq_len=T)
    chunks, keys, stride = gen.kv_layout(d, T)
    assert keys >= 256 * d and keys + T * d <= stride + (256 if d < 16 else 0)       # the keys do not touch value rows
    block = 7 * PAGE
    rows = gen.records(cfg, 0, keys % PAGE, np.zeros((T, d // 2)), np.zeros((T, d // 2)))
    for p in (0, 1, T // 2, T - 1):
        em = Emitter()
        em.key_pointer(X, (block + keys) // PAGE)
        cpu = run(em, {X: rows[p, 6:9]})
        assert cpu.p[2] == block + keys + p * d


def test_position_records_link_to_the_next_and_end_with_page_0():
    cfg = ModelConfig(dim=16, n_heads=2, n_kv_heads=1, max_seq_len=256)
    rows = gen.records(cfg, 0x12345, 2304, np.ones((256, 4)), 2 * np.ones((256, 4)))
    for p in range(255):
        nibble, low, page = rows[p, :3]
        assert (page << 12 | nibble << 8 | low) == 0x12345 + (p + 1) * 25
        assert rows[p, 3] + 256 * rows[p, 4] == p + 1 and rows[p, 5] == p
    assert rows[255, :6].tolist() == [0, 0, 0, 0, 1, 255]
    assert rows[9, 9:].tolist() == [1, -2] * 4 + [2, 1] * 4


def argmax(u):
    """(token chosen by the argmax primitive, cycles of the pass)."""
    em = Emitter()
    em.argmax(Y, len(u), X, 5)
    cpu = run(em, {Y: u, X: [77], 5 * 256: np.where(np.arange(256) < 128, np.arange(256), 0)})
    assert cpu.r[1] == max(u)
    return cpu.d[X] * 256 + (-cpu.r[2]) % 256, cpu.cycles


@pytest.mark.parametrize("n", [1, 2, 5, 255, 256, 257, 300, 512, 700])
def test_argmax_random_and_ties(n):
    rng = np.random.default_rng(n)
    cases = [rng.integers(0, 128, n) for case in range(20)] + [rng.integers(0, 3, n) for case in range(10)]    # many ties
    cases += [np.zeros(n), np.full(n, 127), np.full(n, 60), np.arange(n) % 128, 127 - np.arange(n) % 128]
    for place in {0, n - 1, n // 2, min(n - 1, 255), min(n - 1, 256), min(n - 1, 511), min(n - 1, 512)}:
        u = rng.integers(0, 100, n)
        u[place:] = np.minimum(u[place:], 99)
        u[place] = 100 + int(rng.integers(0, 28))                    # the only maximum
        cases.append(u)
        cases.append(np.where(np.arange(n) >= place, u[place], 0))   # ties after it: the first one wins
    for u in cases:
        token, cycles = argmax(u.astype(np.int64).tolist())
        assert token == int(np.argmax(u))


def test_argmax_costs_20_cycles_per_logit_and_16_per_new_maximum():
    base = argmax([0] * 512)[1]
    assert argmax([0] * 256)[1] == base - 256 * 20 - 4               # one block less: one LDI R3 less
    assert argmax([5] + [0] * 511)[1] == base + 16
    assert argmax([min(g, 127) for g in range(512)])[1] == base + 127 * 16           # the most a pass can cost
    assert argmax([0] * 300)[1] - argmax([0] * 299)[1] == 20
    assert argmax([0] * 299 + [1])[1] - argmax([0] * 300)[1] == 24   # 2 more instructions in a short last block


def test_output_layer_then_argmax():
    rng = np.random.default_rng(0)
    for case in range(5):
        E, y = rng.integers(-127, 128, (300, 24)), rng.integers(-128, 128, 24)
        sh, c = int(rng.integers(5, 10)), int(rng.integers(-3000, 3000))
        em = Emitter()
        em.rows(W, X, Y, 300, 24, sh, c=c, relu=1)
        em.argmax(Y, 300, X - 4, 5)
        cpu = run(em, {W: E, X: y, 5 * 256: np.where(np.arange(256) < 128, np.arange(256), 0)})
        u = rq(c + E @ y, sh, relu=True)
        assert (out(cpu, Y, 300) == u).all() and u.min() >= 0
        assert cpu.d[X - 4] * 256 + (-cpu.r[2]) % 256 == int(np.argmax(u))


def test_the_emitter_sets_a_pointer_only_when_it_has_to():
    em = Emitter()
    em.point(1, 5 * PAGE + 7)
    assert [name for name, args, tag in em.code] == ["LDI", "MVH", "SETP1"]
    em.point(1, 5 * PAGE + 7)
    em.point(1, 5 * PAGE + 9)
    assert [name for name, args, tag in em.code[3:]] == ["SETP1"]
    em.point2(5 * PAGE + 9)
    em.point2(5 * PAGE + 9)
    assert [name for name, args, tag in em.code[4:]] == ["LDI", "MVH", "SETP0", "CPP20"]
    em.emit("MAC", 3)                                                # a stream moves P0 and P1: nothing is known
    em.point(0, 5 * PAGE + 9)
    assert [name for name, args, tag in em.code[9:]] == ["LDI", "MVH", "SETP0"]
    em.point2(5 * PAGE + 9)
    em.emit("ST2", 0, 1)                                             # a store with post-increment moves P2
    em.point2(5 * PAGE + 9)
    assert [name for name, args, tag in em.code[12:]] == ["ST2", "CPP20"]
    with pytest.raises(AssertionError):
        Emitter().rows(W, 2 * PAGE - 64, Y, 1, 64, 0)                # the rewound vector would end at the page end


def test_stream_lengths():
    for n, expected in ((1, [0]), (2, [2]), (255, [255]), (256, [255, 0]), (257, [255, 2]), (510, [255, 255])):
        em = Emitter()
        em.stream(n)
        assert [args[0] for name, args, tag in em.code] == expected


def test_listing_uses_the_notation_of_the_design_document():
    shown = [gen.text(*insn) for insn in (("LD", (0, 1, 1)), ("ST2", (3, 0)), ("LD2", (1, 1)), ("MAC", (0,)), ("MAC", (64,)),
                                          ("CPP20", ()), ("MVP", (2, 3, 1)), ("MVH", (1, 0)), ("SETP0", (0x31C,)),
                                          ("RQ", (0, 7, 1)), ("LUT", (0, 11)), ("MACL", ()), ("DJNZ", (5,)),
                                          ("NRM", (2,)), ("RQV", (0, 2, 0)))]
    assert shown == ["LD R0, [P1]+", "ST R3, [P2]", "LD R1, [P2]+", "MAC", "MAC #64", "CPP P2, P0", "MVP P2, R3, R1",
                     "MVH P1, R0", "SETP0 0x31C", "RQ R0, 7, 1", "LUT R0, 11", "MACL", "DJNZ 0x005", "NRM R2",
                     "RQV R0, R2, 0"]
    from lsllm.ls.cost import dummy_model
    cfg = ModelConfig(vocab_size=8, dim=8, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=4, max_seq_len=4)
    b = gen.build(*dummy_model(cfg, {"norm": "dyt", "ffn": "relu"}))
    lines = gen.listing(b).splitlines()
    assert len(lines) == len(b.words) and lines[-1].split()[:3] == ["%03X" % (len(lines) - 1), "0001", "HALT"]
    assert {line.split(";")[1].strip() for line in lines} <= set(gen.COMPONENTS)


def test_a_rounded_shift_above_15_is_refused():
    with pytest.raises(ValueError, match="16 bits"):
        Emitter().rows(W, X, Y, 1, 4, 16)
    Emitter().rows(W, X, Y, 1, 4, 16, c=0)                           # with an explicit preload any shift is fine
    Emitter().rows(W, X, Y, 1, 4, 15)
