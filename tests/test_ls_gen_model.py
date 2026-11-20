"""The generated LS program against a numpy implementation of SPEC Part C written here, and against lsllm.intref:
same tokens, same logits, same intermediate vectors, same KV cache, same UART bytes, no accumulator overflow."""
import numpy as np
import pytest

from lsllm.config import ModelConfig
from lsllm.ls import gen

SMALL = dict(vocab_size=32, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, hidden_dim=24, max_seq_len=16)
VALUES = np.where(np.arange(256) < 128, np.arange(256), np.arange(256) - 256)    # the int8 value at every table index


def random_model(seed, norm, ffn, wild=False, **shape):
    """(spec, arrays) of SPEC Part E with random content. wild: full-range weights and random tables, so that
    everything saturates; otherwise values that keep the activations spread over the int8 range."""
    rng = np.random.default_rng(seed)
    cfg = ModelConfig(**shape)
    D, L, H, F, V, T, d = cfg.dim, cfg.n_layers, cfg.n_heads, cfg.hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.head_dim
    Gd, sites = cfg.kv_dim, 2 * L + 1

    def weights(*size):
        w = rng.integers(-127, 128, size) if wild else np.round(rng.normal(0, 24, size))
        return np.clip(w, -127, 127).astype(np.int8)

    arrays = {"E": weights(V, D)}
    for l in range(L):
        shapes = {"Wq": (D, D), "Wk": (Gd, D), "Wv": (Gd, D), "Wo": (D, D), "W1": (F, D), "W3": (F, D), "W2": (D, F)}
        for name, size in shapes.items():
            if name != "W3" or ffn == "swiglu":
                arrays["%s_%d" % (name, l)] = weights(*size)
    theta = 10000.0 ** (-np.arange(0, d, 2) / d)
    angle = np.outer(np.arange(T), theta)
    arrays["ROPE_C"], arrays["ROPE_S"] = np.round(127 * np.cos(angle)), np.round(127 * np.sin(angle))
    arrays["EXP"] = np.where(VALUES <= 0, np.round(127 * np.exp(VALUES / rng.uniform(3, 6, (L, 1)))), 0)   # C.6, step 3
    arrays["RECIP"] = np.minimum(127, np.round(8192 / np.maximum(np.arange(256), 1)))
    arrays["SILU"] = np.clip(np.round(VALUES / (1 + np.exp(-VALUES / 16.0))), -127, 127)
    if norm == "dyt":
        arrays["NORM"] = np.round(np.outer(rng.integers(60, 127, sites), np.ones(256)) * np.tanh(VALUES / 40.0))
    else:
        arrays["NORM"] = np.clip(np.round(48 * np.sqrt(64 / (np.maximum(VALUES, 0) + 0.5))), 1, 127) * np.ones((sites, 1))
    if wild:
        arrays["EXP"] = rng.integers(0, 128, (L, 256))
        arrays["EXP"][:, 0] = 127                                # the one property the program relies on: Dsum >= 127
        arrays["RECIP"] = rng.integers(-128, 128, 256)
        arrays["SILU"] = rng.integers(-127, 128, 256)
        arrays["NORM"] = rng.integers(1 if norm == "rms" else -127, 128, (sites, 256))
    low, high = (0, 16) if wild else (5, 9)
    shifts = {name: rng.integers(low, high, L).tolist() for name in ("sh_q", "sh_k", "sh_v", "sh_s", "sh_o", "sh_1", "sh_3", "sh_g")}
    shifts["sh_l"] = int(rng.integers(low, high))
    shifts["sa"] = rng.integers(0 if wild else 8, 24 if wild else 11, sites).tolist()
    shifts["sb"] = rng.integers(low, high, sites).tolist()
    centre = np.array([60 << s for s in shifts["sh_s"]])[:, None]
    c_s = rng.integers(-32768, 32768, (L, H)) if wild else centre + rng.integers(-2000, 2000, (L, H))
    c_l = int(rng.integers(-32768, 32768)) if wild else (40 << shifts["sh_l"]) + int(rng.integers(-500, 500))
    spec = {"config": cfg.to_dict(), "options": {"norm": norm, "ffn": ffn}, "frac": {"u": 4},
            "shifts": shifts, "preloads": {"c_s": np.clip(c_s, -32768, 32767).tolist(), "c_l": c_l}, "tokenizer": ""}
    return spec, {name: np.asarray(value).astype(np.int8) for name, value in arrays.items()}


def rq(acc, sh, relu=False):
    """SPEC B.2, with the 24-bit check."""
    acc = np.asarray(acc, dtype=np.int64)
    assert acc.min() >= -2 ** 23 and acc.max() < 2 ** 23
    t = acc >> sh
    return np.clip(np.maximum(t, 0) if relu else t, -128, 127)


def rqr(acc, sh, relu=False):
    return rq(np.asarray(acc, dtype=np.int64) + ((1 << sh) >> 1), sh, relu)


class Reference:
    """SPEC Part C for one sequence, one token at a time, with the KV cache. `t` keeps the vectors of the last step."""

    def __init__(self, spec, arrays):
        self.cfg, self.opt, self.sh, self.pre = ModelConfig(**spec["config"]), spec["options"], spec["shifts"], spec["preloads"]
        self.A = {name: value.astype(np.int64) for name, value in arrays.items()}
        cfg = self.cfg
        self.K = np.zeros((cfg.n_layers, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim), dtype=np.int64)    # K[l][g][p][j]
        self.V = np.zeros((cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, cfg.max_seq_len), dtype=np.int64)    # V[l][g][j][p]
        self.p = 0
        self.shifts = set()                                      # every k that a head has used

    def lut(self, table, v):
        return table[np.asarray(v) % 256]

    def norm(self, n, x):
        table = self.A["NORM"][n]
        if self.opt["norm"] == "dyt":
            return self.lut(table, x)
        r = self.lut(table, rq((x * x).sum(), self.sh["sa"][n]))
        return rqr(x * r, self.sh["sb"][n])

    def rope(self, head):
        a, b = head[0::2], head[1::2]
        c, s = self.A["ROPE_C"][self.p], self.A["ROPE_S"][self.p]
        return np.concatenate([rqr(a * c - b * s, 7), rqr(a * s + b * c, 7)])

    def step(self, tok):
        cfg, A, sh, p, t = self.cfg, self.A, self.sh, self.p, {}
        d, share = cfg.head_dim, cfg.n_heads // cfg.n_kv_heads
        x = A["E"][tok].copy()
        for l in range(cfg.n_layers):
            y = self.norm(2 * l, x)
            q = rqr(A["Wq_%d" % l] @ y, sh["sh_q"][l])
            k = rqr(A["Wk_%d" % l] @ y, sh["sh_k"][l])
            v = rqr(A["Wv_%d" % l] @ y, sh["sh_v"][l])
            o = np.zeros(cfg.dim, dtype=np.int64)
            for g in range(cfg.n_kv_heads):
                self.K[l, g, p] = self.rope(k[g * d:(g + 1) * d])
                self.V[l, g, :, p] = v[g * d:(g + 1) * d]
            for h in range(cfg.n_heads):
                g = h // share
                qh = self.rope(q[h * d:(h + 1) * d])
                code = rq(self.pre["c_s"][l][h] + self.K[l, g, :p + 1] @ qh, sh["sh_s"][l], relu=True)
                e = self.lut(A["EXP"][l], code - code.max())     # SPEC C.6, steps 2 and 3
                dsum = int(e.sum())
                k = dsum.bit_length()                            # step 5: 7..15 significant bits
                assert 7 <= k <= 15
                r = self.lut(A["RECIP"], dsum >> (k - 7))
                o[h * d:(h + 1) * d] = rqr(rq(self.V[l, g, :, :p + 1] @ e, k) * r, sh["sh_o"][l])
                t["l%d.h%d.e" % (l, h)] = e
                self.shifts.add(k)
            x = rq(256 * x + 128 + A["Wo_%d" % l] @ o, 8)
            y = self.norm(2 * l + 1, x)
            if self.opt["ffn"] == "swiglu":
                g1 = self.lut(A["SILU"], rqr(A["W1_%d" % l] @ y, sh["sh_1"][l]))
                gate = rqr(g1 * rqr(A["W3_%d" % l] @ y, sh["sh_3"][l]), sh["sh_g"][l])
            else:
                gate = rqr(A["W1_%d" % l] @ y, sh["sh_1"][l], relu=True)
            x = rq(256 * x + 128 + A["W2_%d" % l] @ gate, 8)
        t["x"], t["o"], t["g"] = x, o, gate
        t["y"] = self.norm(2 * cfg.n_layers, x)
        t["u"] = rq(self.pre["c_l"] + A["E"] @ t["y"], sh["sh_l"], relu=True)
        self.p, self.t = p + 1, t
        return t["u"]


def cache(cpu, b, cfg):
    """(K, V) as the program left them in the data memory, in the shapes of Reference."""
    L, G, T, d = cfg.n_layers, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim
    K = [[gen.peek(cpu, b, "kv.K%d.%d" % (l, g)).reshape(T, d) for g in range(G)] for l in range(L)]
    V = [[[gen.peek(cpu, b, "kv.V%d.%d.%d" % (l, g, j)) for j in range(d)] for g in range(G)] for l in range(L)]
    return np.array(K), np.array(V)


def generate(spec, arrays, tokens, forced=None):
    """Run the program and the reference side by side for `tokens` tokens and check everything after each one.
    Greedy from token 1, or fed with the token ids `forced` (the choice of the program is still checked)."""
    b = gen.build(spec, arrays)
    cpu, ref, cfg = gen.machine(b), Reference(spec, arrays), ModelConfig(**spec["config"])
    tok, out = 1, []
    for step in range(tokens):
        tok = tok if forced is None else forced[step]
        u = ref.step(tok)
        chosen = gen.run_token(cpu, b, tok)
        for name in ("x", "y", "o", "g", "u"):
            assert (gen.peek(cpu, b, "scratch." + name) == ref.t[name]).all(), (step, name)
        last = "l%d.h%d.e" % (cfg.n_layers - 1, cfg.n_heads - 1)
        assert (gen.peek(cpu, b, "scratch.e")[:step + 1] == ref.t[last]).all()
        K, V = cache(cpu, b, cfg)
        assert (K == ref.K).all() and (V == ref.V).all()
        tok = int(np.argmax(u))                                  # numpy: the first of the largest
        assert chosen == tok
        out.append(tok)
    assert not cpu.overflow
    assert cpu.tx == [byte for tok in out for byte in (0x80 | tok >> 7, tok & 0x7F)]
    return out, cpu, b


@pytest.mark.parametrize("norm", ["dyt", "rms"])
@pytest.mark.parametrize("ffn", ["swiglu", "relu"])
@pytest.mark.parametrize("wild", [False, True])
def test_twelve_greedy_tokens_equal_the_numpy_reference(norm, ffn, wild):
    out, cpu, b = generate(*random_model(7, norm, ffn, wild, **SMALL), tokens=12)
    assert len(out) == 12


def test_the_smallest_and_the_largest_sum_of_weights():
    """One key (Dsum = 127, k = 7), and 256 keys that all have weight 127 (Dsum = 32,512, k = 15)."""
    for p, k in ((0, 7), (255, 15)):
        spec, arrays = random_model(4, "dyt", "relu", **dict(SMALL, max_seq_len=256))
        arrays["EXP"][:] = 127
        b = gen.build(spec, arrays)
        cpu, ref = gen.machine(b), Reference(spec, arrays)
        ref.p = p
        ref.V[:, :, :, :p] = np.random.default_rng(p).integers(-128, 128, ref.V[:, :, :, :p].shape)
        for l in range(2):
            for j in range(8):
                address, size = b.mem["kv.V%d.0.%d" % (l, j)]
                cpu.view[address:address + size] = ref.V[l, 0, j]
        gen.set_position(cpu, b, p)
        u = ref.step(5)
        assert gen.run_token(cpu, b, 5) == int(np.argmax(u)) and ref.shifts == {k}
        assert (gen.peek(cpu, b, "scratch.u") == u).all() and (gen.peek(cpu, b, "scratch.o") == ref.t["o"]).all()
        assert not cpu.overflow


@pytest.mark.parametrize("norm", ["dyt", "rms"])
@pytest.mark.parametrize("ffn", ["swiglu", "relu"])
def test_a_full_context_of_random_tokens_then_halt(norm, ffn):
    spec, arrays = random_model(3, norm, ffn, **SMALL)
    forced = np.random.default_rng(5).integers(0, 32, 16).tolist()
    out, cpu, b = generate(spec, arrays, tokens=16, forced=forced)
    assert len(set(out)) > 4                                     # the choices differ from token to token
    cpu.run(stop=0)                                              # position 16 does not exist: the program stops
    assert cpu.halted and len(cpu.tx) == 32


def at_position(spec, arrays, p, tokens, seed=0):
    """As `generate`, but starting at position p with random int8 contents in the cache of the positions before."""
    rng = np.random.default_rng(seed)
    b = gen.build(spec, arrays)
    cpu, ref, cfg = gen.machine(b), Reference(spec, arrays), ModelConfig(**spec["config"])
    ref.p = p
    ref.K[:, :, :p] = rng.integers(-128, 128, ref.K[:, :, :p].shape)
    ref.V[:, :, :, :p] = rng.integers(-128, 128, ref.V[:, :, :, :p].shape)
    for l in range(cfg.n_layers):
        for g in range(cfg.n_kv_heads):
            address, size = b.mem["kv.K%d.%d" % (l, g)]
            cpu.view[address:address + size] = ref.K[l, g].ravel()
            for j in range(cfg.head_dim):
                address, size = b.mem["kv.V%d.%d.%d" % (l, g, j)]
                cpu.view[address:address + size] = ref.V[l, g, j]
    gen.set_position(cpu, b, p)
    for tok in rng.integers(0, cfg.vocab_size, tokens).tolist():
        u = ref.step(tok)
        chosen = gen.run_token(cpu, b, tok)
        for name in ("x", "y", "o", "g", "u"):
            assert (gen.peek(cpu, b, "scratch." + name) == ref.t[name]).all(), name
        K, V = cache(cpu, b, cfg)
        assert (K == ref.K).all() and (V == ref.V).all()
        assert chosen == int(np.argmax(u))
        assert cpu.tx[-2:] == [0x80 | chosen >> 7, chosen & 0x7F]
    assert not cpu.overflow
    return cpu, b


STORIES = dict(vocab_size=512, dim=64, n_layers=5, n_kv_heads=4, n_heads=8, hidden_dim=172, max_seq_len=256)


@pytest.mark.parametrize("heads, kv_heads, norm, ffn, p", [
    (8, 4, "dyt", "swiglu", 254), (4, 2, "rms", "relu", 254), (2, 2, "dyt", "relu", 254), (2, 2, "rms", "swiglu", 0),
    (8, 4, "rms", "swiglu", 30), (4, 2, "dyt", "swiglu", 127)])
def test_the_stories260k_shapes_at_the_end_of_the_context(heads, kv_heads, norm, ffn, p):
    """Head sizes 8, 16 and 32, V rows that end at the end of a page, two argmax blocks, positions 254 and 255."""
    shape = dict(STORIES, n_heads=heads, n_kv_heads=kv_heads)
    cpu, b = at_position(*random_model(11, norm, ffn, **shape), p=p, tokens=2)
    assert len(b.words) <= 4096


@pytest.mark.parametrize("shape", [
    dict(vocab_size=300, dim=48, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=260, max_seq_len=256),   # d = 24, F > 255
    dict(vocab_size=700, dim=24, n_layers=3, n_heads=6, n_kv_heads=2, hidden_dim=30, max_seq_len=40),     # d = 4, share 3
    dict(vocab_size=300, dim=264, n_layers=1, n_heads=4, n_kv_heads=1, hidden_dim=260, max_seq_len=8),    # D > 255, d = 66,
])                                                                                                        # scratch in 2 pages
def test_unusual_shapes(shape):
    T = shape["max_seq_len"]
    for norm, ffn, wild in (("dyt", "swiglu", False), ("rms", "relu", True)):
        at_position(*random_model(2, norm, ffn, wild, **shape), p=T - 2, tokens=2)
        at_position(*random_model(3, norm, ffn, wild, **shape), p=0, tokens=2)


@pytest.mark.parametrize("norm", ["dyt", "rms"])
@pytest.mark.parametrize("ffn", ["swiglu", "relu"])
@pytest.mark.parametrize("wild", [False, True])
def test_logits_and_tokens_equal_the_integer_reference(norm, ffn, wild, tmp_path):
    """lsllm.intref (the official reference) on the same exported files: 16 tokens, greedy."""
    import json

    from lsllm.intref import IntModel
    spec, arrays = random_model(21, norm, ffn, wild, **dict(SMALL, n_heads=4, n_kv_heads=2))
    with open(tmp_path / "model.json", "w") as f:
        json.dump(spec, f)
    np.savez(tmp_path / "model.npz", **arrays)
    reference = IntModel.load(str(tmp_path))
    b = gen.build(*gen.load(str(tmp_path)))
    cpu, tok = gen.machine(b), 3
    for step in range(16):
        trace = {}
        u = reference.step(tok, trace)
        chosen = gen.run_token(cpu, b, tok)
        assert (gen.peek(cpu, b, "scratch.u") == u).all() and (gen.peek(cpu, b, "scratch.y") == trace["y"][0]).all()
        assert (gen.peek(cpu, b, "scratch.x") == trace["l1.xf"][0]).all()
        tok = int(reference.choose(u[None])[0])
        assert chosen == tok
    assert not cpu.overflow


def test_models_older_than_spec_v04_are_refused_and_shapes_that_do_not_fit_say_why():
    spec, arrays = random_model(0, "dyt", "relu", **SMALL)
    for change in (dict(EXP=arrays["EXP"][0]), dict(EXP=arrays["EXP"] - 1), dict(RECIP=None)):
        old = {name: value for name, value in {**arrays, **change}.items() if value is not None}
        with pytest.raises(ValueError, match="v0.4"):
            gen.build(spec, old)
    spec["shifts"]["sh_o"] = [16, 3]                             # a rounded shift: at most 15
    with pytest.raises(ValueError, match="16 bits"):
        gen.build(spec, arrays)
    from lsllm.ls.cost import dummy_model
    for change in (dict(max_seq_len=512), dict(n_layers=21), dict(vocab_size=5000)):
        with pytest.raises(ValueError):
            gen.build(*random_model(0, "dyt", "swiglu", **dict(SMALL, **change)))
    big = ModelConfig(vocab_size=600, dim=264, n_layers=1, n_heads=4, n_kv_heads=4, hidden_dim=500, max_seq_len=8)
    with pytest.raises(ValueError, match="2\\^19"):
        gen.build(*dummy_model(big, {"norm": "dyt", "ffn": "swiglu"}))
    deep = ModelConfig(vocab_size=32, dim=16, n_layers=20, n_heads=8, n_kv_heads=4, hidden_dim=4, max_seq_len=4)
    with pytest.raises(ValueError, match="words"):
        gen.build(*dummy_model(deep, {"norm": "rms", "ffn": "swiglu"}))
    gen.build(*dummy_model(ModelConfig(n_heads=16, n_kv_heads=8), {"norm": "rms", "ffn": "swiglu"}))    # 16 heads fit


@pytest.mark.parametrize("wild", [False, True])
def test_rms_norm_when_its_multiplier_and_x_are_in_different_pages(wild):
    """V = 1100 makes the token table end so that scratch.r is the last byte used in one page and x starts the
    next: the page change between LUT and the store of r must not use R0."""
    shape = dict(vocab_size=1100, dim=128, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=8, max_seq_len=3)
    spec, arrays = random_model(1, "rms", "relu", wild, **shape)
    b = gen.build(spec, arrays)
    assert b.mem["scratch.r"][0] // gen.PAGE != b.mem["scratch.x"][0] // gen.PAGE
    at_position(spec, arrays, p=0, tokens=3)


def test_every_token_id_is_looked_up_chosen_and_sent_correctly():
    """A model that repeats its input: zero layers' weights, identity norm, rows of E with entries +16 or -16.
    Then u[j] = E[j] . E[token] / 64 is largest for j = token, for every one of 700 token ids."""
    from lsllm.ls.cost import dummy_model
    cfg = ModelConfig(vocab_size=700, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=4, max_seq_len=4)
    spec, arrays = dummy_model(cfg, {"norm": "dyt", "ffn": "relu"})
    signs = (np.arange(700)[:, None] >> np.arange(16)) & 1                        # the bits of the id: all rows differ
    arrays["E"] = (16 - 32 * signs).astype(np.int8)
    arrays["NORM"] = np.resize(VALUES, (3, 256)).astype(np.int8)
    spec["shifts"]["sh_l"] = 6
    b = gen.build(spec, arrays)
    cpu = gen.machine(b)
    for tok in range(700):
        gen.set_position(cpu, b, tok % 4)
        assert gen.run_token(cpu, b, tok) == tok
        u = gen.peek(cpu, b, "scratch.u")
        assert u[tok] == 64 and np.count_nonzero(u == 64) == 1
        assert cpu.tx[-2:] == [0x80 | tok >> 7, tok & 0x7F]
    assert len(cpu.tx) == 1400 and not cpu.overflow


@pytest.mark.parametrize("V", [1600, 2500, 4096])
def test_token_table_of_a_large_vocabulary(V):
    """Columns of the token table that do not fit one page together; the same repeating model as above."""
    from lsllm.ls.cost import dummy_model
    cfg = ModelConfig(vocab_size=V, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=4, max_seq_len=4)
    spec, arrays = dummy_model(cfg, {"norm": "dyt", "ffn": "relu"})
    arrays["E"] = (16 - 32 * ((np.arange(V)[:, None] >> np.arange(16)) & 1)).astype(np.int8)
    arrays["NORM"] = np.resize(VALUES, (3, 256)).astype(np.int8)
    spec["shifts"]["sh_l"] = 6
    b = gen.build(spec, arrays)
    cpu = gen.machine(b)
    for tok in (0, 255, 256, 1535, 1599, V - 1, V // 2 + 77):
        gen.set_position(cpu, b, 0)
        assert gen.run_token(cpu, b, tok) == tok and cpu.tx[-2:] == [0x80 | tok >> 7, tok & 0x7F]


@pytest.mark.parametrize("shape", [
    dict(vocab_size=40, dim=16, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=24, max_seq_len=10, norm="dyt"),
    dict(vocab_size=30, dim=24, n_layers=2, n_heads=2, n_kv_heads=1, hidden_dim=20, max_seq_len=10, ffn="relu")])
def test_a_model_exported_by_the_qat_code_runs_to_the_same_logits(shape, tmp_path):
    """The whole contract of SPEC Part E: float model -> lsllm.qat export -> files -> program -> simulator,
    compared with lsllm.intref on the same files for a full context."""
    import torch

    from lsllm.model import build
    from lsllm.qat.calibrate import calibrate
    from lsllm.qat.export import export
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = ModelConfig(**shape)
    float_model = build(cfg)
    with torch.no_grad():
        for p in float_model.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    reference = export(calibrate(float_model, [torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len))]), str(tmp_path))
    b = gen.build(*gen.load(str(tmp_path)))
    cpu = gen.machine(b)
    for tok in np.random.default_rng(1).integers(0, cfg.vocab_size, cfg.max_seq_len).tolist():
        u = reference.step(tok)
        assert gen.run_token(cpu, b, tok) == int(reference.choose(u[None])[0])
        assert (gen.peek(cpu, b, "scratch.u") == u).all()
    assert not cpu.overflow
