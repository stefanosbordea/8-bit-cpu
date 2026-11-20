"""IntModel against a second implementation written from the SPEC text with plain Python integers and loops;
the batch path, the trace, the 24-bit check, the LFSR and Gumbel-max sampling."""
import json

import numpy as np
import pytest
import torch

from lsllm.config import ModelConfig
from lsllm.intref import IntModel
from lsllm.intref.model import LFSR_RESET, gumbel_table, lfsr_start, lfsr_step, rnd, rnd_tables
from lsllm.model import build
from lsllm.qat.calibrate import calibrate
from lsllm.qat.export import export

torch.set_num_threads(2)
SHAPES = {
    "dyt_swiglu": ModelConfig(vocab_size=40, dim=16, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=24, max_seq_len=10,
                              norm="dyt"),
    "rms_relu": ModelConfig(vocab_size=30, dim=24, n_layers=2, n_heads=2, n_kv_heads=1, hidden_dim=20, max_seq_len=10,
                            ffn="relu"),
    "rms_swiglu": ModelConfig(vocab_size=40, dim=16, n_layers=2, n_heads=4, n_kv_heads=4, hidden_dim=24, max_seq_len=10),
}


def exported(cfg, directory, seed=0):
    """Files of an integer model: a random float model with enlarged weights, calibrated on random tokens."""
    torch.manual_seed(seed)
    float_model = build(cfg)
    with torch.no_grad():
        for p in float_model.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    export(calibrate(float_model, [torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len))]), str(directory))
    return str(directory)


@pytest.fixture(scope="module", params=list(SHAPES))
def directory(request, tmp_path_factory):
    return exported(SHAPES[request.param], tmp_path_factory.mktemp(request.param))


class Loops:
    """SPEC Part C for one sequence, one number at a time."""

    def __init__(self, directory):
        with open(directory + "/model.json") as f:
            self.spec = json.load(f)
        self.A = {name: value.tolist() for name, value in np.load(directory + "/model.npz").items()}
        cfg = self.spec["config"]
        self.D, self.L, self.H, self.G = cfg["dim"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"]
        self.F, self.V, self.d = cfg["hidden_dim"], cfg["vocab_size"], cfg["dim"] // cfg["n_heads"]
        self.K = [[[] for g in range(self.G)] for l in range(self.L)]                               # K[l][g][p][j]
        self.Vc = [[[[] for j in range(self.d)] for g in range(self.G)] for l in range(self.L)]     # V[l][g][j][p]
        self.p = 0

    @staticmethod
    def rq(acc, sh, relu=False):
        assert -2 ** 23 <= acc < 2 ** 23
        t = acc >> sh
        if relu and t < 0:
            t = 0
        return max(-128, min(127, t))

    def rqr(self, acc, sh, relu=False):
        return self.rq(acc + (2 ** (sh - 1) if sh >= 1 else 0), sh, relu)

    @staticmethod
    def lut(table, v):
        return table[v % 256]

    def norm(self, n, x):
        sh, table = self.spec["shifts"], self.A["NORM"][n]
        if self.spec["options"]["norm"] == "dyt":
            return [self.lut(table, v) for v in x]
        r = self.lut(table, self.rq(sum(v * v for v in x), sh["sa"][n]))
        return [self.rqr(v * r, sh["sb"][n]) for v in x]

    def linear(self, W, y, sh, relu=False):
        return [self.rqr(sum(w * v for w, v in zip(row, y)), sh, relu) for row in W]

    def residual(self, x, W, h):
        return [self.rq(256 * x[r] + 128 + sum(w * v for w, v in zip(W[r], h)), 8) for r in range(self.D)]

    def rope(self, head):
        first, second = [], []
        for j in range(self.d // 2):
            a, b = head[2 * j], head[2 * j + 1]
            c, s = self.A["ROPE_C"][self.p][j], self.A["ROPE_S"][self.p][j]
            first.append(self.rqr(a * c - b * s, 7))
            second.append(self.rqr(a * s + b * c, 7))
        return first + second

    def step(self, tok):
        A, sh, pre, d, p = self.A, self.spec["shifts"], self.spec["preloads"], self.d, self.p
        x = [self.rqr(v, sh.get("sh_e", 0)) for v in A["E"][tok]]       # SPEC C.1: the lookup shifts E down to the stream
        for l in range(self.L):
            y = self.norm(2 * l, x)
            q = self.linear(A["Wq_%d" % l], y, sh["sh_q"][l])
            k = self.linear(A["Wk_%d" % l], y, sh["sh_k"][l])
            v = self.linear(A["Wv_%d" % l], y, sh["sh_v"][l])
            for g in range(self.G):
                self.K[l][g].append(self.rope(k[g * d:(g + 1) * d]))
                for j in range(d):
                    self.Vc[l][g][j].append(v[g * d + j])
            o = []
            for h in range(self.H):
                g = h // (self.H // self.G)
                qh = self.rope(q[h * d:(h + 1) * d])
                code = [self.rq(pre["c_s"][l][h] + sum(qh[j] * self.K[l][g][t][j] for j in range(d)), sh["sh_s"][l], True)
                        for t in range(p + 1)]                              # SPEC C.6 step 1: 0..127
                m = max(code)                                               # step 2
                e = [self.lut(A["EXP"][l], c - m) for c in code]            # step 3
                dsum = sum(e)                                               # step 4
                k = dsum.bit_length()                                       # step 5
                mant = dsum >> (k - 7)
                assert 127 <= dsum <= 127 * (p + 1) and 7 <= k <= 15 and 64 <= mant <= 127
                for j in range(d):
                    acc = sum(e[t] * self.Vc[l][g][j][t] for t in range(p + 1))
                    o.append(self.rqr(self.rq(acc, k) * self.lut(A["RECIP"], mant), sh["sh_o"][l]))
            x = self.residual(x, A["Wo_%d" % l], o)
            y = self.norm(2 * l + 1, x)
            if self.spec["options"]["ffn"] == "swiglu":
                g1 = [self.lut(A["SILU"], v) for v in self.linear(A["W1_%d" % l], y, sh["sh_1"][l])]
                h3 = self.linear(A["W3_%d" % l], y, sh["sh_3"][l])
                hidden = [self.rqr(g1[r] * h3[r], sh["sh_g"][l]) for r in range(self.F)]
            else:
                hidden = self.linear(A["W1_%d" % l], y, sh["sh_1"][l], relu=True)
            x = self.residual(x, A["W2_%d" % l], hidden)
        y = self.norm(2 * self.L, x)
        self.p += 1
        return [self.rq(pre["c_l"] + sum(w * v for w, v in zip(row, y)), sh["sh_l"], relu=True) for row in A["E"]]


def test_step_equals_the_loop_implementation(directory):
    model, loops = IntModel.load(directory), Loops(directory)
    tokens = np.random.default_rng(1).integers(0, model.cfg.vocab_size, model.cfg.max_seq_len)
    for t in tokens:
        u = model.step(int(t))
        assert u.shape == (model.cfg.vocab_size,) and u.tolist() == loops.step(int(t))
    assert len(set(u.tolist())) > 5                                         # not a degenerate model
    with pytest.raises(AssertionError):
        model.step(0)                                                       # the context is full


def test_batch_rows_are_independent_sequences(directory):
    model = IntModel.load(directory)
    tokens = np.random.default_rng(2).integers(0, model.cfg.vocab_size, (3, model.cfg.max_seq_len))
    batch = model.forward(tokens)
    assert batch.shape == (3, model.cfg.max_seq_len, model.cfg.vocab_size)
    for row in range(3):
        model.reset()
        assert np.array_equal(batch[row], np.stack([model.step(int(t)) for t in tokens[row]]))
    # select() copies and drops sequences together with their caches
    model.reset(3)
    for t in range(4):
        model.step(tokens[:, t])
    model.select([2, 0, 2])
    assert np.array_equal(model.step(tokens[[2, 0, 2], 4]), batch[[2, 0, 2], 4])


def test_trace_holds_every_value_of_a_step(directory):
    model = IntModel.load(directory)
    tokens = np.random.default_rng(3).integers(0, model.cfg.vocab_size, 6)
    for t in tokens:
        trace = {}
        u = model.step(int(t), trace=trace)
    cfg, sh, p = model.cfg, model.sh, len(tokens) - 1
    swiglu, rms = model.opt["ffn"] == "swiglu", model.opt["norm"] == "rms"
    names = {"x0", "y", "u", "u.acc"}
    for l in range(cfg.n_layers):
        plain = ["ya", "yf", "max", "e", "dsum", "nrm", "mant", "r"] + ["g1"] * swiglu + ["ya.r", "yf.r"] * rms
        streams = (["q", "k", "v", "qr", "kr", "code", "a", "o", "xa", "g", "xf"] + ["h1", "h3"] * swiglu
                   + ["ya", "yf", "ya.m", "yf.m"] * rms)
        names |= {"l%d.%s" % (l, name) for name in plain + streams} | {"l%d.%s.acc" % (l, name) for name in streams}
    names |= {"y.acc", "y.m", "y.m.acc", "y.r"} if rms else set()
    names |= {"x0.acc"} if sh.get("sh_e", 0) else set()                # the embedding lookup is a requantisation then
    assert set(trace) == names
    for name, value in trace.items():
        if name.endswith(".dsum"):
            assert 127 <= value.min() and value.max() <= 127 * (p + 1)
            assert trace[name[:-4] + "nrm"].ravel().tolist() == [int(total).bit_length() for total in value.ravel()]
            assert trace[name[:-4] + "mant"].min() >= 64
        elif name.endswith(".acc"):
            assert -2 ** 23 <= value.min() and value.max() < 2 ** 23, name
        else:
            assert -128 <= value.min() and value.max() <= 127, name
    assert np.array_equal(trace["u"][0], u) and trace["u"].shape == (1, cfg.vocab_size)
    # an accumulator and its result belong together: RQ of SPEC B.2
    assert np.array_equal(np.clip(np.maximum(trace["u.acc"] >> sh["sh_l"], 0), -128, 127), trace["u"])
    assert np.array_equal(np.clip(trace["l1.q.acc"] >> sh["sh_q"][1], -128, 127), trace["l1.q"])
    assert np.array_equal(np.clip(trace["l0.xa.acc"] >> 8, -128, 127), trace["l0.xa"])
    # shapes: one code and one weight per query head and per key seen so far
    share = cfg.n_heads // cfg.n_kv_heads
    assert trace["l0.code"].shape == trace["l0.e"].shape == (1, cfg.n_kv_heads, share, p + 1)
    assert trace["l0.o"].shape == (1, cfg.n_kv_heads, share, cfg.head_dim)
    assert trace["l0.code"].min() >= 0 and np.array_equal(trace["l0.max"][..., 0], trace["l0.code"].max(-1))
    assert trace["l0.e"].min() >= 0 and (trace["l0.e"].max(-1) == 127).all()    # the largest code has weight 127
    # the cache holds what the trace says was written at this position (SPEC C.5)
    assert np.array_equal(model.K[1][0, :, p], trace["l1.kr"][0])
    assert np.array_equal(model.V[1][0, :, :, p].ravel(), trace["l1.v"][0])
    assert set(model.peak) == {name[:-4] for name in names if name.endswith(".acc")}
    assert model.peak["u"] >= np.abs(trace["u.acc"]).max()


def test_greedy_generation(directory):
    model = IntModel.load(directory)
    prompt = [1, 5, 7]
    new = model.generate(prompt, 5)
    assert len(new) == 5 and new == model.generate(prompt, 5)
    model.reset()
    for t in prompt + new[:-1]:
        u = model.step(t)
    assert new[-1] == int(np.flatnonzero(u == u.max())[0])                  # the smallest index of the largest logit
    assert model.choose(np.array([[3, 9, 9, 1], [0, 0, 0, 0]])).tolist() == [1, 0]


def test_lfsr():
    assert lfsr_step(1) == 0x80000000 and lfsr_step(0x80000000) == 0xC0000000 and lfsr_step(1 << 10) == 0x80000200
    # the period is 2^32 - 1: the step, as a matrix over GF(2), has exactly that order (checked on its prime factors)
    step = np.array([[(lfsr_step(1 << col) >> row) & 1 for col in range(32)] for row in range(32)], dtype=np.int64)

    def power(n):
        out, base = np.eye(32, dtype=np.int64), step
        while n:
            if n & 1:
                out = out @ base % 2
            base, n = base @ base % 2, n >> 1
        return out

    full = 2 ** 32 - 1
    assert np.array_equal(power(full), np.eye(32, dtype=np.int64))
    assert all(not np.array_equal(power(full // prime), np.eye(32, dtype=np.int64)) for prime in (3, 5, 17, 257, 65537))
    state = LFSR_RESET
    for g in range(8):
        state = lfsr_step(state)
    assert rnd(LFSR_RESET) == (state, state >> 24)                          # RND: 8 steps, then the upper byte
    assert lfsr_start(0).tolist() == [0xACE1ACE1] and lfsr_start(3, 2).tolist() == [0xACE1ACE1 ^ 3, 0xACE1ACE1 ^ 4]
    assert lfsr_start(0xACE1).tolist() == [0xACE10000]                      # the upper half keeps it away from 0
    states = np.array([0xACE1ACE1, 1, 0xFFFFFFFF, 0x12345678], dtype=np.int64)
    t0, t1, t2, t3 = rnd_tables()                                           # the table form used for batches
    fast = t0[states & 255] ^ t1[(states >> 8) & 255] ^ t2[(states >> 16) & 255] ^ t3[states >> 24]
    assert fast.tolist() == [rnd(int(s))[0] for s in states]


def test_sampling_is_gumbel_max_with_one_rnd_per_token(directory):
    model = IntModel.load(directory)
    rng = np.random.default_rng(4)
    u = rng.integers(0, 128, (3, model.cfg.vocab_size))
    model.reset(3, seed=11)
    states = [int(s) for s in model.lfsr]
    table = gumbel_table(0.8, model.f_l)
    assert table.min() == 0 and table.max() <= 63 and (np.diff(table) >= 0).all()
    for round_ in range(3):
        chosen = model.choose(u, temperature=0.8)
        for row in range(3):                                                # the same with one RND at a time
            candidates = []
            for j in range(model.cfg.vocab_size):
                states[row], byte = rnd(states[row])
                candidates.append((int(u[row, j]) >> 1) + int(table[byte]))
            assert chosen[row] == candidates.index(max(candidates))
        assert model.lfsr.tolist() == states
    assert model.generate([1, 2], 6, temperature=0.8, seed=5) == model.generate([1, 2], 6, temperature=0.8, seed=5)
    assert len({tuple(model.generate([1, 2], 8, temperature=1.5, seed=seed)) for seed in range(5, 11)}) > 1


def test_sampling_follows_the_softmax_of_the_logits(directory):
    """Many draws from fixed logits: token frequencies are close to softmax(logits / temperature)."""
    model = IntModel.load(directory)
    V, rows = model.cfg.vocab_size, 4000
    u = np.zeros((rows, V), dtype=np.int64)
    u[:, :4] = 96 + np.array([0, 8, 16, 24])                                # four tokens that matter, the rest far below
    model.reset(rows, seed=1)
    counts = np.bincount(model.choose(u, temperature=1.0), minlength=V)
    real = u[0, :4] * 2.0 ** -model.f_l
    expected = np.exp(real - real.max()) / np.exp(real - real.max()).sum()
    assert counts[4:].sum() <= rows * 0.01
    assert np.abs(counts[:4] / rows - expected).max() < 0.06, (counts[:4] / rows, expected)


def test_accumulator_overflow_is_caught(tmp_path):
    """A feed-forward stream of 600 terms can leave 24 bits (600 * 127 * 127 > 2^23): the reference must stop."""
    cfg = ModelConfig(vocab_size=20, dim=8, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=600, max_seq_len=6, ffn="relu",
                      norm="dyt")
    directory = exported(cfg, tmp_path)
    arrays = dict(np.load(directory + "/model.npz"))
    IntModel.load(directory).step(3)                                        # fine as exported
    with open(directory + "/model.json") as f:
        spec = json.load(f)
    spec["shifts"]["sh_1"] = [0]                                            # hidden units saturate at 127 ...
    arrays["W1_0"] = np.full_like(arrays["W1_0"], 127)
    arrays["NORM"] = np.full_like(arrays["NORM"], 127)                      # every norm output is 127
    arrays["W2_0"] = np.full_like(arrays["W2_0"], 127)                      # ... and all add up
    with pytest.raises(AssertionError, match="l0.xf: the accumulator leaves 24 bits"):
        IntModel(spec, arrays).step(3)


def test_invalid_models_are_refused(tmp_path):
    cfg = SHAPES["dyt_swiglu"]
    directory = exported(cfg, tmp_path)
    arrays = dict(np.load(directory + "/model.npz"))
    with open(directory + "/model.json") as f:
        spec = json.load(f)
    IntModel(spec, arrays)

    def changed(path, value):
        new = json.loads(json.dumps(spec))
        where = new
        for key in path[:-1]:
            where = where[key]
        where[path[-1]] = value
        return new

    with pytest.raises(AssertionError, match="shift"):
        IntModel(changed(["shifts", "sh_s", 0], 24), arrays)
    with pytest.raises(AssertionError, match="shift sh_q"):     # rqr: the rounding preload 2^15 does not fit 16 bits
        IntModel(changed(["shifts", "sh_q", 0], 16), arrays)
    IntModel(changed(["shifts", "sh_q", 0], 15), arrays)
    with pytest.raises(AssertionError, match="preload"):
        IntModel(changed(["preloads", "c_l"], 40000), arrays)
    with pytest.raises(AssertionError, match="preload"):
        IntModel(changed(["preloads", "c_s", 1, 0], -32769), arrays)
    with pytest.raises(AssertionError, match="EXP is not one table per layer"):
        IntModel(spec, {**arrays, "EXP": arrays["EXP"][0]})     # the single table of exports before SPEC v0.4
    with pytest.raises(AssertionError, match="weights"):
        IntModel(spec, {**arrays, "W2_0": np.full_like(arrays["W2_0"], -128)})
    with pytest.raises(AssertionError, match="int8"):
        IntModel(spec, {**arrays, "E": arrays["E"].astype(np.int16)})
