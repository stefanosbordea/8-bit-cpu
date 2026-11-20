"""The integer model of SPEC Parts C, D and E in numpy integers: what the LS CPU computes, bit for bit.

IntModel.load(directory) reads an exported model. step(token) returns the logits u (0..127) of the next token
and keeps the KV cache, like the CPU. All sequences of a batch advance together, so thousands of benchmark
samples can run at once; one token (an int) is a batch of one sequence.
Names used by the trace and by the accumulator peaks (l = layer number):
    x0                              embedding row, shifted right by sh_e with rounding
    l.ya, l.yf, y                   norm outputs (attention site, feed-forward site, final site)
    l.q, l.k, l.v                   projections;  l.qr, l.kr  after RoPE (as [a' of every pair, b' of every pair])
    l.code, l.max, l.e              attention (SPEC C.6), per query head: score codes 0..127 of the keys, their
                                    largest value, weights 0..127
    l.dsum, l.nrm, l.mant, l.r      sum of the weights (127 .. 127 T, more than a byte), its number of significant
                                    bits (the NRM instruction), its mantissa 64..127, the table value RECIP[mant]
    l.a, l.o                        value sums shifted to int8, head outputs
    l.xa, l.xf                      residual stream after the attention and after the feed-forward update
    l.h1, l.g1, l.h3, l.g           feed-forward
    u                               logits
A name followed by ".acc" is the accumulator (with its preload) before the requantisation that gives the name.
"""
import functools
import json
import os

import numpy as np

from lsllm.config import ModelConfig

ACC_MIN, ACC_MAX = -2 ** 23, 2 ** 23 - 1      # the signed 24-bit accumulator
LFSR_RESET = 0xACE1ACE1                       # the switches are xored into the lower 16 bits


def half(sh):
    """Rounding preload of rqr (SPEC B.2): 2^(sh - 1), and 0 for sh = 0."""
    return (1 << sh) >> 1


def bitlen(n):
    """Number of significant bits of every n in 0..32767, found by comparisons (the NRM instruction): 0 for 0,
    1 for 1, 8 for 128..255. n has a last axis of length 1."""
    return (n >= 1 << np.arange(15)).sum(-1, keepdims=True)


def lfsr_step(s):
    """One step of the 32-bit LFSR of SPEC A.2 (x^32 + x^22 + x^2 + x + 1: every state but 0, period 2^32 - 1)."""
    bit = (s ^ (s >> 10) ^ (s >> 30) ^ (s >> 31)) & 1
    return (s >> 1) | (bit << 31)


def rnd(s):
    """The RND instruction: the LFSR advances 8 steps. Returns (new state, its upper byte)."""
    for g in range(8):
        s = lfsr_step(s)
    return s, s >> 24


def lfsr_start(seed, rows=1):
    """Reset state of the LFSR of every sequence: 0xACE1ACE1 with the switches (seed, seed + 1, ...) xored into
    the lower 16 bits. The upper half keeps it away from 0."""
    return LFSR_RESET ^ ((seed + np.arange(rows, dtype=np.int64)) & 0xFFFF)


@functools.lru_cache(maxsize=None)
def rnd_tables():
    """Four tables of 256 states. The LFSR is linear, so RND of a state is the xor of table[k][byte k of it]:
    sampling for many sequences at once without stepping bit by bit."""
    return np.array([[rnd(byte << (8 * k))[0] for byte in range(256)] for k in range(4)], dtype=np.int64)


def gumbel_table(temperature, f_l):
    """GUMBEL of SPEC Part D: noise for u >> 1, indexed by a random byte. g0 makes the smallest entry 0.
    The table depends on the temperature, so it is not in the exported files: the board's table must be made by
    this function (the only place where the integer reference uses floating point)."""
    scale = temperature * 2.0 ** (f_l - 1)
    noise = -np.log(-np.log((np.arange(256) + 0.5) / 256))
    g0 = np.round(scale * np.log(np.log(512)))
    return np.clip(np.round(g0 + scale * noise), 0, 63).astype(np.int32)


class IntModel:
    def __init__(self, spec, arrays):
        """spec: the content of model.json; arrays: name -> int8 array of model.npz (SPEC Part E)."""
        self.spec, self.cfg, self.opt = spec, ModelConfig(**spec["config"]), spec["options"]
        self.sh, self.pre, self.f_l = spec["shifts"], spec["preloads"], spec["frac"]["u"]
        cfg = self.cfg
        assert cfg.tied_embeddings, "SPEC Part C has one matrix E"
        # sh_s, sh_l (their preloads c_s, c_l hold the rounding) and sa (no rounding) may be 0..23; every other
        # shift belongs to an rqr, whose rounding preload 2^(sh - 1) must fit a signed 16-bit value: at most 15
        for name, value in self.sh.items():
            top = 23 if name in ("sh_s", "sh_l", "sa") else 15
            assert 0 <= np.min(value) and np.max(value) <= top, "the shift %s is outside 0..%d" % (name, top)
        preloads = np.concatenate([np.ravel(v) for v in self.pre.values()])
        assert preloads.min() >= -2 ** 15 and preloads.max() < 2 ** 15, "a preload does not fit 16 bits"
        for name, value in arrays.items():
            assert value.dtype == np.int8, name + " is not int8"
            assert value.min() >= -127 or name[0] != "W" and name != "E", name + ": weights are -127..127"
            setattr(self, name, value.astype(np.int32))       # self.E, self.Wq_0, ..., self.NORM, self.EXP, ...
        assert self.EXP.shape == (cfg.n_layers, 256), "EXP is not one table per layer (an export older than SPEC v0.4?)"
        self.peak = {}          # name -> largest |accumulator| seen so far
        self.trace = None
        self.reset()

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "model.json")) as f:
            spec = json.load(f)
        with np.load(os.path.join(directory, "model.npz")) as arrays:
            return cls(spec, {name: arrays[name] for name in arrays.files})

    def reset(self, rows=1, seed=0):
        """Empty KV cache for `rows` sequences at position 0; `seed` stands for the switches at reset."""
        G, d, T = self.cfg.n_kv_heads, self.cfg.head_dim, self.cfg.max_seq_len
        self.K = [np.zeros((rows, G, T, d), dtype=np.int32) for l in range(self.cfg.n_layers)]    # K[l][row, g, p, j]
        self.V = [np.zeros((rows, G, d, T), dtype=np.int32) for l in range(self.cfg.n_layers)]    # V[l][row, g, j, p]
        self.pos = 0
        self.lfsr = lfsr_start(seed, rows)

    def select(self, rows):
        """Keep the sequences `rows` of the batch. An index may repeat: that copies the sequence."""
        self.K = [k[rows] for k in self.K]
        self.V = [v[rows] for v in self.V]
        self.lfsr = self.lfsr[rows]

    def _note(self, name, value):
        if self.trace is not None:
            self.trace[name] = value
        return value

    def _rq(self, name, acc, sh, relu=False):
        """RQ of SPEC B.2 on accumulator values: floor shift, optional ReLU, clamp to int8."""
        low, high = int(acc.min()), int(acc.max())
        assert ACC_MIN <= low and high <= ACC_MAX, "%s: the accumulator leaves 24 bits (%d .. %d)" % (name, low, high)
        self.peak[name] = max(self.peak.get(name, 0), -low, high)
        self._note(name + ".acc", acc)
        out = acc >> sh                                       # arithmetic shift: rounds toward minus infinity
        if relu:
            out = np.maximum(out, 0)
        return self._note(name, np.clip(out, -128, 127))

    def _rqr(self, name, acc, sh, relu=False):
        """RQ with the rounding preload."""
        return self._rq(name, acc + half(sh), sh, relu)

    def _norm(self, n, name, x):
        """SPEC C.2 at site n."""
        if self.opt["norm"] == "dyt":
            return self._note(name, self.NORM[n][x & 255])
        m = self._rq(name + ".m", (x * x).sum(-1, keepdims=True), self.sh["sa"][n])       # 0..127
        return self._rqr(name, x * self._note(name + ".r", self.NORM[n][m]), self.sh["sb"][n])

    def _rope(self, name, heads):
        """SPEC C.4 on (rows, heads, d) at the current position."""
        a, b = heads[..., 0::2], heads[..., 1::2]
        c, s = self.ROPE_C[self.pos], self.ROPE_S[self.pos]
        return self._rqr(name, np.concatenate([a * c - b * s, a * s + b * c], -1), 7)

    def _divide(self, at, e, A, sh):
        """SPEC C.6 step 5: the value sums A (rows, G, share, d) divided by the sum of the weights e (.., keys)."""
        dsum = self._note(at + "dsum", e.sum(-1, keepdims=True))            # 127 .. 127 T: the largest weight is 127
        k = self._note(at + "nrm", bitlen(dsum))                            # 7 .. 15
        r = self._note(at + "r", self.RECIP[self._note(at + "mant", dsum >> (k - 7))])     # about 8192 / mantissa
        a = self._rq(at + "a", A, k)                                        # |A| <= 128 dsum < 128 * 2^k: fits int8
        return self._rqr(at + "o", a * r, sh)                               # a * r = 64 * A / dsum

    def step(self, tokens, trace=None):
        """Feed one token per sequence; returns the logits u (rows, V) of the next token (shape (V,) for an int).

        trace: a dictionary that receives every intermediate value of this step by name (see the module text).
        """
        cfg, sh, p = self.cfg, self.sh, self.pos
        assert p < cfg.max_seq_len, "the context is full"
        tok = np.atleast_1d(tokens)
        B, G, d, share = len(tok), cfg.n_kv_heads, cfg.head_dim, cfg.n_heads // cfg.n_kv_heads
        self.trace = trace
        first = max(0, p + 1 - cfg.window) if cfg.window else 0
        sh_e = sh.get("sh_e", 0)                                # E may have more fractional bits than the stream (SPEC C.1)
        x = self._rqr("x0", self.E[tok], sh_e) if sh_e else self._note("x0", self.E[tok])
        for l in range(cfg.n_layers):
            at = "l%d." % l
            W = {name: getattr(self, "%s_%d" % (name, l), None) for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W3", "W2")}
            y = self._norm(2 * l, at + "ya", x)
            q = self._rqr(at + "q", y @ W["Wq"].T, sh["sh_q"][l])
            k = self._rqr(at + "k", y @ W["Wk"].T, sh["sh_k"][l])
            v = self._rqr(at + "v", y @ W["Wv"].T, sh["sh_v"][l])
            q = self._rope(at + "qr", q.reshape(B, G, share, d))
            self.K[l][:, :, p] = self._rope(at + "kr", k.reshape(B, G, d))
            self.V[l][:, :, :, p] = v.reshape(B, G, d)
            # the keys of this position: all of them, or the last cfg.window (the CPU then keeps a ring of that many)
            keys, values = self.K[l][:, :, first:p + 1], self.V[l][:, :, :, first:p + 1]
            c_s = np.reshape(self.pre["c_s"][l], (G, share, 1))
            code = self._rq(at + "code", q @ keys.swapaxes(2, 3) + c_s, sh["sh_s"][l], relu=True)   # (B, G, share, p + 1)
            top = self._note(at + "max", code.max(-1, keepdims=True))       # the row maximum m of SPEC C.6
            e = self._note(at + "e", self.EXP[l][(code - top) & 255])       # 127 for the largest code
            o = self._divide(at, e, e @ values.swapaxes(2, 3), sh["sh_o"][l])
            x = self._rq(at + "xa", 256 * x + 128 + o.reshape(B, cfg.dim) @ W["Wo"].T, 8)
            y = self._norm(2 * l + 1, at + "yf", x)
            if self.opt["ffn"] == "swiglu":
                g1 = self._note(at + "g1", self.SILU[self._rqr(at + "h1", y @ W["W1"].T, sh["sh_1"][l]) & 255])
                g = self._rqr(at + "g", g1 * self._rqr(at + "h3", y @ W["W3"].T, sh["sh_3"][l]), sh["sh_g"][l])
            else:
                g = self._rqr(at + "g", y @ W["W1"].T, sh["sh_1"][l], relu=True)
            x = self._rq(at + "xf", 256 * x + 128 + g @ W["W2"].T, 8)
        y = self._norm(2 * cfg.n_layers, "y", x)
        u = self._rq("u", y @ self.E.T + self.pre["c_l"], sh["sh_l"], relu=True)
        self.pos, self.trace = p + 1, None
        return u[0] if np.ndim(tokens) == 0 else u

    def forward(self, tokens):
        """Logits u (rows, time, V) of every position of `tokens` (rows, time), from an empty cache."""
        tokens = np.asarray(tokens)
        self.reset(len(tokens))
        return np.stack([self.step(tokens[:, t]) for t in range(tokens.shape[1])], 1)

    def choose(self, u, temperature=0.0, allowed=None):
        """Next token of every sequence from its logits u (rows, V): the smallest index of the largest value.

        temperature > 0: Gumbel-max sampling (SPEC Part D), one RND per token id, in the order of the ids.
        allowed: (rows, V) of booleans from the syntax filter; a token that is not allowed is never chosen
        (the CPU skips it in the argmax loop). A row in which nothing is allowed is left unfiltered.
        """
        if temperature > 0:
            t0, t1, t2, t3 = rnd_tables()
            state, drawn = self.lfsr, np.empty(u.shape, dtype=np.int64)
            for j in range(u.shape[1]):                 # one RND per token id, in the order of the ids
                state = t0[state & 255] ^ t1[(state >> 8) & 255] ^ t2[(state >> 16) & 255] ^ t3[state >> 24]
                drawn[:, j] = state >> 24
            self.lfsr = state
            u = (u >> 1) + gumbel_table(temperature, self.f_l)[drawn]
        if allowed is not None:
            u = np.where(allowed | ~allowed.any(-1, keepdims=True), u, -1)
        return u.argmax(-1)

    def generate(self, prompt_ids, max_new_tokens, temperature=0.0, seed=0):
        """The new token ids after the prompt, one sequence, greedy (temperature 0) or sampled."""
        self.reset(seed=seed)
        for t in prompt_ids:
            u = self.step([t])
        new = []
        while len(new) < max_new_tokens:
            new.append(int(self.choose(u, temperature)[0]))
            if len(new) < max_new_tokens:
                u = self.step(new[-1:])
        return new
