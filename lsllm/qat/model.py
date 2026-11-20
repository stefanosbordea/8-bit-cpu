"""Quantization-aware training: the integer graph of SPEC Part C, exact, for a whole (batch, time) tensor at once.

Every tensor holds whole numbers below 2^24, which float32 represents exactly, so forward() returns the same
logits as the integer reference (lsllm.intref) and as the CPU, bit for bit.
The trained parameters are float ("latent") weights in real units; the forward pass rounds them to int8.
Gradients are straight-through: rounding, floor shifts and clamps pass the gradient unchanged, a table passes
the slope of the smooth function it samples, a ReLU stops the gradient where its input is negative.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lsllm.qat import tables


class Requant(torch.autograd.Function):
    """rq of SPEC B.2: floor(acc / 2^sh), ReLU if asked, clamp to lo..127. Gradient: that of acc / 2^sh."""

    @staticmethod
    def forward(ctx, acc, sh, relu, lo):
        ctx.scale = 2.0 ** -sh
        ctx.cut = acc < 0 if relu else None
        return torch.floor(acc * ctx.scale).clamp_(0 if relu else lo, 127)

    @staticmethod
    def backward(ctx, grad):
        grad = grad * ctx.scale
        return (grad if ctx.cut is None else grad.masked_fill(ctx.cut, 0.0)), None, None, None


class Lookup(torch.autograd.Function):
    """table[value] for int8 values (index = unsigned byte). Gradient: slope[value]."""

    @staticmethod
    def forward(ctx, value, table, slope):
        index = value.to(torch.int32).bitwise_and_(255)
        ctx.index, ctx.slope = index.to(torch.uint8), slope     # one byte per element is kept for the backward pass
        return table[index]

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.slope[ctx.index.to(torch.int32)], None, None


def rq(acc, sh, relu=False, lo=-128):
    return Requant.apply(acc, sh, relu, lo)


def rqr(acc, sh, relu=False):
    """rq with the rounding preload 2^(sh - 1) (0 for sh = 0)."""
    return rq(acc + ((1 << sh) >> 1), sh, relu)


def ste(exact, smooth):
    """The value of `exact` with the gradient of `smooth`."""
    return smooth + (exact - smooth).detach()


class QatModel(nn.Module):
    def __init__(self, cfg, options, frac, shifts, latent, score_offset, logit_offset, alpha=None, gain=1.0):
        """cfg: ModelConfig. options: {"norm": "dyt" | "rms", "ffn": "swiglu" | "relu"}.
        frac: fractional bits of every tensor ("x", "u", "h1", "g1", "E": int; "y": per norm site; others: per layer;
        "s": the score codes of a layer, 2^f_s codes per nat, the kappa of SPEC C.6).
        shifts: as in SPEC Part E. latent: name ("E", "Wq_0", ...) -> float weights in real units.
        score_offset (layers, heads) and logit_offset: constants in nats behind the preloads c_s and c_l. They are
        not trained: attention weights and the loss do not change when all scores of a row, or all logits, move
        together, so the offsets only decide what is cut off at the ends of the code range.
        alpha: the DyT slope of every norm site. gain: a factor inside the table of the final norm."""
        super().__init__()
        self.cfg, self.options, self.frac, self.shifts = cfg, options, frac, shifts
        self.alpha, self.gain, self.tokenizer = alpha, gain, ""
        self.notes = []         # what calibration wants the reader of model.json to know
        self.latent = nn.ParameterDict({name: nn.Parameter(torch.as_tensor(w, dtype=torch.float32).clone())
                                        for name, w in latent.items()})
        self.register_buffer("score_offset", torch.as_tensor(score_offset, dtype=torch.float32).clone())
        self.logit_offset = float(logit_offset)

        sites = range(2 * cfg.n_layers + 1)
        amp = [2.0 ** frac["y"][n] * (gain if n == 2 * cfg.n_layers else 1.0) for n in sites]
        if options["norm"] == "dyt":
            norm = [tables.dyt(alpha[n], frac["x"], amp[n]) for n in sites]
        else:
            norm = [tables.rms(cfg.dim, shifts["sa"][n], shifts["sb"][n], amp[n]) for n in sites]
        exp = [tables.exp(2.0 ** frac["s"][l]) for l in range(cfg.n_layers)]
        made = {"NORM": [np.stack(part) for part in zip(*norm)], "EXP": [np.stack(part) for part in zip(*exp)],
                "RECIP": tables.recip()}
        if options["ffn"] == "swiglu":
            made["SILU"] = tables.silu(frac["h1"], frac["g1"])
        for name, (table, slope) in made.items():
            self.register_buffer(name, torch.tensor(table, dtype=torch.float32), persistent=False)
            self.register_buffer(name + "_slope", torch.tensor(slope, dtype=torch.float32), persistent=False)
        for name, table in zip(("ROPE_C", "ROPE_S"), tables.rope(cfg.max_seq_len, cfg.head_dim)):
            self.register_buffer(name, torch.tensor(table, dtype=torch.float32), persistent=False)
        self.register_buffer("POW", 2.0 ** torch.arange(15.0), persistent=False)       # 1, 2, .., 16384
        seen = torch.ones(cfg.max_seq_len, cfg.max_seq_len).tril()          # seen[t, s]: position t attends to key s
        if cfg.window:
            seen = seen.triu(1 - cfg.window)                                # only the last `window` keys
        self.register_buffer("causal", seen, persistent=False)

    def scale(self, key):
        """2^f of a weight matrix; key is "E" or "<name>_<layer>"."""
        name, _, layer = key.partition("_")
        return 2.0 ** (self.frac[name][int(layer)] if layer else self.frac[name])

    def weight(self, key):
        """The int8 weights of a matrix (as float32): round(latent * 2^f), clamped to -127..127."""
        w = self.latent[key] * self.scale(key)
        return ste(w.round().clamp(-127, 127), w)

    def c_s(self, l):
        """Preload of the scores of every head of layer l: 2^f_s codes per nat plus the rounding half, times 2^sh_s."""
        c = (2.0 ** self.frac["s"][l] * self.score_offset[l] + 0.5) * 2.0 ** self.shifts["sh_s"][l]
        return c.round().clamp(-2 ** 15, 2 ** 15 - 1)

    def c_l(self):
        """Preload of the logits: 2^f_l steps per nat plus the rounding half, times 2^sh_l."""
        c = round((self.logit_offset * 2.0 ** self.frac["u"] + 0.5) * 2.0 ** self.shifts["sh_l"])
        return min(max(c, -2 ** 15), 2 ** 15 - 1)

    def norm(self, n, x):
        """SPEC C.2 at site n."""
        if self.options["norm"] == "dyt":
            return Lookup.apply(x, self.NORM[n], self.NORM_slope[n])
        m = rq((x * x).sum(-1, keepdim=True), self.shifts["sa"][n])
        return rqr(x * Lookup.apply(m, self.NORM[n], self.NORM_slope[n]), self.shifts["sb"][n])

    def rope(self, heads):
        """SPEC C.4 on (batch, time, heads, d): every head becomes [a' of every pair, b' of every pair]."""
        T = heads.shape[1]
        a, b = heads[..., 0::2], heads[..., 1::2]
        c, s = self.ROPE_C[:T, None], self.ROPE_S[:T, None]
        return torch.cat([rqr(a * c - b * s, 7), rqr(a * s + b * c, 7)], -1)

    def divide(self, e, A, sh):
        """SPEC C.6 step 5: the value sums A divided by the sum of the weights e, with the reciprocal table."""
        dsum = e.sum(-1, keepdim=True)                          # 127 .. 127 T
        k = (dsum >= self.POW).sum(-1, keepdim=True)            # its significant bits, found by comparisons: 7..15
        scale = self.POW[15 - k] / 256                          # 2^(7 - k): the mantissa dsum * scale is in 64..127
        mant = ste(torch.floor(dsum * scale), dsum * scale)
        a = ste(torch.floor(A * scale / 128), A * scale / 128)  # rq(A, k); it cannot leave int8
        return rqr(a * Lookup.apply(mant, self.RECIP, self.RECIP_slope), sh)

    def forward(self, tokens, targets=None):
        """tokens (batch, time) -> integer logits u (batch, time, vocab) in 0..127, and the cross-entropy of
        u / 2^f_l (the logits at their real scale) if targets are given."""
        cfg, sh = self.cfg, self.shifts
        B, T = tokens.shape
        G, d, share = cfg.n_kv_heads, cfg.head_dim, cfg.n_heads // cfg.n_kv_heads
        E = self.weight("E")
        x = E[tokens]
        if sh.get("sh_e", 0):                                   # E holds more fractional bits than the stream: SPEC C.1
            x = rqr(x, sh["sh_e"])
        for l in range(cfg.n_layers):
            W = lambda name: self.weight("%s_%d" % (name, l))
            y = self.norm(2 * l, x)
            # queries (batch, kv heads, share, time, d); keys and values (batch, kv heads, 1, time, d)
            q = self.rope(rqr(y @ W("Wq").T, sh["sh_q"][l]).view(B, T, cfg.n_heads, d))
            q = q.view(B, T, G, share, d).permute(0, 2, 3, 1, 4)
            k = self.rope(rqr(y @ W("Wk").T, sh["sh_k"][l]).view(B, T, G, d)).transpose(1, 2)[:, :, None]
            v = rqr(y @ W("Wv").T, sh["sh_v"][l]).view(B, T, G, d).transpose(1, 2)[:, :, None]
            c_s = self.c_s(l).view(1, G, share, 1, 1)
            # score codes 0..127 (batch, kv heads, share, time, time); the floor at 0 is a clamp for the gradient;
            # later keys get code 0, which leaves the row maximum alone, and weight 0
            code = rq(q @ k.transpose(-2, -1) + c_s, sh["sh_s"][l], lo=0) * self.causal[:T, :T]
            top = code.amax(-1, keepdim=True)
            e = Lookup.apply(code - top, self.EXP[l], self.EXP_slope[l]) * self.causal[:T, :T]
            o = self.divide(e, e @ v, sh["sh_o"][l]).permute(0, 3, 1, 2, 4).reshape(B, T, cfg.dim)
            x = rq(256 * x + 128 + o @ W("Wo").T, 8)
            y = self.norm(2 * l + 1, x)
            if self.options["ffn"] == "swiglu":
                g1 = Lookup.apply(rqr(y @ W("W1").T, sh["sh_1"][l]), self.SILU, self.SILU_slope)
                g = rqr(g1 * rqr(y @ W("W3").T, sh["sh_3"][l]), sh["sh_g"][l])
            else:
                g = rqr(y @ W("W1").T, sh["sh_1"][l], relu=True)
            x = rq(256 * x + 128 + g @ W("W2").T, 8)
        y = self.norm(2 * cfg.n_layers, x)
        # the floor of the logits at 0 is treated as a clamp, not as a ReLU: a token below it can still be pulled up
        u = rq(y @ E.T + self.c_l(), sh["sh_l"], lo=0)
        if targets is None:
            return u, None
        return u, F.cross_entropy((u * 2.0 ** -self.frac["u"]).view(-1, cfg.vocab_size), targets.reshape(-1))

    @torch.no_grad()
    def clip(self):
        """Keep every latent weight inside what int8 holds at its scale (call after each optimiser step)."""
        for key, w in self.latent.items():
            w.clamp_(-127 / self.scale(key), 127 / self.scale(key))

    @torch.no_grad()
    def integers(self):
        """(int8 arrays of model.npz, preloads of model.json): SPEC Part E, exactly what forward() computes with."""
        arrays = {key: self.weight(key) for key in self.latent}
        arrays.update({name: getattr(self, name) for name in ("NORM", "SILU", "EXP", "RECIP", "ROPE_C", "ROPE_S")
                       if hasattr(self, name)})
        preloads = {"c_s": [self.c_s(l).int().tolist() for l in range(self.cfg.n_layers)], "c_l": self.c_l()}
        return {name: a.cpu().numpy().astype(np.int8) for name, a in arrays.items()}, preloads
