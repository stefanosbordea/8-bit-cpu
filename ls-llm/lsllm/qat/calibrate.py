"""From a trained float model to a QatModel that starts as close to it as int8 allows (SPEC B.5).

Every tensor gets a power-of-two scale, real value = integer / 2^f:
- a weight matrix: the largest f at which its largest weight still fits in 127;
- an activation tensor: the largest f at which the pct-th percentile of its size on calibration data fits
  (larger values saturate; the percentile is taken per batch, the largest batch counts);
- a matrix that writes into the residual stream has no free scale (SPEC B.3: f_W + f_h = f_x + 8): the scale of
  the vector h it reads is set halfway between "the weights fit" and "h fits".
What the integer graph has no place for is folded into the float weights first, without changing the function:
per-channel norm gains go into the matrices that read the norm, the embedding scale into E, 1/sqrt(d) into Wq.
A shift of an rqr is at most 15, because its rounding preload 2^(sh - 1) must fit a signed 16-bit value.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from lsllm.model.llama import rotate
from lsllm.qat.model import QatModel

F_S, TARGET = 2, 80             # attention (SPEC C.6, D-arch-3): 2^2 = 4 score codes per nat; the median over the
                                # positions of the largest score of a head is put at code 80
PRELOAD = 2 ** 15 - 1           # largest 16-bit preload
RQR_MAX = 15                    # largest shift of an rqr: its rounding preload 2^(sh - 1) must fit 16 bits


def fbits(size):
    """Largest f with size * 2^f <= 127."""
    return math.floor(math.log2(127.0 / max(size, 1e-9)))


def linear(fw, f_in, f_out):
    """(shift, weight scale) of y = rqr(W x, sh) for an output scale f_out <= fw + f_in: sh = fw + f_in - f_out."""
    fw = min(fw, f_out + RQR_MAX - f_in)        # only for tiny weights: a coarser weight scale keeps sh at 15
    return fw + f_in - f_out, fw


def residual(f_x, w_top, h_size):
    """Scale of the vector h read by a residual matrix W: the weights fit from lo on, h fits up to hi."""
    lo, hi = f_x + 8 - fbits(w_top), fbits(h_size)
    return (lo + hi + 1) // 2


def measure(model, sites, token_batches, pct):
    """Run the float model on the batches. Returns (size, top): size[name] is the pct-th percentile of |tensor|
    (largest batch), top[l] for every head of layer l the median over the positions of its largest score."""
    cfg, size, top, queries, hooks = model.cfg, {}, {}, {}, []

    def keep(name, value, pick=max):
        size[name] = pick(size.get(name, float(value)), float(value))

    def level(tensor, q=pct):
        return np.percentile(tensor.cpu().numpy(), q)

    def watch(module, function):
        hooks.append(module.register_forward_hook(function))

    def norm_hook(n):
        def hook(module, args, out):
            x = args[0]
            keep("x", level(x.abs()))
            if cfg.norm != "dyt":
                S = x.pow(2).sum(-1)                                    # sum of squares per position
                keep("S_hi%d" % n, level(S))
                keep("S_lo%d" % n, level(S, 100 - pct), min)
                keep("y%d" % n, level((x * torch.rsqrt(S[..., None] / cfg.dim)).abs()))
        return hook

    def out_hook(name, also=None):
        def hook(module, args, out):
            keep(name, level(out.abs()))
            if also:
                keep(also, level(F.silu(out).abs()))
        return hook

    def in_hook(name):
        return lambda module, args, out: keep(name, level(args[0].abs()))

    def score_hook(l):
        def hook(module, args, out):                                    # called on the keys; the queries came first
            B, T, _ = out.shape
            G, d, share = cfg.n_kv_heads, cfg.head_dim, cfg.n_heads // cfg.n_kv_heads
            q = rotate(queries[l].view(B, T, cfg.n_heads, d).transpose(1, 2), model.cos[:T], model.sin[:T])
            k = rotate(out.view(B, T, G, d).transpose(1, 2), model.cos[:T], model.sin[:T])
            scores = q.view(B, G, share, T, d) @ k[:, :, None].transpose(-2, -1) / math.sqrt(d) + model.hide[:T, :T]
            now = torch.amax(scores, -1).view(B, cfg.n_heads, T).transpose(0, 1).reshape(cfg.n_heads, -1)
            top.setdefault(l, []).append(now.median(dim=-1).values)
        return hook

    for n, site in enumerate(sites):
        watch(site, norm_hook(n))
    for l, layer in enumerate(model.layers):
        att, ffn = layer.attention, layer.feed_forward
        watch(att.wq, out_hook("q%d" % l))
        watch(att.wk, out_hook("k%d" % l))
        watch(att.wv, out_hook("v%d" % l))
        watch(att.wo, in_hook("o%d" % l))
        watch(att.wq, lambda module, args, out, l=l: queries.update({l: out}))
        watch(att.wk, score_hook(l))
        if cfg.ffn == "swiglu":
            watch(ffn.w1, out_hook("h1", "g1"))
            watch(ffn.w3, out_hook("h3%d" % l))
        watch(ffn.w2, in_hook("g%d" % l))

    def top_hook(module, args, out):
        top = out.max(-1).values                                        # the largest logit of every position
        keep("top_hi", level(top))
        keep("top_lo", level(top, 100 - pct), min)

    watch(model.output, top_hook)
    for tokens in token_batches:
        model(tokens)
    for hook in hooks:
        hook.remove()
    return size, {l: torch.stack(rows).mean(0) for l, rows in top.items()}


@torch.no_grad()
def calibrate(model, token_batches, pct=99.9, logit_bits=None, stream_bits=None):
    """model: a trained lsllm.model.llama.Llama with softmax attention; token_batches: (batch, time) token
    tensors of real data. Returns a QatModel; its `notes` list what calibration could not do well.
    Attention (SPEC C.6) is softmax again, with the scores coded in 7 bits relative to the largest one of the
    row: nothing is folded, only the offset of every head is chosen, so that few codes fall off the ends 0 and 127.
    logit_bits: codes per nat of the logits as a power of two, if finer than the rule below would choose. The rule
    keeps the largest logit of every position inside 0..127; a finer scale lets the strongest of them stop at 127.
    stream_bits: fractional bits of the residual stream, if more than the rule would choose. The rule fits the 99.9th
    percentile of |x| over all sites and dimensions; a model with one very large dimension (seen after long training:
    one dimension of the last site up to 39 while the others stay below 10) then loses 3 bits for all the others.
    More bits let that dimension stop at +-127; QAT has to make up for it."""
    cfg = model.cfg
    assert cfg.tied_embeddings, "SPEC Part C has one matrix E"
    if cfg.attn != "softmax":
        raise ValueError("attention %r has no integer form (SPEC C.6 is softmax)" % cfg.attn)
    L, D, d, V = cfg.n_layers, cfg.dim, cfg.head_dim, cfg.vocab_size
    dyt, swiglu, last = cfg.norm == "dyt", cfg.ffn == "swiglu", 2 * cfg.n_layers
    sites = [norm for layer in model.layers for norm in (layer.attention_norm, layer.ffn_norm)] + [model.norm]
    size, score_top = measure(model, sites, token_batches, pct)
    notes = []

    # 1. the float weights, with everything folded in that the integer graph has no place for
    def gain(n):                                    # per-channel gain of norm site n, if it has one
        return getattr(sites[n], "weight", torch.ones(D))

    emb = float(getattr(model, "emb_scale", 1.0))   # the stream starts as emb * E[token]
    W, b = {"E": model.output.weight * emb}, []
    for l, layer in enumerate(model.layers):
        att, ffn = layer.attention, layer.feed_forward
        # scores are q.k / sqrt(d); the integer RoPE tables are 127/128 of a rotation, for q and for k
        W["Wq_%d" % l] = att.wq.weight * gain(2 * l) * ((128 / 127) ** 2 / math.sqrt(d))
        W["Wk_%d" % l] = att.wk.weight * gain(2 * l)
        W["Wv_%d" % l] = att.wv.weight * gain(2 * l)
        W["Wo_%d" % l] = att.wo.weight
        W["W1_%d" % l] = ffn.w1.weight * gain(2 * l + 1)
        if swiglu:
            W["W3_%d" % l] = ffn.w3.weight * gain(2 * l + 1)
        W["W2_%d" % l] = ffn.w2.weight
    # a per-channel gain in front of the tied matrix E cannot be folded: its mean goes into the final table,
    # together with 1 / emb, which turns emb * E back into E for the logits
    amp = [1.0] * last + [float(gain(last).mean()) / emb]
    top = {name: float(w.abs().max()) for name, w in W.items()}

    # 2. scales of the stream, the norm outputs and the logits
    f = {"x": fbits(size["x"])}
    if stream_bits is not None and stream_bits > f["x"]:
        notes.append("stream: %d fractional bits by request instead of %d; values beyond %.2f stop there (99.9th percentile on the data: %.2f)"
                     % (stream_bits, f["x"], 127 * 2.0 ** -stream_bits, size["x"]))
        f["x"] = stream_bits
    alpha = [float(site.alpha) for site in sites] if dyt else None
    if dyt:
        # a dyt table is flat (+-127) beyond |alpha x| = atanh(127/128): no site reads a difference there,
        # so the stream needs no more range than the site with the smallest alpha can tell apart
        flat = math.atanh(127 / 128) / max(min(abs(a) for a in alpha), 1e-9)
        f["x"] = max(f["x"], fbits(flat))
    # x = rqr(E[token], sh_e) (SPEC C.1): E keeps the scale at which its largest weight fits int8, which is what the
    # output layer reads; the embedding lookup shifts it down to the scale of the stream
    f["E"] = min(fbits(top["E"]), f["x"] + RQR_MAX)
    f["x"] = min(f["x"], f["E"])
    f["y"] = [math.floor(math.log2(128 / amp[n])) if dyt else fbits(amp[n] * size["y%d" % n])
              for n in range(last + 1)]
    # u = (z + offset) * 2^f_l in 0..127 must hold the largest logit of a position, which lies between top_lo and
    # top_hi on the data, and a tail of ln(V) + 3 nats below it (then the tokens floored at 0 weigh under 5%).
    # Of the offsets (in nats) that do this, the one nearest to 0 is taken: the preload c_l = offset * 2^(f_l + sh_l)
    # has 16 bits, so a small offset allows a large shift sh_l, which means more bits for the final norm output
    tail = math.log(V) + 3
    f["u"] = min(fbits(size["top_hi"] - size["top_lo"] + tail), f["E"] + f["y"][last])
    offset = min(max(0.0, tail - size["top_lo"]), 127 * 2.0 ** -f["u"] - size["top_hi"])
    if logit_bits is not None and min(logit_bits, f["E"] + f["y"][last]) > f["u"]:
        # A model that knows texts by heart has very large logits where it is sure, which makes the rule choose
        # coarse codes for all positions. Finer codes keep the tail below a weak top; a top above the range stops
        # at 127, where it still wins unless a second token gets there too (QAT sees that case and trains it away).
        f["u"] = min(logit_bits, f["E"] + f["y"][last])
        offset = max(0.0, tail - size["top_lo"])
        notes.append("logits: %d codes per nat by request; a logit above %.1f nats stops at code 127 (largest on the data: %.1f)"
                     % (2 ** f["u"], 127 * 2.0 ** -f["u"] - offset, size["top_hi"]))
    cap = min(23, math.floor(math.log2(PRELOAD / (abs(offset) * 2.0 ** f["u"] + 1))))   # so that c_l fits 16 bits
    f["y"][last] -= max(0, f["E"] + f["y"][last] - f["u"] - cap)
    sh = {"sh_l": f["E"] + f["y"][last] - f["u"], "sh_e": f["E"] - f["x"]}

    # 3. layer by layer
    for name in ("q", "k", "v", "o", "s", "g", "Wq", "Wk", "Wv", "Wo", "W1", "W2") + ("h3", "W3") * swiglu:
        f[name] = [0] * L
    for name in ("sh_q", "sh_k", "sh_v", "sh_s", "sh_o", "sh_1") + ("sh_3", "sh_g") * swiglu:
        sh[name] = [0] * L
    fw = {name: fbits(value) for name, value in top.items()}
    if swiglu:                                                  # one SILU table for all layers: one h1, g1 scale
        f["h1"] = min([fbits(size["h1"])] + [fw["W1_%d" % l] + f["y"][2 * l + 1] for l in range(L)])
        f["g1"] = fbits(size["g1"])
    for l in range(L):
        ya, yf = f["y"][2 * l], f["y"][2 * l + 1]
        tag = "_%d" % l
        # attention scores: code = 4 * (q.k + b) = (c_s + sum q k) >> sh_s with sh_s = f_q + f_k - 2, and b such
        # that the usual largest score gets the code TARGET. The shift must be small enough for the preload
        # c_s = (TARGET - 4 * usual largest score + 0.5) * 2^sh_s to fit 16 bits
        cap = math.floor(math.log2(PRELOAD / (float((TARGET - 2 ** F_S * score_top[l]).abs().max()) + 1)))
        want = {"q": fbits(size["q%d" % l] * (128 / 127) ** 2 / math.sqrt(d)), "k": fbits(size["k%d" % l]),
                "v": fbits(size["v%d" % l])}
        want = {name: min(value, fw["W" + name + tag] + ya) for name, value in want.items()}
        while want["q"] + want["k"] - F_S > cap:
            want["q" if want["q"] >= want["k"] else "k"] -= 1
        for name in ("q", "k", "v"):
            f[name][l] = want[name]
            sh["sh_" + name][l], f["W" + name][l] = linear(fw["W" + name + tag], ya, want[name])
        sh["sh_s"][l] = max(0, f["q"][l] + f["k"][l] - F_S)
        f["s"][l] = f["q"][l] + f["k"][l] - sh["sh_s"][l]
        if f["s"][l] < F_S:     # q and k are coarse and a shift cannot go left: fewer codes per nat, same table rule
            notes.append("layer %d: the scores have %g codes per nat instead of %d (f_q + f_k = %d)"
                         % (l, 2.0 ** f["s"][l], 2 ** F_S, f["s"][l]))
        b.append(TARGET * 2.0 ** -f["s"][l] - score_top[l])     # the offset of every head, in nats
        # head outputs: o = (64 * weighted mean of v) >> sh_o (SPEC C.6 step 5)
        most = f["v"][l] + 6
        f["o"][l] = residual(f["x"], top["Wo" + tag], size["o%d" % l])
        f["o"][l] = min(max(f["o"][l], most - RQR_MAX), most)
        sh["sh_o"][l], f["Wo"][l] = most - f["o"][l], f["x"] + 8 - f["o"][l]
        # feed-forward
        f["g"][l] = residual(f["x"], top["W2" + tag], size["g%d" % l])
        if swiglu:
            sh["sh_1"][l], f["W1"][l] = linear(fw["W1" + tag], yf, f["h1"])
            f["h3"][l] = min(fbits(size["h3%d" % l]), fw["W3" + tag] + yf)
            sh["sh_3"][l], f["W3"][l] = linear(fw["W3" + tag], yf, f["h3"][l])
            f["g"][l] = min(max(f["g"][l], f["g1"] + f["h3"][l] - RQR_MAX), f["g1"] + f["h3"][l])
            sh["sh_g"][l] = f["g1"] + f["h3"][l] - f["g"][l]
        else:
            f["g"][l] = min(f["g"][l], fw["W1" + tag] + yf)
            sh["sh_1"][l], f["W1"][l] = linear(fw["W1" + tag], yf, f["g"][l])
        f["W2"][l] = f["x"] + 8 - f["g"][l]

    # 4. rms norm: m = sum(x^2) >> sa must stay below 128; the table multiplier r must fit 127 for the small sums
    if not dyt:
        sh["sa"], sh["sb"] = [0] * (last + 1), [0] * (last + 1)
        for n in range(last + 1):
            hi, lo = size["S_hi%d" % n] * 4.0 ** f["x"], size["S_lo%d" % n] * 4.0 ** f["x"]     # in integer units
            sa = max(0, math.floor(math.log2(max(hi, 1.0) / 128)) + 1)
            r = amp[n] * 2.0 ** f["y"][n] * math.sqrt(D / ((math.floor(lo / 2 ** sa) + 0.5) * 2 ** sa))
            sh["sa"][n], sh["sb"][n] = sa, min(max(math.floor(math.log2(127 / r)), 0), RQR_MAX)

    options = {"norm": "dyt" if dyt else "rms", "ffn": cfg.ffn}
    qat = QatModel(cfg, options, f, sh, W, torch.stack(b), offset, alpha, amp[last])
    # 5. how much of int8 every matrix uses at its scale (E and the residual matrices cannot choose theirs)
    for key in qat.latent:
        used = int(qat.weight(key).abs().max())
        if used == 0:
            notes.append("%s rounds to all zeros: it does nothing in the integer model" % key)
        elif used < 32:
            notes.append("%s uses only -%d..%d of -127..127" % (key, used, used))
    qat.notes = notes
    return qat
