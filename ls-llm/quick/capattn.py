"""Float stand-in for the integer attention of the LS CPU ("cap-referenced softmax").

Scores get a per-head offset (free on the CPU: the accumulator preload), are coded as int8 at 8 codes per nat,
go through a 7-bit exp table referenced to the top code 127 (weights 0..127, window of about 5.5 nats, no search
for the row maximum), and the mixed values are divided by the sum of the weights (reciprocal table on the CPU).
"""
import math

import torch
from torch import nn

from lsllm.model import llama

CODES_PER_NAT = 8.0


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def cap_forward(self, x, cos, sin, hide):
    cfg = self.cfg
    B, T, _ = x.shape
    share = cfg.n_heads // cfg.n_kv_heads
    q = llama.rotate(self.wq(x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2), cos, sin)
    k = llama.rotate(self.wk(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2), cos, sin)
    v = self.wv(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
    q, k, v = q.view(B, cfg.n_kv_heads, share, T, cfg.head_dim), k[:, :, None], v[:, :, None]
    scores = (q / math.sqrt(cfg.head_dim)) @ k.transpose(-2, -1) + self.cap_b.view(1, cfg.n_kv_heads, share, 1, 1)
    code = ste_round(scores * CODES_PER_NAT)
    code = code + (code.clamp(-128, 127) - code).detach()
    weights = ste_round(127.0 * torch.exp((code - 127.0) / CODES_PER_NAT))
    weights = weights.masked_fill(~torch.isfinite(hide), 0.0)
    if getattr(self, "cap_norm", True):
        out = (weights @ v) / weights.sum(-1, keepdim=True).clamp_min(1.0)
    else:
        out = (weights @ v) / 127.0
    out = out.view(B, cfg.n_heads, T, cfg.head_dim).transpose(1, 2).reshape(B, T, cfg.dim)
    return self.wo(out)


def install(model, normalise=True, margin_nats=1.0):
    """Switch every attention block of `model` to the cap-referenced form."""
    cfg = model.cfg
    start = 127.0 / CODES_PER_NAT - margin_nats - (0.0 if normalise else math.log(cfg.max_seq_len))
    for layer in model.layers:
        att = layer.attention
        att.cap_b = nn.Parameter(torch.full((cfg.n_heads,), start))
        att.cap_norm = normalise
        att.forward = cap_forward.__get__(att)
    return model
