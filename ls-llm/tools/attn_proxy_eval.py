#!/usr/bin/env python3
"""Float evaluation of the integer attention design on a trained softmax checkpoint (no training).

Scheme: score code = clamp(round(kappa * (score + offset_head)), 0, 127) (unsigned 7-bit, so the CPU can find the
row maximum with SUB + sign table); weight = round(127 * exp((code - rowmax) / kappa)); output = mix / sum.
offset_head puts the median row maximum of the head at code `target`.
Usage: .venv/bin/python tools/attn_proxy_eval.py CKPT
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from lsllm.model import llama

torch.set_num_threads(2)
model, saved = llama.load(sys.argv[1])
model.eval()
cfg = model.cfg
T = cfg.max_seq_len
val = np.fromfile(saved["tok"] + "/val.bin", dtype=np.uint16).astype(np.int64)
starts = np.linspace(0, len(val) - T - 1, 40).astype(np.int64)
rows = torch.from_numpy(np.stack([val[g:g + T + 1] for g in starts]))
x, y = rows[:, :-1], rows[:, 1:]
SET = {"kappa": None, "target": 100, "offsets": None, "clip": [0, 0, 0]}


def forward(self, xx, cos, sin, hide):
    B, TT, _ = xx.shape
    share = cfg.n_heads // cfg.n_kv_heads
    q = llama.rotate(self.wq(xx).view(B, TT, cfg.n_heads, cfg.head_dim).transpose(1, 2), cos, sin)
    k = llama.rotate(self.wk(xx).view(B, TT, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2), cos, sin)
    v = self.wv(xx).view(B, TT, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
    q, k, v = q.view(B, cfg.n_kv_heads, share, TT, cfg.head_dim), k[:, :, None], v[:, :, None]
    scores = (q / math.sqrt(cfg.head_dim)) @ k.transpose(-2, -1) + hide
    if SET["kappa"] is None:                               # calibration pass / plain softmax
        self.rowmax = scores.amax(-1)
        w = torch.softmax(scores, -1)
    else:
        seen = torch.isfinite(scores)
        code = torch.round(SET["kappa"] * (torch.where(seen, scores, torch.zeros_like(scores)) + self.offset.view(1, cfg.n_kv_heads, share, 1, 1)))
        SET["clip"][0] += int(((code < 0) & seen).sum()); SET["clip"][1] += int(((code > 127) & seen).sum()); SET["clip"][2] += int(seen.sum())
        code = code.clamp(0, 127)
        top = torch.where(seen, code, torch.zeros_like(code)).amax(-1, keepdim=True)
        w = torch.round(127.0 * torch.exp((code - top) / SET["kappa"])) * seen
        w = w / w.sum(-1, keepdim=True)
    out = (w @ v).view(B, cfg.n_heads, TT, cfg.head_dim).transpose(1, 2).reshape(B, TT, cfg.dim)
    return self.wo(out)


llama.Attention.forward = forward


@torch.no_grad()
def loss():
    return sum(model(x[g:g + 8], y[g:g + 8])[1].item() for g in range(0, len(x), 8)) / (len(x) // 8)


base = loss()
med = []
with torch.no_grad():
    model(x[:16])
for layer in model.layers:
    med.append(layer.attention.rowmax.transpose(0, 2).reshape(cfg.n_kv_heads * (cfg.n_heads // cfg.n_kv_heads), -1).median(dim=1).values if False else
               layer.attention.rowmax.permute(1, 2, 0, 3).reshape(cfg.n_heads, -1).median(dim=1).values)
print("softmax %.4f" % base)
for kappa in (3.0, 4.0, 5.0):
    for target in (72, 84, 96):
        SET["kappa"], SET["clip"] = kappa, [0, 0, 0]
        for layer, m in zip(model.layers, med):
            layer.attention.offset = target / kappa - m
        v = loss()
        print("kappa %.0f target %3d: %.4f (+%.4f)  codes clipped low %.2f%% high %.2f%%" % (kappa, target, v, v - base, 100 * SET["clip"][0] / SET["clip"][2], 100 * SET["clip"][1] / SET["clip"][2]))
