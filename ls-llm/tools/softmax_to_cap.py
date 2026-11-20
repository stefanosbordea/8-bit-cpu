#!/usr/bin/env python3
"""Turn a float checkpoint with softmax attention into the integer-style attention of SPEC C.6.

The weights are copied. The per-head offset is set from data: for every head, the median over positions of the
largest score in the row is moved to `margin` nats below the top code of the exp table, so that the tokens a head
attends to land in the table's window. Prints the validation loss before and after (no training).

    .venv/bin/python tools/softmax_to_cap.py runs/A/ckpt.pt runs/B/ckpt.pt [cappow2|capnorm] [margin]
"""
import json
import math
import sys
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, ".")
from lsllm.config import ModelConfig
from lsllm.model.llama import CODES_PER_NAT, build, load, rotate

torch.set_num_threads(2)
src, dst = sys.argv[1], sys.argv[2]
attn = sys.argv[3] if len(sys.argv) > 3 else "cappow2"
margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
model, saved = load(src)
model.eval()
cfg = model.cfg
T = cfg.max_seq_len
val = np.fromfile(saved["tok"] + "/val.bin", dtype=np.uint16).astype(np.int64)
starts = np.linspace(0, len(val) - T - 1, 24).astype(np.int64)
rows = torch.from_numpy(np.stack([val[g:g + T + 1] for g in starts]))
x_tok, y_tok = rows[:, :-1], rows[:, 1:]


@torch.no_grad()
def row_maxima(model, tokens):
    """Per layer and head: the largest attention score of every row (batch x time values)."""
    out = []
    x = model.tok_embeddings(tokens)
    cos, sin, hide = model.cos[:T], model.sin[:T], model.hide[:T, :T]
    share = cfg.n_heads // cfg.n_kv_heads
    B = tokens.shape[0]
    for layer in model.layers:
        att = layer.attention
        h = layer.attention_norm(x)
        q = rotate(att.wq(h).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2), cos, sin)
        k = rotate(att.wk(h).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2), cos, sin)
        q, k = q.view(B, cfg.n_kv_heads, share, T, cfg.head_dim), k[:, :, None]
        scores = (q / math.sqrt(cfg.head_dim)) @ k.transpose(-2, -1) + hide
        out.append(scores.amax(-1).reshape(B, cfg.n_heads, T))
        x = layer(x, cos, sin, hide)
    return out


@torch.no_grad()
def val_loss(m):
    return sum(m(x_tok[g:g + 8], y_tok[g:g + 8])[1].item() for g in range(0, len(x_tok), 8)) / (len(x_tok) // 8)


before = val_loss(model)
new = build(replace(cfg, attn=attn))
state = new.state_dict()
state.update(model.state_dict())
spread = []
for g, maxima in enumerate(row_maxima(model, x_tok)):
    med = maxima.transpose(0, 1).reshape(cfg.n_heads, -1).median(dim=1).values
    lo = maxima.transpose(0, 1).reshape(cfg.n_heads, -1).quantile(0.1, dim=1)
    hi = maxima.transpose(0, 1).reshape(cfg.n_heads, -1).quantile(0.9, dim=1)
    state["layers.%d.attention.cap_b" % g] = 127.0 / CODES_PER_NAT - margin - med
    spread.append([round(float(v), 2) for v in (hi - lo)])
new.load_state_dict(state)
new.eval()
after = val_loss(new)
torch.save({"config": new.cfg.to_dict(), "model": new.state_dict(), "step": saved.get("step", 0), "val_loss": after, "tok": saved["tok"]}, dst)
print(json.dumps({"attn": attn, "margin_nats": margin, "val_softmax": round(before, 4), "val_converted": round(after, 4),
                  "row_max_spread_10_to_90_percent_nats_per_layer_head": spread}))
