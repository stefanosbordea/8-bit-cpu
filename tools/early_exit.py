"""Early exit, measured: if the logits were read after layer k instead of after the last layer, how often would the
chosen token be the same, and how sure does the model have to be for that? Validation text, float model.

For a threshold on the largest probability after layer k the script reports the share of positions that would
exit (probability above the threshold) and how many of those pick the same token as the full model. With the
cycle counts of the LS program this gives the speed that early exit could buy: an exit after layer k needs one
more output layer for every token and saves the layers behind k (except their key and value projections, which
later tokens still need) for the tokens that exit.

    PYTHONPATH=. .venv/bin/python tools/early_exit.py runs/NAME/ckpt.pt
"""
import os
import sys

import numpy as np
import torch

from lsllm.config import ModelConfig
from lsllm.ls import cost
from lsllm.model.llama import load

model, saved = load(sys.argv[1], "cpu")
model.eval()
cfg, T = model.cfg, model.cfg.max_seq_len
val = np.fromfile(os.path.join(saved["tok"], "val.bin"), dtype=np.uint16).astype(np.int64)
starts = np.linspace(0, len(val) - T - 1, 64).astype(np.int64)
tokens = torch.from_numpy(val[starts[:, None] + np.arange(T)])
with torch.no_grad():
    x = model.tok_embeddings(tokens)
    cos, sin, hide = model.cos[:T], model.sin[:T], model.hide[:T, :T]
    probs = []
    for layer in model.layers:
        x = layer(x, cos, sin, hide)
        probs.append(torch.softmax(model.output(model.norm(x)), -1))
final = probs[-1].argmax(-1)
options = {"norm": "rms", "ffn": cfg.ffn}
shape = ModelConfig(**dict(cfg.to_dict(), attn="softmax", window=0))
total, parts = cost.cycles_per_token(shape, options, 63)
out_layer = parts["output"] + parts["argmax"]
per_layer = (total - out_layer - parts["embedding"] - parts["book"]) / cfg.n_layers
kv_only = (parts["qkv"] * 2 / 3 + parts["norm"] / (2 * cfg.n_layers + 1) * cfg.n_layers) / cfg.n_layers    # norm + k, v of a skipped layer
print("model %s: %d layers; cycles at context 64: %d per token, %d per layer, %d for output layer and argmax" %
      (sys.argv[1], cfg.n_layers, total, per_layer, out_layer))
print("exit after layer | threshold | share of positions that exit | same token as the full model | cycles per token | speed-up")
for k in range(1, cfg.n_layers):
    top, pick = probs[k - 1].max(-1)
    saved_cycles = (cfg.n_layers - k) * (per_layer - kv_only)
    for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
        exits = top > threshold
        share = exits.float().mean().item()
        same = (pick[exits] == final[exits]).float().mean().item() if exits.any() else float("nan")
        cycles = total + out_layer - share * (saved_cycles + out_layer)     # the exit test is paid by all, the last output layer only by the rest
        print("%d | %.2f | %.1f%% | %.1f%% | %d | %.3f" % (k, threshold, 100 * share, 100 * same, cycles, total / cycles))
