#!/usr/bin/env python3
"""Validation loss of a checkpoint per data source (human-written functions, Qwen-written simple functions).
Usage: .venv/bin/python tools/val_by_source.py runs/NAME/ckpt.pt [data/final/f512a]"""
import json
import math
import sys

import torch

sys.path.insert(0, ".")
from lsllm.model.llama import load
from lsllm.tok import Tok

torch.set_num_threads(2)
ckpt, base = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "data/final/f512a")
model, saved = load(ckpt)
model.eval()
tok = Tok(saved["tok"] + "/tok.model")
T = model.cfg.max_seq_len
out = {}
for name in ("human_val", "qwen_val"):
    nats = tokens = nbytes = 0
    for line in open("%s/%s.jsonl" % (base, name)):
        text = json.loads(line)["text"]
        ids = tok.encode(text, bos=True, eos=True)[:T + 1]
        if len(ids) < 3:
            continue
        x = torch.tensor([ids[:-1]])
        y = torch.tensor([ids[1:]])
        with torch.no_grad():
            loss = model(x, y)[1].item()
        nats += loss * (len(ids) - 1)
        tokens += len(ids) - 1
        nbytes += len(text)
    out[name] = {"nats_per_token": round(nats / tokens, 4), "bits_per_byte": round(nats / (nbytes * math.log(2)), 4), "tokens": tokens}
print(json.dumps(out))
