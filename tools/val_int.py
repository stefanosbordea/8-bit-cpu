"""Validation loss of integer models (the cross-entropy of their integer logits), all tokens and body tokens only.

    PYTHONPATH=. .venv/bin/python tools/val_int.py DATA_DIR MODEL_DIR [MODEL_DIR ...]

The windows are those of lsllm.train.run and lsllm.qat.train, so the numbers compare with their logs.
"""
import os
import sys

import numpy as np

from lsllm.intref import IntModel
from lsllm.tok import Tok
from lsllm.train.run import prompt_flags

data = sys.argv[1]
val = np.fromfile(os.path.join(data, "val.bin"), dtype=np.uint16)
flags = prompt_flags(val, Tok(os.path.join(data, "tok.model")))
for folder in sys.argv[2:]:
    model = IntModel.load(folder)
    T = model.cfg.max_seq_len
    starts = np.linspace(0, len(val) - T - 1, max(1, min(512, len(val) // (T + 1)))).astype(np.int64)
    rows = val[starts[:, None] + np.arange(T + 1)].astype(np.int64)
    part = flags[starts[:, None] + np.arange(T + 1)][:, 1:]
    total = body = n_body = same = 0.0
    for g in range(0, len(rows), 64):
        u = model.forward(rows[g:g + 64, :-1]) * 2.0 ** -model.f_l
        u = u - u.max(-1, keepdims=True)
        logp = u - np.log(np.exp(u).sum(-1, keepdims=True))
        each = -np.take_along_axis(logp, rows[g:g + 64, 1:, None], -1)[..., 0]
        keep = 1.0 - part[g:g + 64]
        total += each.sum()
        body += (each * keep).sum()
        n_body += keep.sum()
    print("%s: val %.4f nats/token, body only %.4f (%d windows)" % (folder, total / (len(rows) * T), body / n_body, len(rows)))
