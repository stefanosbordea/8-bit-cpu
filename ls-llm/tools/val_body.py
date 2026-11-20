"""Validation loss of a float checkpoint, over all tokens and over the body tokens only (see lsllm.train.run).

    PYTHONPATH=. .venv/bin/python tools/val_body.py runs/NAME/ckpt.pt [data/tok/NAME]
"""
import os
import sys

import numpy as np

from lsllm.model.llama import load
from lsllm.tok import Tok
from lsllm.train.run import evaluate_parts, prompt_flags, windows

model, saved = load(sys.argv[1], "cpu")
data = sys.argv[2] if len(sys.argv) > 2 else saved["tok"]
T = model.cfg.max_seq_len
val = np.fromfile(os.path.join(data, "val.bin"), dtype=np.uint16)
starts = np.linspace(0, len(val) - T - 1, max(1, min(512, len(val) // (T + 1)))).astype(np.int64)
flags = prompt_flags(val, Tok(os.path.join(data, "tok.model")))
total, body = evaluate_parts(model.eval(), windows(val, starts, T), windows(flags, starts, T), 16)
print("%s: val %.4f nats/token, body only %.4f (%d windows)" % (sys.argv[1], total, body, len(starts)))
