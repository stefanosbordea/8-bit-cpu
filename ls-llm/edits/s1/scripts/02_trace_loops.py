"""Run 2 (trace): prompt + baseline completion of the 8 repeating answers and of every fifth other answer.
Keeps residual stream, feed-forward units, head outputs, attention sharpness and logits in trace_02.pkl.
Run from the repository root: PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/02_trace_loops.py"""
import pickle

from common import DIR, encode, load, repeats, tokenizer
from lsllm.edit.session import Session

NAMES = ["x0", "y", "u", "u.acc"] + ["l%d.%s" % (l, n) for l in (0, 1)
                                     for n in ("ya", "yf", "g", "o", "xa", "xf", "max", "dsum", "qr", "kr", "v")]
s = Session(DIR)
tok = tokenizer(s)
tasks, rows = load(1)
chosen, total = [], 0
order = [g for g in range(200) if repeats(rows[g]["completion"])] + list(range(2, 200, 5))
for g in dict.fromkeys(order):
    ids, n = encode(tok, tasks[g]["prompt"], rows[g]["completion"])
    if total + len(ids) > 4096:
        continue
    total += len(ids)
    chosen.append({"task": g, "ids": ids, "n_prompt": n, "repeats": repeats(rows[g]["completion"])})
out = s.trace([c["ids"] for c in chosen], NAMES, "baseline: 8 repeating answers and %d others, %d tokens" % (len(chosen) - 8, total))
for c, o in zip(chosen, out):
    c["trace"] = o
with open(DIR + "/trace_02.pkl", "wb") as f:
    pickle.dump(chosen, f)
print(len(chosen), total, s.left())
