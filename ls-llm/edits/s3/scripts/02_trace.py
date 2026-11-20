"""Run 2 (trace): practice prompts followed by their reference solutions, teacher forced.
Keeps every fixed-shape intermediate value that the hardware knobs act on, in edits/s3/trace_02.npz."""
import numpy as np
from common import DIR, open_session, practice

s, tok = open_session()
tasks = practice()
seqs, used, total = [], [], 0
for g in range(0, 200, 4):
    ids = tok.encode(tasks[g]["canonical_solution"], bos=True)
    if total + len(ids) > 4096:
        break
    seqs.append(ids); used.append(g); total += len(ids)
names = ["x0", "u", "u.acc", "y", "y.m", "y.r"]
for l in range(2):
    at = "l%d." % l
    names += [at + n for n in ("ya", "ya.m", "ya.r", "yf", "yf.m", "yf.r", "q", "k", "v", "qr", "kr", "max", "dsum",
                               "mant", "a", "o", "xa", "xf", "g")]
out = s.trace(seqs, names, "prompt + reference solution of %d practice tasks, knob statistics" % len(seqs))
keep = {"used": np.array(used), "lengths": np.array([len(q) for q in seqs]),
        "tokens": np.concatenate([np.array(q) for q in seqs])}
for name in names:
    keep[name] = np.concatenate([o[name] for o in out])
np.savez_compressed(DIR + "/trace_02.npz", **keep)
print(len(seqs), "sequences", total, "tokens; budget left", s.left())
