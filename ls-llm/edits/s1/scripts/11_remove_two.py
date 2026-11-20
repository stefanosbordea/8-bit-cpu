"""One budgeted evaluate of direction removals with a different direction and alpha per tensor (script 05 takes one
direction for all its tensors). Items: tensor:file:name:alpha, e.g. Wo_0:04:l0_xa:1 Wo_1:04:l1_xa:1.
The edit is taken back after the run unless --keep is given.
Run from the repository root, e.g.:
PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/11_remove_two.py Wo_0:04:l0_xa:1 Wo_1:04:l1_xa:1"""
import argparse

import numpy as np

from common import DIR, report, restore
from lsllm.edit.session import Session

ap = argparse.ArgumentParser()
ap.add_argument("items", nargs="+")
ap.add_argument("--keep", action="store_true")
ap.add_argument("--base", type=int, default=1)
a = ap.parse_args()

s = Session(DIR)
note = "remove directions " + " ".join(a.items)
saved = {}
for item in a.items:
    tensor, which, name, alpha = item.split(":")
    u = np.load("%s/directions_%s.npz" % (DIR, which))[name]
    W = s.get(tensor)
    saved[tensor] = W.copy()
    print(tensor, "changed", s.put(tensor, np.clip(np.round(W - float(alpha) * np.outer(u, u @ W)), -127, 127), note)["changed"])
report(s.evaluate(note), a.base)
if not a.keep:
    restore(s, saved, "undo: " + note)
    print("taken back")
