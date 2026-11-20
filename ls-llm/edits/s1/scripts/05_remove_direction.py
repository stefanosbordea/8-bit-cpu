"""One budgeted evaluate of: remove a direction u of the residual stream from what a layer writes,
    W <- clamp(round(W - alpha * u (u^T W)), -127, 127)      for W = W2_l (feed-forward) or Wo_l (attention).
u comes from directions_03.npz (plain difference of means) or directions_04.npz (matched pairs).
The edit is taken back after the run unless --keep is given.
Run from the repository root, e.g.:
PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/05_remove_direction.py --tensors W2_1 --direction 04:l1_xf --alpha 1"""
import argparse

import numpy as np

from common import DIR, report, restore
from lsllm.edit.session import Session

ap = argparse.ArgumentParser()
ap.add_argument("--tensors", nargs="+", required=True)
ap.add_argument("--direction", required=True, help="<03 or 04>:<name>, e.g. 04:l1_xf")
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--keep", action="store_true")
ap.add_argument("--base", type=int, default=1, help="evaluate run to compare with")
a = ap.parse_args()

s = Session(DIR)
which, name = a.direction.split(":")
u = np.load("%s/directions_%s.npz" % (DIR, which))[name]
note = "remove direction %s from %s, alpha %g" % (a.direction, "+".join(a.tensors), a.alpha)
saved = {}
for tensor in a.tensors:
    W = s.get(tensor)
    saved[tensor] = W.copy()
    new = np.clip(np.round(W - a.alpha * np.outer(u, u @ W)), -127, 127)
    entry = s.put(tensor, new, note)
    print(tensor, "changed", entry["changed"], "of", W.size, "largest change", int(np.abs(new - W).max()))
report(s.evaluate(note), a.base)
if not a.keep:
    restore(s, saved, "undo: " + note)
    print("taken back")
