"""One budgeted evaluate of: rebuild the attention weight table of a layer from its formula with another decay,
    EXP[l][0] = 127, EXP[l][256 - d] = round(127 exp(-d / kappa)) for d = 1..127, 0 elsewhere   (SPEC C.6; kappa is 4.0 in the export).
A smaller kappa makes the layer attend more sharply. Several --layer l:kappa may be given.
The edit is taken back after the run unless --keep is given.
Run from the repository root, e.g.:
PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/09_exp_kappa.py --layer 1:3.2"""
import argparse

import numpy as np

from common import DIR, report, restore
from lsllm.edit.session import Session

ap = argparse.ArgumentParser()
ap.add_argument("--layer", nargs="+", required=True, help="layer:kappa")
ap.add_argument("--keep", action="store_true")
ap.add_argument("--base", type=int, default=1, help="evaluate run to compare with")
a = ap.parse_args()

s = Session(DIR)
note = "EXP table with kappa " + " ".join(a.layer)
EXP = s.get("EXP")
saved = {"EXP": EXP.copy()}
for item in a.layer:
    l, kappa = item.split(":")
    d = np.arange(1, 128)
    EXP[int(l), 256 - d] = np.round(127 * np.exp(-d / float(kappa)))
print("EXP changed", s.put("EXP", EXP, note)["changed"])
report(s.evaluate(note), a.base)
if not a.keep:
    restore(s, saved, "undo: " + note)
    print("taken back")
