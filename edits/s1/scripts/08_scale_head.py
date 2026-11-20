"""One budgeted evaluate of: scale what attention heads write, Wo_l[:, 40 h : 40 h + 40] <- round(factor * ...),
on top of the model that is in the session directory. Several --head l:h:factor may be given.
The edit is taken back after the run unless --keep is given.
Run from the repository root, e.g.:
PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/08_scale_head.py --head 1:0:1.25"""
import argparse

import numpy as np

from common import DIR, report, restore
from lsllm.edit.session import Session

ap = argparse.ArgumentParser()
ap.add_argument("--head", nargs="+", required=True, help="layer:head:factor")
ap.add_argument("--keep", action="store_true")
ap.add_argument("--base", type=int, default=1, help="evaluate run to compare with")
a = ap.parse_args()

s = Session(DIR)
note = "scale Wo of heads " + " ".join(a.head)
new, saved = {}, {}
for item in a.head:
    l, h, factor = item.split(":")
    name = "Wo_" + l
    if name not in new:
        new[name] = s.get(name)
        saved[name] = new[name].copy()
    block = new[name][:, 40 * int(h):40 * int(h) + 40]
    new[name][:, 40 * int(h):40 * int(h) + 40] = np.clip(np.round(float(factor) * block), -127, 127)
for name, W in new.items():
    print(name, "changed", s.put(name, W, note)["changed"])
report(s.evaluate(note), a.base)
if not a.keep:
    restore(s, saved, "undo: " + note)
    print("taken back")
