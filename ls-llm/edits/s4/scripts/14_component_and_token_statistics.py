"""No model run. (a) margin contribution of the big parts at decisive / wrong positions (which part to scale?);
(b) per token: how often wrongly chosen / missed over all 3834 reference positions; (c) is there a constant
direction in y (needed for a per-token bias through an E row)?"""
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, D

A, rows, tokens = load(["y"])
z = np.load(D + "attribution.npz")
units, heads, base, margin, ok = z["units"], z["heads"], z["base"], z["margin"], z["ok"]
near = np.abs(margin) <= 8
for label, m in (("all", slice(None)), ("decisive", near), ("wrong", ~ok), ("decisive wrong", near & ~ok), ("decisive right", near & ok)):
    print("%-15s n %4d  margin %6.2f  layer-0 stream %6.2f  heads %s  units %6.2f" % (label, len(margin[m]), margin[m].mean(), base[m].mean(), heads[m].mean(0).round(2), units[m].sum(1).mean()))
ref, top = rows[:, 3], rows[:, 4]
fp, fn, rc = Counter(top[~ok]), Counter(ref[~ok]), Counter(ref)
print("\ntoken: reference count / wrongly chosen / missed")
for t in sorted(set(fp) | set(fn), key=lambda t: -abs(fp[t] - fn[t]))[:36]:
    print("  %-9r ref %3d  wrongly chosen %3d  missed %3d" % (P(t), rc[t], fp[t], fn[t]))
y = A["y"][rows[:, 2]].astype(float)
m = y.mean(0)
print("\n|mean y| %.1f, mean |y| %.1f; projection of y on the mean direction: mean %.1f sd %.1f" % (
    np.linalg.norm(m), np.linalg.norm(y, axis=1).mean(), (y @ m / np.linalg.norm(m)).mean(), (y @ m / np.linalg.norm(m)).std()))
