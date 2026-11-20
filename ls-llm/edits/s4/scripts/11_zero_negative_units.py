"""Edit 2 + run 7. Hypothesis: the layer-1 feed-forward units whose margin contribution at the reference tokens is
negative in all four statistics (all body positions and decisive ones, even and odd tasks) only hedge or push
wrong tokens on this kind of text; greedy decoding gains if they are silent.
Edit: zero their W2_1 columns (60 units, 136 is already zero). One-shot rule, no search."""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report, D
from lib import load
from lsllm.edit.session import Session

A, rows, tokens = load([])
z = np.load(D + "attribution.npz")
units, margin = z["units"], z["margin"]
near, even = np.abs(margin) <= 8, rows[:, 0] % 2 == 0
stats = [units[m].sum(0) for m in (even, ~even, near & even, near & ~even)]
chosen = np.flatnonzero(np.all([st < 0 for st in stats], 0))
print(len(chosen), "units:", chosen.tolist(), "predicted mean margin gain %.2f codes" % (-units[:, chosen].sum() / len(rows)))
s = Session("edits/s4")
w = s.get("W2_1"); w[:, chosen] = 0
print(s.put("W2_1", w, "zero the W2_1 columns of the %d units with negative margin contribution in all four statistics" % len(chosen))["changed"], "numbers changed")
report(s, "W2_1: 60 negative-attribution columns zeroed")
