"""No model run. Where is a layer-1 feed-forward unit active in the traced reference bodies (state <tag>)?
Prints the strongest positions: the text before, the reference token, the greedy token, the unit's margin
contribution there. python 19_unit_contexts.py <tag> <unit> ..."""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, D

tag = sys.argv[1]
A, rows, tokens = load(["l1.g"], tag)
att = np.load(D + "attribution_%s.npz" % tag)["units"]
g = A["l1.g"][rows[:, 2]]
for k in map(int, sys.argv[2:]):
    print("unit %d: active at %.2f of body positions, mean when active %.1f" % (k, (g[:, k] > 0).mean(), g[g[:, k] > 0, k].mean()))
    worst = np.argsort(att[:, k])[:7]
    best = np.argsort(-att[:, k])[:5]
    for label, idx in (("hurts", worst), ("helps", best)):
        for r in idx:
            p = rows[r, 2]
            before = "".join(P(t) for t in tokens[max(0, p - 7):p + 1]).replace("\n", "\\n")
            print("   %s %6.1f  act %3d  ...%-40s ref %-8r greedy %r" % (label, att[r, k], g[r, k], before[-40:], P(rows[r, 3]), P(rows[r, 4])))
