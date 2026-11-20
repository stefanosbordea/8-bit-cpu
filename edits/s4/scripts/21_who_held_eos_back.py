"""No model run. After 16 units were zeroed the answers end too early (EOS wrongly on top at 49 reference
positions, 24 at the baseline; answers 35.6 characters on average, references 43.7). Which of the zeroed units
held EOS (and ')' , ']') back? For every zeroed unit, with the BASELINE activations and the ORIGINAL column:
its contribution to logit(t) - logit(reference) at the positions where t is wrongly on top in state e6, and to
logit(t) - logit(best other) where t is the reference. Also the same with the column replaced by its projection
on E[t] (the planned repair)."""
import json
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, D
from lsllm.edit.session import Session

s = Session("edits/s4")
E = s.get("E").astype(float)
W2 = s.get("W2_1").astype(float)
original = W2.copy()                                   # rebuild the original W2_1 from the edit log
for e in reversed(s.edits()):
    if e["tensor"] == "W2_1":
        original.ravel()[e["positions"]] = e["old"]
zeroed = [k for k in range(219) if not W2[:, k].any()]
print("zeroed units:", zeroed)
A0, rows0, _ = load(["y", "l1.xf", "l1.g"])
A6, rows6, _ = load([], "e6")
assert (rows0[:, :4] == rows6[:, :4]).all()
pos, ref = rows0[:, 2], rows0[:, 3]
gain = np.linalg.norm(A0["y"][pos], axis=1) / np.linalg.norm(A0["l1.xf"][pos], axis=1) / 512
g = A0["l1.g"][pos]
for t in (2, 13, 15):
    wrong = (rows6[:, 4] == t) & (ref != t)
    right = ref == t
    e_hat = E[t] / np.linalg.norm(E[t])
    print("\ntoken %r: wrongly on top at %d positions (e6), reference at %d" % (P(t), wrong.sum(), right.sum()))
    print("  unit | E[t].w | mean effect on logit(t)-logit(ref) where wrongly on top | on logit(t)-logit(other) where reference | the same two with the projected column")
    for k in zeroed:
        w = original[:, k]
        wp = (w @ e_hat) * e_hat
        out = []
        for col in (w, wp):
            a = (g[wrong, k] * ((E[t] - E[ref[wrong]]) @ col) / 256 * gain[wrong]).mean()
            b = (g[right, k] * ((E[t] - E[rows0[right, 5]]) @ col) / 256 * gain[right]).mean()
            out += [a, b]
        print("  %4d | %7.0f | %6.2f | %6.2f | %6.2f | %6.2f" % (k, E[t] @ w, out[0], out[1], out[2], out[3]))
