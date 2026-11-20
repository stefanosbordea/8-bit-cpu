"""Edit + run. A per-token logit shift through the E row: E[t] += round(shift * 2^sh_l * m / |m|^2), where m is
the mean of the final norm output y over the body positions of the reference traces (state e6). Then
dE . y / 2^sh_l is `shift` codes on average. Tokens and shifts come from script 18 (threshold statistics of the
traced logits): '== ' is missed at 24 of 40 reference positions and almost never chosen wrongly; 'a ' (the
argument a in a body) is missed at 26 of 34.
python 23_raise_tokens.py <token id>:<shift> ..."""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lib import load, P
from lsllm.edit.session import Session

s = Session("edits/s4")
A, rows, tokens = load(["y"], "e6")
y = A["y"][rows[:, 2]].astype(float)
m = y.mean(0)
E = s.get("E")
notes = []
for arg in sys.argv[1:]:
    t, shift = arg.split(":")
    t, shift = int(t), float(shift)
    delta = np.round(shift * 512 * m / (m @ m)).astype(int)
    got = y @ delta / 512
    print("token %r: %d elements changed, largest step %d, |dE| %.1f of |E| %.1f; shift at body positions %.1f +- %.1f codes" % (
        P(t), (delta != 0).sum(), np.abs(delta).max(), np.linalg.norm(delta), np.linalg.norm(E[t]), got.mean(), got.std()))
    E[t] += delta
    notes.append("%r %+g codes" % (P(t), shift))
entry = s.put("E", E, "E rows moved along the mean of y over body positions: " + ", ".join(notes))
print(entry["changed"], "numbers changed")
report(s, "E rows shifted: " + ", ".join(notes))
