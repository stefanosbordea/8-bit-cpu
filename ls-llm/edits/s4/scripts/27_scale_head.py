"""Edit + run. Scale what one head of layer 1 writes: Wo_1[:, 40 h : 40 h + 40] *= factor (rounded).
Head 0 is the copy head (script 26: +8.4 codes of margin at positions that continue a word of the prompt, +7.3
where a word of the prompt begins, -1.3 elsewhere; decisive positions +311 / +310 codes in the two halves).
Hypothesis: the NameErrors (a word of the docstring or a wrong ending instead of the argument) are copies that
lose against the feed-forward prior; a louder copy head wins more of them.
python 27_scale_head.py <head> <factor>"""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

head, factor = int(sys.argv[1]), float(sys.argv[2])
s = Session("edits/s4")
w = s.get("Wo_1").astype(float)
w[:, 40 * head:40 * head + 40] *= factor
w = np.clip(np.round(w), -127, 127).astype(int)
print(s.put("Wo_1", w, "Wo_1 columns of head %d times %.2f: the copy head made louder" % (head, factor))["changed"], "numbers changed; largest |w| now", np.abs(w).max())
report(s, "Wo_1 head %d x %.2f" % (head, factor))
