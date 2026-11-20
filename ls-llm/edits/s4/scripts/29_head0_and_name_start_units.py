"""Edits + run on the 16-unit state: (a) Wo_1 head 0 x 1.25 (run 27: syntax_ok 0.95, tests_passed 0.1237);
(b) zero the W2_1 columns given on the command line. Hypothesis for 54, 78, 191, 83, 26: at the positions where
a word of the prompt must begin (right at only 55 %) these units work against the reference in both halves and
are not needed elsewhere; 54 writes digits ('1' '2' '3' '5'), the habit `return 2 - 5` for `return n - 5`.
python 29_head0_and_name_start_units.py 54 78 191 83 26"""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

units = [int(a) for a in sys.argv[1:]]
s = Session("edits/s4")
w = s.get("Wo_1").astype(float)
w[:, :40] *= 1.25
print(s.put("Wo_1", np.round(w).astype(int), "Wo_1 columns of head 0 times 1.25: the copy head made louder")["changed"], "numbers changed in Wo_1")
if units:
    w = s.get("W2_1"); w[:, units] = 0
    print(s.put("W2_1", w, "zero W2_1 columns %s: against the reference where a prompt word must begin, both halves" % units)["changed"], "numbers changed in W2_1")
report(s, "Wo_1 head 0 x 1.25; W2_1 columns %s zeroed" % units)
