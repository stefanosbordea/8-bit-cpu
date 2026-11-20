"""Edit + run. Second round on the re-trace of state e3 (8 units zeroed; reference accuracy 0.680 -> 0.695).
Rule: units negative in all four statistics of the NEW attribution, most negative at decisive positions first.
python 20_zero_round2_units.py 19 133 212 102"""
import sys
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

units = [int(a) for a in sys.argv[1:]]
s = Session("edits/s4")
w = s.get("W2_1"); w[:, units] = 0
print(s.put("W2_1", w, "zero W2_1 columns %s: negative margin contribution in all four statistics of the re-trace e3" % units)["changed"], "numbers changed")
report(s, "W2_1: also columns %s zeroed" % units)
