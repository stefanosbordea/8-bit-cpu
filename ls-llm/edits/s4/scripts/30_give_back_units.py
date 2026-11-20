"""Edit + run. Give zeroed W2_1 columns their original numbers back (from the edit log).
Hypothesis for 10 and 194 (script 21): they held EOS back by 5.4 and 3.8 codes exactly where EOS is wrong;
without them the answers end too early (35.6 characters, references 43.7; EOS wrongly on top 49 times, 24 at
the baseline). Their own attribution was negative, so this is a trade.
python 30_give_back_units.py 10 194"""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

units = [int(a) for a in sys.argv[1:]]
s = Session("edits/s4")
original = s.get("W2_1")
for e in reversed(s.edits()):
    if e["tensor"] == "W2_1":
        original.ravel()[e["positions"]] = e["old"]          # back to the exported W2_1
w = s.get("W2_1")
w[:, units] = original[:, units]
print(s.put("W2_1", w, "W2_1 columns %s get their exported numbers back: they hold EOS back where it is wrong" % units)["changed"], "numbers changed")
print("zero columns now:", [k for k in range(219) if not w[:, k].any()])
report(s, "W2_1 columns %s given back" % units)
