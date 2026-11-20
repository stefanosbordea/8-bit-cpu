"""Edit 1 + run 6. Hypothesis: feed-forward unit 136 of layer 1 writes 'if ', '* ', 'not ' and pushes ' ' down;
it is the unit with the most negative margin contribution at decisive reference positions (-331 codes, negative
in both halves of the tasks) and matches the habit `x if ...` / `x * ...` where the reference has `x + ...`.
Edit: zero column 136 of W2_1."""
import sys
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

s = Session("edits/s4")
w = s.get("W2_1"); w[:, 136] = 0
print(s.put("W2_1", w, "zero column 136 of W2_1: the unit writes 'if ', '* ', 'not ' after operands")["changed"], "numbers changed")
report(s, "W2_1 column 136 zeroed")
