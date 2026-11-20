"""Undo edit 2 (60 units: syntax_ok 0.93 -> 0.75, loss 1.65 -> 2.31: the model is damaged), then edit 3 + run 8.
Hypothesis: only the units that are most negative at DECISIVE positions (|margin| <= 8 codes) in both halves hurt
greedy decoding; the rest hedge in a useful way. Edit: zero W2_1 columns 136 (already), 194, 207, 70, 10, 93, 206, 67."""
import sys
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

s = Session("edits/s4")
print("undo:", s.undo()["changed"], "numbers restored")
w = s.get("W2_1"); w[:, [194, 207, 70, 10, 93, 206, 67]] = 0
print(s.put("W2_1", w, "zero W2_1 columns 194, 207, 70, 10, 93, 206, 67: most negative at decisive positions in both halves")["changed"], "numbers changed")
report(s, "W2_1: columns 136, 194, 207, 70, 10, 93, 206, 67 zeroed")
