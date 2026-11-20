"""Edit 4 + run 9. Same rule as edit 3, the next eight units of the decisive ranking (negative in both halves):
161, 102, 19, 218, 123, 121, 197, 16. Does the gain continue or does the damage of edit 2 begin here?"""
import sys
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

s = Session("edits/s4")
w = s.get("W2_1"); w[:, [161, 102, 19, 218, 123, 121, 197, 16]] = 0
print(s.put("W2_1", w, "zero W2_1 columns 161, 102, 19, 218, 123, 121, 197, 16: ranks 9-16 of the decisive ranking")["changed"], "numbers changed")
report(s, "W2_1: 16 columns zeroed (top 16 of the decisive ranking)")
