"""Edit + run, from the attribution of the re-trace e6 (16 units zeroed). Last round of this kind.
(a) Wo_1 columns (head dimensions of layer 1) that are negative in all four statistics, the eight most negative
    at decisive positions: 157, 149, 51, 145, 83, 96, 57, 92.
(b) W2_1 columns 209, 34, 62: negative in all four statistics; the units that write digits (161, 123, 16) or
    hold EOS / line breaks back (94, 182, 106, 121) are left alone: edit 4 showed that operands and structure
    break without them.
python 24_zero_head_dims_and_units.py dims|units|both"""
import sys
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

what = sys.argv[1]
s = Session("edits/s4")
note = []
if what in ("dims", "both"):
    w = s.get("Wo_1"); w[:, [157, 149, 51, 145, 83, 96, 57, 92]] = 0
    print(s.put("Wo_1", w, "zero Wo_1 columns 157, 149, 51, 145, 83, 96, 57, 92: negative in all four statistics of the re-trace e6")["changed"], "numbers changed")
    note.append("Wo_1 columns 157, 149, 51, 145, 83, 96, 57, 92 zeroed")
if what in ("units", "both"):
    w = s.get("W2_1"); w[:, [209, 34, 62]] = 0
    print(s.put("W2_1", w, "zero W2_1 columns 209, 34, 62: negative in all four statistics of the re-trace e6")["changed"], "numbers changed")
    note.append("W2_1 columns 209, 34, 62 zeroed")
report(s, "; ".join(note))
