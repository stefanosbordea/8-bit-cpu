"""No model run. Put the working model back to the state it had after log entry N of edits.jsonl, through
Session.put / put_number (logged). Needed because Session.undo called twice re-applies an edit (it takes back
the last LOG ENTRY, and the first undo is itself an entry): that happened once, before run 23.
python restore.py <N>"""
import sys
from lsllm.edit.session import Session

n = int(sys.argv[1])
s = Session("edits/s4")
arrays = {name: s.get(name) for name in s.arrays}
numbers = {}
for e in reversed(s.edits()[n:]):
    if e["tensor"].startswith("model.json:"):
        numbers[e["tensor"]] = e["old"][0]
    else:
        arrays[e["tensor"]].ravel()[e["positions"]] = e["old"]
for name, values in arrays.items():
    entry = s.put(name, values, "restore %s to its state after log entry %d" % (name, n))
    if entry:
        print("restored", name, entry["changed"], "numbers")
for key, value in numbers.items():
    path = tuple(int(k) if k.isdigit() else k for k in key[len("model.json:"):].split("."))
    if s.put_number(path, value, "restore to the state after log entry %d" % n):
        print("restored", key)
print("zero W2_1 columns:", [k for k in range(219) if not s.get("W2_1")[:, k].any()])
print("zero Wo_1 columns:", [k for k in range(160) if not s.get("Wo_1")[:, k].any()])
print("EXP[1] distance 1..4:", s.get("EXP")[1][255:251:-1].tolist(), " EXP[0]:", s.get("EXP")[0][255:251:-1].tolist())
