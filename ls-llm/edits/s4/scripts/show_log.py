"""No model run. The edit log and the run table (the inline listings of the session, saved as a script), and the
difference between the working model and the exported base."""
import json
import numpy as np
from lsllm.edit.session import Session
s = Session("edits/s4")
for e in s.edits():
    print(e["edit"], e["after_run"], e["tensor"], e["changed"], e["reason"][:110])
for r in s.budget["runs"]:
    x = r.get("result", {})
    print(r["run"], r["kind"], r["edits_so_far"], x.get("pass@1", "-"), x.get("syntax_ok", "-"), x.get("runs_ok", "-"), x.get("tests_passed", "-"), x.get("practice_loss", "-"), "|", r["note"][:90])
base = np.load(s.budget["base"] + "/model.npz")
for name in base.files:
    d = int((base[name] != s.arrays[name]).sum())
    if d:
        print(name, "differs from the base in", d, "numbers")
print("zero W2_1 columns:", [k for k in range(219) if not s.get("W2_1")[:, k].any()])
print("zero Wo_1 columns:", [k for k in range(160) if not s.get("Wo_1")[:, k].any()])
