"""Final run: check that the session directory holds the kept model (base + the three logged edits of run 15:
Wq_1 row 38, Wk_1 row 38, E row 2), then evaluate it once more. The result must equal run 15."""
import json
import numpy as np
from lsllm.edit.session import Session
s = Session("edits/s2")
base = np.load(s.budget["base"] + "/model.npz")
spec = json.load(open(s.budget["base"] + "/model.json"))
assert (spec["shifts"], spec["preloads"]) == (s.spec["shifts"], s.spec["preloads"]), "a number of model.json differs from the base"
diff = {k: np.argwhere(base[k] != s.arrays[k]) for k in base.files if (base[k] != s.arrays[k]).any()}
assert set(diff) == {"Wq_1", "Wk_1", "E"}, sorted(diff)
assert set(diff["Wq_1"][:, 0]) == {38} and set(diff["Wk_1"][:, 0]) == {38} and set(diff["E"][:, 0]) == {2}
print("differs from the base in: Wq_1 row 38 (%d values), Wk_1 row 38 (%d values), E row 2 (%d values)" % tuple(len(diff[k]) for k in ("Wq_1", "Wk_1", "E")))
r = s.evaluate("final: the kept model again (signature preference 3 codes + end bias 40)")
print(json.dumps(r, indent=1))
r15 = next(run["result"] for run in s.budget["runs"] if run["run"] == 15)
same = all(r[k] == r15[k] for k in ("practice_loss", "pass@1", "syntax_ok", "runs_ok", "tests_passed", "outcomes", "errors", "repeating"))
a = open("edits/s2/eval_15.jsonl").read(); b = open("edits/s2/eval_%02d.jsonl" % r["run"]).read()
print("same scores as run 15:", same, "| same completions:", a == b)
