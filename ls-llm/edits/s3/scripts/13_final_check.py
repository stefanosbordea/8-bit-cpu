"""No budget: what differs between the session's model and the model it started from, and the budget used."""
import json
import numpy as np
from common import open_session

s, tok = open_session()
base = s.budget["base"]
with np.load(base + "/model.npz") as old:
    for name in old.files:
        where = np.flatnonzero(old[name] != s.arrays[name])
        if len(where):
            print(name, "differs at", len(where), "places; flat index, old, new of the first 12:",
                  list(zip(where[:12].tolist(), old[name].ravel()[where[:12]].tolist(), s.arrays[name].ravel()[where[:12]].tolist())))
            if name == "NORM":
                print("  sites touched:", sorted(set((where // 256).tolist())))
with open(base + "/model.json") as f:
    spec = json.load(f)
print("shifts equal:", spec["shifts"] == s.spec["shifts"], " preloads equal:", spec["preloads"] == s.spec["preloads"])
print("runs used %d of %d; log entries %d" % (len(s.budget["runs"]), s.budget["limit"], len(s.edits())))
print("last run:", json.dumps(s.budget["runs"][-1]["result"]))
