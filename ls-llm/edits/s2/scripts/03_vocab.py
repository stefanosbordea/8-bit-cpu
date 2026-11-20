"""No budget: the vocabulary, the tokens of some prompts, and the sizes of the arrays."""
import json
import numpy as np
from lsllm.edit.session import Session
from lsllm.tok import Tok
s = Session("edits/s2")
tok = Tok(s.spec["tokenizer"] + "/tok.model")
print(" ".join("%d:%r" % (t, tok.pieces[t]) for t in range(512)))
tasks = [json.loads(line) for line in open("data/bench/custom_practice.jsonl")]
for g in (0, 9, 30, 63, 66, 104, 111):
    ids = tok.encode(tasks[g]["canonical_solution"])
    print(len(ids), "|".join(tok.pieces[t] for t in ids))
for name, a in s.arrays.items():
    print(name, a.shape, int(a.min()), int(a.max()), "mean|.| %.1f" % np.abs(a.astype(int)).mean())
print("prompt lengths", sorted(len(tok.encode(t["prompt"])) for t in tasks)[::20])
