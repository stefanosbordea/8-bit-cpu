"""Run 2 (trace): prompt + reference solution + end of text of about 55 practice tasks, teacher forced.
Saves the logits u and the last-layer values to edits/s4/trace_ref.npz for the analysis scripts."""
import json
import numpy as np
from lsllm.edit.session import Session
from lsllm.tok import Tok

s = Session("edits/s4")
tok = Tok(s.spec["tokenizer"] + "/tok.model")
tasks = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
order = list(range(0, 200, 4)) + list(range(2, 200, 16))
chosen, seqs, total = [], [], 0
for n in order:
    ids = tok.encode(tasks[n]["canonical_solution"], bos=True, eos=True)
    if total + len(ids) > 4096:
        continue
    chosen.append(n); seqs.append(ids); total += len(ids)
print(len(chosen), "tasks,", total, "tokens")
names = ["u", "y", "l1.xf", "l1.xa", "l1.g", "l1.o", "l0.xf"]
out = s.trace(seqs, names, "reference solutions of %d practice tasks, teacher forced: u and last-layer values" % len(chosen))
save = {"tasks": np.array(chosen), "lengths": np.array([len(q) for q in seqs]), "tokens": np.concatenate(seqs)}
for name in names:
    save[name] = np.concatenate([o[name] for o in out]).astype(np.int16)
np.savez_compressed("edits/s4/trace_ref.npz", **save)
print("budget left", s.left())
