"""Run 2 (trace): the reference solutions of the shortest practice tasks, teacher forced.
Keeps everything needed to rebuild the attention weights (qr, kr: `l.e` itself cannot be stacked over positions)
and to attribute logits to heads and feed-forward units. Saved to edits/s2/trace_refs.pkl."""
import json, pickle
from lsllm.edit.session import Session
from lsllm.tok import Tok
s = Session("edits/s2")
tok = Tok(s.spec["tokenizer"] + "/tok.model")
tasks = [json.loads(line) for line in open("data/bench/custom_practice.jsonl")]
seqs = sorted(((tok.encode(t["canonical_solution"]), len(tok.encode(t["prompt"])), g) for g, t in enumerate(tasks)),
              key=lambda item: len(item[0]))
chosen, total = [], 0
for ids, n_prompt, g in seqs:
    if total + len(ids) > 4096:
        break
    chosen.append((ids, n_prompt, g)); total += len(ids)
names = ["x0", "u", "y", "y.r"] + ["l%d.%s" % (l, n) for l in (0, 1) for n in ("ya", "qr", "kr", "v", "o", "xa", "yf", "g", "xf", "dsum", "max")]
out = s.trace([ids for ids, _, _ in chosen], names, "reference solutions of the %d shortest tasks, teacher forced" % len(chosen))
pickle.dump({"seqs": chosen, "trace": out}, open("edits/s2/trace_refs.pkl", "wb"))
print(len(chosen), "tasks", total, "tokens; budget left", s.left())
