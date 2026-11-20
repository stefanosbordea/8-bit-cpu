"""Runs 3-5 (trace): the reference solutions of the other 142 practice tasks, teacher forced, so that the
attribution statistics rest on all 200 tasks. Saves edits/s4/trace_ref_b<k>.npz."""
import json
import numpy as np
from lsllm.edit.session import Session
from lsllm.tok import Tok

s = Session("edits/s4")
tok = Tok(s.spec["tokenizer"] + "/tok.model")
tasks = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
done = set(np.load("edits/s4/trace_ref.npz")["tasks"].tolist())
rest = [(n, tok.encode(tasks[n]["canonical_solution"], bos=True, eos=True)) for n in range(200) if n not in done]
rest.sort(key=lambda item: -len(item[1]))
bins = [[], [], []]                                   # first-fit decreasing into three runs of at most 4096 tokens
for n, ids in rest:
    for b in bins:
        if sum(len(q) for _, q in b) + len(ids) <= 4096:
            b.append((n, ids)); break
    else:
        raise SystemExit("task %d does not fit" % n)
names = ["u", "y", "y.r", "l1.xf", "l1.xa", "l1.g", "l1.o", "l0.xf", "l0.xa", "l0.g", "l0.o", "x0",
         "l0.dsum", "l0.max", "l1.dsum", "l1.max", "l1.yf", "l1.ya"]
for k, b in enumerate(bins):
    b.sort()
    out = s.trace([ids for _, ids in b], names, "reference solutions of %d more practice tasks, teacher forced" % len(b))
    save = {"tasks": np.array([n for n, _ in b]), "lengths": np.array([len(q) for _, q in b]),
            "tokens": np.concatenate([q for _, q in b])}
    for name in names:
        save[name] = np.concatenate([o[name].reshape(len(o[name]), -1) for o in out]).astype(np.int16)
    np.savez_compressed("edits/s4/trace_ref_b%d.npz" % k, **save)
    print(k, len(b), "tasks", sum(len(q) for _, q in b), "tokens; budget left", s.left())
