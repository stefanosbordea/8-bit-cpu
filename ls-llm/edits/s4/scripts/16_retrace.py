"""4 trace runs: the reference solutions of all 200 practice tasks, teacher forced, on the CURRENT working model.
python 16_retrace.py <tag>  writes edits/s4/trace_<tag>_<k>.npz (same grouping of tasks as the first traces)."""
import json
import sys
import numpy as np
from lsllm.edit.session import Session
from lsllm.tok import Tok

tag = sys.argv[1]
s = Session("edits/s4")
tok = Tok(s.spec["tokenizer"] + "/tok.model")
tasks = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
names = ["u", "y", "l1.xf", "l1.xa", "l1.g", "l1.o", "l0.xf", "l0.g", "l0.o", "l0.xa", "x0"]
for k, old in enumerate(("trace_ref.npz", "trace_ref_b0.npz", "trace_ref_b1.npz", "trace_ref_b2.npz")):
    numbers = np.load("edits/s4/" + old)["tasks"].tolist()
    seqs = [tok.encode(tasks[n]["canonical_solution"], bos=True, eos=True) for n in numbers]
    out = s.trace(seqs, names, "reference solutions of %d practice tasks, teacher forced, state '%s'" % (len(seqs), tag))
    save = {"tasks": np.array(numbers), "lengths": np.array([len(q) for q in seqs]), "tokens": np.concatenate(seqs)}
    for name in names:
        save[name] = np.concatenate([o[name].reshape(len(o[name]), -1) for o in out]).astype(np.int16)
    np.savez_compressed("edits/s4/trace_%s_%d.npz" % (tag, k), **save)
    print(k, len(seqs), "tasks; budget left", s.left())
