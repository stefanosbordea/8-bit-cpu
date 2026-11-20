"""Run 8 (trace, on the unedited model): prompt + the model's own baseline completion (eval_01) of every task that
ended in a NameError, then tasks with other outcomes until 4096 tokens are full. Saved to edits/s2/trace_own.pkl."""
import json, pickle
from lsllm.edit.session import Session
from lsllm.tok import Tok
s = Session("edits/s2")
assert not any((s.arrays[k] != v).any() for k, v in __import__("numpy").load(s.budget["base"] + "/model.npz").items()), "not the base model"
tok = Tok(s.spec["tokenizer"] + "/tok.model")
tasks = [json.loads(line) for line in open("data/bench/custom_practice.jsonl")]
samples = [json.loads(line) for line in open("edits/s2/eval_01.jsonl")]
order = sorted(range(200), key=lambda g: ("NameError" not in samples[g]["detail"], g % 3, g))
chosen, total = [], 0
for g in order:
    try:
        ids = tok.encode(tasks[g]["prompt"] + samples[g]["completion"])
    except ValueError:          # a completion that is not Python cannot be tokenised again
        continue
    if total + len(ids) > 4096:
        continue
    chosen.append((ids, len(tok.encode(tasks[g]["prompt"])), g)); total += len(ids)
names = ["x0", "u", "y", "y.r"] + ["l%d.%s" % (l, n) for l in (0, 1) for n in ("ya", "qr", "kr", "v", "o", "xa", "yf", "g", "xf", "dsum", "max")]
out = s.trace([ids for ids, _, _ in chosen], names, "prompt + own baseline completion of %d tasks (all NameError tasks first)" % len(chosen))
pickle.dump({"seqs": chosen, "trace": out}, open("edits/s2/trace_own.pkl", "wb"))
print(len(chosen), "tasks", total, "tokens; budget left", s.left())
