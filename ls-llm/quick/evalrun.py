"""Fast path: evaluate a checkpoint on a benchmark split through the sandboxed harness."""
import json
import sys

import torch

from lsllm.bench import evaluate, load_tasks
from quick.complete import QuickCompleter

ckpt, bench, split = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 1
temp = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
use_filter = len(sys.argv) > 6 and sys.argv[6] == "filter"
torch.set_num_threads(2)
tasks = load_tasks(bench, split=split)
rep = evaluate(QuickCompleter(ckpt, use_filter=use_filter), tasks, n_samples=n, temperature=temp, max_new_tokens=96, seed=0, workers=3)
print(json.dumps({"ckpt": ckpt, "bench": bench, "split": split, "n_tasks": rep.n_tasks, "n_samples": n, "temperature": temp, "filter": use_filter,
                  "metrics": rep.metrics, "outcomes": rep.outcomes, "truncated": rep.truncated}))
by = {}
for s, t in zip(rep.samples[::n], tasks):
    key = t.meta.get("stratum") or "all"
    by.setdefault(key, [0, 0])
    by[key][1] += 1
if n == 1 and tasks and tasks[0].meta.get("stratum"):
    for s, t in zip(rep.samples, tasks):
        by[t.meta["stratum"]][0] += s["outcome"] == "pass"
    print(json.dumps({"pass_by_stratum": {k: "%d/%d" % tuple(v) for k, v in by.items()}}))
json.dump(rep.samples, open(ckpt.replace("ckpt.pt", "samples_%s_%s%s.json" % (bench, split, "_filter" if use_filter else "")), "w"))
