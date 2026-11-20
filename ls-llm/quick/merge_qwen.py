"""All usable Qwen functions in one file: the training functions of data/final/f512c plus the functions of later
generation folders, under the same rules as quick/build_final.py: not a benchmark name or solution (overlap rules,
read again now), not the code of a validation function, at most two functions with the same code fingerprint.

    python -m quick.merge_qwen OUT_FILE data/qwen/s data/qwen/t ...
"""
import json
import sys

from lsllm.alpha import alpha_key
from lsllm.data.corpus.overlap import load_benchmarks, overlap
from quick.synth import benchmark_keys

out, folders = sys.argv[1], sys.argv[2:]
read = lambda path: [json.loads(line) for line in open(path)]
names, keys = benchmark_keys()
bench = load_benchmarks()
val_keys = {alpha_key(r["text"]) for r in read("data/final/f512a/qwen_val.jsonl")}
kept, seen, per_key, count = [], set(), {}, {"old": 0, "new": 0, "dropped_overlap": 0, "dropped_duplicate": 0, "dropped_like_validation": 0}
for source, rows in [("old", read("data/final/f512c/qwen_train.jsonl"))] + [("new", read(folder + "/all.jsonl")) for folder in folders]:
    for r in rows:
        key = alpha_key(r["text"])
        if r["name"] in names or key in keys or overlap(r["name"], r["text"], key, bench):
            count["dropped_overlap"] += 1
        elif key in val_keys:
            count["dropped_like_validation"] += 1
        elif r["text"] in seen or per_key.get(key, 0) >= 2:
            count["dropped_duplicate"] += 1
        else:
            seen.add(r["text"])
            per_key[key] = per_key.get(key, 0) + 1
            kept.append(r)
            count[source] += 1
with open(out, "w") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")
print(json.dumps(dict(count, total=len(kept), distinct_code=len(per_key))))
