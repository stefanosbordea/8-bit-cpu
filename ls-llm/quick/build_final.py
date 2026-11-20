"""Assemble the training set: human-written functions (corpus + CodeSearchNet subset) and Qwen-written simple
functions (repeated K times so that simple documented functions are a fixed share of the token stream).
Usage: python -m quick.build_final OUT_NAME K"""
import glob
import json
import os
import random
import subprocess
import sys

from lsllm.alpha import alpha_key

out, k = sys.argv[1], int(sys.argv[2])
max_lines = int(sys.argv[3]) if len(sys.argv) > 3 else 40
need_doc = len(sys.argv) > 4 and sys.argv[4] == "doc"
fits = lambda r: r["n_lines"] <= max_lines and (r.get("has_docstring") or not need_doc)
os.makedirs("data/final/" + sys.argv[1], exist_ok=True)
rng = random.Random(0)


def rows(path, keep=lambda r: True):
    return [r for r in map(json.loads, open(path)) if keep(r)]


human_train = rows("data/corpus/train.jsonl", lambda r: r["kind"] == "function" and fits(r)) + rows("data/csn/train.jsonl", fits)
human_val = rows("data/corpus/val.jsonl", lambda r: r["kind"] == "function" and fits(r)) + rows("data/csn/val.jsonl", fits)
texts = set()
human_train = [r for r in human_train if not (r["text"] in texts or texts.add(r["text"]))]
human_val = [r for r in human_val if r["text"] not in texts]
from quick.synth import benchmark_keys
from lsllm.data.corpus.overlap import load_benchmarks, overlap
bench_names, bench_keys = benchmark_keys()          # read again now: the practice tasks may be newer than the data
bench = load_benchmarks()
before = len(human_train)
human_train = [r for r in human_train if not overlap(r["name"], r["text"], alpha_key(r["text"]), bench)]
human_val = [r for r in human_val if not overlap(r["name"], r["text"], alpha_key(r["text"]), bench)]
print("human functions removed by the benchmark overlap rules at build time:", before - len(human_train))
seen, per_key, qwen, dropped = set(), {}, [], 0
for path in sorted(glob.glob("data/qwen/*/all.jsonl")):
    for r in rows(path):
        key = alpha_key(r["text"])
        if r["name"] in bench_names or key in bench_keys or overlap(r["name"], r["text"], key, bench):
            dropped += 1
            continue
        if r["text"] in seen or per_key.get(key, 0) >= 2:
            continue
        seen.add(r["text"])
        per_key[key] = per_key.get(key, 0) + 1
        qwen.append(r)
rng.shuffle(qwen)
n_val = max(50, len(qwen) // 20)
qwen_val, qwen_train = qwen[:n_val], qwen[n_val:]
for name, data in (("human_train", human_train), ("human_val", human_val), ("qwen_train", qwen_train), ("qwen_val", qwen_val)):
    with open("data/final/%s/%s.jsonl" % (out, name), "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
size = lambda data: round(sum(len(r["text"]) for r in data) / 1e6, 2)
print(json.dumps({"human_train": [len(human_train), size(human_train)], "human_val": len(human_val), "qwen_train": [len(qwen_train), size(qwen_train)],
                  "qwen_val": len(qwen_val), "qwen_distinct_code": len(per_key), "qwen_removed_by_overlap_at_build": dropped, "repeat": k}))
base = "data/final/%s/" % out
cmd = [".venv/bin/python", "-m", "lsllm.data.tokenize_corpus", "--inputs", base + "human_train.jsonl"] + [base + "qwen_train.jsonl"] * k + \
      ["--val-inputs", base + "human_val.jsonl", base + "qwen_val.jsonl", "--vocab-size", "512", "--variant", "B+", "--out", "data/tok/" + out]
subprocess.run(cmd, check=True)
print(open("data/tok/%s/meta.json" % out).read())
