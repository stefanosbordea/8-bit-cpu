"""CodeSearchNet (Python) subset under permissive licences only: plain functions, canonical, overlap-filtered.

The original zip is read by HTTP range requests (no 940 MB download). The licence of every repository comes from
the dataset's own table python_licenses.pkl, read with an unpickler that refuses every class (the table is a
third-party pickle). Usage: python -m quick.csn OUT_DIR SHARD [SHARD ...]   e.g. valid_0 test_0 train_0
"""
import gzip
import hashlib
import importlib.util
import io
import json
import os
import pickle
import sys
import zipfile

from lsllm.alpha import alpha_key
from lsllm.canon import canon
from lsllm.data.corpus.overlap import load_benchmarks, overlap

spec = importlib.util.spec_from_file_location("m08", "research/08_dataset_measurements.py")
m08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m08)

COPYLEFT = ("GNU GENERAL PUBLIC", "GNU LESSER", "GNU AFFERO", "GNU LIBRARY", "Mozilla Public", "ShareAlike", "Eclipse Public", "CeCILL", "European Union Public")
MARKS = {"MIT": "Permission is hereby granted, free of charge", "BSD": "Redistribution and use in source and binary forms",
         "Apache-2.0": "Apache License", "ISC": "Permission to use, copy, modify, and/or distribute this software",
         "Unlicense": "This is free and unencumbered software released into the public domain"}


class NoClasses(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("classes are not allowed: %s.%s" % (module, name))


def licence_of(text):
    """SPDX-like name of a permissive licence text, or None (copyleft, unknown or mixed)."""
    flat = " ".join(text.split())
    if any(mark.lower() in flat.lower() for mark in COPYLEFT):
        return None
    hits = [name for name, mark in MARKS.items() if mark.lower() in flat.lower()]
    if hits == ["Apache-2.0"] and "Version 2.0" not in flat:
        return None
    return hits[0] if len(hits) == 1 else None


def main(out, shards):
    os.makedirs(out, exist_ok=True)
    zf = zipfile.ZipFile(m08.HttpRangeFile(m08.CSN_ZIP))
    names = zf.namelist()
    cached = "/Volumes/CARMIX-WORK/ls-llm-work/python_licenses.pkl"
    raw = open(cached, "rb").read() if os.path.exists(cached) else zf.read([n for n in names if n.endswith("python_licenses.pkl")][0])
    table = NoClasses(io.BytesIO(raw)).load()
    del raw
    allowed, count = {}, {"repos": len(table), "permissive": 0}
    for repo, value in table.items():        # value: list of (path of a licence file, its text); every file must be permissive
        found = [licence_of(text) for _, text in value if isinstance(text, str)]
        if found and all(found):
            allowed[repo] = found[0]
    count["permissive"] = len(allowed)
    del table
    bench = load_benchmarks()
    seen_text, seen_key = set(), set()
    for path in ("data/corpus/train.jsonl", "data/corpus/val.jsonl"):
        if os.path.exists(path):
            for line in open(path):
                rec = json.loads(line)
                seen_text.add(rec["text"])
    stats = {"rows": 0, "method": 0, "licence": 0, "not_canonical": 0, "size": 0, "overlap": 0, "duplicate": 0, "kept": 0}
    used = {}
    with open(out + "/train.jsonl", "w") as ftrain, open(out + "/val.jsonl", "w") as fval:
        for shard in shards:
            part = shard.split("_")[0]
            member = [n for n in names if n.endswith("jsonl/%s/python_%s.jsonl.gz" % (part, shard))][0]
            for line in gzip.open(io.BytesIO(zf.read(member)), "rt"):
                row = json.loads(line)
                stats["rows"] += 1
                if "." in row["func_name"]:
                    stats["method"] += 1
                    continue
                if row["repo"] not in allowed:
                    stats["licence"] += 1
                    continue
                text = canon(row["original_string"])
                if text is None or not text.startswith("def "):
                    stats["not_canonical"] += 1
                    continue
                lines = text.count("\n") + 1
                if lines < 2 or lines > 40 or len(text) > 1500 or "<NL>" in text or "<IND>" in text or "<DED" in text or "<LF>" in text:
                    stats["size"] += 1
                    continue
                key = alpha_key(text)
                if overlap(row["func_name"], text, key, bench):
                    stats["overlap"] += 1
                    continue
                if text in seen_text or key in seen_key:
                    stats["duplicate"] += 1
                    continue
                seen_text.add(text)
                seen_key.add(key)
                rec = {"text": text, "name": row["func_name"], "source": "CodeSearchNet/" + row["repo"], "licence": allowed[row["repo"]], "path": row["path"],
                       "kind": "function", "n_lines": lines, "has_docstring": '"""' in text.split("\n", 2)[1] if lines > 1 else False}
                (fval if int(hashlib.sha1(text.encode()).hexdigest(), 16) % 50 == 0 else ftrain).write(json.dumps(rec) + "\n")
                used[row["repo"]] = allowed[row["repo"]]
                stats["kept"] += 1
            print(shard, json.dumps(stats), flush=True)
    json.dump({"licence_table": count, "stats": stats, "shards": shards, "repositories_used": len(used), "licences_used": sorted(set(used.values()))}, open(out + "/stats.json", "w"), indent=1)
    json.dump(used, open(out + "/repos.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
