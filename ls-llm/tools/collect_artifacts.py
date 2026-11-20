#!/usr/bin/env python3
"""artifacts/: the final models as files in the repository (runs/ is not tracked).

    PYTHONPATH=. .venv/bin/python tools/collect_artifacts.py runs/final.json

Per model of the manifest: the float checkpoint and its result.json, the integer model (model.json, model.npz) and
its result.json; for the board model also the repair record and the request texts it writes.
"""
import json
import os
import shutil
import sys

spec = json.load(open(sys.argv[1]))
wanted = [(entry.get("folder") or entry["label"].lower().replace(" ", "_"), entry["float"], entry.get("int")) for entry in spec["models"]]
board = spec.get("board_model")
if board:
    wanted.append(("board", board["float"], board["int"]))
lines = ["# Final models", "", "Copied by `tools/collect_artifacts.py` from the run folders named in `runs/final.json`.", "",
         "| folder | float checkpoint from | integer model from | files |", "|---|---|---|---|"]
for name, ckpt, integer in wanted:
    out = os.path.join("artifacts", name)
    os.makedirs(out, exist_ok=True)
    files = []
    for source, target in ((ckpt, "float.pt"), (os.path.join(os.path.dirname(ckpt), "result.json"), "float_result.json")):
        if os.path.exists(source):
            shutil.copy(source, os.path.join(out, target))
            files.append(target)
    for item in ("model.json", "model.npz", "result.json", "qat_result.json", "polish.json", "polish.jsonl", "texts.jsonl"):
        if integer and os.path.exists(os.path.join(integer, item)):
            shutil.copy(os.path.join(integer, item), os.path.join(out, item))
            files.append(item)
    lines.append("| `artifacts/%s` | `%s` | `%s` | %s |" % (name, ckpt, integer or "-", ", ".join(files)))
os.makedirs("artifacts/tokenizer", exist_ok=True)
folders = sorted({json.load(open(os.path.join(integer, "model.json")))["tokenizer"] for name, ckpt, integer in wanted if integer})
for item in ("tok.model", "tok.vocab"):
    shutil.copy(os.path.join(folders[0], item), os.path.join("artifacts/tokenizer", item))
same = all(open(os.path.join(folder, "tok.model"), "rb").read() == open(os.path.join(folders[0], "tok.model"), "rb").read() for folder in folders)
lines += ["", "`artifacts/tokenizer/` holds the SentencePiece model of all of them (`tok.model`, `tok.vocab`; the same file in %s: %s)." % (
              ", ".join("`%s`" % folder for folder in folders), "checked" if same else "NOT the same, check"),
          "Each `model.json` names its tokenizer folder under `data/tok/`, which is not tracked: on a fresh copy of the repository, put the two",
          "files there (for example `data/tok/f512p/`) before running a tool on a model.", "",
          "The board model is `artifacts/board`: `PYTHONPATH=. .venv/bin/python tools/board_ask.py artifacts/board --list`."]
open("artifacts/README.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
