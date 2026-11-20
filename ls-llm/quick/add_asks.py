"""A token folder = an existing one plus the training texts of the request suite, repeated K times.

    python -m quick.add_asks OUT BASE K [--clean] [--more FILE]

Every round is a new random order of all texts, so a window never shows the same program twice in a row.
--clean keeps only the texts that a benchmark model may see (`benchmark_clean` in data/asks/train_texts.jsonl);
without it the folder is for the board model only and must not be used for benchmark scores.
--more adds further verified texts (data/asks/qwen_texts.jsonl).
"""
import json
import random
import shutil
import sys
from array import array
from pathlib import Path

from lsllm.tok import Tok

out, base, k = Path("data/tok") / sys.argv[1], Path("data/tok") / sys.argv[2], int(sys.argv[3])
clean = "--clean" in sys.argv
rows = [json.loads(line) for line in open("data/asks/train_texts.jsonl")]
if "--more" in sys.argv:
    have = {row["name"] for row in rows}
    for line in open(sys.argv[sys.argv.index("--more") + 1]):
        row = json.loads(line)
        if row["name"] not in have:
            rows.append(dict(row, benchmark_clean=False))      # checked against the overlap rules by the caller if needed
if clean:
    rows = [row for row in rows if row["benchmark_clean"]]
out.mkdir(parents=True, exist_ok=True)
for name in ("tok.model", "tok.vocab", "val.bin", "val_general.bin"):
    if (base / name).exists():
        shutil.copy(base / name, out / name)
tok, rng, ids = Tok(out / "tok.model"), random.Random(0), array("H")
encoded = [tok.encode(row["text"], bos=True, eos=True) for row in rows]
for g in range(k):
    for doc in rng.sample(encoded, len(encoded)):
        ids.extend(doc)
with open(out / "train.bin", "wb") as f:
    f.write((base / "train.bin").read_bytes())
    ids.tofile(f)
meta = json.loads((base / "meta.json").read_text())
meta.update(train_tokens=meta["train_tokens"] + len(ids), train_docs=meta["train_docs"] + k * len(rows), asks_functions=len(rows), asks_repeats=k,
            asks_tokens=len(ids), asks_share_of_tokens=round(len(ids) / (meta["train_tokens"] + len(ids)), 3),
            asks_note="benchmark-clean texts only" if clean else "all request-suite texts: a board model, not for benchmark scores", made_from=str(base))
(out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(json.dumps({key: meta[key] for key in ("train_tokens", "asks_functions", "asks_repeats", "asks_tokens", "asks_share_of_tokens", "asks_note")}))
