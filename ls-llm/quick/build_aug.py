"""Training set with renaming augmentation (lsllm.data.rename): every function appears once as written and
several times with new parameter and variable names. Tokenizer and validation set are those of BASE, unchanged.
Usage: python -m quick.build_aug OUT BASE HUMAN_COPIES QWEN_ORIGINALS QWEN_COPIES [SHARE]"""
import json
import random
import shutil
import sys
from pathlib import Path

from lsllm.data.rename import name_pool, rename
from lsllm.data.tokenize_corpus import encode_texts
from lsllm.tok import Tok

out, base = Path("data/tok") / sys.argv[1], Path("data/tok") / sys.argv[2]
human_copies, qwen_originals, qwen_copies = map(int, sys.argv[3:6])
share = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0             # probability that a given name is replaced
read = lambda path: [json.loads(line)["text"] for line in open(path)]
human, qwen = read("data/final/f512a/human_train.jsonl"), read("data/final/%s/qwen_train.jsonl" % sys.argv[2])
rng = random.Random(0)
pool = name_pool(human)
texts, made = list(human) + qwen * qwen_originals, {"human": 0, "qwen": 0}
for kind, source, copies in (("human", human, human_copies), ("qwen", qwen, qwen_copies)):
    for text in source:
        seen = {text}
        for g in range(copies):
            new = rename(text, rng, pool, share)
            if new and new not in seen:
                seen.add(new)
                texts.append(new)
                made[kind] += 1
rng.shuffle(texts)
out.mkdir(parents=True, exist_ok=True)
for name in ("tok.model", "tok.vocab", "val.bin"):
    shutil.copy(base / name, out / name)
count = encode_texts(Tok(out / "tok.model"), texts, out / "train.bin")
meta = json.loads((base / "meta.json").read_text())
meta.update({"train_" + key: value for key, value in count.items()})
meta.update({"tokenizer": "copied from data/tok/" + sys.argv[2], "augmentation": "lsllm.data.rename", "name_pool": len(pool[0]),
             "rename_share": share, "human_renamed_copies": made["human"], "qwen_repeat": qwen_originals, "qwen_renamed_copies": made["qwen"]})
(out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(json.dumps(meta, indent=1))
print("pool sample:", pool[0][:60])
