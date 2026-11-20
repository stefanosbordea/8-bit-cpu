"""Give token folders the second validation set (held-out self-contained functions and Qwen functions, data/tok/val2):
the old validation file is kept as val_general.bin. Usage: python -m quick.use_val2 NAME [NAME ...]"""
import json
import shutil
import sys

v2 = json.load(open("data/tok/val2/meta.json"))
for name in sys.argv[1:]:
    folder = "data/tok/" + name
    meta = json.load(open(folder + "/meta.json"))
    if meta.get("validation") == "val2":
        continue
    shutil.copy(folder + "/val.bin", folder + "/val_general.bin")
    shutil.copy("data/tok/val2/val.bin", folder + "/val.bin")
    meta.update({key: v2[key] for key in ("val_docs", "val_tokens", "val_bytes", "val_dropped", "bytes_per_token")})
    meta["validation"] = "val2"
    meta["validation_note"] = "held-out self-contained human functions and 51 Qwen functions (counts in data/tok/val2/meta.json); the general validation text is val_general.bin"
    json.dump(meta, open(folder + "/meta.json", "w"), indent=1)
    print(name, "now validates on val2:", meta["val_tokens"], "tokens")
