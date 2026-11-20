"""Practice-solution loss (tools/practice_loss.py) of every run folder under the given directories, cached in
practice_loss.json next to each checkpoint.   PYTHONPATH=. .venv/bin/python tools/practice_all.py runs/p runs/q runs/x"""
import glob
import json
import os
import sys

from tools.practice_loss import losses

for folder in sys.argv[1:]:
    for ckpt in sorted(glob.glob(os.path.join(folder, "*", "ckpt.pt"))):
        out = os.path.join(os.path.dirname(ckpt), "practice_loss.json")
        if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(ckpt):
            continue
        try:
            got = losses(ckpt)
        except Exception as error:
            print(ckpt, "failed:", error)
            continue
        json.dump({bench: {"nats_per_token": v[0], "nats_per_solution": v[1], "tasks": v[2]} for bench, v in got.items()}, open(out, "w"), indent=1)
        print(ckpt, {bench: round(v[1], 1) for bench, v in got.items()}, flush=True)
