#!/usr/bin/env python3
"""Make the Google Colab bundle: a zip with the code and the token streams, and a notebook that trains with it.

    .venv/bin/python tools/make_colab_bundle.py

Writes notebooks/lsllm_colab_bundle.zip and notebooks/train_on_colab.ipynb. The notebook runs the same trainer
(`lsllm.train.run`, `lsllm.qat.train`) with larger budgets than the laptop allows; checkpoints are downloaded at the end.
Run it again whenever the final data set or shape changes (DATA, SHAPE_A, SHAPE_B below).
"""
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
DATA = "data/tok/f512v1"                      # the training set of the final runs (D-data-7)
KEEP = ["lsllm", "third_party/llama2c", DATA, "data/bench", "docs/SPEC.md", "tools/practice_loss.py", "tools/val_body.py"]

with zipfile.ZipFile("notebooks/lsllm_colab_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for top in KEEP:
        if os.path.isfile(top):
            z.write(top)
        for folder, _, files in os.walk(top):
            for name in files:
                if "__pycache__" not in folder and not name.endswith(".pyc"):
                    z.write(os.path.join(folder, name))


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "\n".join(lines)}


def text(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


SHAPE_A = "--n-layers 2 --dim 160 --n-heads 4 --n-kv-heads 1 --ffn relu --hidden-dim 219"        # D-shape-2
SHAPE_B = "--n-layers 4 --dim 192 --n-heads 4 --n-kv-heads 1 --ffn relu --hidden-dim 510"        # Tier B (weights in DDR2)
COMMON = "--data %s --device cuda --batch-size 128 --prompt-weight 0.1 --eval-every 2000" % DATA
cells = [
    text("# LS-LLM: longer training runs on a Colab GPU",
         "The laptop (M1, 8 GB) trains the Tier A model at about 70,000 tokens per second, so the runs of the report are short",
         "(10,000 steps of 64 x 256 tokens). This notebook runs the same code with the same data on a Colab GPU for longer.",
         "Upload `lsllm_colab_bundle.zip` (made by `tools/make_colab_bundle.py`) when asked.",
         "Runtime > Change runtime type > GPU before you start."),
    code("from google.colab import files", "up = files.upload()", "!unzip -q -o lsllm_colab_bundle.zip", "!pip -q install sentencepiece", "!nvidia-smi | head -15"),
    text("## Tier A float model: the shape of the report, 8 times more tokens"),
    code("!python -m lsllm.train.run --out runs/f/A_long %s %s --steps 40000 --lr 1e-3" % (COMMON, SHAPE_A)),
    text("## Tier B float model"),
    code("!python -m lsllm.train.run --out runs/f/B_long %s %s --steps 40000 --lr 6e-4" % (COMMON, SHAPE_B)),
    text("## Quantization-aware training (the forward pass is the arithmetic of the LS CPU, bit for bit)"),
    code("!python -m lsllm.qat.train --float runs/f/A_long/ckpt.pt --data %s --out runs/qat/A_long --steps 6000 --kd-weight 0.5 --batch-size 32 --device cuda" % DATA,
         "!python -m lsllm.qat.train --float runs/f/B_long/ckpt.pt --data %s --out runs/qat/B_long --steps 6000 --kd-weight 0.5 --batch-size 32 --device cuda" % DATA),
    text("## Download the results",
         "Copy the downloaded `runs/` folders into the project on the laptop, then run `tools/finalize.py` for each model",
         "(benchmarks in the sandbox, export, speed documents) and `tools/report.py`."),
    code("!zip -q -r runs.zip runs", "files.download('runs.zip')"),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("notebooks/train_on_colab.ipynb", "w"), indent=1)
print("wrote notebooks/train_on_colab.ipynb and notebooks/lsllm_colab_bundle.zip (%.1f MB)" % (os.path.getsize("notebooks/lsllm_colab_bundle.zip") / 1e6))
