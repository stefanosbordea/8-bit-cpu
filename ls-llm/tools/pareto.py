"""Shape search table and Pareto chart: quality (validation bits per byte, practice-set scores) against speed
(tokens per second on the LS CPU at 100 MHz, from the cycle cost model) for every run in a directory.

    .venv/bin/python tools/pareto.py runs/p,runs/q docs/SHAPE_SEARCH.md docs/pareto.png
"""
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lsllm.config import ModelConfig
from lsllm.ls import cost

CLOCK = 100e6
POSITIONS = (15, 63, 255)                   # context 16, 64, 256
BUDGET = 110 * 4096                         # Tier A: the data memory image must fit 110 of the 135 block RAMs


def practice(name):
    """{(benchmark, filter): metrics} of the practice-set rows recorded for the model `name`."""
    found = {}
    for line in open("results/results.jsonl"):
        r = json.loads(line)
        if r["model"] in (name, name + "+filter") and r["split"] == "practice":
            found[r["benchmark"], r["filter"]] = r["metrics"]
    return found


def front(points):
    """Names of the points (name, speed, loss) that no other point beats in both speed and loss."""
    return {a[0] for a in points if not any(b[1] >= a[1] and b[2] <= a[2] and b[1:] != a[1:] for b in points)}


def main(runs, out_md, out_png):
    rows = []
    for path in sorted(path for folder in runs.split(",") for path in glob.glob(os.path.join(folder, "*", "result.json"))):
        r = json.load(open(path))
        if r.get("failed"):
            continue
        name, cfg = os.path.basename(os.path.dirname(path)), ModelConfig(**r["config"])
        options = {"norm": "dyt" if cfg.norm == "dyt" else "rms", "ffn": cfg.ffn}
        shape = ModelConfig(**dict(r["config"], attn="softmax", window=0))
        cycles, image = [], None
        for pos in POSITIONS:
            reach = min(pos, cfg.window - 1) if cfg.window else pos     # a window of w keys costs like context w
            try:                                                        # the complete program: prompt, stop rule, syntax filter
                total, parts, build = cost.token_cycles(shape, options, reach, stop=True, syntax_filter=cfg.vocab_size == 512)
                cycles.append(total)
                image = len(build.image)
            except ValueError:                                          # does not fit the 2^19 bytes of data memory: no program, no cycle count
                cycles.append(float("inf"))
        bpb = r["val_bpb"]
        per_token = r["val_loss"] / (r["val_bpb"] * math.log(2))             # bytes per token of the validation text
        byte_rate = [CLOCK / c * per_token for c in cycles]
        scores = practice(name)
        cached = os.path.join(os.path.dirname(path), "practice_loss.json")      # written by tools/practice_all.py
        solution = json.load(open(cached)) if os.path.exists(cached) else {}
        rows.append(dict(name=name, cfg=cfg, params=r["params"], kv=cfg.kv_cache_bytes(), image=image, val=r["val_loss"], bpb=bpb,
                         steps=r["steps"], data=os.path.basename(r["args"]["data"]), cycles=cycles,
                         rate=[CLOCK / c for c in cycles], byte_rate=byte_rate, scores=scores, solution=solution))
    rows.sort(key=lambda row: row["bpb"])
    best = front([(row["name"], row["byte_rate"][1], row["bpb"]) for row in rows if row["image"] and row["image"] <= BUDGET])
    lines = ["# Shape search", "",
             "Every run: %s. Validation loss on the same validation text for every run, in bits per byte so that" % runs,
             "vocabularies can be compared. Speed: cycles of one token from the cost model (`lsllm.ls.cost`, L1 machine,",
             "100 MHz, complete program with prompt buffer, stop rule and syntax filter) at context 16 / 64 / 256.",
             "Memory = the data memory image that the generator lays out for the shape: weights, E, KV cache, lookup tables,",
             "position records with the RoPE tables, filter tables, scratch. The Tier A budget is %s bytes (110 block RAMs)." % "{:,}".format(BUDGET),
             "`front` marks the runs inside the budget that no other such run beats in both bits per byte and bytes per second at context 64.", "",
             "| run | data | steps | L | D | heads | KV heads | F | ffn | vocab | window | parameters | KV cache bytes | data memory bytes "
             "| val nats/token | val bits/byte | cycles/token @16 | @64 | @256 | tokens/s @16 | @64 | @256 | bytes/s @64 "
             "| custom practice syntax ok / runs ok / pass@1 | MBPP practice syntax ok / runs ok / pass@1 "
             "| practice-solution loss, nats per solution: custom / MBPP | front |",
             "|" + "---|" * 26]
    for row in rows:
        c = row["cfg"]
        show = lambda bench: " / ".join("%.1f%%" % (100 * row["scores"][bench, "off"][m]) for m in ("syntax_ok", "runs_ok", "pass@1")) \
            if (bench, "off") in row["scores"] else "-"
        cells = [row["name"], row["data"], row["steps"], c.n_layers, c.dim, c.n_heads, c.n_kv_heads, c.hidden_dim, c.ffn, c.vocab_size,
                 c.window or "-", "{:,}".format(row["params"]), "{:,}".format(row["kv"]),
                 ("{:,}".format(row["image"]) + ("" if row["image"] <= BUDGET else " (over budget)")) if row["image"] else "more than 524,288",
                 "%.4f" % row["val"], "%.4f" % row["bpb"]] + ["{:,}".format(v) if v != float("inf") else "-" for v in row["cycles"]] + \
                ["%.0f" % v if v else "-" for v in row["rate"]] + ["%.0f" % row["byte_rate"][1] if row["byte_rate"][1] else "-", show("custom"), show("mbpp"),
                                                     " / ".join("%.1f" % row["solution"][b]["nats_per_solution"] for b in ("custom", "mbpp")) if row["solution"] else "-",
                                                     "yes" if row["name"] in best else ""]
        lines.append("| " + " | ".join(str(v) for v in cells) + " |")
    open(out_md, "w").write("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in rows:
        if not row["byte_rate"][1]:
            continue
        on, inside = row["name"] in best, bool(row["image"]) and row["image"] <= BUDGET
        ax.scatter(row["byte_rate"][1], row["bpb"], s=46, color="#1f6fb4" if on else "#9aa3ad", zorder=3, marker="o" if inside else "x")
        ax.annotate(row["name"], (row["byte_rate"][1], row["bpb"]), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    line = sorted((row["byte_rate"][1], row["bpb"]) for row in rows if row["name"] in best)
    ax.plot([p[0] for p in line], [p[1] for p in line], color="#1f6fb4", linewidth=1, zorder=2)
    ax.set_xlabel("speed on the LS CPU at 100 MHz, context 64 (bytes of Python per second)")
    ax.set_ylabel("validation loss (bits per byte, lower is better)")
    ax.set_title("Shape search: quality against speed (blue: Pareto front; x: over the memory budget)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("\n".join(lines[7:]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
