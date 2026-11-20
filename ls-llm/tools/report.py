"""REPORT.md, written from the recorded files: every number in it is read from a result file, never typed.

    PYTHONPATH=. .venv/bin/python tools/report.py runs/final.json

Needs: docs/tables/*.md (tools/report_tables.py), docs/SPEED.md and docs/HARDWARE_CHANGES.md (tools/speed_report.py),
docs/SHAPE_SEARCH.md and docs/pareto.png (tools/pareto.py), docs/TIER_B.md (tools/tierb.py), EDITS.md (tools/edits_md.py)
and the logs of tools/finalize.py under results/logs/. A part whose file does not exist yet is reported as missing.
"""
import json
import os
import re
import sys

from lsllm.config import ModelConfig

num = lambda v: "{:,}".format(int(round(v)))


def read(path, default="(not produced yet: `%s`)"):
    return open(path).read().strip() if os.path.exists(path) else default % path


def section(path, title):
    """The text of one `## title` section of a generated markdown file, without its heading."""
    text = read(path, "")
    match = re.search(r"^## %s\n(.*?)(?=^## |\Z)" % re.escape(title), text, re.S | re.M)
    return match.group(1).strip() if match else "(not produced yet: section '%s' of `%s`)" % (title, path)


def result(path):
    return json.load(open(path)) if os.path.exists(path) else None


def run_row(folder, name):
    r = result(os.path.join(folder, name, "result.json"))
    return r if r and not r.get("failed") else None


def tierb_intro(spec):
    """The trained Tier B model in words, from its result files and its row of docs/TIER_B.md."""
    entry = next((m for m in spec["models"] if m["label"] == "Tier B"), None)
    if entry is None:
        return "No Tier B model was trained; the table below is the cost model alone."
    f = result(os.path.join(os.path.dirname(entry["float"]), "result.json"))
    q = result(os.path.join(entry["int"], "result.json")) if entry.get("int") else None
    a = result(os.path.join(os.path.dirname(spec["models"][0]["float"]), "result.json"))
    c = f["config"]
    text = ("**The trained Tier B model** (`%s`%s): %d layers, width %d, %d query heads on %d key/value head, ReLU feed-forward of %d: %s parameters, "
            "%.2f MiB of int8 weights, which do not fit the block RAM. It was trained for %s steps in all on benchmark-clean text (`%s`; Tier A: %s steps), so it is "
            "less trained than Tier A and still better: validation loss %.4f nats per token in float%s (Tier A float: %.4f), on the body tokens %.4f%s. "
            "Its benchmark rows are in section 3. " % (
                os.path.dirname(entry["float"]), ", int8 `%s`" % entry["int"] if q else "", c["n_layers"], c["dim"], c["n_heads"], c["n_kv_heads"], c["hidden_dim"],
                num(f["params"]), f["params"] / 2 ** 20, num(entry.get("steps_total", f["steps"])), f["args"]["data"],
                num(spec["models"][0].get("steps_total", a["steps"])), f["val_loss"], " and %.4f in int8" % q["val_loss"] if q else "",
                a["val_loss"], f["val_body"], " / %.4f" % q["val_body"] if q and q.get("val_body") else ""))
    row = re.search(r"^\| trained: .*$", read("docs/TIER_B.md", ""), re.M)
    if row:
        cells = [cell.strip() for cell in row.group(0).strip("|").split("|")]
        text += ("By the cost model below it needs %s cycles per token at context 64 and %s at 256: **%s and %s tokens per second**. "
                 "These are extrapolated counts, not counted on a generated program: the generator and the simulator stop at 2^19 bytes of data memory, "
                 "and this model needs the DDR2 changes of `docs/HARDWARE_CHANGES.md`, section 3." % (cells[8], cells[10], cells[9], cells[11]))
    return text


def techniques():
    """The table of measured speed techniques, from the search runs (600 steps each, same data, same seed)."""
    from lsllm.ls import cost
    rows = []

    def speed(cfg, n):
        plain = ModelConfig(**dict(cfg.to_dict(), attn="softmax", window=0))
        span = (min(n, cfg.window) if cfg.window else n) - 1
        options = {"norm": "rms", "ffn": cfg.ffn}
        try:
            return cost.token_cycles(plain, options, span, stop=True, syntax_filter=cfg.vocab_size == 512)[0]
        except ValueError:
            return None

    def line(what, variant, folder, name, note=""):
        r = run_row(folder, name)
        if r is None:
            return
        cfg = ModelConfig(**r["config"])
        c64, c256 = speed(cfg, 64), speed(cfg, 256)
        per_token = r["val_loss"] / (r["val_bpb"] * 0.6931471805599453)
        rows.append([what, variant, num(r["params"]), "%.4f" % r["val_bpb"], num(c64) if c64 else "-", num(c256) if c256 else "-",
                     "%.0f" % (1e8 / c64 * per_token) if c64 else "-", "%.0f" % (1e8 / c256 * per_token) if c256 else "-", note])

    for name, variant in (("L5d64h8k4", "8 heads, 4 KV heads (stories260K)"), ("L5d64h4k2", "4 heads, 2 KV heads"), ("L5d64h2k2", "2 heads, 2 KV heads"),
                          ("L5d64h2k1", "2 heads, 1 KV head"), ("L5d64h1k1", "1 head")):
        line("heads and shared KV heads (5 x 64, SwiGLU 172)", variant, "runs/p", name)
    line("feed-forward at equal parameters (5 x 64)", "SwiGLU, F = 172", "runs/p", "L5d64h2k1")
    line("feed-forward at equal parameters (5 x 64)", "ReLU, F = 255", "runs/p", "L5d64h2k1relu")
    line("feed-forward at equal parameters (3 x 96)", "SwiGLU, F = 272", "runs/p", "L3d96h2k1F272")
    line("feed-forward at equal parameters (3 x 96)", "ReLU, F = 408", "runs/p", "L3d96h2k1reluF408")
    line("vocabulary (5 x 64, 2 heads)", "512 tokens", "runs/p", "L5d64h2k1")
    line("vocabulary (5 x 64, 2 heads)", "256 tokens", "runs/p", "L5d64h2k1v256", "fewer bytes per token: compare bits per byte and bytes per second")
    line("attention window (5 x 64, 2 heads)", "none", "runs/p", "L5d64h2k1")
    line("attention window (5 x 64, 2 heads)", "last 64 keys", "runs/p", "L5d64h2k1w64")
    header = ["technique", "variant", "parameters", "val bits/byte", "cycles/token @64", "cycles/token @256", "bytes/s @64", "bytes/s @256", "note"]
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def main(manifest):
    spec = json.load(open(manifest))
    a = spec["models"][0]
    name = os.path.basename(os.path.dirname(a["float"]))
    fl = result(os.path.join(os.path.dirname(a["float"]), "result.json"))
    q = result(os.path.join(a["int"], "result.json")) if a.get("int") else None
    cfg = ModelConfig(**fl["config"])
    data_meta = result(os.path.join(fl["args"]["data"], "meta.json")) or {}
    shape = "%d layers, width %d, %d query heads on %d key/value head%s, %s feed-forward of %d, vocabulary %d, context %d%s" % (
        cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, "" if cfg.n_kv_heads == 1 else "s", "ReLU" if cfg.ffn == "relu" else "SwiGLU", cfg.hidden_dim,
        cfg.vocab_size, cfg.max_seq_len, ", attention window %d" % cfg.window if cfg.window else "")
    compare = read("results/logs/compare_%s.txt" % name)
    overflow = read("results/logs/overflow_%s.txt" % name)
    overflow_last = overflow.split("\n")[-1]
    early = read("results/logs/early_exit_%s.txt" % name)
    out = []
    w = out.append
    w("# LS-LLM: a language model that writes Python on the LS 8-bit CPU")
    w("")
    w("Final report. Generated by `tools/report.py %s`; every number is read from a file that a run wrote" % manifest)
    w("(`results/results.jsonl`, `runs/*/result.json`, the exported integer model, the generated LS program).")
    w("Practice-set scores and benchmark scores are in separate tables and are never mixed.")
    w("")
    w("## 1. Summary")
    w("")
    w(read("docs/tables/summary.md"))
    w("")
    w("## 2. The model in one table")
    w("")
    w("Tier A is the main model: everything it needs is in block RAM. Tier B keeps its weights in the DDR2 (section 12).")
    w("Cycles and tokens per second: level L1 of `docs/SPEC.md`, 100 MHz, complete program, greedy decoding with the syntax filter.")
    w("")
    w(read("docs/tables/models.md"))
    w("")
    w("Shape of Tier A: %s; %s parameters, all int8." % (shape, num(fl["params"])))
    if q:
        w("Validation loss on the same %s validation tokens: float %.4f nats per token, int8 %.4f (straight after calibration, before" % (
            num(data_meta.get("val_tokens", 0)), q["float_val_loss"], q["val_loss"]))
        w("quantization-aware training: %.4f). The int8 number is the loss of the integers that the CPU computes: the training" % q["calibrated_val_loss"])
        w("forward pass and the integer reference gave identical logits on the check windows (`bit_exact`: %s)." % q["bit_exact"])
    w("")
    w("## 3. Benchmark scores")
    w("")
    w("HumanEval (164 tasks), MBPP test split (500 tasks) and the custom set of 200 very simple tasks (test half). pass@1 with")
    w("greedy decoding, and pass@1 / pass@10 from 10 samples per task at temperature 0.8 (unbiased estimator). `syntax ok` =")
    w("the completed function parses, `runs ok` = it runs through the tests without an exception (they may still fail).")
    w("Generated code was executed only inside the sandbox (`lsllm/sandbox.py`, no network, no file access, time limit).")
    w("Every model went through the same harness, the same prompts and the same budget of new tokens.")
    w("")
    w(read("docs/tables/scores_test.md"))
    w("")
    w("### The custom set by stratum")
    w("")
    w("The 200 custom tasks come from 40 families. For 24 families (120 tasks, \"near\") the generated training text contains")
    w("functions of the same kind, though never the task itself; for 16 families (80 tasks, \"far\") it contains nothing of the")
    w("kind. A pass on a near task shows what was learned, a pass on a far task would show generalisation. Greedy decoding.")
    w("")
    w(read("docs/tables/strata.md"))
    w("")
    w("### Practice-set scores (not benchmark scores)")
    w("")
    w("These tasks were used to make decisions (shape, data mix, decoding, weight edits). They are shown so that the decisions can be checked.")
    w("")
    w(read("docs/tables/scores_practice.md"))
    w("")
    w("### Programs on request (recall of trained programs, not a benchmark)")
    w("")
    w(read("docs/tables/asks.md"))
    w("")
    w("### Published numbers of other small code models (not run by us)")
    w("")
    w(read("docs/tables/published.md"))
    w("")
    w("## 4. Research summary")
    w("")
    w(read("docs/tables/research.md"))
    w("")
    w("## 5. Data and tokenizer")
    w("")
    w(read("docs/tables/data.md"))
    w("")
    w("## 6. Shape search and the Pareto front")
    w("")
    w(read("docs/tables/shape.md"))
    w("")
    w("![Pareto chart](docs/pareto.png)")
    w("")
    w(read("docs/SHAPE_SEARCH.md").split("\n\n", 1)[-1])
    w("")
    w("## 7. Speed")
    w("")
    w("### 7.1 Cycles per token and tokens per second")
    w("")
    w(section("docs/SPEED.md", "Cycles per token and tokens per second"))
    w("")
    w("### 7.2 Where the cycles go")
    w("")
    w(section("docs/SPEED.md", "Where the cycles go (generated token, greedy, filter on)"))
    w("")
    w("### 7.3 What the stream instruction is worth")
    w("")
    w(section("docs/SPEED.md", "What the stream instruction is worth"))
    w("")
    w("### 7.4 Techniques that were tried and measured")
    w("")
    w("Each pair below differs in one thing only; all runs are 600 training steps on the same data with the same seed, so the")
    w("validation numbers rank the variants but are far from converged. Speed is counted on the generated program of each shape.")
    w("")
    w(techniques())
    w("")
    w(read("docs/tables/techniques.md"))
    w("")
    w("Early exit, measured on the final float model (`tools/early_exit.py`):")
    w("")
    w("```")
    w(early)
    w("```")
    w("")
    w("### 7.5 Hardware changes")
    w("")
    w("Full list with costs: `docs/HARDWARE_CHANGES.md`. Section 1 there is what the model needs (level L1; all main numbers assume")
    w("exactly that). The optional changes, never assumed in the main numbers:")
    w("")
    w(section("docs/HARDWARE_CHANGES.md", "2. Optional changes that make it faster (level L2)").split("\n\n", 1)[-1])
    w("")
    w("### 7.6 Other small machines that run a small language model")
    w("")
    w(read("docs/tables/systems.md"))
    w(read("docs/tables/systems_ours.md", ""))
    w("")
    w("## 8. Integer arithmetic: float against int8")
    w("")
    w(read("docs/tables/integer.md"))
    w("")
    w("Float model against integer model on the same prompts (`lsllm.intref.compare`):")
    w("")
    w("```")
    w(compare)
    w("```")
    w("")
    w("Accumulator use (`lsllm.intref.overflow`; the limit of the signed 24-bit accumulator is 8,388,607): " + overflow_last + ".")
    w("The full table per stream is in `results/logs/overflow_%s.txt`." % name)
    w("")
    w("## 9. Syntax filter")
    w("")
    w(read("docs/tables/filter.md"))
    w("")
    w("## 10. Weight-space editing")
    w("")
    w(read("docs/tables/editing.md"))
    w("")
    w("Held-out check, test sets, greedy (from `EDITS.md`, where the practice scores and every edit are listed as well):")
    w("")
    w(section("EDITS.md", "Held-out check: test sets before and after (greedy; measured after the sessions ended, never seen by them)"))
    w("")
    w(read("docs/tables/editing_outcome.md", ""))
    w("")
    w("## 11. Exports for the board")
    w("")
    w(read("docs/tables/exports.md"))
    w("")
    w("## 12. Tier B: weights in the DDR2")
    w("")
    w(tierb_intro(spec))
    w("")
    w(re.sub(r"(?m)^## ", "### ", read("docs/TIER_B.md").split("\n", 1)[-1].strip()))
    w("")
    w("## 13. What is not verified, and limits")
    w("")
    w(read("docs/tables/limits.md"))
    w("")
    w("## 14. Reproduce")
    w("")
    w(read("docs/tables/reproduce.md"))
    open("REPORT.md", "w").write("\n".join(out) + "\n")
    print("wrote REPORT.md: %d lines" % len(out))


if __name__ == "__main__":
    main(sys.argv[1])
