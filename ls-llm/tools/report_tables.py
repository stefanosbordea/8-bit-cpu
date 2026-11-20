"""Tables of the final report, made from the recorded files (results/results.jsonl, runs/*/result.json, the
exported integer models). Nothing here is typed by hand: every number is read from the file that the run wrote.

    PYTHONPATH=. .venv/bin/python tools/report_tables.py runs/final.json        writes docs/tables/*.md

runs/final.json names the models of the report:
    {"models": [{"label": "Tier A", "float": "runs/f/NAME/ckpt.pt", "int": "runs/qat/NAME", "edited": "edits/s2"}, ...],
     "comparators": ["stories260K (karpathy/tinyllamas, as published)", "ngram-432KB"]}
"""
import json
import os
import sys

from lsllm.config import ModelConfig
from lsllm.ls import cost

BENCHES = (("custom", "custom 200"), ("mbpp", "MBPP"), ("humaneval", "HumanEval"))
pct = lambda v: "-" if v is None else "%.1f%%" % (100 * v)
num = lambda v: "{:,}".format(int(round(v)))


def results():
    rows = [json.loads(line) for line in open("results/results.jsonl")]
    latest = {}
    for r in rows:                                  # a later run of the same setting replaces the earlier one
        key = (r["model"].replace("+filter", ""), r["variant"], r.get("edit", ""), r["filter"] or "off", r["decoding"]["temperature"] > 0,
               r["n_samples"], r["benchmark"], r["split"], r["decoding"]["max_new_tokens"])
        latest[key] = r
    return latest


def score_rows(latest, model, variant, edit, split, label):
    """Rows of the score table for one model: filter off/on x greedy/sampled, all benchmarks side by side."""
    out = []
    for flt in ("off", "on", "on, no docstring"):
        for sampled in (False, True):
            cells, found = [], False
            for bench, _ in BENCHES:
                got = [r for k, r in latest.items() if k[:5] == (model, variant, edit, flt, sampled) and k[6] == bench and k[7] == split
                       and (k[5] >= 10) == sampled]
                if not got:
                    cells += ["-"] * 4
                    continue
                found, m = True, got[-1]["metrics"]
                cells += [pct(m["pass@1"]), pct(m["pass@10"]), pct(m["syntax_ok"]), pct(m["runs_ok"])]
            if found:
                out.append([label, variant, edit or "-", flt, "10 samples, T = 0.8" if sampled else "greedy"] + cells)
    return out


def score_table(rows):
    header = ["model", "arithmetic", "weight edit", "syntax filter", "decoding"]
    for _, name in BENCHES:
        header += [name + " pass@1", name + " pass@10", name + " syntax ok", name + " runs ok"]
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def strata(latest, wanted):
    """docs/tables/strata.md: the custom test set split into near families (the generated training text covers the
    family, not the task) and far families (it does not), greedy rows only, read from the stored samples."""
    import gzip
    tasks = {}
    for line in open("data/bench/custom_test.jsonl"):
        task = json.loads(line)
        tasks[task["task_id"]] = task["meta"]["stratum"]
    total = {name: sum(1 for stratum in tasks.values() if stratum == name) for name in ("near", "far")}
    lines = ["| model | arithmetic | weight edit | syntax filter | near families: passed of %d | far families: passed of %d |" % (total["near"], total["far"]),
             "|---|---|---|---|---|---|"]
    found = {}
    for key, r in sorted(latest.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        model, variant, edit, flt, sampled, n, bench, split, tokens = key
        path = os.path.join("results", r.get("samples", ""))
        if bench != "custom" or split != "test" or sampled or n != 1 or not os.path.exists(path) or (model, variant) not in wanted:
            continue
        passed = {"near": 0, "far": 0}
        for line in gzip.open(path, "rt"):
            sample = json.loads(line)
            if sample["outcome"] == "pass" and sample["task_id"] in tasks:
                passed[tasks[sample["task_id"]]] += 1
        lines.append("| %s | %s | %s | %s | %d (%.1f%%) | %d (%.1f%%) |" % (model, variant, edit or "-", flt, passed["near"], 100 * passed["near"] / total["near"],
                                                                   passed["far"], 100 * passed["far"] / total["far"]))
        if sum(passed.values()) >= sum(found.get((model, variant, edit), (0, 0))):          # the better of filter on and off
            found[(model, variant, edit)] = (passed["near"], passed["far"])
    open("docs/tables/strata.md", "w").write("\n".join(lines) + "\n")
    return found


def model_facts(entry):
    """Shape, memory, validation losses and speed of one model of the manifest."""
    r = json.load(open(os.path.join(os.path.dirname(entry["float"]), "result.json")))
    cfg = ModelConfig(**r["config"])
    facts = {"label": entry["label"], "cfg": cfg, "params": r["params"], "val_float": r["val_loss"], "val_float_bpb": r["val_bpb"],
             "val_float_body": r.get("val_body"), "steps": r["steps"], "data": r["args"]["data"]}
    if entry.get("int") and os.path.exists(os.path.join(entry["int"], "result.json")):
        q = json.load(open(os.path.join(entry["int"], "result.json")))
        facts.update(val_int=q["val_loss"], val_int_bpb=q["val_bpb"], val_calibrated=q["calibrated_val_loss"], bit_exact=q["bit_exact"],
                     val_int_body=q.get("val_body"), val_calibrated_body=q.get("calibrated_val_body"))
    meta_path = os.path.join(r["args"]["data"], "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    facts["validation"] = "self-contained functions (val2)" if meta.get("validation") == "val2" else "general functions"
    options = {"norm": "rms" if cfg.norm != "dyt" else "dyt", "ffn": cfg.ffn}
    plain = ModelConfig(**dict(cfg.to_dict(), attn="softmax", window=0))
    span = lambda n: (min(n, cfg.window) if cfg.window else n) - 1
    try:
        facts["cycles"] = [cost.token_cycles(plain, options, span(n), stop=True, syntax_filter=True)[0] for n in (16, 64, 256)]
        facts["speedup"] = [cost.classic(plain, options, span(n))[0] / cost.cycles_per_token(plain, options, span(n))[0] for n in (16, 64, 256)]
        b = cost.token_cycles(plain, options, 0, stop=True, syntax_filter=True)[2]
        facts["memory"], facts["words"] = len(b.image), len(b.code)
    except Exception as error:                      # the model does not fit the block RAM (Tier B): see docs/TIER_B.md
        facts["cycles"] = facts["speedup"] = None
        facts["memory"], facts["note"] = cfg.n_params() + cfg.kv_cache_bytes(), "does not fit the data memory: %s" % str(error)[:60]
    return facts


def summary(spec, facts, latest, strata_of):
    """docs/tables/summary.md: the headline numbers, each read from the same files as the tables."""
    f = facts[0]
    c = f["cfg"]
    lines = []
    gains = c.dim * (2 * c.n_layers + 1)
    lines.append("- **Model (Tier A).** %s trained parameters, all carried by int8 values (%s int8 matrix entries; the %d gains of the norms are folded "
                 "into matrices and tables): %d layers, width %d, %d query heads sharing %d key/value head%s, %s feed-forward of %d, "
                 "512 tokens, context %d%s. Its complete data memory image is %s bytes (%d of the 135 block RAMs, plus one for the program), the program %s words." % (
                     num(f["params"]), num(f["params"] - gains), gains, c.n_layers, c.dim, c.n_heads, c.n_kv_heads, "" if c.n_kv_heads == 1 else "s",
                     "ReLU" if c.ffn == "relu" else "SwiGLU", c.hidden_dim, c.max_seq_len, ", attention window %d" % c.window if c.window else "",
                     num(f["memory"]), -(-f["memory"] // 4096), num(f.get("words", 0))))
    if f["cycles"]:
        lines.append("- **Speed (level L1, 100 MHz, greedy with the syntax filter).** %s / %s / %s cycles per token = **%s / %s / %s tokens per second** at "
                     "context 16 / 64 / 256. The stream instruction makes the same program %s / %s / %s times faster than one classic 4-cycle `MAC` per product." % (
                         *[num(v) for v in f["cycles"]], *["%.0f" % (1e8 / v) for v in f["cycles"]], *["%.2f" % v for v in f["speedup"]]))
    if "val_int" in f:
        lines.append("- **Integer arithmetic.** Validation loss %.4f nats per token in float and %.4f with the integers that the CPU computes (%+.4f); "
                     "training forward pass, integer reference and generated LS program give identical tokens." % (f["val_float"], f["val_int"], f["val_int"] - f["val_float"]))

    def best(model, variant, bench, key="pass@1", edit="", sampled=False):
        rows = [r for k, r in latest.items() if k[0] == model and k[1] == variant and k[2] == edit and k[6] == bench and k[7] == "test"
                and k[4] == sampled and (k[5] >= 10) == sampled]
        return max((r["metrics"][key] for r in rows if r["metrics"][key] is not None), default=None)

    def params(model, variant):
        return next((num(r["params"]) for k, r in latest.items() if k[0] == model and k[1] == variant and r.get("params")), "-")

    def line(label, model, variant, edit=""):
        near, far = strata_of.get((model, variant, edit), ("-", "-"))
        return "| %s | %s | %s | %s | %s | %s |" % (
            label, params(model, variant), " | ".join(pct(best(model, variant, bench, edit=edit)) for bench, _ in BENCHES), near, far,
            " | ".join(pct(best(model, variant, bench, "pass@10", edit, True)) for bench, _ in BENCHES))

    lines += ["- **Scores on the test sets** (pass@1 greedy, best of filter on and off; pass@10 from 10 samples at temperature 0.8). Every model went",
              "  through the same harness; the comparators are public models run as published. Details and the ladder below pass@1: section 3.", "",
              "  | model | parameters | custom 200 | MBPP | HumanEval | custom: near families, tasks passed of 120 | custom: far families, of 80 | custom 200 pass@10 | MBPP pass@10 | HumanEval pass@10 |",
              "  |---|---|---|---|---|---|---|---|---|---|"]
    for entry in spec["models"]:
        float_name = os.path.basename(os.path.dirname(entry["float"]))
        lines.append("  " + line("**%s, float**" % entry["label"], float_name, "float"))
        if entry.get("int"):
            lines.append("  " + line("**%s, int8 (what the CPU computes)**" % entry["label"], os.path.basename(entry["int"]), "int8"))
    for name in spec.get("comparators", []):
        variant = next((k[1] for k in latest if k[0] == name), "")
        lines.append("  " + line(name.replace(" (as published, plain continuation)", "").replace(" (karpathy/tinyllamas, as published)", ""), name, variant))
    ours = [strata_of.get((os.path.basename(os.path.dirname(entry["float"])), "float", ""), (0, 0)) for entry in spec["models"]]
    ours += [strata_of.get((os.path.basename(entry["int"]), "int8", ""), (0, 0)) for entry in spec["models"] if entry.get("int")]
    far = max(pair[1] for pair in ours)
    lines += ["", "  Near families are kinds of task that the generated part of the training text also covers (other tasks, same kind); far families are not.",
              "  Our models pass %s. The two larger public models pass several far-family tasks." % (
                  "only near-family tasks" if far == 0 else "near-family tasks and at most %d far-family task%s" % (far, "" if far == 1 else "s")), ""]
    for entry in spec.get("board", []):
        if entry.get("verify") and os.path.exists(entry["verify"]):
            rows = json.load(open(entry["verify"]))
            ordered = sorted(row["cycles"] for row in rows)
            lines.append("- **Programs on request (board model, int8, on the simulated CPU).** Of %d requests such as \"make me a bubble sort program\", "
                         "%d come back as a program that passes the tests of its request; in all %d the CPU program sends exactly the tokens of the integer reference. "
                         "Median %.2f s per program at 100 MHz, longest %.2f s. This is recall of programs the board model was trained on, not a benchmark (section 3)."
                         % (len(rows), sum(row["outcome"] == "pass" for row in rows), sum(row["same_as_reference"] for row in rows),
                            ordered[len(ordered) // 2] / 1e8, ordered[-1] / 1e8))
    with_edits = next((entry for entry in spec["models"] if entry.get("edited") and entry.get("int")), None)
    edited = with_edits["edited"] if with_edits else []
    if edited:
        name = os.path.basename(with_edits["int"])
        got = [best(name, "int8", "custom", edit="session " + os.path.basename(folder)) for folder in edited]
        if all(v is not None for v in got):
            lines.append("- **Weight editing without gradients (%d sessions, on the %s).** Practice scores rose, the held-out custom score did not: %s unedited, "
                         "%s after the sessions. No edited model is used (section 10)." % (
                             len(edited), "final int8 model" if with_edits is spec["models"][0] else "earlier Tier A model", pct(best(name, "int8", "custom")),
                             ", ".join(pct(v) for v in got)))
    lines.append("- **What was not done.** Nothing ran on the board or in Vivado; cycle counts come from a simulator of the proposed CPU (section 13).")
    open("docs/tables/summary.md", "w").write("\n".join(lines) + "\n")


def main(manifest):
    spec = json.load(open(manifest))
    latest = results()
    os.makedirs("docs/tables", exist_ok=True)
    test, practice, facts = [], [], []
    for entry in spec["models"]:
        name = os.path.basename(os.path.dirname(entry["float"]))
        variants = [(name, "float", "")]
        if entry.get("int"):
            variants.append((os.path.basename(entry["int"]), "int8", ""))
        for session in entry.get("edited", []):
            variants.append((os.path.basename(entry["int"]), "int8", "session " + os.path.basename(session)))
        for model, variant, edit in variants:
            test += score_rows(latest, model, variant, edit, "test", entry["label"])
            practice += score_rows(latest, model, variant, edit, "practice", entry["label"])
        facts.append(model_facts(entry))
    for name in spec.get("comparators", []):
        variant = next((k[1] for k in latest if k[0] == name), "")
        test += score_rows(latest, name, variant, "", "test", name)
    open("docs/tables/scores_test.md", "w").write("\n".join(score_table(test)) + "\n")
    open("docs/tables/scores_practice.md", "w").write("\n".join(score_table(practice)) + "\n")
    header = ["model", "layers", "width", "heads / KV heads", "feed-forward", "window", "parameters", "data memory on the CPU (bytes)", "program (words)",
              "validation text", "val loss float, all tokens (nats/token)", "val loss int8, all tokens", "int8 after calibration only",
              "val loss on body tokens: float / int8", "bits per byte float / int8",
              "cycles per token @16 / @64 / @256", "tokens/s @16 / @64 / @256", "stream speed-up vs classic MAC @16 / @64 / @256"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for f in facts:
        c = f["cfg"]
        cyc = " / ".join(num(v) for v in f["cycles"]) if f["cycles"] else "see docs/TIER_B.md"
        tps = " / ".join("%.1f" % (1e8 / v) for v in f["cycles"]) if f["cycles"] else "see docs/TIER_B.md"
        spd = " / ".join("%.2f" % v for v in f["speedup"]) if f["speedup"] else "-"
        lines.append("| " + " | ".join(str(v) for v in [
            f["label"], c.n_layers, c.dim, "%d / %d" % (c.n_heads, c.n_kv_heads), "%s, %d" % (c.ffn, c.hidden_dim), c.window or "-", num(f["params"]),
            num(f["memory"]), num(f["words"]) if "words" in f else "-", f["validation"], "%.4f" % f["val_float"],
            "%.4f" % f["val_int"] if "val_int" in f else "-", "%.4f" % f["val_calibrated"] if "val_calibrated" in f else "-",
            "%s / %s" % ("%.4f" % f["val_float_body"] if f.get("val_float_body") else "-", "%.4f" % f["val_int_body"] if f.get("val_int_body") else "-"),
            "%.4f / %s" % (f["val_float_bpb"], "%.4f" % f["val_int_bpb"] if "val_int_bpb" in f else "-"), cyc, tps, spd]) + " |")
    open("docs/tables/models.md", "w").write("\n".join(lines) + "\n")
    f = facts[0]
    if f["cycles"]:
        c = f["cfg"]
        open("docs/tables/systems_ours.md", "w").write(
            "| **this project (simulated, not on hardware)** | LS CPU (level L1) on an Artix-7, 100 MHz, one multiply-add per clock | own model, %s parameters, "
            "%d layers, width %d | int8 weights and activations, 24-bit sums, shifts and tables only | %s at context 16, %s at 64, %s at 256 |\n" % (
                num(f["params"]), c.n_layers, c.dim, *["%.0f" % (1e8 / v) for v in f["cycles"]]))
    wanted = {(name, next((k[1] for k in latest if k[0] == name), "")) for name in spec.get("comparators", [])}
    for entry in spec["models"]:
        wanted.add((os.path.basename(os.path.dirname(entry["float"])), "float"))
        if entry.get("int"):
            wanted.add((os.path.basename(entry["int"]), "int8"))
    summary(spec, facts, latest, strata(latest, wanted))
    print("\n".join(lines))
    print("\n".join(score_table(test)[:6]))


if __name__ == "__main__":
    main(sys.argv[1])
