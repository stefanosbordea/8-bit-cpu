"""The record of every reported score: results/results.jsonl, rendered as results/RESULTS.md."""
import gzip
import json
import re
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
COLUMNS = ("date", "model", "params", "variant", "decoding", "filter", "edit", "benchmark", "tasks", "samples",
           "pass@1", "pass@10", "syntax_ok", "runs_ok", "empty", "notes")


def record(report, params=None, variant="", filter="", edit="", notes="", folder=RESULTS):
    """Append one evaluation to results.jsonl, keep its samples for inspection and render RESULTS.md again.

    `report` comes from lsllm.bench.evaluate. The other arguments describe the model: number of parameters,
    variant (e.g. "float", "int8"), decoding filter and weight edit. Returns the line that was written.
    """
    now = time.localtime()
    name = "%s_%s_%s_%s" % (time.strftime("%Y%m%d-%H%M%S", now), report.model, report.benchmark, report.split)
    samples = Path("samples") / (re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".jsonl.gz")
    (folder / "samples").mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / samples, "wt", encoding="utf-8") as f:
        for sample in report.samples:
            f.write(json.dumps(sample) + "\n")
    row = {"date": time.strftime("%Y-%m-%d %H:%M", now), "model": report.model, "params": params, "variant": variant,
           "decoding": report.decoding, "filter": filter, "edit": edit, "benchmark": report.benchmark,
           "split": report.split, "n_tasks": report.n_tasks, "n_samples": report.n_samples,
           "metrics": report.metrics, "notes": notes, "outcomes": report.outcomes, "truncated": report.truncated,
           "samples": str(samples)}
    with open(folder / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    render(folder)
    return row


def _percent(value):
    return "-" if value is None else "%.2f%%" % (100 * value)


def _table(rows):
    if not rows:
        return "No results yet.\n"
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        decoding, metrics = row["decoding"], row["metrics"]
        notes = row["notes"]
        if row.get("truncated"):
            notes = ("%d prompts truncated. " % row["truncated"] + notes).strip()
        cells = [row["date"], row["model"], row["params"] or "", row["variant"],
                 "T=%s, %s tokens, seed %s" % (decoding["temperature"], decoding["max_new_tokens"], decoding["seed"]),
                 row["filter"], row["edit"], row["benchmark"], row["n_tasks"], row["n_samples"],
                 _percent(metrics["pass@1"]), _percent(metrics["pass@10"]), _percent(metrics["syntax_ok"]),
                 _percent(metrics["runs_ok"]), _percent(metrics.get("empty")), notes]
        lines.append("| " + " | ".join(str(cell).replace("|", "/") for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def render(folder=RESULTS):
    """Write RESULTS.md from results.jsonl. Practice-split rows never share a table with test-split rows."""
    path = folder / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    text = ("# RESULTS\n\nRendered from `results/results.jsonl` by `lsllm.bench.results.render()`. Do not edit by hand.\n"
            "pass@1 and pass@10 are per task (pass@10 is the unbiased estimator of Chen et al. 2021 and needs at\n"
            "least 10 samples per task); syntax_ok and runs_ok are per sample. The column empty is the share of samples where\n"
            "the model added nothing: such a sample is only the prompt (def + docstring), which parses and runs, so\n"
            "syntax_ok and runs_ok of a model with a high empty share say little.\n\n"
            "## Benchmark scores (test splits)\n\n" + _table([row for row in rows if row["split"] == "test"]) +
            "\n## Practice-set scores (NOT benchmark scores: these tasks are used for tuning)\n\n" +
            _table([row for row in rows if row["split"] != "test"]))
    (folder / "RESULTS.md").write_text(text, encoding="utf-8")
