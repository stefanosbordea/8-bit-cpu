"""results.jsonl gets one line per evaluation; RESULTS.md keeps test and practice scores apart."""
import gzip
import json

from lsllm.bench.harness import Report
from lsllm.bench.results import record, render


def report(model, split, pass_at_1):
    return Report(model=model, benchmark="mbpp", split=split, n_tasks=2, n_samples=1,
                  decoding={"temperature": 0.0, "max_new_tokens": 128, "seed": 0},
                  metrics={"pass@1": pass_at_1, "pass@10": None, "syntax_ok": 1.0, "runs_ok": 0.5},
                  outcomes={"pass": 1, "wrong": 0, "error": 1, "timeout": 0, "syntax": 0}, truncated=3,
                  samples=[{"task_id": "mbpp/11", "outcome": "pass", "detail": "", "completion": "\n    return 1"},
                           {"task_id": "mbpp/12", "outcome": "error", "detail": "error: NameError()", "completion": "\n    return y"}])


def test_record_appends_a_line_and_keeps_the_samples(tmp_path):
    row = record(report("model/a:1", "test", 0.5), params=260000, variant="int8", filter="grammar", edit="none",
                 notes="first | try", folder=tmp_path)
    record(report("model-b", "practice", 0.25), folder=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0] == row
    for name in ("date", "model", "params", "variant", "decoding", "filter", "edit", "benchmark", "split", "n_tasks",
                 "n_samples", "metrics", "notes"):
        assert name in rows[0] and name in rows[1]
    assert (rows[0]["model"], rows[0]["params"], rows[0]["split"], rows[0]["metrics"]["pass@1"]) == ("model/a:1", 260000, "test", 0.5)
    assert (rows[1]["params"], rows[1]["variant"], rows[1]["split"]) == (None, "", "practice")
    with gzip.open(tmp_path / rows[0]["samples"], "rt") as f:
        samples = [json.loads(line) for line in f]
    assert samples == report("x", "test", 0).samples


def test_render_separates_test_and_practice(tmp_path):
    render(tmp_path)
    assert (tmp_path / "RESULTS.md").read_text().count("No results yet.") == 2
    record(report("model-a", "test", 0.5), params=260000, variant="int8", notes="first | try", folder=tmp_path)
    record(report("model-b", "practice", 0.25), folder=tmp_path)
    text = (tmp_path / "RESULTS.md").read_text()
    top, bottom = text.split("## Practice-set scores (NOT benchmark scores")
    assert "## Benchmark scores (test splits)" in top
    assert "model-a" in top and "model-b" not in top and "50.00%" in top and "260000" in top
    assert "model-b" in bottom and "model-a" not in bottom and "25.00%" in bottom
    assert "3 prompts truncated. first / try" in top        # a "|" in a note cannot break the table
    assert all(line.count("|") == 17 for line in text.splitlines() if line.startswith("|"))
