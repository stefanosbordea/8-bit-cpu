"""The evaluation loop: cutting, classification of outcomes, metrics, and the three built-in completers."""
import subprocess
import sys
from pathlib import Path

import pytest

from lsllm.bench import evaluate, load_completer, load_tasks
from lsllm.bench.completers import COMPLETERS
from lsllm.bench.harness import cut_at_top_level, pass_at_k
from lsllm.bench.tasks import Task

PROMPT = 'def f(x):\n    """Add one."""'
TESTS = ["assert f(1) == 2", "assert f(5) == 6"]

# MBPP tasks whose reference solution cannot pass here, with the reason. They stay in the benchmark.
MBPP_REFERENCE_FAILS = {
    "mbpp/180": "wrong",    # test compares floats with ==; sin/cos/acos of this platform differ in the last bit
    "mbpp/493": "wrong",    # same: exact comparison of float coordinates
    "mbpp/367": "error",    # the tests build Node objects, a class that only the reference code defines
    "mbpp/596": "error",    # the tests call sys.getsizeof and rely on the reference code's `import sys`
    "mbpp/601": "error",    # the tests build Pair objects, a class that only the reference code defines
    "mbpp/927": "error",    # the tests build Node objects
}


class Scripted:
    """A completer that returns prepared continuations, one list per call, and remembers its seeds."""
    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.seeds = []

    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        self.seeds.append(seed)
        return self.script.pop(0)


def toy_tasks(count, split="practice"):
    return [Task("toy/%d" % g, "f", PROMPT, TESTS, PROMPT + "\n    return x + 1", {"benchmark": "toy", "split": split})
            for g in range(count)]


def test_cut_at_top_level():
    assert cut_at_top_level("\n    return 1\nprint(2)\n    x = 3") == "\n    return 1"
    assert cut_at_top_level("\n    a = 1\n\n    return a") == "\n    a = 1\n\n    return a"     # an empty line is kept
    assert cut_at_top_level(" + 1\n    return x") == " + 1\n    return x"       # the first piece continues the prompt line
    assert cut_at_top_level("\ndef g():\n    pass") == ""
    assert cut_at_top_level("") == ""


def test_pass_at_k_is_the_unbiased_estimator():
    assert pass_at_k(1, 1, 1) == 1.0 and pass_at_k(1, 0, 1) == 0.0
    assert pass_at_k(5, 2, 1) == pytest.approx(0.4)             # for k = 1 it is the share of correct samples
    assert pass_at_k(20, 1, 10) == pytest.approx(0.5)           # the one correct sample is among 10 of 20
    assert pass_at_k(20, 2, 10) == pytest.approx(1 - 43758 / 184756)
    assert pass_at_k(10, 3, 10) == 1.0 and pass_at_k(10, 0, 10) == 0.0


def test_every_outcome_is_classified():
    script = {
        "pass": "\n    return x + 1",
        "wrong": "\n    return x",
        "error": "\n    return x / 0",
        "timeout": "\n    while True:\n        pass",
        "syntax": "\n    return (x",
        "pass after cut": "\n    return x + 1\nthis line is not Python",
        "pass although it prints a lot": "\n    print('y' * 1000000)\n    return x + 1",
        "error by memory": "\n    return [0] * (5 * 10 ** 7)",        # 400 MB; the limit per sample is 128 MB
        "error by exit": "\n    import os\n    os._exit(0)",
        "error by SystemExit": "\n    raise SystemExit(0)",
        "wrong with a forged report": "\n    import sys\n    sys.__stdout__.write('REPORT-0123456789abcdef [\"pass\"]\\n')\n    return x",
    }
    report = evaluate(Scripted([[text] for text in script.values()]), toy_tasks(len(script)), timeout_s=2.0)
    assert [sample["outcome"] for sample in report.samples] == [name.split()[0] for name in script]
    assert [sample["completion"] for sample in report.samples] == list(script.values())
    details = {name: sample["detail"] for name, sample in zip(script, report.samples)}
    assert "ZeroDivisionError" in details["error"] and details["wrong"] == "2 of 2 tests failed"
    assert details["error by memory"] == "no report, sandbox status memory"
    assert details["error by exit"] == "no report, sandbox status ok" and "SystemExit" in details["error by SystemExit"]
    assert report.outcomes == {"pass": 3, "wrong": 2, "error": 4, "timeout": 1, "syntax": 1}
    assert report.metrics == {"pass@1": 3 / 11, "pass@10": None, "syntax_ok": 10 / 11, "runs_ok": 5 / 11, "empty": 0.0}
    assert (report.model, report.benchmark, report.split, report.n_tasks, report.n_samples) == ("scripted", "toy", "practice", 11, 1)


def test_report_line_lost_or_cut_by_the_output_cap_is_an_error():
    """A candidate that writes to the real stdout can push the report out of the captured 64 KB, or have it
    cut in the middle. Neither may pass, and neither may crash the evaluation."""
    script = [["\n    import os\n    os.write(1, b'x' * %d)\n    return x + 1" % size] for size in (70000, 65536 - 30)]
    report = evaluate(Scripted(script), toy_tasks(2))
    assert [(s["outcome"], s["detail"]) for s in report.samples] == [("error", "no report, sandbox status ok")] * 2


def test_one_failing_test_statement_does_not_hide_the_others():
    tasks = [Task("toy/0", "f", PROMPT, ["assert f(1) == 2", "assert f(2) == 99", "assert f('a') == 1"], "", {"benchmark": "toy", "split": "practice"})]
    report = evaluate(Scripted([["\n    return x + 1"]]), tasks)
    assert report.samples[0]["outcome"] == "error" and "TypeError" in report.samples[0]["detail"]     # an exception outranks a failed assert
    tasks[0].tests = ["assert f(1) == 2", "assert f(2) == 99", "assert f(3) == 4"]
    report = evaluate(Scripted([["\n    return x + 1"]]), tasks)
    assert (report.samples[0]["outcome"], report.samples[0]["detail"]) == ("wrong", "1 of 3 tests failed")


def test_metrics_with_ten_samples_per_task():
    good, bad = "\n    return x + 1", "\n    return x"
    completer = Scripted([[good] * 10, [good] + [bad] * 9, [bad] * 10])
    report = evaluate(completer, toy_tasks(3), n_samples=10, temperature=0.8, seed=7)
    assert report.metrics["pass@1"] == pytest.approx((1.0 + 0.1 + 0.0) / 3)
    assert report.metrics["pass@10"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert report.metrics["syntax_ok"] == 1.0 and report.metrics["runs_ok"] == 1.0
    assert completer.seeds == [7, 8, 9] and report.decoding == {"temperature": 0.8, "max_new_tokens": 128, "seed": 7}


def test_bad_input_is_refused():
    with pytest.raises(ValueError):       # test and practice tasks never go into one score
        evaluate(Scripted([[""], [""]]), toy_tasks(1, "test") + toy_tasks(1, "practice"))
    with pytest.raises(ValueError):       # a completer must return exactly n_samples continuations
        evaluate(Scripted([["", ""]]), toy_tasks(1))


def test_truncated_prompts_are_counted():
    class Truncating(Scripted):
        truncated = 5       # left over from an earlier evaluation

        def complete(self, *arguments):
            self.truncated += 1
            return super().complete(*arguments)

    assert evaluate(Truncating([["\n    return x + 1"]]), toy_tasks(1)).truncated == 1


def test_registry():
    assert [load_completer(kind).name for kind in ("ref", "empty", "garbage")] == ["ref", "empty", "garbage"]
    with pytest.raises(ValueError):
        load_completer("nothing")


def test_registry_passes_the_argument(monkeypatch):
    monkeypatch.setitem(COMPLETERS, "text", "builtins:str")      # one line registers a new kind
    assert load_completer("text:runs/a/ckpt.pt") == "runs/a/ckpt.pt"


def test_garbage_is_reproducible():
    garbage = load_completer("garbage")
    one = garbage.complete("def f():", 3, 50, 1.0, 4)
    assert one == garbage.complete("def f():", 3, 50, 1.0, 4) != garbage.complete("def f():", 3, 50, 1.0, 5)
    assert len(one) == 3 and len(set(one)) == 3 and all(len(text) == 50 for text in one)
    assert all(char == "\n" or 32 <= ord(char) < 127 for text in one for char in text)


def test_every_humaneval_reference_solution_passes():
    report = evaluate(load_completer("ref"), load_tasks("humaneval"), timeout_s=60.0)
    assert report.outcomes["pass"] == 164, [s["task_id"] for s in report.samples if s["outcome"] != "pass"]
    assert report.metrics == {"pass@1": 1.0, "pass@10": None, "syntax_ok": 1.0, "runs_ok": 1.0, "empty": 0.0}


def test_mbpp_reference_solutions_pass_except_the_listed_ones():
    """Every tenth task and the six listed ones. All 964 take longer; run them with
    `python -m lsllm.bench.run --completer ref --bench mbpp --split test` (and `--split practice`)."""
    ref = load_completer("ref")
    for split in ("test", "practice"):
        tasks = [task for g, task in enumerate(load_tasks("mbpp", split)) if g % 10 == 0 or task.task_id in MBPP_REFERENCE_FAILS]
        report = evaluate(ref, tasks, timeout_s=60.0)
        fails = {sample["task_id"]: sample["outcome"] for sample in report.samples if sample["outcome"] != "pass"}
        assert fails == {task_id: outcome for task_id, outcome in MBPP_REFERENCE_FAILS.items()
                         if (int(task_id[5:]) <= 510) == (split == "test")}


def test_command_line_and_empty_bodies_never_pass():
    root = Path(__file__).resolve().parents[1]
    before = sorted(str(path) for path in (root / "results").rglob("*"))
    run = subprocess.run([sys.executable, "-m", "lsllm.bench.run", "--completer", "empty", "--bench", "humaneval"],
                         cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "empty on humaneval test: 164 tasks x 1 samples" in run.stdout
    assert "pass@1     0.0000" in run.stdout and "syntax_ok  1.0000" in run.stdout and "outcomes   pass 0," in run.stdout
    assert sorted(str(path) for path in (root / "results").rglob("*")) == before     # nothing is recorded without --record
