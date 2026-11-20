"""Score a completer on benchmark tasks. Generated code is only parsed here; it runs only in lsllm.sandbox."""
import ast
import json
import math
import secrets
import warnings
from collections import Counter
from dataclasses import dataclass

from lsllm.sandbox import run_many, run_python

OUTCOMES = ("pass", "wrong", "error", "timeout", "syntax")
MEMORY_MB = 128      # per sample; four run at a time (the largest reference solution needs 11 MB)

# Runs inside the sandbox, after three lines that define PROGRAM, TESTS and MARK. It defines the candidate,
# runs every test statement on its own and reports one line. What the candidate prints is thrown away.
DRIVER = '''
import json, sys
report = sys.stdout
class Sink:
    def write(self, text):
        return len(text)
    def flush(self):
        pass
sys.stdout = sys.stderr = Sink()
space = {"__name__": "__main__"}
try:
    exec(PROGRAM, space)
    steps = []
    for test in TESTS:
        try:
            exec(test, space)
            steps.append("pass")
        except AssertionError:
            steps.append("wrong")
        except BaseException as problem:
            steps.append("error: " + repr(problem)[:200])
except BaseException as problem:
    steps = ["error: " + repr(problem)[:200]]
report.write(MARK + json.dumps(steps) + "\\n")
report.flush()
'''


@dataclass
class Report:
    model: str         # completer.name
    benchmark: str
    split: str         # "test" or "practice", taken from the tasks, so a practice run cannot be mislabelled
    n_tasks: int
    n_samples: int     # per task
    decoding: dict     # temperature, max_new_tokens, seed
    metrics: dict      # pass@1, pass@10 (None below 10 samples), syntax_ok, runs_ok
    outcomes: dict     # number of samples per outcome
    truncated: int     # prompts the completer had to shorten (0 if it does not count them)
    samples: list      # one dict per sample: task_id, outcome, detail, completion


def cut_at_top_level(continuation):
    """Drop everything from the first line that starts in column 0: there the function has ended."""
    lines = continuation.split("\n")
    for g in range(1, len(lines)):          # lines[0] continues the last line of the prompt
        if lines[g] and not lines[g][0].isspace():
            return "\n".join(lines[:g])
    return continuation


def pass_at_k(n, c, k):
    """Unbiased pass@k of Chen et al. 2021 for one task: the chance that at least one of k samples, drawn
    without replacement from n samples of which c are correct, is correct."""
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _parses(source):
    """Syntax check. Parsing builds a tree; it does not execute anything."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(source)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def _outcome(result, mark):
    """Turn the sandbox result of one driver run into (outcome, detail)."""
    if result.status == "timeout":
        return "timeout", ""
    # complete lines only: the captured output is cut at 64 KB, which can cut the report line itself
    lines = [line for line in result.stdout.split("\n")[:-1] if line.startswith(mark)]
    if not lines:                           # killed, crashed, exited early, or wrote 64 KB before the report
        return "error", "no report, sandbox status " + result.status
    steps = json.loads(lines[-1][len(mark):])
    errors = [step for step in steps if step.startswith("error")]
    if errors:
        return "error", errors[0]
    if "wrong" in steps:
        return "wrong", "%d of %d tests failed" % (steps.count("wrong"), len(steps))
    return "pass", ""


def evaluate(completer, tasks, n_samples=1, temperature=0.0, max_new_tokens=128, seed=0, timeout_s=5.0, workers=4):
    """Let `completer` continue the prompt of every task `n_samples` times and test the programs in the sandbox.

    One program = prompt + continuation (cut at the first line in column 0), followed by the task's tests.
    Outcome of a sample: "syntax" (does not parse), "timeout", "error" (an exception anywhere, or the run was
    killed), "wrong" (it ran, an assert failed), "pass". Task number g is sampled with seed + g.
    """
    labels = {(task.meta.get("benchmark"), task.meta.get("split")) for task in tasks}
    if len(labels) != 1:
        raise ValueError("the tasks must come from one benchmark and one split")
    benchmark, split = labels.pop()
    check = run_python("print('sandbox works')")
    if check.stdout != "sandbox works\n":
        raise RuntimeError("the sandbox does not work on this machine: " + check.stderr)
    mark = "REPORT-" + secrets.token_hex(8) + " "      # random, so a candidate cannot print a fake report
    truncated = getattr(completer, "truncated", 0)
    samples, drivers = [], []
    for g, task in enumerate(tasks):
        texts = completer.complete(task.prompt, n_samples, max_new_tokens, temperature, seed + g)
        if len(texts) != n_samples:
            raise ValueError("%s returned %d continuations instead of %d" % (completer.name, len(texts), n_samples))
        for text in texts:
            program = task.prompt + cut_at_top_level(text)
            parses = _parses(program)
            if parses:
                drivers.append("PROGRAM = %r\nTESTS = %r\nMARK = %r\n" % (program, task.tests, mark) + DRIVER)
            samples.append({"task_id": task.task_id, "outcome": None if parses else "syntax", "detail": "",
                            "completion": text, "empty": parses and not cut_at_top_level(text).strip()})
    results = iter(run_many(drivers, workers, timeout_s, MEMORY_MB))
    for sample in samples:
        if sample["outcome"] is None:
            sample["outcome"], sample["detail"] = _outcome(next(results), mark)
    passed = Counter(sample["task_id"] for sample in samples if sample["outcome"] == "pass")
    count = Counter(sample["outcome"] for sample in samples)

    def mean_pass_at(k):
        return sum(pass_at_k(n_samples, passed[task.task_id], k) for task in tasks) / len(tasks)

    metrics = {"pass@1": mean_pass_at(1), "pass@10": mean_pass_at(10) if n_samples >= 10 else None,
               "syntax_ok": 1 - count["syntax"] / len(samples),
               "runs_ok": (count["pass"] + count["wrong"]) / len(samples),
               # a completer that adds nothing leaves "def + docstring", which parses and runs: such samples are
               # counted here so that syntax_ok and runs_ok can be read honestly
               "empty": sum(sample["empty"] for sample in samples) / len(samples)}
    return Report(model=completer.name, benchmark=benchmark, split=split, n_tasks=len(tasks), n_samples=n_samples,
                  decoding={"temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed},
                  metrics=metrics, outcomes={name: count[name] for name in OUTCOMES},
                  truncated=getattr(completer, "truncated", 0) - truncated, samples=samples)
