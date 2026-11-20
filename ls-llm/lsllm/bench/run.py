"""Command line: score one completer on one benchmark.

.venv/bin/python -m lsllm.bench.run --completer ref --bench humaneval
"""
import argparse
import time

from lsllm.bench import evaluate, load_completer, load_tasks
from lsllm.bench.results import record


def main():
    parser = argparse.ArgumentParser(description="Score a completer on a benchmark; generated code runs in the sandbox.")
    parser.add_argument("--completer", required=True, help='"ref", "empty", "garbage" or "<kind>:<argument>"')
    parser.add_argument("--bench", required=True, choices=["humaneval", "mbpp", "custom"])
    parser.add_argument("--split", default="test", choices=["test", "practice"])
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--record", action="store_true", help="append the result to results/results.jsonl")
    args = parser.parse_args()
    completer = load_completer(args.completer)
    tasks = load_tasks(args.bench, args.split)
    start = time.monotonic()
    report = evaluate(completer, tasks, args.n_samples, args.temperature, args.max_new_tokens)
    print("%s on %s %s: %d tasks x %d samples in %.1f s" % (report.model, report.benchmark, report.split,
                                                          report.n_tasks, report.n_samples, time.monotonic() - start))
    for name, value in report.metrics.items():
        print("  %-10s %s" % (name, "-" if value is None else "%.4f" % value))
    print("  outcomes   " + ", ".join("%s %d" % item for item in report.outcomes.items()))
    if report.truncated:
        print("  %d prompts were truncated by the completer" % report.truncated)
    if args.record:
        # a completer may describe itself in `info`: any of params, variant, filter, edit, notes
        row = record(report, **getattr(completer, "info", {}))
        print("  recorded in results/results.jsonl, samples in results/" + row["samples"])


if __name__ == "__main__":
    main()
