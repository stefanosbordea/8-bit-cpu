"""Benchmark harness: tasks (HumanEval, MBPP, custom), evaluation in the sandbox, results table."""
from lsllm.bench.tasks import Task, load_tasks
from lsllm.bench.harness import Report, evaluate
from lsllm.bench.completers import load_completer
