"""The custom benchmark: very simple Python tasks with unit tests, in a test split and a practice split.

Task sources: tasks_<split>/*.py (format in build.py). Built files: data/bench/custom_<split>.jsonl.
Families and their stratum (near or far): data/bench/custom_families.json. A function that fits a far family
counts as far, even when the wording of a near family fits it too.
Build: .venv/bin/python -m lsllm.bench.custom.build        Check: .venv/bin/python -m lsllm.bench.custom.check
"""
