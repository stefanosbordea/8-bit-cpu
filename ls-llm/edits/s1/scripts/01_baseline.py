"""Run 1: the baseline scores. Run from the repository root: PYTHONPATH=. .venv/bin/python edits/s1/scripts/01_baseline.py"""
import json
from lsllm.edit.session import Session
print(json.dumps(Session("edits/s1").evaluate("baseline"), indent=1))
