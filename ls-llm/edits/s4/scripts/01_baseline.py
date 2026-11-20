"""Run 1: the baseline score of the untouched model."""
import json
from lsllm.edit.session import Session
s = Session("edits/s4")
print(json.dumps(s.evaluate("baseline"), indent=1))
