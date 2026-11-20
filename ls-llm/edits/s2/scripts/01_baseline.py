"""Run 1: the baseline scores of the untouched model."""
import json
from lsllm.edit.session import Session
print(json.dumps(Session("edits/s2").evaluate("baseline"), indent=1))
