"""Shared by the scripts of session s3: the session, the tokenizer, the practice tasks."""
import json
import sys

sys.path.insert(0, ".")
from lsllm.edit.session import Session
from lsllm.tok import Tok

DIR = "edits/s3"


def open_session():
    s = Session(DIR)
    return s, Tok(s.spec["tokenizer"] + "/tok.model")


def practice():
    with open("data/bench/custom_practice.jsonl") as f:
        return [json.loads(line) for line in f]


def completions(run):
    with open("%s/eval_%02d.jsonl" % (DIR, run)) as f:
        return [json.loads(line) for line in f]


def show(result):
    print(json.dumps({k: result[k] for k in ("run", "note", "pass@1", "tests_passed", "syntax_ok", "runs_ok",
                                             "practice_loss", "errors", "repeating", "budget_left")}))


import numpy as np


def exp_table(kappa, floor=1):
    """One row of EXP after SPEC C.6: [0] = 127, [256 - d] = round(127 exp(-d / kappa)) for d = 1..127, else 0.
    Entries below `floor` are set to 0."""
    row = np.zeros(256, dtype=np.int64)
    row[0] = 127
    for d in range(1, 128):
        value = int(np.floor(127 * np.exp(-d / kappa) + 0.5))
        row[256 - d] = value if value >= floor else 0
    return row


def trial(s, edits, note, keep=False):
    """Apply `edits` (a list of ("array name", new values, reason) or (("preloads", ...), number, reason)),
    spend one evaluate, and take the edits back unless keep. Returns the scores."""
    before = []
    for target, value, reason in edits:
        if isinstance(target, tuple):
            node = s.spec
            for key in target:
                node = node[key]
            before.append((target, node))
            s.put_number(target, value, reason)
        else:
            before.append((target, s.get(target)))
            s.put(target, value, reason)
    result = s.evaluate(note)
    show(result)
    if not keep:
        for target, old in reversed(before):
            if isinstance(target, tuple):
                s.put_number(target, old, "take back: " + note)
            else:
                s.put(target, old, "take back: " + note)
    return result
