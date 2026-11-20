"""Shared helpers of the session scripts (no model runs here)."""
import json
import re
from collections import Counter

from lsllm.tok import Tok

DIR = "edits/s1"


def load(run):
    """(tasks, samples of eval_<run>.jsonl)."""
    tasks = [json.loads(line) for line in open("data/bench/custom_practice.jsonl")]
    rows = [json.loads(line) for line in open("%s/eval_%02d.jsonl" % (DIR, run))]
    return tasks, rows


def credit(row):
    if row["outcome"] == "pass":
        return 1.0
    if row["outcome"] == "wrong":
        failed, total = map(int, re.match(r"(\d+) of (\d+)", row["detail"]).groups())
        return 1 - failed / total
    return 0.0


def repeats(text):
    """The 'repeating' rule of Session.evaluate."""
    words = text.split()
    grams = Counter(tuple(words[g:g + 3]) for g in range(len(words) - 2))
    return bool(grams) and max(grams.values()) >= 3


def encode(tok, prompt, completion):
    """(ids, number of prompt ids) of prompt + completion. A completion that was cut inside a line cannot be
    marked as a whole: its complete lines are marked with a stand-in last line `x`, then the real last line
    replaces the stand-in. The ids decode to the exact text."""
    from lsllm.structure import to_markers
    n = len(tok.encode(prompt))
    text = prompt + completion
    try:
        return tok.encode(text), n
    except ValueError:
        head, last = text.rsplit("\n", 1)
        indent = last[:len(last) - len(last.lstrip())]
        marked = to_markers(head + "\n" + indent + "x")
        assert marked.endswith("x")
        ids = [tok.bos_id] + tok.sp.encode(marked[:-1] + last.lstrip())
        assert tok.decode(ids) == text, text
        return ids, n


def tokenizer(session):
    return Tok(session.spec["tokenizer"] + "/tok.model")


def report(result, base_run=1):
    """One line of scores of an evaluate result and the tasks whose credit changed against eval_<base_run>."""
    tasks, old = load(base_run)
    _, new = load(result["run"])
    up = [(t["task_id"][-3:], round(credit(a), 2), round(credit(b), 2)) for t, a, b in zip(tasks, old, new) if credit(b) > credit(a)]
    down = [(t["task_id"][-3:], round(credit(a), 2), round(credit(b), 2)) for t, a, b in zip(tasks, old, new) if credit(b) < credit(a)]
    changed = sum(a["completion"] != b["completion"] for a, b in zip(old, new))
    print("run %d  pass@1 %.3f  tests_passed %.4f  syntax_ok %.3f  runs_ok %.3f  loss %.4f  repeating %.3f  errors %s" % (
        result["run"], result["pass@1"], result["tests_passed"], result["syntax_ok"], result["runs_ok"],
        result["practice_loss"], result["repeating"], result["errors"]))
    print("   against run %d: %d answers changed; credit up %s; credit down %s" % (base_run, changed, up, down))


def restore(session, saved, why):
    """Put back the arrays `saved` (name -> old array): what Session.undo does, for an edit of several tensors."""
    for name, old in saved.items():
        session.put(name, old, why)
