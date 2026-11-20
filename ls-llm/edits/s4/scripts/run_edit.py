"""Helper used by the numbered edit scripts: apply edits, evaluate once (one budgeted run), print the scores and
a comparison with the baseline answers. No model run of its own beyond Session.evaluate."""
import json
import os
import sys
from lsllm.edit.session import Session

D = "edits/s4/"


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def report(s, note):
    r = s.evaluate(note)
    tasks = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
    new = [json.loads(l) for l in open(D + "eval_%02d.jsonl" % r["run"])]
    old = [json.loads(l) for l in open(D + "eval_01.jsonl")]
    refs = [t["canonical_solution"][len(t["prompt"]):] for t in tasks]
    exact = sum(n["completion"] == ref for n, ref in zip(new, refs))
    prefix = sum(common_prefix(n["completion"], ref) / len(ref) for n, ref in zip(new, refs)) / len(refs)
    changed = sum(n["completion"] != o["completion"] for n, o in zip(new, old))
    print(json.dumps({k: r[k] for k in ("run", "budget_left", "practice_loss", "pass@1", "syntax_ok", "runs_ok", "tests_passed", "errors", "repeating")}))
    print("exact = reference: %d; mean share of the reference reproduced as a prefix: %.4f; answers changed vs baseline: %d" % (exact, prefix, changed))
    for n, (a, o) in enumerate(zip(new, old)):
        if (a["outcome"] == "pass") != (o["outcome"] == "pass"):
            print("  task %d: %s -> %s | %r" % (n, o["outcome"], a["outcome"], a["completion"]))
    passed = [n for n, a in enumerate(new) if a["outcome"] == "pass"]
    print("passes:", passed)
    return r
