"""One budgeted trace: a finer yardstick than pass@1. Greedy answers are chaotic (any edit changes most of them), so
pass@1 moves by +-2 tasks for no reason. Here every third practice task is fed with its REFERENCE solution
(practice data, custom_practice.jsonl) and the 7-bit logits are read at the tokens of the solution body:
    nll      mean of -log softmax(u / 2^f_u) of the reference token
    top1     share of body tokens that are the greedy choice (no syntax filter)
    solved   tasks whose whole body is the greedy choice
The same tokens for every model, so two models are compared token by token (paired), with a standard error over tasks.
Optional edits on top of the model in the session directory, taken back after the trace:
    --remove Wo_0:04:l0_xa:1     remove a direction (as script 05)      --head 1:1:1.25     scale a head (as script 08)
Run from the repository root, e.g.:
PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/10_trace_references.py --remove Wo_0:04:l0_xa:1"""
import argparse
import json

import numpy as np

from common import DIR, restore, tokenizer
from lsllm.edit.session import Session

ap = argparse.ArgumentParser()
ap.add_argument("--remove", nargs="*", default=[])
ap.add_argument("--head", nargs="*", default=[])
ap.add_argument("--kappa", nargs="*", default=[], help="layer:kappa (as script 09)")
a = ap.parse_args()

s = Session(DIR)
tok = tokenizer(s)
tasks = [json.loads(line) for line in open("data/bench/custom_practice.jsonl")]
seqs, total = [], 0
for g in range(0, 200, 3):
    ids, n = tok.encode(tasks[g]["canonical_solution"]), len(tok.encode(tasks[g]["prompt"]))
    if total + len(ids) <= 4096:
        total += len(ids)
        seqs.append((g, ids, n))

note = "reference solutions of %d tasks, %d tokens; edits: %s" % (
    len(seqs), total, " ".join(a.remove + a.head + a.kappa) or "none")
new, saved = {}, {}


def array(name):
    if name not in new:
        new[name] = s.get(name)
        saved[name] = new[name].copy()
    return new[name]


for item in a.remove:
    tensor, which, name, alpha = item.split(":")
    u = np.load("%s/directions_%s.npz" % (DIR, which))[name]
    W = array(tensor)
    W[:] = np.clip(np.round(W - float(alpha) * np.outer(u, u @ W)), -127, 127)
for item in a.head:
    l, h, factor = item.split(":")
    W = array("Wo_" + l)
    W[:, 40 * int(h):40 * int(h) + 40] = np.clip(np.round(float(factor) * W[:, 40 * int(h):40 * int(h) + 40]), -127, 127)
for item in a.kappa:
    l, kappa = item.split(":")
    d = np.arange(1, 128)
    array("EXP")[int(l), 256 - d] = np.round(127 * np.exp(-d / float(kappa)))
for name, W in new.items():
    s.put(name, W, "for a trace: " + note)

out = s.trace([ids for g, ids, n in seqs], ["u"], note)
run = len(s.budget["runs"])
restore(s, saved, "undo after trace %d" % run)

task, nll, hit = [], [], []
for (g, ids, n), o in zip(seqs, out):
    logits = o["u"][n - 1:len(ids) - 1] / 2.0 ** s.spec["frac"]["u"]
    logp = logits - logits.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    target = np.array(ids[n:])
    nll += list(-logp[np.arange(len(target)), target])
    hit += list(o["u"][n - 1:len(ids) - 1].argmax(-1) == target)
    task += [g] * len(target)
task, nll, hit = np.array(task), np.array(nll), np.array(hit)
np.savez("%s/refs_%02d.npz" % (DIR, run), task=task, nll=nll, hit=hit)
solved = sum(hit[task == g].all() for g in np.unique(task))
print("run %d (%s): %d body tokens of %d tasks: nll %.4f  top1 %.4f  solved %d" % (run, note, len(nll), len(seqs), nll.mean(), hit.mean(), solved))
try:
    first = sorted(f for f in __import__("os").listdir(DIR) if f.startswith("refs_"))[0]
    base = np.load(DIR + "/" + first)
    if first != "refs_%02d.npz" % run:
        d_nll = np.array([(nll - base["nll"])[task == g].sum() for g in np.unique(task)]) / len(nll) * len(np.unique(task))
        d_hit = np.array([(hit.astype(float) - base["hit"])[task == g].sum() for g in np.unique(task)]) / len(nll) * len(np.unique(task))
        se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
        print("   against %s: nll %+.4f +- %.4f   top1 %+.4f +- %.4f   (mean +- standard error over tasks)" % (
            first, d_nll.mean(), se(d_nll), d_hit.mean(), se(d_hit)))
except IndexError:
    pass
