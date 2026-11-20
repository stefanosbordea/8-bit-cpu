"""No budget: the same copy test as script 27 (1) for every head of layer 1 (they share the values, each has its
own columns of Wo_1), and for the docstring argument mentions."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_refs")
A = {k: v.astype(np.int64) for k, v in s.arrays.items()}; E = A["E"]
digit = {t for t in range(512) if tok.pieces[t].strip(" .").isdigit()}
num = lambda t: tok.pieces[t].strip(" .")
close_id = tok.sp.piece_to_id("):")
for h in range(4):
    ranks, same = [], []
    for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
        ids = list(ids); hi = ids.index(close_id)
        for p in range(hi + 3, n_prompt - 1):
            if ids[p] in digit:
                logit = E @ (A["Wo_1"][:, 40 * h:40 * h + 40] @ tr["l1.v"][p])
                ranks.append(int((logit > logit[ids[p]]).sum()))
                same.append(min(int((logit > logit[t]).sum()) for t in digit if num(t) == num(ids[p])))
    print("head %d: docstring digit keys: own token median rank %d, top 5 %.2f; any token of the same number: median %d, top 5 %.2f" % (
        h, np.median(ranks), np.mean(np.array(ranks) < 5), np.median(same), np.mean(np.array(same) < 5)))
