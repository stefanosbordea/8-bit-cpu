"""No budget: could the copy head (layer 1 head 0) copy NUMBERS from the docstring if it looked at them?
(1) what head 0 would write if all its weight were on one docstring digit: push the traced value l1.v of that key
    through Wo_1 (columns of head 0) and E, and look where the digit itself ranks among the 512 logits;
    the same for signature tokens as a yardstick (the head is known to copy those);
(2) is "a digit is due" readable at the query (class means of l1.ya, measured on the other half of the tasks);
(3) are docstring digits readable at the key."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_refs")
A = {k: v.astype(np.int64) for k, v in s.arrays.items()}; E = A["E"]
digit = {t for t in range(512) if tok.pieces[t].strip(" .").isdigit()}
num = lambda t: tok.pieces[t].strip(" .")
open_id, close_id, comma = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):"), tok.sp.piece_to_id(",▁")
ranks_d, ranks_s, same_d = [], [], []
due, rest, grp_due, grp_rest, kd, ko = [], [], [], [], [], []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = list(ids); lo, hi = ids.index(open_id), ids.index(close_id)
    for p in range(lo + 1, n_prompt - 1):
        if (p > hi + 2 and ids[p] in digit) or (p < hi and ids[p] != comma):
            logit = E @ (A["Wo_1"][:, :40] @ tr["l1.v"][p])
            rank = int((logit > logit[ids[p]]).sum())
            if ids[p] in digit:
                ranks_d.append(rank)
                best = max((t for t in digit if num(t) == num(ids[p])), key=lambda t: logit[t])
                same_d.append(int((logit > logit[best]).sum()))
            else:
                ranks_s.append(rank)
        if p > hi + 2:
            (kd if ids[p] in digit else ko).append(tr["l1.ya"][p])
    for p in range(n_prompt, len(ids) - 1):
        (due if ids[p + 1] in digit else rest).append(tr["l1.ya"][p]); (grp_due if ids[p + 1] in digit else grp_rest).append(g)
print("(1) rank of the key's own token in what head 0 would write: docstring digits median %d, in the top 5: %.2f (%d keys); best token for the same number: median %d, top 5: %.2f" % (
    np.median(ranks_d), np.mean(np.array(ranks_d) < 5), len(ranks_d), np.median(same_d), np.mean(np.array(same_d) < 5)))
print("    signature tokens: median %d, in the top 5: %.2f (%d keys)" % (np.median(ranks_s), np.mean(np.array(ranks_s) < 5), len(ranks_s)))
due, rest, grp_due, grp_rest = np.array(due, float), np.array(rest, float), np.array(grp_due), np.array(grp_rest)
for half in (0, 1):
    a, b = grp_due % 2 == half, grp_rest % 2 == half
    v = due[a].mean(0) - rest[b].mean(0); v /= np.linalg.norm(v)
    print("(2) digit due, direction from half %d measured on the other: due %.1f +- %.1f (%d), rest %.1f +- %.1f" % (half, (due[~a] @ v).mean(), (due[~a] @ v).std(), (~a).sum(), (rest[~b] @ v).mean(), (rest[~b] @ v).std()))
kd, ko = np.array(kd, float), np.array(ko, float)
v = kd.mean(0) - ko.mean(0); v /= np.linalg.norm(v)
print("(3) docstring digits %.1f +- %.1f (%d), other docstring tokens %.1f +- %.1f" % ((kd @ v).mean(), (kd @ v).std(), len(kd), (ko @ v).mean(), (ko @ v).std()))
np.savez("edits/s2/digit.npz", q_due=due.mean(0), q_rest=rest.mean(0), k_digit=kd.mean(0), k_other=ko.mean(0))
