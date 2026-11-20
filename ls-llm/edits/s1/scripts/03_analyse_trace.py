"""Analysis of trace_02.pkl (no model run): the repetition direction, ties of the 7-bit logits, attention sharpness.
Run from the repository root: PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/03_analyse_trace.py"""
import pickle

import numpy as np

from common import DIR, tokenizer
from lsllm.edit.session import Session

s = Session(DIR)
tok = tokenizer(s)
data = pickle.load(open(DIR + "/trace_02.pkl", "rb"))


def looping(ids, p):
    """True if the token after position p continues a cycle: for a period P of 2..8 the last 2P tokens up to
    p are two equal halves and the next token repeats the one P back."""
    for P in range(2, 9):
        if p + 1 < len(ids) and p + 1 - 2 * P >= 0 and ids[p + 1 - 2 * P:p + 1 - P] == ids[p + 1 - P:p + 1] \
                and ids[p + 1] == ids[p + 1 - P]:
            return True
    return False


loop, normal = [], []                      # (sequence, position)
for q, c in enumerate(data):
    for p in range(c["n_prompt"] - 1, len(c["ids"]) - 1):
        (loop if looping(c["ids"], p) else normal).append((q, p))
print("looping positions %d (in %d answers), normal answer positions %d" % (
    len(loop), len({q for q, p in loop}), len(normal)))


def rows(name, where):
    return np.stack([data[q]["trace"][name][p].ravel() for q, p in where]).astype(np.float64)


directions = {}
for name in ("l0.xa", "l0.xf", "l1.xa", "l1.xf", "y"):
    a, b = rows(name, loop), rows(name, normal)
    diff = a.mean(0) - b.mean(0)
    u = diff / np.linalg.norm(diff)
    pa, pb = a @ u, b @ u
    dprime = (pa.mean() - pb.mean()) / np.sqrt((pa.var() + pb.var()) / 2)
    directions[name] = u
    print("%-6s |mean diff| %6.2f  mean |x| %6.2f  projections loop %6.2f +- %5.2f normal %6.2f +- %5.2f  d' %.2f" % (
        name, np.linalg.norm(diff), np.linalg.norm(b, axis=1).mean(), pa.mean(), pa.std(), pb.mean(), pb.std(), dprime))
np.savez(DIR + "/directions_03.npz", **{k.replace(".", "_"): v for k, v in directions.items()})

# what the layers write along the direction of the stream after them
for l in (0, 1):
    u = directions["l%d.xf" % l]
    W2, Wo = s.get("W2_%d" % l), s.get("Wo_%d" % l)
    g_loop, g_norm = rows("l%d.g" % l, loop), rows("l%d.g" % l, normal)
    o_loop, o_norm = rows("l%d.o" % l, loop), rows("l%d.o" % l, normal)
    print("layer %d: feed-forward writes along u (stream units): loop %.2f normal %.2f | attention: loop %.2f normal %.2f" % (
        l, (g_loop @ W2.T @ u).mean() / 256, (g_norm @ W2.T @ u).mean() / 256,
        (o_loop @ Wo.T @ u).mean() / 256, (o_norm @ Wo.T @ u).mean() / 256))
    share = np.linalg.norm(u @ W2) ** 2 / np.linalg.norm(W2) ** 2
    print("   share of |W2|^2 along u: %.4f (1/160 = %.4f); of |Wo|^2: %.4f" % (
        share, 1 / 160, np.linalg.norm(u @ Wo) ** 2 / np.linalg.norm(Wo) ** 2))

# which tokens the direction of the last stream favours (E is the output matrix)
E = s.get("E").astype(np.float64)
for name in ("l1.xf", "y"):
    score = E @ directions[name]
    print(name, "direction favours:", [tok.pieces[t] for t in np.argsort(-score)[:15]],
          "| disfavours:", [tok.pieces[t] for t in np.argsort(score)[:15]])

# ties of the 7-bit logits: greedy takes the smallest id among the largest u
answer = loop + normal
u, acc = rows("u", answer), rows("u.acc", answer)
top = u.max(1)
tied = (u == top[:, None]).sum(1)
print("answer positions %d: largest logit code mean %.1f min %d max %d; saturated at 127: %d" % (
    len(answer), top.mean(), top.min(), top.max(), (top == 127).sum()))
print("positions with a tie for the largest code: %d (%.1f%%); argmax of the code != argmax of the accumulator: %d" % (
    (tied > 1).sum(), 100 * (tied > 1).mean(), (u.argmax(1) != acc.argmax(1)).sum()))
gap = np.sort(acc, 1)
print("gap between first and second accumulator, in codes (1 code = 512): quartiles", np.percentile((gap[:, -1] - gap[:, -2]) / 512, [10, 25, 50, 75, 90]).round(2))

# attention sharpness: sum of the weights / 127 = number of keys that count, and the largest score code
for l in (0, 1):
    dsum, mx = rows("l%d.dsum" % l, answer) / 127, rows("l%d.max" % l, answer)
    print("layer %d heads: keys that count (median) %s  | largest code median %s, share at 127: %s, share at 0: %s" % (
        l, np.median(dsum, 0).round(1), np.median(mx, 0), (mx == 127).mean(0).round(2), (mx == 0).mean(0).round(2)))
