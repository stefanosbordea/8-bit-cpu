"""No model run. Positions where a word of the prompt must begin (the reference token stands in the prompt, but
not after the previous token): right at only 55 %. Which feed-forward units of layer 1 work against the
reference there, in both halves of the tasks? (attribution of state e6)"""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, D, tok, tasks
from lsllm.edit.session import Session

s = Session("edits/s4")
E, W2 = s.get("E"), s.get("W2_1")
A, rows, tokens = load([], "e6")
z = np.load(D + "attribution_e6.npz")
units, margin, ok = z["units"], z["margin"], z["ok"]
pos, ref = rows[:, 2], rows[:, 3]
prev = tokens[pos]
letters = np.array([P(t)[:1].isalpha() or P(t)[:1] == "_" for t in range(512)])
pairs = {n: None for n in range(200)}
for n in range(200):
    ids = tok.encode(tasks[n]["prompt"])
    pairs[n] = (set(zip(ids, ids[1:])), set(ids))
ind = np.array([(int(a), int(b)) in pairs[n][0] for n, a, b in zip(rows[:, 0], prev, ref)]) & letters[prev] & letters[ref]
first = np.array([int(b) in pairs[n][1] for n, b in zip(rows[:, 0], ref)]) & letters[ref] & ~ind
even = rows[:, 0] % 2 == 0
near = np.abs(margin) <= 8
fe, fo = units[first & even].sum(0), units[first & ~even].sum(0)
re_, ro = units[~first & even].sum(0), units[~first & ~even].sum(0)
dec = units[near].sum(0)
out = E @ W2
print("name-start positions: %d, right %.3f" % (first.sum(), ok[first].mean()))
print("unit | name-start even odd | elsewhere even odd | decisive all | writes")
for k in np.argsort(fe + fo)[:20]:
    print("%4d | %7.1f %7.1f | %7.1f %7.1f | %7.1f | %s" % (k, fe[k], fo[k], re_[k], ro[k], dec[k], " ".join(repr(P(t)) for t in np.argsort(-out[:, k])[:7])))
