"""Analysis of trace_02.pkl (no model run): a repetition direction from matched pairs. The plain difference of
means (script 03) mostly says "digits and '* '", because the loops are `1024 * 1024 * ...`. Here a looping
position is compared with the same place of the FIRST turn of its own cycle (same tokens around it), so the
content cancels and what is left is "this has been written before".
Run from the repository root: PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/04_matched_direction.py"""
import pickle

import numpy as np

from common import DIR, tokenizer
from lsllm.edit.session import Session

s = Session(DIR)
tok = tokenizer(s)
data = pickle.load(open(DIR + "/trace_02.pkl", "rb"))
plain = np.load(DIR + "/directions_03.npz")

pairs, normal = [], []           # (sequence, looping position, matched first-turn position)
for q, c in enumerate(data):
    ids = c["ids"]
    for p in range(c["n_prompt"] - 1, len(ids) - 1):
        for P in range(2, 9):
            if p + 1 - 2 * P >= 0 and ids[p + 1 - 2 * P:p + 1 - P] == ids[p + 1 - P:p + 1] and ids[p + 1] == ids[p + 1 - P]:
                first = p
                while first - P >= c["n_prompt"] and ids[first - P] == ids[first]:
                    first -= P
                pairs.append((q, p, first))
                break
        else:
            normal.append((q, p))
first_turn = {(q, f) for q, p, f in pairs}
print("pairs %d, different first-turn positions %d, in answers %s" % (
    len(pairs), len(first_turn), sorted({data[q]["task"] for q, p, f in pairs})))

E = s.get("E").astype(np.float64)
out = {}
for name in ("l0.xa", "l0.xf", "l1.xa", "l1.xf"):
    x = lambda q, p: data[q]["trace"][name][p].astype(np.float64)
    # every answer counts once: mean over its pairs, then mean over answers (064 alone has half of the pairs)
    per_answer = {}
    for q, p, f in pairs:
        per_answer.setdefault(q, []).append(x(q, p) - x(q, f))
    diff = np.mean([np.mean(v, 0) for v in per_answer.values()], 0)
    u = diff / np.linalg.norm(diff)
    out[name.replace(".", "_")] = u
    pl = np.array([x(q, p) @ u for q, p, f in pairs])
    pf = np.array([x(q, f) @ u for q, f in first_turn])
    pn = np.array([x(q, p) @ u for q, p in normal])
    agree = np.mean([np.mean(v, 0) @ u > 0 for v in per_answer.values()])
    print("%-6s |diff| %6.2f  cos with the plain direction %.2f  projection: looping %6.1f +- %5.1f, first turn %6.1f +- %5.1f, "
          "other %6.1f +- %5.1f; answers whose own difference points along u: %.0f%%" % (
              name, np.linalg.norm(diff), u @ plain[name.replace(".", "_")], pl.mean(), pl.std(), pf.mean(), pf.std(),
              pn.mean(), pn.std(), 100 * agree))
    score = E @ u
    print("       favours", [tok.pieces[t] for t in np.argsort(-score)[:12]], "disfavours", [tok.pieces[t] for t in np.argsort(score)[:12]])
for l in (0, 1):
    u = out["l%d_xf" % l]
    W2, Wo = s.get("W2_%d" % l).astype(np.float64), s.get("Wo_%d" % l).astype(np.float64)
    g = lambda q, p: data[q]["trace"]["l%d.g" % l][p].astype(np.float64)
    o = lambda q, p: data[q]["trace"]["l%d.o" % l][p].ravel().astype(np.float64)
    print("layer %d writes along the matched direction (stream units): feed-forward looping %.1f first turn %.1f other %.1f | "
          "attention looping %.1f first turn %.1f other %.1f" % (
              l, np.mean([g(q, p) @ W2.T @ u for q, p, f in pairs]) / 256, np.mean([g(q, f) @ W2.T @ u for q, f in first_turn]) / 256,
              np.mean([g(q, p) @ W2.T @ u for q, p in normal]) / 256,
              np.mean([o(q, p) @ Wo.T @ u for q, p, f in pairs]) / 256, np.mean([o(q, f) @ Wo.T @ u for q, f in first_turn]) / 256,
              np.mean([o(q, p) @ Wo.T @ u for q, p in normal]) / 256))
np.savez(DIR + "/directions_04.npz", **out)
