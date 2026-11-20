"""No budget: at body positions where the right next token is a token of the signature, which parts of the model
vote for it and which against it (direct attribution of the logit difference right - chosen, or right - runner-up)."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load()
A = {k: v.astype(np.int64) for k, v in s.arrays.items()}
E = A["E"]
close_id = tok.sp.piece_to_id("):")
parts_fail, parts_ok, cases = [], [], []
unit_fail = np.zeros((2, 219)); unit_ok = np.zeros((2, 219)); n_fail = n_ok = 0
h0 = {"fail": [], "ok": []}
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids); P = len(ids)
    u = tr["u"]; pred = u.argmax(-1)
    sig_end = list(ids).index(close_id)
    e1, _ = attention(s, tr, 1)
    share1 = e1 / e1.sum(-1, keepdims=True)
    for p in range(n_prompt - 1, P - 1):
        ref = ids[p + 1]
        if ref not in ids[1:sig_end + 1] or tok.pieces[ref] in ("(", ")", ", "):
            continue
        ok = pred[p] == ref
        other = pred[p] if not ok else np.argsort(u[p])[-2]
        dE = E[ref] - E[other]
        comp = {"x0": tr["x0"][p] @ dE}
        for l in (0, 1):
            o = tr["l%d.o" % l][p].reshape(4, 40)
            Wo = A["Wo_%d" % l].reshape(160, 4, 40)
            for h in range(4):
                comp["a%d.%d" % (l, h)] = (Wo[:, h] @ o[h]) @ dE / 256
            units = (A["W2_%d" % l].T @ dE) * tr["l%d.g" % l][p] / 256           # per unit
            comp["f%d" % l] = units.sum()
            (unit_ok if ok else unit_fail)[l] += units
        comp["total"] = tr["l1.xf"][p] @ dE
        comp["du"] = u[p, ref] - u[p, other]
        (parts_ok if ok else parts_fail).append(comp)
        n_ok += ok; n_fail += not ok
        target = np.flatnonzero(ids[:p + 1] == ref)
        h0["ok" if ok else "fail"].append((share1[p, 0][target].sum(), share1[p, 0][:sig_end + 1].sum(), tr["l1.dsum"][p, 0, 0, 0] / 127))
        if not ok:
            cases.append((g, p, "".join(tok.pieces[t] for t in ids[max(n_prompt, p - 6):p + 1]), tok.pieces[ref], tok.pieces[other], share1[p, 0][target].sum()))
print("positions: right %d, wrong %d" % (n_ok, n_fail))
keys = list(parts_ok[0])
print("%-8s %10s %10s" % ("part", "wrong", "right"))
for k in keys:
    print("%-8s %10.1f %10.1f" % (k, np.mean([c[k] for c in parts_fail]), np.mean([c[k] for c in parts_ok])))
for tag in ("fail", "ok"):
    arr = np.array(h0[tag])
    print("layer 1 head 0 when %-4s: share on copies of the right token %.3f, on the signature %.3f, sum of weights / 127 = %.2f" % (tag, *arr.mean(0)))
for l in (0, 1):
    order = np.argsort(unit_fail[l])
    print("layer %d units most against the right token when wrong (unit: sum wrong / sum right):" % l,
          " ".join("%d:%.0f/%.0f" % (k, unit_fail[l][k], unit_ok[l][k]) for k in order[:14]))
    print("          most for it:", " ".join("%d:%.0f/%.0f" % (k, unit_fail[l][k], unit_ok[l][k]) for k in order[-8:]))
for c in cases:
    print("%3d %3d %-40r right %-8r chosen %-8r h0 share on right %.2f" % c)
