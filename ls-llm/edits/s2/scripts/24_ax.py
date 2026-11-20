"""No budget: the habitual two-letter names (`ax`, `dx`): after a bare one-letter argument the model writes `x `.
At those positions of the own-completion trace (base model): the logits of the candidates and which parts vote
for `x ` against the lone space (the token that would let an operator follow)."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_own")
A = {k: v.astype(np.int64) for k, v in s.arrays.items()}; E = A["E"]
x_sp, space = tok.sp.piece_to_id("x▁"), tok.sp.piece_to_id("▁")
bare = {tok.sp.piece_to_id(c) for c in "abcdns"}
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids)
    for p in range(n_prompt, len(ids) - 1):
        if ids[p] in bare and ids[p + 1] == x_sp:
            u = tr["u"][p].astype(int)
            top = np.argsort(u)[::-1][:6]
            print("task %d after %r: top %s" % (g, "".join(tok.pieces[t] for t in ids[n_prompt + 1:p + 1]), " ".join("%r:%d" % (tok.pieces[t], u[t]) for t in top)), "| space %d eos %d" % (u[space], u[tok.eos_id]))
            dE = E[x_sp] - E[space]
            comp = {"x0": tr["x0"][p] @ dE}
            for l in (0, 1):
                o = tr["l%d.o" % l][p].reshape(4, 40); Wo = A["Wo_%d" % l].reshape(160, 4, 40)
                for h in range(4):
                    comp["a%d.%d" % (l, h)] = round(float((Wo[:, h] @ o[h]) @ dE / 256))
                units = (A["W2_%d" % l].T @ dE) * tr["l%d.g" % l][p] / 256
                comp["f%d" % l] = round(float(units.sum()))
                if l == 1:
                    order = np.argsort(units)[::-1]
                    comp["f1 top units"] = " ".join("%d:%.0f" % (k, units[k]) for k in order[:6])
            print("   ", comp)
