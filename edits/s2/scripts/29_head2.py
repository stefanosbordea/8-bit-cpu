"""No budget: where do the heads of layer 1 look when a digit is due (reference trace, base model), split by
whether the right number stands in the docstring; and how many different numbers the docstring has."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load("trace_refs")
digit = {t for t in range(512) if tok.pieces[t].strip(" .").isdigit()}
num = lambda t: tok.pieces[t].strip(" .")
close_id = tok.sp.piece_to_id("):")
rows = []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids); hi = list(ids).index(close_id); pred = tr["u"].argmax(-1)
    e, _ = attention(s, tr, 1)
    doc = [t for t in range(hi + 3, n_prompt - 1) if ids[t] in digit]
    for p in range(n_prompt, len(ids) - 1):
        if ids[p + 1] in digit:
            right = [t for t in doc if num(ids[t]) == num(ids[p + 1])]
            w = e[p, :, :p + 1] / e[p, :, :p + 1].sum(-1, keepdims=True)
            rows.append([len(right) > 0, len(doc), pred[p] == ids[p + 1]] + [w[h, doc].sum() for h in range(4)] + [w[h, right].sum() for h in range(4)])
            if len(doc):
                print("  task %3d %-22r due %-4r model %-5r docstring digits %s" % (g, "".join(tok.pieces[t] for t in ids[n_prompt + 1:p + 1])[-22:], tok.pieces[ids[p + 1]], tok.pieces[pred[p]], [tok.pieces[ids[t]] for t in doc]))
r = np.array(rows, float)
for flag, title in ((1, "right number in the docstring"), (0, "right number not in the docstring")):
    q = r[r[:, 0] == flag]
    print("%s: %d positions, docstring has digits in %d, model right %.2f; share on docstring digits per head %s; on the right number %s" % (
        title, len(q), (q[:, 1] > 0).sum(), q[:, 2].mean(), np.round(q[:, 3:7].mean(0), 2), np.round(q[:, 7:11].mean(0), 2)))
