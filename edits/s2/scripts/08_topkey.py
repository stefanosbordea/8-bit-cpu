"""No budget: at the positions where a signature token is due, is the key with the largest weight of layer 1 head 0
a copy of the right token? (If yes, a sharper head 0 helps; if no, sharpening copies the wrong thing harder.)"""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load()
close_id = tok.sp.piece_to_id("):")
res = {True: [], False: []}
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids); pred = tr["u"].argmax(-1); sig_end = list(ids).index(close_id)
    e, code = attention(s, tr, 1)
    for p in range(n_prompt - 1, len(ids) - 1):
        ref = ids[p + 1]
        if ref not in ids[1:sig_end + 1] or tok.pieces[ref] in ("(", ")", ", "):
            continue
        w = e[p, 0, :p + 1]
        best = np.flatnonzero(w == w.max())
        res[pred[p] == ref].append(((ids[best] == ref).mean(), (ids[best] == pred[p]).mean(), best.max() >= n_prompt, best.max() <= sig_end))
for ok in (False, True):
    arr = np.array(res[ok], dtype=float)
    print("model %s (%d): top key of head 0 is the right token %.2f, is the chosen token %.2f, lies in the body %.2f, in the signature %.2f" % (
        "right" if ok else "wrong", len(arr), *arr.mean(0)))
