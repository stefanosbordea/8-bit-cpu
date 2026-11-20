"""No budget: in the repeat cases of script 21, is the weight of head 0 for the repeated name on its copy in the
signature, in the docstring or in the body? Also: over ALL body positions where an argument is due, the share of
head 0 on body keys."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load("trace_refs")
open_id, close_id, comma = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):"), tok.sp.piece_to_id(",▁")
rows = []; allb = []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids); pred = tr["u"].argmax(-1)
    lo, hi = list(ids).index(open_id), list(ids).index(close_id)
    args = set(ids[lo + 1:hi]) - {comma}
    e, _ = attention(s, tr, 1)
    base = lambda t: tok.pieces[t].strip()
    for p in range(n_prompt, len(ids) - 1):
        ref, got = ids[p + 1], pred[p]
        w = e[p, 0, :p + 1] / e[p, 0, :p + 1].sum()
        if ref in args:
            allb.append(w[n_prompt:].sum())
        body = {base(t) for t in ids[n_prompt:p + 1]}
        if ref in args and got != ref and base(got) != base(ref) and base(got) in body and base(got) in {base(a) for a in args}:
            where = lambda name, a, b: sum(w[t] for t in range(a, min(b, p + 1)) if base(ids[t]) == name)
            rows.append((where(base(got), lo, hi + 1), where(base(got), hi + 1, n_prompt), where(base(got), n_prompt, p + 1),
                         where(base(ref), lo, hi + 1), where(base(ref), hi + 1, n_prompt), where(base(ref), n_prompt, p + 1)))
r = np.array(rows)
print("repeated name: signature %.2f docstring %.2f body %.2f | right name: signature %.2f docstring %.2f body %.2f  (mean of %d cases)" % (*r.mean(0), len(r)))
print("all positions where an argument is due: share of head 0 on body keys %.3f" % np.mean(allb))
