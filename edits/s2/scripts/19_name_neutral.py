"""No budget: the key direction signature - docstring also lifts the tokens of the FUNCTION NAME a little (they lie
at +21, the docstring at -18): names such as `letter`, `length`, `digit` are copied from there. A second direction
without that: remove from it the part along (function name - docstring). How well does it still separate?"""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_refs")
_, _, _, own = load("trace_own")
open_id, close_id = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):")
cls = {"name": [], "sig": [], "doc": [], "body": []}
for d_ in (data, own):
    for (ids, n_prompt, g), tr in zip(d_["seqs"], d_["trace"]):
        ids = list(ids); lo, hi = ids.index(open_id), ids.index(close_id)
        y = tr["l1.ya"]
        cls["name"].append(y[2:lo]); cls["sig"].append(y[lo + 1:hi]); cls["doc"].append(y[hi + 3:n_prompt - 1]); cls["body"].append(y[n_prompt + 1:])
cls = {k: np.concatenate(v).astype(float) for k, v in cls.items()}
mean = {k: v.mean(0) for k, v in cls.items()}
d = mean["sig"] - mean["doc"]; n = mean["name"] - mean["doc"]
d2 = d - (d @ n) / (n @ n) * n
for title, v in (("signature - docstring", d), ("the same without (function name - docstring)", d2)):
    u = v / np.linalg.norm(v)
    print(title + ": " + "  ".join("%s %.1f +- %.1f" % (k, (x @ u).mean(), (x @ u).std()) for k, x in cls.items()))
