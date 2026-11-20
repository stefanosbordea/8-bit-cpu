"""No budget: feed-forward units of layer 1. For the units that vote most for the undefined name at the fatal
positions (script 12): mean activation inside the docstring and inside the body (run 2 trace), and the tokens they
push. Layer 1 is the last layer, so what it writes at prompt positions never reaches the answer: a unit that
serves docstring text can be damped without touching how the prompt is read."""
import json, re, sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_refs")
_, _, _, own = load("trace_own")
A = {k: v.astype(np.int64) for k, v in s.arrays.items()}; E = A["E"]
samples = [json.loads(line) for line in open("edits/s2/eval_01.jsonl")]
close_id = tok.sp.piece_to_id("):")
doc, body = [], []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    hi = list(ids).index(close_id)
    doc.append(tr["l1.g"][hi + 3:n_prompt - 1]); body.append(tr["l1.g"][n_prompt:])
doc, body = np.concatenate(doc).astype(float), np.concatenate(body).astype(float)
fatal = np.zeros(219); right = np.zeros(219); n = 0
open_id, comma = tok.sp.piece_to_id("("), tok.sp.piece_to_id(",▁")
for (ids, n_prompt, g), tr in zip(own["seqs"], own["trace"]):
    m = re.match(r"error: NameError\(\"name '(\w+)' is not defined", samples[g]["detail"])
    if not m:
        continue
    ids = np.array(ids)
    at = len(tasks[g]["prompt"]) + re.search(r"(?<![\w.])%s(?!\w)" % m.group(1), samples[g]["completion"]).start()
    p = next(k for k in range(1, len(ids)) if len(tok.decode(ids[1:k + 1])) > at)
    lo, hi = list(ids).index(open_id), list(ids).index(close_id)
    args = [t for t in ids[lo + 1:hi] if t != comma]
    best = max(set(args), key=lambda t: tr["u"][p - 1, t])
    if best == ids[p]:
        continue
    fatal += (A["W2_1"].T @ (E[ids[p]] - E[best])) * tr["l1.g"][p - 1] / 256; n += 1
push = A["W2_1"].T @ E.T
order = np.argsort(fatal)[::-1]
print("%d fatal positions whose first token is not an argument token. unit: votes for the undefined name (sum) | mean activation docstring / body | pushes" % n)
for k in list(order[:14]) + list(order[-5:]):
    top = np.argsort(push[k])[::-1][:9]
    print("unit %3d %6.0f | %5.1f / %5.1f | %s" % (k, fatal[k], doc[:, k].mean(), body[:, k].mean(), " ".join(repr(tok.pieces[t]) for t in top)))
ratio = (doc.mean(0) + 1) / (body.mean(0) + 1)
print("units most active in the docstring relative to the body:", " ".join("%d:%.1f/%.1f(%.0f)" % (k, doc[:, k].mean(), body[:, k].mean(), fatal[k]) for k in np.argsort(ratio)[::-1][:16]))
print("sum of votes of all units %.0f; of the 16 docstring units %.0f" % (fatal.sum(), fatal[np.argsort(ratio)[::-1][:16]].sum()))
