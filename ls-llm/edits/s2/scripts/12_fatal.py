"""No budget: in the model's own NameError completions, the position where the undefined name starts.
Where does layer 1 head 0 look there, and which parts of the model vote for the chosen token over the best
token of the argument list?"""
import json, re, sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load("trace_own")
A = {k: v.astype(np.int64) for k, v in s.arrays.items()}
E = A["E"]
samples = [json.loads(line) for line in open("edits/s2/eval_01.jsonl")]
open_id, close_id, comma = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):"), tok.sp.piece_to_id(",▁")
agree = [0, 0]; rows = []; comps = []; unit = np.zeros((2, 219))
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids); u = tr["u"]; pred = u.argmax(-1)
    body_ok = pred[n_prompt - 1:-1] == ids[n_prompt:]
    agree[0] += body_ok.sum(); agree[1] += len(body_ok)
    m = re.match(r"error: NameError\(\"name '(\w+)' is not defined", samples[g]["detail"])
    if not m:
        continue
    name = m.group(1)
    text = tasks[g]["prompt"] + samples[g]["completion"]
    at = len(tasks[g]["prompt"]) + re.search(r"(?<![\w.])%s(?!\w)" % name, samples[g]["completion"]).start()
    p = next(k for k in range(1, len(ids)) if len(tok.decode(ids[1:k + 1])) > at)   # token that starts the name
    lo, hi = list(ids).index(open_id), list(ids).index(close_id)
    args = [t for t in ids[lo + 1:hi] if t != comma]
    e, _ = attention(s, tr, 1)
    w = e[p - 1, 0, :p] / e[p - 1, 0, :p].sum()
    doc_end = n_prompt
    best_arg = max(set(args), key=lambda t: u[p - 1, t])
    top = int(np.argmax(w))
    rows.append((g, name, tok.pieces[ids[p]], tok.pieces[best_arg], u[p - 1, ids[p]] - u[p - 1, best_arg], w[lo:hi + 1].sum(), w[hi + 1:doc_end].sum(), w[doc_end:].sum(),
                 tok.pieces[ids[top]], "sig" if top <= hi else "doc" if top < doc_end else "body", bool(body_ok[:p - n_prompt + 1].all())))
    dE = E[ids[p]] - E[best_arg]
    comp = {"x0": tr["x0"][p - 1] @ dE}
    for l in (0, 1):
        o = tr["l%d.o" % l][p - 1].reshape(4, 40); Wo = A["Wo_%d" % l].reshape(160, 4, 40)
        for h in range(4):
            comp["a%d.%d" % (l, h)] = (Wo[:, h] @ o[h]) @ dE / 256
        units = (A["W2_%d" % l].T @ dE) * tr["l%d.g" % l][p - 1] / 256
        comp["f%d" % l] = units.sum(); unit[l] += units
    comp["total"] = tr["l1.xf"][p - 1] @ dE
    comps.append(comp)
print("tokens of the completions that the model predicts again when re-tokenised: %d of %d" % tuple(agree))
print("task name  first-token best-arg  logit gap | head 0 share: signature docstring body | top key (where) | same path")
for r in rows:
    print("%3d %-18s %-7r %-7r %3d | %.2f %.2f %.2f | %-7r %-4s | %s" % r)
arr = np.array([r[5:8] for r in rows])
print("mean share of head 0 at the fatal position: signature %.2f docstring %.2f body %.2f" % tuple(arr.mean(0)))
print("top key in: ", {k: sum(r[9] == k for r in rows) for k in ("sig", "doc", "body")})
print("votes for the chosen (undefined) first token against the best argument token, mean over %d cases:" % len(comps))
for k in comps[0]:
    print("  %-6s %8.1f   (positive in %d cases)" % (k, np.mean([c[k] for c in comps]), sum(c[k] > 0 for c in comps)))
for l in (0, 1):
    order = np.argsort(unit[l])[::-1]
    print("layer %d units for the undefined name:" % l, " ".join("%d:%.0f" % (k, unit[l][k]) for k in order[:12]), "| against:", " ".join("%d:%.0f" % (k, unit[l][k]) for k in order[-6:]))
