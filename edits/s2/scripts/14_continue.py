"""No budget: NameErrors whose first token is right (the name starts like an argument) and a later token is not.
At the first token that leaves the argument: what should come (the token after the same token in the signature),
what came, where layer 1 head 0 looks, and which parts vote for the wrong token."""
import json, re, sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load("trace_own")
A = {k: v.astype(np.int64) for k, v in s.arrays.items()}
E = A["E"]
samples = [json.loads(line) for line in open("edits/s2/eval_01.jsonl")]
open_id, close_id, comma = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):"), tok.sp.piece_to_id(",▁")
comps = []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    m = re.match(r"error: NameError\(\"name '(\w+)' is not defined", samples[g]["detail"])
    if not m:
        continue
    ids = np.array(ids); u = tr["u"]
    at = len(tasks[g]["prompt"]) + re.search(r"(?<![\w.])%s(?!\w)" % m.group(1), samples[g]["completion"]).start()
    p = next(k for k in range(1, len(ids)) if len(tok.decode(ids[1:k + 1])) > at)
    lo, hi = list(ids).index(open_id), list(ids).index(close_id)
    sig = list(ids[lo + 1:hi + 1])
    if ids[p] not in sig[:-1]:
        continue
    k = lo + 1 + sig.index(ids[p])            # the same token in the signature
    q = p
    while ids[q + 1] == ids[k + 1]:
        q += 1; k += 1
    right, chosen = ids[k + 1], ids[q + 1]
    e, _ = attention(s, tr, 1)
    w = e[q, 0, :q + 1] / e[q, 0, :q + 1].sum()
    top = int(np.argmax(w))
    dE = E[chosen] - E[right]
    comp = {"x0": tr["x0"][q] @ dE}
    for l in (0, 1):
        o = tr["l%d.o" % l][q].reshape(4, 40); Wo = A["Wo_%d" % l].reshape(160, 4, 40)
        for h in range(4):
            comp["a%d.%d" % (l, h)] = (Wo[:, h] @ o[h]) @ dE / 256
        comp["f%d" % l] = ((A["W2_%d" % l].T @ dE) * tr["l%d.g" % l][q] / 256).sum()
    comp["total"] = tr["l1.xf"][q] @ dE
    comps.append(comp)
    print("%3d %-10s after %-6r right %-6r chosen %-6r gap %3d | head 0: on the right key %.2f, signature %.2f, docstring %.2f, top key %r at %s | cos(E right, E chosen) %.2f" % (
        g, m.group(1), tok.pieces[ids[q]], tok.pieces[right], tok.pieces[chosen], u[q, chosen] - u[q, right], w[k + 1], w[lo:hi + 1].sum(), w[hi + 1:n_prompt].sum(),
        tok.pieces[ids[top]], "sig" if top <= hi else "doc" if top < n_prompt else "body",
        E[right] @ E[chosen] / np.linalg.norm(E[right]) / np.linalg.norm(E[chosen])))
print("votes for the chosen token against the right continuation, mean over %d cases:" % len(comps))
for k in comps[0]:
    print("  %-6s %8.1f   (positive in %d cases)" % (k, np.mean([c[k] for c in comps]), sum(c[k] > 0 for c in comps)))
