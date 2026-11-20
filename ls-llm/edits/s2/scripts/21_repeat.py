"""No budget: (1) how many completions of a run repeat one name around an operator (`b < b`, `nums > nums`);
(2) on the reference trace (base model): where the right token is an argument and the model instead repeats a name
that the body already has, which parts vote for the repeat?"""
import json, re, sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
for run in (1, 15):
    comps = [json.loads(l)["completion"] for l in open("edits/s2/eval_%02d.jsonl" % run)]
    rep = [c for c in comps if re.search(r"\b([A-Za-z_]\w*) (?:[<>+*/%-]|==|!=|<=|>=|and|or|in|//|\*\*) \1\b", c)]
    print("run %d: %d of 200 completions have `X op X`" % (run, len(rep)))
s, tok, tasks, data = load("trace_refs")
A = {k: v.astype(np.int64) for k, v in s.arrays.items()}; E = A["E"]
open_id, close_id, comma = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):"), tok.sp.piece_to_id(",▁")
comps = []; shares = []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids); u = tr["u"]; pred = u.argmax(-1)
    lo, hi = list(ids).index(open_id), list(ids).index(close_id)
    args = set(ids[lo + 1:hi]) - {comma}
    e, _ = attention(s, tr, 1)
    for p in range(n_prompt, len(ids) - 1):
        ref, got = ids[p + 1], pred[p]
        base = lambda t: tok.pieces[t].strip()
        body = {base(t) for t in ids[n_prompt:p + 1]}
        if ref in args and got != ref and base(got) != base(ref) and base(got) in body and (got in args or base(got) in {base(a) for a in args}):
            dE = E[got] - E[ref]
            comp = {"x0": tr["x0"][p] @ dE}
            for l in (0, 1):
                o = tr["l%d.o" % l][p].reshape(4, 40); Wo = A["Wo_%d" % l].reshape(160, 4, 40)
                for h in range(4):
                    comp["a%d.%d" % (l, h)] = (Wo[:, h] @ o[h]) @ dE / 256
                comp["f%d" % l] = ((A["W2_%d" % l].T @ dE) * tr["l%d.g" % l][p] / 256).sum()
            comps.append(comp)
            w = e[p, 0, :p + 1] / e[p, 0, :p + 1].sum()
            key = lambda name: sum(w[t] for t in range(p + 1) if base(ids[t]) == name)
            shares.append((key(base(ref)), key(base(got))))
            print("  task %3d %-28r right %-5r repeated %-5r head 0 share on right %.2f on repeated %.2f" % (g, "".join(tok.pieces[t] for t in ids[n_prompt + 1:p + 1])[-28:], tok.pieces[ref], tok.pieces[got], *shares[-1]))
print("votes for the repeated name against the right argument, mean over %d cases:" % len(comps))
for k in comps[0]:
    print("  %-6s %8.1f   (positive in %d cases)" % (k, np.mean([c[k] for c in comps]), sum(c[k] > 0 for c in comps)))
