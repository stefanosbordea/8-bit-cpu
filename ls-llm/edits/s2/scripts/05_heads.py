"""No budget: what the teacher-forced trace says. Accuracy on the body, and what every head looks at."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load()
stats = np.zeros((2, 4, 6)); count = 0
hit = [0, 0, 0, 0]
rows = []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids); P = len(ids)
    pred = tr["u"].argmax(-1)                       # prediction of token p + 1 at position p
    sig_end = list(ids).index(tok.sp.piece_to_id("):")) if tok.sp.piece_to_id("):") in ids else 0
    for p in range(n_prompt - 1, P - 1):
        ok = pred[p] == ids[p + 1]
        in_sig = ids[p + 1] in ids[1:sig_end + 1]
        hit[0] += ok; hit[1] += 1
        rows.append((g, p, ok, in_sig))
    for l in (0, 1):
        e, code = attention(s, tr, l)
        share = e / e.sum(-1, keepdims=True)
        for p in range(n_prompt - 1, P - 1):
            same = np.flatnonzero(ids[:p] == ids[p])                    # earlier occurrences of the current token
            after = same + 1                                            # the token after them: what an induction head reads
            target = np.flatnonzero(ids[:p + 1] == ids[p + 1])          # earlier occurrences of the token to be written
            for h in range(4):
                w = share[p, h]
                stats[l, h] += [w[p], w[p - 1], w[same].sum(), w[after[after <= p]].sum() if len(same) else 0, w[target].sum(), w[:sig_end + 1].sum()]
        if l == 0:
            count += P - n_prompt
print("teacher-forced accuracy on the body: %d of %d" % (hit[0], hit[1]))
rows = np.array(rows)
print("  next token occurs in the signature: %.3f (%d)   other: %.3f" % (rows[rows[:, 3] == 1, 2].mean(), (rows[:, 3] == 1).sum(), rows[rows[:, 3] == 0, 2].mean()))
print("mean share of the attention at body positions on: self, previous, same token earlier, after same token (induction), earlier copies of the right next token, signature")
for l in (0, 1):
    for h in range(4):
        print("layer %d head %d  " % (l, h) + "  ".join("%.3f" % v for v in stats[l, h] / count))
