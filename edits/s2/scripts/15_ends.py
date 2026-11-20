"""No budget: does the model stop where the reference solution stops? Teacher-forced trace of the references:
at the last token, the logits of the tokens that end the answer (<DED1>, end of text) against the best other token;
at every earlier body position, how far the end tokens are below the right token (the room for an end bias)."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_refs")
ends = [tok.eos_id] + list(tok.ded_ids)
print("end tokens:", ends, [tok.sp.id_to_piece(t) for t in ends], "nl", tok.nl_id)
gaps, room, first = [], [], []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    u = tr["u"].astype(int)
    last = u[-1]
    other = np.delete(np.arange(512), ends)
    best_other = other[np.argmax(last[other])]
    gaps.append((last[ends].max() - last[best_other], tok.pieces[best_other], tok.sp.id_to_piece(int(ends[int(np.argmax(last[ends]))]))))
    for p in range(n_prompt, len(ids) - 1):
        room.append(u[p, ids[p + 1]] - u[p, ends].max())
        first.append(u[p].max() - u[p, ends].max())
gaps_v = np.array([g[0] for g in gaps])
print("at the true end: end token wins in %d of %d; gap (end - best other) percentiles 10/25/50/75: %s" % ((gaps_v > 0).sum(), len(gaps_v), np.percentile(gaps_v, [10, 25, 50, 75])))
print("what wins instead:", sorted(((g[1], g[0]) for g in gaps if g[0] <= 0), key=lambda x: x[1]))
print("which end token is the best one:", {k: sum(g[2] == k for g in gaps) for k in set(g[2] for g in gaps)})
room, first = np.array(room), np.array(first)
print("inside the body (%d positions): right token minus best end token, percentiles 1/5/10/25/50: %s" % (len(room), np.percentile(room, [1, 5, 10, 25, 50])))
print("  chosen token minus best end token, percentiles 1/5/10/25/50: %s; share with gap <= 4: %.3f, <= 8: %.3f, <= 12: %.3f" % (
    np.percentile(first, [1, 5, 10, 25, 50]), (first <= 4).mean(), (first <= 8).mean(), (first <= 12).mean()))
y = np.concatenate([tr["y"][n:] for (ids, n, g), tr in zip(data["seqs"], data["trace"])]).astype(float)
m = y.mean(0)
print("final norm output y on the body: |mean| %.1f, rms length %.1f; along the mean: %.1f +- %.1f" % (np.linalg.norm(m), np.sqrt((y ** 2).sum(1).mean()), (y @ m / np.linalg.norm(m)).mean(), (y @ m / np.linalg.norm(m)).std()))
