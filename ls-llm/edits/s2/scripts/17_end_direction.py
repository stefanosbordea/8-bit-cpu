"""No budget: a direction of the final norm output y that says "the reference ends here": mean of y at the last
token of the 79 traced references minus the mean of y at the other body positions. Saved to edits/s2/end.npz."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_refs")
E = s.arrays["E"].astype(float)
end = np.array([tr["y"][-1] for tr in data["trace"]], dtype=float)
other = np.concatenate([tr["y"][n:-1] for (ids, n, g), tr in zip(data["seqs"], data["trace"])]).astype(float)
d = end.mean(0) - other.mean(0)
unit = d / np.linalg.norm(d)
print("along the direction: end %.1f +- %.1f [5%%: %.0f], other %.1f +- %.1f [95%%: %.0f, 99%%: %.0f]" % (
    (end @ unit).mean(), (end @ unit).std(), np.percentile(end @ unit, 5), (other @ unit).mean(), (other @ unit).std(), np.percentile(other @ unit, 95), np.percentile(other @ unit, 99)))
print("E row of </s>: length %.1f, cosine with the direction %.2f; logit units per unit of the direction: 1 / 512" % (np.linalg.norm(E[tok.eos_id]), E[tok.eos_id] @ unit / np.linalg.norm(E[tok.eos_id])))
gap = np.array([tr["u"][-1].astype(int)[[tok.eos_id] + list(tok.ded_ids)].max() - np.delete(tr["u"][-1].astype(int), [tok.eos_id] + list(tok.ded_ids)).max() for tr in data["trace"]])
proj = end @ unit
print("true ends that the model misses: gap and projection:", sorted(zip(gap[gap <= 0].tolist(), np.round(proj[gap <= 0]).tolist())))
np.savez("edits/s2/end.npz", unit=unit, end=(end @ unit).mean(), other=(other @ unit).mean())
