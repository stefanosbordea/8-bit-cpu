"""No budget: can the input of layer 1 (l1.ya, what Wq_1 and Wk_1 read) tell signature, docstring and body apart
by one direction (difference of class means)? And how much does every RoPE pair of head 0 contribute to its scores?
Class means are saved to edits/s2/regions.npz for the edit scripts."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load("trace_refs")
_, _, _, own = load("trace_own")
open_id, close_id, q3 = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):"), tok.sp.piece_to_id('"""')
cls = {"name": [], "sig": [], "doc": [], "body": []}
qa, ka = [], []
for d_ in (data, own):
    for (ids, n_prompt, g), tr in zip(d_["seqs"], d_["trace"]):
        ids = list(ids); lo, hi = ids.index(open_id), ids.index(close_id)
        y = tr["l1.ya"]
        cls["name"].append(y[2:lo]); cls["sig"].append(y[lo + 1:hi]); cls["doc"].append(y[hi + 3:n_prompt - 1]); cls["body"].append(y[n_prompt + 1:])
        if d_ is data:
            qa.append(tr["l1.qr"][n_prompt:, 0, 0]); ka.append(tr["l1.kr"][:, 0])
cls = {k: np.concatenate(v).astype(float) for k, v in cls.items()}
mean = {k: v.mean(0) for k, v in cls.items()}
print({k: (len(v), round(float(np.linalg.norm(mean[k])), 1), round(float(np.sqrt((v ** 2).sum(1).mean())), 1)) for k, v in cls.items()}, "(count, |mean|, rms length)")
def show(title, d):
    d = d / np.linalg.norm(d)
    print(title + ": " + "  ".join("%s %.1f +- %.1f [%.0f..%.0f]" % (k, (v @ d).mean(), (v @ d).std(), np.percentile(v @ d, 5), np.percentile(v @ d, 95)) for k, v in cls.items()))
show("signature - docstring", mean["sig"] - mean["doc"])
show("body mean (a constant for queries?)", mean["body"])
show("body - prompt", mean["body"] - (mean["sig"] + mean["doc"]) / 2)
d = mean["sig"] - mean["doc"]; b = mean["body"] - (mean["sig"] + mean["doc"]) / 2
show("signature - docstring, without the body direction", d - (d @ b) / (b @ b) * b)
np.savez("edits/s2/regions.npz", **mean)
qa, ka = np.concatenate(qa).astype(float), np.concatenate(ka).astype(float)
print("head 0 of layer 1, per RoPE pair j (a at index j, b at 20 + j): rms q_a, rms k_a, rms q_b, rms k_b, theta_j")
for j in range(20):
    print("  j %2d  q_a %5.1f k_a %5.1f  q_b %5.1f k_b %5.1f  theta %.5f   mean q_a %5.1f" % (j, np.sqrt((qa[:, j] ** 2).mean()), np.sqrt((ka[:, j] ** 2).mean()),
          np.sqrt((qa[:, 20 + j] ** 2).mean()), np.sqrt((ka[:, 20 + j] ** 2).mean()), 10000 ** (-2 * j / 40), qa[:, j].mean()))
