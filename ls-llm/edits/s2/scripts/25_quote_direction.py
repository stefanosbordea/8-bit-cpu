"""No budget: the kept model writes a triple quote inside the body in 3 of 200 answers (the base model in none).
A direction of the final norm output y that is large in the body and zero, on average, where a triple quote is
really due (the two quotes of the prompt): body mean minus quote mean, then made orthogonal to the quote mean.
Saved to edits/s2/quote.npz."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_refs")
q3 = tok.sp.piece_to_id('"""')
due, body = [], []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids)
    due.append(tr["y"][:-1][ids[1:] == q3]); body.append(tr["y"][n_prompt:])
due, body = np.concatenate(due).astype(float), np.concatenate(body).astype(float)
m = due.mean(0)
d = body.mean(0) - m
d = d - (d @ m) / (m @ m) * m
u = d / np.linalg.norm(d)
print("along the direction: body %.1f +- %.1f [5%%: %.0f], quote due %.1f +- %.1f (%d positions)" % ((body @ u).mean(), (body @ u).std(), np.percentile(body @ u, 5), (due @ u).mean(), (due @ u).std(), len(due)))
print("E row of the triple quote: token %d, cosine with the direction %.2f" % (q3, s.arrays["E"][q3] @ u / np.linalg.norm(s.arrays["E"][q3].astype(float))))
np.savez("edits/s2/quote.npz", unit=u, body=(body @ u).mean())
