"""No budget: can the input of layer 1 at a body position tell that a NAME comes next? Class means of l1.ya over
the body positions of the traced references where the next token is a signature token, against the other body
positions. Checked on the fatal positions of the model's own NameError completions (run 8), which were not used
for the means. Saved to edits/s2/due.npz."""
import json, re, sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_refs")
_, _, _, own = load("trace_own")
samples = [json.loads(line) for line in open("edits/s2/eval_01.jsonl")]
open_id, close_id, comma = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):"), tok.sp.piece_to_id(",▁")
due, rest = [], []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids); lo, hi = list(ids).index(open_id), list(ids).index(close_id)
    args = set(ids[lo + 1:hi]) - {comma}
    for p in range(n_prompt, len(ids) - 1):
        (due if ids[p + 1] in args else rest).append(tr["l1.ya"][p])
due, rest = np.array(due, dtype=float), np.array(rest, dtype=float)
fatal, starts = [], []
for (ids, n_prompt, g), tr in zip(own["seqs"], own["trace"]):
    m = re.match(r"error: NameError\(\"name '(\w+)' is not defined", samples[g]["detail"])
    if m:
        at = len(tasks[g]["prompt"]) + re.search(r"(?<![\w.])%s(?!\w)" % m.group(1), samples[g]["completion"]).start()
        p = next(k for k in range(1, len(ids)) if len(tok.decode(ids[1:k + 1])) > at)
        fatal.append(tr["l1.ya"][p - 1])
fatal = np.array(fatal, dtype=float)
d = due.mean(0) - rest.mean(0)
u = d / np.linalg.norm(d)
mid = (due.mean(0) + rest.mean(0)) / 2 @ u
print("name due (%d): %.1f +- %.1f   other body positions (%d): %.1f +- %.1f   fatal positions of the own NameErrors (%d): %.1f +- %.1f" % (
    len(due), (due @ u).mean(), (due @ u).std(), len(rest), (rest @ u).mean(), (rest @ u).std(), len(fatal), (fatal @ u).mean(), (fatal @ u).std()))
print("share above the midpoint %.1f: name due %.2f, other %.2f, fatal %.2f" % (mid, (due @ u > mid).mean(), (rest @ u > mid).mean(), (fatal @ u > mid).mean()))
np.savez("edits/s2/due.npz", unit=u, mean_due=due.mean(0), mean_rest=rest.mean(0))
