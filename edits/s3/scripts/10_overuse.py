"""No budget: which tokens does the model write more often than the reference solutions of the practice tasks?
Counts over the 200 bodies (reference) and over the completions of the unedited model (runs 1, 21, 22 are identical;
the other runs are perturbed models and show whether the excess is stable)."""
import sys
from collections import Counter
from common import completions, open_session, practice

s, tok = open_session()
tasks = practice()


def body_ids(task, completion):
    full = tok.encode(task["prompt"] + completion, bos=True)
    return full[len(tok.encode(task["prompt"], bos=True)):]


ref = Counter()
ref_len = []
for task in tasks:
    ids = body_ids(task, task["canonical_solution"][len(task["prompt"]):])
    ref.update(ids); ref_len.append(len(ids))
runs = [int(a) for a in sys.argv[1:]] or [1]
got, got_len = Counter(), []
for run in runs:
    for task, row in zip(tasks, completions(run)):
        try:
            ids = body_ids(task, row["completion"])
        except ValueError:
            continue
        got.update(ids); got_len.append(len(ids))
scale = 1.0 / len(runs)
print("body tokens per task: reference %.1f, model %.1f" % (sum(ref_len) / 200, sum(got_len) / 200 * scale))
rows = sorted(((got[t] * scale - ref[t], t) for t in set(ref) | set(got)), reverse=True)
print("most over-used (model per run - reference, model, reference, token id, piece):")
for diff, t in rows[:28]:
    print("  %+7.1f %7.1f %5d  %3d %r" % (diff, got[t] * scale, ref[t], t, tok.pieces[t]))
print("most under-used:")
for diff, t in rows[-22:]:
    print("  %+7.1f %7.1f %5d  %3d %r" % (diff, got[t] * scale, ref[t], t, tok.pieces[t]))
