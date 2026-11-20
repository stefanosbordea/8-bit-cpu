"""No budget (trace of run 2, base model): (1) positions where a digit is due: is it in the docstring, is the model
right, where does layer 1 head 0 look; (2) the first name of the body: which argument is it in the reference,
which does the model choose, and how does head 0 split its weight over the arguments."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load("trace_refs")
digit = {t for t in range(512) if tok.pieces[t].strip(" .") .isdigit()}
open_id, close_id, comma = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):"), tok.sp.piece_to_id(",▁")
rows = []; firsts = []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    ids = np.array(ids); pred = tr["u"].argmax(-1)
    lo, hi = list(ids).index(open_id), list(ids).index(close_id)
    e, _ = attention(s, tr, 1)
    for p in range(n_prompt - 1, len(ids) - 1):
        ref = ids[p + 1]
        if ref in digit:
            w = e[p, 0, :p + 1] / e[p, 0, :p + 1].sum()
            doc = np.arange(hi + 1, n_prompt)
            same_number = [t for t in doc if tok.pieces[ids[t]].strip(" .") == tok.pieces[ref].strip(" .")]
            any_digit = [t for t in doc if ids[t] in digit]
            rows.append((pred[p] == ref, len(same_number) > 0, w[same_number].sum(), w[any_digit].sum(), w[lo:hi + 1].sum(), pred[p] in digit))
    # the first name of the body
    args, cur = [], []
    for t in range(lo + 1, hi + 1):
        if ids[t] in (comma, close_id):
            args.append(cur); cur = []
        else:
            cur.append(t)
    p = n_prompt        # position of the first body token (<NL> or <IND>.. then return): find the first argument token in the body
    q = next((k for k in range(n_prompt, len(ids)) if any(ids[k] == ids[a[0]] for a in args)), None)
    if q is not None and tok.pieces[ids[q - 1]] in ("return ", "(", "[", "not "):
        w = e[q - 1, 0, :q] / e[q - 1, 0, :q].sum()
        which = next(n for n, a in enumerate(args) if ids[a[0]] == ids[q])
        chosen = next((n for n, a in enumerate(args) if ids[a[0]] == pred[q - 1]), -1)
        firsts.append((len(args), which, chosen, [round(float(w[a].sum()), 2) for a in args], tok.pieces[pred[q - 1]], g))
rows = np.array(rows, dtype=float)
print("digit due at %d positions: model right %.2f; the number is in the docstring %.2f; model writes some digit %.2f" % (len(rows), rows[:, 0].mean(), rows[:, 1].mean(), rows[:, 5].mean()))
print("  head 0 share on copies of the right number %.2f, on any digit of the docstring %.2f, on the signature %.2f" % tuple(rows[:, 2:5].mean(0)))
print("  right when in docstring %.2f (%d), when not %.2f (%d)" % (rows[rows[:, 1] == 1, 0].mean(), (rows[:, 1] == 1).sum(), rows[rows[:, 1] == 0, 0].mean(), (rows[:, 1] == 0).sum()))
many = [f for f in firsts if f[0] > 1]
print("first name of the body, tasks with several arguments: %d; the reference uses argument 1 in %d; the model chooses the reference's in %d, argument 1 in %d, the last in %d, no argument in %d" % (
    len(many), sum(f[1] == 0 for f in many), sum(f[1] == f[2] for f in many), sum(f[2] == 0 for f in many), sum(f[2] == f[0] - 1 for f in many), sum(f[2] == -1 for f in many)))
for f in many:
    print("   task %3d args %d  reference %d  model %2d %-8r head 0 shares %s" % (f[5], f[0], f[1] + 1, f[2] + 1, f[4], f[3]))
