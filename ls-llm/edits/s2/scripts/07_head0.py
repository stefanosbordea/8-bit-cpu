"""No budget: layer 1 head 0 (the copy head) in numbers: sizes of its weights, its queries and score codes on the
trace, the EXP table, and the tokens that the feed-forward units of layer 1 push when they vote against copying."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load, attention
s, tok, tasks, data = load()
A = {k: v.astype(np.int64) for k, v in s.arrays.items()}
for l in (0, 1):
    print("EXP[%d] first values:" % l, A["EXP"][l][0], A["EXP"][l][255:215:-1].tolist())
    for h in range(4):
        print("layer %d head %d: max|Wo col| %d  rms Wo %.2f   max|Wq row| %d  rms Wq %.2f" % (
            l, h, np.abs(A["Wo_%d" % l][:, h * 40:h * 40 + 40]).max(), np.sqrt((A["Wo_%d" % l][:, h * 40:h * 40 + 40] ** 2).mean()),
            np.abs(A["Wq_%d" % l][h * 40:h * 40 + 40]).max(), np.sqrt((A["Wq_%d" % l][h * 40:h * 40 + 40] ** 2).mean())))
qs, tops, os_, xs, lows = [], [], [], [], []
for (ids, n_prompt, g), tr in zip(data["seqs"], data["trace"]):
    e, code = attention(s, tr, 1)
    qs.append(np.abs(tr["l1.qr"][n_prompt:, 0]).reshape(-1, 4, 40)); tops.append(tr["l1.max"][n_prompt:, 0, :, 0])
    os_.append(np.abs(tr["l1.o"][n_prompt:]).reshape(-1, 4, 40)); xs.append(np.abs(tr["l1.xf"][n_prompt:]))
    lows.append(np.array([[(code[p, h, :p + 1] == 0).mean() for h in range(4)] for p in range(n_prompt, len(ids))]))
qs, tops, os_, xs, lows = map(np.concatenate, (qs, tops, os_, xs, lows))
for h in range(4):
    print("layer 1 head %d on the body: |q| mean %.1f max %d, share of |q| >= 100: %.4f; row maximum mean %.1f, at 127: %.3f; keys at code 0: %.2f; |o| mean %.1f max %d" % (
        h, qs[:, h].mean(), qs[:, h].max(), (qs[:, h] >= 100).mean(), tops[:, h].mean(), (tops[:, h] == 127).mean(), lows[:, h].mean(), os_[:, h].mean(), os_[:, h].max()))
print("final residual |x|: mean %.1f, share at the int8 limit %.4f" % (xs.mean(), (xs >= 127).mean()))
push = A["W2_1"].T @ A["E"].T          # (units, tokens): what a unit of layer 1 adds to every logit (up to the norm gain)
for k in (209, 3, 93, 214, 191, 54, 168, 79, 102, 104, 90, 159):
    order = np.argsort(push[k])
    print("unit %3d  up: %s   down: %s" % (k, " ".join(repr(tok.pieces[t]) for t in order[::-1][:10]), " ".join(repr(tok.pieces[t]) for t in order[:8])))
