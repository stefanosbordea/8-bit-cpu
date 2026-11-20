"""No model run. The argument a: in the signature it is the token 'a' (434), in a body `a < b` it is 'a ' (67),
which is also the article of every docstring. Is 'a ' written too rarely, what wins instead, and which part of
the last layer decides? (state e6; weights only for the similarities)"""
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, D, tok
from lsllm.edit.session import Session

s = Session("edits/s4")
E, Wo, W2 = s.get("E").astype(float), s.get("Wo_1").astype(float), s.get("W2_1").astype(float)
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
pairs = [(t, tok.pieces.index(tok.pieces[t] + " ")) for t in range(429, 512) if tok.pieces[t] + " " in tok.pieces]
print("letter / letter+space: cosine of the E rows, norms")
for a, b in pairs:
    print("  %r %r  cos %.2f  |E| %.0f %.0f" % (P(a), P(b), cos(E[a], E[b]), np.linalg.norm(E[a]), np.linalg.norm(E[b])))
A, rows, tokens = load(["y", "l1.xf", "l1.g", "l1.o", "l0.xf"], "e6")
pos, ref, top = rows[:, 2], rows[:, 3], rows[:, 4]
u = A["u"]
for t in (67, 449, 434, 33, 207, 315):
    m = ref == t
    print("\nreference %r: %d positions, greedy right %d; greedy instead: %s" % (P(t), m.sum(), (top[m] == t).sum(), Counter(P(x) for x in top[m & (top != t)]).most_common(8)))
m = (ref == 67) & (top != 67)
x, y = A["l1.xf"][pos], A["y"][pos]
gain = np.linalg.norm(y, axis=1) / np.linalg.norm(x, axis=1) / 512
d = E[ref] - E[top]
base = (A["l0.xf"][pos] * d).sum(1) * gain
heads = np.stack([((A["l1.o"][pos][:, 40 * h:40 * h + 40] @ Wo[:, 40 * h:40 * h + 40].T) * d).sum(1) / 256 * gain for h in range(4)], 1)
units = (A["l1.g"][pos] * (d @ W2)).sum(1) / 256 * gain
print("\n'a ' missed (%d positions): mean logit gap %.1f = layer-0 stream %.1f + heads %s + units %.1f" % (m.sum(), (u[pos, ref] - u[pos, top])[m].mean(), base[m].mean(), heads[m].mean(0).round(1), units[m].mean()))
for r in np.flatnonzero(m)[:26]:
    p = pos[r]
    print("   ...%-36s greedy %-6r gap %3d | l0 %5.1f heads %s units %5.1f" % ("".join(P(t) for t in tokens[max(0, p - 6):p + 1]).replace("\n", "\\n")[-36:], P(top[r]),
          u[p, ref[r]] - u[p, top[r]], base[r], heads[r].round(1), units[r]))
