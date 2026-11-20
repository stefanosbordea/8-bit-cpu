"""No model run. Direct logit attribution on the reference traces (all 200 tasks): the residual stream before the
final norm is  l0.xf + Wo_1 o / 256 + W2_1 g / 256, and the logit of token t is (gain) * x . E[t] / 2^sh_l.
For every body position: how much does each part (layer-0 stream, the 4 heads of layer 1, each of the 219
feed-forward units of layer 1) add to the margin  logit(reference) - logit(best other allowed token)?"""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, D
from lsllm.edit.session import Session

s = Session("edits/s4")
E, Wo, W2 = s.get("E"), s.get("Wo_1"), s.get("W2_1")
A, rows, tokens = load(["y", "l1.xf", "l1.xa", "l1.g", "l1.o", "l0.xf"])
pos, ref, top, alt = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
print("body positions", len(rows), "accuracy %.4f" % (ref == top).mean())
d = (E[ref] - E[alt]).astype(float)                                       # margin direction per position
x, y = A["l1.xf"][pos].astype(float), A["y"][pos].astype(float)
gain = np.linalg.norm(y, axis=1) / np.linalg.norm(x, axis=1) / 2 ** s.spec["shifts"]["sh_l"]   # codes per unit of x . E
margin = A["u"][pos, ref] - A["u"][pos, alt]
print("margin from y: corr with real margin %.3f" % np.corrcoef((y * d).sum(1) / 512, margin)[0, 1])
base = (A["l0.xf"][pos] * d).sum(1) * gain
o, g = A["l1.o"][pos].astype(float), A["l1.g"][pos].astype(float)
heads = np.stack([((o[:, 40 * h:40 * h + 40] @ Wo[:, 40 * h:40 * h + 40].T) * d).sum(1) / 256 * gain for h in range(4)], 1)
units = g * (d @ W2) / 256 * gain[:, None]                                # (positions, 219)
ok = ref == top
print("mean margin %.2f codes; layer-0 stream %.2f; heads %s; all units %.2f" % (margin.mean(), base.mean(), heads.mean(0).round(2), units.sum(1).mean()))
print("  at wrong positions: margin %.2f; layer-0 %.2f; heads %s; units %.2f" % (margin[~ok].mean(), base[~ok].mean(), heads[~ok].mean(0).round(2), units[~ok].sum(1).mean()))
tot, bad, good = units.sum(0), units[~ok].sum(0), units[ok].sum(0)
active = (g > 0).mean(0)
order = np.argsort(tot)
print("\nfeed-forward units of layer 1 with the most negative total margin contribution (codes summed over %d positions):" % len(rows))
for k in order[:25]:
    print("  unit %3d total %8.1f  at wrong %8.1f  at right %8.1f  active %.2f  |W2 col| %.0f" % (k, tot[k], bad[k], good[k], active[k], np.abs(W2[:, k]).sum()))
print("most positive:")
for k in order[-10:]:
    print("  unit %3d total %8.1f  at wrong %8.1f  at right %8.1f  active %.2f" % (k, tot[k], bad[k], good[k], active[k]))
print("units with negative total: %d, their sum %.0f; sum of all %.0f" % ((tot < 0).sum(), tot[tot < 0].sum(), tot.sum()))
# split halves by task number (even / odd) to see whether a unit's sign is stable
even = rows[:, 0] % 2 == 0
te, to_ = units[even].sum(0), units[~even].sum(0)
print("correlation of unit totals between even and odd tasks: %.3f" % np.corrcoef(te, to_)[0, 1])
print("negative in both halves:", [int(k) for k in order if te[k] < 0 and to_[k] < 0])
np.savez(D + "attribution.npz", units=units, heads=heads, base=base, margin=margin, ok=ok)
