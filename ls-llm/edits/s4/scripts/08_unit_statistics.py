"""No model run. Per feed-forward unit of layer 1 (and per head dimension): margin contribution summed over all
body positions, over the decisive ones (|margin| <= 8 codes = 2 nats), in both halves of the tasks; and which
tokens the unit writes (top entries of E @ W2[:, k])."""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, D
from lsllm.edit.session import Session

s = Session("edits/s4")
E, Wo, W2 = s.get("E"), s.get("Wo_1"), s.get("W2_1")
A, rows, tokens = load(["y", "l1.xf", "l1.g", "l1.o"])
z = np.load(D + "attribution.npz")
units, margin, ok = z["units"], z["margin"], z["ok"]
near = np.abs(margin) <= 8
even = rows[:, 0] % 2 == 0
print("positions %d, decisive %d (of them right %d)" % (len(rows), near.sum(), (near & ok).sum()))
tot, dec = units.sum(0), units[near].sum(0)
de, do = units[near & even].sum(0), units[near & ~even].sum(0)
print("corr of decisive totals between halves %.3f" % np.corrcoef(de, do)[0, 1])
order = np.argsort(dec)
out = E @ W2                                                             # (tokens, units): what a unit writes
for k in list(order[:30]) + list(order[-8:]):
    hi, lo = np.argsort(-out[:, k])[:6], np.argsort(out[:, k])[:6]
    print("unit %3d decisive %7.1f (even %6.1f odd %6.1f) all %7.1f | up: %s | down: %s" % (
        k, dec[k], de[k], do[k], tot[k], " ".join(repr(P(t)) for t in hi), " ".join(repr(P(t)) for t in lo)))
print("\nunits negative on decisive positions in both halves:", [int(k) for k in order if de[k] < 0 and do[k] < 0])
print("units negative on decisive AND on all positions in both halves:",
      [int(k) for k in order if de[k] < 0 and do[k] < 0 and units[even].sum(0)[k] < 0 and units[~even].sum(0)[k] < 0])
# head dimensions of layer 1
o = A["l1.o"][rows[:, 2]].astype(float)
x, y = A["l1.xf"][rows[:, 2]].astype(float), A["y"][rows[:, 2]].astype(float)
gain = np.linalg.norm(y, axis=1) / np.linalg.norm(x, axis=1) / 512
d = (E[rows[:, 3]] - E[rows[:, 5]]).astype(float)
dims = o * (d @ Wo) / 256 * gain[:, None]
np.save(D + "attribution_headdims.npy", dims)
for h in range(4):
    part = dims[:, 40 * h:40 * h + 40]
    print("head %d: all %.0f decisive %.0f wrong %.0f | most negative dims (all): %s" % (h, part.sum(), part[near].sum(), part[~ok].sum(),
          [(int(40 * h + j), round(float(part[:, j].sum()))) for j in np.argsort(part.sum(0))[:5]]))
