"""No model run. Attribution on a re-trace (python 17_attribution_tag.py <tag>): accuracy at the reference
tokens, margin contribution of every layer-1 feed-forward unit and head dimension (all / decisive positions,
even / odd tasks), token over- and under-prediction. Uses the CURRENT weights of the session (must be the state
the trace was made with). Saves attribution_<tag>.npz."""
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, D
from lsllm.edit.session import Session

tag = sys.argv[1]
s = Session("edits/s4")
E, Wo, W2 = s.get("E"), s.get("Wo_1"), s.get("W2_1")
A, rows, tokens = load(["y", "l1.xf", "l1.g", "l1.o", "l0.xf"], tag)
pos, ref, top, alt = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
ok = ref == top
even = rows[:, 0] % 2 == 0
print("body positions %d, greedy = reference %.4f (even tasks %.4f, odd %.4f)" % (len(rows), ok.mean(), ok[even].mean(), ok[~even].mean()))
first_error = [rows[rows[:, 0] == n][:, 3:5] for n in range(200)]
print("tasks with every reference token greedy: %d" % sum((r[:, 0] == r[:, 1]).all() for r in first_error))
d = (E[ref] - E[alt]).astype(float)
x, y = A["l1.xf"][pos].astype(float), A["y"][pos].astype(float)
gain = np.linalg.norm(y, axis=1) / np.linalg.norm(x, axis=1) / 512
margin = A["u"][pos, ref] - A["u"][pos, alt]
near = np.abs(margin) <= 8
units = A["l1.g"][pos] * (d @ W2) / 256 * gain[:, None]
dims = A["l1.o"][pos] * (d @ Wo) / 256 * gain[:, None]
base = (A["l0.xf"][pos] * d).sum(1) * gain
print("mean margin %.2f; decisive positions %d; layer-0 stream %.2f, heads %s, units %.2f" % (margin.mean(), near.sum(), base.mean(), dims.reshape(len(dims), 4, 40).sum(2).mean(0).round(2), units.sum(1).mean()))
out = E @ W2
for name, part, write in (("unit", units, out), ("head dim", dims, E @ Wo)):
    dec, de, do = part[near].sum(0), part[near & even].sum(0), part[near & ~even].sum(0)
    ae, ao = part[even].sum(0), part[~even].sum(0)
    print("\n%s: most negative at decisive positions (dec even odd | all even odd | writes | suppresses)" % name)
    for k in np.argsort(dec)[:22]:
        hi, lo = np.argsort(-write[:, k])[:5], np.argsort(write[:, k])[:5]
        print("  %3d  %7.1f %6.1f %6.1f | %7.1f %7.1f | %s | %s" % (k, dec[k], de[k], do[k], ae[k], ao[k], " ".join(repr(P(t)) for t in hi), " ".join(repr(P(t)) for t in lo)))
    print("  negative in all four:", [int(k) for k in np.argsort(dec) if max(de[k], do[k], ae[k], ao[k]) < 0])
fp, fn, rc = Counter(top[~ok]), Counter(ref[~ok]), Counter(ref)
print("\ntoken: reference count / wrongly chosen / missed")
for t in sorted(set(fp) | set(fn), key=lambda t: -abs(fp[t] - fn[t]))[:30]:
    print("  %-9r ref %3d  wrongly chosen %3d  missed %3d" % (P(t), rc[t], fp[t], fn[t]))
np.savez(D + "attribution_%s.npz" % tag, units=units, dims=dims, base=base, margin=margin, ok=ok)
