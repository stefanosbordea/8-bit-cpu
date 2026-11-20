"""No budget: statistics of the trace of run 2 for every hardware knob.
The score codes are rebuilt from the traced rotated queries and keys (the trace cannot return them: their length grows)."""
import numpy as np
from common import open_session, practice

np.set_printoptions(linewidth=220, precision=2, suppress=True)
s, tok = open_session()
t = np.load("edits/s3/trace_02.npz")
lengths, tokens = t["lengths"], t["tokens"]
starts = np.concatenate([[0], np.cumsum(lengths)])
tasks = practice()
# where the body starts in every sequence (after the prompt)
body = np.zeros(len(tokens), bool)
for n, g in enumerate(t["used"]):
    p = len(tok.encode(tasks[g]["prompt"], bos=True))
    body[starts[n] + p - 1:starts[n + 1] - 1] = True       # positions whose NEXT token is a body token
print("positions", len(tokens), "of which predict a body token", body.sum())

print("\n== norm sites: index m into NORM (0..127), table value r")
for n, name in enumerate(["l0.ya", "l0.yf", "l1.ya", "l1.yf", "y"]):
    m, r = t[name + ".m"].ravel(), t[name + ".r"].ravel()
    y = t[name]
    print("site %d %-6s m: min %3d p5 %3d med %3d p95 %3d max %3d | m=127: %.3f | r=127 (table clamp): %.3f | r med %d | |y| mean %.1f, |y|>=127: %.4f"
          % (n, name, m.min(), np.percentile(m, 5), np.median(m), np.percentile(m, 95), m.max(), (m == 127).mean(),
             (r == 127).mean(), np.median(r), np.abs(y).mean(), (np.abs(y) >= 127).mean()))

print("\n== residual stream saturation (|x| >= 127)")
for name in ["x0", "l0.xa", "l0.xf", "l1.xa", "l1.xf"]:
    x = t[name]
    print("%-6s |x| mean %.1f  rms %.1f  saturated %.4f" % (name, np.abs(x).mean(), np.sqrt((x.astype(float) ** 2).mean()), (np.abs(x) >= 127).mean()))
for l in range(2):
    at = "l%d." % l
    for name in ("q", "k", "v", "a", "o", "g"):
        x = t[at + name]
        print("%-6s |.| mean %.1f  max %d  at the clamp %.4f  zero %.3f" % (at + name, np.abs(x).mean(), np.abs(x).max(), ((x >= 127) | (x <= -128)).mean(), (x == 0).mean()))

print("\n== attention: row maximum code, sum of weights (dsum / 127 = number of keys at full weight), per head")
G, d = 1, 40
for l in range(2):
    at = "l%d." % l
    mx, ds = t[at + "max"].reshape(-1, 4), t[at + "dsum"].reshape(-1, 4)
    for h in range(4):
        print("layer %d head %d  max code: p5 %3d med %3d p95 %3d  =127: %.3f  <23: %.3f | dsum/127: med %.2f p95 %.2f max %.1f"
              % (l, h, np.percentile(mx[:, h], 5), np.median(mx[:, h]), np.percentile(mx[:, h], 95), (mx[:, h] == 127).mean(),
                 (mx[:, h] < 23).mean(), np.median(ds[:, h]) / 127, np.percentile(ds[:, h], 95) / 127, ds[:, h].max() / 127))

print("\n== score codes rebuilt from qr and kr: share of keys at code 0, at 127, several keys at 127, weight entropy")
exp_tab = s.get("EXP")
stats = {}
for l in range(2):
    c_s, sh = np.array(s.spec["preloads"]["c_s"][l]), s.spec["shifts"]["sh_s"][l]
    qr, kr = t["l%d.qr" % l].reshape(-1, 4, d), t["l%d.kr" % l].reshape(-1, d)
    zero, top, ties, neff, check, n, wself, raw_all = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), 0, 0, np.zeros(4), [[] for h in range(4)]
    for q in range(len(lengths)):
        for p in range(starts[q], starts[q + 1]):
            keys = kr[starts[q]:p + 1]
            raw = qr[p] @ keys.T + c_s[:, None]                 # (4, keys)
            code = np.clip(np.maximum(raw >> sh, 0), -128, 127)
            m = code.max(-1, keepdims=True)
            check += (m.ravel() != t["l%d.max" % l][p].ravel()).sum()
            e = exp_tab[l][(code - m) & 255]
            w = e / e.sum(-1, keepdims=True)
            if p - starts[q] >= 8:
                zero += (code == 0).mean(-1); top += (code == 127).mean(-1); ties += (code == 127).sum(-1) > 1
                neff += np.exp(-(w * np.log(np.maximum(w, 1e-9))).sum(-1)); wself += w[:, -1]; n += 1
                for h in range(4):
                    raw_all[h].append((raw[h] - raw[h].max()) / 2.0 ** sh)
    print("layer %d: rebuilt maximum differs from the traced one at %d places" % (l, check))
    for h in range(4):
        gaps = np.concatenate(raw_all[h])
        print("  head %d keys at code 0: %.3f  at 127: %.4f  rows with >1 key at 127: %.3f  effective keys %.2f  weight on self %.2f | keys within 4 codes of max %.3f, within 23: %.3f"
              % (h, zero[h] / n, top[h] / n, ties[h] / n, neff[h] / n, wself[h] / n, (gaps > -4).mean(), (gaps > -23).mean()))

print("\n== logits")
u, acc = t["u"], t["u.acc"]
top = u.max(-1)
nxt = np.concatenate([tokens[1:], [0]])
last = np.zeros(len(tokens), bool); last[starts[1:] - 1] = True
ok = body & ~last
print("top logit: p5 %d med %d p95 %d  =127: %.3f  | tokens at 0: %.3f" % (np.percentile(top, 5), np.median(top), np.percentile(top, 95), (top == 127).mean(), (u == 0).mean()))
tied = (u == top[:, None]).sum(-1)
print("positions with a tie at the top: all %.3f, body %.3f; mean tied tokens %.2f" % ((tied > 1).mean(), (tied[ok] > 1).mean(), tied.mean()))
pred = u.argmax(-1)
print("teacher-forced greedy accuracy (no filter): all %.3f body %.3f" % ((pred[~last] == nxt[~last]).mean(), (pred[ok] == nxt[ok]).mean()))
ref_u = u[np.arange(len(u)), nxt]
print("body: reference token is among the tied top %.3f, strictly the argmax %.3f" % ((ref_u[ok] == top[ok]).mean(), (pred[ok] == nxt[ok]).mean()))
gap = top - ref_u
print("body: gap top - reference logit (codes, 4 per nat): ", np.percentile(gap[ok], [25, 50, 75, 90]))
lg = u / 4.0; lg = lg - lg.max(-1, keepdims=True); lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
print("teacher-forced loss: all %.3f body %.3f" % (-lp[np.arange(len(u)), nxt][~last].mean(), -lp[np.arange(len(u)), nxt][ok].mean()))
