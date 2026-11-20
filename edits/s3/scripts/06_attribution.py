"""No budget: direct logit attribution on the trace of run 2 (weights and traced activations only).
The final residual stream is x0 + Wo_0 o0 + W2_0 g0 + Wo_1 o1 + W2_1 g1 (each /256, up to rounding), and the greedy
choice is the largest E @ x. For body positions: how much does every part add to the margin
(reference token minus the best other token), and what would the choice be if a part of the LAST layer were
written s times as strongly (exact up to rounding for layer 1, because nothing reads it but the output)."""
import numpy as np
from common import open_session, practice

np.set_printoptions(linewidth=220, precision=3, suppress=True)
s, tok = open_session()
t = np.load("edits/s3/trace_02.npz")
lengths, tokens = t["lengths"], t["tokens"]
starts = np.concatenate([[0], np.cumsum(lengths)])
tasks = practice()
body = np.zeros(len(tokens), bool)
for n, g in enumerate(t["used"]):
    p = len(tok.encode(tasks[g]["prompt"], bos=True))
    body[starts[n] + p - 1:starts[n + 1] - 1] = True
nxt = np.concatenate([tokens[1:], [0]])
E = s.get("E").astype(float)
parts = {"x0": t["x0"].astype(float),
         "attn0": t["l0.o"].reshape(len(tokens), -1) @ s.get("Wo_0").T / 256.0,
         "ffn0": t["l0.g"] @ s.get("W2_0").T / 256.0,
         "attn1": t["l1.o"].reshape(len(tokens), -1) @ s.get("Wo_1").T / 256.0,
         "ffn1": t["l1.g"] @ s.get("W2_1").T / 256.0}
total = sum(parts.values())
print("rounding check: |sum of parts - l1.xf| mean %.2f, |l1.xf| mean %.2f" % (np.abs(total - t["l1.xf"]).mean(), np.abs(t["l1.xf"]).mean()))
rows = np.flatnonzero(body)
ref = nxt[rows]


def accuracy(x):
    return ((x[rows] @ E.T).argmax(-1) == ref).mean()


print("body positions %d; accuracy of argmax E @ l1.xf %.4f, of E @ sum of parts %.4f, of traced u %.4f"
      % (len(rows), accuracy(t["l1.xf"].astype(float)), accuracy(total), (t["u"][rows].argmax(-1) == ref).mean()))
logit = total[rows] @ E.T
other = logit.copy(); other[np.arange(len(rows)), ref] = -1e9
rival = other.argmax(-1)
direction = E[ref] - E[rival]
right = logit[np.arange(len(rows)), ref] > other.max(-1)
print("\nmean contribution to the margin (reference - best rival), in units of E @ x:")
print("%-6s %10s %10s %10s" % ("part", "all", "when right", "when wrong"))
for name, x in parts.items():
    m = (x[rows] * direction).sum(-1)
    print("%-6s %10.1f %10.1f %10.1f   share of wrong positions where the part favours the reference: %.2f" % (name, m.mean(), m[right].mean(), m[~right].mean(), (m[~right] > 0).mean()))
print("\nwhat-if accuracy on body positions when a layer-1 part is scaled:")
for name in ("attn1", "ffn1"):
    print(name, " ".join("s=%.2f: %.4f" % (sc, accuracy(total + (sc - 1) * parts[name])) for sc in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)))
print("both layer-1 parts (= the embedding and layer 0 weaker):", " ".join("s=%.2f: %.4f" % (sc, accuracy(total + (sc - 1) * (parts["attn1"] + parts["ffn1"]))) for sc in (0.8, 0.9, 1.1, 1.2, 1.35, 1.5)))
# the wrong choices: is the rival a name-like token, is the reference a token of the prompt
kinds = {"ref is in the prompt": [], "rival is in the prompt": []}
seq_of = np.searchsorted(starts, rows, side="right") - 1
for n, p in enumerate(rows):
    q = seq_of[n]
    g = t["used"][q]
    prompt = set(tok.encode(tasks[g]["prompt"], bos=True))
    kinds["ref is in the prompt"].append(ref[n] in prompt)
    kinds["rival is in the prompt"].append(rival[n] in prompt)
for name, v in kinds.items():
    v = np.array(v)
    print("%s: all %.2f, at wrong positions %.2f" % (name, v.mean(), v[~right].mean()))
inp = np.array(kinds["ref is in the prompt"])
print("accuracy when the reference token is in the prompt %.3f (n %d), when it is not %.3f (n %d)" % (right[inp].mean(), inp.sum(), right[~inp].mean(), (~inp).sum()))
