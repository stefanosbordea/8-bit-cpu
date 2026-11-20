"""No model run. Induction positions: body positions where the pair (previous token, reference token) already
stands in the prompt (the second piece of an argument name, such as 'ite' -> 'ms'). There the model should copy;
38 baseline answers die of a NameError (item for items, day for days). Which part of the last layer carries
the copy, and how right is the model there? (state e6)"""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, D, tok, tasks
from lsllm.edit.session import Session

s = Session("edits/s4")
E, Wo, W2 = s.get("E").astype(float), s.get("Wo_1").astype(float), s.get("W2_1").astype(float)
A, rows, tokens = load(["y", "l1.xf", "l1.g", "l1.o", "l0.xf"], "e6")
pos, ref, top, alt = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
prompt_pairs = {}
for n in range(200):
    ids = tok.encode(tasks[n]["prompt"])
    prompt_pairs[n] = set(zip(ids, ids[1:]))
prev = tokens[pos]
letters = np.array([P(t)[:1].isalpha() or P(t)[:1] == "_" for t in range(512)])
ind = np.array([(int(a), int(b)) in prompt_pairs[n] for n, a, b in zip(rows[:, 0], prev, ref)]) & letters[prev] & letters[ref]
first = np.array([any(b == q for _, q in prompt_pairs[n]) for n, b in zip(rows[:, 0], ref)]) & letters[ref] & ~ind
ok = ref == top
x, y = A["l1.xf"][pos], A["y"][pos]
gain = np.linalg.norm(y, axis=1) / np.linalg.norm(x, axis=1) / 512
d = E[ref] - E[alt]
base = (A["l0.xf"][pos] * d).sum(1) * gain
heads = np.stack([((A["l1.o"][pos][:, 40 * h:40 * h + 40] @ Wo[:, 40 * h:40 * h + 40].T) * d).sum(1) / 256 * gain for h in range(4)], 1)
units = (A["l1.g"][pos] * (d @ W2)).sum(1) / 256 * gain
margin = A["u"][pos, ref] - A["u"][pos, alt]
for label, m in (("induction (2nd piece of a prompt word)", ind), ("word piece that is in the prompt, not after its prompt neighbour", first), ("all other", ~ind & ~first)):
    print("%-65s n %4d  right %.3f  margin %6.2f = layer-0 %6.2f + heads %s + units %6.2f" % (label, m.sum(), ok[m].mean(), margin[m].mean(), base[m].mean(), heads[m].mean(0).round(2), units[m].mean()))
    w = m & ~ok
    print("%-65s n %4d              margin %6.2f = layer-0 %6.2f + heads %s + units %6.2f" % ("   of these wrong", w.sum(), margin[w].mean(), base[w].mean(), heads[w].mean(0).round(2), units[w].mean()))
from collections import Counter
w = ind & ~ok
print("wrong induction positions: (previous, reference <- greedy)", Counter((P(a), P(b), P(c)) for a, b, c in zip(prev[w], ref[w], top[w])).most_common(25))
