"""No model run. From trace_ref.npz: at every body position of the reference solutions, the (syntax-filtered)
greedy token against the reference token. Which tokens are systematically too high or too low?"""
import json
from collections import Counter
import numpy as np
from lsllm.filter_table import TableFilter
from lsllm.tok import Tok

tok = Tok("data/tok/f512v2/tok.model")
filt = TableFilter(tok)
tasks = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
z = np.load("edits/s4/trace_ref.npz")
u, tokens, ends = z["u"].astype(int), z["tokens"], np.cumsum(z["lengths"])
P = lambda t: repr(tok.pieces[t]) if t != 2 else "EOS"
rows = []          # (task, body index, position in the trace, reference token, greedy token, u ref, u greedy, rank)
for n, end in zip(z["tasks"], ends):
    start = end - z["lengths"][list(ends).index(end)]
    ids = tokens[start:end].tolist()
    n_prompt = len(tok.encode(tasks[n]["prompt"]))
    state = filt.start_ids(ids[:n_prompt])
    for g in range(n_prompt, len(ids)):
        logits = np.where(filt.allowed(state), u[start + g - 1], -1)
        ref, top = ids[g], int(logits.argmax())
        rows.append((n, g - n_prompt, start + g - 1, ref, top, logits[ref], logits[top], int((logits > logits[ref]).sum()), len(ids) - 1 - g))
        filt.step(state, ref)
rows = np.array(rows)
np.save("edits/s4/trace_ref_rows.npy", rows)
print("body positions:", len(rows), " greedy = reference: %.3f" % (rows[:, 3] == rows[:, 4]).mean(), " reference allowed:", (rows[:, 5] >= 0).mean())
for b in range(6):
    r = rows[rows[:, 1] == b]
    print("body token %d: acc %.2f  ref %s  greedy %s" % (b, (r[:, 3] == r[:, 4]).mean(),
          Counter(P(t) for t in r[:, 3]).most_common(6), Counter(P(t) for t in r[:, 4]).most_common(6)))
wrong = rows[rows[:, 3] != rows[:, 4]]
print("\nmost common (reference <- greedy instead), with mean logit gap in codes (4 codes = 1 nat):")
pairs = Counter((r[3], r[4]) for r in wrong)
for (ref, top), c in pairs.most_common(40):
    gap = np.mean([r[6] - r[5] for r in wrong if r[3] == ref and r[4] == top])
    print("  %3d  %-10s <- %-10s gap %.1f" % (c, P(ref), P(top), gap))
print("\nper token: times reference / times greedy / greedy when wrong / missed")
ref_c, top_c, fp, fn = Counter(rows[:, 3]), Counter(rows[:, 4]), Counter(wrong[:, 4]), Counter(wrong[:, 3])
for t in sorted(set(ref_c) | set(top_c), key=lambda t: -(fp[t] + fn[t]))[:40]:
    print("  %-10s ref %3d greedy %3d  wrongly chosen %3d  missed %3d" % (P(t), ref_c[t], top_c[t], fp[t], fn[t]))
last = rows[rows[:, 8] == 0]
print("\nend of text at the reference end: greedy is EOS in %d of %d; mean gap %.1f; ranks %s" % (
    (last[:, 4] == 2).sum(), len(last), (last[:, 6] - last[:, 5]).mean(), sorted(last[:, 7].tolist())))
print("greedy tokens there:", Counter(P(t) for t in last[:, 4]).most_common(12))
early = rows[(rows[:, 4] == 2) & (rows[:, 3] != 2)]
print("EOS chosen too early: %d times, reference then was %s" % (len(early), Counter(P(t) for t in early[:, 3]).most_common(10)))
