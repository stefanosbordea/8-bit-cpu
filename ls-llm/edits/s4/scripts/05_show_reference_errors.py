"""No model run. Prints every traced reference body with the positions where the filtered greedy token differs:
reference token, its logit, and the top three allowed tokens with logits."""
import json
import numpy as np
from lsllm.filter_table import TableFilter
from lsllm.tok import Tok

tok = Tok("data/tok/f512v2/tok.model")
filt = TableFilter(tok)
tasks = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
z = np.load("edits/s4/trace_ref.npz")
u, tokens, ends = z["u"].astype(int), z["tokens"], np.cumsum(z["lengths"])
P = lambda t: tok.pieces[t] if t != 2 else "EOS"
for n, end, length in zip(z["tasks"], ends, z["lengths"]):
    start = end - length
    ids = tokens[start:end].tolist()
    n_prompt = len(tok.encode(tasks[n]["prompt"]))
    state = filt.start_ids(ids[:n_prompt])
    print("\n#%d %s | %s" % (n, tasks[n]["meta"]["family"], tasks[n]["prompt"].split('"""')[1][:110]))
    line = []
    for g in range(n_prompt, len(ids)):
        logits = np.where(filt.allowed(state), u[start + g - 1], -1)
        ref = ids[g]
        top = np.argsort(-logits, kind="stable")[:3]
        if top[0] == ref:
            line.append(P(ref))
        else:
            line.append("{%s %d | %s}" % (P(ref), logits[ref], " ".join("%s %d" % (P(t), logits[t]) for t in top)))
        filt.step(state, ref)
    print("  " + "·".join(line).replace("\n", "\\n"))
