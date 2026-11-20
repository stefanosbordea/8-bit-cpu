"""Shared loading for the analysis scripts (no model run): the four reference traces as one table."""
import json
import numpy as np
from lsllm.filter_table import TableFilter
from lsllm.tok import Tok

D = "edits/s4/"
tok = Tok("data/tok/f512v2/tok.model")
filt = TableFilter(tok)
tasks = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
P = lambda t: tok.pieces[t] if t != 2 else "EOS"


def load(names, tag=None):
    """Arrays `names` of all traced positions, and rows: one per body position with columns
    task, body index, trace position (where the logits for this token are), reference token, greedy token,
    best allowed token that is not the reference, tokens left after this one."""
    names_ = ("trace_ref.npz", "trace_ref_b0.npz", "trace_ref_b1.npz", "trace_ref_b2.npz") if tag is None else ["trace_%s_%d.npz" % (tag, k) for k in range(4)]
    files = [np.load(D + f) for f in names_]
    arrays = {name: np.concatenate([z[name].reshape(len(z[name]), -1) for z in files if name in z.files]).astype(np.int64) for name in names + ["u"]}
    tokens = np.concatenate([z["tokens"] for z in files])
    lengths = np.concatenate([z["lengths"] for z in files])
    numbers = np.concatenate([z["tasks"] for z in files])
    rows, start = [], 0
    for n, length in zip(numbers, lengths):
        ids = tokens[start:start + length].tolist()
        n_prompt = len(tok.encode(tasks[n]["prompt"]))
        state = filt.start_ids(ids[:n_prompt])
        for g in range(n_prompt, length):
            allowed = np.array(filt.allowed(state))
            logits = np.where(allowed, arrays["u"][start + g - 1], -1)
            ref = ids[g]
            other = logits.copy(); other[ref] = -1
            rows.append((n, g - n_prompt, start + g - 1, ref, int(logits.argmax()), int(other.argmax()), length - 1 - g))
            filt.step(state, ref)
        start += length
    return arrays, np.array(rows), tokens
