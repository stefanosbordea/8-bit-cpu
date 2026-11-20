"""Baseline at the same memory budget: a pruned token n-gram model (orders 1 to 6, longest-match backoff).

Storage model used for the budget: one record per kept n-gram = (n - 1) context tokens and the next token at
9 bits each plus an 8-bit count, rounded up to whole bytes: 3, 4, 5, 6, 7, 8 bytes for orders 1..6.
The count threshold is the smallest one for which all kept records fit the budget.

    .venv/bin/python -m lsllm.baselines.ngram data/tok/f512a 450000 runs/base/ngram.pkl
"""
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

from lsllm.model.completer import LINE_BREAKS
from lsllm.tok import Tok

BYTES = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}


def train(data_dir, budget_bytes, out):
    ids = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16).tolist()
    bos = Tok(os.path.join(data_dir, "tok.model")).bos_id
    counts = {n: Counter() for n in BYTES}
    for n in BYTES:
        for g in range(len(ids) - n + 1):
            gram = tuple(ids[g:g + n])
            if bos not in gram[1:]:              # never continue across the start of a document
                counts[n][gram] += 1
    lo, hi = 1, 1 << 20
    while lo < hi:
        mid = (lo + hi) // 2
        size = BYTES[1] * len(counts[1]) + sum(BYTES[n] * sum(1 for c in counts[n].values() if c >= mid) for n in BYTES if n > 1)
        lo, hi = (lo, mid) if size <= budget_bytes else (mid + 1, hi)
    table, used = defaultdict(dict), BYTES[1] * len(counts[1])
    for n in BYTES:
        for gram, c in counts[n].items():
            if n == 1 or c >= lo:
                table[gram[:-1]][gram[-1]] = c
                used += BYTES[n] if n > 1 else 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pickle.dump({"table": dict(table), "tok": data_dir, "bytes": used, "threshold": lo}, open(out, "wb"))
    print(json.dumps({"ngrams": sum(len(v) for v in table.values()), "bytes": used, "count_threshold": lo, "budget": budget_bytes}))


class NgramCompleter:
    def __init__(self, path):
        saved = pickle.load(open(path, "rb"))
        self.table, self.truncated = saved["table"], 0
        self.tok = Tok(os.path.join(saved["tok"], "tok.model"))
        self.name = "ngram-%dKB" % round(saved["bytes"] / 1000)
        self.info = {"params": sum(len(v) for v in self.table.values()), "variant": "n-gram", "filter": "off",
                     "notes": "%d bytes by the storage model in lsllm/baselines/ngram.py; params = number of n-grams" % saved["bytes"]}

    def distribution(self, ids):
        for n in range(5, -1, -1):
            context = tuple(ids[len(ids) - n:]) if n else ()
            if len(context) == n and context in self.table:
                return self.table[context]
        return {self.tok.eos_id: 1}

    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        tok = self.tok
        prompt_ids = tok.encode(prompt, bos=True, eos=False)
        rng = torch.Generator().manual_seed(seed)
        texts = []
        for _ in range(n if temperature > 0 else 1):
            ids, level, answer = list(prompt_ids), tok.level_after(prompt_ids), []
            for _ in range(max_new_tokens):
                dist = self.distribution(ids)
                choices = sorted(dist)
                if temperature > 0:
                    p = torch.tensor([float(dist[t]) for t in choices]) ** (1.0 / temperature)
                    t = choices[int(torch.multinomial(p / p.sum(), 1, generator=rng))]
                else:
                    t = max(choices, key=lambda c: (dist[c], -c))
                level = tok.level_after([t], level)
                if t == tok.eos_id or (level == 0 and any(m in tok.pieces[t] for m in LINE_BREAKS)):
                    break
                answer.append(t)
                ids.append(t)
            texts.append(tok.decode(prompt_ids + answer)[len(prompt):])
        return texts * (n if temperature == 0 else 1)


if __name__ == "__main__":
    train(sys.argv[1], int(sys.argv[2]), sys.argv[3])
