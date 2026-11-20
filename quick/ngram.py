"""Baseline at the same memory budget: a pruned token n-gram model (orders 1..6, longest-match backoff).

Storage model for the budget: one record per kept n-gram = (n - 1) context tokens + next token at 9 bits each
+ an 8-bit count, rounded up to whole bytes: 3, 4, 5, 6, 7, 8 bytes for orders 1..6.
"""
import json
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import sentencepiece as spm
import torch

from quick.markers import DED, IND, from_markers, level_after, to_markers

BYTES = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}


def train(data_dir, budget_bytes, out):
    ids = np.fromfile(data_dir + "/train.bin", dtype=np.uint16).tolist()
    counts = {n: Counter() for n in BYTES}
    for n in BYTES:
        for g in range(len(ids) - n + 1):
            gram = tuple(ids[g:g + n])
            if 1 not in gram[1:]:               # never continue across a document start
                counts[n][gram] += 1
    lo, hi = 1, 1 << 20
    while lo < hi:                               # smallest count threshold that fits the budget
        mid = (lo + hi) // 2
        size = sum(BYTES[n] * sum(1 for c in counts[n].values() if c >= mid) for n in BYTES if n > 1) + BYTES[1] * len(counts[1])
        lo, hi = (lo, mid) if size <= budget_bytes else (mid + 1, hi)
    table = defaultdict(dict)
    used = BYTES[1] * len(counts[1])
    for n in BYTES:
        for gram, c in counts[n].items():
            if n == 1 or c >= lo:
                table[gram[:-1]][gram[-1]] = c
                used += BYTES[n] if n > 1 else 0
    pickle.dump({"table": dict(table), "tok": data_dir, "bytes": used, "threshold": lo}, open(out, "wb"))
    print(json.dumps({"ngrams": sum(len(v) for v in table.values()), "bytes": used, "count_threshold": lo, "budget": budget_bytes}))


class NgramCompleter:
    def __init__(self, path):
        d = pickle.load(open(path, "rb"))
        self.table, self.name, self.truncated = d["table"], path, 0
        self.sp = spm.SentencePieceProcessor(model_file=d["tok"] + "/tok.model")
        self.delta = {}
        for g in range(self.sp.get_piece_size()):
            piece = self.sp.id_to_piece(g)
            self.delta[g] = 1 if piece == IND else -(DED.index(piece) + 1) if piece in DED else 0

    def dist(self, ids):
        for n in range(5, -1, -1):
            ctx = tuple(ids[len(ids) - n:]) if n else ()
            if len(ctx) == n and ctx in self.table:
                return self.table[ctx]
        return {2: 1}

    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        marked = to_markers(prompt)
        start = level_after(marked)
        gen = torch.Generator().manual_seed(seed)
        outs = []
        for _ in range(1 if temperature == 0 else n):
            ids, level, new = [1] + self.sp.encode(marked), start, []
            for _ in range(max_new_tokens):
                d = self.dist(ids)
                toks = sorted(d)
                if temperature == 0:
                    tok = max(toks, key=lambda t: (d[t], -t))
                else:
                    p = torch.tensor([float(d[t]) for t in toks]) ** (1.0 / temperature)
                    tok = toks[int(torch.multinomial(p / p.sum(), 1, generator=gen))]
                level += self.delta[tok]
                if tok == 2 or (self.delta[tok] < 0 and level <= 0):
                    break
                new.append(tok)
                ids.append(tok)
            outs.append(from_markers(self.sp.decode(new), level=start))
        return outs * n if temperature == 0 else outs


if __name__ == "__main__":
    train(sys.argv[1], int(sys.argv[2]), sys.argv[3])
