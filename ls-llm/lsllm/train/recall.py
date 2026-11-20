"""Request texts as aligned training rows and as a recall check (board models only, never benchmark models).

A request on the board starts the context: <s> def name( and the model writes the rest. A row here is exactly
that: <s>, the text, </s>; the tokens of the request and the padding behind </s> do not count in the loss.
`check` says for every text whether greedy decoding reproduces it: at every position behind the opening
parenthesis the wanted token must beat every other token (by one code or more for integer logits). The syntax
filter of the board only removes other tokens, so a text that passes here is written exactly on the board.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F


class Recall:
    def __init__(self, paths, tok, T):
        texts = {}
        for path in paths:
            with open(path) as f:
                for row in map(json.loads, f):
                    texts.setdefault(row["name"], row["text"])      # the first file that has a name decides its text
        self.names = list(texts)
        rows, mask = np.full((len(texts), T + 1), tok.eos_id, np.int64), np.zeros((len(texts), T), np.float32)
        self.lengths = np.zeros(len(texts), np.int64)
        for g, (name, text) in enumerate(texts.items()):
            prompt = tok.encode("def %s(" % name)
            ids = tok.encode_request("def %s(" % name, text)       # the tokens of the request, then the tokens of the rest
            assert len(ids) <= T + 1, name + ": longer than the context"
            rows[g, :len(ids)] = ids
            mask[g, len(prompt) - 1:len(ids) - 1] = 1.0             # target j is token j + 1
            self.lengths[g] = len(ids)
        self.rows, self.mask = torch.from_numpy(rows), torch.from_numpy(mask)
        self.failing = np.ones(len(texts), bool)
        self.order = np.argsort(self.lengths, kind="stable")        # texts from short to long
        self.rank = np.argsort(self.order)                          # place of every text in that order

    def batch(self, n, rng, boost):
        """n rows cut to the longest text among them, and their loss mask. A failing text is `1 + boost` times as
        likely. One text is drawn first, the others among the 2n texts nearest to it in length: rows of similar
        length need little padding (the median text has 75 tokens, the longest 228)."""
        p = 1.0 + boost * self.failing
        first = rng.choice(len(p), p=p / p.sum())
        low = max(0, min(int(self.rank[first]) - n, len(p) - 2 * n))
        near = self.order[low:low + 2 * n]
        pick = rng.choice(near, size=min(n, len(near)), replace=False, p=p[near] / p[near].sum())
        longest = int(self.lengths[pick].max())
        return self.rows[pick, :longest], self.mask[pick, :longest - 1]

    @staticmethod
    def loss(logits, rows, mask):
        """Mean cross-entropy over the tokens the model has to write; logits are those of rows[:, :-1]."""
        each = F.cross_entropy(logits.reshape(-1, logits.size(-1)), rows[:, 1:].reshape(-1), reduction="none")
        return (each * mask.reshape(-1)).sum() / mask.sum()

    @torch.no_grad()
    def check(self, logits_of, device, least=0.0, batch_size=32):
        """Smallest margin (wanted logit minus best other logit) of every text; marks a text as failing if it is
        not above `least` (0 for float logits; for integer logits pass 0 as well: a margin of one code is above it)."""
        margins = []
        for g in range(0, len(self.names), batch_size):
            longest = int(self.lengths[g:g + batch_size].max())
            rows, mask = self.rows[g:g + batch_size, :longest].to(device), self.mask[g:g + batch_size, :longest - 1].to(device)
            logits, want = logits_of(rows[:, :-1]), rows[:, 1:, None]
            margin = logits.gather(-1, want).squeeze(-1) - logits.scatter(-1, want, float("-inf")).max(-1).values
            margins.append(torch.where(mask > 0, margin, torch.full_like(margin, float("inf"))).min(1).values.cpu())
        margins = torch.cat(margins).numpy()
        self.failing = ~(margins > least)
        return margins
