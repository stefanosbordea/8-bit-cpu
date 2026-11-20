"""Weights only (no model run): which attention head COPIES? For head h of layer l the path
token t (embedding E[t]) -> Wv -> the head's 40 columns of Wo -> output matrix E gives a V x V table C[j, t]:
how much attending to token t raises the logit of token j. A copying head has a large positive diagonal.
Also the same for the direct path (embedding -> logits) and the feed-forward layers' share of the stream.
Run from the repository root: PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/07_copy_heads.py"""
import numpy as np

from common import DIR, tokenizer
from lsllm.edit.session import Session

s = Session(DIR)
tok = tokenizer(s)
E = s.get("E").astype(np.float64)
En = E / np.linalg.norm(E, axis=1, keepdims=True)
names = [t for t in range(512) if tok.pieces[t].strip() and tok.pieces[t].strip().isidentifier()]
print("vocabulary rows that are identifier pieces:", len(names))
for l in (0, 1):
    Wv, Wo = s.get("Wv_%d" % l).astype(np.float64), s.get("Wo_%d" % l).astype(np.float64)
    print("layer %d: |Wo| per head block %s, largest |Wo| %d" % (
        l, [round(float(np.linalg.norm(Wo[:, 40 * h:40 * h + 40])), 1) for h in range(4)], int(np.abs(Wo).max())))
    for h in range(4):
        C = En @ Wo[:, 40 * h:40 * h + 40] @ Wv @ En.T
        diag, off = np.diag(C), C[~np.eye(512, dtype=bool)]
        rank = (C > np.diag(C)[None, :]).sum(0)          # per attended token t: how many tokens j beat t itself
        print("   head %d: diagonal mean %8.1f (identifier pieces %8.1f)  all entries mean %6.1f std %7.1f  -> z %.2f; "
              "tokens that are their own top-5 output: %.0f%%" % (
                  h, diag.mean(), diag[names].mean(), off.mean(), off.std(), (diag.mean() - off.mean()) / off.std(),
                  100 * (rank < 5).mean()))
