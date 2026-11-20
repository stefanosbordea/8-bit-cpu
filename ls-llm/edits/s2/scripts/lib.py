"""Helpers for the analysis scripts: no model run here, only numpy on traced values and on the weights."""
import json, pickle
import numpy as np
from lsllm.edit.session import Session
from lsllm.tok import Tok

def load(name="trace_refs"):
    s = Session("edits/s2")
    tok = Tok(s.spec["tokenizer"] + "/tok.model")
    tasks = [json.loads(line) for line in open("data/bench/custom_practice.jsonl")]
    data = pickle.load(open("edits/s2/%s.pkl" % name, "rb"))
    return s, tok, tasks, data

def attention(s, tr, l):
    """The weights e (positions, heads, keys) of layer l, rebuilt from the traced rotated queries and keys
    (SPEC C.6 steps 1 to 3), checked against the traced row maximum and weight sum."""
    q, k = tr["l%d.qr" % l][:, 0].astype(np.int64), tr["l%d.kr" % l][:, 0].astype(np.int64)     # (P, 4, 40), (P, 40)
    c_s = np.array(s.spec["preloads"]["c_s"][l])[None, :, None]
    code = np.clip((q @ k.T + c_s) >> s.spec["shifts"]["sh_s"][l], 0, 127)                         # (P, 4, P)
    P = len(k)
    mask = np.tril(np.ones((P, P), dtype=bool))[:, None, :]
    code = np.where(mask, code, -1)
    top = code.max(-1, keepdims=True)
    e = np.where(mask, s.arrays["EXP"][l].astype(np.int64)[(code - top) & 255], 0)
    assert (top[:, :, 0] == tr["l%d.max" % l][:, 0, :, 0]).all() and (e.sum(-1) == tr["l%d.dsum" % l][:, 0, :, 0]).all()
    return e, code
