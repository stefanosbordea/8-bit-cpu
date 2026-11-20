"""Knobs (d) and (b). Arguments: recip:FACTOR (RECIP[64..127] scaled, clamped to 127: every head output of both layers
is written FACTOR times as strongly) or cs:LAYER:CODES (c_s of every head of the layer moved by CODES score codes)."""
import sys
import numpy as np
from common import open_session, trial

s, tok = open_session()
for arg in sys.argv[1:]:
    kind, *rest = arg.split(":")
    if kind == "recip":
        factor = float(rest[0])
        table = s.get("RECIP")
        table[64:128] = np.clip(np.floor(table[64:128] * factor + 0.5), 1, 127)
        trial(s, [("RECIP", table, "RECIP[64..127] times %.2f" % factor)], "RECIP x %.2f" % factor)
    else:
        layers, codes = ([0, 1] if rest[0] == "all" else [int(rest[0])]), int(rest[1])
        edits = []
        for l in layers:
            for h in range(4):
                old = s.spec["preloads"]["c_s"][l][h]
                edits.append((("preloads", "c_s", l, h), old + codes * 2 ** s.spec["shifts"]["sh_s"][l],
                              "c_s[%d][%d] moved by %d codes" % (l, h, codes)))
        trial(s, edits, "c_s layer %s moved by %d codes" % (rest[0], codes))
