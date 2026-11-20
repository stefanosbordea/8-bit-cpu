"""Knob (c): the table NORM[site] scaled by a factor (rounded, clamped to 1..127). Arguments site:factor ...
Sites 0 and 2 feed q, k, v of layer 0 and 1 (scores scale with factor^2, values with factor);
sites 1 and 3 feed the feed-forward layers (ReLU without bias: the layer writes factor times as much);
site 4 feeds the logits: there c_l is lowered so that the median top logit (code 73 in the trace) stays where it is."""
import sys
import numpy as np
from common import open_session, trial

s, tok = open_session()
for site, factor in [(int(a.split(":")[0]), float(a.split(":")[1])) for a in sys.argv[1:]]:
    table = s.get("NORM")
    table[site] = np.clip(np.floor(table[site] * factor + 0.5), 1, 127)
    edits = [("NORM", table, "NORM[%d] times %.2f" % (site, factor))]
    if site == 4:
        c_l = s.spec["preloads"]["c_l"]
        new = int(round(c_l - (factor - 1) * 73 * 2 ** s.spec["shifts"]["sh_l"]))
        edits.append((("preloads", "c_l"), new, "c_l %d -> %d: keeps the median top logit at code 73 under NORM[4] times %.2f" % (c_l, new, factor)))
    trial(s, edits, "NORM[%d] x %.2f" % (site, factor))
