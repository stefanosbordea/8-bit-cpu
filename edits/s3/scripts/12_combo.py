"""Several knobs in one trial. Each argument is one trial: knobs joined by "+", a knob is
exp:LAYER:KAPPA, norm:SITE:FACTOR or recip:FACTOR (the same edits as scripts 04, 07, 08, applied to the current model).
--keep keeps the edits of the (single) trial."""
import sys
import numpy as np
from common import exp_table, open_session, trial

s, tok = open_session()
keep = "--keep" in sys.argv
for arg in [a for a in sys.argv[1:] if a != "--keep"]:
    exp, norm, recip, reasons = s.get("EXP"), s.get("NORM"), s.get("RECIP"), {"EXP": [], "NORM": [], "RECIP": []}
    for knob in arg.split("+"):
        kind, *rest = knob.split(":")
        if kind == "exp":
            exp[int(rest[0])] = exp_table(float(rest[1]))
            reasons["EXP"].append("EXP[%s] rebuilt with kappa %s" % tuple(rest))
        elif kind == "norm":
            site, factor = int(rest[0]), float(rest[1])
            norm[site] = np.clip(np.floor(norm[site] * factor + 0.5), 1, 127)
            reasons["NORM"].append("NORM[%d] times %s" % (site, rest[1]))
        else:
            recip[64:128] = np.clip(np.floor(recip[64:128] * float(rest[0]) + 0.5), 1, 127)
            reasons["RECIP"].append("RECIP[64..127] times %s" % rest[0])
    edits = [(name, value, "; ".join(reasons[name])) for name, value in (("EXP", exp), ("NORM", norm), ("RECIP", recip)) if reasons[name]]
    trial(s, edits, arg, keep=keep)
