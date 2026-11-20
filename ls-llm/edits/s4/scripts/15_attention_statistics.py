"""No model run. From the traces: per head, the largest score code (saturation at 127?) and the effective number
of attended keys dsum / 127, at body positions; and the EXP tables."""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, D
from lsllm.edit.session import Session

s = Session("edits/s4")
print("EXP[0] at distance 1..12:", s.get("EXP")[0][255:243:-1].tolist())
print("EXP[1] at distance 1..12:", s.get("EXP")[1][255:243:-1].tolist())
files = [np.load(D + "trace_ref_b%d.npz" % k) for k in range(3)]
for l in range(2):
    mx = np.concatenate([z["l%d.max" % l] for z in files]).reshape(-1, 4)
    ds = np.concatenate([z["l%d.dsum" % l] for z in files]).reshape(-1, 4) / 127
    print("layer %d: max code mean %s, share at 127 %s, share below 40 %s" % (l, mx.mean(0).round(1), (mx == 127).mean(0).round(3), (mx < 40).mean(0).round(3)))
    print("         keys attended (dsum/127): median %s mean %s; share with < 1.5 keys %s" % (np.median(ds, 0).round(2), ds.mean(0).round(2), (ds < 1.5).mean(0).round(2)))
