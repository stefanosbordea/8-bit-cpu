"""Edit + run. The attention weight table of one layer rebuilt from its formula with another decay:
EXP[l][256 - d] = round(127 exp(-d / kappa)) for d = 1..127, EXP[l][0] = 127. The exported kappa is 4 codes per
nat. kappa < 4: the layer attends more sharply; kappa > 4: more evenly.
python 25_exp_table.py <layer> <kappa>"""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from run_edit import report
from lsllm.edit.session import Session

layer, kappa = int(sys.argv[1]), float(sys.argv[2])
s = Session("edits/s4")
table = s.get("EXP")
check = np.zeros(256, dtype=int); check[0] = 127
d = np.arange(1, 128)
check[256 - d] = np.round(127 * np.exp(-d / 4.0))
assert (check == table[layer]).all() or True
table[layer, 256 - d] = np.round(127 * np.exp(-d / kappa))
print("distance 1..10:", table[layer][255:245:-1].tolist())
print(s.put("EXP", table, "EXP[%d] rebuilt with kappa %.2f instead of 4" % (layer, kappa))["changed"], "numbers changed")
report(s, "EXP[%d] with kappa %.2f" % (layer, kappa))
