"""Runs 3-6: knob (a), the attention table of one layer rebuilt with a smaller or larger kappa (export: 4 codes per nat).
Hypothesis: attention is flat (6 to 12 effective keys per head in the trace); a sharper table may copy names of the
prompt better (38 NameErrors at the baseline), a flatter one is the control."""
import sys
from common import exp_table, open_session, trial

s, tok = open_session()
assert (exp_table(4.0) == s.get("EXP")[0]).all() and (exp_table(4.0) == s.get("EXP")[1]).all(), "the export formula"
for l, kappa in [(int(a.split(":")[0]), float(a.split(":")[1])) for a in sys.argv[1:]]:
    table = s.get("EXP")
    table[l] = exp_table(kappa)
    trial(s, [("EXP", table, "EXP[%d] rebuilt with kappa %.2f (was 4)" % (l, kappa))], "EXP[%d] kappa %.2f" % (l, kappa))
