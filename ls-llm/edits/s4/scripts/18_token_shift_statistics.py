"""No model run. For a token t and a logit shift b (codes), from the traced logits of state <tag>: how many
reference positions have t wrongly on top by less than |b| (would be released if t is lowered), how many have t
correctly on top with a margin of at most |b| (would be lost), and the same for raising t.
python 18_token_shift_statistics.py <tag> <token id> ..."""
import sys
import numpy as np
sys.path.insert(0, "edits/s4/scripts")
from lib import load, P, filt, tok

tag = sys.argv[1]
A, rows, tokens = load([], tag)
u = A["u"]
pos, ref, top, alt = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
for t in map(int, sys.argv[2:]):
    ut = u[pos, t]
    lowered_good = (top == t) & (ref != t)                 # t wrongly on top: gap to the reference
    gap_ref = ut - u[pos, ref]
    lowered_bad = (top == t) & (ref == t)                  # t correctly on top: margin to the runner-up
    margin = ut - u[pos, alt]
    raised_good = (ref == t) & (top != t)                  # t missed: gap to the top
    gap_top = u[pos, top] - ut
    print("token %r: wrongly on top %d, correctly on top %d, missed %d" % (P(t), lowered_good.sum(), lowered_bad.sum(), raised_good.sum()))
    for b in (1, 2, 3, 4, 6, 8, 12):
        # lowering by b: the reference wins only if it was the runner-up among the others
        second = np.where(np.arange(512)[None] == t, -1, u[pos]).max(1)       # best other token, ignoring the filter (upper bound of what is released)
        released = (lowered_good & (gap_ref <= b - 1) & (u[pos, ref] >= second)).sum()
        lost = (lowered_bad & (margin <= b)).sum()
        gained = (raised_good & (gap_top <= b - 1)).sum()
        # raising t by b breaks positions where another token is the right top and t (allowed) is within b
        t_allowed_close = (ref != t) & (top == ref) & (u[pos, ref] - ut <= b) & (ut >= 0)
        print("   shift %2d: lower -> fixes %3d, breaks %3d | raise -> fixes %3d, breaks <= %3d" % (b, released, lost, gained, t_allowed_close.sum()))
