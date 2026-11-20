"""No budget: the docstring also names the arguments ("Return a minus b"), often as the token with the trailing
space that the body needs. Does the input of layer 1 know that a docstring token is an argument mention?
(1) where the mentions lie along the signature - docstring direction of script 13;
(2) the direction (mentions - other docstring tokens) and its separation, split in two halves of the tasks:
    means from one half, separation measured on the other."""
import sys
import numpy as np
sys.path.insert(0, "edits/s2/scripts")
from lib import load
s, tok, tasks, data = load("trace_refs")
_, _, _, own = load("trace_own")
open_id, close_id, comma = tok.sp.piece_to_id("("), tok.sp.piece_to_id("):"), tok.sp.piece_to_id(",▁")
base = lambda t: tok.pieces[t].strip(" .,")
men, oth, grp_m, grp_o = [], [], [], []
seen = set()
for d_ in (data, own):
    for (ids, n_prompt, g), tr in zip(d_["seqs"], d_["trace"]):
        if g in seen:
            continue
        seen.add(g)
        ids = list(ids); lo, hi = ids.index(open_id), ids.index(close_id)
        # arguments as strings, and docstring tokens that start a mention of one (whole word match on the text)
        args, cur = [], ""
        for t in ids[lo + 1:hi + 1]:
            if t in (comma, close_id):
                args.append(cur); cur = ""
            else:
                cur += tok.pieces[t]
        for p in range(hi + 3, n_prompt - 1):
            hit = False
            for a in args:
                text, k = "", p
                while k < n_prompt - 1 and len(text) < len(a):
                    text += tok.pieces[ids[k]]; k += 1
                before = tok.pieces[ids[p - 1]][-1:] if p > hi + 3 else " "
                if text.rstrip(" .,") == a and (text != a or not tok.pieces[ids[k]][:1].isalnum()) and not before.isalnum() and (len(text) == len(a) or not text[len(a)].isalnum()):
                    hit = True
            (men if hit else oth).append(tr["l1.ya"][p]); (grp_m if hit else grp_o).append(g)
men, oth, grp_m, grp_o = np.array(men, float), np.array(oth, float), np.array(grp_m), np.array(grp_o)
mean = np.load("edits/s2/regions.npz")
d = mean["sig"] - mean["doc"]; d /= np.linalg.norm(d)
print("%d mentions, %d other docstring tokens" % (len(men), len(oth)))
print("along signature - docstring: mentions %.1f +- %.1f, other docstring tokens %.1f +- %.1f (signature 94, docstring -18)" % ((men @ d).mean(), (men @ d).std(), (oth @ d).mean(), (oth @ d).std()))
for half in (0, 1):
    a, b = grp_m % 2 == half, grp_o % 2 == half
    v = men[a].mean(0) - oth[b].mean(0); v /= np.linalg.norm(v)
    print("direction from tasks with number %% 2 == %d, measured on the others: mentions %.1f +- %.1f, other %.1f +- %.1f" % (
        half, (men[~a] @ v).mean(), (men[~a] @ v).std(), (oth[~b] @ v).mean(), (oth[~b] @ v).std()))
v = men.mean(0) - oth.mean(0)
np.savez("edits/s2/mention.npz", mean_men=men.mean(0), mean_oth=oth.mean(0))
