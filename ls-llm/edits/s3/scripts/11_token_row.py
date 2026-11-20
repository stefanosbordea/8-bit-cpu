"""Outside the four knobs: rows of E scaled (a row is how likely the token is written; the embedding is tied, the
norm at site 0 rescales what is read). Arguments TOKEN_ID:FACTOR ... are applied together in ONE trial; --keep keeps them."""
import sys
import numpy as np
from common import open_session, trial

s, tok = open_session()
args = [a for a in sys.argv[1:] if a != "--keep"]
E = s.get("E")
notes = []
for token, factor in [(int(a.split(":")[0]), float(a.split(":")[1])) for a in args]:
    E[token] = np.floor(E[token] * factor + 0.5)
    notes.append("E[%d] (%r) x %.2f" % (token, tok.pieces[token], factor))
trial(s, [("E", E, "; ".join(notes))], "; ".join(notes), keep="--keep" in sys.argv)
