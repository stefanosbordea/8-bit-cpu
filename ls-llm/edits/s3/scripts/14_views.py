"""No budget: the read-only views that were first typed inline during the session (saved afterwards for the record).
  tables            the EXP, NORM and RECIP tables of the model
  sample RUN        every 7th completion of a run next to the prompt and the reference
  chains RUN        completions of a run that contain and / or / else, with the reference
  names RUN ...     the names of the NameErrors over the runs, and the E rows of the chain tokens
  diff RUN_A RUN_B  the first 30 completions that differ between two runs"""
import re
import sys
from collections import Counter
import numpy as np
from common import completions, open_session, practice

np.set_printoptions(linewidth=250)
s, tok = open_session()
tasks = practice()
view, runs = sys.argv[1], [int(a) for a in sys.argv[2:]]
flat = lambda text: text.strip().replace("\n", " | ")
if view == "tables":
    for l in range(2):
        print("EXP[%d] by distance d:" % l, [int(s.arrays["EXP"][l][(-d) & 255]) for d in range(128)])
    for n in range(5):
        print("NORM[%d]:" % n, s.arrays["NORM"][n][:128].tolist())
    print("RECIP[60..129]:", s.arrays["RECIP"][60:130].tolist())
elif view == "sample":
    rows = completions(runs[0])
    for g in range(0, 200, 7):
        print("=====", g, rows[g]["outcome"], (rows[g]["detail"] or "")[:100])
        print(tasks[g]["prompt"]); print("--- completion:", rows[g]["completion"]); print("--- reference:", tasks[g]["canonical_solution"])
elif view == "chains":
    for task, row in zip(tasks, completions(runs[0])):
        if any(w in row["completion"] for w in (" and ", " or ", " else ")):
            print(row["outcome"][:5].ljust(5), "|", task["prompt"].split('"""')[1][:90])
            print("      got:", flat(row["completion"])[:120]); print("      ref:", flat(task["canonical_solution"][len(task["prompt"]):])[:120])
elif view == "names":
    names = Counter()
    for run in runs:
        for row in completions(run):
            m = re.search(r"NameError\(\"name '(\w+)'", row["detail"] or "")
            if m:
                names[m.group(1)] += 1
    print(names.most_common(40))
    E = s.get("E")
    for t in (135, 95, 201, 143, 71, 4, 13):
        print(t, repr(tok.pieces[t]), "row norm %.1f" % np.linalg.norm(E[t]))
    print("median row norm %.1f" % np.median(np.linalg.norm(E, axis=1)))
elif view == "diff":
    a, b = completions(runs[0]), completions(runs[1])
    shown = 0
    for task, x, y in zip(tasks, a, b):
        if x["completion"] != y["completion"] and shown < 30:
            shown += 1
            print(task["entry_point"], "|", x["outcome"], "->", y["outcome"]); print("   was:", flat(x["completion"])[:100]); print("   now:", flat(y["completion"])[:100])
    print("bodies with ' and ':", sum(" and " in r["completion"] for r in a), "->", sum(" and " in r["completion"] for r in b))
