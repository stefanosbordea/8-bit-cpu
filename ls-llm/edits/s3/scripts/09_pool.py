"""No budget: the completions of every run for the tasks that passed in at least one run."""
import glob
from collections import Counter
from common import DIR, completions, practice

tasks = practice()
runs = sorted(int(p[-8:-6]) for p in glob.glob(DIR + "/eval_*.jsonl"))
rows = {run: completions(run) for run in runs}
pool = sorted({g for run in runs for g, r in enumerate(rows[run]) if r["outcome"] == "pass"})
print("passes per task over %d runs:" % len(runs), Counter(tasks[g]["entry_point"] for run in runs for g in pool if rows[run][g]["outcome"] == "pass").most_common())
for g in pool:
    print("=====", tasks[g]["entry_point"], "|", tasks[g]["prompt"].split('"""')[1])
    seen = {}
    for run in runs:
        seen.setdefault((rows[run][g]["outcome"], rows[run][g]["completion"].strip()), []).append(run)
    for (outcome, text), where in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        print("  %-6s runs %s: %s" % (outcome, where, text[:110]))
