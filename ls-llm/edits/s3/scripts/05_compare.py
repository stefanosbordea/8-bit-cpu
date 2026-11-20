"""No budget: which tasks pass in which run, and how many completions differ from the baseline run."""
import sys
from common import completions, practice

tasks = practice()
base = completions(1)
runs = [int(a) for a in sys.argv[1:] if a != "-v"]
for run in runs:
    rows = completions(run)
    passed = [g for g, r in enumerate(rows) if r["outcome"] == "pass"]
    changed = sum(r["completion"] != b["completion"] for r, b in zip(rows, base))
    print("run %2d: passes %s  completions that differ from run 1: %d" % (run, [tasks[g]["entry_point"] for g in passed], changed))
    if "-v" in sys.argv:
        for g in passed:
            print("   ", tasks[g]["entry_point"], "|", rows[g]["completion"].strip())
