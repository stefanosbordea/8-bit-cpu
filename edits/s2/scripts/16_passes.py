"""No budget: which tasks pass in which evaluate run."""
import glob, json
tasks = [json.loads(line) for line in open("data/bench/custom_practice.jsonl")]
notes = {r["run"]: r["note"] for r in json.load(open("edits/s2/budget.json"))["runs"]}
seen = {}
for path in sorted(glob.glob("edits/s2/eval_*.jsonl")):
    run = int(path[-8:-6])
    got = [g for g, line in enumerate(open(path)) if json.loads(line)["outcome"] == "pass"]
    for g in got:
        seen.setdefault(g, []).append(run)
    print("run %2d %-40s passes: %s" % (run, notes[run][:40], " ".join("%03d" % g for g in got)))
for g in sorted(seen):
    print("%03d in runs %s: %s" % (g, seen[g], tasks[g]["canonical_solution"][len(tasks[g]["prompt"]):].strip().replace("\n", "\\n")[:60]))
