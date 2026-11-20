"""Print the completions of an evaluate run: python 02_show.py NN [outcome]"""
import json, sys
run = int(sys.argv[1]); want = sys.argv[2] if len(sys.argv) > 2 else None
tasks = [json.loads(line) for line in open("data/bench/custom_practice.jsonl")]
for task, line in zip(tasks, open("edits/s2/eval_%02d.jsonl" % run)):
    s = json.loads(line)
    if want and s["outcome"] != want and want not in s["detail"]:
        continue
    print("=== %s [%s] %s | %s" % (task["task_id"][-3:], task["meta"]["family"], s["outcome"], s["detail"][:90]))
    print(task["prompt"] + "\x1b[0m>>>" + s["completion"])
    print("   REF:", task["canonical_solution"][len(task["prompt"]):].strip().replace("\n", " \\n "))
