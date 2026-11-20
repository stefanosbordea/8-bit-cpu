"""No budget: compare the completions of two evaluate runs: python 10_compare.py A B [all]"""
import json, re, sys
a, b = int(sys.argv[1]), int(sys.argv[2])
tasks = [json.loads(line) for line in open("data/bench/custom_practice.jsonl")]
def credit(s):
    if s["outcome"] == "pass":
        return 1.0
    if s["outcome"] == "wrong":
        failed, total = map(int, re.match(r"(\d+) of (\d+)", s["detail"]).groups())
        return 1 - failed / total
    return 0.0
A = [json.loads(l) for l in open("edits/s2/eval_%02d.jsonl" % a)]
B = [json.loads(l) for l in open("edits/s2/eval_%02d.jsonl" % b)]
print("same completion: %d of 200; credit up in %d, down in %d" % (sum(x["completion"] == y["completion"] for x, y in zip(A, B)),
      sum(credit(y) > credit(x) for x, y in zip(A, B)), sum(credit(y) < credit(x) for x, y in zip(A, B))))
for t, x, y in zip(tasks, A, B):
    if credit(x) != credit(y) or (len(sys.argv) > 3 and x["completion"] != y["completion"]):
        print("%s %.2f -> %.2f | %s || %s || REF %s" % (t["task_id"][-3:], credit(x), credit(y), x["completion"].strip()[:70], y["completion"].strip()[:70],
              t["canonical_solution"][len(t["prompt"]):].strip()[:50].replace("\n", "\\n")))
