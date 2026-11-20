"""No model run. Compare two evaluate outputs: python 10_compare_evals.py 01 06 [show]"""
import json, sys
sys.path.insert(0, "edits/s4/scripts")
from run_edit import common_prefix, D
a, b = sys.argv[1], sys.argv[2]
tasks = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
refs = [t["canonical_solution"][len(t["prompt"]):] for t in tasks]
for run in (a, b):
    S = [json.loads(l) for l in open(D + "eval_%s.jsonl" % run)]
    print(run, "exact %d prefix share %.4f" % (sum(x["completion"] == r for x, r in zip(S, refs)),
          sum(common_prefix(x["completion"], r) / len(r) for x, r in zip(S, refs)) / 200))
if len(sys.argv) > 3:
    A = [json.loads(l) for l in open(D + "eval_%s.jsonl" % a)]
    B = [json.loads(l) for l in open(D + "eval_%s.jsonl" % b)]
    for n, (x, y) in enumerate(zip(A, B)):
        if x["completion"] != y["completion"]:
            print(n, x["outcome"], "->", y["outcome"], "|", repr(x["completion"][5:]), "->", repr(y["completion"][5:]), "| REF", repr(refs[n][5:]))
