"""No model run. The answers of one evaluate run next to the reference solutions (the inline listings of the
session, saved as a script). python show_completions.py <run, e.g. 01> [first task] [last task]"""
import json
import sys
run = sys.argv[1]
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 199)
S = [json.loads(l) for l in open("edits/s4/eval_%s.jsonl" % run)]
T = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
for n, (s, t) in enumerate(zip(S, T)):
    if lo <= n <= hi:
        print(n, t["meta"]["family"], s["outcome"], s["detail"][:40], "|", t["prompt"].split('"""')[1][:90], "|", repr(s["completion"][5:]),
              "| REF", repr(t["canonical_solution"][len(t["prompt"]) + 5:]))
