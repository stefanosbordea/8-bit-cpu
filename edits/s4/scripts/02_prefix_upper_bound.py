"""No model run. (Cuts are by character, because answers that do not parse cannot be tokenised again.)
 For every baseline completion: would a shorter answer (cut after some token) pass the practice
tests? Upper bound of what "stop earlier" can gain. Runs the cut programs in the repository sandbox."""
import ast, json, sys
from lsllm.sandbox import run_many

run = sys.argv[1] if len(sys.argv) > 1 else "01"
samples = [json.loads(l) for l in open("edits/s4/eval_%s.jsonl" % run)]
tasks = [json.loads(l) for l in open("data/bench/custom_practice.jsonl")]
DRIVER = """
import json
try:
    exec(PROGRAM)
    out = []
    for t in TESTS:
        try:
            exec(t); out.append(1)
        except Exception:
            out.append(0)
except Exception:
    out = [0] * len(TESTS)
print("R " + json.dumps(out))
"""
jobs, codes = [], []
for n, (s, t) in enumerate(zip(samples, tasks)):
    seen, text = set(), s["completion"]
    for cut in range(2, len(text) + 1):          # cut after every character; only cuts that parse are tested
        program = t["prompt"] + text[:cut].rstrip()
        if program in seen:
            continue
        seen.add(program)
        try:
            ast.parse(program)
        except SyntaxError:
            continue
        jobs.append((n, cut, len(text), program[len(t["prompt"]):]))
        codes.append("PROGRAM = %r\nTESTS = %r\n" % (program, t["tests"]) + DRIVER)
results = run_many(codes, 4, 5.0, 512)
best = {}
for (n, cut, full, text), r in zip(jobs, results):
    line = [l for l in r.stdout.split("\n") if l.startswith("R ")]
    got = json.loads(line[-1][2:]) if line else [0]
    share = sum(got) / len(got)
    if share > best.get(n, (0,))[0] or n not in best:
        best[n] = (share, cut, full, text)
    if share == 1 and best[n][0] == 1 and cut < best[n][1]:
        best[n] = (share, cut, full, text)
passes = [n for n in best if best[n][0] == 1]
print("tasks where some cut passes:", len(passes))
for n in passes:
    share, cut, full, text = best[n]
    print(n, tasks[n]["meta"]["family"], samples[n]["outcome"], "cut %d of %d" % (cut, full), repr(text), "| full:", repr(samples[n]["completion"]))
json.dump({str(n): best[n][:3] for n in best}, open("edits/s4/prefix_%s.json" % run, "w"))
