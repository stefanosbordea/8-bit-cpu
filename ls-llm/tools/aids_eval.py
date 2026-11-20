"""Practice-set scores of a float checkpoint for several settings of the decoding aids (never run on test sets).

    PYTHONPATH=. .venv/bin/python tools/aids_eval.py runs/NAME/ckpt.pt [custom|mbpp]
"""
import re
import sys
from collections import Counter

from lsllm.bench import evaluate, load_tasks
from lsllm.model.completer import TorchCompleter

SETTINGS = [(0.0, 0), (1.0, 0), (2.0, 0), (3.0, 0), (0.0, 4), (2.0, 4), (3.0, 4), (4.0, 4)]
tasks = load_tasks(sys.argv[2] if len(sys.argv) > 2 else "custom", "practice")
completer = TorchCompleter(sys.argv[1], syntax_filter=True)
print("prompt bonus | repeat limit | pass@1 | syntax ok | runs ok | tests passed | NameError | repeating")
for bias, no_repeat in SETTINGS:
    completer.prompt_bias, completer.no_repeat = bias, no_repeat
    report = evaluate(completer, tasks, 1, 0.0, 96)
    errors, credit, repeats = Counter(), 0.0, 0
    for sample in report.samples:
        if sample["outcome"] == "error":
            errors[sample["detail"].split("(")[0].replace("error: ", "")] += 1
        elif sample["outcome"] == "pass":
            credit += 1
        elif sample["outcome"] == "wrong":
            failed, total = map(int, re.match(r"(\d+) of (\d+)", sample["detail"]).groups())
            credit += 1 - failed / total
        words = sample["completion"].split()
        grams = Counter(tuple(words[g:g + 3]) for g in range(len(words) - 2))
        repeats += bool(grams) and max(grams.values()) >= 3
    m = report.metrics
    print("%4.1f | %d | %.3f | %.3f | %.3f | %.4f | %d | %.3f" % (bias, no_repeat, m["pass@1"], m["syntax_ok"], m["runs_ok"],
                                                           credit / len(tasks), errors["NameError"], repeats / len(tasks)), flush=True)
