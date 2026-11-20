"""Near-copies of benchmark solutions: a training function that contains at least half of the 8-token runs of the
body of a benchmark solution. The exact overlap rules (same name, same code up to renaming, same docstring;
lsllm/data/corpus/overlap.py) do not catch a function that differs from a solution in one or two tokens.

    drop = NearCopies()                  # reads every benchmark task (HumanEval, MBPP, custom; test and practice)
    drop(text)                           # True if `text` is a near-copy of some solution

Python tokens with their names are compared; comments, docstrings and layout are ignored, and so is the signature
line (it is given in the prompt). A solution body of fewer than 11 tokens has fewer than 4 runs and is not
checked: any code shares such a body, and the exact rules already cover it.
"""
import io
import tokenize
from collections import defaultdict

N, LEAST_RUNS, SHARE = 8, 4, 0.5
SKIP = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)


def runs(text):
    """The set of runs of N consecutive Python tokens of the body of the function `text`."""
    body = "def f():\n" + (text.split("\n", 1)[1] if "\n" in text else "")
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(body).readline):
            if tok.type in SKIP or tok.type == tokenize.STRING and tok.string.startswith(('"""', "'''")):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set()
    return {tuple(out[g:g + N]) for g in range(len(out) - N + 1)}


class NearCopies:
    def __init__(self, tasks=None):
        if tasks is None:
            from lsllm.bench import load_tasks
            tasks = [task for bench, split in (("humaneval", "test"), ("mbpp", "test"), ("mbpp", "practice"), ("custom", "test"),
                                               ("custom", "practice")) for task in load_tasks(bench, split)]
        self.size, self.where = {}, defaultdict(set)        # runs per solution; run -> the solutions that have it
        for task in tasks:
            mine = runs(task.canonical_solution)
            if len(mine) >= LEAST_RUNS:
                self.size[task.task_id] = len(mine)
                for run in mine:
                    self.where[run].add(task.task_id)

    def match(self, text):
        """(task id, share of its runs found in `text`) of the solution that `text` covers best, or (None, 0.0)."""
        hits = defaultdict(int)
        for run in runs(text):
            for task_id in self.where.get(run, ()):
                hits[task_id] += 1
        if not hits:
            return None, 0.0
        best = max(hits, key=lambda task_id: hits[task_id] / self.size[task_id])
        return best, hits[best] / self.size[best]

    def __call__(self, text):
        return self.match(text)[1] >= SHARE
