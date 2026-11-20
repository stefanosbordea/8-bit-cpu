"""Near-copies of benchmark solutions in a training set: for every benchmark task, the training function that
contains the largest share of the solution body's 8-token runs (Python tokens, names kept).

A function that shares most runs of a solution is a near-copy that the exact rules (same name, same code up to
renaming, same docstring) do not catch. The script lists them; `--write FILE` saves the texts of the functions
above the threshold so that a build can leave them out.

    PYTHONPATH=. .venv/bin/python tools/contamination.py FILE_WITH_TEXTS.jsonl [--threshold 0.5] [--write FILE]
"""
import argparse
import io
import json
import tokenize
from collections import defaultdict

from lsllm.bench import load_tasks

N = 8


def runs(source):
    """The set of runs of N consecutive Python tokens of `source` (comments, docstrings and layout removed)."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
                continue
            if tok.type == tokenize.STRING and tok.string.startswith(('"""', "'''")):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set()
    return {tuple(out[g:g + N]) for g in range(len(out) - N + 1)}


def body(text):
    """The text after the signature line (the signature is given in the prompt and is not what a copy is about)."""
    return text.split("\n", 1)[1] if "\n" in text else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("texts")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--write")
    args = p.parse_args()
    texts = list(dict.fromkeys(json.loads(line)["text"] for line in open(args.texts)))
    index = defaultdict(set)
    for g, text in enumerate(texts):
        for run in runs("def f():\n" + body(text)):
            index[run].add(g)
    flagged, rows = set(), []
    for bench, split in (("humaneval", "test"), ("mbpp", "test"), ("mbpp", "practice"), ("custom", "test"), ("custom", "practice")):
        over = 0
        for task in load_tasks(bench, split):
            solution = runs("def f():\n" + body(task.canonical_solution))
            if len(solution) < 4:               # a body of fewer than 12 tokens: any code shares its runs
                continue
            hits = defaultdict(int)
            for run in solution:
                for g in index.get(run, ()):
                    hits[g] += 1
            if not hits:
                continue
            best = max(hits, key=hits.get)
            share = hits[best] / len(solution)
            if share >= args.threshold:
                over += 1
                rows.append((share, bench, split, task.task_id, best))
                flagged.update(g for g, n in hits.items() if n / len(solution) >= args.threshold)
        print("%s %s: %d tasks have a training function with at least %.0f%% of the runs of their solution" % (bench, split, over, 100 * args.threshold))
    rows.sort(reverse=True)
    for share, bench, split, task_id, g in rows[:8]:
        print("---- %.0f%% of %s (%s)" % (100 * share, task_id, split))
        print(texts[g][:400])
    print("%d training functions flagged of %d" % (len(flagged), len(texts)))
    if args.write:
        with open(args.write, "w") as f:
            for g in sorted(flagged):
                f.write(json.dumps({"text": texts[g]}) + "\n")


if __name__ == "__main__":
    main()
