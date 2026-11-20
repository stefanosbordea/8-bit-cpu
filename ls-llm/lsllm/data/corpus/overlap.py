"""Benchmark overlap filter: which corpus functions must be kept out of the training data.

A corpus function is dropped when
  name:       its name is the entry point of a benchmark task, or
  alpha:      its alpha key (lsllm.alpha) equals that of a function in a benchmark reference solution, or
  docstring:  its docstring equals a benchmark prompt docstring (compared after collapsing whitespace).

Benchmark files are read as they are found: raw HumanEval records (prompt, entry_point, canonical_solution = body),
raw MBPP records (text, code, test_list) and Task records of docs/INTERFACES.md section 4 (entry_point, prompt,
canonical_solution = full source). New files, such as the custom tasks, are picked up by running the build again.
"""
import ast
import glob
import gzip
import json

from lsllm.alpha import alpha_key


def _functions(src):
    """All function definitions in `src` (none if it does not parse)."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _squash(text):
    """Collapse whitespace, so that two docstrings compare equal however they are wrapped."""
    return " ".join(text.split())


def load_benchmarks(pattern="data/bench/*.jsonl*"):
    """Read every benchmark file that matches `pattern`; return {"names": set, "keys": set, "docs": set}."""
    bench = {"names": set(), "keys": set(), "docs": set()}
    for path in sorted(glob.glob(pattern)):
        with (gzip.open if path.endswith(".gz") else open)(path, "rt") as lines:
            for line in lines:
                _add_task(json.loads(line), bench)
    return bench


def _add_task(row, bench):
    """Add the entry point, the reference solution keys and the prompt docstrings of one benchmark record."""
    prompt = row.get("prompt", "")
    solution = row.get("code") or row.get("canonical_solution", "")
    if not _functions(solution):            # HumanEval stores only the function body: put the prompt in front
        solution = prompt + solution
    funcs = _functions(solution)
    if "entry_point" in row:
        bench["names"].add(row["entry_point"])
    else:                                   # MBPP: the entry point is the solution function that the tests call
        tests = row["test_list"] + row.get("challenge_test_list", [])
        called = {node.id for test in tests for node in ast.walk(ast.parse(test)) if isinstance(node, ast.Name)}
        bench["names"] |= {fn.name for fn in funcs} & called
    bench["keys"] |= {alpha_key(ast.unparse(fn)) for fn in funcs}
    docs = [ast.get_docstring(fn) for fn in _functions(prompt)] + [row.get("text")]
    bench["docs"] |= {_squash(doc) for doc in docs if doc}


def overlap(name, text, key, bench):
    """Return the rules ("name", "alpha", "docstring") by which a corpus function overlaps the benchmarks.

    `text` is the canonical source of the function, `name` its name and `key` its alpha key.
    """
    doc = ast.get_docstring(ast.parse(text).body[0])
    hits = {"name": name in bench["names"],
            "alpha": key in bench["keys"],
            "docstring": bool(doc) and _squash(doc) in bench["docs"]}
    return [rule for rule, hit in hits.items() if hit]
