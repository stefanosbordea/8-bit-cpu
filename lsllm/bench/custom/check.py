"""Checker for the custom benchmark. Run: .venv/bin/python -m lsllm.bench.custom.check

For data/bench/custom_test.jsonl and custom_practice.jsonl (each one only if it exists) it verifies that
  - every task has exactly the fields of docs/INTERFACES.md section 4 and the task ids count up from 000,
  - the entry point is a snake_case name that is neither a keyword nor a builtin,
  - the prompt is a canonical stub: a def line with plain parameter names (no type hints) and a one-line
    docstring of the form "Return ... .",
  - the canonical solution is canonical, starts with the prompt, is one function with 1 to 6 body lines and
    has no import and no input/output,
  - there are 4 to 6 tests, each one a canonical single-line assert that uses the function,
  - the reference solution passes its tests (run in a separate Python process with a timeout),
  - meta names a family of data/bench/custom_families.json with its stratum, every family has exactly
    5 tasks per file, and there are 24 near and 16 far families,
  - no function name and no alpha key (lsllm.alpha) is used twice, inside a file or across the two files,
  - no function name and no alpha key is shared with a function of a HumanEval or MBPP reference solution.
It prints one line per problem and exits with status 1 if there is any.
"""
import ast
import builtins
import gzip
import json
import keyword
import re
import subprocess
import sys
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lsllm.alpha import alpha_key
from lsllm.canon import canon

BENCH = Path(__file__).resolve().parents[3] / "data" / "bench"
SPLITS = ("test", "practice")
FIELDS = {"task_id": str, "entry_point": str, "prompt": str, "tests": list, "canonical_solution": str, "meta": dict}
PER_FAMILY = 5                              # tasks of every family in each split
STRATA = {"near": 24, "far": 16}            # number of families per stratum
BANNED = {"print", "input", "open", "eval", "exec", "__import__"}


def load(split):
    """Tasks of one split (an empty list if the file does not exist)."""
    path = BENCH / ("custom_%s.jsonl" % split)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def is_assert(test, name):
    """True if `test` is one canonical single-line assert statement that uses the function `name`."""
    if not isinstance(test, str) or "\n" in test or canon(test) != test:
        return False
    statement = ast.parse(test).body[0]
    return isinstance(statement, ast.Assert) and name in {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)}


def static_problems(task, task_id, families):
    """Problems of one task that show without running it. `task_id` is the id it should have."""
    if not isinstance(task, dict) or set(task) != set(FIELDS) or any(not isinstance(task[f], FIELDS[f]) for f in FIELDS):
        return ["the fields are not " + ", ".join(FIELDS)]
    name, prompt, tests, solution, meta = (task[field] for field in list(FIELDS)[1:])
    problems = []
    if task["task_id"] != task_id:
        problems.append("the task id should be " + task_id)
    if not re.fullmatch("[a-z][a-z0-9_]*", name) or keyword.iskeyword(name) or hasattr(builtins, name):
        problems.append("the entry point is not a snake_case name that is free to use")
    stub = r'def %s\(\w+(, \w+)*\):\n    """Return [^\n]*\."""' % re.escape(name)
    if canon(prompt) != prompt or not re.fullmatch(stub, prompt):
        problems.append("the prompt is not a canonical def line without type hints plus a one-line docstring 'Return ... .'")
    if canon(solution) != solution or not solution.startswith(prompt + "\n"):
        problems.append("the solution is not canonical or does not start with the prompt")
    else:
        tree = ast.parse(solution)
        if len(tree.body) != 1 or not 1 <= solution.count("\n") - prompt.count("\n") <= 6:
            problems.append("the solution is not one function with 1 to 6 body lines")
        imports = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
        if imports or BANNED & {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}:
            problems.append("the solution has an import or does input/output")
    if not 4 <= len(tests) <= 6:
        problems.append("there are %d tests, allowed are 4 to 6" % len(tests))
    problems += ["this test is not a canonical single-line assert that uses the function: %r" % (test,)
                 for test in tests if not is_assert(test, name)]
    info = families.get(meta.get("family")) if isinstance(meta.get("family"), str) else None
    if info is None or meta != {"family": meta["family"], "stratum": info["stratum"]}:
        problems.append("meta is not a family of custom_families.json with its stratum")
    return problems


def run_problems(task):
    """Run the reference solution and then its tests in a separate Python process; return the problem, if any."""
    code = "\n".join([task["canonical_solution"]] + task["tests"])
    try:
        done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return ["the reference solution does not finish its tests within 10 seconds"]
    if done.returncode == 0:
        return []
    where = re.search(r"line (\d+), in <module>", done.stderr)        # the line of `code` that raised the error
    line = code.split("\n")[int(where.group(1)) - 1] if where else "exit code %d" % done.returncode
    return ["the reference solution fails: %s (%s)" % (line, done.stderr.strip().split("\n")[-1])]


def task_problems(task, task_id, families):
    """All problems of one task on its own. The reference solution is run only if the task is well formed."""
    return static_problems(task, task_id, families) or run_problems(task)


def family_problems(families):
    """Problems of the families file: {family: {"stratum": "near" | "far", "description": text}}."""
    problems = ["family %s needs a stratum (near or far) and a description" % family
                for family, info in families.items()
                if set(info) != {"stratum", "description"} or info["stratum"] not in STRATA or not info["description"]]
    counts = dict(Counter(info.get("stratum") for info in families.values()))
    if counts != STRATA:
        problems.append("families per stratum are %s, expected %s" % (counts, STRATA))
    return problems


def reference_functions():
    """Every function of the HumanEval and MBPP reference solutions: {lower-case name: task}, {alpha key: task}."""
    sources = []
    with gzip.open(BENCH / "HumanEval.jsonl.gz", "rt") as lines:
        for row in map(json.loads, lines):
            sources.append((row["task_id"], row["prompt"] + row["canonical_solution"]))
    with open(BENCH / "mbpp.jsonl") as lines:
        for row in map(json.loads, lines):
            sources.append(("mbpp/%d" % row["task_id"], row["code"]))
    names, keys = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)      # some MBPP solutions write "\d" in a plain string
        for task_id, source in sources:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.FunctionDef):
                    names.setdefault(node.name.lower(), task_id)
                    keys.setdefault(alpha_key(ast.unparse(node)), task_id)
    return names, keys


def problems():
    """All problems of the custom benchmark files, one text line each."""
    families = json.loads((BENCH / "custom_families.json").read_text())
    found = ["custom_families.json: " + text for text in family_problems(families)]
    names, keys = reference_functions()         # the custom tasks are added to these as they are read
    for split in SPLITS:
        tasks = load(split)
        ids = ["custom/%s/%03d" % (split, number) for number in range(len(tasks))]
        with ThreadPoolExecutor(4) as pool:     # four reference solutions run at the same time
            issues_of = list(pool.map(task_problems, tasks, ids, [families] * len(tasks)))
        counts = Counter()
        for task, task_id, issues in zip(tasks, ids, issues_of):
            if not issues:
                counts[task["meta"]["family"]] += 1
                name, key = task["entry_point"], alpha_key(task["canonical_solution"])
                if name in names:
                    issues.append("the function name is also used by " + names[name])
                if key in keys:
                    issues.append("the solution is the same code, up to renaming, as " + keys[key])
                names.setdefault(name, task_id)
                keys.setdefault(key, task_id)
            label = "%s %s" % (task_id, task.get("entry_point")) if isinstance(task, dict) else task_id
            found += ["%s: %s" % (label, text) for text in issues]
        if tasks:
            found += ["custom_%s.jsonl: family %s has %d correct tasks, expected %d" % (split, family, counts[family], PER_FAMILY)
                      for family in families if counts[family] != PER_FAMILY]
    return found


if __name__ == "__main__":
    found = problems()
    for text in found:
        print(text)
    print("custom tasks: %d test, %d practice, %d problems" % (len(load("test")), len(load("practice")), len(found)))
    sys.exit(1 if found else 0)
