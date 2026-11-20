"""Benchmark tasks in one shape: HumanEval, MBPP and the custom set.

Every prompt is canonical source (lsllm.canon) that ends inside a function: optional imports and helpers,
then `def ...:` and a docstring. The model continues it with the body.
"""
import ast
import gzip
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from lsllm.canon import canon

DATA = Path(__file__).resolve().parents[2] / "data" / "bench"

# Canonical source is printable ASCII. These are all the other characters that occur in HumanEval prompts
# (the arrow and the less-or-equal sign) and in MBPP task texts, with what we write instead.
ASCII = str.maketrans({"➞": "->", "≤": "<=", "‘": "'", "’": "'", "–": "-",
                       "…": "...", "²": "^2", "³": "^3"})


@dataclass
class Task:
    task_id: str             # "HumanEval/0", "mbpp/11", "custom/test/017"
    entry_point: str         # name of the function under test
    prompt: str              # canonical stub that the model continues
    tests: list              # test statements, run one by one after the candidate is defined
    canonical_solution: str  # canonical full source of a correct solution; it starts with the prompt
    meta: dict               # "benchmark", "split" and benchmark-specific extras


def _task(task_id, entry_point, stub, solution, tests, meta):
    """Make stub and solution canonical. A task that cannot be built is an error, never dropped silently."""
    prompt, full = canon(stub), canon(solution)
    if prompt is None or full is None or not full.startswith(prompt):
        raise ValueError(task_id + ": the prompt or the reference solution cannot be made canonical")
    return Task(task_id, entry_point, prompt, tests, full, meta)


def _humaneval():
    """HumanEval (Chen et al. 2021). The tests are one block: the dataset's check function and its call."""
    tasks = []
    with gzip.open(DATA / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stub = row["prompt"].translate(ASCII)
            tests = [row["test"] + "\ncheck(" + row["entry_point"] + ")"]
            meta = {"ascii_replaced": True} if stub != row["prompt"] else {}
            tasks.append(_task(row["task_id"], row["entry_point"], stub, stub + row["canonical_solution"], tests, meta))
    return tasks


def _mbpp(split):
    """MBPP (Austin et al. 2021) with prompts in the MultiPL-E style (Cassano et al. 2022).

    MBPP gives a sentence and three asserts but no signature. The prompt is the signature of the function that
    the first assert calls, taken from the reference code, with the sentence as its docstring. The reference
    solution is put into the same shape: its imports, helpers and constants move inside that function
    (except module-level assignments to names that are parameters there: inside, the parameter is meant).
    Splits by task id as in the MBPP paper: 11-510 test; 511-600 validation and 601-974 train are our practice.
    """
    first, last = (11, 510) if split == "test" else (511, 974)
    tasks = []
    with open(DATA / "mbpp.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not first <= row["task_id"] <= last:
                continue
            task_id = "mbpp/%d" % row["task_id"]
            tree = ast.parse(row["code"])
            functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            called = [node.func.id for node in ast.walk(ast.parse(row["test_list"][0]))
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            entry = next((name for name in called if name in functions), None)
            if entry is None:
                raise ValueError(task_id + ": the first assert calls no function of the reference code")
            fn = functions[entry]
            text = row["text"].translate(ASCII)
            docstring = ast.Expr(ast.Constant(text))
            parameters = {node.arg for node in ast.walk(fn.args) if isinstance(node, ast.arg)}
            shadowed = [node for node in tree.body if isinstance(node, ast.Assign)
                        and all(isinstance(target, ast.Name) and target.id in parameters for target in node.targets)]
            rest = [node for node in tree.body if node is not fn and node not in shadowed]
            body = fn.body
            fn.body = [docstring]
            stub = ast.unparse(fn)
            fn.body = [docstring] + rest + body
            tests = ([row["test_setup_code"]] if row["test_setup_code"] else []) + row["test_list"]
            meta = {"ascii_replaced": True} if text != row["text"] else {}
            tasks.append(_task(task_id, entry, stub, ast.unparse(fn), tests, meta))
    return tasks


def _custom(split):
    """The custom tasks, written by lsllm.bench.custom with the fields of Task."""
    with open(DATA / ("custom_%s.jsonl" % split), encoding="utf-8") as f:
        return [Task(**json.loads(line)) for line in f if line.strip()]


def load_tasks(benchmark, split="test"):
    """Tasks of "humaneval" (test only), "mbpp" or "custom"; split is "test" or "practice"."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)      # some MBPP solutions use outdated string escapes
        if benchmark == "humaneval" and split == "test":
            tasks = _humaneval()
        elif benchmark == "mbpp" and split in ("test", "practice"):
            tasks = _mbpp(split)
        elif benchmark == "custom" and split in ("test", "practice"):
            tasks = _custom(split)
        else:
            raise ValueError("no benchmark %r with split %r" % (benchmark, split))
    for task in tasks:
        task.meta.update(benchmark=benchmark, split=split)
    return tasks
