"""Build data/bench/custom_<split>.jsonl from the task source files in lsllm/bench/custom/tasks_<split>/.

A task source file is plain Python. It is only parsed here, never imported:

    family = "arith_one"                # family of the tasks that follow (a key of custom_families.json)

    def double_number(n):               # one task: a function with a one-line docstring ...
        '''Return n multiplied by 2.'''
        return n * 2
    assert double_number(3) == 6        # ... followed by its tests, one assert per line

The prompt of a task is the def line plus the docstring. Tasks are numbered in the order of the families in
data/bench/custom_families.json, so inserting a task shifts the ids after it.
Run: .venv/bin/python -m lsllm.bench.custom.build
"""
import ast
import json
from pathlib import Path

from lsllm.canon import canon

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2] / "data" / "bench"


def read_tasks(path):
    """Tasks of one source file, as dicts that still lack the task id and the stratum."""
    tasks, family = [], None
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            family = node.value.value
        elif isinstance(node, ast.FunctionDef):
            solution = canon(ast.unparse(node))
            assert solution, "canon() rejects the function %s" % node.name
            prompt = "\n".join(solution.split("\n")[:2])
            tasks.append({"entry_point": node.name, "prompt": prompt, "tests": [],
                          "canonical_solution": solution, "meta": {"family": family}})
        elif isinstance(node, ast.Assert):
            tasks[-1]["tests"].append(ast.unparse(node))
    return tasks


def records(split):
    """All tasks of one split as Task records (docs/INTERFACES.md section 4), in family order."""
    families = json.loads((BENCH / "custom_families.json").read_text())
    order = list(families)
    tasks = [task for path in sorted((HERE / ("tasks_" + split)).glob("*.py")) for task in read_tasks(path)]
    tasks.sort(key=lambda task: order.index(task["meta"]["family"]))
    for number, task in enumerate(tasks):
        task["meta"]["stratum"] = families[task["meta"]["family"]]["stratum"]
        tasks[number] = {"task_id": "custom/%s/%03d" % (split, number), **task}
    return tasks


if __name__ == "__main__":
    for split in ("test", "practice"):
        if (HERE / ("tasks_" + split)).is_dir():
            rows = records(split)
            (BENCH / ("custom_%s.jsonl" % split)).write_text("".join(json.dumps(row) + "\n" for row in rows))
            print("custom_%s.jsonl: %d tasks" % (split, len(rows)))
