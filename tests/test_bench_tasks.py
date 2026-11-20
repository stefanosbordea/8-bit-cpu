"""The three task sets load completely and in the shape the harness relies on."""
import json

import pytest

from lsllm.bench import load_tasks
from lsllm.bench import tasks as tasks_module
from lsllm.canon import canon


def check_shape(tasks, benchmark, split):
    for task in tasks:
        # canon accepts them: valid syntax, printable ASCII only (canon is not idempotent for HumanEval/51 and
        # HumanEval/109, so `canon(prompt) == prompt` cannot be asserted)
        assert canon(task.prompt) is not None and canon(task.canonical_solution) is not None, task.task_id
        assert task.canonical_solution.startswith(task.prompt + "\n    "), task.task_id
        assert task.tests and all(task.entry_point in test for test in task.tests[-1:]), task.task_id
        assert (task.meta["benchmark"], task.meta["split"]) == (benchmark, split)
    assert len({task.task_id for task in tasks}) == len(tasks)


@pytest.fixture(scope="module")
def mbpp():
    return {split: load_tasks("mbpp", split) for split in ("test", "practice")}


def test_humaneval():
    tasks = load_tasks("humaneval")
    check_shape(tasks, "humaneval", "test")
    assert [task.task_id for task in tasks] == ["HumanEval/%d" % g for g in range(164)]
    assert tasks[0].prompt.startswith("from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n")
    assert all(len(task.tests) == 1 and task.tests[0].endswith("\ncheck(%s)" % task.entry_point) for task in tasks)
    # ten prompts contain an arrow or a less-or-equal sign, written as -> and <= here
    replaced = [task.task_id for task in tasks if task.meta.get("ascii_replaced")]
    assert replaced == ["HumanEval/%d" % g for g in (72, 74, 84, 92, 125, 126, 132, 134, 137, 147)]
    assert "will_it_fly([1, 2], 5) -> False" in tasks[72].prompt and "0 <= N <= 10000" in tasks[84].prompt


def test_humaneval_is_a_test_set_only():
    with pytest.raises(ValueError):
        load_tasks("humaneval", "practice")
    with pytest.raises(ValueError):
        load_tasks("mbpp", "train")


def test_mbpp_splits_follow_the_paper(mbpp):
    test, practice = mbpp["test"], mbpp["practice"]
    check_shape(test, "mbpp", "test")
    check_shape(practice, "mbpp", "practice")
    assert [task.task_id for task in test] == ["mbpp/%d" % g for g in range(11, 511)]
    assert [task.task_id for task in practice] == ["mbpp/%d" % g for g in range(511, 975)]
    assert sum(bool(task.meta.get("ascii_replaced")) for task in test + practice) == 9


def test_mbpp_prompt_is_signature_and_docstring(mbpp):
    tasks = {task.task_id: task for task in mbpp["test"]}
    first = tasks["mbpp/11"]
    assert first.entry_point == "remove_Occ"
    assert first.prompt == ('def remove_Occ(s, ch):\n    """Write a python function to remove first and last occurrence '
                            'of a given character from the string."""')
    assert first.tests == ['assert remove_Occ("hello","l") == "heo"', 'assert remove_Occ("abcda","a") == "bcd"',
                           'assert remove_Occ("PHP","P") == "H"']
    # imports and helpers of the reference code move into the function, they are not part of the prompt
    helper = tasks["mbpp/223"]
    assert helper.prompt.count("def ") == 1 and "\n    def binary_search(arr, low, high, x):\n" in helper.canonical_solution
    assert "\n    import re\n" in tasks["mbpp/16"].canonical_solution and "import" not in tasks["mbpp/16"].prompt
    assert len(tasks["mbpp/367"].tests) == 4        # test_setup_code comes first, then the three asserts
    assert "n'th smart number" in tasks["mbpp/122"].prompt     # the typographic apostrophe became ASCII


def test_custom_tasks_are_read_from_json_lines(tmp_path, monkeypatch):
    row = {"task_id": "custom/practice/001", "entry_point": "double", "prompt": 'def double(x):\n    """Return 2x."""',
           "tests": ["assert double(2) == 4"], "canonical_solution": 'def double(x):\n    """Return 2x."""\n    return 2 * x',
           "meta": {"family": "arithmetic", "stratum": "near"}}
    (tmp_path / "custom_practice.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(tasks_module, "DATA", tmp_path)
    tasks = load_tasks("custom", "practice")
    check_shape(tasks, "custom", "practice")
    assert tasks[0].entry_point == "double" and tasks[0].meta["family"] == "arithmetic"
    with pytest.raises(FileNotFoundError):
        load_tasks("custom", "test")
