"""Tests of the custom benchmark: the checker passes on the real task files and catches broken ones."""
import json
import shutil
from collections import Counter

from lsllm.bench.custom import build, check

FAMILIES = json.loads((check.BENCH / "custom_families.json").read_text())


def test_task_files_have_no_problems():
    assert check.problems() == []


def test_task_files_match_their_sources():
    for split in check.SPLITS:
        if (build.HERE / ("tasks_" + split)).is_dir():
            assert check.load(split) == build.records(split), "rebuild with: .venv/bin/python -m lsllm.bench.custom.build"


def test_test_split_has_200_tasks_in_two_strata():
    tasks = check.load("test")
    assert len(tasks) == 200
    assert Counter(task["meta"]["stratum"] for task in tasks) == {"near": 120, "far": 80}
    assert Counter(task["meta"]["family"] for task in tasks) == dict.fromkeys(FAMILIES, 5)


def test_checker_catches_broken_tasks():
    good = check.load("test")[0]                    # double_number(n), body: return n * 2
    prompt, solution, tests = good["prompt"], good["canonical_solution"], good["tests"]
    broken = [
        ("fields", {"extra": 1}),
        ("task id", {"task_id": "custom/test/1"}),
        ("entry point", {"entry_point": "doubleNumber"}),
        ("prompt", {"prompt": prompt.replace("(n)", "(n: int)")}),                             # type hint
        ("prompt", {"prompt": prompt.replace("Return", "Returns")}),                           # docstring form
        ("start with the prompt", {"canonical_solution": solution.replace("by 2.", "by two.")}),
        ("not canonical", {"canonical_solution": solution.replace("n * 2", "n*2")}),
        ("1 to 6 body lines", {"canonical_solution": solution.replace("return", "n = n\n    " * 6 + "return")}),
        ("input/output", {"canonical_solution": solution.replace("return", "print(n)\n    return")}),
        ("fails", {"canonical_solution": solution.replace("n * 2", "n * 3")}),
        ("3 tests", {"tests": tests[:3]}),
        ("7 tests", {"tests": tests + tests[:2]}),
        ("this test", {"tests": tests[:4] + ["assert 1 == 1"]}),                               # does not use the function
        ("this test", {"tests": tests[:4] + ["assert double_number(1) == 2  # comment"]}),     # not canonical
        ("this test", {"tests": tests[:4] + ["x = double_number(1)"]}),                        # not an assert
        ("meta", {"meta": {"family": "arith_one", "stratum": "far"}}),
        ("meta", {"meta": {"family": "no_such_family", "stratum": "near"}}),
    ]
    assert check.task_problems(good, "custom/test/000", FAMILIES) == []
    for word, changes in broken:
        found = check.task_problems(dict(good, **changes), "custom/test/000", FAMILIES)
        assert any(word in text for text in found), (word, changes, found)


def test_checker_catches_duplicates_and_benchmark_copies(tmp_path, monkeypatch):
    for name in ("custom_families.json", "HumanEval.jsonl.gz", "mbpp.jsonl"):
        shutil.copy(check.BENCH / name, tmp_path / name)
    tasks = check.load("test")[:5]                  # the five tasks of the family arith_one
    stub = 'def strlen(s):\n    """Return the number of characters of the string s."""'
    strlen = {"task_id": "custom/practice/001", "entry_point": "strlen", "prompt": stub,
              "tests": ["assert strlen('') == 0", "assert strlen('a') == 1", "assert strlen('ab') == 2", "assert strlen('a b') == 3"],
              "canonical_solution": stub + "\n    return len(s)", "meta": {"family": "string_case_length", "stratum": "near"}}
    practice = [dict(tasks[0], task_id="custom/practice/000"), strlen]
    for name, rows in (("custom_test.jsonl", tasks), ("custom_practice.jsonl", practice)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(check, "BENCH", tmp_path)
    found = check.problems()
    for text in ("custom/practice/000 double_number: the function name is also used by custom/test/000",
                 "custom/practice/000 double_number: the solution is the same code, up to renaming, as custom/test/000",
                 "custom/practice/001 strlen: the function name is also used by HumanEval/23",
                 "custom/practice/001 strlen: the solution is the same code, up to renaming, as HumanEval/23",
                 "custom_test.jsonl: family arith_two has 0 correct tasks, expected 5"):
        assert text in found
    assert not any(text.startswith("custom/test/") for text in found)
