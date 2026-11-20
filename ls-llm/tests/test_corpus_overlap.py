import gzip
import json
import os
import re

import pytest

from lsllm.alpha import alpha_key
from lsllm.data.corpus.overlap import load_benchmarks, overlap

pytestmark = pytest.mark.filterwarnings("ignore::SyntaxWarning")    # MBPP solutions contain "\d" in plain strings

HUMANEVAL = {"task_id": "HumanEval/x", "entry_point": "is_even",
             "prompt": 'def helper(n):\n    """ Halve n. """\n    return n // 2\n\n\ndef is_even(n: int) -> bool:\n'
                       '    """ Say whether n\n    is even.\n    >>> is_even(2)\n    True\n    """\n',
             "canonical_solution": "    return n % 2 == 0\n", "test": "def check(candidate):\n    assert candidate(2)\n"}
MBPP = {"task_id": 1, "text": "Write a function to triple a number.",
        "code": "def aux(v):\r\n\treturn v * 3\r\ndef triple_it(v):\r\n\treturn aux(v)",
        "test_list": ["assert triple_it(2) == 6"], "challenge_test_list": []}
TASK = {"task_id": "custom/test/001", "entry_point": "last_item", "tests": ["assert last_item([1, 2]) == 2"],
        "prompt": 'def last_item(xs):\n    """Return the last item of xs."""',
        "canonical_solution": 'def last_item(xs):\n    """Return the last item of xs."""\n    return xs[-1]',
        "meta": {"family": "lists", "stratum": "near"}}


def write(path, rows):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "wt") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")


def hits(src, bench):
    name = re.match(r"def (\w+)", src).group(1)
    return overlap(name, src, alpha_key(src), bench)


def test_three_file_formats(tmp_path):
    write(tmp_path / "HumanEval.jsonl.gz", [HUMANEVAL])
    write(tmp_path / "mbpp.jsonl", [MBPP])
    bench = load_benchmarks(str(tmp_path / "*.jsonl*"))
    assert bench["names"] == {"is_even", "triple_it"}       # entry points only, not the helpers
    assert bench["docs"] == {"Halve n.", "Say whether n is even. >>> is_even(2) True", "Write a function to triple a number."}
    assert len(bench["keys"]) == 4                           # helper and entry point of both tasks

    assert hits("def is_even(k):\n    return not k & 1", bench) == ["name"]
    assert hits("def parity(k):\n    return k % 2 == 0", bench) == ["alpha"]
    assert hits("def times_three(q):\n    return q * 3", bench) == ["alpha"]           # equals the MBPP helper
    assert hits('def other(k):\n    """Say whether n is even.\n\n    >>> is_even(2)\n    True"""\n    return 1', bench) == ["docstring"]
    assert hits('def other(k):\n    """Write a function to   triple a number."""\n    return 1', bench) == ["docstring"]
    assert hits('def is_even(n):\n    """Halve n."""\n    return n % 2 == 0', bench) == ["name", "alpha", "docstring"]
    assert hits('def fine(k):\n    """Say whether k is odd."""\n    return k % 2 == 1', bench) == []

    # the custom task files appear later: running the loader again picks them up
    assert hits("def last_item(seq):\n    return seq[len(seq) - 1]", bench) == []
    write(tmp_path / "custom_test.jsonl", [TASK])
    bench = load_benchmarks(str(tmp_path / "*.jsonl*"))
    assert hits("def last_item(seq):\n    return seq[len(seq) - 1]", bench) == ["name"]
    assert hits("def final(seq):\n    return seq[-1]", bench) == ["alpha"]
    assert hits('def final(seq):\n    """Return the last item of xs."""\n    return seq.pop()', bench) == ["docstring"]


@pytest.mark.skipif(not os.path.exists("data/bench/mbpp.jsonl"), reason="benchmark files not downloaded")
def test_every_real_task_is_covered():
    bench = load_benchmarks()
    with gzip.open("data/bench/HumanEval.jsonl.gz", "rt") as lines:
        humaneval = [json.loads(line) for line in lines]
    with open("data/bench/mbpp.jsonl") as lines:
        mbpp = [json.loads(line) for line in lines]
    assert len(humaneval) == 164 and len(mbpp) == 974
    for row in humaneval:
        assert row["entry_point"] in bench["names"]
        assert alpha_key(row["prompt"] + row["canonical_solution"]) in bench["keys"]
    for row in mbpp:
        called = re.findall(r"\w+", " ".join(row["test_list"]))
        assert bench["names"] & set(called), row["task_id"]     # the tested function is a known entry point
        assert alpha_key(row["code"]) in bench["keys"]
        assert " ".join(row["text"].split()) in bench["docs"]
    for name in ("data/bench/custom_test.jsonl", "data/bench/custom_practice.jsonl"):
        if os.path.exists(name):
            with open(name) as lines:
                for line in lines:
                    assert json.loads(line)["entry_point"] in bench["names"]
