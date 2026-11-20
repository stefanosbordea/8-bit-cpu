"""The request suite: the request rule, and that every stored solution is an unchanged corpus function that passes."""
import gzip
import json
import secrets

from lsllm.asks import ASKS, SOLUTIONS, MISSING, load_jsonl, plain, rename, request_to_prompt
from lsllm.asks.mine import MAX_TOKENS, ROOT, TOKENIZER, program
from lsllm.asks.run import RefCompleter, run
from lsllm.bench.harness import DRIVER, MEMORY_MB, _outcome
from lsllm.data.selfcontained import self_contained
from lsllm.sandbox import run_many
from lsllm.tok import Tok

ASK = {ask["id"]: ask for ask in load_jsonl(ASKS)}
SOLVED = load_jsonl(SOLUTIONS)


def test_plain_form():
    assert plain('def f(n: int) -> int:\n    """Doc."""\n    return n') == ("def f(n: int) -> int:\n    return n", False)
    assert plain("def f(x: List[int], k: int=2) -> Optional[int]:\n    y: int = k\n    return y") == ("def f(x, k=2):\n    y = k\n    return y", True)
    assert plain('def f():\n    """Only a docstring."""') == (None, False)


def test_rename_leaves_attributes():
    assert rename("def pop(array):\n    return array.pop()", "pop", "stack_pop") == "def stack_pop(array):\n    return array.pop()"
    assert rename("def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)", "fib", "fibonacci") == \
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)"


def test_request_rule():
    assert request_to_prompt("bubble sort") == "def bubble_sort("
    assert request_to_prompt("Make me a bubble sort program") == "def bubble_sort("
    assert request_to_prompt("Can you write a function for binary search in Python, please?") == "def binary_search("
    assert request_to_prompt("celsius to fahrenheit") == "def celsius_fahrenheit("


def test_every_phrasing_maps_to_the_name():
    assert len(ASK) >= 60 and len(ASK) == len(load_jsonl(ASKS))
    for ask in ASK.values():
        assert ask["request"] == ask["requests"][0] and len(ask["requests"]) >= 3
        for request in ask["requests"]:
            assert request_to_prompt(request) == "def " + ask["name"] + "(", request


def test_every_ask_is_solved_or_missing():
    assert sorted([row["id"] for row in SOLVED] + [row["id"] for row in load_jsonl(MISSING)]) == sorted(ASK)


def test_solutions_fit_the_model():
    tok = Tok(str(ROOT / TOKENIZER))
    for row in SOLVED:
        form, hints_removed = plain(row["text"])
        assert self_contained(form) and hints_removed == row["hints_removed"], row["id"]
        assert len(tok.encode(form)) == row["tokens_without_docstring"] <= MAX_TOKENS, row["id"]
        assert row["tokens"] is None or row["tokens"] == len(tok.encode(row["text"])), row["id"]


def test_solutions_are_unchanged_corpus_lines():
    wanted = {}                                     # file -> {line number: solution}
    for row in SOLVED:
        wanted.setdefault(row["file"], {})[row["line"]] = row
    for name, rows in wanted.items():
        with (gzip.open if name.endswith(".gz") else open)(ROOT / name, "rt", encoding="utf-8") as lines:
            for number, line in enumerate(lines):
                if number in rows:
                    record, row = json.loads(line), rows.pop(number)
                    assert record["text"] == row["text"] and record["name"] == row["original_name"], row["id"]
                    assert (record["source"], record["licence"], record["path"]) == (row["source"], row["licence"], row["path"])
        assert not rows, "lines not found in " + name


def test_solutions_pass_in_plain_form_under_the_ask_name():
    mark = "REPORT-" + secrets.token_hex(8) + " "
    drivers = ["PROGRAM = %r\nTESTS = %r\nMARK = %r\n" % (program(ASK[row["id"]], plain(row["text"])[0], row["original_name"]),
                                                         ASK[row["id"]]["tests"], mark) + DRIVER for row in SOLVED]
    outcomes = [_outcome(result, mark) for result in run_many(drivers, 2, 5.0, MEMORY_MB)]
    assert [(row["id"], outcome) for row, outcome in zip(SOLVED, outcomes) if outcome[0] != "pass"] == []


def test_reference_run_passes_every_solved_ask():
    solved = {row["id"] for row in SOLVED}
    rows = run(RefCompleter(), list(ASK.values()))
    assert [row["id"] for row in rows if row["id"] in solved and row["outcome"] != "pass"] == []
    assert all(row["outcome"] == "syntax" for row in rows if row["id"] not in solved)
