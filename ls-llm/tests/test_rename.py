import random

from lsllm.alpha import alpha_key
from lsllm.data.rename import name_pool, names_of, rename

POOL = (["value", "items", "count", "text", "total", "left", "right", "k", "m", "x", "y", "z"], [1.0] * 12)


def many(text, runs=30):
    out = [rename(text, random.Random(seed), POOL) for seed in range(runs)]
    return [o for o in out if o]


def test_parameters_locals_and_docstring_change_together():
    src = 'def add_twice(n, step):\n    """Return n plus step, twice."""\n    result = n + step\n    return result + step'
    for out in many(src):
        params, local = names_of(out)
        assert out.startswith("def add_twice(") and alpha_key(out) == alpha_key(src)
        scope = {}
        exec(out, scope)
        assert scope["add_twice"](3, 4) == 11
        doc = out.split('"""')[1]
        assert all(p in doc for p in params)                # the docstring talks about the new names
        assert "n" not in params or "step" not in params or out == src
    assert len(set(many(src))) > 5


def test_function_word_parameter_is_kept_when_the_docstring_uses_it():
    src = 'def first(a, b):\n    """Return a if it is larger than b."""\n    return a if a > b else b'
    for out in many(src):
        assert "def first(a, " in out and "Return a if it is larger than" in out


def test_keywords_calls_and_attributes_are_left_alone():
    src = 'def order(items, key):\n    """Sort."""\n    return sorted(items, key=key).copy()'
    for out in many(src):
        assert "sorted(" in out and ".copy()" in out and "key=" in out
        assert ", key)" in out and "key=key" in out             # a name used as a keyword is never renamed
        scope = {}
        exec(out, scope)
        assert scope["order"]([3, 1, 2], lambda v: -v) == [3, 2, 1]


def test_recursion_comprehension_and_except_still_work():
    src = ('def walk(n, seen):\n    """Count."""\n    try:\n        total = sum((v for v in seen if v > n))\n'
           '    except TypeError as err:\n        return str(err)\n    return total if n <= 0 else walk(n - 1, seen)')
    for out in many(src):
        scope = {}
        exec(out, scope)
        assert scope["walk"](2, [1, 2, 3]) == 6
        assert scope["walk"](2, 5).startswith("'int' object")


def test_unsafe_functions_are_skipped():
    for src in ('def f(n):\n    return locals()["n"]', 'def f(n):\n    global n2\n    n2 = n\n    return n2',
                'def f(n):\n    class A:\n        pass\n    return n', 'x = 1'):
        assert rename(src, random.Random(0), POOL) is None


def test_name_pool_counts_functions_and_drops_reserved_names():
    texts = ['def f(list, n, a):\n    total = n\n    return total'] * 3 + ['def g(n):\n    return n']
    names, weights = name_pool(texts, least=3)
    assert names == ["n", "total"] and weights[0] > weights[1]


def test_renumber_changes_docstring_and_code_together():
    from lsllm.data.renumber import renumber
    src = 'def scale(n):\n    """Return n multiplied by 10."""\n    return n * 10'
    seen = set()
    for seed in range(20):
        out = renumber(src, random.Random(seed))
        number = int(out.split("multiplied by ")[1].split(".")[0])
        assert number != 10 and out.endswith("return n * %d" % number)
        seen.add(number)
    assert len(seen) > 5
    named = renumber('def times_ten(n):\n    """Return n multiplied by 10."""\n    return n * 10', random.Random(2))
    number = int(named.split("multiplied by ")[1].split(".")[0])
    assert named.startswith("def times_%s(n):" % {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 11: "eleven",
                                                  12: "twelve", 15: "fifteen", 20: "twenty", 30: "thirty", 50: "fifty", 100: "hundred"}[number])
    assert renumber('def larger_than_10(n):\n    """Return True if n is larger than 10."""\n    return n > 10', random.Random(0)).split("(")[0][-2:].strip("_").isdigit()
    assert "def count_" in renumber('def count_fives(items):\n    """Return how often 5 occurs."""\n    return items.count(5)', random.Random(0))
    from lsllm.data.renumber import rename_number
    assert rename_number("count_fives", 5, 6) == "count_sixes" and rename_number("add_one", 1, 7) == "add_seven"
    assert rename_number("from_2_to_5", 2, 3) is None and rename_number("double_plus_two", 2, 3) is None
    assert rename_number("first_three", 3, 4) is None and rename_number("scale", 3, 4) == "scale"
    for left_alone in ('def double(n):\n    """Return n multiplied by 2."""\n    return n * 2',                     # the name says it another way
                       'def f(n):\n    """Return n plus 2, twice 2."""\n    return n + 2',                       # twice in the docstring
                       'def f(n):\n    """Return the numbers from 1 to n."""\n    return list(range(1, n + 1))',  # twice in the code
                       'def f(n):\n    """Return n plus two."""\n    return n + 2',                              # not written as digits
                       'def f(n):\n    """Return n plus 2.5."""\n    return n + 2.5',                            # not a whole number
                       'def f(n):\n    return n + 2'):
        assert renumber(left_alone, random.Random(0)) is None
    out = renumber('def f(items):\n    """Return the first 3 items, at most 100."""\n    return items[:3]', random.Random(1))
    assert "at most 100" in out and "items[:3]" not in out


def test_near_copies_of_benchmark_solutions_are_found():
    from types import SimpleNamespace

    from lsllm.data.nearcopy import NearCopies
    solution = "def f(words, n):\n    \"\"\"Doc.\"\"\"\n    out = []\n    for word in words:\n        if len(word) > n:\n            out.append(word.upper())\n    return out"
    drop = NearCopies([SimpleNamespace(task_id="t/1", canonical_solution=solution), SimpleNamespace(task_id="t/2", canonical_solution="def g(a, b):\n    return a + b")])
    assert drop(solution.replace("def f(", "def other_name(")) and drop.match(solution)[1] == 1.0
    assert drop(solution.replace("> n", ">= n"))                        # one token changed: still a near-copy
    assert not drop(solution.replace("word", "item").replace("out", "kept"))   # renamed everywhere: the exact rule's job (alpha key)
    assert not drop("def g(a, b):\n    return a + b") and not drop("def h(x):\n    return [x] * 3")     # short bodies are not checked
