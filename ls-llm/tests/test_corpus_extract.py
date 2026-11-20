import collections

from lsllm.canon import canon
from lsllm.data.corpus.extract import extract, skip_path, stable_canon

SRC = b'''
import os

def add(a, b):
    """Sum."""  # a comment
    return a + b

async def fetch(x):
    return x

def test_add():
    assert add(1, 2) == 3

class Stack:
    """A stack."""
    def push(self, item):
        """Put `item` on top."""
        def inner():
            return item
        self.items.append(inner())

    @property
    def top(self):
        return self.items[-1]

    class Inner:
        def deep(self):
            return 1

def cafe():
    return "caf\xc3\xa9"

def marker():
    return "<IND>"
'''


def test_functions_and_methods():
    counts = collections.Counter()
    records = extract(SRC, counts)
    assert [(rec["name"], rec["kind"]) for rec in records] == [("add", "function"), ("push", "method"), ("top", "method")]
    assert records[0] == {"text": 'def add(a, b):\n    """Sum."""\n    return a + b', "name": "add",
                          "kind": "function", "n_lines": 3, "has_docstring": True}
    assert records[1]["text"].startswith('def push(self, item):\n    """Put `item` on top."""\n    def inner():')
    assert records[2]["text"] == "@property\ndef top(self):\n    return self.items[-1]"
    assert not records[2]["has_docstring"]
    assert counts == {"extracted": 3, "async": 1, "test_function": 1, "not_canonical": 1, "marker": 1}


def test_length_limits():
    body = "".join("    x%d = %d\n" % (g, g) for g in range(39))
    long_line = "def wide():\n    return '%s'\n" % ("a" * 1500)
    counts = collections.Counter()
    records = extract("def fits():\n" + body + "def too_long():\n" + body + "    y = 0\n" + long_line, counts)
    assert [rec["name"] for rec in records] == ["fits"] and records[0]["n_lines"] == 40
    assert counts == {"extracted": 1, "lines": 1, "chars": 1}


def test_python2_idioms_are_dropped():
    src = ("def a(n):\n    return [k for k in xrange(n)]\n"
           "def b(d):\n    return sorted(d.iteritems())\n"
           "def c(s):\n    return isinstance(s, basestring)\n"
           "def ok(d, n):\n    return [k for k in range(n)] + sorted(d.items())\n")
    counts = collections.Counter()
    assert [rec["name"] for rec in extract(src, counts)] == ["ok"]
    assert counts == {"python2": 3, "extracted": 1}


def test_file_that_does_not_parse():
    counts = collections.Counter()
    assert extract("def f(:\n    print 'python 2'", counts) == []
    assert counts == {"file_not_python_3_12": 1}


def test_text_is_a_fixed_point_of_canon():
    # one canon pass leaves these docstrings in a state that a second pass changes again
    sources = ['def f():\n    """\n        A   B\n       / \\\\\n    """\n    return 1\n',
               'def g():\n    """Text.\n        more\n            """\n    return 1\n']
    for src in sources:
        assert canon(canon(src)) != canon(src)
        text = stable_canon(src)
        assert canon(text) == text
        assert extract(src, collections.Counter())[0]["text"] == text
    assert stable_canon("def f(:") is None


def test_skip_path():
    for path in ["Lib/test/x.py", "a/tests/b.py", "a/test_b.py", "a/b_test.py", "conftest.py", "numpy/testing/x.py",
                 "idlelib/idle_test/a.py", "Tests/b.py"]:
        assert skip_path(path) == "test", path
    for path in ["pip/_vendor/a.py", "sklearn/externals/a.py", "third_party/a.py"]:
        assert skip_path(path) == "vendored", path
    for path in ["Lib/doctest.py", "latest.py", "contest/a.py", "Lib/unittest/case.py", "maths/prime_check.py"]:
        assert skip_path(path) is None, path
