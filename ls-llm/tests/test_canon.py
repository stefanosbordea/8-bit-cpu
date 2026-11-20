import ast

from lsllm.canon import canon

SRC = '''
import math   # a comment

def area(r):
    """Area of a circle.

       >>> area(1)
       3.14

    Second paragraph.   
    """
    if r < 0: raise ValueError("neg")

    def helper(x):
        """Nested
        doc."""
        return x*x

    return math.pi*helper(r)
'''

EXPECTED = '''import math
def area(r):
    """Area of a circle.

       >>> area(1)
       3.14

    Second paragraph."""
    if r < 0:
        raise ValueError('neg')
    def helper(x):
        """Nested
        doc."""
        return x * x
    return math.pi * helper(r)'''


def test_canon_layout():
    assert canon(SRC) == EXPECTED


def test_canon_idempotent_and_valid():
    once = canon(SRC)
    assert canon(once) == once
    ast.parse(once)


def test_canon_rejects():
    assert canon("def f(:") is None
    assert canon("x = 'café'") is None
    assert canon("") is None


def test_canon_stub_only_docstring():
    stub = 'def add(a, b):\n    """Return a plus b."""\n'
    assert canon(stub) == 'def add(a, b):\n    """Return a plus b."""'
