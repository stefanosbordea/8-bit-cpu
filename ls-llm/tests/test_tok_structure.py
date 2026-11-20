"""Structure markers: exact examples, round trip on the whole bootstrap set, streaming, rejected inputs."""
import json
import re
from pathlib import Path

import pytest

from lsllm.structure import MARKERS, MarkerDecoder, from_markers, to_markers

BOOT = Path(__file__).resolve().parents[1] / "data/tok/bootstrap"
PARTS = re.compile("(" + "|".join(MARKERS) + ")")

CANON = '''import math
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

MARKED = ('import math<NL>def area(r):<IND>"""Area of a circle.<LF><LF>   >>> area(1)<LF>   3.14<LF><LF>'
          'Second paragraph."""<NL>if r < 0:<IND>raise ValueError(\'neg\')<DED1>def helper(x):<IND>"""Nested<LF>'
          'doc."""<NL>return x * x<DED1>return math.pi * helper(r)')


@pytest.fixture(scope="module")
def texts():
    """All bootstrap functions (made by data/tok/make_bootstrap.py)."""
    return [json.loads(line)["text"] for name in ("train", "val") for line in open(BOOT / (name + ".jsonl"))]


def test_example():
    assert to_markers(CANON) == MARKED
    assert from_markers(MARKED) == CANON


def test_stub_prompts():
    assert to_markers("def f(x):") == "def f(x):"
    assert to_markers('def f(x):\n    """Doc."""') == 'def f(x):<IND>"""Doc."""'


def test_round_trip_on_bootstrap(texts):
    assert len(texts) >= 5000
    rejected = 0
    for text in texts:
        try:
            marked = to_markers(text)
        except ValueError:
            rejected += 1
            continue
        assert from_markers(marked) == text
        assert "\n" not in marked
        assert len(PARTS.findall(marked)) == text.count("\n")                 # one marker per line break
        assert not re.search(r"(<NL>|<IND>|<DED\d>) ", marked)                 # indentation is never spaces
    assert rejected <= len(texts) // 1000                                     # only dedents of more than 5 levels


def test_streaming_equals_batch(texts):
    for text in texts[::7]:
        try:
            marked = to_markers(text)
        except ValueError:
            continue
        pieces = []                 # the finest stream there is: every marker and every character on its own
        for part in PARTS.split(marked):
            pieces += [part] if part in MARKERS else list(part)
        decoder = MarkerDecoder()
        assert "".join(decoder.feed(piece) for piece in pieces) == from_markers(marked) == text


def test_levels():
    assert from_markers("a<IND>b<IND>c<DED2>d") == "a\n    b\n        c\nd"
    assert from_markers("a<DED1>b<NL>c", level=2) == "a\n    b\n    c"
    assert from_markers('"""a<LF><LF>b<LF>  c"""<NL>d', level=1) == '"""a\n\n    b\n      c"""\n    d'
    assert from_markers("a<DED3>b", level=1) == "a\nb"                         # the level never goes below 0
    decoder = MarkerDecoder(level=1)
    decoder.feed("x<IND>y<IND>z<DED1>")
    assert decoder.level == 2


def test_deepest_dedent():
    nest = "".join("    " * depth + "if a:\n" for depth in range(5))
    text = nest + "    " * 5 + "b = 1\nc = 2"
    assert to_markers(text).endswith("b = 1<DED5>c = 2")
    with pytest.raises(ValueError):                                           # six levels back: no marker for it
        to_markers("if a:\n" + nest.replace("if", "    if") + "    " * 6 + "b = 1\nc = 2")


@pytest.mark.parametrize("text", [
    "x = '<NL>'",               # contains a marker
    "s = 'a<LF>b'",
    "x = (1,\n     2)",         # line break inside brackets
    "x = 1\n\ny = 2",           # blank line
    "x = 1\n",                  # trailing line break
    "if a:\n  b = 1",           # indentation is not 4 spaces
    "x = 1\n# note\ny = 2",     # comment line
    "def f(:\n    '''",         # not Python
])
def test_rejects_text_that_is_not_canonical(text):
    with pytest.raises(ValueError):
        to_markers(text)
