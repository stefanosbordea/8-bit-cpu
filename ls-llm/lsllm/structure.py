"""Structure markers: every line break of canonical source becomes exactly one marker.

    <NL>            line break, same indentation level
    <IND>           line break, one level deeper
    <DED1>..<DED5>  line break, k levels back
    <LF>            line break inside a multi-line docstring

Indentation is never written as spaces. After <NL>, <IND> and <DEDk> the next line starts at the new level.
After <LF> a line with text starts at the level of its block and an empty line stays empty, so an empty
docstring line reads <LF><LF>. There is no marker before the first line and none after the last.
"""
import io
import re
import tokenize

MARKERS = ["<NL>", "<IND>", "<DED1>", "<DED2>", "<DED3>", "<DED4>", "<DED5>", "<LF>"]
STEP = {"<NL>": 0, "<IND>": 1, "<DED1>": -1, "<DED2>": -2, "<DED3>": -3, "<DED4>": -4, "<DED5>": -5}
_SPLIT = re.compile("(" + "|".join(MARKERS) + ")")


def to_markers(text):
    """Canonical text -> marker text. Raises ValueError if markers cannot reproduce the text exactly."""
    lines = text.split("\n")
    cut = [("", 0)] * len(lines)    # per line: the marker in front of it, and how many indentation spaces to drop
    level = 0                       # indentation level of the current logical line
    step = 0                        # INDENT tokens minus DEDENT tokens since the last logical line
    fresh = False                   # True between the end of a logical line and the first token of the next
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.NEWLINE:
                fresh, step = True, 0
            elif tok.type == tokenize.INDENT:
                step += 1
            elif tok.type == tokenize.DEDENT:
                step -= 1
            elif tok.type != tokenize.ENDMARKER:
                row = tok.start[0] - 1
                if fresh:
                    level += step
                    marker = "<IND>" if step == 1 else "<NL>" if step == 0 else "<DED%d>" % -step
                    cut[row] = (marker, 4 * level)
                    fresh = False
                for inner in range(row + 1, tok.end[0]):    # a token that spans lines is a multi-line docstring
                    cut[inner] = ("<LF>", 4 * level)
    except (tokenize.TokenError, SyntaxError) as err:
        raise ValueError("not Python source: %s" % err) from err
    out = "".join(marker + line[width:] for (marker, width), line in zip(cut, lines))
    if from_markers(out) != text:   # catches everything that is not canonical, and text that contains a marker
        raise ValueError("text is not canonical, markers cannot reproduce it")
    return out


class MarkerDecoder:
    """Streaming from_markers: feed pieces of marker text one at a time, every call returns the new text.

    A marker is recognised when it lies completely inside one piece.
    """

    def __init__(self, level=0):
        self.level = level      # indentation level of the current line
        self.owed = False       # a docstring line has started and has not received its indentation yet

    def feed(self, piece):
        out = []
        for part in _SPLIT.split(piece):
            if part == "<LF>":
                out.append("\n")
                self.owed = True
            elif part in STEP:
                self.level = max(0, self.level + STEP[part])
                out.append("\n" + "    " * self.level)
                self.owed = False
            elif part:
                if self.owed:
                    out.append("    " * self.level)
                    self.owed = False
                out.append(part)
        return "".join(out)


def from_markers(marker_text, level=0):
    """Marker text -> text. `level` is the indentation level before the first character."""
    return MarkerDecoder(level).feed(marker_text)
