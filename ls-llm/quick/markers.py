"""Fast-path structure markers (same contract as docs/INTERFACES.md section 2)."""
import io
import re
import tokenize

NL, IND, LF = "<NL>", "<IND>", "<LF>"
DED = ["<DED%d>" % k for k in range(1, 6)]
MARKERS = [NL, IND, LF] + DED
_SPLIT = re.compile(r"(<NL>|<IND>|<LF>|<DED[1-5]>)")


def to_markers(text):
    lines = text.split("\n")
    inner = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            inner.update(range(tok.start[0], tok.end[0]))
    out, level = [], None
    for row, line in enumerate(lines):
        if row in inner:
            pad = "    " * level
            out.append(LF + (line[len(pad):] if line.startswith(pad) else line.strip()))
            continue
        body = line.lstrip(" ")
        new = (len(line) - len(body)) // 4
        if level is None:
            mark = ""
        elif new == level:
            mark = NL
        elif new == level + 1:
            mark = IND
        elif 1 <= level - new <= 5:
            mark = DED[level - new - 1]
        else:
            raise ValueError("indentation jump")
        level = new
        out.append(mark + body)
    return "".join(out)


def from_markers(marked, level=0):
    out, after_lf = [], False
    for piece in _SPLIT.split(marked):
        if not piece:
            continue
        if piece == LF:
            out.append("\n")
            after_lf = True
        elif piece in (NL, IND) or piece in DED:
            level += 1 if piece == IND else 0 if piece == NL else -(DED.index(piece) + 1)
            out.append("\n" + "    " * level)
            after_lf = False
        else:
            out.append(("    " * level if after_lf else "") + piece)
            after_lf = False
    return "".join(out)


def level_after(marked, level=0):
    for piece in _SPLIT.findall(marked):
        level += 1 if piece == IND else -(DED.index(piece) + 1) if piece in DED else 0
    return level
