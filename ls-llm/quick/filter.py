"""Syntax filter: a small state machine over marker text that says which next tokens keep the code well formed.

State: lexer mode (code, '...' string, "..." string, docstring), bracket stack, indentation level, and three facts
about the current line (has content, first word, last non-space character). Rules:
1. a closing bracket must match the innermost open bracket;
2. no line break inside brackets or inside a one-line string; inside a docstring only <LF>;
3. after a line that ends with ':' the next line break must be <IND>; otherwise <IND> is forbidden;
4. a line that starts with a block keyword must end with ':' before any line break;
5. <DEDk> may not go below level 0; no empty lines; end-of-text only where a line could end.
"""
import re

_SYM = re.compile(r"<NL>|<IND>|<LF>|<DED[1-5]>|.", re.S)
BLOCK = {"def", "if", "elif", "else", "for", "while", "try", "except", "finally", "with", "class", "async"}
OPEN, CLOSE = "([{", ")]}"


class State:
    __slots__ = ("mode", "esc", "quotes", "stack", "level", "content", "word", "word_done", "last")

    def __init__(self, level=0):
        self.mode, self.esc, self.quotes, self.stack, self.level = "code", False, 0, "", level
        self.content, self.word, self.word_done, self.last = False, "", False, ""

    def key(self):
        return (self.mode, self.esc, self.quotes, self.stack, self.level, self.content, self.word if not self.word_done or self.word in BLOCK else "", self.word_done, self.last)

    def clone(self):
        s = State()
        for name in State.__slots__:
            setattr(s, name, getattr(self, name))
        return s

    def can_end_line(self):
        if self.mode != "code" or self.stack or not self.content or self.quotes:
            return False
        return (self.word in BLOCK) == (self.last == ":")

    def feed(self, text):
        """Consume marker text. Returns False as soon as it breaks a rule."""
        for sym in _SYM.findall(text):
            if len(sym) > 1:                                   # a marker
                if sym == "<LF>":
                    if self.mode != "doc":
                        return False
                    continue
                if not self.can_end_line():
                    return False
                if sym == "<IND>":
                    if self.last != ":":
                        return False
                    self.level += 1
                else:
                    if self.last == ":":
                        return False
                    self.level -= 0 if sym == "<NL>" else int(sym[4])
                    if self.level < 0:
                        return False
                self.content, self.word, self.word_done, self.last = False, "", False, ""
                continue
            if self.mode == "code":
                if sym == '"' or sym == "'":
                    self.mode, self.quotes = ("s2", 1) if sym == '"' else ("s1", 0)
                elif sym in OPEN:
                    self.stack += sym
                elif sym in CLOSE:
                    if not self.stack or OPEN.index(self.stack[-1]) != CLOSE.index(sym):
                        return False
                    self.stack = self.stack[:-1]
                elif sym == "\n":
                    return False
                if sym != " ":
                    self.content, self.last = True, sym
                    if not self.word_done:
                        if sym.isalnum() or sym == "_":
                            self.word += sym
                        else:
                            self.word_done = True
                elif self.word:
                    self.word_done = True
            elif self.mode == "doc":
                self.quotes = self.quotes + 1 if sym == '"' and not self.esc else 0
                self.esc = sym == "\\" and not self.esc
                if self.quotes == 3:
                    self.mode, self.quotes = "code", 0
            else:                                              # one-line string
                if sym == "\n":
                    return False
                if self.esc:
                    self.esc = False
                elif sym == "\\":
                    self.esc, self.quotes = True, 0
                elif self.mode == "s2" and sym == '"':
                    if self.quotes == 1:                       # "" so far: empty string, or the start of a docstring
                        self.quotes = 2
                        self.mode = "code2"
                        continue
                    self.mode, self.quotes = "code", 0
                elif self.mode == "s1" and sym == "'":
                    self.mode = "code"
                else:
                    self.quotes = 0
                continue
            if self.mode == "code2":                           # just saw "": a third quote opens a docstring
                pass
        return True


def _fix_code2(state, sym):
    """Resolve the state right after two double quotes."""
    if sym == '"':
        state.mode, state.quotes = "doc", 0
        return True
    state.mode, state.quotes = "code", 0
    return False


_feed_inner = State.feed


def feed(self, text):
    for sym in _SYM.findall(text):
        if self.mode == "code2":
            if _fix_code2(self, sym):
                continue
        if not _feed_inner(self, sym):
            return False
    return True


State.feed = feed


class Filter:
    """Masks for a vocabulary of marker-text pieces. eos_id is allowed where a line could end."""

    def __init__(self, pieces, eos_id):
        self.pieces, self.eos_id, self.cache = pieces, eos_id, {}

    def allowed(self, state):
        key = state.key()
        if key not in self.cache:
            ok = []
            for g, piece in enumerate(self.pieces):
                if g == self.eos_id:
                    ok.append(state.mode == "code" and state.can_end_line())
                elif piece is None:
                    ok.append(False)
                else:
                    ok.append(state.clone().feed(piece))
            self.cache[key] = ok
        return self.cache[key]
