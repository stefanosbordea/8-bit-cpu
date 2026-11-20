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
        word = self.word if not self.word_done or self.word in BLOCK else ""
        return (self.mode, self.esc, self.quotes, self.stack, self.level, self.content, word, self.word_done, self.last)

    def clone(self):
        s = State()
        for name in State.__slots__:
            setattr(s, name, getattr(self, name))
        return s

    def can_end_line(self):
        """May a line break or the end of the text come now?"""
        if self.mode != "code" or self.stack or not self.content:
            return False
        return (self.word in BLOCK) == (self.last == ":")

    def feed(self, text):
        """Consume marker text. Returns False as soon as it breaks a rule (the state is then unusable)."""
        for sym in _SYM.findall(text):
            if self.mode == "two":                             # just saw "": a third quote opens a docstring
                if sym == '"':
                    self.mode, self.quotes = "doc", 0
                    continue
                self.mode = "code"
            if len(sym) > 1:                                   # a marker
                if sym == "<LF>":
                    if self.mode != "doc":
                        return False
                    continue
                if not self.can_end_line() or (sym == "<IND>") != (self.last == ":"):
                    return False
                self.level += 1 if sym == "<IND>" else 0 if sym == "<NL>" else -int(sym[4])
                if self.level < 0:
                    return False
                self.content, self.word, self.word_done, self.last = False, "", False, ""
            elif self.mode == "code":
                if sym == "\n":
                    return False
                if sym == '"':
                    self.mode, self.quotes = "s2", 1
                elif sym == "'":
                    self.mode = "s1"
                elif sym in OPEN:
                    self.stack += sym
                elif sym in CLOSE:
                    if not self.stack or OPEN.index(self.stack[-1]) != CLOSE.index(sym):
                        return False
                    self.stack = self.stack[:-1]
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
            else:                                              # inside a one-line string
                if sym == "\n":
                    return False
                if self.esc:
                    self.esc = False
                elif sym == "\\":
                    self.esc, self.quotes = True, 0
                elif self.mode == "s2" and sym == '"':
                    self.mode = "two" if self.quotes == 1 else "code"
                    self.quotes = 0
                elif self.mode == "s1" and sym == "'":
                    self.mode = "code"
                else:
                    self.quotes = 0
        return True


class SyntaxFilter:
    """Which tokens of a tokenizer may come next. `tok` is a lsllm.tok.Tok (pieces = marker text of every token)."""

    def __init__(self, tok):
        self.pieces, self.eos_id, self.cache = tok.pieces, tok.eos_id, {}

    def start(self, marker_text):
        state = State()
        state.feed(marker_text)
        return state

    def allowed(self, state):
        key = state.key()
        if key not in self.cache:
            self.cache[key] = [state.can_end_line() if g == self.eos_id else bool(piece) and state.clone().feed(piece)
                               for g, piece in enumerate(self.pieces)]
        return self.cache[key]

    def step(self, state, token):
        if self.pieces[token]:
            state.feed(self.pieces[token])
