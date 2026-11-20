"""Syntax filter as tables: the form of lsllm/filter.py that the LS CPU can run. Explained in docs/FILTER.md.

Every token id has two bytes: a class (25 classes: what the token is for brackets, strings and line breaks) and
a word column (what it does to the first word of a line). The state is five small numbers. Before a token is
chosen the state is reduced to one of 17 "abstract states"; a token of class c may come if
ALLOW[abstract state][c] is 1. After the choice, step() updates the state with two table lookups (LEX, WORD)
and a few tests on the class. Characters are looked at only here, when the tables are built.

Rules: a closing bracket must match the innermost open one (at most MAX_DEPTH are open); no line break inside
brackets or inside a one-line string; inside a triple-quoted string only the docstring line markers; after a
line that ends with ":" the next line break is <IND>, otherwise <IND> is forbidden; a line that begins with a
block keyword must end with ":" and no other line may; <DEDk> never goes below level 0; no empty line; the
text ends only where a line may end. In code there is no "#", no backslash and no space at the start of a line.
"""
BLOCK = {"def", "if", "elif", "else", "for", "while", "try", "except", "finally", "with", "class", "async"}
MAX_DEPTH = 8                                   # open brackets; a ninth may not be opened
ROW, WORD_ROW = 32, 64                          # bytes per row of the exported tables (25 classes, 49 word states)

# Token classes. OPEN + k and CLOSE + k are the brackets ( [ { and ) ] } for k = 0, 1, 2; DED1 + k - 1 is <DEDk>.
NEVER, END, NL, IND, DED1, DOCLINE, OPEN, CLOSE, CLOSE_COLON = 0, 1, 2, 3, 4, 9, 10, 13, 16
SQ, DQ, TQ, BACKSLASH, HASH, COLON, SPACE, PLAIN = 17, 18, 19, 20, 21, 22, 23, 24
N_CLASS, N_ABSTRACT = 25, 17
FIXED = {"(": OPEN, "[": OPEN + 1, "{": OPEN + 2, ")": CLOSE, "]": CLOSE + 1, "}": CLOSE + 2, "):": CLOSE_COLON,
         "'": SQ, '"': DQ, '"""': TQ, "\\": BACKSLASH, "#": HASH, ":": COLON, ",": PLAIN, ", ": PLAIN, "()": PLAIN}

# Lexer states: 0 code; 1, 2 just after '' or ""; 3..8 in a one-line string; 9..16 in a triple-quoted string.
# fresh = just opened, esc and doce = after a backslash, doc0..doc2 = that many closing quotes seen.
LEX_STATES = [("code", "")] + [(name, q) for name in ("empty", "fresh", "str", "esc", "doc0", "doc1", "doc2", "doce")
                               for q in "'\""]

# States of the first word of a line. 0..12: the line begins with a block keyword (0: the word is over, as after
# `if `; 1..12: the word so far is exactly that keyword, as after `else`). 13: it does not. 14: nothing but
# spaces yet. 15 and up: the word so far is the beginning of a keyword (`whi`), the next token decides.
WORDS = ["<keyword>"] + sorted(BLOCK) + ["<other>", ""] + sorted({k[:n] for k in BLOCK for n in range(1, len(k))})
KEYWORD, OTHER, LINE_START = 0, len(BLOCK) + 1, len(BLOCK) + 2


def lex_char(state, ch):
    """The string automaton for one character: a quote, a backslash, or "x" for any other character."""
    name, q = state
    if name in ("code", "empty"):
        if name == "empty" and ch == q:
            return ("doc0", q)                          # the third quote in a row opens a triple-quoted string
        return ("fresh", ch) if ch in "'\"" else ("code", "")
    if name in ("esc", "doce"):                         # the character after a backslash is taken as it is
        return ("str" if name == "esc" else "doc0", q)
    if ch == "\\":
        return ("esc" if name in ("fresh", "str") else "doce", q)
    if name in ("fresh", "str"):
        if ch != q:
            return ("str", q)
        return ("empty", q) if name == "fresh" else ("code", "")
    if ch != q:                                         # doc0, doc1, doc2: count the closing quotes
        return ("doc0", q)
    return {"doc0": ("doc1", q), "doc1": ("doc2", q), "doc2": ("code", "")}[name]


def lex_token(state, c):
    """Lexer state after a token of class c: the automaton run over the characters of the token."""
    if c == DOCLINE:                                    # a new line starts the quote count and the escape again
        return ("doc0", state[1]) if state[0].startswith("doc") else state
    for ch in {SQ: "'", DQ: '"', TQ: '"""', BACKSLASH: "\\"}.get(c, "x"):
        state = lex_char(state, ch)
    return state


def word_token(w, piece):
    """State of the first word of the line after a token with the text `piece` (in code)."""
    for ch in piece:
        if w in (KEYWORD, OTHER):
            break                                       # the first word is over: nothing changes any more
        if ch.isalnum() or ch == "_":
            longer = WORDS[w] + ch
            w = WORDS.index(longer) if longer in WORDS else OTHER
        elif ch != " " or w != LINE_START:              # the word ends here: was it a keyword?
            w = KEYWORD if w < OTHER else OTHER
    return w


def allow(a, c):
    """May a token of class c come in abstract state a?"""
    if c == NEVER:
        return False
    if c in (NL, END):
        return a >= 11                                  # the line may end
    if c == IND:
        return a == 10                                  # a keyword line that ends with ":"
    if DED1 <= c < DOCLINE:
        return a >= 11 + c - DED1 + 1                   # <DEDk> needs level k or more
    if c == DOCLINE:
        return a == 1
    if a < 2:
        return True                                     # in a string every text may come
    if c in (HASH, BACKSLASH):
        return False                                    # in code: no comment, no line continuation
    if c == SPACE:
        return a != 8                                   # no space at the start of a line
    if OPEN <= c < CLOSE:
        return not (a < 8 and a % 2)                    # not at the maximum depth
    if CLOSE <= c <= CLOSE_COLON:
        return a < 8 and (a - 2) // 2 == (c - CLOSE) % 3
    return True


LEX = [[LEX_STATES.index(lex_token(state, c)) for c in range(N_CLASS)] for state in LEX_STATES]
ALLOW = [[int(allow(a, c)) for c in range(N_CLASS)] for a in range(N_ABSTRACT)]


def token_class(piece):
    """Class of a token that is not a marker, from its text."""
    if piece in FIXED:
        return FIXED[piece]
    if set(piece) & set("()[]{}'\":,#\\"):
        raise ValueError("token %r mixes a structural character with text: the tokenizer is not variant B+" % piece)
    if not piece.strip(" "):
        return SPACE
    if piece[0] == " ":
        raise ValueError("token %r begins with a space and goes on: not supported" % piece)
    return PLAIN


class State:
    """What the filter remembers: 5 bytes and the bracket stack (2 bits per level) on the CPU."""

    def __init__(self):
        self.lex = 0                # lexer state, an index of LEX_STATES
        self.word = LINE_START      # state of the first word of the line, an index of WORDS
        self.colon = 0              # 1: the last token of the line that is not spaces is ":" or "):"
        self.level = 0              # indentation level
        self.stack = []             # types (0, 1, 2) of the open brackets, the innermost last


class TableFilter:
    """Which tokens of a tokenizer may come next. `tok` is a lsllm.tok.Tok of variant B+."""

    def __init__(self, tok, triple_quotes=True):
        """triple_quotes=False: the model may not open a triple-quoted string (it then never writes a docstring of
        its own, which is where a small model tends to run on); closing one that the prompt opened stays possible."""
        self.tok = tok
        # the ALLOW table of this filter: abstract state 1 is "inside a triple-quoted string"
        self.allow = [[int(bit and (triple_quotes or c != TQ or a == 1)) for c, bit in enumerate(row)] for a, row in enumerate(ALLOW)]
        special = {tok.unk_id: NEVER, tok.bos_id: NEVER, tok.eos_id: END, tok.nl_id: NL, tok.ind_id: IND,
                   tok.lf_id: DOCLINE, tok.blank_id: DOCLINE}
        special.update((t, DED1 + k) for k, t in enumerate(tok.ded_ids))
        self.cls = [special[t] if t in special else token_class(piece) for t, piece in enumerate(tok.pieces)]
        # what every token does to each word state; tokens that do the same share one column of the table
        effects = [tuple(word_token(w, "" if t in special else piece) for w in range(len(WORDS)))
                   for t, piece in enumerate(tok.pieces)]
        self.word = sorted(set(effects))                # word[column][word state] = next word state
        self.col = [self.word.index(effect) for effect in effects]
        self.masks = [[bool(row[c]) for c in self.cls] for row in self.allow]

    def start(self, marker_text):
        """State after a prompt given as marker text."""
        state = State()
        for token in self.tok.sp.encode(marker_text):
            self.step(state, token)
        return state

    def start_ids(self, ids):
        """State after a prompt given as token ids (works for a prompt that ends in the middle of a line)."""
        state = State()
        for token in ids:
            self.step(state, token)
        return state

    def index(self, state):
        """Abstract state 0..16: all that `allowed` needs to know about the state."""
        if state.lex >= 9:
            return 1                                    # in a triple-quoted string
        if state.lex >= 3:
            return 0                                    # in a one-line string
        if state.stack:                                 # in brackets: type of the innermost, and is the stack full
            return 2 + 2 * state.stack[-1] + (len(state.stack) == MAX_DEPTH)
        if state.word == LINE_START:
            return 8
        if (state.word < OTHER) != state.colon:
            return 9                                    # keyword line without ":", or ":" without a keyword
        return 10 if state.colon else 11 + min(state.level, 5)

    def allowed(self, state):
        """One bool per token id: may this token come next? (The list is shared: do not change it.)"""
        return self.masks[self.index(state)]

    def step(self, state, token):
        """Advance `state` over one token, in place. Only the two bytes of the token are used."""
        c = self.cls[token]
        if c <= END:
            return                                      # a control token is no text
        if state.lex < 3:                               # in code
            if NL <= c < DOCLINE:                       # a line break: new level, new line
                state.level = max(0, state.level + (0, 1, -1, -2, -3, -4, -5)[c - NL])
                state.word, state.colon = LINE_START, 0
            else:
                if OPEN <= c < CLOSE and len(state.stack) < MAX_DEPTH:
                    state.stack.append(c - OPEN)
                elif CLOSE <= c <= CLOSE_COLON and state.stack:
                    state.stack.pop()
                state.word = self.word[self.col[token]][state.word]
                if c != SPACE:
                    state.colon = int(c in (COLON, CLOSE_COLON))
        state.lex = LEX[state.lex][c]

    def token_table(self):
        """Two bytes per token id t: byte t is its class, byte vocab_size + t is its word column."""
        return bytes(self.cls + self.col)

    def allow_table(self):
        """ALLOW as bytes: byte ROW * a + c is 1 if class c may come in abstract state a."""
        return b"".join(bytes(row).ljust(ROW, b"\0") for row in self.allow)

    def lexer_table(self):
        """LEX as bytes: byte ROW * s + c is the lexer state after a token of class c in lexer state s."""
        return b"".join(bytes(row).ljust(ROW, b"\0") for row in LEX)

    def word_table(self):
        """WORD as bytes: byte WORD_ROW * k + w is the word state after a token of column k in word state w."""
        return b"".join(bytes(row).ljust(WORD_ROW, b"\0") for row in self.word)

    def sizes(self):
        sizes = {"classes": N_CLASS, "abstract_states": N_ABSTRACT, "lexer_states": len(LEX),
                 "word_states": len(WORDS), "word_columns": len(self.word), "state_bytes": 5 + MAX_DEPTH // 4,
                 "token_table_bytes": len(self.token_table()), "allow_table_bytes": len(self.allow_table()),
                 "lexer_table_bytes": len(self.lexer_table()), "word_table_bytes": len(self.word_table())}
        sizes["total_table_bytes"] = sum(v for k, v in sizes.items() if k.endswith("_table_bytes"))
        return sizes
