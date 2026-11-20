"""Table filter on real functions: acceptance, agreement with the character filter, the exported bytes, random text.

    .venv/bin/python tests/test_filter_table_corpus.py      prints the measurements on 6,000 functions (about 40 s)
"""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsllm import filter as char_filter
from lsllm import filter_table as ft
from lsllm.structure import to_markers
from lsllm.tok import Tok

TOK_FILE = "data/tok/f512a/tok.model"
FILES = ["data/final/f512a/human_val.jsonl", "data/final/f512a/human_train.jsonl"]
ADDED = {"comment_or_backslash", "leading_space", "end_after_colon"}     # rules the character filter does not have


def load_texts(limit):
    texts = []
    for name in FILES:
        with open(name) as f:
            for line in f:
                if len(texts) < limit:
                    texts.append(json.loads(line)["text"])
    return texts


def measure_acceptance(flt, texts):
    """Share of functions that the table filter lets pass from the first token to the end of the text."""
    tok, accepted, by_char, rejected = flt.tok, 0, 0, {}
    for text in texts:
        state = char_filter.State()
        by_char += state.feed(to_markers(text)) and state.can_end_line()
        ids = tok.encode(text, bos=False) + [tok.eos_id]
        state = flt.start("")
        for g, t in enumerate(ids):
            if not flt.allowed(state)[t]:
                line = tok.decode(ids[:g]).split("\n")[-1].strip()
                if line.startswith(("match ", "case ")):
                    why = "match statement"
                elif len(state.stack) == ft.MAX_DEPTH:
                    why = "more than %d open brackets" % ft.MAX_DEPTH
                else:
                    why = "unexplained"
                rejected.setdefault(why, []).append(line)
                break
            flt.step(state, t)
        else:
            accepted += 1
    return {"functions": len(texts), "accepted": accepted, "accepted_by_character_filter": by_char,
            "rejected": {why: len(lines) for why, lines in rejected.items()}, "rejected_lines": rejected}


def reason(c, char_allows, a, triple):
    """Why the two filters disagree about a token of class c in abstract state a."""
    if char_allows and c in (ft.HASH, ft.BACKSLASH):
        return "comment_or_backslash"
    if char_allows and c == ft.SPACE and a == 8:
        return "leading_space"
    if char_allows and c == ft.END and a == 10:
        return "end_after_colon"
    if char_allows and ft.OPEN <= c < ft.CLOSE and a in (3, 5, 7):
        return "depth_limit"
    if triple:
        return "triple_single_quote"                    # the character filter does not know ''' strings
    return "unexplained"


def measure_agreement(flt, texts):
    """At every position of every function: is the allowed set of the two filters the same? If not, why not?

    A function is followed up to the first token that one of the two filters refuses.
    """
    tok, cache = flt.tok, {}
    count = {"positions": 0, "identical": 0, "identical_apart_from_added_rules": 0,
             "table_more_permissive": 0, "table_more_restrictive": 0}
    for text in texts:
        ids = tok.encode(text, bos=False) + [tok.eos_id]
        chars, state, seen = char_filter.State(), flt.start(""), ""
        for t in ids:
            a = flt.index(state)
            key = (chars.key(), a, "'''" in seen)
            if key not in cache:
                mask = [chars.can_end_line() if u == tok.eos_id else bool(piece) and chars.clone().feed(piece)
                        for u, piece in enumerate(tok.pieces)]
                differ = [u for u in range(tok.vocab_size) if mask[u] != flt.masks[a][u]]
                cache[key] = ({reason(flt.cls[u], mask[u], a, key[2]) for u in differ},
                              any(not mask[u] for u in differ), any(mask[u] for u in differ))
            reasons, permissive, restrictive = cache[key]
            count["positions"] += 1
            count["identical"] += not reasons
            count["identical_apart_from_added_rules"] += reasons <= ADDED
            count["table_more_permissive"] += permissive
            count["table_more_restrictive"] += restrictive
            for why in reasons:
                count[why] = count.get(why, 0) + 1
            if not flt.masks[a][t] or not (chars.can_end_line() if t == tok.eos_id else chars.feed(tok.pieces[t])):
                break
            flt.step(state, t)
            seen += tok.pieces[t]
    return count


def byte_step(flt, vocab, s, t):
    """The step as the CPU makes it: it reads only the exported bytes. s = [lex, word, colon, level, stack]."""
    tokens, lexer, word = flt.token_table(), flt.lexer_table(), flt.word_table()
    c, k = tokens[t], tokens[vocab + t]
    if c <= 1:
        return
    if s[0] < 3:
        if c < 9:
            s[3] = max(0, s[3] + (0, 1, -1, -2, -3, -4, -5)[c - 2])
            s[1], s[2] = 14, 0
        else:
            if 10 <= c < 13 and len(s[4]) < 8:
                s[4].append(c - 10)
            elif 13 <= c <= 16 and s[4]:
                s[4].pop()
            s[1] = word[64 * k + s[1]]
            if c != 23:
                s[2] = int(c in (16, 22))
    s[0] = lexer[32 * s[0] + c]


def byte_index(s):
    """The abstract state as the CPU computes it."""
    lex, word, colon, level, stack = s
    if lex >= 3:
        return int(lex >= 9)
    if stack:
        return 2 + 2 * stack[-1] + (len(stack) == 8)
    if word == 14:
        return 8
    if (word < 13) != colon:
        return 9
    return 10 if colon else 11 + min(level, 5)


PAIR = {")": "(", "]": "[", "}": "{"}


def scan(text):
    """A checker that knows nothing of tokens or tables: reads plain text character by character.

    Returns (error or None, open brackets, open quote). Errors: a closing bracket that does not match, a line
    break inside brackets or inside a one-line string, an empty line, indentation that is not a multiple of 4,
    a deeper line that does not follow a ":" and a ":" that is not followed by a deeper line.
    """
    stack, quote, indent, last, g = [], "", 0, "", 0
    while g < len(text):
        ch = text[g]
        if quote:
            if ch == "\\":
                g += 1                                  # the next character is escaped
            elif text.startswith(quote, g):
                g += len(quote) - 1
                quote = ""
            elif ch == "\n" and len(quote) == 1:
                return "line break in a string", stack, quote
        elif ch in "'\"":
            quote = ch * 3 if text.startswith(ch * 3, g) else ch
            g += len(quote) - 1
            last = ch
        elif ch == "\n":
            spaces = len(text) - g - 1 - len(text[g + 1:].lstrip(" "))
            if stack:
                return "line break in brackets", stack, quote
            if not last:
                return "empty line", stack, quote
            if spaces % 4 or spaces > indent + 4 or (spaces > indent) != (last == ":"):
                return "indentation", stack, quote
            indent, last = spaces, ""
            g += spaces
        else:
            if ch in "([{":
                stack.append(ch)
            elif ch in PAIR and (not stack or stack.pop() != PAIR[ch]):
                return "bracket mismatch", stack, quote
            if ch != " ":
                last = ch
        g += 1
    return None, stack, quote


def random_text(flt, header, seed, steps=200, masked=True):
    """Text made of random tokens after a header (marker text): a random allowed class, then a random token of it.

    Line breaks, colons and closing brackets are chosen 10 times as often as opening brackets, triple quotes and
    the end of the text, and every second line begins with a block keyword: so blocks inside blocks appear.
    """
    tok, rng = flt.tok, random.Random(seed)
    state, ids = flt.start(header), tok.sp.encode(header)
    keywords = [t for t in range(tok.vocab_size) if flt.word[flt.col[t]][ft.LINE_START] < ft.OTHER]    # `if `, `else`, ..
    weight = [1.0] * ft.N_CLASS
    for c in range(ft.N_CLASS):
        if c in (ft.END, ft.TQ) or ft.OPEN <= c < ft.CLOSE:
            weight[c] = 0.3
        elif c < ft.DOCLINE or c == ft.COLON or ft.CLOSE <= c <= ft.CLOSE_COLON:
            weight[c] = 3.0
    for _ in range(steps):
        mask = flt.allowed(state) if masked else [c != ft.NEVER for c in flt.cls]
        classes = sorted({flt.cls[t] for t in range(tok.vocab_size) if mask[t]})
        c = rng.choices(classes, [weight[c] for c in classes])[0]
        t = rng.choice([t for t in range(tok.vocab_size) if mask[t] and flt.cls[t] == c])
        if masked and state.word == ft.LINE_START and state.lex == 0 and not state.stack and rng.random() < 0.5:
            t = rng.choice(keywords)
        if t == tok.eos_id:
            return tok.decode(ids), state, True
        flt.step(state, t)
        ids.append(t)
    return tok.decode(ids), state, False


@pytest.fixture(scope="module")
def flt():
    if not all(os.path.exists(name) for name in [TOK_FILE] + FILES):
        pytest.skip("tokenizer or corpus files are missing")
    return ft.TableFilter(Tok(TOK_FILE))


@pytest.fixture(scope="module")
def texts():
    return load_texts(6000)


def test_real_functions_pass(flt, texts):
    m = measure_acceptance(flt, texts)
    print({k: v for k, v in m.items() if k != "rejected_lines"})
    assert m["functions"] == 6000
    assert m["accepted"] >= 0.999 * m["functions"]                  # measured: 5,997 of 6,000
    assert m["accepted"] >= m["accepted_by_character_filter"]       # measured: 5,984 for lsllm/filter.py
    assert set(m["rejected"]) <= {"match statement"}                # every rejection is explained


def test_agreement_with_the_character_filter(flt, texts):
    m = measure_agreement(flt, texts)
    print(m)
    assert m["positions"] > 900_000
    assert m.get("unexplained", 0) == 0                             # every difference has a known reason
    assert m["identical_apart_from_added_rules"] >= 0.999 * m["positions"]      # measured: 99.97 percent
    assert 0.35 * m["positions"] < m["identical"] < 0.45 * m["positions"]       # identical only inside strings
    assert m["table_more_permissive"] <= m.get("triple_single_quote", 0) < 0.001 * m["positions"]
    assert m.get("depth_limit", 0) < 0.0001 * m["positions"]


def test_exported_bytes_give_the_same_filter(flt, texts):
    tok = flt.tok
    for text in texts[:300]:
        state, s = flt.start(""), [0, 14, 0, 0, []]
        for t in tok.encode(text, bos=False):
            flt.step(state, t)
            byte_step(flt, tok.vocab_size, s, t)
            assert s == [state.lex, state.word, state.colon, state.level, state.stack]
            assert byte_index(s) == flt.index(state)


def test_the_checker_finds_errors():
    for text, error in (("x = (1]", "bracket mismatch"), ("x = 1)", "bracket mismatch"), ("x = (1,\n2)", "line break in brackets"),
                        ("x = 'a\nb'", "line break in a string"), ("if x:\ny = 1", "indentation"),
                        ("x = 1\n    y = 2", "indentation"), ("if x:\n  y = 1", "indentation"), ("x = 1\n\ny = 2", "empty line")):
        assert scan(text)[0] == error, text
    good = 'def f(a):\n    """Doc (text\n\n    more]"""\n    if a:\n        return [a, {1: \')\'}]\n    return (a)'
    assert scan(good) == (None, [], "")


HEADERS = ["def f(a, b):", 'def total(numbers):<IND>"""Return the sum.', "def g(x):<IND>y = [x, (1, 2"]     # marker text


def test_random_tokens_under_the_mask_give_well_formed_text(flt):
    assert to_markers(HEADERS[0]) == HEADERS[0]
    ended = breaks = strings = deep = 0
    for seed in range(240):
        text, state, at_end = random_text(flt, HEADERS[seed % 3], seed)
        error, stack, quote = scan(text)
        assert error is None, (seed, error, text)
        assert len(stack) == len(state.stack) <= ft.MAX_DEPTH and bool(quote) == (state.lex >= 3), (seed, text)
        if at_end:                                      # the end of the text: nothing open, no block left empty
            assert not stack and not quote and text.rstrip(" ")[-1] != ":", (seed, text)
        ended, breaks = ended + at_end, breaks + text.count("\n")
        strings, deep = strings + text.count("'"), deep + text.count("\n        ")    # lines at level 2 or deeper
    assert ended > 5 and breaks > 1000 and strings > 1000 and deep > 100    # measured: 16, 1,545, 2,319, 162


def test_random_tokens_without_the_mask_fail_the_checker(flt):
    errors = sum(scan(random_text(flt, HEADERS[0], seed, masked=False)[0])[0] is not None for seed in range(40))
    assert errors >= 30                                 # measured: 40 of 40


if __name__ == "__main__":
    table = ft.TableFilter(Tok(TOK_FILE))
    functions = load_texts(6000)
    print(json.dumps(table.sizes(), indent=1))
    print(json.dumps(measure_acceptance(table, functions), indent=1))
    print(json.dumps(measure_agreement(table, functions), indent=1))
