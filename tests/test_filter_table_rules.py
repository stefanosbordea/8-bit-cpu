"""Table filter: one test per rule, on the tokenizer data/tok/f512a. Prompts are written as marker text."""
import pytest

from lsllm import filter_table as ft
from lsllm.tok import Tok

TOK_FILE = "data/tok/f512a/tok.model"
BREAKS = ["<NL>", "<IND>", "<DED1>", "<LF>", "<BLANK>", "<END>"]


@pytest.fixture(scope="module")
def flt():
    try:
        return ft.TableFilter(Tok(TOK_FILE))
    except OSError:
        pytest.skip("tokenizer %s is missing" % TOK_FILE)


def can(flt, marker_text, names):
    """Which of `names` (token texts, <BLANK> or <END>) may come after the prompt, as a string of 1 and 0."""
    ids = {"<END>": flt.tok.eos_id, "<BLANK>": flt.tok.blank_id}
    mask = flt.allowed(flt.start(marker_text))
    return "".join(str(int(mask[ids[name] if name in ids else flt.tok.sp.piece_to_id(name.replace(" ", "▁"))]))
                   for name in names)


def test_every_token_has_a_class_and_structure_never_hides_in_text(flt):
    tok = flt.tok
    assert len(flt.cls) == len(flt.col) == tok.vocab_size
    for t, piece in enumerate(tok.pieces):
        if flt.cls[t] in (ft.PLAIN, ft.SPACE):
            assert piece in (",", ", ", "()") or piece and not set(piece) & set("()[]{}'\":,#\\")
    assert [flt.cls[t] for t in (tok.unk_id, tok.bos_id, tok.eos_id, tok.nl_id, tok.ind_id, tok.lf_id, tok.blank_id)] \
        == [ft.NEVER, ft.NEVER, ft.END, ft.NL, ft.IND, ft.DOCLINE, ft.DOCLINE]
    assert [flt.cls[t] for t in tok.ded_ids] == [ft.DED1 + k for k in range(5)]
    for text in ("a(", "x: ", "it's", " x"):
        with pytest.raises(ValueError):
            ft.token_class(text)


def test_closing_bracket_must_match_the_innermost(flt):
    closers = [")", "]", "}", "):"]
    assert can(flt, "x = (", closers) == "1001"
    assert can(flt, "x = (a[", closers) == "0100"
    assert can(flt, "x = (a[{", closers) == "0010"
    assert can(flt, "x = (a[{1}", closers) == "0100"         # after the } the [ is innermost again
    assert can(flt, "x = (a[{1}]", closers) == "1001"
    assert can(flt, "x = (a[{1}])", closers) == "0000"       # nothing is open
    assert can(flt, "x = ", closers) == "0000"
    assert can(flt, "", closers) == "0000"
    assert can(flt, "def f(a", closers) == "1001"
    assert can(flt, "x = f()", closers) == "0000"            # the token () opens and closes
    assert can(flt, "x = (f()", closers) == "1001"


def test_brackets_inside_strings_do_not_count(flt):
    assert can(flt, "x = (')'", [")", "]"]) == "10"
    assert can(flt, "x = '('", ["<NL>", ")"]) == "10"
    assert can(flt, 'x = ["a]', ["]"]) == "1"                # inside the string: plain text
    assert can(flt, 'x = ["a]"', ["]", ")"]) == "10"


def test_at_most_max_depth_open_brackets(flt):
    openers = ["(", "[", "{"]
    assert can(flt, "x = " + "(" * (ft.MAX_DEPTH - 1), openers + [")"]) == "1111"
    assert can(flt, "x = " + "(" * ft.MAX_DEPTH, openers + [")", "]"]) == "00010"
    assert can(flt, "x = " + "[" * ft.MAX_DEPTH + "]", openers + ["]"]) == "1111"


def test_no_line_break_inside_brackets(flt):
    assert can(flt, "x = (a", BREAKS) == "000000"
    assert can(flt, "x = (a)[", BREAKS) == "000000"
    assert can(flt, "x = (a)", BREAKS) == "100001"
    assert can(flt, "def f(a, b", BREAKS) == "000000"


def test_no_line_break_inside_a_one_line_string(flt):
    for quote in "'\"":
        assert can(flt, "x = " + quote + "ab", BREAKS) == "000000"
        assert can(flt, "x = " + quote + "ab" + quote, BREAKS) == "100001"
        assert can(flt, "x = " + quote + "ab\\" + quote, BREAKS) == "000000"           # an escaped quote does not close
        assert can(flt, "x = " + quote + "ab\\\\" + quote, BREAKS) == "100001"         # an escaped backslash, then the quote
        assert can(flt, "x = " + quote + quote, BREAKS) == "100001"                    # the empty string
    assert can(flt, "x = 'a\"b", BREAKS + ["'"]) == "0000001"                          # the other quote is text


def test_inside_a_triple_quoted_string_only_docstring_line_markers(flt):
    for quotes in ('"""', "'''"):
        head = "def f():<IND>" + quotes + "Text"
        assert can(flt, head, BREAKS) == "000110"
        assert can(flt, head + "<LF>more (text", BREAKS + [")", "#", "\\"]) == "000110111"
        assert can(flt, head + quotes, BREAKS) == "101001"       # closed: a line break of code again (level 1)
        assert can(flt, head + quotes[0], BREAKS) == "000110"    # one or two quotes do not close
        assert can(flt, head + quotes[:2], BREAKS) == "000110"
        assert can(flt, head + "\\" + quotes, BREAKS) == "000110"          # an escaped quote is not the first of three
        assert can(flt, head + quotes[:2] + "<LF>" + quotes[0], BREAKS) == "000110"    # quotes on two lines do not add up
    assert can(flt, "x = 1", ["<LF>", "<BLANK>"]) == "00"        # never in code


def test_indent_exactly_after_a_line_that_ends_with_a_colon(flt):
    assert can(flt, "if x:", BREAKS) == "010000"
    assert can(flt, "def f(a):", BREAKS) == "010000"             # the token ):
    assert can(flt, "def f(a) -> int:", BREAKS) == "010000"
    assert can(flt, "else:  ", BREAKS) == "010000"               # spaces after the colon change nothing
    assert can(flt, "x = 1", BREAKS) == "100001"
    assert can(flt, "if x:<IND>return 1", BREAKS) == "101001"
    assert can(flt, "x = {1: 2}", BREAKS) == "100001"            # a colon inside brackets is not the end of the line
    assert can(flt, "x = ':'", BREAKS) == "100001"               # nor inside a string


KEYWORD_LINES = ["def f()", "if x", "elif x", "else", "for x in y", "while x", "try", "except", "except E", "finally",
                 "with x", "class A", "async def f()", "if(x)", "while(x)"]


@pytest.mark.parametrize("line", KEYWORD_LINES)
def test_a_line_that_begins_with_a_block_keyword_must_end_with_a_colon(flt, line):
    assert can(flt, line, BREAKS) == "000000"
    assert can(flt, line + ":", BREAKS) == "010000"
    assert can(flt, line + ": x", BREAKS) == "000000"
    assert can(flt, "if a:<IND>" + line, BREAKS) == "000000"
    assert can(flt, "if a:<IND>" + line + ":", BREAKS) == "010000"


@pytest.mark.parametrize("line", ["elsewhere = 1", "format(x)", "classes.append(x)", "tryit()", "define = 2", "iff",
                                  "x = 1 if y else 2", "return [x for x in y]", "_if = 1", "e = 1", "whil", "el",
                                  "@with", "for_ = 1", "asyncio.run(x)", "withx", "if_", "else_ = 0"])
def test_other_lines_end_without_a_colon(flt, line):
    assert can(flt, line, BREAKS) == "100001"
    assert can(flt, line + ":", BREAKS) == "000000"              # only a keyword line may end with a colon


def test_dedent_never_goes_below_level_zero(flt):
    deds = ["<DED1>", "<DED2>", "<DED3>", "<DED4>", "<DED5>"]
    assert can(flt, "x = 1", deds) == "00000"
    assert can(flt, "def f():<IND>x = 1", deds) == "10000"
    assert can(flt, "def f():<IND>if x:<IND>y = 1", deds) == "11000"
    assert can(flt, "def f():<IND>if x:<IND>y = 1<DED1>z = 2", deds) == "10000"
    deep = "".join("if x:<IND>" for _ in range(7)) + "y = 1"
    assert can(flt, deep, deds) == "11111"
    assert can(flt, deep + "<DED5>y = 1", deds) == "11000"
    assert can(flt, "def f():<IND>if x", deds) == "00000"        # and only where the line may end


def test_no_empty_line_and_no_space_at_the_start_of_a_line(flt):
    for head in ("", "x = 1<NL>", "if x:<IND>", "if x:<IND>y = 1<DED1>"):
        assert can(flt, head, BREAKS + [" ", "    ", "x", "("]) == "0000000011"
    assert can(flt, "x", [" ", "    "]) == "11"


def test_end_of_text_only_where_a_line_may_end(flt):
    for text, expect in (("x = 1", "1"), ("def f():<IND>return 1", "1"), ("x = (1", "0"), ("x = 'a", "0"),
                         ("if x", "0"), ("if x:", "0"), ("def f():<IND>", "0"), ('def f():<IND>"""Doc', "0"),
                         ('def f():<IND>"""Doc"""', "1"), ("", "0")):
        assert can(flt, text, ["<END>"]) == expect, text


def test_no_comment_and_no_backslash_in_code(flt):
    assert can(flt, "x = 1 ", ["#", "\\"]) == "00"
    assert can(flt, "", ["#", "\\"]) == "00"
    assert can(flt, "x = (1, ", ["#", "\\"]) == "00"
    assert can(flt, "x = '", ["#", "\\"]) == "11"


def test_control_tokens_are_never_allowed_and_something_always_is(flt):
    for mask in flt.masks:
        assert not mask[flt.tok.unk_id] and not mask[flt.tok.bos_id]
        assert sum(mask) > 400


def test_exported_tables(flt):
    vocab, sizes = flt.tok.vocab_size, flt.sizes()
    tokens, allow_bytes = flt.token_table(), flt.allow_table()
    assert len(tokens) == sizes["token_table_bytes"] == 2 * vocab
    assert len(allow_bytes) == sizes["allow_table_bytes"] == 32 * 17
    assert len(flt.lexer_table()) == sizes["lexer_table_bytes"] == 32 * 17
    assert len(flt.word_table()) == sizes["word_table_bytes"] == 64 * sizes["word_columns"]
    assert sizes["classes"] == 25 and sizes["abstract_states"] == 17 and sizes["word_states"] == 49
    assert max(tokens[:vocab]) < 25 and max(tokens[vocab:]) < sizes["word_columns"] <= 256
    assert sizes["total_table_bytes"] == len(tokens) + len(allow_bytes) + len(flt.lexer_table()) + len(flt.word_table())
    # the test the CPU makes: one class byte, one byte of the row of the abstract state
    for a in range(17):
        assert [bool(allow_bytes[32 * a + tokens[t]]) for t in range(vocab)] == flt.masks[a]


def test_start_gives_the_state_of_the_prompt_tokens(flt):
    from lsllm.structure import to_markers
    prompt = 'def f(a, b):\n    """Return the sum.\n\n    Both are numbers."""'
    by_token = flt.start("")
    assert vars(by_token) == {"lex": 0, "word": ft.LINE_START, "colon": 0, "level": 0, "stack": []}
    for t in flt.tok.encode(prompt, bos=False):
        flt.step(by_token, t)
    assert vars(flt.start(to_markers(prompt))) == vars(by_token)
    state = flt.start(to_markers(prompt) + "<NL>if a:<IND>return [a, (b")
    assert (state.lex, state.colon, state.level, state.stack, flt.index(state)) == (0, 0, 2, [1, 0], 2)
    before = vars(state).copy()
    flt.step(state, flt.tok.eos_id)                 # control tokens change nothing
    flt.step(state, flt.tok.bos_id)
    assert vars(state) == before
