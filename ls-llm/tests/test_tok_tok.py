"""Tok on the two bootstrap tokenizers: exact round trip, levels, decoding in parts, cutting at any token."""
import json
from pathlib import Path

import pytest
import sentencepiece as spm

from lsllm.data.tokenize_corpus import STRUCTURAL, VARIANTS
from lsllm.structure import MARKERS, MarkerDecoder
from lsllm.tok import Tok

TOK_DIR = Path(__file__).resolve().parents[1] / "data/tok"


@pytest.fixture(scope="module")
def texts():
    """5,000 training functions and all validation functions of the bootstrap set."""
    out = []
    for name, limit in (("train", 5000), ("val", None)):
        lines = open(TOK_DIR / "bootstrap" / (name + ".jsonl")).readlines()[:limit]
        out += [json.loads(line)["text"] for line in lines]
    return out


@pytest.fixture(scope="module", params=[256, 512])
def tok(request):
    tok = Tok(TOK_DIR / ("boot%d" % request.param) / "tok.model")
    assert tok.vocab_size == request.param
    return tok


@pytest.fixture(scope="module")
def encoded(tok, texts):
    """(text, ids) for every text; the two functions with a dedent of more than 5 levels are left out."""
    pairs = []
    for text in texts:
        try:
            pairs.append((text, tok.encode(text)))
        except ValueError:
            pass
    assert len(pairs) >= 5000 and len(pairs) >= len(texts) - 5
    return pairs


def test_vocabulary(tok):
    structure = [tok.blank_id, tok.nl_id, tok.ind_id, *tok.ded_ids, tok.lf_id]
    assert len(set(structure + [tok.unk_id, tok.bos_id, tok.eos_id])) == 12
    assert [tok.pieces[t] for t in structure[1:]] == MARKERS
    assert tok.pieces[tok.blank_id] == "<LF>"
    assert len(tok.pieces) == tok.vocab_size
    for code in range(32, 127):                     # every printable ASCII character is a token of its own
        assert chr(code) in tok.pieces
    for t, piece in enumerate(tok.pieces):          # only the marker tokens show a marker
        assert (t in structure) == any(marker in piece for marker in MARKERS)


def test_refuses_a_model_without_blank_token(tmp_path):
    spm.SentencePieceTrainer.train(sentence_iterator=iter(["def f(x):<IND>return x"] * 50), vocab_size=24,
                                   hard_vocab_limit=False, model_prefix=str(tmp_path / "plain"),
                                   user_defined_symbols=MARKERS, minloglevel=2)
    with pytest.raises(ValueError):
        Tok(tmp_path / "plain.model")


def test_structural_tokens_stay_single(tok):
    """Variant B+: a piece with a structural character is one of the fixed structural tokens, nothing else."""
    assert json.loads((TOK_DIR / ("boot%d" % tok.vocab_size) / "meta.json").read_text())["variant"] == "B+"
    for piece in tok.pieces:
        if set(piece) & set(STRUCTURAL):
            assert piece in VARIANTS["B+"]
    assert [tok.pieces[t] for t in range(12, 28)] == VARIANTS["B+"]


def test_structural_strings_are_one_token(tok):
    pieces = [tok.pieces[t] for t in tok.encode('def f(a, b):\n    """Doc."""\n    return g(), [a, b]', bos=False)]
    for single in ("(", ", ", "):", '"""', "()", "[", "]"):
        assert single in pieces
    assert "," not in pieces and ")" not in pieces and '"' not in pieces


def test_round_trip(tok, encoded):
    for text, ids in encoded:
        assert tok.decode(ids) == text
        assert ids[0] == tok.bos_id and tok.unk_id not in ids and tok.eos_id not in ids
        assert max(ids) < tok.vocab_size


def test_bos_eos(tok):
    text = "def f(x):\n    return x"
    plain = tok.encode(text, bos=False)
    assert tok.encode(text) == [tok.bos_id] + plain
    assert tok.encode(text, bos=True, eos=True) == [tok.bos_id] + plain + [tok.eos_id]
    assert tok.decode([tok.bos_id] + plain + [tok.eos_id]) == text


def test_level_tracking(tok, encoded):
    breaks = {tok.nl_id, tok.ind_id, *tok.ded_ids}
    for text, ids in encoded:
        lines = text.split("\n")
        level, row = 0, 0
        for t in ids:
            level = tok.level_after([t], level)
            if t in breaks or t in (tok.lf_id, tok.blank_id):
                row += 1
            if t in breaks:                         # the new line starts with exactly `level` indentation steps
                assert lines[row].startswith("    " * level) and lines[row][4 * level] != " "
        assert row == len(lines) - 1
        assert level == tok.level_after(ids)


def test_level_rules(tok):
    assert tok.level_after([tok.ind_id, tok.ind_id, tok.nl_id, tok.lf_id, tok.blank_id]) == 2
    assert tok.level_after([tok.ded_ids[2]], level=5) == 2
    assert tok.level_after([tok.ded_ids[4]], level=1) == 0          # never below 0
    assert tok.decode([tok.ded_ids[0]], level=2) == "\n    "
    assert tok.decode([tok.lf_id, tok.blank_id, tok.lf_id], level=1) == "\n    \n\n    "


def test_decoding_token_by_token_equals_batch(tok, encoded):
    for text, ids in encoded:
        level, parts = 0, []
        decoder, stream = MarkerDecoder(), []
        for t in ids:
            parts.append(tok.decode([t], level))
            level = tok.level_after([t], level)
            stream.append(decoder.feed(tok.pieces[t]))              # the same through the marker-text decoder
        assert "".join(parts) == "".join(stream) == text


def test_cut_at_any_token(tok, encoded):
    for text, ids in encoded[::12]:
        for cut in range(len(ids) + 1):
            head, rest = ids[:cut], ids[cut:]
            assert tok.decode(head) + tok.decode(rest, level=tok.level_after(head)) == text


def test_empty_docstring_line(tok):
    text = 'def f(x):\n    """One.\n\n    Two."""\n    return x'
    ids = tok.encode(text)
    assert ids.count(tok.blank_id) == 1 and ids.count(tok.lf_id) == 1
    assert ids.index(tok.blank_id) + 1 == ids.index(tok.lf_id)
    assert tok.decode(ids) == text


@pytest.mark.parametrize("text", ["x = 'caf\u00e9'", "x = '\u2581'", "x = '<IND>'", "x = (1,\n     2)", "\tx = 1"])
def test_encode_rejects(tok, text):
    with pytest.raises(ValueError):
        tok.encode(text)


def test_control_symbol_is_plain_text(tok):
    text = "x = '<BLANK><s></s><unk>'"
    assert tok.decode(tok.encode(text)) == text
