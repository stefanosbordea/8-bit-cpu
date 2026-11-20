"""TorchCompleter: stopping rules, truncation, batched sampling, determinism (with a stand-in tokenizer)."""
import pytest
import torch

from lsllm.config import ModelConfig
from lsllm.model import build
from lsllm.model.completer import TorchCompleter

PROMPT = "def f(x):\n    y = x"      # ends inside the function, at indentation level 1


class ByteTok:
    """Stand-in for lsllm.tok.Tok (INTERFACES section 2): one token per character, one per line-break marker."""

    def __init__(self):
        self.pieces = ["", "", "", "<NL>", "<IND>", "<DED1>", "<DED2>"] + [chr(code) for code in range(32, 127)]
        self.vocab_size, self.bos_id, self.eos_id = len(self.pieces), 1, 2
        self.index = {piece: t for t, piece in enumerate(self.pieces) if piece}
        self.step = {self.index["<NL>"]: 0, self.index["<IND>"]: 1, self.index["<DED1>"]: -1, self.index["<DED2>"]: -2}

    def encode(self, text, bos=True, eos=False):
        ids, level = [], 0
        for g, line in enumerate(text.split("\n")):
            body = line.lstrip(" ")
            new = (len(line) - len(body)) // 4
            if g:
                ids.append(self.index["<NL>" if new == level else "<IND>" if new > level else "<DED%d>" % (level - new)])
            level = new
            ids += [self.index[ch] for ch in body]
        return [self.bos_id] * bos + ids + [self.eos_id] * eos

    def decode(self, ids, level=0):
        out = []
        for t in ids:
            if t in self.step:
                level = max(0, level + self.step[t])
                out.append("\n" + "    " * level)
            else:
                out.append(self.pieces[t])
        return "".join(out)

    def level_after(self, ids, level=0):
        for t in ids:
            level = max(0, level + self.step.get(t, 0))
        return level


class Scripted:
    """Stands for the model: whatever it is given, token number g of the answer is script[g]."""

    def __init__(self, cfg, script, start):
        self.cfg, self.script, self.start, self.inputs = cfg, script, start, []

    def __call__(self, seq):
        self.inputs.append(seq.clone())
        logits = torch.zeros(seq.shape[0], seq.shape[1], self.cfg.vocab_size)
        logits[:, -1, self.script[len(self.inputs) - 1]] = 50.0      # one call per generated token
        return logits, None


@pytest.fixture
def completer(tmp_path):
    tok = ByteTok()
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=32, max_seq_len=64)
    (tmp_path / "tinyrun").mkdir()
    ckpt = {"config": cfg.to_dict(), "model": build(cfg).state_dict(), "step": 0, "val_loss": 9.9, "tok": "unused"}
    torch.save(ckpt, tmp_path / "tinyrun" / "ckpt.pt")
    return TorchCompleter(str(tmp_path / "tinyrun" / "ckpt.pt"), tok)


def scripted_answer(completer, prompt, script_text, max_new_tokens=40, n=1, temperature=0.0, extra=()):
    """Run complete() with a model that continues the prompt with script_text, then `extra`, then 'z' for ever."""
    tok = completer.tok
    ids = tok.encode(prompt)
    script = tok.encode(prompt + script_text)[len(ids):] + list(extra) + [tok.index["z"]] * 100
    context = min(len(ids), 64 - min(max_new_tokens, 63))
    completer.model = Scripted(completer.model.cfg, script, context)
    return completer.complete(prompt, n, max_new_tokens, temperature, seed=0)


def test_stops_when_a_line_break_returns_to_level_zero(completer):
    # the first character of the script continues the last prompt line
    answer = scripted_answer(completer, PROMPT, " + 1\n    if y:\n        return y\n    return 0\nprint(f(1))")
    assert answer == [" + 1\n    if y:\n        return y\n    return 0"]
    assert PROMPT + answer[0] == completer.tok.decode(completer.tok.encode(PROMPT + answer[0]))


def test_a_double_dedent_to_level_zero_stops_too(completer):
    assert scripted_answer(completer, PROMPT, "\n    if y:\n        return y\nprint(2)") == ["\n    if y:\n        return y"]


def test_stops_at_end_of_text_token(completer):
    assert scripted_answer(completer, PROMPT, " + 1", extra=[completer.tok.eos_id]) == [" + 1"]


def test_stops_after_max_new_tokens(completer):
    assert scripted_answer(completer, PROMPT, "", max_new_tokens=5) == ["zzzzz"]
    assert len(completer.model.inputs) == 5


def test_prompt_at_level_zero_ends_with_its_line(completer):
    assert scripted_answer(completer, "x = 1", " + 2\ny = 3") == [" + 2"]


def test_greedy_gives_n_equal_answers_from_one_row(completer):
    answer = scripted_answer(completer, PROMPT, " + 1\n    return y\nq", n=3)
    assert answer == [" + 1\n    return y"] * 3
    assert all(seq.shape[0] == 1 for seq in completer.model.inputs)


def test_long_prompt_keeps_its_last_tokens_and_is_counted(completer):
    tok = completer.tok
    long_prompt = "def f(x):\n    y = " + " + ".join(["x"] * 40)            # 172 tokens, the context holds 64
    ids = tok.encode(long_prompt)
    answer = scripted_answer(completer, long_prompt, " + 1\n    return y\nq", max_new_tokens=24)
    assert answer == [" + 1\n    return y"]                                # still the text after the whole prompt
    assert completer.truncated == 1
    assert completer.model.inputs[0][0].tolist() == ids[-63:]               # as much of the prompt as fits
    assert max(seq.shape[1] for seq in completer.model.inputs) <= 64        # the window slides, never grows

    scripted_answer(completer, PROMPT, " + 1\n    return y\nq", max_new_tokens=24)
    assert completer.truncated == 1                                         # a short prompt is not counted
    scripted_answer(completer, long_prompt, "", max_new_tokens=50)
    assert completer.truncated == 2 and completer.model.inputs[0].shape[1] == 63


def test_sampling_is_batched_stops_per_sample_and_repeats_for_a_seed(completer):
    model, sizes = completer.model, []

    def recording(seq):
        sizes.append(seq.shape[0])
        return model(seq)

    completer.model = recording
    recording.cfg = model.cfg
    first = completer.complete(PROMPT, 8, 40, 1.0, seed=7)
    assert len(first) == 8 and sizes[0] == 8 and sizes == sorted(sizes, reverse=True) and sizes[-1] < 8
    for text in first:              # no sample goes on after a line break back to column 0
        assert all(line.startswith("    ") for line in text.split("\n")[1:])
    assert completer.complete(PROMPT, 8, 40, 1.0, seed=7) == first
    assert completer.complete(PROMPT, 8, 40, 1.0, seed=8) != first
    assert len(set(first)) > 1


def test_prompt_that_the_tokenizer_cannot_reproduce_is_refused(completer):
    with pytest.raises(ValueError):
        completer.complete("def f(x):\n  return x", 1, 10, 0.0, seed=0)     # two spaces: not canonical


def test_name_and_tokenizer_check(completer, tmp_path):
    assert completer.name == "tinyrun" and completer.truncated == 0
    assert completer.info == {"params": completer.model.cfg.n_params(), "variant": "float", "filter": "off"}
    wrong = ByteTok()
    wrong.vocab_size += 1
    with pytest.raises(AssertionError):
        TorchCompleter(str(tmp_path / "tinyrun" / "ckpt.pt"), wrong)
