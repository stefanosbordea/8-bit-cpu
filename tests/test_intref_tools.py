"""IntCompleter (stopping rules, truncation, batched sampling, registry), compare and overflow."""
import os

import numpy as np
import pytest
import torch

from lsllm.bench import load_completer
from lsllm.config import ModelConfig
from lsllm.intref import IntModel
from lsllm.intref.compare import compare
from lsllm.intref.completer import IntCompleter
from lsllm.intref.overflow import bounds, report
from lsllm.model import build
from lsllm.qat.calibrate import calibrate
from lsllm.qat.export import export

torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "def f(x):\n    y = x"      # ends inside the function, at indentation level 1


class CharTok:
    """Stand-in for lsllm.tok.Tok (INTERFACES section 2): one token per character, one per line-break marker."""

    def __init__(self):
        self.pieces = ["", "", "", "<NL>", "<IND>", "<DED1>"] + [chr(code) for code in range(32, 127)]
        self.vocab_size, self.bos_id, self.eos_id = len(self.pieces), 1, 2
        self.index = {piece: t for t, piece in enumerate(self.pieces) if piece}
        self.change = {self.index["<NL>"]: 0, self.index["<IND>"]: 1, self.index["<DED1>"]: -1}

    def encode(self, text, bos=True, eos=False):
        ids, level = [], 0
        for g, line in enumerate(text.split("\n")):
            body = line.lstrip(" ")
            new = (len(line) - len(body)) // 4
            if g:
                ids.append(self.index[["<DED1>", "<NL>", "<IND>"][new - level + 1]])
            level = new
            ids += [self.index[ch] for ch in body]
        return [self.bos_id] * bos + ids + [self.eos_id] * eos

    def decode(self, ids, level=0):
        out = []
        for t in ids:
            if t in self.change:
                level = max(0, level + self.change[t])
                out.append("\n" + "    " * level)
            else:
                out.append(self.pieces[t])
        return "".join(out)

    def level_after(self, ids, level=0):
        for t in ids:
            level = max(0, level + self.change.get(t, 0))
        return level


def tiny(vocab_size, directory, tokenizer="", seq_len=48):
    """(float model, directory of its integer model): random weights, enlarged so that the logits differ."""
    cfg = ModelConfig(vocab_size=vocab_size, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=24,
                      max_seq_len=seq_len, norm="dyt")
    torch.manual_seed(0)
    float_model = build(cfg)
    with torch.no_grad():
        for p in float_model.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    qat = calibrate(float_model, [torch.randint(0, vocab_size, (4, seq_len))])
    qat.tokenizer = tokenizer
    export(qat, str(directory))
    return float_model.eval(), str(directory)


class Scripted(IntModel):
    """The integer model, but its logits always choose the next token of a script."""

    def step(self, tokens, trace=None):
        u = np.zeros_like(super().step(tokens))
        u[:, self.script[self.pos - self.start]] = 127
        return u


@pytest.fixture
def completer(tmp_path):
    tok = CharTok()
    completer = IntCompleter(tiny(tok.vocab_size, tmp_path)[1], tok)
    completer.directory = str(tmp_path)         # for these tests only: the scripted model reads the arrays again
    return completer


def scripted_answer(completer, script_text, max_new_tokens=40, n=1, temperature=0.0, extra=()):
    """complete() with a model that continues PROMPT with script_text, then `extra`, then 'z' for ever."""
    tok = completer.tok
    ids = tok.encode(PROMPT)
    model = Scripted(completer.model.spec, {k: v for k, v in np.load(completer.directory + "/model.npz").items()})
    model.script = tok.encode(PROMPT + script_text)[len(ids):] + list(extra) + [tok.index["z"]] * 100
    model.start = min(len(ids), model.cfg.max_seq_len - min(max_new_tokens, model.cfg.max_seq_len - 1))
    completer.model = model
    return completer.complete(PROMPT, n, max_new_tokens, temperature, seed=0)


def test_stops_when_the_code_returns_to_level_0(completer):
    tok = completer.tok
    assert scripted_answer(completer, " + 1\n    return y", extra=[tok.index["<DED1>"]]) == [" + 1\n    return y"]
    assert scripted_answer(completer, " + 1", extra=[tok.eos_id]) == [" + 1"]              # end of text
    assert scripted_answer(completer, " + 1", n=3) == [" + 1" + "z" * 36] * 3               # greedy: n equal answers
    assert scripted_answer(completer, " + 1", max_new_tokens=6) == [" + 1zz"]


def test_long_prompt_keeps_its_last_tokens(completer):
    T = completer.model.cfg.max_seq_len
    before = completer.truncated
    answer = scripted_answer(completer, "", max_new_tokens=T - 5)                           # 5 positions for the prompt
    assert completer.truncated == before + 1 and answer == ["z" * (T - 5)]
    assert completer.model.pos == T - 1                                                     # the context was used up
    assert completer.info["notes"].startswith("no sliding window")                          # unlike TorchCompleter


def test_sampling_runs_all_samples_in_one_batch(completer):
    first = completer.complete(PROMPT, 5, 12, 1.0, seed=3)
    assert len(first) == 5 and len(set(first)) > 1
    assert first == completer.complete(PROMPT, 5, 12, 1.0, seed=3)
    assert first != completer.complete(PROMPT, 5, 12, 1.0, seed=4)
    for text in first:
        assert len(completer.tok.encode(PROMPT + text)) <= len(completer.tok.encode(PROMPT)) + 12
    # sample g of seed s uses the LFSR start of seed s + g: the first sample of seed 4 is the second of seed 3
    assert completer.complete(PROMPT, 1, 12, 1.0, seed=4)[0] == first[1]
    greedy = completer.complete(PROMPT, 2, 12, 0.0, seed=0)
    assert greedy[0] == greedy[1]
    assert completer.name == os.path.basename(completer.directory) and completer.info["variant"] == "int8"


def test_registry_and_tokenizer_of_the_model(tmp_path):
    """"int:<directory>" through the harness registry, with the tokenizer named in model.json."""
    tok_dir = os.path.join(ROOT, "data", "tok", "f512a")
    if not os.path.exists(os.path.join(tok_dir, "tok.model")):
        pytest.skip("data/tok/f512a is not on this machine")
    directory = tiny(512, tmp_path, tokenizer=tok_dir)[1]
    completer = load_completer("int:" + directory)
    assert isinstance(completer, IntCompleter)
    prompt = 'def add(a, b):\n    """Return the sum."""\n    return a'
    ids = completer.tok.encode(prompt)
    assert ids[0] == completer.tok.bos_id and completer.tok.decode(ids) == prompt
    assert completer.tok.level_after(ids) == 1
    texts = completer.complete(prompt, 2, 8, 0.0, seed=0)
    assert len(texts) == 2 and texts[0] == texts[1]


class Shifted:
    """Stand-in float model: the logits of an integer model, turned upside down at one position."""

    def __init__(self, directory, position):
        self.model, self.position = IntModel.load(directory), position

    def __call__(self, seq):
        logits = torch.tensor(self.model.forward(seq.numpy()) * 2.0 ** -self.model.f_l)
        if seq.shape[1] > self.position:
            logits[:, self.position] *= -1
        return logits, None


def test_compare_finds_the_first_difference(tmp_path):
    directory = tiny(40, tmp_path)[1]
    reference = IntModel.load(directory)
    prompts = [[1, 5, 9, 3], [1, 7, 7, 2, 8, 30]]
    same = compare(Shifted(directory, 10 ** 6), reference, prompts, max_new_tokens=8)       # never different
    assert same["same"] == 1.0 and same["identical"] == 2
    assert same["rows"] == [{"same": 1.0, "first": None}] * 2
    assert all(abs(row["difference"]) < 1e-9 for row in same["largest"])
    # position 6 predicts new token 3 of the first prompt (4 tokens) and new token 1 of the second (6 tokens)
    result = compare(Shifted(directory, 6), reference, prompts, max_new_tokens=8)
    assert result["rows"] == [{"same": 7 / 8, "first": 3}, {"same": 7 / 8, "first": 1}]
    assert result["same"] == 7 / 8 and result["identical"] == 0
    assert {(row["prompt"], row["new_token"]) for row in result["largest"][:2]} == {(0, 3), (1, 1)}
    assert abs(result["largest"][0]["difference"]) > 1.0


def test_compare_float_model_with_its_integer_model(tmp_path):
    float_model, directory = tiny(40, tmp_path)
    prompts = np.random.default_rng(0).integers(0, 40, (6, 5)).tolist()
    result = compare(float_model, IntModel.load(directory), prompts, max_new_tokens=10)
    assert result["same"] > 0.8 and len(result["largest"]) == 5
    assert abs(result["largest"][0]["difference"]) < 1.0                                    # nats


def test_overflow_report(tmp_path):
    directory = tiny(40, tmp_path, seq_len=12)[1]
    model = IntModel.load(directory)
    tokens = np.random.default_rng(0).integers(0, 40, (3, 12))
    rows = report(model, tokens)
    assert {row["stream"] for row in rows} == set(model.peak)                               # every stream has a bound
    for row in rows:
        assert 0 < row["seen"] <= row["bound"] < 2 ** 23, row
    worst = bounds(model)
    assert worst["u"] == abs(model.pre["c_l"]) + 128 * np.abs(model.E).sum(1).max()
    assert worst["l0.xa"] == 256 * 128 + 128 + 128 * np.abs(model.Wo_0).sum(1).max()
    assert worst["l0.a"] == 12 * 127 * 128                                                  # weights times values
    assert worst["l0.o"] == (1 << model.sh["sh_o"][0] >> 1) + 128 * 127
    assert worst["l0.code"] == np.abs(model.pre["c_s"][0]).max() + 8 * 128 * 128
