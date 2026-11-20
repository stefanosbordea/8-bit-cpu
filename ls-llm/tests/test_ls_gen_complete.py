"""The complete program (lsllm.ls.gen.build with prompt=True): prompt feeding with the prefill skip, the stop rule,
the syntax filter and Gumbel-max sampling, against lsllm.intref.IntModel and lsllm.filter_table.TableFilter."""
import json

import numpy as np
import pytest

from lsllm.config import ModelConfig
from lsllm.filter_table import State, TableFilter
from lsllm.intref import IntModel
from lsllm.ls import cost, gen
from lsllm.tok import Tok
from test_ls_gen_model import SMALL, random_model

TOK = Tok("data/tok/f512a/tok.model")
TF = TableFilter(TOK)
WITH_TOK = dict(SMALL, vocab_size=512, max_seq_len=48)
PROMPTS = [[TOK.bos_id] + TOK.sp.encode(text) for text in (       # prompts as marker text (lsllm.structure)
    "def f(x):<IND>", "def last(items):<IND>k = len(items) - 1<NL>while items[k] % 2:<IND>k = k - 1<DED1>",
    "x = [1, (2, {3: '", 'def g(a, b):<IND>"""Add.<LF><LF>More', "class A:<IND>def m(self):<IND>if self.x:<IND>return (1")]


def run(spec, arrays, ids, sw=0, **options):
    """(tokens sent, CPU, Build) of the complete program on the prompt `ids`, run until HALT."""
    b = gen.build(spec, arrays, prompt=True, **options)
    cpu = gen.machine(b, sw=sw)
    gen.set_prompt(cpu.d, b, ids)
    cpu.run(limit=50_000_000)
    assert cpu.halted and not cpu.overflow
    return gen.sent(cpu), cpu, b


def reference(spec, arrays, ids, stop=False, syntax_filter=False, temperature=0.0, seed=0):
    """What the program must send: the loop of lsllm.intref.completer.IntCompleter for one sample, written with
    IntModel, TableFilter and Tok directly. The token that ends the text is sent too."""
    model = IntModel(spec, arrays)
    model.reset(seed=seed)
    state, level, new = State(), TOK.level_after(ids) if stop else 0, []
    for t in ids:
        u = model.step([t])
        TF.step(state, t)
    while True:
        allowed = np.array([TF.allowed(state)]) if syntax_filter else None
        t = int(model.choose(u, temperature, allowed)[0])
        new.append(t)
        if stop:
            level = TOK.level_after([t], level)
            if t == TOK.eos_id or level == 0 and any(m in TOK.pieces[t] for m in ("<NL>", "<DED")):
                return new
        if model.pos == model.cfg.max_seq_len:
            return new
        TF.step(state, t)
        u = model.step([t])


@pytest.mark.parametrize("norm, ffn", [("dyt", "swiglu"), ("rms", "relu"), ("rms", "swiglu")])
@pytest.mark.parametrize("shape", [SMALL, dict(SMALL, n_heads=4, n_kv_heads=2, vocab_size=300, max_seq_len=12)])
@pytest.mark.parametrize("length", [1, 2, 7])
def test_prompt_then_greedy_tokens_until_the_context_is_full(norm, ffn, shape, length):
    spec, arrays = random_model(length, norm, ffn, **shape)
    T, V = shape["max_seq_len"], shape["vocab_size"]
    ids = np.random.default_rng(length).integers(0, V, length).tolist()
    sent, cpu, b = run(spec, arrays, ids)
    model = IntModel(spec, arrays)
    assert sent == model.generate(ids, T - length + 1) and len(sent) == T - length + 1
    assert sent == reference(spec, arrays, ids)


def test_a_prompt_that_fills_the_context_gives_one_token_and_a_wrong_length_is_refused():
    spec, arrays = random_model(5, "rms", "swiglu", **SMALL)
    ids = list(range(16))
    assert run(spec, arrays, ids)[0] == IntModel(spec, arrays).generate(ids, 1)
    b = gen.build(spec, arrays, prompt=True)
    for bad in ([], list(range(17))):
        with pytest.raises(ValueError):
            gen.set_prompt(b.image, b, bad)
    gen.set_prompt(b.image, b, [3, 1])                           # the prompt can also be written into the image
    cpu = gen.machine(b)
    cpu.run()
    assert gen.sent(cpu) == IntModel(spec, arrays).generate([3, 1], 15)


def test_the_options_are_off_by_default_and_the_filter_needs_its_tables():
    spec, arrays = random_model(1, "dyt", "relu", **SMALL)
    assert not any(tag in gen.EXTRA for name, args, tag in gen.build(spec, arrays).code)
    with pytest.raises(ValueError):
        gen.build(spec, arrays, syntax_filter=True)
    with pytest.raises(ValueError):
        gen.build(spec, arrays, tf=TF)                           # tables of 512 tokens, a model of 32


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("syntax_filter", [False, True])
def test_stop_rule_and_syntax_filter_equal_the_completer_loop(seed, syntax_filter):
    spec, arrays = random_model(seed, "rms", "swiglu", **WITH_TOK)
    ids = PROMPTS[seed]
    sent, cpu, b = run(spec, arrays, ids, tf=TF, syntax_filter=syntax_filter)
    assert sent == reference(spec, arrays, ids, stop=True, syntax_filter=syntax_filter)


@pytest.mark.parametrize("seed", range(6))
def test_filter_state_after_random_tokens_equals_the_table_filter(seed):
    """Prompts of random token ids (also the control tokens, unmatched brackets, line breaks in strings)."""
    rng = np.random.default_rng(seed)
    spec, arrays = random_model(seed, "dyt", "relu", **dict(WITH_TOK, n_layers=1, max_seq_len=64))
    special = [t for t in range(512) if TF.cls[t] != 24]
    ids = [int(rng.choice(special)) if rng.random() < 0.6 else int(rng.integers(512)) for g in range(60)]
    sent, cpu, b = run(spec, arrays, ids, tf=TF, syntax_filter=True)
    assert sent == reference(spec, arrays, ids, stop=True, syntax_filter=True)
    state = State()
    for t in ids + sent[:-1]:
        TF.step(state, t)
    cells = b.mem["filter.rows"][0] + 256 * gen.ROWS
    lex, level, word, colon, depth = (cpu.d[cells + g] & 255 for g in (1, 2, 5, 6, 7))
    if len(sent) > 1 or True:                                    # the state the last choice was made in
        assert (lex, level, word, colon, depth) == (state.lex, state.level, state.word, state.colon, len(state.stack))


def test_filter_through_deep_brackets_and_a_row_in_which_nothing_is_allowed():
    """Nine opening brackets (the ninth is not pushed), then the model chooses under the filter."""
    spec, arrays = random_model(3, "dyt", "relu", **dict(WITH_TOK, n_layers=1))
    ids = [TOK.bos_id] + TOK.sp.encode("x = ((([[[{{{")
    sent, cpu, b = run(spec, arrays, ids, tf=TF, syntax_filter=True)
    assert sent == reference(spec, arrays, ids, stop=True, syntax_filter=True) and len(sent) > 8

    class Nothing(TableFilter):                                  # no token of the vocabulary is allowed in any state
        def token_table(self):
            return bytes([0] * 512 + self.col)
    spec, arrays = random_model(0, "rms", "swiglu", **WITH_TOK)
    plain = run(spec, arrays, ids, temperature=1.5)[0]           # sampled, so that the tokens vary
    assert run(spec, arrays, ids, tf=Nothing(TOK), syntax_filter=True, temperature=1.5)[0] == plain and len(set(plain)) > 3


def test_the_first_allowed_token_wins_when_every_logit_is_zero():
    """Token 0 is never allowed; with equal logits the reference takes the first allowed id."""
    spec, arrays = cost.dummy_model(ModelConfig(**dict(WITH_TOK, n_layers=1, max_seq_len=8)), {"norm": "dyt", "ffn": "relu"})
    spec["frac"] = {"u": 4}
    sent = run(spec, arrays, PROMPTS[0][:3], tf=TF, syntax_filter=True)[0]
    assert sent == reference(spec, arrays, PROMPTS[0][:3], stop=True, syntax_filter=True) and 0 not in sent


TEXTS = ["while x:<IND>try:<IND>y = (1, [2, {3: 'a\\'b'}])<DED1>except E:<IND>pass<DED2>elif z:<IND>return 1",
         'def f(a):<IND>"""Doc \\" x.<LF><LF>More ( text.<LF>"""<NL>for k in a:<IND>if k:<IND>with a as b:<IND>class C:<IND>'
         "async def g():<IND>else_ = ''<DED5>x = 1<DED1>finally:<IND>y = \"\" + '#'",
         "if a:<IND>if b:<IND>if c:<IND>if d:<IND>if e:<IND>if f:<IND>if g:<IND>x<DED5>y<DED2>z<NL>else :<IND>w<DED4>",
         "x = ((([[[{{{(1)}}}]]])))<NL>y = ':' \\ # (<NL>  whi le<NL>el if:<NL>tr y :<IND>"]


@pytest.mark.parametrize("seed", range(10))
def test_filter_state_and_allowed_classes_after_every_token(seed):
    """The program without `prompt` (the test bench feeds the tokens): after every token the state bytes, the
    bracket stack and the copied row of ALLOW equal TableFilter; the choice is the best allowed logit."""
    rng = np.random.default_rng(seed)
    spec, arrays = random_model(seed, "dyt", "relu", **dict(WITH_TOK, n_layers=1, max_seq_len=96))
    special = [t for t in range(512) if TF.cls[t] != 24]
    ids = TOK.sp.encode(TEXTS[seed]) if seed < len(TEXTS) else [
        int(rng.choice(special)) if rng.random() < 0.7 else int(rng.integers(512)) for g in range(96)]
    ids = ([TOK.bos_id] + ids)[:96]
    b = gen.build(spec, arrays, tf=TF, syntax_filter=True)
    cpu, state, level = gen.machine(b), State(), 0
    cells = b.mem["filter.rows"][0] + 256 * gen.ROWS
    table = 256 * [args[1] for name, args, tag in b.code if name == "LUT" and tag == "filter"][-1]     # ALLOWED
    seen = set()
    for t in ids:
        chosen = gen.run_token(cpu, b, t)
        TF.step(state, t)
        level = TOK.level_after([t], level)
        got = [cpu.d[cells + g] & 255 for g in (0, 1, 2, 5, 6, 7)]
        assert got == [level, state.lex, state.level, state.word, state.colon, len(state.stack)]
        top = 3 + 2 * state.stack[-1] + (len(state.stack) == 8) if state.stack else 0
        assert cpu.d[cells - 256 + len(state.stack)] == top
        allowed = TF.allowed(state)
        assert [cpu.d[table + c] != 0 for c in TF.cls] == allowed
        u = gen.peek(cpu, b, "scratch.u")
        assert chosen == int(np.where(allowed, u, -1).argmax())
        seen.add((TF.index(state), state.lex, len(state.stack)))
    assert len(seen) > 4


@pytest.mark.parametrize("temperature, seed", [(0.5, 0), (1.0, 7), (1.0, 0xBEEF), (2.5, 3)])
@pytest.mark.parametrize("syntax_filter", [False, True])
def test_sampling_equals_the_integer_reference(temperature, seed, syntax_filter):
    spec, arrays = random_model(seed, "rms", "swiglu", **WITH_TOK)
    ids = PROMPTS[seed % 5]
    options = dict(tf=TF, syntax_filter=True) if syntax_filter else {}
    sent, cpu, b = run(spec, arrays, ids, sw=seed, temperature=temperature, **options)
    assert sent == reference(spec, arrays, ids, stop=syntax_filter, syntax_filter=syntax_filter, temperature=temperature, seed=seed)
    if not syntax_filter:
        new = IntModel(spec, arrays).generate(ids, len(sent), temperature, seed)
        assert sent == new and len(set(sent)) > 3
        assert sent != run(spec, arrays, ids, sw=seed + 1, temperature=temperature)[0]      # other switches, other text


def test_sampling_on_a_small_vocabulary_that_is_not_a_multiple_of_256():
    shape = dict(SMALL, vocab_size=300)
    spec, arrays = random_model(2, "dyt", "swiglu", **shape)
    for seed in (1, 2):
        assert run(spec, arrays, [5, 299], sw=seed, temperature=1.5)[0] == IntModel(spec, arrays).generate([5, 299], 15, 1.5, seed)


def test_a_prefill_token_skips_the_final_norm_the_output_layer_the_choice_and_the_uart():
    spec, arrays = random_model(4, "rms", "swiglu", **SMALL)
    b = gen.build(spec, arrays, prompt=True)
    cpu = gen.machine(b)
    gen.set_prompt(cpu.d, b, [3, 4, 5])
    seen = []
    for g in range(4):
        before = list(cpu.count)
        cpu.run(stop=0)
        ran = {tag for a, (name, args, tag) in enumerate(b.code) if cpu.count[a] > before[a]}
        seen.append((cpu.cycles, len(cpu.tx), ran))
    V, D = 32, 16
    for g in (0, 1):                                             # two prefill tokens: nothing sent, no output layer
        assert seen[g][1] == 0 and not seen[g][2] & {"output", "argmax"}
    assert seen[2][1] == 2 and seen[3][1] == 4 and {"output", "argmax"} <= seen[2][2]
    cfg, options = ModelConfig(**SMALL), {"norm": "rms", "ffn": "swiglu"}
    prefill, parts, _ = cost.token_cycles(cfg, options, 1, prefill=True)
    generated, full, _ = cost.token_cycles(cfg, options, 1)
    assert seen[1][0] - seen[0][0] == prefill == 19200 and generated == 21632
    assert generated == cost.cycles_per_token(cfg, options, 1)[0] + 40       # two tests for a prompt token: 10 instructions
    assert parts["output"] == parts["argmax"] == 0 and full["output"] == 1312 and full["argmax"] == 20 * V + 32
    saved = {name: full[name] - parts[name] for name in full if full[name] != parts[name]}
    assert saved == {"output": 1312, "argmax": 672, "norm": 464, "book": 52, "prompt": -68} and sum(saved.values()) == 2432


def test_cycles_of_the_stop_rule_the_filter_and_sampling_on_a_small_model():
    cfg, options = ModelConfig(**dict(SMALL, vocab_size=512)), {"norm": "rms", "ffn": "swiglu"}
    plain, a, _ = cost.token_cycles(cfg, options, 3, stop=True)
    filtered, f, _ = cost.token_cycles(cfg, options, 3, syntax_filter=True)
    sampled, s, _ = cost.token_cycles(cfg, options, 3, stop=True, temperature=1.0)
    both, fs, _ = cost.token_cycles(cfg, options, 3, syntax_filter=True, temperature=1.0)
    prefill, pf, _ = cost.token_cycles(cfg, options, 3, prefill=True, syntax_filter=True)
    assert (plain, filtered, sampled, both, prefill) == (51188, 57936, 65536, 72284, 20136)
    assert plain == cost.cycles_per_token(cfg, options, 3)[0] + 172          # 43 instructions: prompt tests, class, level, stop test
    assert s["sample"] == fs["sample"] == 28 * 512 + 12 and sampled - plain == s["sample"]
    assert both - filtered == s["sample"] and "sample" not in pf
    forbidden = sum(not ok for ok in cost.tokenizer_filter().masks[11])      # a line that may end, at level 0
    assert filtered - plain == 12 * 512 + 604 and forbidden == 16            # per token id: class, table, branch
    assert pf["filter"] == 220 and a["argmax"] == 20 * 512 + 36
