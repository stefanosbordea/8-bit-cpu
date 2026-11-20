"""Asking the board: a prompt table chosen by the switches (gen.build(prompts=...)) and prompts typed over the
UART with the proposed instruction RX (gen.build(uart_prompt=True)). Against lsllm.intref.IntModel and the loop of
lsllm.intref.completer.IntCompleter, written here with IntModel, TableFilter and Tok."""
import numpy as np
import pytest

from lsllm.filter_table import State
from lsllm.intref import IntModel
from lsllm.ls import gen
from lsllm.ls.isa import TABLE, TABLE_L1, decode, encode
from lsllm.ls.iss import L1
from test_ls_gen_complete import PROMPTS, TF, TOK, WITH_TOK
from test_ls_gen_model import SMALL, random_model

SHAPE = dict(WITH_TOK, window=7)
ALL = dict(tf=TF, syntax_filter=True, temperature=1.0)


def model(seed, **shape):
    spec, arrays = random_model(seed, "rms", "swiglu", **shape)
    spec["shifts"]["sh_e"] = 2
    return spec, arrays


def answer(spec, arrays, ids, stop=False, syntax_filter=False, temperature=0.0, seed=0, lfsr=None):
    """(tokens the program must send for the prompt `ids`, LFSR state afterwards). lfsr: the state to go on from."""
    ref = IntModel(spec, arrays)
    ref.reset(seed=seed)
    if lfsr is not None:
        ref.lfsr = lfsr
    state, level, new = State(), TOK.level_after(ids) if stop else 0, []
    for t in ids:
        u = ref.step([t])
        TF.step(state, t)
    while True:
        allowed = np.array([TF.allowed(state)]) if syntax_filter else None
        t = int(ref.choose(u, temperature, allowed)[0])
        new.append(t)
        if stop:
            level = TOK.level_after([t], level)
            if t == TOK.eos_id or level == 0 and any(m in TOK.pieces[t] for m in ("<NL>", "<DED")):
                return new, ref.lfsr
        if ref.pos == ref.cfg.max_seq_len:
            return new, ref.lfsr
        TF.step(state, t)
        u = ref.step([t])


def kinds(options):
    return dict(stop="tf" in options, syntax_filter=options.get("syntax_filter", False), temperature=options.get("temperature", 0.0))


# ---- the instruction RX and the simulator ----

def test_rx_encoding_and_a_cpu_that_waits_for_the_uart():
    assert encode("RX", 2, table=TABLE_L1) == 0x0808 and decode(0x0808, TABLE_L1)[0].name == "RX"
    assert "RX" not in TABLE                                     # not an instruction of the baseline CPU
    words = [encode(*insn, table=TABLE_L1) for insn in (("RX", 0), ("TX", 0), ("RX", 1), ("TX", 1), ("HALT",))]
    cpu = L1(words, rx=[5])
    cpu.run()
    assert cpu.waiting and not cpu.halted and cpu.tx == [5] and cpu.pc == 2 and cpu.cycles == 8
    cpu.run()                                                    # still nothing: it stays where it is, nothing is counted
    assert cpu.waiting and cpu.pc == 2 and cpu.cycles == 8
    cpu.feed([200])
    cpu.run()
    assert cpu.halted and not cpu.waiting and cpu.tx == [5, 200] and cpu.cycles == 20


def test_token_stream_with_end_marks():
    class Sent:
        tx = [0x80, 5, 0x83, 0x7F, 0xFF, 0x81, 0, 0xFF, 0x9F, 0x7F]
    assert gen.answers(Sent) == [[5, 511], [128], [4095]] and gen.sent(Sent) == [5, 511, 128, 4095]
    assert gen.frame([5, 511, 128, 4095]) == [0x80, 5, 0x83, 0x7F, 0x81, 0, 0x9F, 0x7F, 0xFF]


# ---- the prompt table ----

@pytest.mark.parametrize("options", [{}, dict(tf=TF), ALL, dict(temperature=1.5)], ids=["plain", "stop", "filter+sampling", "sampling"])
def test_the_switches_choose_the_prompt(options):
    spec, arrays = model(2, **SHAPE)
    b = gen.build(spec, arrays, prompts=PROMPTS[:4], **options)
    assert gen.top(b) > 0 and all(tag == "start" for name, args, tag in b.code[:gen.top(b)])
    for sw, n in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (63, 0), (64 + 2, 2), (0x80 + 1, 1), (0x300 + 3, 3), (0xFF, 0)):
        cpu = gen.machine(b, sw=sw)
        cpu.run(limit=50_000_000)
        assert cpu.halted and not cpu.overflow
        assert gen.sent(cpu) == answer(spec, arrays, PROMPTS[n], seed=sw, **kinds(options))[0], (sw, n)
    if not options:
        assert gen.sent(cpu) == IntModel(spec, arrays).generate(PROMPTS[0], 48 - len(PROMPTS[0]) + 1)


def test_other_prompts_are_written_into_the_table_of_an_image():
    spec, arrays = model(3, **SHAPE)
    b = gen.build(spec, arrays, prompts=PROMPTS[:3], tf=TF)
    new = [PROMPTS[4][:9], [300, 5, 129], PROMPTS[1]]
    gen.set_prompts(b.image, b, new)
    for sw in range(4):
        cpu = gen.machine(b, sw=sw)
        cpu.run(limit=50_000_000)
        assert gen.sent(cpu) == answer(spec, arrays, new[sw % 3 if sw < 3 else 0], stop=True)[0]
    for bad in ([], [[]], [[1]] * 65, [list(range(49))], PROMPTS[:5]):      # none, empty, too many, too long, too many bytes
        with pytest.raises(ValueError):
            gen.set_prompts(b.image, b, bad)
    with pytest.raises(ValueError):
        gen.build(spec, arrays, prompts=PROMPTS[:2], uart_prompt=True)


def same_program(b, plain, tail):
    """After its start-up piece, b is the complete program `plain` (apart from its last `tail` instructions and
    from branch targets), on the same memory map."""
    mine = [(name, tag) for name, args, tag in b.code[gen.top(b):len(b.code) - tail]]
    return mine == [(name, tag) for name, args, tag in plain.code[:len(plain.code) - (1 if tail else 0)]] and \
        all(b.mem[name] == plain.mem[name] for name in plain.mem)


@pytest.mark.parametrize("options", [{}, ALL], ids=["plain", "filter+sampling"])
def test_start_up_of_the_prompt_table_costs_92_cycles_and_24_per_token(options):
    spec, arrays = model(4, **SHAPE)
    b = gen.build(spec, arrays, prompts=[[5] * 3, [6] * 20], **options)
    plain = gen.build(spec, arrays, prompt=True, **options)
    cycles = []
    for sw in (0, 1):
        cpu = gen.machine(b, sw=sw)
        cpu.run(stop=gen.top(b))
        cycles.append(cpu.cycles)
    assert cycles == [92 + 24 * 3, 92 + 24 * 20] and gen.top(b) == 29 == len(b.words) - len(plain.words)
    assert b.mem["prompts.index"][1] == 256 and b.mem["prompts.table"][1] == 2 * 23
    assert 256 + 46 <= len(b.image) - len(plain.image) < 256 + 46 + 256      # the table is last in memory, at a multiple of 256
    assert same_program(b, plain, 0) and gen.top(plain) == 0


# ---- prompts over the UART ----

@pytest.mark.parametrize("options", [{}, dict(tf=TF), dict(tf=TF, syntax_filter=True), ALL], ids=["plain", "stop", "filter", "filter+sampling"])
def test_prompts_over_the_uart_give_the_answers_of_fresh_runs(options):
    spec, arrays = model(5, **SHAPE)
    b = gen.build(spec, arrays, uart_prompt=True, **options)
    cpu, lfsr, want = gen.machine(b, sw=9), None, []
    cpu.feed(gen.frame(PROMPTS[1]) + gen.frame(PROMPTS[3]))
    cpu.run(limit=100_000_000)
    for ids in (PROMPTS[1], PROMPTS[3]):
        new, lfsr = answer(spec, arrays, ids, seed=9, lfsr=lfsr, **kinds(options))     # the LFSR is not reset between answers
        want.append(new)
    assert cpu.waiting and not cpu.halted and not cpu.overflow
    assert gen.answers(cpu) == want + [[]] and cpu.tx.count(gen.END) == 2
    cpu.feed(gen.frame(PROMPTS[0]))                              # a third prompt later
    cpu.run(limit=100_000_000)
    assert gen.answers(cpu)[2] == answer(spec, arrays, PROMPTS[0], seed=9, lfsr=lfsr, **kinds(options))[0] and cpu.waiting
    if "temperature" not in options:                             # the second answer does not depend on the first
        alone = gen.machine(b, sw=9)
        alone.feed(gen.frame(PROMPTS[3]))
        alone.run(limit=100_000_000)
        assert gen.answers(alone)[0] == want[1] != want[0]


def test_uart_prompt_that_is_too_long_empty_or_arrives_in_pieces():
    spec, arrays = model(6, **SHAPE)
    b = gen.build(spec, arrays, uart_prompt=True, tf=TF, syntax_filter=True)
    long = (PROMPTS[1] * 5)[:60]                                 # 60 tokens, the buffer holds 48: the rest is dropped
    data = [gen.END] + gen.frame(long) + [gen.END, gen.END] + gen.frame(PROMPTS[2])
    cpu = gen.machine(b)
    for byte in data:                                            # byte by byte: the CPU waits in between
        cpu.feed([byte])
        cpu.run(limit=100_000_000)
        assert cpu.waiting
    assert gen.answers(cpu) == [answer(spec, arrays, long[:48], True, True)[0], answer(spec, arrays, PROMPTS[2], True, True)[0], []]
    assert len(gen.answers(cpu)[0]) == 1                         # the context was full after the prompt


def test_uart_prompts_on_a_small_model_without_tokenizer_tables():
    spec, arrays = model(7, **dict(SMALL, vocab_size=300, window=5))
    b = gen.build(spec, arrays, uart_prompt=True)
    cpu = gen.machine(b)
    prompts = [[299, 128, 5], [127], [255, 256, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 130]]
    for ids in prompts:
        cpu.feed(gen.frame(ids))
    cpu.run(limit=100_000_000)
    assert gen.answers(cpu) == [IntModel(spec, arrays).generate(ids, 16 - len(ids) + 1) for ids in prompts] + [[]]
    assert gen.peek(cpu, b, "prompt.ids")[[0, 1, 256, 257]].tolist() == [0, 0, 0, 130 - 256]      # high bytes, low bytes


@pytest.mark.parametrize("options, fixed, words", [({}, 100, 42), (dict(tf=TF), 148, 54), (ALL, 160, 57)], ids=["plain", "stop", "filter+sampling"])
def test_start_up_of_the_uart_prompt_costs_76_cycles_per_token(options, fixed, words):
    spec, arrays = model(8, **SHAPE)
    b, plain = gen.build(spec, arrays, uart_prompt=True, **options), gen.build(spec, arrays, prompt=True, **options)
    cycles = {}
    for n, ids in ((1, [5]), (2, [5, 6]), (10, [5] * 10), (-1, [200])):
        cpu = gen.machine(b)
        cpu.feed(gen.frame(ids))
        cpu.run(stop=gen.top(b))
        cycles[n] = cpu.cycles
    assert cycles[2] - cycles[1] == 76 and cycles[10] - cycles[1] == 9 * 76
    assert cycles[-1] - cycles[1] == 8                           # bit 7 of the low byte: two instructions more
    assert cycles[1] == fixed + 76 and gen.top(b) == words and len(b.words) - len(plain.words) == words + 2
    assert (b.image == plain.image).all() and same_program(b, plain, 3)      # no more data memory, the same program
    assert [name for name, args, tag in b.code[-3:]] == ["LDI", "TX", "JMP"] and plain.code[-1][0] == "HALT"


def test_filter_and_stop_state_are_reset_before_the_second_prompt():
    """The first prompt fills the context and ends inside a string, in three brackets, at level 3. The state bytes
    are read when the first token of the second answer has been sent: they are those of the second prompt alone."""
    spec, arrays = model(9, **SHAPE)
    b = gen.build(spec, arrays, uart_prompt=True, tf=TF, syntax_filter=True)
    first = ([TOK.bos_id] + TOK.sp.encode("if a:<IND>if b:<IND>if c:<IND>x = ((('abc" + " d" * 60))[:48]
    second = [TOK.bos_id] + TOK.sp.encode("x = [1")              # no line break: the word state comes from the reset
    stale, fresh = State(), State()
    for t in first:
        TF.step(stale, t)
    for t in second:
        TF.step(fresh, t)
    assert len(first) == 48 and (stale.lex, stale.level, len(stale.stack)) == (5, 3, 3) and TOK.level_after(first) == 3
    cpu = gen.machine(b)
    cpu.feed(gen.frame(first) + gen.frame(second))
    while gen.END not in cpu.tx or len(cpu.tx) < cpu.tx.index(gen.END) + 3:
        cpu.run(stop=gen.top(b), limit=100_000_000)              # one pass of the program per token
        assert not cpu.waiting
    cells = b.mem["filter.rows"][0] + 256 * gen.ROWS
    got = [cpu.d[cells + g] & 255 for g in (0, 1, 2, 5, 6, 7)]
    assert got == [0, fresh.lex, fresh.level, fresh.word, fresh.colon, len(fresh.stack)] == [0, 0, 0, 13, 0, 1]
    cpu.run(limit=100_000_000)
    assert cpu.waiting and gen.answers(cpu) == [answer(spec, arrays, first, True, True)[0], answer(spec, arrays, second, True, True)[0], []]
