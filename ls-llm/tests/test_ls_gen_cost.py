"""The cost model: its cycles are those of the generated program in the simulator, for any weights; the classic
comparisons and the what-if switches follow their one-line definitions; the memory regions add up."""
import math

import numpy as np
import pytest

from lsllm.config import ModelConfig
from lsllm.ls import cost, gen
from lsllm.ls.gen import PAGE, Emitter
from lsllm.ls.iss import L1
from test_ls_gen_model import SMALL, STORIES, random_model

OPTIONS = [{"norm": norm, "ffn": ffn} for norm in ("dyt", "rms") for ffn in ("swiglu", "relu")]


@pytest.mark.parametrize("options", OPTIONS, ids=lambda o: o["norm"] + "_" + o["ffn"])
@pytest.mark.parametrize("shape, pos", [(SMALL, 0), (SMALL, 9), (SMALL, 15), (STORIES, 255), (dict(STORIES, n_heads=2, n_kv_heads=2), 100)])
def test_cost_on_zero_weights_equals_the_cycles_with_random_weights(options, shape, pos):
    cfg = ModelConfig(**shape)
    total, parts = cost.cycles_per_token(cfg, options, pos)
    assert total == sum(parts.values())
    spec, arrays = random_model(pos, options["norm"], options["ffn"], **shape)
    b = gen.build(spec, arrays)
    cpu = gen.machine(b)
    rng = np.random.default_rng(pos)
    for name, (address, size) in b.mem.items():                 # random cache, random weights
        if name.startswith("kv."):
            cpu.view[address:address + size] = rng.integers(-128, 128, size)
    gen.set_position(cpu, b, pos)
    gen.run_token(cpu, b, 3)
    real = cost.price(b, cpu.count, cpu.elems)
    assert sum(real.values()) == cpu.cycles
    found = [cpu.count[a] for a, (name, args, tag) in enumerate(b.code) if tag == "argmax" and name == "MOV"]
    extra = sum(found) * 16 + (found[-1] * 8 if cfg.vocab_size % 256 else 0)      # new maxima met by the argmax
    assert real["argmax"] == parts["argmax"] + extra and 0 < sum(found) <= 127
    bit7 = 12 * (cpu.tx[-1] != cpu.d[b.mem["scratch.token"][0] + 1] & 255)        # the low byte had its bit 7 set
    assert real["book"] == parts["book"] + bit7
    assert all(real[name] == parts[name] for name in gen.COMPONENTS if name not in ("argmax", "book"))
    assert cpu.cycles == total + extra + bit7


def test_sending_a_token_costs_12_cycles_more_when_bit_7_of_its_low_byte_is_set():
    cfg = ModelConfig(vocab_size=300, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=4, max_seq_len=4)
    spec, arrays = cost.dummy_model(cfg, OPTIONS[1])
    cycles = {}
    for token in (5, 127, 128, 255, 256, 299):                   # the token whose row of E is not 0 wins the argmax
        arrays["E"][:] = 0
        arrays["E"][token] = 16
        arrays["NORM"][:] = 16
        b = gen.build(spec, arrays)
        cpu = gen.machine(b)
        assert gen.run_token(cpu, b, token) == token and cpu.tx == [0x80 | token >> 7, token & 0x7F]
        new_maximum = 16 if token < 256 else 24                  # one new maximum; 24 in the short last block
        cycles[token] = cpu.cycles - cost.cycles_per_token(cfg, OPTIONS[1], 0)[0] - new_maximum
    assert cycles == {5: 0, 127: 0, 128: 12, 255: 12, 256: 0, 299: 0}


def test_one_more_position_costs_one_more_key_per_head_and_one_more_product_per_value_row():
    cfg = ModelConfig()                                          # stories260K: H = 8, d = 8, D = 64, L = 5
    options = OPTIONS[0]
    for pos in (0, 14, 254):
        a, b = cost.cycles_per_token(cfg, options, pos)[1], cost.cycles_per_token(cfg, options, pos + 1)[1]
        assert b["scores"] - a["scores"] == 5 * 8 * (8 + 40 + 20)    # code and maximum, then the weight
        assert b["mix"] - a["mix"] == 5 * (64 + 8)                # 64 value rows and 8 sums of weights per layer
        assert all(a[name] == b[name] for name in a if name not in ("scores", "mix"))
    one = cost.cycles_per_token(ModelConfig(n_layers=1), options, 0)[1]
    two = cost.cycles_per_token(ModelConfig(n_layers=1, n_heads=16, n_kv_heads=4, dim=128), options, 0)[1]
    assert two["book"] - one["book"] == 8 * 212                  # a head more: its record (13 bytes) and 14 instructions


def test_classic_comparisons_follow_their_definitions():
    cfg = ModelConfig(**SMALL)
    for options in OPTIONS:
        b, count, elems = cost.profile(cfg, options, 7)
        total = cost.cycles_per_token(cfg, options, 7)[0]
        a = c = 0
        for address, (name, args, tag) in enumerate(b.code):     # price every executed instruction again, by hand
            if name == "MACL" or name == "MAC" and args[0]:
                n = 8 if name == "MACL" else args[0]             # LEN = position + 1
                assert elems[address] == n * count[address]
                a, c = a + 4 * n * count[address], c + 8 * n * count[address]
            else:
                a, c = a + 4 * count[address], c + 4 * count[address]
        assert cost.classic(cfg, options, 7) == (a, c) and total < a < c
        products, streams, singles = cost.multiply_adds(cfg, options, 7)
        alone = [n for n, (name, args, tag) in zip(count, b.code) if (name, args) == ("MAC", (0,)) or name == "MUL"]
        assert products == sum(elems) and singles == sum(alone)  # MUL: the reciprocal product of every value row
    assert cost.classic_words(ModelConfig(), OPTIONS[0]) > 4096  # classic (a) does not fit the ROM


def emitted(build, data=None):
    """Counts of a small program made of primitives: (emitter, count, elems)."""
    em = Emitter()
    build(em)
    em.emit("HALT")
    cpu = L1(em.words(), np.zeros(1 << 16, dtype=np.int8))
    cpu.len = 100
    cpu.run()
    return em, cpu.count, cpu.elems


def total(triple, **switches):
    return sum(cost.price(*triple, **switches).values())


def test_what_if_switches_on_single_loops():
    X, Y, W = PAGE + 16, 2 * PAGE, 3 * PAGE
    rows = emitted(lambda em: em.rows(W, X, Y, 10, 64, 7))
    setup = total(rows) - 10 * (64 + 24)
    assert setup == 4 * 13                                       # preload 2, P1 3, P2 4, page of P0 2, LDI R3, HALT
    assert total(rows, fused=True) == setup + 10 * (64 + 3)
    assert total(rows, lanes=2) == setup + 10 * (32 + 24) and total(rows, lanes=4) == setup + 10 * (16 + 24)
    assert total(rows, ordinary=3) == 3 * 13 + 10 * (64 + 4 + 5 * 3)
    assert total(rows, fused=True, lanes=4, ordinary=3) == 3 * 13 + 10 * (16 + 3)
    assert total(rows, stream_lut=True, signed_branch=True) == total(rows)
    odd = emitted(lambda em: em.rows(W, X, Y, 3, 300, 7))        # MAC #255 then MAC #45
    assert total(odd, lanes=4) == total(odd) - 3 * (300 - math.ceil(255 / 4) - math.ceil(45 / 4))
    table = emitted(lambda em: em.rows(W, X, Y, 10, 64, 7, lut=2))
    assert total(table) == setup + 10 * (64 + 28) and total(table, stream_lut=True) == setup + 10 * (64 + 24)
    assert total(table, fused=True) == setup + 10 * (64 + 3)
    residual = emitted(lambda em: em.residual(W, X, Y, 10, 64))
    assert total(residual, fused=True) == total(residual) - 10 * (64 + 28) + 10 * (64 + 3 + 4)    # LD of x[j] stays
    mapped = emitted(lambda em: em.table_map(X, Y, 64, 1))
    assert total(mapped, stream_lut=True) == total(mapped) - 64 * 16 + 64 + 4 and total(mapped, fused=True) == total(mapped)
    # one or two products per value: not a fused row. Saved by 4 lanes: nothing on a classic MAC, 1 cycle on each
    # of the 8 MAC #2 of a RoPE head, 50 - 13 cycles on the sum of squares of an rms norm
    for loop, saved in ((lambda em: em.product(X, W, Y, 50, 3), 0), (lambda em: em.rope(X, 8, W, Y), 8),
                        (lambda em: em.rms(X, Y, 50, 1, 3, 3, X - 1), 37)):
        elem = emitted(loop)
        assert total(elem, fused=True) == total(elem) and total(elem, lanes=4) == total(elem) - saved
    mix = emitted(lambda em: em.mix(X, X + 300, Y, W, [(0, 16)], 3, 9, True))        # LEN = 100
    assert total(mix, fused=True) == total(mix)                  # a value row requantises twice: not a fused row
    assert total(mix, lanes=4) == total(mix) - 17 * 75           # 17 streams of 100
    scores = emitted(lambda em: em.scores(X, X + 50, Y, 8, 3, X + 40, 5, 6))         # 256 keys, d = 8
    setup = total(scores) - 256 * (8 + 40 + 20)
    assert total(scores, fused=True) == setup + 256 * (8 + 3 + 16 + 20)              # LD, SUB, LUT, ADD stay
    # no SUB, LUT, ADD and one read of the maximum; 12: the three instructions that stand for the argmax of the logits
    assert total(scores, hw_argmax=True) == setup + 256 * (8 + 28 + 20) + 4 + 12
    assert total(scores, stream_lut=True) == setup + 256 * (8 + 40 + 1) + 4          # the weights: a table stream
    assert total(scores, signed_branch=True) == total(scores)
    best = emitted(lambda em: em.argmax(Y, 512, X, 5))
    em = best[0]
    em.code[:] = [(name, args, "argmax") for name, args, tag in em.code]
    assert total(best) - total(best, signed_branch=True) == 512 * 4 and total(best, stream_lut=True) == total(best)
    assert total(best, hw_argmax=True) == 12 and total(best, hw_argmax=True, ordinary=3) == 9


@pytest.mark.parametrize("heads, kv_heads", [(8, 4), (4, 2), (2, 2)])
def test_what_ifs_never_cost_more_and_all_together_is_the_cheapest(heads, kv_heads):
    cfg = ModelConfig(n_heads=heads, n_kv_heads=kv_heads)
    for options in OPTIONS:
        base = cost.cycles_per_token(cfg, options, 255)[0]
        each = {name: cost.cycles_per_token(cfg, options, 255, **switches)[0] for name, (switches, text) in cost.L2.items()}
        assert all(value < base for value in each.values())
        assert each["all together"] == min(each.values()) and each["4 MAC lanes"] < each["2 MAC lanes"]


@pytest.mark.parametrize("heads, kv_heads", [(8, 4), (4, 2), (2, 2)])
def test_memory_regions(heads, kv_heads):
    for ffn in ("swiglu", "relu"):
        cfg = ModelConfig(n_heads=heads, n_kv_heads=kv_heads, ffn=ffn)
        b = cost.profile(cfg, {"norm": "dyt", "ffn": ffn}, 0)[0]
        regions = cost.memory(b)
        assert regions["total"] == len(b.image) <= 1 << 19 and regions["unused (alignment)"] >= 0
        assert regions["KV cache: keys"] + regions["KV cache: values"] == cfg.kv_cache_bytes()
        assert regions["weight matrices"] + regions["embedding E"] == cfg.n_params() - (2 * cfg.n_layers + 1) * cfg.dim
        spans = sorted(b.mem.values())
        assert all(a + size <= b_ for (a, size), (b_, size_) in zip(spans, spans[1:]))        # nothing overlaps
        assert len(b.words) <= 4096


def test_report_has_every_table():
    text = cost.report()
    assert text.count("### Cycles per token at L1") == 3 and text.count("### What-if hardware changes") == 3
    assert text.count("| dyt | swiglu | 256 |") == 3 and "all together" in text
