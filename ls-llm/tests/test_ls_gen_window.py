"""The attention window (cfg.window = W): the KV cache of the generated program is a ring of W slots. Logits and
tokens against lsllm.intref.IntModel with the same window; the cost stops growing at context W."""
import numpy as np
import pytest

from lsllm.config import ModelConfig
from lsllm.intref import IntModel
from lsllm.ls import cost, gen
from test_ls_gen_complete import PROMPTS, TF, WITH_TOK, reference, run
from test_ls_gen_model import SMALL, STORIES, random_model

OPTIONS = {"norm": "rms", "ffn": "swiglu"}


def side_by_side(spec, arrays, tokens):
    """Feed `tokens` to the program without options and to IntModel: same logits, same choice at every position.
    Returns the logits of every position."""
    b = gen.build(spec, arrays)
    cpu, model, out = gen.machine(b), IntModel(spec, arrays), []
    for tok in tokens:
        u = model.step(tok)
        chosen = gen.run_token(cpu, b, tok)
        assert (gen.peek(cpu, b, "scratch.u") == u).all() and chosen == int(np.argmax(u))
        out.append(tuple(u))
    assert not cpu.overflow
    return out, cpu, b


@pytest.mark.parametrize("window", [1, 2, 3, 5, 8, 16, 23, 24])
@pytest.mark.parametrize("norm, ffn, heads, kv_heads", [("rms", "swiglu", 2, 1), ("dyt", "relu", 4, 2)])
def test_logits_equal_the_integer_reference_through_several_wraps(window, norm, ffn, heads, kv_heads):
    shape = dict(SMALL, max_seq_len=24, n_heads=heads, n_kv_heads=kv_heads, window=window)
    spec, arrays = random_model(window, norm, ffn, **shape)
    tokens = np.random.default_rng(window).integers(0, 32, 24).tolist()
    out, cpu, b = side_by_side(spec, arrays, tokens)
    assert len(set(out)) > 12                                    # the logits differ from position to position
    cpu.run(stop=0)                                              # position 24 does not exist
    assert cpu.halted
    cfg = ModelConfig(**shape)                                   # what the ring holds at the end: the last W positions
    model = IntModel(spec, arrays)
    for tok in tokens:
        model.step(tok)
    for p in range(24 - window, 24):
        for g in range(kv_heads):
            K = gen.peek(cpu, b, "kv.K0.%d" % g).reshape(window, cfg.head_dim)
            assert (K[p % window] == model.K[0][0, g, p]).all()
            V = [gen.peek(cpu, b, "kv.V0.%d.%d" % (g, j))[p % window] for j in range(cfg.head_dim)]
            assert (V == model.V[0][0, g, :, p]).all()


def test_a_window_changes_the_logits_and_window_t_does_not():
    tokens = np.random.default_rng(1).integers(0, 32, 16).tolist()
    spec, arrays = random_model(1, "rms", "swiglu", **SMALL)
    logits = {}
    for window in (0, 5, 16):
        spec["config"]["window"] = window
        model = IntModel(spec, arrays)
        logits[window] = np.array([model.step(t) for t in tokens])
        side_by_side(spec, arrays, tokens)
    assert (logits[0] == logits[16]).all() and (logits[0][:5] == logits[5][:5]).all()
    assert (logits[0][5:] != logits[5][5:]).any()                # from position 5 on the window forgets keys


@pytest.mark.parametrize("heads, kv_heads, window, p", [(2, 1, 64, 200), (8, 4, 100, 254), (2, 2, 256, 254), (4, 2, 128, 127)])
def test_the_stories260k_shapes_late_in_the_context(heads, kv_heads, window, p):
    """Head sizes 8, 16, 32; the ring is filled with random values for the W - 1 positions before p; 2 tokens."""
    shape = dict(STORIES, n_heads=heads, n_kv_heads=kv_heads, window=window)
    spec, arrays = random_model(p, "rms", "swiglu", **shape)
    cfg, rng = ModelConfig(**shape), np.random.default_rng(p)
    b = gen.build(spec, arrays)
    cpu, model = gen.machine(b), IntModel(spec, arrays)
    model.pos = p
    for l in range(cfg.n_layers):
        for g in range(kv_heads):
            (ka, size), d = b.mem["kv.K%d.%d" % (l, g)], cfg.head_dim
            for t in range(max(0, p + 1 - window), p):
                model.K[l][0, g, t] = rng.integers(-128, 128, d)
                model.V[l][0, g, :, t] = rng.integers(-128, 128, d)
                cpu.view[ka + (t % window) * d:ka + (t % window + 1) * d] = model.K[l][0, g, t]
                for j in range(d):
                    cpu.view[b.mem["kv.V%d.%d.%d" % (l, g, j)][0] + t % window] = model.V[l][0, g, j, t]
    gen.set_position(cpu, b, p)
    for tok in rng.integers(0, 512, 2).tolist():
        u = model.step(tok)
        assert gen.run_token(cpu, b, tok) == int(np.argmax(u)) and (gen.peek(cpu, b, "scratch.u") == u).all()
    assert not cpu.overflow and len(b.words) <= 4096


@pytest.mark.parametrize("window", [1, 7, 16])
@pytest.mark.parametrize("options", [{}, dict(tf=TF), dict(tf=TF, syntax_filter=True), dict(tf=TF, syntax_filter=True, temperature=1.0),
                                     dict(temperature=1.5)], ids=["plain", "stop", "filter", "filter+sampling", "sampling"])
def test_the_complete_program_with_a_window(window, options):
    """Prompts of 5 to 18 tokens (longer than the window), then generation to the stop rule or the full context."""
    spec, arrays = random_model(window, "rms", "swiglu", **dict(WITH_TOK, window=window))
    for seed, ids in enumerate(PROMPTS[:3]):
        sent, cpu, b = run(spec, arrays, ids, sw=seed, **options)
        assert sent == reference(spec, arrays, ids, stop="tf" in options, syntax_filter=options.get("syntax_filter", False),
                                 temperature=options.get("temperature", 0.0), seed=seed)
    if not options:
        assert sent == IntModel(spec, arrays).generate(ids, 48 - len(ids) + 1) and len(ids) > window or window == 16


def test_records_of_a_ring_and_the_limits_of_the_window():
    cfg = ModelConfig(**dict(SMALL, max_seq_len=12, window=5))
    d = cfg.head_dim
    rows = gen.records(cfg, 0x3000, 0x900, np.zeros((12, d // 2)), np.zeros((12, d // 2)))
    assert rows.shape == (12, 10 + 2 * d)
    for p in range(12):
        key = 0x900 + (p % 5) * d
        assert rows[p, 3:9].tolist() == [min(p + 1, 5), 0, p, key >> 8, key & 255, 0] and rows[p, -1] == p % 5
    plain = gen.records(ModelConfig(**dict(SMALL, max_seq_len=12)), 0x3000, 0x900, np.zeros((12, d // 2)), np.zeros((12, d // 2)))
    assert plain.shape == (12, 9 + 2 * d) and plain[11, 3:6].tolist() == [12, 0, 11]
    for window in (-1, 17):
        with pytest.raises(ValueError):
            gen.build(*random_model(0, "dyt", "relu", **dict(SMALL, window=window)))


def test_cost_stops_growing_at_the_window_and_the_cache_is_smaller():
    cfg, whole = ModelConfig(n_heads=2, n_kv_heads=1, window=64), ModelConfig(n_heads=2, n_kv_heads=1)
    at = {p: cost.cycles_per_token(cfg, OPTIONS, p)[0] for p in (15, 62, 63, 64, 100, 255)}
    assert at[63] == at[64] == at[100] == at[255] and at[62] < at[63]
    # against the program without a window: one more record byte (LD, ST, DJNZ = 12 cycles), and a value row of
    # fewer than 256 bytes never ends at the end of its page, so the page of P1 is not set again for every row
    # (LD, MVH = 8 cycles for each of the 32 rows, 2 heads, 5 layers)
    change = 12 - 8 * 32 * 2 * 5
    for p in (15, 63):
        assert at[p] == cost.cycles_per_token(whole, OPTIONS, p)[0] + change
    assert at[255] == 514292 and cost.cycles_per_token(whole, OPTIONS, 255)[0] == 756840
    assert cost.multiply_adds(cfg, OPTIONS, 255) == cost.multiply_adds(whole, OPTIONS, 63)
    # classic (a) words: 3 MACL per layer (sum of the weights, two chunks of value rows) are W instead of T words,
    # and the two instructions that set the page again are gone in both chunks of every layer
    assert cost.classic_words(whole, OPTIONS) - cost.classic_words(cfg, OPTIONS) == 3 * 5 * (256 - 64) + 2 * 2 * 5
    small, big = (cost.memory(cost.profile(c, OPTIONS, 0)[0]) for c in (cfg, whole))
    L, G, d = 5, 1, 32
    assert small["KV cache: keys"] == L * G * 64 * d and big["KV cache: keys"] == L * G * 256 * d
    assert small["KV cache: values"] == L * G * d * 64 and big["KV cache: values"] == L * G * d * 256
    # keys of a layer: 8,192 -> 2,048 bytes; blocks are whole pages (4 x 4,096 less), the last one ends where its keys end
    assert (big["total"], small["total"]) == (374016, 351488) and big["total"] - small["total"] == 4 * 4096 + 6144
    for kw in (dict(), dict(prefill=True), dict(syntax_filter=True, temperature=1.0)):  # the complete program too
        assert cost.token_cycles(cfg, OPTIONS, 200, **kw)[0] == cost.token_cycles(cfg, OPTIONS, 63, **kw)[0] \
            == cost.token_cycles(whole, OPTIONS, 63, **kw)[0] + change


def test_cost_on_zero_weights_equals_the_cycles_with_random_weights_and_a_window():
    shape = dict(SMALL, window=6)
    cfg = ModelConfig(**shape)
    spec, arrays = random_model(2, "rms", "swiglu", **shape)
    b = gen.build(spec, arrays)
    cpu = gen.machine(b)
    for p, tok in enumerate(np.random.default_rng(0).integers(0, 32, 16).tolist()):
        before = cpu.cycles
        gen.run_token(cpu, b, tok)
        total, parts = cost.cycles_per_token(cfg, OPTIONS, p)
        assert parts["scores"] == cost.cycles_per_token(cfg, OPTIONS, min(p, 5))[1]["scores"]
        extra = cpu.cycles - before - total                      # new maxima of the argmax, bit 7 of the token id
        assert 0 <= extra <= 127 * 16 + 12 and extra % 4 == 0
