"""The embedding shift of SPEC C.1: x[k] = rqr(E[token][k], sh_e). The generated program against the formula and
against lsllm.intref.IntModel; sh_e = 0 or absent is the plain copy and the program of before."""
import numpy as np
import pytest

from lsllm.config import ModelConfig
from lsllm.intref import IntModel
from lsllm.ls import cost, gen
from test_ls_gen_complete import PROMPTS, TF, WITH_TOK, reference, run
from test_ls_gen_model import SMALL, random_model

EDGES = [-127, -126, -65, -64, -63, -4, -3, -2, -1, 0, 1, 2, 3, 63, 64, 126, 127, -1, -1, 1, 5, -5, 100, -100]


def model(seed, sh_e, **shape):
    """A random model with the embedding shift sh_e (None: the entry is absent) and edge values in row 0 of E."""
    spec, arrays = random_model(seed, "rms", "swiglu", **shape)
    if sh_e is not None:
        spec["shifts"]["sh_e"] = sh_e
    arrays["E"][0] = np.resize(EDGES, arrays["E"].shape[1])
    arrays["E"][1] = np.random.default_rng(seed).integers(-127, 128, arrays["E"].shape[1])
    return spec, arrays


@pytest.mark.parametrize("sh_e", [0, 1, 3, 7, 15])
@pytest.mark.parametrize("heads, window", [(2, 0), (4, 5)])
def test_embedding_row_and_logits_equal_the_formula_and_the_integer_reference(sh_e, heads, window):
    spec, arrays = model(sh_e, sh_e, **dict(SMALL, dim=24, n_heads=heads, window=window))
    b = gen.build(spec, arrays)
    cpu, ref = gen.machine(b), IntModel(spec, arrays)
    first_norm = next(a for a, (name, args, tag) in enumerate(b.code) if tag == "norm")
    for tok in [0, 1] + np.random.default_rng(sh_e).integers(0, 32, 14).tolist():
        gen.set_token(cpu, b, tok)
        cpu.run(stop=first_norm)                                 # the embedding is done, nothing else yet
        row = arrays["E"][tok].astype(np.int64)
        want = (row + ((1 << sh_e) >> 1)) >> sh_e                # rqr: add half, shift toward minus infinity
        assert (gen.peek(cpu, b, "scratch.x") == want).all()
        trace = {}
        u = ref.step(tok, trace)
        assert (trace["x0"][0] == want).all()
        cpu.run(stop=0)
        assert (gen.peek(cpu, b, "scratch.u") == u).all()
    assert not cpu.overflow


def test_the_rounding_edges():
    """E = -1 with sh_e = 1 gives 0 (half up), -3 gives -1, 127 gives 64; with sh_e = 15 everything gives 0."""
    for sh_e, want in ((1, {-1: 0, -3: -1, -2: -1, 1: 1, 127: 64, -127: -63, 0: 0}), (7, {-64: 0, -65: -1, 63: 0, 64: 1, 127: 1, -127: -1}),
                       (15, {-127: 0, 127: 0})):
        spec, arrays = model(3, sh_e, **SMALL)
        arrays["E"][7] = np.resize(list(want), 16)
        b = gen.build(spec, arrays)
        cpu = gen.machine(b)
        gen.set_token(cpu, b, 7)
        cpu.run(stop=next(a for a, (name, args, tag) in enumerate(b.code) if tag == "norm"))
        assert gen.peek(cpu, b, "scratch.x").tolist() == np.resize(list(want.values()), 16).tolist()


def test_shift_0_or_absent_is_the_program_of_before_and_a_shift_of_16_is_refused():
    absent, zero, one = (gen.build(*model(1, sh_e, **SMALL)) for sh_e in (None, 0, 1))
    assert absent.words == zero.words and (absent.image == zero.image).all() and absent.mem == zero.mem
    assert not any(name == "LUT" and tag == "embedding" for name, args, tag in zero.code)
    assert len(one.words) == len(zero.words) + 1 and one.mem["tables"][1] == zero.mem["tables"][1] + 256   # one LUT, one table
    for bad in (16, -1):
        with pytest.raises(ValueError):
            gen.build(*model(1, bad, **SMALL))


@pytest.mark.parametrize("sh_e", [1, 3, 7])
@pytest.mark.parametrize("options", [{}, dict(tf=TF, syntax_filter=True), dict(tf=TF, syntax_filter=True, temperature=1.0)],
                         ids=["plain", "filter", "filter+sampling"])
def test_the_complete_program_with_an_embedding_shift_and_a_window(sh_e, options):
    spec, arrays = model(sh_e, sh_e, **dict(WITH_TOK, window=7))
    for seed, ids in enumerate(PROMPTS[:2]):
        sent, cpu, b = run(spec, arrays, ids, sw=seed, **options)
        assert sent == reference(spec, arrays, ids, stop="tf" in options, syntax_filter=options.get("syntax_filter", False),
                                 temperature=options.get("temperature", 0.0), seed=seed)


def test_the_shift_costs_4_cycles_per_element_of_the_row():
    options = {"norm": "rms", "ffn": "swiglu"}
    for dim, shape in ((64, dict()), (160, dict(dim=160, n_layers=2, n_heads=4, n_kv_heads=1, hidden_dim=219))):
        cfg = ModelConfig(**shape)                               # stories260K, and the Tier A shape of D-shape-2
        for pos in (0, 63):
            total, parts = cost.cycles_per_token(cfg, options, pos)
            shifted, parts_s = cost.cycles_per_token(cfg, dict(options, sh_e=3), pos)
            assert shifted - total == parts_s["embedding"] - parts["embedding"] == 4 * dim
            assert cost.cycles_per_token(cfg, dict(options, sh_e=0), pos)[0] == total
        for kw in (dict(), dict(prefill=True), dict(syntax_filter=True, temperature=1.0)):
            assert cost.token_cycles(cfg, dict(options, sh_e=9), 20, **kw)[0] == cost.token_cycles(cfg, options, 20, **kw)[0] + 4 * dim
    spec, arrays = cost.dummy_model(ModelConfig(), dict(options, sh_e=5))
    assert spec["shifts"]["sh_e"] == 5 and spec["options"] == options
    assert "sh_e" not in cost.dummy_model(ModelConfig(), options)[0]["shifts"]
