"""QatModel.forward and IntModel.step give the same logits at every position, bit for bit; the gradient rules."""
import numpy as np
import pytest
import torch

from lsllm.config import ModelConfig
from lsllm.intref.model import bitlen
from lsllm.model import build
from lsllm.qat.calibrate import calibrate
from lsllm.qat.export import export
from lsllm.qat.model import Lookup, rq, rqr

torch.set_num_threads(2)
SHAPES = {
    "dyt_swiglu": ModelConfig(vocab_size=40, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=24, max_seq_len=12,
                              norm="dyt"),
    "rms_relu": ModelConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=40, max_seq_len=20,
                            ffn="relu"),
    "rms_swiglu": ModelConfig(vocab_size=50, dim=24, n_layers=2, n_heads=2, n_kv_heads=2, hidden_dim=36, max_seq_len=16),
    "dyt_relu": ModelConfig(vocab_size=33, dim=32, n_layers=3, n_heads=8, n_kv_heads=2, hidden_dim=20, max_seq_len=9,
                            norm="dyt", ffn="relu"),
    "rms_swiglu_4_2": ModelConfig(vocab_size=40, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=24,
                                  max_seq_len=14),
}


def calibrated(cfg, seed=0, rows=3):
    """A QatModel calibrated from a random float model (weights enlarged so that the logits differ), and tokens."""
    torch.manual_seed(seed)
    float_model = build(cfg)
    with torch.no_grad():
        for p in float_model.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    tokens = torch.randint(0, cfg.vocab_size, (rows, cfg.max_seq_len))
    return calibrate(float_model, [tokens]), tokens


def scramble(model, seed):
    """Random int8 weights of every size, random offsets and changed shifts: a model no training would produce,
    in which clamps, floors and zero shifts all occur."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for key, w in model.latent.items():
            size = rng.choice([127, 40, 6])
            w.copy_(torch.from_numpy(rng.integers(-size, size + 1, w.shape) / model.scale(key)))
        model.score_offset.copy_(torch.from_numpy(rng.uniform(4, 15, model.score_offset.shape)))
    model.logit_offset = rng.uniform(-3, 8)
    for name, value in model.shifts.items():
        if name not in ("sa", "sb"):                    # these two belong to the rms table
            move = rng.integers(-3, 3, np.shape(value))
            model.shifts[name] = np.clip(np.add(value, move), 0, 15).tolist()
    model.shifts["sh_l"] = int(rng.integers(7, 12))
    model.shifts["sh_o"][0] = 0                         # a shift of 0 has no rounding preload
    return model


def both(model, tokens, tmp_path):
    """Integer logits from the QAT forward pass and from the integer reference reading the exported files."""
    with torch.no_grad():
        u = model(tokens)[0].numpy()
    return u, export(model, str(tmp_path)).forward(tokens.numpy())


@pytest.mark.parametrize("shape", SHAPES)
def test_calibrated_model_is_bit_exact(shape, tmp_path):
    model, tokens = calibrated(SHAPES[shape])
    u, u_int = both(model, tokens, tmp_path)
    assert u.shape == (*tokens.shape, SHAPES[shape].vocab_size)
    assert np.array_equal(u, u_int)
    assert u.min() >= 0 and u.max() <= 127 and len(np.unique(u)) > 10       # 7-bit logits that are not all equal


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_model_is_bit_exact(shape, seed, tmp_path):
    model, tokens = calibrated(SHAPES[shape], seed)
    u, u_int = both(scramble(model, seed), tokens, tmp_path)
    assert np.array_equal(u, u_int)


def test_export_writes_part_e(tmp_path):
    cfg = SHAPES["rms_swiglu"]
    model, tokens = calibrated(cfg)
    model.tokenizer = "data/tok/some"
    reference = export(model, str(tmp_path))
    spec, arrays = reference.spec, np.load(tmp_path / "model.npz")
    assert set(spec) >= {"config", "options", "frac", "shifts", "preloads", "tokenizer"}
    assert spec["config"] == cfg.to_dict() and spec["tokenizer"] == "data/tok/some"
    assert spec["options"] == {"norm": "rms", "ffn": "swiglu"}
    L, D, d = cfg.n_layers, cfg.dim, cfg.head_dim
    shapes = {"E": (cfg.vocab_size, D), "Wq_1": (D, D), "Wk_1": (cfg.kv_dim, D), "Wv_0": (cfg.kv_dim, D),
              "Wo_0": (D, D), "W1_0": (cfg.hidden_dim, D), "W3_1": (cfg.hidden_dim, D), "W2_1": (D, cfg.hidden_dim),
              "NORM": (2 * L + 1, 256), "SILU": (256,), "EXP": (L, 256), "RECIP": (256,),
              "ROPE_C": (cfg.max_seq_len, d // 2), "ROPE_S": (cfg.max_seq_len, d // 2)}
    for name, shape in shapes.items():
        assert arrays[name].shape == shape and arrays[name].dtype == np.int8, name
    assert len(arrays.files) == 6 + 1 + 7 * L
    for name in ("sh_q", "sh_k", "sh_v", "sh_s", "sh_o", "sh_1", "sh_3", "sh_g"):
        assert len(spec["shifts"][name]) == L and all(0 <= s <= 23 for s in spec["shifts"][name])
    assert len(spec["shifts"]["sa"]) == len(spec["shifts"]["sb"]) == 2 * L + 1
    assert np.shape(spec["preloads"]["c_s"]) == (L, cfg.n_heads) and isinstance(spec["preloads"]["c_l"], int)
    # the residual matrices obey SPEC B.3: f_W + f_h = f_x + 8
    frac = spec["frac"]
    for l in range(L):
        assert frac["Wo"][l] + frac["o"][l] == frac["x"] + 8 and frac["W2"][l] + frac["g"][l] == frac["x"] + 8
    # tables: EXP of a layer is indexed by the unsigned byte of (code - row maximum) = 0, -1, .., -127, at
    # 2^f_s = 4 codes per nat; RECIP by the mantissa 64..127
    assert frac["s"] == [2] * L and frac["s"][0] == frac["q"][0] + frac["k"][0] - spec["shifts"]["sh_s"][0]
    exp = arrays["EXP"][0].astype(int)
    assert exp[0] == 127 and exp[256 - 4] == 47 and exp[256 - 22] == 1 and exp[256 - 23] == 0     # 127 / e = 46.7
    assert (exp[1:129] == 0).all() and np.array_equal(arrays["EXP"][0], arrays["EXP"][1])
    assert arrays["RECIP"][64] == 127 and arrays["RECIP"][65] == 126 and arrays["RECIP"][127] == 65   # 8192 / mantissa


def test_export_refuses_a_model_that_could_overflow(tmp_path):
    """600 hidden units, every weight 127: the worst case of a residual stream is 117% of the accumulator."""
    cfg = ModelConfig(vocab_size=20, dim=8, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=600, max_seq_len=6,
                      ffn="relu", norm="dyt")
    model, tokens = calibrated(cfg)
    export(model, str(tmp_path))                        # fine as calibrated
    with torch.no_grad():
        for key, w in model.latent.items():
            w.fill_(127 / model.scale(key))
    with pytest.raises(ValueError, match="l0.xf: worst-case accumulator 9786496 is outside 24 bits"):
        export(model, str(tmp_path))


def test_requant_gradient_passes_floor_and_clamp_but_not_relu():
    acc = torch.tensor([-70000.0, -300.0, 5.0, 300.0, 70000.0], requires_grad=True)
    out = rq(acc, 4)
    assert out.tolist() == [-128, -19, 0, 18, 127]                          # floor(-300 / 16) = -19
    out.sum().backward()
    assert acc.grad.tolist() == [1 / 16] * 5                                # also where the clamp is active
    acc.grad = None
    out = rqr(acc, 4, relu=True)
    assert out.tolist() == [0, 0, 0, 19, 127]                               # (300 + 8) >> 4 = 19
    out.sum().backward()
    assert acc.grad.tolist() == [0, 0, 1 / 16, 1 / 16, 1 / 16]


def test_lookup_uses_the_table_forward_and_the_slope_backward():
    table = torch.arange(256.0) * 2                                         # entry of index v
    slope = torch.arange(256.0) / 4
    value = torch.tensor([0.0, 5.0, 127.0, -128.0, -1.0], requires_grad=True)
    out = Lookup.apply(value, table, slope)
    assert out.tolist() == [0, 10, 254, 256, 510]                           # -128 is index 128, -1 is index 255
    (out * torch.tensor([1.0, 1, 1, 1, 2])).sum().backward()
    assert value.grad.tolist() == [0, 1.25, 31.75, 32, 127.5]


@pytest.mark.parametrize("shape", SHAPES)
def test_every_parameter_gets_a_gradient(shape):
    model, tokens = calibrated(SHAPES[shape])
    loss = model(tokens[:, :-1], tokens[:, 1:])[1]
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_model_stays_bit_exact_while_it_trains(tmp_path):
    """Optimiser steps change the model (here on one batch); it stays exportable and equal to the reference."""
    model, tokens = calibrated(SHAPES["dyt_swiglu"], rows=8)
    x, y = tokens[:, :-1], tokens[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    before = model.integers()[0]
    first = model(x, y)[1].item()
    for step in range(20):
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.clip()
    assert model(x, y)[1].item() < first
    assert any((before[name] != array).any() for name, array in model.integers()[0].items())
    u, u_int = both(model, tokens, tmp_path)
    assert np.array_equal(u, u_int)
    for key, w in model.latent.items():                                     # clip() keeps the latents inside int8
        assert (w.abs() * model.scale(key)).max() <= 127


def test_division_is_exact_for_every_size_of_the_sum(tmp_path):
    """SPEC C.6 step 5 for sums of weights from 127 to 127 * 256, far more than the short contexts of the models
    above reach: around every power of two, where the number of bits changes, and 2000 random sums. Also the
    comparison form of NRM against int.bit_length for every 15-bit number."""
    every = np.arange(2 ** 15)
    assert bitlen(every[:, None])[:, 0].tolist() == [int(n).bit_length() for n in every]
    model, tokens = calibrated(SHAPES["rms_swiglu_4_2"])
    reference = export(model, str(tmp_path))
    rng = np.random.default_rng(0)
    edges = [2 ** k + step for k in range(7, 15) for step in (-1, 0, 1)]
    sums = np.array([127, 127 * 256] + edges + rng.integers(127, 127 * 256, 2000).tolist())
    e = np.zeros((len(sums), 256), dtype=np.int64)
    for row, total in enumerate(sums):                  # weights 127, 127, .., rest, 0, .. that add up to the sum
        e[row, :total // 127] = 127
        if total % 127:
            e[row, total // 127] = total % 127
    assert np.array_equal(e.sum(1), sums)
    A = e @ rng.integers(-128, 128, (256, 5))           # five value components per row of weights
    got = model.divide(torch.tensor(e, dtype=torch.float32), torch.tensor(A, dtype=torch.float32), 6)
    want = reference._divide("test.", e, A, 6)          # shift 6: the output is the weighted mean of the values
    assert np.array_equal(got.numpy(), want) and len(np.unique(want)) > 50
    assert np.abs(want - A / sums[:, None]).max() < 3.0     # a is floored to 6 or 7 bits and RECIP is rounded



def test_attention_window_is_the_same_in_qat_and_in_the_integer_reference(tmp_path):
    """A model with cfg.window attends to the last `window` keys only, identically in both implementations."""
    import numpy as np
    import torch

    from lsllm.config import ModelConfig
    from lsllm.model.llama import build
    from lsllm.qat.calibrate import calibrate
    from lsllm.qat.export import export

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, hidden_dim=24, max_seq_len=24, window=5)
    tokens = torch.randint(0, 64, (3, 24))
    model = calibrate(build(cfg).eval(), [tokens])
    reference = export(model, str(tmp_path / "m"))
    with torch.no_grad():
        u = model(tokens)[0].numpy()
    assert np.array_equal(u, reference.forward(tokens.numpy()))
    # the window matters: the same integers without a window give other logits from position 5 on, not before
    from lsllm.intref import IntModel
    spec = dict(reference.spec, config=dict(reference.spec["config"], window=0))
    with np.load(str(tmp_path / "m" / "model.npz")) as arrays:
        full = IntModel(spec, {name: arrays[name] for name in arrays.files}).forward(tokens.numpy())
    assert np.array_equal(full[:, :5], u[:, :5]) and not np.array_equal(full[:, 5:], u[:, 5:])
