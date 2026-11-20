"""The float model: equality with upstream llama2.c, parameter counts, the shape options, causality."""
import importlib.util
import math
import os
from dataclasses import replace

import pytest
import torch

from lsllm.config import ModelConfig
from lsllm.model import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL = ModelConfig(vocab_size=50, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=40, max_seq_len=24)
OPTIONS = {
    "default": {},
    "untied": {"tied_embeddings": False},
    "relu": {"ffn": "relu"},
    "dyt": {"norm": "dyt"},
    "sigmoid": {"attn": "sigmoid"},
    "window": {"window": 5},
    "all": {"tied_embeddings": False, "ffn": "relu", "norm": "dyt", "attn": "sigmoid", "window": 5},
}


def upstream_module():
    """third_party/llama2c/model.py, loaded by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("llama2c_model", os.path.join(ROOT, "third_party/llama2c/model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def randomize(model, seed):
    """Large random weights, so that attention is sharp and every path matters."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3 if p.dim() == 2 else torch.rand_like(p) + 0.5)


def count(model):
    return sum(p.numel() for p in model.parameters())


def test_default_is_stories260k():
    cfg = ModelConfig()
    assert count(build(cfg)) == 260032 == cfg.n_params()


@pytest.mark.parametrize("big_weights", [False, True])
def test_same_logits_as_upstream(big_weights):
    up = upstream_module()
    torch.manual_seed(0)
    ref = up.Transformer(up.ModelArgs(dim=64, n_layers=5, n_heads=8, n_kv_heads=4, vocab_size=512, multiple_of=4,
                                      max_seq_len=256, dropout=0.0)).eval()
    assert count(ref) == 260032
    if big_weights:
        randomize(ref, 1)
    mine = build(ModelConfig()).eval()
    mine.load_state_dict(ref.state_dict())          # same names, same shapes
    tokens, targets = torch.randint(0, 512, (2, 256)), torch.randint(0, 512, (2, 256))
    with torch.no_grad():
        want = ref(tokens, targets)                  # upstream returns all logits when targets are given
        got, loss = mine(tokens, targets)
    assert got.shape == want.shape == (2, 256, 512)
    # float32 rounding only: measured 4e-7 at initialisation scale, 1e-4 with logits of size 12
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-4 if big_weights else 1e-5)
    assert abs(loss.item() - ref.last_loss.item()) < 1e-4


@pytest.mark.parametrize("extra", [{}, {"window": 5}])
def test_flash_equals_written_out_attention(extra):
    cfg = replace(SMALL, **extra)
    plain, flash = build(cfg).double().eval(), build(cfg, flash=True).double().eval()
    randomize(plain, 2)
    flash.load_state_dict(plain.state_dict())
    tokens = torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len))
    with torch.no_grad():
        assert torch.allclose(plain(tokens)[0], flash(tokens)[0], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", OPTIONS)
def test_option_shapes_loss_and_gradients(name):
    cfg = replace(SMALL, **OPTIONS[name])
    model = build(cfg)
    # ModelConfig.n_params does not count the few scalars of DyT (alpha per norm) and sigmoid (b)
    scalars = (2 * cfg.n_layers + 1) * (cfg.norm == "dyt") + cfg.n_layers * (cfg.attn == "sigmoid")
    assert count(model) == cfg.n_params() + scalars
    tokens = torch.randint(0, cfg.vocab_size, (3, cfg.max_seq_len))
    logits, loss = model(tokens[:, :-1], tokens[:, 1:])
    assert logits.shape == (3, cfg.max_seq_len - 1, cfg.vocab_size)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5     # untrained: close to a uniform guess
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model(tokens)[1] is None                               # no targets, no loss


@pytest.mark.parametrize("name", OPTIONS)
def test_future_tokens_never_change_earlier_logits(name):
    cfg = replace(SMALL, **OPTIONS[name])
    model = build(cfg).eval()
    randomize(model, 3)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    changed = tokens.clone()
    changed[:, 10] = (changed[:, 10] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = model(tokens)[0], model(changed)[0]
    assert torch.equal(a[:, :10], b[:, :10])
    assert not torch.allclose(a[:, 10], b[:, 10])


@pytest.mark.parametrize("n_layers", [1, 2])
def test_window_limits_how_far_back_a_position_sees(n_layers):
    W = 5
    cfg = replace(SMALL, n_layers=n_layers, window=W)
    model = build(cfg).eval()
    randomize(model, 4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    changed = tokens.clone()
    changed[:, 3] = (changed[:, 3] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = model(tokens)[0], model(changed)[0]
    reach = 3 + n_layers * (W - 1)          # every layer passes information W - 1 positions forward
    assert not torch.allclose(a[:, reach], b[:, reach])
    assert torch.equal(a[:, reach + 1:], b[:, reach + 1:])


def test_sigmoid_bias_starts_at_minus_log_context():
    model = build(replace(SMALL, attn="sigmoid"))
    assert all(abs(layer.attention.b.item() + math.log(SMALL.max_seq_len)) < 1e-6 for layer in model.layers)


def test_dropout_only_acts_in_training_mode():
    model = build(SMALL, dropout=0.5)
    tokens = torch.randint(0, SMALL.vocab_size, (2, SMALL.max_seq_len))
    assert not torch.equal(model(tokens)[0], model(tokens)[0])
    model.eval()
    assert torch.equal(model(tokens)[0], model(tokens)[0])


def test_bad_shapes_are_rejected():
    with pytest.raises(AssertionError):
        build(replace(SMALL, n_heads=5))
    with pytest.raises(AssertionError):
        build(replace(SMALL, norm="layernorm"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps_matches_cpu():
    model = build(SMALL).eval()
    tokens = torch.randint(0, SMALL.vocab_size, (2, SMALL.max_seq_len))
    with torch.no_grad():
        cpu = model(tokens)[0]
        mps = model.to("mps")(tokens.to("mps"))[0].cpu()
    assert torch.allclose(cpu, mps, atol=1e-5)
