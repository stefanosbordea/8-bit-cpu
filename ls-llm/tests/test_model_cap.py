import torch

from lsllm.config import ModelConfig
from lsllm.model.llama import build


def test_cap_attention_is_causal_and_trains():
    for attn in ("cap", "capnorm", "cappow2", "maxnorm", "maxpow2"):
        torch.manual_seed(0)
        model = build(ModelConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=48, max_seq_len=32, attn=attn))
        x = torch.randint(0, 64, (2, 32))
        logits, loss = model(x, x)
        loss.backward()
        assert logits.shape == (2, 32, 64) and torch.isfinite(loss)
        assert attn.startswith("max") or all(layer.attention.cap_b.grad is not None for layer in model.layers)
        y = x.clone()
        y[:, 20:] = 0
        with torch.no_grad():
            assert torch.allclose(model(x)[0][:, :20], model(y)[0][:, :20], atol=1e-5)
