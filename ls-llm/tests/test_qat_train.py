"""lsllm.qat.train end to end on a tiny model: a few steps on the CPU, the files it writes, the numbers in them."""
import json
import math

import numpy as np
import pytest
import torch

from lsllm.config import ModelConfig
from lsllm.intref import IntModel
from lsllm.model import build
from lsllm.qat import train

torch.set_num_threads(2)
V, T = 32, 16


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A toy token stream, a float checkpoint trained on it for 40 steps, and one QAT run with distillation."""
    root = tmp_path_factory.mktemp("qat")
    rng = np.random.default_rng(0)
    successors = rng.permuted(np.tile(np.arange(V), (V, 1)), axis=1)[:, :3]
    stream = np.zeros(30000, dtype=np.uint16)
    for g in range(1, len(stream)):
        stream[g] = successors[stream[g - 1], rng.choice(3, p=[0.7, 0.2, 0.1])]
    data = root / "data"
    data.mkdir()
    stream[:28000].tofile(data / "train.bin")
    stream[28000:].tofile(data / "val.bin")
    (data / "meta.json").write_text(json.dumps({"vocab_size": V, "val_tokens": 2000, "val_bytes": 5000}))

    cfg = ModelConfig(vocab_size=V, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=48, max_seq_len=T)
    torch.manual_seed(0)
    model = build(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for step in range(40):
        rows = train.windows(stream[:28000], rng.integers(0, 28000 - T, size=16), T)
        loss = model(rows[:, :-1], rows[:, 1:])[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 40, "val_loss": 0.0, "tok": str(data)},
               root / "ckpt.pt")
    out = root / "out"
    result = train.main(["--float", str(root / "ckpt.pt"), "--data", str(data), "--out", str(out), "--steps", "30",
                         "--batch-size", "16", "--eval-every", "10", "--eval-windows", "24", "--calib-windows", "8",
                         "--kd-weight", "0.5", "--lr", "0.2"])
    return result, out, data


def test_result_numbers(run):
    result, out, data = run
    assert json.loads((out / "result.json").read_text()) == json.loads(json.dumps(result))
    assert result["bit_exact"] is True
    assert result["val_loss"] <= result["calibrated_val_loss"]              # the best model is kept, step 0 included
    assert result["val_loss"] < result["calibrated_val_loss"]               # and 30 steps do help here
    assert abs(result["calibrated_val_loss"] - result["float_val_loss"]) < 0.1
    assert result["val_bpb"] == pytest.approx(result["val_loss"] * 2000 / (5000 * math.log(2)))
    assert result["steps"] == 30 and result["tokens_seen"] == 30 * 16 * T and result["device"] == "cpu"
    assert 0 < result["best_step"] <= 30


def test_written_model_gives_the_reported_loss(run):
    """The loss in result.json is the loss of the files: recomputed here with the integer reference alone."""
    result, out, data = run
    reference = IntModel.load(str(out))
    assert reference.spec["tokenizer"] == str(data)
    val = np.fromfile(data / "val.bin", dtype=np.uint16).astype(np.int64)
    starts = np.linspace(0, len(val) - T - 1, 24).astype(np.int64)
    rows = val[starts[:, None] + np.arange(T + 1)]
    logits = reference.forward(rows[:, :-1]) * 2.0 ** -reference.f_l
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    loss = -np.take_along_axis(logp, rows[:, 1:, None], -1).mean()
    assert loss == pytest.approx(result["val_loss"], abs=1e-5)


def test_log(run):
    result, out, data = run
    lines = (out / "log.txt").read_text().splitlines()
    assert lines[0].startswith("float val ") and "calibrated integer val" in lines[0]
    assert [line.split()[1] for line in lines if line.startswith("step ")] == ["10", "20", "30"]
    assert lines[-1].startswith("done: integer val ") and lines[-1].endswith("identical: True")
