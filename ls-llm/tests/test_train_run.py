"""The training run: files it writes, learning on a toy stream, continuing, distillation, the schedule."""
import json
import math

import numpy as np
import pytest
import torch

from lsllm.model import load
from lsllm.train import run

TINY = ["--dim", "16", "--n-layers", "1", "--n-heads", "2", "--n-kv-heads", "1", "--hidden-dim", "32",
        "--max-seq-len", "16", "--batch-size", "16", "--eval-every", "20", "--lr", "0.01", "--device", "cpu",
        "--threads", "1"]
V = 20


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """A toy token stream: counting upwards modulo V, which a tiny model learns within 100 steps."""
    folder = tmp_path_factory.mktemp("toy")
    for name, n in (("train", 5000), ("val", 800)):
        (np.arange(n) % V).astype(np.uint16).tofile(folder / f"{name}.bin")
    meta = {"vocab_size": V, "train_tokens": 5000, "val_tokens": 800, "val_bytes": 2000}
    (folder / "meta.json").write_text(json.dumps(meta))
    return str(folder)


@pytest.fixture(scope="module")
def first_run(data, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("runs") / "first")
    return out, run.main(["--out", out, "--data", data, "--steps", "100"] + TINY)


def test_run_learns_and_writes_the_three_files(data, first_run):
    out, result = first_run
    assert result["val_loss"] < 0.3                                 # a uniform guess costs ln 20 = 3.0
    assert result["val_bpb"] == pytest.approx(result["val_loss"] * 800 / (2000 * math.log(2)))
    assert result["steps"] == 100 and result["tokens_seen"] == 100 * 16 * 16 and result["device"] == "cpu"
    with open(out + "/result.json") as f:
        assert json.load(f)["val_loss"] == result["val_loss"]
    lines = open(out + "/log.txt").read().splitlines()
    assert [line.split()[1] for line in lines if line.startswith("step")] == ["20", "40", "60", "80", "100"]

    model, ckpt = load(out + "/ckpt.pt")
    assert set(ckpt) == {"config", "model", "step", "val_loss", "tok"}
    assert ckpt["tok"] == data and ckpt["val_loss"] == result["val_loss"] and ckpt["step"] == result["best_step"]
    assert ckpt["config"]["vocab_size"] == V and ckpt["config"]["dim"] == 16     # vocabulary taken from meta.json
    # the stored validation loss can be recomputed from the checkpoint: evenly spaced windows of val.bin
    val = np.fromfile(data + "/val.bin", dtype=np.uint16)
    rows = run.windows(val, np.linspace(0, len(val) - 17, 800 // 17).astype(np.int64), 16)
    assert run.evaluate(model, rows, 16) == pytest.approx(result["val_loss"], abs=1e-5)


def test_same_seed_gives_the_same_result(data, first_run, tmp_path):
    again = run.main(["--out", str(tmp_path / "again"), "--data", data, "--steps", "100"] + TINY)
    assert again["val_loss"] == pytest.approx(first_run[1]["val_loss"], abs=1e-6)
    other = run.main(["--out", str(tmp_path / "other"), "--data", data, "--steps", "100", "--seed", "1"] + TINY)
    assert other["val_loss"] != first_run[1]["val_loss"]


def test_init_from_continues_with_the_shape_of_the_checkpoint(data, first_run, tmp_path):
    out = str(tmp_path / "more")
    result = run.main(["--out", out, "--data", data, "--steps", "20", "--init-from", first_run[0] + "/ckpt.pt",
                       "--batch-size", "16", "--eval-every", "20", "--device", "cpu", "--lr", "1e-3"])
    ckpt = torch.load(out + "/ckpt.pt")
    assert ckpt["config"]["dim"] == 16 and ckpt["step"] == first_run[1]["best_step"] + 20
    assert result["val_loss"] < first_run[1]["val_loss"] + 0.05      # it starts from the trained weights


def test_distillation_from_a_teacher(data, first_run, tmp_path):
    result = run.main(["--out", str(tmp_path / "kd"), "--data", data, "--steps", "60", "--teacher",
                       first_run[0] + "/ckpt.pt", "--kd-weight", "0.5", "--kd-temp", "2"] + TINY)
    assert result["val_loss"] < 1.0


def test_shape_options_and_dropout_train(data, tmp_path):
    result = run.main(["--out", str(tmp_path / "opts"), "--data", data, "--steps", "60", "--no-tied-embeddings",
                       "--ffn", "relu", "--norm", "dyt", "--attn", "sigmoid", "--window", "4", "--dropout", "0.1"] + TINY)
    assert result["config"]["tied_embeddings"] is False and result["config"]["window"] == 4
    assert result["val_loss"] < 1.0


def test_data_vocabulary_must_fit(data, tmp_path):
    with pytest.raises(AssertionError):
        run.main(["--out", str(tmp_path / "bad"), "--data", data, "--steps", "1", "--vocab-size", "10"] + TINY)


def test_learning_rate_schedule():
    rates = [run.lr_at(step, 100, 1.0, 10) for step in range(100)]
    assert rates[0] == pytest.approx(0.1) and rates[9] == pytest.approx(1.0) == max(rates)
    assert all(a > b for a, b in zip(rates[10:], rates[11:]))       # falls after the warm-up
    assert 0 < rates[-1] < 0.01
