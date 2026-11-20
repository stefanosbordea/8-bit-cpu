"""tokenize_corpus on a small sample: files, counts, document layout, subsampling, dropped documents."""
import json
from pathlib import Path

import numpy as np

from lsllm.data.tokenize_corpus import build, sample
from lsllm.structure import to_markers
from lsllm.tok import Tok

BOOT = Path(__file__).resolve().parents[1] / "data/tok/bootstrap"


def write_sample(path, source, count, extra=()):
    lines = open(source).readlines()[:count] + [json.dumps({"text": text}) + "\n" for text in extra]
    path.write_text("".join(lines))
    return [json.loads(line)["text"] for line in lines]


def representable(texts):
    out = []
    for text in texts:
        try:
            to_markers(text)
            out.append(text)
        except ValueError:
            pass
    return out


def test_build(tmp_path):
    deep = "".join("    " * depth + "if a:\n" for depth in range(7)) + "    " * 7 + "b = 1\nc = 2"
    train = write_sample(tmp_path / "train.jsonl", BOOT / "train.jsonl", 600)
    val = write_sample(tmp_path / "val.jsonl", BOOT / "val.jsonl", 40, extra=[deep, "x = '<NL>'", "x = 'caf\u00e9'"])
    kept = {"train": representable(train), "val": representable(val)[:-1]}     # markers accept the last, Tok does not
    out = tmp_path / "tok256"
    meta = build([tmp_path / "train.jsonl"], [tmp_path / "val.jsonl"], 256, out)
    assert meta == json.loads((out / "meta.json").read_text())
    assert meta["vocab_size"] == 256
    assert meta["val_dropped"] >= 3                                 # the dedent of 7 levels, the marker, the accent
    for split, texts in (("train", train), ("val", val)):
        assert meta[split + "_docs"] == len(kept[split])
        assert meta[split + "_dropped"] == len(texts) - len(kept[split])
        assert meta[split + "_bytes"] == sum(len(text) for text in kept[split])
    assert meta["bytes_per_token"] == round(meta["val_bytes"] / meta["val_tokens"], 4)

    tok = Tok(out / "tok.model")
    assert tok.vocab_size == 256
    for split, texts in kept.items():
        ids = np.fromfile(out / (split + ".bin"), dtype=np.uint16)
        assert len(ids) == meta[split + "_tokens"] and ids.max() < 256
        assert (ids == tok.bos_id).sum() == (ids == tok.eos_id).sum() == len(texts)
        starts, ends = np.flatnonzero(ids == tok.bos_id), np.flatnonzero(ids == tok.eos_id)
        for doc in (0, len(texts) // 2, len(texts) - 1):            # each document is BOS, tokens, EOS
            assert tok.decode(ids[starts[doc]:ends[doc] + 1]) == texts[doc]


def test_max_per_input(tmp_path):
    write_sample(tmp_path / "a.jsonl", BOOT / "train.jsonl", 500)
    write_sample(tmp_path / "b.jsonl", BOOT / "val.jsonl", 300)
    meta = build([tmp_path / "a.jsonl", tmp_path / "b.jsonl"], [tmp_path / "b.jsonl"], 256, tmp_path / "out",
                 max_per_input=200)
    assert (meta["train_docs"], meta["val_docs"]) == (400, 200)


def test_sample():
    assert sample(iter(range(5)), 10) == [0, 1, 2, 3, 4]
    picked = sample(iter(range(1000)), 10)
    assert len(set(picked)) == 10 and picked == sample(iter(range(1000)), 10)
    assert max(picked) > 100                        # not simply the first items
