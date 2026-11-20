"""Small pieces of the board-model tools: the request tokenisation, the play of an entry of E, the request rows."""
import json

import numpy as np

from lsllm.intref.polish import free_step, play
from lsllm.tok import Tok
from lsllm.train.recall import Recall

TOK = Tok("artifacts/tokenizer/tok.model")


def test_request_text_begins_with_the_tokens_of_the_request():
    for name, text in (("hello_world", "def hello_world():\n    return 'Hello, World!'"), ("add", "def add(a, b):\n    return a + b")):
        prompt = TOK.encode("def %s(" % name)
        ids = TOK.encode_request("def %s(" % name, text)
        assert ids[:len(prompt)] == prompt and ids[-1] == TOK.eos_id
        assert TOK.decode(ids[1:-1]) == text


def test_play_keeps_the_input_embedding():
    row = np.arange(-127, 128, dtype=np.int8)
    for sh_e in (1, 2, 3):
        half = (1 << sh_e) >> 1
        low, high = play(row, sh_e)
        assert (low <= 0).all() and (high >= 0).all()
        for value, a, b in zip(row.astype(int), low, high):
            same = {(value + step + half) >> sh_e for step in range(a, b + 1)}
            assert same == {(value + half) >> sh_e} and -127 <= value + a and value + b <= 127
            for step in (a - 1, b + 1):                 # one further leaves the bucket or int8
                assert not (-127 <= value + step <= 127 and (value + step + half) >> sh_e == (value + half) >> sh_e)
    assert free_step(np.array([3], np.int8), 0, 1, 3) is False and free_step(np.array([2], np.int8), 0, 1, 3) is True


def test_request_rows_count_only_what_the_model_writes(tmp_path):
    path = tmp_path / "texts.jsonl"
    texts = {"add": "def add(a, b):\n    return a + b", "hello_world": "def hello_world():\n    return 'Hello, World!'"}
    path.write_text("".join(json.dumps({"name": name, "text": text}) + "\n" for name, text in texts.items()))
    recall = Recall([str(path)], TOK, 256)
    for g, name in enumerate(recall.names):
        prompt = TOK.encode("def %s(" % name)
        length = int(recall.lengths[g])
        mask = recall.mask[g].numpy()
        assert mask[:len(prompt) - 1].sum() == 0                    # the request itself is given
        assert mask[len(prompt) - 1:length - 1].all()               # everything behind it counts, the end token included
        assert mask[length - 1:].sum() == 0                         # padding does not
        assert int(recall.rows[g, length - 1]) == TOK.eos_id
    rows, mask = recall.batch(2, np.random.default_rng(0), 3.0)
    assert rows.shape[1] == int(recall.lengths.max()) and mask.shape[1] == rows.shape[1] - 1
