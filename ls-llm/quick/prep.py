"""Fast path: marker text -> SentencePiece -> uint16 token streams."""
import json
import os
import sys

import numpy as np
import sentencepiece as spm

from quick.markers import MARKERS, from_markers, to_markers


def load(paths, keep=None):
    docs = []
    for path in paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if keep is None or keep(rec):
                    docs.append(rec["text"])
    return docs


def marked(docs):
    out = []
    for text in docs:
        if any(m in text for m in MARKERS):
            continue
        try:
            m = to_markers(text)
        except Exception:
            continue
        if from_markers(m) == text:
            out.append(m)
    return out


def main(out, vocab, train_paths, val_paths):
    os.makedirs(out, exist_ok=True)
    train, val = marked(load(train_paths)), marked(load(val_paths))
    with open(out + "/train.txt", "w") as f:
        f.write("\n".join(train) + "\n")
    spm.SentencePieceTrainer.train(
        input=out + "/train.txt", model_prefix=out + "/tok", vocab_size=vocab, model_type="bpe",
        user_defined_symbols=MARKERS, normalization_rule_name="identity", remove_extra_whitespaces=False,
        add_dummy_prefix=False, allow_whitespace_only_pieces=True, character_coverage=1.0,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, max_sentence_length=8192, num_threads=4,
        input_sentence_size=400000, shuffle_input_sentence=True, minloglevel=2)
    sp = spm.SentencePieceProcessor(model_file=out + "/tok.model")
    meta = {"vocab_size": vocab}
    for name, docs in (("train", train), ("val", val)):
        ids, kept, nbytes = [], 0, 0
        for g in range(0, len(docs), 2000):
            chunk = docs[g:g + 2000]
            for doc, enc in zip(chunk, sp.encode(chunk)):
                if sp.decode(enc) != doc or 0 in enc:
                    continue
                ids += [1] + enc + [2]
                kept += 1
                nbytes += len(from_markers(doc))
        np.array(ids, dtype=np.uint16).tofile(out + "/%s.bin" % name)
        meta[name + "_tokens"], meta[name + "_docs"], meta[name + "_bytes"] = len(ids), kept, nbytes
        meta[name + "_dropped"] = len(docs) - kept
    meta["bytes_per_token"] = round(meta["train_bytes"] / meta["train_tokens"], 3)
    json.dump(meta, open(out + "/meta.json", "w"), indent=1)
    os.remove(out + "/train.txt")
    print(json.dumps(meta))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4].split(","))
