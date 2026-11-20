"""Train a SentencePiece tokenizer on marker text and write the token files of a corpus.

    .venv/bin/python -m lsllm.data.tokenize_corpus --inputs data/corpus/train.jsonl \
        --val-inputs data/corpus/val.jsonl --vocab-size 512 --out data/tok/main512

Inputs are JSON-lines files with the canonical source of one function in the field "text".
Output in --out: tok.model, tok.vocab, train.bin, val.bin and meta.json. The .bin files are numpy uint16,
documents concatenated, each document = BOS, tokens, EOS.
"""
import argparse
import json
import random
from array import array
from pathlib import Path

import sentencepiece as spm

from lsllm.structure import MARKERS, to_markers
from lsllm.tok import BLANK, Tok

PRINTABLE = "".join(chr(c) for c in range(33, 127))    # printable ASCII without the space
STRUCTURAL = list("()[]{}'\":,#\\")
VARIANTS = {                                           # strings that are always one token of their own
    "A": [],                                           # none: free merges
    "B": STRUCTURAL,                                   # the structural characters
    "B+": STRUCTURAL + ['"""', ", ", "()", "):"],      # and the four most frequent strings made of them
}
MAX_TRAIN_DOCS = 200_000                               # the tokenizer itself is trained on at most this many documents


def sample(items, k):
    """At most k items of an iterator of unknown length, chosen uniformly at random (reservoir sampling, seed 0)."""
    rng = random.Random(0)
    keep = []
    for n, item in enumerate(items):
        if n < k:
            keep.append(item)
        else:
            slot = rng.randrange(n + 1)
            if slot < k:
                keep[slot] = item
    return keep


def read_texts(paths, max_per_input=None):
    """Yield the "text" of every record; with max_per_input, a random sample of that many records per file."""
    for path in paths:
        with open(path) as f:
            for line in f if max_per_input is None else sample(f, max_per_input):
                yield json.loads(line)["text"]


def marker_texts(texts):
    """Yield the marker text of every text that markers can represent."""
    for text in texts:
        try:
            yield to_markers(text)
        except ValueError:
            pass


def train_tokenizer(texts, vocab_size, model_file, variant="B+", **options):
    """Train SentencePiece on the marker text of `texts` and write `model_file` (must end in .model).

    variant: a key of VARIANTS. options: SentencePiece settings that replace the ones below.
    """
    single = VARIANTS[variant]
    settings = dict(
        model_type="bpe",
        user_defined_symbols=MARKERS + [s.replace(" ", "\u2581") for s in single],    # never merged with neighbours
        control_symbols=[BLANK],                               # an id that only Tok.encode produces
        required_chars="".join(c for c in PRINTABLE if c not in single),
        character_coverage=1.0,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        split_by_unicode_script=False, split_by_number=False,   # a piece may mix letters, digits and punctuation
        treat_whitespace_as_suffix=True,                        # a space belongs to the piece in front of it
        max_sentence_length=1_000_000,                         # one document is one sentence, never skip one
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        minloglevel=2,
    )
    settings.update(options)
    spm.SentencePieceTrainer.train(sentence_iterator=iter(sample(marker_texts(texts), MAX_TRAIN_DOCS)),
                                   model_prefix=str(model_file)[:-len(".model")], vocab_size=vocab_size, **settings)


def encode_texts(tok, texts, bin_file):
    """Write the token stream of `texts` to bin_file; return the counts of documents, tokens, bytes, dropped."""
    ids = array("H")
    count = {"docs": 0, "tokens": 0, "bytes": 0, "dropped": 0}
    for text in texts:
        try:
            ids.extend(tok.encode(text, bos=True, eos=True))
        except ValueError:
            count["dropped"] += 1
            continue
        count["docs"] += 1
        count["bytes"] += len(text)
    count["tokens"] = len(ids)
    with open(bin_file, "wb") as f:
        ids.tofile(f)
    return count


def build(inputs, val_inputs, vocab_size, out, max_per_input=None, variant="B+"):
    """Train the tokenizer on `inputs`, encode `inputs` and `val_inputs`, write everything to the directory `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train_tokenizer(read_texts(inputs, max_per_input), vocab_size, out / "tok.model", variant)
    tok = Tok(out / "tok.model")
    assert tok.vocab_size == vocab_size
    meta = {"vocab_size": vocab_size, "variant": variant}
    for split, paths in (("train", inputs), ("val", val_inputs)):
        count = encode_texts(tok, read_texts(paths, max_per_input), out / (split + ".bin"))
        meta.update({split + "_" + key: value for key, value in count.items()})
    # tokens include one BOS and one EOS per document, exactly as stored in val.bin
    meta["bytes_per_token"] = round(meta["val_bytes"] / meta["val_tokens"], 4)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON-lines files for training")
    parser.add_argument("--val-inputs", nargs="+", required=True, help="JSON-lines files for validation")
    parser.add_argument("--vocab-size", type=int, choices=[256, 512], required=True)
    parser.add_argument("--out", required=True, help="output directory, e.g. data/tok/main512")
    parser.add_argument("--max-per-input", type=int, help="use only a random sample of this many records per file")
    parser.add_argument("--variant", choices=list(VARIANTS), default="B+", help="tokens forced to stay single")
    args = parser.parse_args()
    meta = build(args.inputs, args.val_inputs, args.vocab_size, args.out, args.max_per_input, args.variant)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
