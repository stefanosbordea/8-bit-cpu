"""Compare tokenizer settings on the bootstrap set. Reproduces the numbers of the tokenizer decisions in DECISIONS.md.

    .venv/bin/python data/tok/compare.py

Variants (lsllm.data.tokenize_corpus.VARIANTS): A = free merges; B = the structural characters
( ) [ ] { } ' " : , # \\ are always single tokens; B+ = B, and four frequent strings made of structural
characters are single tokens too (three double quotes, comma + space, "()" and "):").
For vocabulary 256 and 512 and every variant, SentencePiece BPE is trained on bootstrap/train.jsonl with every
combination of three splitting settings; the unigram model is trained with the chosen combination only.
Everything is measured on bootstrap/val.jsonl. Token counts include BOS and EOS. Results: data/tok/compare.json.
"""
import itertools
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lsllm.data.tokenize_corpus import VARIANTS, marker_texts, read_texts, train_tokenizer  # noqa: E402
from lsllm.tok import Tok  # noqa: E402

FLAGS = ("mixed_pieces", "space_as_suffix", "pieces_across_spaces")
CHOSEN = (True, True, False)       # the settings that lsllm.data.tokenize_corpus uses


def measure(tok, texts, marked):
    """Sequence length statistics of the tokenizer on a list of canonical texts and their marker texts."""
    lengths, counts = [], Counter()
    for ids in tok.sp.encode(marked):           # raw SentencePiece ids, so that unknown tokens would show up
        lengths.append(len(ids) + 2)            # with BOS and EOS
        counts.update(ids)
    total = sum(lengths)
    return {"tokens_mean": round(total / len(lengths), 1), "tokens_median": statistics.median(lengths),
            "bytes_per_token": round(sum(map(len, texts)) / total, 3), "unknown_share": counts[tok.unk_id] / total,
            "within_256": round(sum(n <= 256 for n in lengths) / len(lengths), 3),
            "top40": [[tok.pieces[t], n] for t, n in counts.most_common(40)]}


def run(train, val, val_marked, vocab, variant, model_type, mixed, suffix, across, folder):
    """Train one tokenizer and return its row of results."""
    start = time.time()
    train_tokenizer(train, vocab, folder / "tok.model", variant, model_type=model_type,
                    split_by_unicode_script=not mixed, split_by_number=not mixed,
                    treat_whitespace_as_suffix=suffix, split_by_whitespace=not across)
    tok = Tok(folder / "tok.model")
    assert tok.vocab_size == vocab
    row = {"vocab": vocab, "variant": variant, "model_type": model_type, "mixed_pieces": mixed,
           "space_as_suffix": suffix, "pieces_across_spaces": across}
    row.update(measure(tok, val, val_marked))
    row["train_seconds"] = round(time.time() - start, 1)
    print("%(vocab)d %(variant)-2s %(model_type)-7s mixed=%(mixed_pieces)d suffix=%(space_as_suffix)d "
          "across=%(pieces_across_spaces)d  tokens/function mean %(tokens_mean).1f median %(tokens_median)g  "
          "bytes/token %(bytes_per_token).3f  within 256: %(within_256).3f  unknown %(unknown_share)g" % row,
          flush=True)
    return row


def main():
    boot = ROOT / "data/tok/bootstrap"
    train = list(read_texts([boot / "train.jsonl"]))
    val = list(read_texts([boot / "val.jsonl"]))
    val_marked = list(marker_texts(val))
    assert len(val) == len(val_marked)          # every validation function can be written with markers
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        for vocab, variant in itertools.product((256, 512), VARIANTS):
            for flags in itertools.product((False, True), repeat=3):
                rows.append(run(train, val, val_marked, vocab, variant, "bpe", *flags, Path(folder)))
            rows.append(run(train, val, val_marked, vocab, variant, "unigram", *CHOSEN, Path(folder)))
    (ROOT / "data/tok/compare.json").write_text(json.dumps(rows, indent=1) + "\n")
    print("\nchosen settings (mixed pieces, space as suffix, no pieces across spaces); validation functions: %d, "
          "mean %.0f bytes" % (len(val), statistics.mean(map(len, val))))
    chosen = {(r["vocab"], r["variant"], r["model_type"]): r for r in rows
              if tuple(r[flag] for flag in FLAGS) == CHOSEN}
    for (vocab, variant, model_type), row in chosen.items():
        free = chosen[vocab, "A", model_type]["tokens_mean"]
        print("%d %-2s %-7s mean %.1f (%+.1f%% against A) median %g bytes/token %.3f" % (
            vocab, variant, model_type, row["tokens_mean"], 100 * (row["tokens_mean"] / free - 1),
            row["tokens_median"], row["bytes_per_token"]))
        print("    top 40:", " ".join(repr(piece) for piece, _ in row["top40"]))


if __name__ == "__main__":
    main()
