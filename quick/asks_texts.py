"""The training texts of the request suite: every verified human implementation (data/asks/solutions.jsonl) and
every implementation that passed the tests but was too long with its docstring (data/asks/missing.jsonl), each
turned into the form the board model has to write after a request, by mechanical steps only:
the function is renamed to the name of the ask (identifiers only, not attributes), its docstring is removed, type
hints that need an import are removed, the text is canonicalised (lsllm.asks.plain, lsllm.asks.rename). The result must be at most 230 tokens and must pass the tests of the ask in the sandbox again.

    python -m quick.asks_texts            writes data/asks/train_texts.jsonl

Every row says whether the text is also allowed in a benchmark model (`benchmark_clean`): it is not if it has the
name or the code of a benchmark task (under its own name or the name of the ask), is a near-copy of a benchmark
solution (lsllm/data/nearcopy.py), or comes from the functions that the overlap filter took out of the corpus.
"""
import gzip
import json

from lsllm.alpha import alpha_key
from lsllm.asks import ASKS, DATA, MISSING, SOLUTIONS, load_jsonl, rename
from lsllm.asks import plain as plain_form
from lsllm.asks.run import DRIVER, MEMORY_MB, _outcome
from lsllm.canon import canon
from lsllm.data.corpus.overlap import load_benchmarks, overlap
from lsllm.data.nearcopy import NearCopies
from lsllm.sandbox import run_many
from lsllm.tok import Tok

LIMIT = 230                 # tokens with the begin token: the usual limit of a request text
LONGEST = 254               # what the context of 256 can hold (begin token, text, end token); used only where nothing shorter exists
asks = {ask["id"]: ask for ask in load_jsonl(ASKS)}
tok = Tok("data/tok/f512v2/tok.model")


def line_of(path, number):
    with (gzip.open if path.endswith(".gz") else open)(path, "rt", encoding="utf-8") as lines:
        for g, line in enumerate(lines):
            if g == number:
                return json.loads(line)


def plain(text, own_name, name, limit=LIMIT):
    """The form the board model writes (lsllm.asks.plain and rename: no docstring, no type hints that need an
    import, the name of the ask, canonical); None if nothing is left, if it has more than `limit` tokens or if it
    cannot be tokenised."""
    try:
        out = plain_form(text)[0]
        out = rename(out, own_name, name) if out else None
        return out if out and len(tok.encode(out)) <= limit else None
    except (ValueError, SyntaxError):
        return None


def main():
    candidates = []
    for row in load_jsonl(SOLUTIONS):
        candidates.append((row["id"], row["text"], row["original_name"], row["source"], row.get("licence_checked", row["licence"]), row.get("file", "")))
    for row in load_jsonl(MISSING):
        for found in row.get("passing_but_too_long", []):
            record = line_of(found["file"], found["line"])
            candidates.append((row["id"], record["text"], found["original_name"], found["source"], found["licence"], found["file"]))
    bench, near = load_benchmarks(), NearCopies()
    rows, programs = [], []
    for ask_id, text, own_name, source, licence, file in candidates:
        ask = asks[ask_id]
        final = plain(text, own_name, ask["name"], LONGEST)
        if final is None or not final.startswith("def %s(" % ask["name"]):
            continue
        removed = file.endswith("all_functions.jsonl.gz") or overlap(own_name, text, alpha_key(text), bench)
        rows.append({"id": ask_id, "name": ask["name"], "text": final, "source": source, "licence": licence, "tokens": len(tok.encode(final)),
                     "original_name": own_name, "removed_by_overlap_filter": bool(removed)})
        programs.append(final)
    mark = "REPORT-asks "
    results = run_many(["PROGRAM = %r\nTESTS = %r\nMARK = %r\n" % (program, asks[row["id"]]["tests"], mark) + DRIVER for row, program in zip(rows, programs)], 2, 5.0, MEMORY_MB)
    kept, seen = [], set()
    for row, result in zip(rows, results):
        outcome = _outcome(result, mark)[0]
        if outcome != "pass" or row["id"] in seen:
            continue
        seen.add(row["id"])
        row["benchmark_clean"] = not (row.pop("removed_by_overlap_filter") or overlap(row["name"], row["text"], alpha_key(row["text"]), bench) or near(row["text"]))
        kept.append(row)
    short = {row["id"] for row in kept if row["tokens"] <= LIMIT}
    kept.sort(key=lambda row: row["tokens"] > LIMIT)       # a text within the usual limit comes first
    long, kept = [row for row in kept if row["tokens"] > LIMIT and row["id"] not in short], [row for row in kept if row["tokens"] <= LIMIT]
    with open(DATA / "train_texts.jsonl", "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    with open(DATA / "long_texts.jsonl", "w") as f:        # 231 to 254 tokens: for a request that has no shorter text
        for row in long:
            f.write(json.dumps(row) + "\n")
    print("long texts (only used where no shorter text exists):", [(row["name"], row["tokens"]) for row in long])
    lengths = sorted(row["tokens"] for row in kept)
    print(json.dumps({"asks": len(asks), "candidates": len(candidates), "texts": len(kept), "benchmark_clean": sum(row["benchmark_clean"] for row in kept),
                      "tokens_median": lengths[len(lengths) // 2], "tokens_max": lengths[-1], "tokens_total": sum(lengths)}))
    print("not clean for a benchmark model:", sorted(row["name"] for row in kept if not row["benchmark_clean"]))
    print("asks without a text:", sorted(set(ask["name"] for ask in asks.values()) - {row["name"] for row in kept}))


if __name__ == "__main__":
    main()
