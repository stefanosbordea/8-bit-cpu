"""Find a verified human-written implementation for every ask that has none yet.

    .venv/bin/python -m lsllm.asks.mine

Solutions already stored in data/asks/solutions.jsonl are kept as they are (a model may have been trained on
them); delete the file to choose all of them again. For every other ask, candidates are corpus functions chosen
by name: tier 0 the ask's name, tier 1 one of its `aliases`, tier 2 a name that contains every word of the ask's
name. A candidate is judged in its plain form (lsllm.asks.plain: without docstring, without type hints that
would not run alone, canonical). It is kept if it is one top-level function (corpus kind "function"), its plain
form is self-contained (lsllm.data.selfcontained), has at most MAX_TOKENS tokens and passes the ask's tests in
the sandbox when it carries the ask's name (lsllm.asks.rename: the function's identifier is replaced, as the
training step does it). The solution of an ask is the passing candidate of the lowest tier and, within the tier,
of the fewest tokens.
The stored `text` is always the unchanged corpus text; `tokens` is its length (null if its docstring is not
ASCII), `tokens_without_docstring` the length of the plain form, `hints_removed` says whether the plain form
has no type hints. Nothing here writes or edits an implementation.

Candidates whose plain form is too long are tested as well and listed in missing.jsonl if they pass.
"""
import collections
import gzip
import json
import secrets
import warnings
from pathlib import Path

from lsllm.asks import ASKS, MISSING, SOLUTIONS, load_jsonl, plain, rename
from lsllm.bench.harness import DRIVER, MEMORY_MB, _outcome
from lsllm.data.selfcontained import self_contained
from lsllm.sandbox import run_many
from lsllm.tok import Tok

ROOT = Path(__file__).resolve().parents[2]
# The first three are the training corpus. The last is the corpus before the benchmark-overlap filter of
# lsllm.data.corpus.build (same repositories, same licences); it still has the functions that are named like
# a benchmark task, such as gcd or binary_search. Rows of repositories rejected later are skipped.
FILES = ["data/corpus/train.jsonl", "data/csn/train.jsonl", "data/csn2/train.jsonl",
         "data/corpus/all_functions.jsonl.gz"]
TOKENIZER = "data/tok/f512v2/tok.model"
MAX_TOKENS = 230        # with the begin token; the model's context is 256
WORKERS = 2


def rejected_repos():
    """Repositories whose functions the corpus builder does not use."""
    rows = json.loads((ROOT / "lsllm/data/corpus/repos.json").read_text())
    fetched = json.loads((ROOT / "data/corpus/fetched.json").read_text())
    return {row["repo"] for row in rows if "rejected" in row} | {repo for repo, info in fetched.items() if "rejected" in info}


def corpus_functions():
    """Yield (file, line number, record) for every plain function of the corpus files."""
    rejected = rejected_repos()
    for name in FILES:
        with (gzip.open if name.endswith(".gz") else open)(ROOT / name, "rt", encoding="utf-8") as lines:
            for number, line in enumerate(lines):
                record = json.loads(line)
                if record["kind"] == "function" and record["source"] not in rejected:
                    yield name, number, record


def tier(ask, name):
    """0 if `name` is the ask's name, 1 for an alias, 2 if it contains all words of the ask's name, else None."""
    if name == ask["name"]:
        return 0
    if name in ask.get("aliases", []):
        return 1
    if all(word in name.lower() for word in ask["name"].split("_")):
        return 2
    return None


def program(ask, form, own_name):
    """The candidate as it is tested: its plain form under the ask's name."""
    return form if own_name == ask["name"] else rename(form, own_name, ask["name"])


def length(tok, text):
    """Number of tokens of `text` with the begin token; None if the tokenizer cannot represent it."""
    try:
        return len(tok.encode(text))
    except ValueError:
        return None


def main():
    warnings.simplefilter("ignore", SyntaxWarning)       # corpus strings with escapes like "\d"
    asks = load_jsonl(ASKS)
    tok = Tok(str(ROOT / TOKENIZER))
    kept = {row["id"]: row for row in load_jsonl(SOLUTIONS)} if SOLUTIONS.exists() else {}
    for row in kept.values():                        # rows written before these two fields existed
        form, row["hints_removed"] = plain(row["text"])
        row["tokens_without_docstring"] = length(tok, form)
    todo = [ask for ask in asks if ask["id"] not in kept]
    candidates = collections.defaultdict(list)       # ask id -> candidates that reach the sandbox
    reasons = collections.defaultdict(collections.Counter)
    seen = set()                                     # (ask id, text): the same function occurs in several files
    for file, number, record in corpus_functions():
        for ask in todo:
            level = tier(ask, record["name"])
            if level is None or (ask["id"], record["text"]) in seen:
                continue
            seen.add((ask["id"], record["text"]))
            form, hints_removed = plain(record["text"])
            if form is None or not self_contained(form):
                reasons[ask["id"]]["not self-contained"] += 1
                continue
            short = length(tok, form)
            if short is None:
                reasons[ask["id"]]["not representable by the tokenizer"] += 1
                continue
            if short > MAX_TOKENS:
                reasons[ask["id"]]["more than %d tokens without docstring" % MAX_TOKENS] += 1
            candidates[ask["id"]].append({"tier": level, "form": form, "tokens_without_docstring": short,
                                          "hints_removed": hints_removed, "file": file, "line": number, **record})
    mark = "REPORT-" + secrets.token_hex(8) + " "
    jobs = [(ask, cand) for ask in todo for cand in candidates[ask["id"]]]
    drivers = ["PROGRAM = %r\nTESTS = %r\nMARK = %r\n" % (program(ask, cand["form"], cand["name"]), ask["tests"], mark)
               + DRIVER for ask, cand in jobs]
    passing, too_long = collections.defaultdict(list), collections.defaultdict(list)
    for (ask, cand), result in zip(jobs, run_many(drivers, WORKERS, 5.0, MEMORY_MB)):
        outcome = _outcome(result, mark)[0]
        if cand["tokens_without_docstring"] > MAX_TOKENS:
            if outcome == "pass":
                too_long[ask["id"]].append(cand)
        elif outcome == "pass":
            passing[ask["id"]].append(cand)
        else:
            reasons[ask["id"]]["tests: " + outcome] += 1
    with open(SOLUTIONS, "w", encoding="utf-8") as solutions, open(MISSING, "w", encoding="utf-8") as missing:
        for ask in asks:
            tried = sum(cand["tokens_without_docstring"] <= MAX_TOKENS for cand in candidates[ask["id"]])
            if ask["id"] in kept:
                solutions.write(json.dumps(kept[ask["id"]]) + "\n")
            elif passing[ask["id"]]:
                best = min(passing[ask["id"]], key=lambda cand: (cand["tier"], cand["tokens_without_docstring"]))
                solutions.write(json.dumps({"id": ask["id"], "name": ask["name"], "text": best["text"], "source": best["source"],
                                            "licence": best["licence"], "path": best["path"], "original_name": best["name"],
                                            "tokens": length(tok, best["text"]), "file": best["file"], "line": best["line"],
                                            "passing": len(passing[ask["id"]]), "tried": tried,
                                            "hints_removed": best["hints_removed"],
                                            "tokens_without_docstring": best["tokens_without_docstring"]}) + "\n")
            else:
                failure = reasons[ask["id"]].most_common(1)
                missing.write(json.dumps({"id": ask["id"], "name": ask["name"], "tried": tried,
                                          "most_common_failure": failure[0][0] if failure else "no function with such a name",
                                          "failures": dict(reasons[ask["id"]]),
                                          "passing_but_too_long": sorted(
                                              ({"original_name": cand["name"], "source": cand["source"], "licence": cand["licence"],
                                                "path": cand["path"], "file": cand["file"], "line": cand["line"],
                                                "tokens": length(tok, cand["text"]),
                                                "tokens_without_docstring": cand["tokens_without_docstring"]}
                                               for cand in too_long[ask["id"]]), key=lambda row: row["tokens_without_docstring"])}) + "\n")
    rows = load_jsonl(MISSING)
    print("%d asks, %d with a verified human implementation (%d kept, %d chosen now), %d missing; %d candidates run in the sandbox"
          % (len(asks), len(asks) - len(rows), len(kept), len(asks) - len(rows) - len(kept), len(rows), len(jobs)))
    for row in rows:
        print("  missing %-30s tried %3d  %s" % (row["name"], row["tried"], row["most_common_failure"]))


if __name__ == "__main__":
    main()
