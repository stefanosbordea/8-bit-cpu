"""Request texts from human functions that call helper functions of their own file.

    python -m quick.asks_bundle            writes data/asks/bundled_texts.jsonl

For every ask without a human text of the usual length (data/asks/train_texts.jsonl): a corpus function
with the name of the ask or one of its other names is not self-contained only because it calls functions that
stand next to it in the same file (heap_sort calls heapify). The helpers are moved inside the function, unchanged,
at the top of its body; docstrings go, the function gets the name of the ask, the result is put in canonical form.
All steps are mechanical; no line of code is written. The text must have at most LONGEST tokens, be
self-contained and pass the tests of the ask in the sandbox; the shortest one is kept.
"""
import ast
import gzip
import json

from lsllm.asks import ASKS, DATA, load_jsonl, rename
from lsllm.asks.run import DRIVER, MEMORY_MB, _outcome
from lsllm.canon import canon
from lsllm.data.selfcontained import BUILTINS, free_names, self_contained
from lsllm.sandbox import run_many
from lsllm.tok import Tok
from quick.asks_texts import LONGEST

# licensed text; the last file also holds the functions that the benchmark-overlap filter took out of the corpus (a
# function called heap_sort has the name of a benchmark task): fine for a board model, never for a benchmark model
FILES = ["data/corpus/train.jsonl", "data/csn/train.jsonl", "data/csn2/train.jsonl", "data/corpus/all_functions.jsonl.gz"]


def no_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node, clean=False) is not None and len(node.body) > 1:
            node.body = node.body[1:]
    return tree


def bundle(record, same_file, depth=3):
    """The function of `record` with the functions it calls from its own file moved inside; None if something else is missing."""
    main = ast.parse(record["text"]).body[0]
    inside, todo, have = [], [record["text"]], {record["name"]}
    for lap in range(depth):
        need = set()
        for text in todo:
            need |= {name for name in (free_names(text) or set()) if name not in BUILTINS and name not in have}
        if not need:
            break
        if not all(name in same_file for name in need):
            return None
        todo = [same_file[name]["text"] for name in sorted(need)]
        inside = [ast.parse(text).body[0] for text in todo] + inside
        have |= need
    else:
        return None
    if not inside:
        return None                                     # self-contained already: the miner has judged it
    doc = 1 if ast.get_docstring(main, clean=False) is not None else 0
    main.body = main.body[:doc] + inside + main.body[doc:]
    return ast.unparse(no_docstrings(ast.Module(body=[main], type_ignores=[])))


def main():
    tok = Tok("data/tok/f512v2/tok.model")
    have = set()
    for name in ("train_texts.jsonl",):               # a human text with helpers is preferred to a generated or a very long one
        have |= {row["name"] for row in load_jsonl(DATA / name)}
    asks = [ask for ask in load_jsonl(ASKS) if ask["name"] not in have]
    wanted = {}
    for ask in asks:
        for name in [ask["name"]] + ask.get("aliases", []):
            wanted.setdefault(name, []).append(ask)
    by_file, candidates = {}, []
    for path in FILES:
        with (gzip.open if path.endswith(".gz") else open)(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                record = json.loads(line)
                by_file.setdefault((record.get("source"), record.get("path")), {}).setdefault(record["name"], record)
                if record["name"] in wanted:
                    candidates.append(record)
    rows, programs = [], []
    for record in candidates:
        try:
            text = bundle(record, by_file[(record.get("source"), record.get("path"))])
        except (SyntaxError, ValueError, RecursionError):
            continue
        if text is None:
            continue
        for ask in wanted[record["name"]]:
            try:
                final = canon(rename(text, record["name"], ask["name"]))
                tokens = len(tok.encode(final))
            except (SyntaxError, ValueError):
                continue
            if tokens > LONGEST or not final.startswith("def %s(" % ask["name"]) or not self_contained(final, set()):
                continue
            rows.append({"id": ask["id"], "name": ask["name"], "text": final, "tokens": tokens, "source": record["source"], "path": record["path"],
                         "licence": record["licence"], "original_name": record["name"],
                         "note": "helper functions of the same file moved inside the function, unchanged; docstrings removed; renamed"})
            programs.append("PROGRAM = %r\nTESTS = %r\nMARK = %r\n" % (final, ask["tests"], "REPORT-b ") + DRIVER)
    kept = {}
    for row, result in zip(rows, run_many(programs, 2, 5.0, MEMORY_MB)):
        if _outcome(result, "REPORT-b ")[0] == "pass" and (row["name"] not in kept or row["tokens"] < kept[row["name"]]["tokens"]):
            kept[row["name"]] = row
    with open(DATA / "bundled_texts.jsonl", "w") as f:
        for row in kept.values():
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"asks_without_text": len(asks), "candidates_with_helpers_in_their_file": len(rows), "texts": len(kept)}))
    print("bundled:", [(row["name"], row["tokens"], row["source"]) for row in kept.values()])
    print("still without a text:", sorted(ask["name"] for ask in asks if ask["name"] not in kept))


if __name__ == "__main__":
    main()
