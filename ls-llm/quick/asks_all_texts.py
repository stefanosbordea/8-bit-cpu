"""One verified file with the training text of every request: python -m quick.asks_all_texts OUT.jsonl [KEEP.jsonl]

KEEP.jsonl: the text file of the board model that training continues from. Its texts stay (a model that knows a
program is not asked to learn another one for the same request), as long as they still pass.

Order of preference per request: a human function within 230 tokens (train_texts.jsonl), a human function with its
helpers moved inside (bundled_texts.jsonl), a text of the local Qwen model (qwen_texts.jsonl), a human function of
231 to 254 tokens (long_texts.jsonl). Every text is run once more against the tests of its request as they are
now (tests may have been corrected since the text was stored) and must be accepted token by token by the syntax
filter of the board; a text that fails is left out and named.
"""
import json
import sys

import numpy as np

from lsllm.asks import ASKS, DATA, load_jsonl
from lsllm.asks.run import DRIVER, MEMORY_MB, _outcome
from lsllm.filter_table import TableFilter
from lsllm.sandbox import run_many
from lsllm.tok import Tok

KINDS = (("train_texts.jsonl", "human"), ("bundled_texts.jsonl", "human, helpers moved inside"), ("qwen_texts.jsonl", "Qwen2.5-Coder-1.5B"),
         ("long_texts.jsonl", "human, 231 to 254 tokens"))


def accepted(tok, flt, name, text):
    """True if the board's filter allows every token of the text after the request."""
    prompt = tok.encode("def %s(" % name, bos=True)
    try:
        ids = tok.encode_request("def %s(" % name, text)
    except ValueError:
        return False
    if len(ids) > 257:
        return False
    state = flt.start_ids(prompt)
    for token in ids[len(prompt):]:
        if not np.asarray(flt.allowed(state))[token]:
            return False
        if token != tok.eos_id:
            flt.step(state, token)
    return True


def main(out, keep=None):
    asks = {ask["name"]: ask for ask in load_jsonl(ASKS)}
    tok = Tok("data/tok/f512v2/tok.model")
    flt = TableFilter(tok, triple_quotes=False)
    rows = []
    if keep:
        rows += [dict(row, kind=row.get("kind") or ("Qwen2.5-Coder-1.5B" if row["source"].startswith("Qwen") else "human")) for row in load_jsonl(keep)
                 if row["name"] in asks]
    for name, kind in KINDS:
        if (DATA / name).exists():
            rows += [dict(row, kind=kind) for row in load_jsonl(DATA / name) if row["name"] in asks]
    results = run_many(["PROGRAM = %r\nTESTS = %r\nMARK = %r\n" % (row["text"], asks[row["name"]]["tests"], "REPORT-t ") + DRIVER for row in rows], 2, 5.0, MEMORY_MB)
    kept, dropped = {}, []
    for row, result in zip(rows, results):
        if row["name"] in kept:
            continue
        if _outcome(result, "REPORT-t ")[0] != "pass":
            dropped.append((row["name"], row["kind"], "fails the tests"))
        elif not accepted(tok, flt, row["name"], row["text"]):
            dropped.append((row["name"], row["kind"], "not accepted by the filter or too long"))
        else:
            kept[row["name"]] = row
    with open(out, "w") as f:
        for ask in asks.values():                       # in the order of the suite
            if ask["name"] in kept:
                f.write(json.dumps(kept[ask["name"]]) + "\n")
    kinds = {}
    for row in kept.values():
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(json.dumps({"asks": len(asks), "texts": len(kept), "by_kind": kinds, "tokens": sum(len(tok.encode(row["text"])) for row in kept.values())}))
    print("left out:", dropped)
    print("without a text:", sorted(set(asks) - set(kept)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
