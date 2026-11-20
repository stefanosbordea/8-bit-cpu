"""Recall test: does a model write the programs it was trained on when asked for them by name?

    .venv/bin/python -m lsllm.asks.run --completer torchp:runs/f/A2/ckpt.pt
    .venv/bin/python -m lsllm.asks.run --completer intp:DIR --json results/asks.json
    .venv/bin/python -m lsllm.asks.run --completer asksref

For every ask the first request phrasing becomes the prompt (lsllm.asks.request_to_prompt), the completer
continues it once (greedy), the result is cut at the first line in column 0 and run with the ask's tests in the
sandbox. Outcomes: pass, wrong (an assert failed), error (an exception), syntax (does not parse), timeout.
This measures recall of trained programs. It is not a benchmark and is never reported as one.
"""
import argparse
import json
import secrets

from lsllm.asks import ASKS, SOLUTIONS, load_jsonl, plain, rename, request_to_prompt
from lsllm.bench.completers import load_completer
from lsllm.bench.harness import DRIVER, MEMORY_MB, OUTCOMES, _outcome, _parses, cut_at_top_level
from lsllm.sandbox import run_many


class RefCompleter:
    """Returns the stored human implementation of the ask in its plain form under the ask's name
    (lsllm.asks.plain, lsllm.asks.rename; kind "asksref"). Expected: every ask with a solution passes."""
    name = "asksref"

    def __init__(self):
        self.continuation = {}      # prompt -> rest of the stored solution
        for row in load_jsonl(SOLUTIONS):
            prompt = "def " + row["name"] + "("
            text = rename(plain(row["text"])[0], row["original_name"], row["name"])
            if not text.startswith(prompt):
                raise ValueError(row["id"] + ": the stored solution does not start with " + prompt)
            self.continuation[prompt] = text[len(prompt):]

    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        return [self.continuation.get(prompt, "")] * n      # an ask without a solution: empty, a syntax error


def run(completer, asks, max_new_tokens=230, workers=2):
    """One row per ask: id, request, prompt, outcome, detail, completion."""
    mark = "REPORT-" + secrets.token_hex(8) + " "
    rows, drivers = [], []
    tok, context = getattr(completer, "tok", None), getattr(getattr(getattr(completer, "model", None), "cfg", None), "max_seq_len", None)
    for g, ask in enumerate(asks):
        prompt = request_to_prompt(ask["request"])
        # a model of ours writes until its context is full, like the program on the board (a few texts need more than 230 tokens)
        limit = context - len(tok.encode(prompt, bos=True)) if tok is not None and context else max_new_tokens
        text = completer.complete(prompt, 1, limit, 0.0, g)[0]
        program = prompt + cut_at_top_level(text)
        parses = _parses(program)
        if parses:
            drivers.append("PROGRAM = %r\nTESTS = %r\nMARK = %r\n" % (program, ask["tests"], mark) + DRIVER)
        rows.append({"id": ask["id"], "request": ask["request"], "prompt": prompt, "outcome": None if parses else "syntax",
                     "detail": "", "completion": text})
    results = iter(run_many(drivers, workers, 5.0, MEMORY_MB))
    for row in rows:
        if row["outcome"] is None:
            row["outcome"], row["detail"] = _outcome(next(results), mark)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Recall test of trained programs (not a benchmark).")
    parser.add_argument("--completer", required=True, help='"asksref" or "<kind>:<argument>", e.g. torchp:runs/f/A2/ckpt.pt')
    parser.add_argument("--max-new-tokens", type=int, default=230)
    parser.add_argument("--json", help="write the rows to this file")
    args = parser.parse_args()
    asks = load_jsonl(ASKS)
    rows = run(load_completer(args.completer), asks, args.max_new_tokens)
    solved = {row["id"] for row in load_jsonl(SOLUTIONS)}
    for row in rows:
        print("%-34s %-8s %s" % (row["id"], row["outcome"], "" if row["id"] in solved else "(no stored solution)"))
    print("recall test, %s: %d asks, " % (args.completer, len(rows))
          + ", ".join("%s %d" % (name, sum(row["outcome"] == name for row in rows)) for name in OUTCOMES))
    print("asks with a stored solution: %d of %d pass" % (sum(row["outcome"] == "pass" for row in rows if row["id"] in solved), len(solved)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(rows, out, indent=1)


if __name__ == "__main__":
    main()
