#!/usr/bin/env python3
"""Every request of the suite on the simulated CPU.

    PYTHONPATH=. .venv/bin/python tools/board_verify.py MODEL_DIR [--json FILE] [--table FILE] [--first N] [--port DEVICE]

With --port the same requests go over the UART to the board (or to tools/board_emulator.py) and the time of every
answer is measured on the PC: this is the measurement of tokens per second on the hardware.

For each request the prompt `def <name>(` goes to the generated LS program over the simulated UART (the program
is built once, with uart_prompt and the syntax filter in the no-docstring mode, and answers one request after
the other like the board). The tokens it sends are compared with the integer reference (lsllm.intref.completer,
same filter, greedy), turned into text, and the text is run against the tests of the request in the sandbox.
This is a recall test of the board model, not a benchmark.
"""
import argparse
import json
import time

from lsllm.asks import ASKS, load_jsonl
from lsllm.asks.run import run
from lsllm.filter_table import TableFilter
from lsllm.intref.completer import IntCompleter
from lsllm.ls import gen
from lsllm.tok import Tok

p = argparse.ArgumentParser()
p.add_argument("model", help="exported integer model (folder with model.json and model.npz)")
p.add_argument("--json", help="write one row per request to this file")
p.add_argument("--table", help="write the summary as a markdown table to this file")
p.add_argument("--first", type=int, help="only the first N requests")
p.add_argument("--port", help="a serial device: the requests go to the board (or to tools/board_emulator.py) instead of the simulator")
p.add_argument("--baud", type=int, default=115200)
args = p.parse_args()
spec, arrays = gen.load(args.model)
tok = Tok(spec["tokenizer"] + "/tok.model")
T = spec["config"]["max_seq_len"]
asks = load_jsonl(ASKS)[:args.first]
b = gen.build(spec, arrays, tf=TableFilter(tok, triple_quotes=False), syntax_filter=True, uart_prompt=True)
cpu = gen.machine(b)
reference = IntCompleter(args.model, syntax_filter="plain")
rows, before, started = [], 0, time.time()
if args.port:
    from lsllm.ls.uart import ask as ask_board, open_port
    port = open_port(args.port, args.baud)
for ask in asks:
    prompt = "def %s(" % ask["name"]
    ids = tok.encode(prompt, bos=True)
    seconds = None
    if args.port:                                       # the board answers; the time is measured on the PC
        answer, seconds = ask_board(port, ids)
    else:
        cpu.feed(bytes(gen.frame(ids)))
        cpu.run(limit=600_000_000)
        answer = gen.answers(cpu)[-2]                   # the last closed answer; its last token is the one that ended the text
    text = tok.decode(ids + answer[:-1])
    wanted = prompt + reference.complete(prompt, 1, T - len(ids), 0.0, 0)[0]
    rows.append({"id": ask["id"], "name": ask["name"], "category": ask["category"], "prompt_tokens": len(ids), "tokens_sent": len(answer),
                 "cycles": cpu.cycles - before if seconds is None else None, "seconds_measured": seconds, "same_as_reference": text == wanted, "text": text})
    before = cpu.cycles


class Sent:
    """The texts the simulated CPU sent, as a completer for the test run of lsllm.asks.run."""
    name = "ls-simulator"

    def __init__(self, texts):
        self.texts = texts

    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        return [self.texts[prompt][len(prompt):]] * n


for row, tested in zip(rows, run(Sent({"def %s(" % row["name"]: row["text"] for row in rows}), asks)):
    row["outcome"], row["detail"] = tested["outcome"], tested["detail"]
passed = [row for row in rows if row["outcome"] == "pass"]
same = sum(row["same_as_reference"] for row in rows)
tokens = sum(row["tokens_sent"] for row in rows)
lines = ["| | |", "|---|---|",
         "| requests sent to %s | %d |" % ("the board on " + args.port if args.port else "the simulated CPU", len(rows)),
         "| answers that pass the tests of their request | %d |" % len(passed),
         "| answers equal to the integer reference, token by token | %d of %d |" % (same, len(rows)),
         "| tokens sent | {:,} |".format(tokens)]
if args.port:                                           # measured: wall-clock time on the PC, UART transfer included
    total = sum(row["seconds_measured"] for row in rows)
    slowest = max(rows, key=lambda row: row["seconds_measured"])
    lines += ["| time for all requests, measured on the PC | %.1f s |" % total,
              "| time per request: median, longest | %.2f s, %.2f s (`%s`, %d tokens) |" % (
                  sorted(row["seconds_measured"] for row in rows)[len(rows) // 2], slowest["seconds_measured"], slowest["name"], slowest["tokens_sent"]),
              "| tokens per second over all requests, measured (reading the prompt and the UART included) | %.1f |" % (tokens / total)]
else:
    cycles = sum(row["cycles"] for row in rows)
    slowest = max(rows, key=lambda row: row["cycles"])
    lines += ["| cycles | {:,} |".format(cycles),
              "| time at 100 MHz, all requests | %.1f s |" % (cycles / 1e8),
              "| time per request: median, longest | %.2f s, %.2f s (`%s`, %d tokens) |" % (
                  sorted(row["cycles"] for row in rows)[len(rows) // 2] / 1e8, slowest["cycles"] / 1e8, slowest["name"], slowest["tokens_sent"]),
              "| tokens per second over all requests (prompt tokens included in the time) | %.1f |" % (tokens / (cycles / 1e8)),
              "| program words, memory image | {:,} words, {:,} bytes |".format(len(b.words), len(b.image))]
print("\n".join(lines))
print("not passed:", ", ".join("%s (%s)" % (row["name"], row["outcome"]) for row in rows if row["outcome"] != "pass") or "none")
print("%s in %.0f s" % ("asked over the UART" if args.port else "simulated", time.time() - started))
if args.json:
    json.dump(rows, open(args.json, "w"), indent=1)
if args.table:
    open(args.table, "w").write("\n".join(lines) + "\n")
