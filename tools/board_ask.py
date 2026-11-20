#!/usr/bin/env python3
"""Type a request, get the program from the LS CPU.

    PYTHONPATH=. .venv/bin/python tools/board_ask.py MODEL_DIR "make me a bubble sort program" --simulate
    PYTHONPATH=. .venv/bin/python tools/board_ask.py MODEL_DIR "bubble sort" --port /dev/tty.usbserial-XXXX

The request becomes the prompt `def bubble_sort(` (lsllm.asks.Resolver), the prompt becomes token ids, and
the ids go to the CPU over the UART: two bytes per token (0x80 | (id >> 7), then id & 0x7F), then the byte 0xFF.
The CPU (program built with uart_prompt, docs/SPEC.md Part D; it needs the proposed `RX` instruction) answers with
tokens in the same framing and ends with 0xFF. --simulate runs that program in the cycle-exact simulator instead
of a board and reports the cycles. --port needs `pyserial` (115200 baud, 8N1).
"""
import argparse
import sys

from lsllm.asks import Resolver
from lsllm.filter_table import TableFilter
from lsllm.ls import gen
from lsllm.tok import Tok

p = argparse.ArgumentParser()
p.add_argument("model", help="exported integer model (folder with model.json and model.npz)")
p.add_argument("request", nargs="*", help="one or more requests; the board answers them one after the other")
p.add_argument("--anyway", action="store_true", help="send a request that is not in the suite as typed")
p.add_argument("--list", action="store_true", help="print the requests this model was trained to answer (MODEL_DIR/texts.jsonl) and stop")
p.add_argument("--simulate", action="store_true")
p.add_argument("--port")
p.add_argument("--baud", type=int, default=115200)
args = p.parse_args()
if args.list:
    import json
    import os
    path = os.path.join(args.model, "texts.jsonl")
    names = sorted({json.loads(line)["name"] for line in open(path)}) if os.path.exists(path) else []
    print("%d requests with a trained program:" % len(names) if names else "this model folder has no texts.jsonl (not a board model)")
    print(", ".join(name.replace("_", " ") for name in names))
    sys.exit(0)
if not args.request:
    sys.exit("give at least one request, or --list")
spec, arrays = gen.load(args.model)
tok = Tok(spec["tokenizer"] + "/tok.model")
resolve = Resolver()                                    # the PC decides which request of the suite is meant
found = [resolve(request) for request in args.request]
prompts = ["def %s(" % name for name, rule in found]
for request, (name, rule) in zip(args.request, found):
    if rule not in ("name", None):
        print("[%r -> def %s(: found by the rule '%s']" % (request, name, rule))
unknown = [request for request, (name, rule) in zip(args.request, found) if rule is None]
if unknown and not args.anyway:                         # the model writes the programs it was trained on, nothing else
    for request in unknown:
        near = resolve.suggest(request)
        print("%r is not one of the requests this model knows.%s" % (request, " Close to it: " + ", ".join(name.replace("_", " ") for name in near) + "." if near else ""))
    print("--list prints all requests; --anyway sends the words as typed (the answer will look like Python and will rarely be right).")
    keep = [g for g, (name, rule) in enumerate(found) if rule is not None]
    args.request, found, prompts = [args.request[g] for g in keep], [found[g] for g in keep], [prompts[g] for g in keep]
    if not prompts:
        sys.exit(1)
ids = [tok.encode(prompt, bos=True) for prompt in prompts]


def show(prompt, answer):
    print(prompt + tok.decode(answer, tok.level_after(tok.encode(prompt, bos=True))))


if args.simulate:
    b = gen.build(spec, arrays, tf=TableFilter(tok, triple_quotes=False), syntax_filter=True, uart_prompt=True)
    cpu = gen.machine(b)
    before = 0
    for prompt, one in zip(prompts, ids):
        cpu.feed(bytes(gen.frame(one)))
        cpu.run(limit=400_000_000)
        answer = gen.answers(cpu)[-2]                   # the last closed answer; the open one after it is empty
        show(prompt, [t for t in answer[:-1]])          # its last token is the one that ended the text
        print("    [%d prompt tokens, %d tokens sent, %s cycles = %.2f s at 100 MHz]\n" % (
            len(one), len(answer), "{:,}".format(cpu.cycles - before), (cpu.cycles - before) / 1e8))
        before = cpu.cycles
elif args.port:
    from lsllm.ls.uart import ask, open_port
    port = open_port(args.port, args.baud)
    for prompt, one in zip(prompts, ids):
        answer, seconds = ask(port, one)
        show(prompt, answer[:-1])
        print("    [%d prompt tokens, %d tokens received in %.2f s = %.1f tokens per second, measured on the PC]\n" % (
            len(one), len(answer), seconds, len(answer) / seconds if seconds else 0.0))
else:
    sys.exit("give --simulate or --port")
