#!/usr/bin/env python3
"""Ask the model for a program by name, on the laptop.

    PYTHONPATH=. .venv/bin/python tools/ask.py MODEL "make me a bubble sort program" [--simulate]

MODEL is an exported integer model (a folder with model.json) or a float checkpoint (ckpt.pt). The request is
turned into the prompt `def bubble_sort(` by lsllm.asks.Resolver (the name, another listed name, a name inside a
longer sentence, or a typing error of one); the model writes the rest (syntax filter on, no docstring). If the request is one of the request suite, its tests are run in the
sandbox and the outcome is printed. --simulate (integer models only) also runs the generated LS program in the
cycle-exact simulator and prints the cycles and the time the board would need at 100 MHz.
"""
import argparse
import os

from lsllm.asks import ASKS, Resolver, load_jsonl
from lsllm.bench.harness import cut_at_top_level

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("request")
p.add_argument("--max-new-tokens", type=int, default=230)
p.add_argument("--simulate", action="store_true")
args = p.parse_args()
name, rule = Resolver()(args.request)                 # the PC decides which request of the suite is meant
prompt = "def %s(" % name
print("[request -> %s%s]" % (prompt, {"name": "", None: "; not one of the request suite"}.get(rule, "; found by the rule '%s'" % rule)))
integer = os.path.isdir(args.model)
if integer:
    from lsllm.intref.completer import IntCompleter
    completer = IntCompleter(args.model, syntax_filter="plain")
else:
    from lsllm.model.completer import TorchCompleter
    completer = TorchCompleter(args.model, syntax_filter="plain")
text = completer.complete(prompt, 1, args.max_new_tokens, 0.0, 0)[0]
program = prompt + cut_at_top_level(text)
print(program)
ask = next((a for a in load_jsonl(ASKS) if "def %s(" % a["name"] == prompt), None)
if ask:
    from lsllm.asks.run import run

    class Fixed:
        name = "fixed"

        def complete(self, prompt, n, max_new_tokens, temperature, seed):
            return [text] * n

    row = run(Fixed(), [ask])[0]
    print("\ntests of the request suite: %s %s" % (row["outcome"], row["detail"]))
else:
    print("\n(not one of the request suite: no tests to run)")
if args.simulate and integer:
    from lsllm.filter_table import TableFilter
    from lsllm.ls import gen
    spec, arrays = gen.load(args.model)
    tf = TableFilter(completer.tok, triple_quotes=False)
    b = gen.build(spec, arrays, prompt=True, tf=tf, syntax_filter=True)
    ids = completer.tok.encode(prompt, bos=True)
    gen.set_prompt(b.image, b, ids)
    cpu = gen.machine(b)
    cpu.run(limit=200_000_000)
    sent = gen.sent(cpu)
    print("LS program in the simulator: %d prompt tokens, %d tokens sent, %s cycles = %.2f s at 100 MHz; same text: %s" % (
        len(ids), len(sent), "{:,}".format(cpu.cycles), cpu.cycles / 1e8, completer.tok.decode(ids + sent[:-1]) == program))
