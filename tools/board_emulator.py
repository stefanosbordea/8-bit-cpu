#!/usr/bin/env python3
"""A stand-in for the board on a pseudo-terminal: the generated LS program in the cycle-exact simulator, speaking the UART protocol.

    PYTHONPATH=. .venv/bin/python tools/board_emulator.py MODEL_DIR [--speed 1.0]

Prints the name of a serial device and waits. In another terminal use that name as the port:
    PYTHONPATH=. .venv/bin/python tools/board_ask.py MODEL_DIR "bubble sort" --port /dev/ttysNNN
The answer is held back until the time the CPU would need at 100 MHz has passed (--speed 2 is twice as fast, 0
does not wait), so the laptop side can be rehearsed and timed without the board. This is the simulator, not hardware.
"""
import argparse
import os
import pty
import time
import tty

from lsllm.filter_table import TableFilter
from lsllm.ls import gen
from lsllm.tok import Tok

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--speed", type=float, default=1.0, help="1: answers take the predicted time at 100 MHz; 0: as fast as the simulator")
args = p.parse_args()
spec, arrays = gen.load(args.model)
tok = Tok(spec["tokenizer"] + "/tok.model")
b = gen.build(spec, arrays, tf=TableFilter(tok, triple_quotes=False), syntax_filter=True, uart_prompt=True)
cpu = gen.machine(b)
master, slave = pty.openpty()
tty.setraw(slave)
print(os.ttyname(slave), flush=True)
sent, before, pending = 0, 0, bytearray()
while True:
    data = os.read(master, 4096)
    if not data:
        break
    pending += data
    while gen.END in pending:                           # one whole request has arrived
        cut = pending.index(gen.END) + 1
        request, pending = bytes(pending[:cut]), pending[cut:]
        started = time.perf_counter()
        cpu.feed(request)
        cpu.run(limit=2_000_000_000)
        seconds = (cpu.cycles - before) / 1e8
        before = cpu.cycles
        if args.speed > 0:
            time.sleep(max(0.0, seconds / args.speed - (time.perf_counter() - started)))
        os.write(master, bytes(cpu.tx[sent:]))
        print("request of %d bytes: %d bytes sent, %s cycles = %.2f s at 100 MHz" % (len(request), len(cpu.tx) - sent, "{:,}".format(int(seconds * 1e8)), seconds), flush=True)
        sent = len(cpu.tx)
