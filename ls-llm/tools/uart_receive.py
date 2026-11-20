#!/usr/bin/env python3
"""Receive tokens from the LS CPU over the USB-UART bridge and print the Python text live.

Frame (docs/SPEC.md Part D): two bytes per token, first 0x80 | (id >> 7), then id & 0x7F. A byte with the top bit
set always starts a token, so the receiver re-synchronises by itself after a lost byte. 115200 baud, 8 data bits,
no parity, 1 stop bit. The end-of-text token ends a function: the script prints a blank line and starts again.

    python tools/uart_receive.py --port /dev/tty.usbserial-XXXX --tok data/tok/f512/tok.model
    python tools/uart_receive.py --replay tokens.bin --tok data/tok/f512/tok.model     (no board needed)

Needs `pyserial` for --port (pip install pyserial); --replay needs nothing but this repository.
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
from lsllm.tok import Tok


def frames(read_byte):
    """Yield token ids from a byte source that returns an int, or None at the end."""
    high = None
    while True:
        b = read_byte()
        if b is None:
            return
        if b & 0x80:
            high = b & 0x7F
        elif high is not None:
            yield (high << 7) | b
            high = None


def encode(ids):
    """The bytes the CPU sends for a list of token ids (used by tests and by --replay files)."""
    return bytes(x for t in ids for x in (0x80 | (t >> 7), t & 0x7F))


def main():
    ap = argparse.ArgumentParser(description="Print the text the LS CPU generates.")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--replay", help="file with raw frame bytes instead of a serial port")
    ap.add_argument("--tok", required=True, help="tokenizer model file (tok.model)")
    ap.add_argument("--level", type=int, default=0, help="indentation level at the start")
    args = ap.parse_args()
    tok = Tok(args.tok)
    if args.replay:
        data = iter(open(args.replay, "rb").read())
        read_byte = lambda: next(data, None)
    else:
        import serial
        port = serial.Serial(args.port, args.baud, bytesize=8, parity="N", stopbits=1, timeout=None)
        read_byte = lambda: port.read(1)[0]
    level, count, start = args.level, 0, time.time()
    for t in frames(read_byte):
        count += 1
        if t == tok.eos_id:
            sys.stdout.write("\n\n")
            level = args.level
        elif t < tok.vocab_size:
            sys.stdout.write(tok.decode([t], level))
            level = tok.level_after([t], level)
        sys.stdout.flush()
    sys.stderr.write("\n%d tokens in %.1f s\n" % (count, time.time() - start))


if __name__ == "__main__":
    main()
