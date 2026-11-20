"""The PC side of the UART protocol of a program built with uart_prompt (docs/SPEC.md, Part D).

A request is the token ids of the prompt, two bytes per token (0x80 | id >> 7, then id & 0x7F), then the byte 0xFF.
The CPU answers in the same framing and ends with 0xFF; the last token before it is the one that ended the text.
115,200 baud, 8 data bits, no parity, 1 stop bit. Needs `pyserial`.
"""
import time

from lsllm.ls.gen import END, frame


def open_port(name, baud=115200):
    import serial
    return serial.Serial(name, baud, bytesize=8, parity="N", stopbits=1, timeout=None)


def ask(port, ids, timeout=120.0):
    """Send one request; returns (token ids of the answer, seconds from the last request byte to the end mark)."""
    port.reset_input_buffer()
    port.write(bytes(frame(ids)))
    port.flush()
    started, answer, high = time.perf_counter(), [], None
    port.timeout = timeout
    while True:
        got = port.read(1)
        if not got:
            raise TimeoutError("no end mark from the board within %.0f s (%d tokens received)" % (timeout, len(answer)))
        byte = got[0]
        if byte == END:
            return answer, time.perf_counter() - started
        if byte & 0x80:
            high = byte & 0x7F
        elif high is not None:
            answer.append((high << 7) | byte)
            high = None
