# UART: between the board and the laptop

- 115200 baud, 8 data bits, no parity, 1 stop bit (8N1), idle high, least significant bit first.
- Frame (docs/SPEC.md Part D): two bytes per token id, first `0x80 | (id >> 7)`, then `id & 0x7F`.
  A byte with the top bit set always starts a token, so the receiver finds the frame again after a lost byte.
  Example: token id 300 = 0x12C is sent as `0x82`, `0x2C`.
- The stream of one answer: the generated tokens only (the prompt is not sent back), one frame each as soon as
  the token is chosen. The last frame is the token that ended the text (the end-of-text token, or a line break
  that brings the indentation level to 0): an end mark, which the host drops. If the context fills up first
  there is no end mark. Then the CPU halts; a new run needs a reset.
- The CPU sends a byte with `TX Rs` (SPEC A.2): the byte goes into the transmit FIFO, the instruction waits only
  if the FIFO is full. 2 bytes per token at 115200 baud allow 5,760 tokens per second, more than the CPU computes.
- At 100 MHz one bit lasts 100,000,000 / 115,200 = 868.06 clock cycles: a divider of 868 is 0.006% off.

## Pins (Nexys A7-100T, FT2232HQ USB-UART bridge; reference manual section 6)

| FPGA pin | name in the Digilent master XDC | direction |
|---|---|---|
| D4 | `UART_RXD_OUT` | FPGA output: **the transmit pin of the CPU** (data to the PC) |
| C4 | `UART_TXD_IN` | FPGA input (data from the PC): not used, this program has no UART receiver |

The Digilent names are from the point of view of the PC, which is why the FPGA's output is called RXD. Source in
this repository: `research/09_fpga_costs.md` ("C4 = UART_TXD_IN (FPGA input, data from the PC), D4 = UART_RXD_OUT
(FPGA output, data to the PC)", checked there against the reference manual section 6, the master XDC and
Digilent's GPIO demo, where `UART_TXD : out` is constrained to D4). Take the constraint lines (pin and
IOSTANDARD) from the master XDC of your board and compare: https://github.com/Digilent/digilent-xdc

## On the laptop

`tools/uart_receive.py` reads the port, joins the two bytes of each token and prints the Python text as it arrives.
`--level` is the indentation level after the prompt (0 for the prompt of the test vectors):

```
python tools/uart_receive.py --port /dev/tty.usbserial-XXXX --tok data/tok/f512p/tok.model --level 0
python tools/uart_receive.py --replay OUT_DIR/vectors/uart.bin --tok data/tok/f512p/tok.model --level 0
```

(From the root of the repository; OUT_DIR is this directory.)
`--port` needs `pyserial`; `--replay` reads a file of raw frame bytes: `vectors/uart.bin` holds the bytes this
export must send for the prompt of the test vectors, so the laptop side can be tried without a board. The script
prints every token it receives, the end mark too (a line break, or a blank line for the end-of-text token).
