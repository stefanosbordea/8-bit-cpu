# UART: between the board and the laptop

- 115200 baud, 8 data bits, no parity, 1 stop bit (8N1), idle high, least significant bit first.
- Frame (docs/SPEC.md Part D): two bytes per token id, first `0x80 | (id >> 7)`, then `id & 0x7F`.
  A byte with the top bit set always starts a token, so the receiver finds the frame again after a lost byte.
  Example: token id 300 = 0x12C is sent as `0x82`, `0x2C`.
- The stream of one answer: the generated tokens only (the prompt is not sent back), one frame each as soon as
  the token is chosen. The last frame is the token that ended the text (the end-of-text token, or a line break
  that brings the indentation level to 0): an end mark, which the host drops. If the context fills up first
  there is no end mark. Then the CPU sends the end byte `0xFF` and waits for the next prompt; it never halts.
- The CPU sends a byte with `TX Rs` (SPEC A.2): the byte goes into the transmit FIFO, the instruction waits only
  if the FIFO is full. 2 bytes per token at 115200 baud allow 5,760 tokens per second, more than the CPU computes.
- At 100 MHz one bit lasts 100,000,000 / 115,200 = 868.06 clock cycles: a divider of 868 is 0.006% off.

## From the laptop to the board: the prompt

This program takes its prompts from the UART. It needs a UART receiver and the proposed instruction `RX Rd`
(docs/SPEC.md A.2: Rd <- the next byte of the receive FIFO, the CPU waits while the FIFO is empty).

1. After reset, and after every answer, the program waits for a prompt.
2. The host sends the token ids of the prompt in the same framing, two bytes per id (`0x80 | (id >> 7)`, then
   `id & 0x7F`), the BOS token first, and then the byte `0xFF`. No token id starts with `0xFF` (ids are below 4,096).
3. Tokens beyond min(T, 255) are read and dropped; a prompt without a token is ignored (`lsllm.ls.gen.build`).
4. The board answers: one frame per generated token, the end mark included, then the byte `0xFF`.
5. Back to 1. The KV cache position, the stop level and the filter state start again for every prompt; the
   random generator of the sampling is not reset between answers.

`lsllm.ls.gen.frame(ids)` gives the bytes of a prompt, `lsllm.ls.gen.answers(cpu)` splits what the simulator sent.
On the laptop `tools/board_ask.py MODEL_DIR "make me a bubble sort program" --port ...` turns a request into
a prompt, sends it and prints the answer (`--simulate` runs the same program in the simulator).

## Pins (Nexys A7-100T, FT2232HQ USB-UART bridge; reference manual section 6)

| FPGA pin | name in the Digilent master XDC | direction |
|---|---|---|
| D4 | `UART_RXD_OUT` | FPGA output: **the transmit pin of the CPU** (data to the PC) |
| C4 | `UART_TXD_IN` | FPGA input (data from the PC): **the receive pin of the CPU**, the prompts arrive here |

The Digilent names are from the point of view of the PC, which is why the FPGA's output is called RXD. Source in
this repository: `research/09_fpga_costs.md` ("C4 = UART_TXD_IN (FPGA input, data from the PC), D4 = UART_RXD_OUT
(FPGA output, data to the PC)", checked there against the reference manual section 6, the master XDC and
Digilent's GPIO demo, where `UART_TXD : out` is constrained to D4). Take the constraint lines (pin and
IOSTANDARD) from the master XDC of your board and compare: https://github.com/Digilent/digilent-xdc

## On the laptop

`tools/uart_receive.py` reads the port, joins the two bytes of each token and prints the Python text as it arrives.
`--level` is the indentation level after the prompt (1 for the prompt of the test vectors):

```
python tools/uart_receive.py --port /dev/tty.usbserial-XXXX --tok data/tok/f512p/tok.model --level 1
python tools/uart_receive.py --replay OUT_DIR/vectors/uart.bin --tok data/tok/f512p/tok.model --level 1
```

(From the root of the repository; OUT_DIR is this directory.)
`--port` needs `pyserial`; `--replay` reads a file of raw frame bytes: `vectors/uart.bin` holds the bytes this
export must send for the prompt of the test vectors, so the laptop side can be tried without a board. The script
prints every token it receives, the end mark too (a line break, or a blank line for the end-of-text token).
