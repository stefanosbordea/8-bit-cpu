# Level 1 in nine steps: task cards for the CPU's owners

The model needs the instructions of level 1 (`docs/SPEC.md`, Part A; encodings in `TABLE_L1`, `lsllm/ls/isa.py`).
These cards order that work so that every step ends with a check against known numbers. They name the modules of
the finished level-0 CPU as they are today. No VHDL is given here: the modules are written by the team. Each card
has two halves that can be written by different people and meet at the stated interface.

Rules that every step must keep, because all predicted numbers rest on them:

- every instruction takes 4 clocks; a stream of n products takes n more;
- clocks spent waiting in `RX` for a byte are not counted;
- the eight level-0 programs keep their cycle counts and results (112, 300, 560, 1,328, 80, 172, 304, 56).

What to compare with: `export/conformance/NAME/` (program, data, expected result) and the behavioural model
`export/vhdl_reference/ReferenceL1.vhd`, which runs the same files in the same simulator (`docs/VHDL_REFERENCE.md`).

| step | what changes | modules today | half A | half B | check (expected cycles, OUT) |
|---|---|---|---|---|---|
| 1 | Program memory of 4,096 words, loaded from a hex file, in block RAM | `ProgramMemory`, `ControlFSM`, `LSCore` | the memory with a registered read and a file to load | FETCH so that the registered read still fits 4 clocks; program start from address 0 | the eight level-0 programs, unchanged numbers |
| 2 | One data memory D with two read ports and one write port, loaded from a hex file; P0 and P1 of 20 bits; `SETP0`/`SETP1` set the low 12 bits, `MVH` the upper 8 | `DataMemories`, `Decoder`, `LSCore` | the memory and its ports | the pointers: width, `MVP` keeping the upper bits, `MVH` | `mac_stream`: 76, 0xA3 |
| 3 | Store pointer P2; `CPP`; `LD` and `ST` through P2 (bit 7 of the word); `ST` through P0 | `DataMemories`, `Decoder` | P2 and the six copies | the address and write-enable selection for `LD`, `ST` | `pointers`: 52, 0x5C. `store_load`: 96, 0xE0 |
| 4 | LEN register (12 bits), `MVL`, `MACL` (bit 8 of the `MAC` word) | `ControlFSM`, `Decoder`, `LSCore` | LEN and `MVL` | the stream counter widened from 8 to 12 bits and loaded from LEN | `macl`: 352, 0x17. `matrix_row_loop`: 236, 0x5E |
| 5 | `LUT Rd, #t`: Rd becomes D[256 t + Rd], Z and N follow | `Decoder`, `DataMemories`, `ResultFlags`, `RegisterFile` | the address 256 t + Rd on a read port inside the 4 clocks | write-back to Rd and the flags | `lut`: 48, 0x0D |
| 6 | `NRM Rd` (significant bits of ACC, 24 for a negative ACC) and `RQV` (shift taken from a register, bit 8 of the `RQ` word) | `Requantiser`, `Decoder`, `LSCore` | the priority encoder | the shift-amount selection in front of the requantiser | `nrm_rqv`: 84, 0xA6 |
| 7 | UART: `TX Rs` with a 16-byte FIFO, `RX Rd` that waits for a byte; 115,200 baud, 8N1 | new modules, `ControlFSM`, `TopLevel`, constraints | transmitter and FIFO | receiver, and the wait state that the cycle counter does not count | `tx`: 40, bytes 41 8A CB 45. `rx_echo`: 28, bytes 12 46 |
| 8 | `RND Rd`: 32-bit LFSR, 8 shifts per call, reset value 0xACE1ACE1 xor the switches | new module, `Decoder` | the LFSR | the instruction and write-back | `rnd`: 52, bytes 83 31 6E 2F |
| 9 | The model: `export/board_uart/mem/program.hex` and `data.hex` (1,036 words, 450,432 bytes) | the whole CPU | simulation: request `def square(` as UART bytes | board: `tools/board_verify.py artifacts/board --port DEVICE` | simulation: 17 bytes sent after 6,122,980 cycles, the text `def square(n): return n * n`. Board: every answer checked, tokens per second measured |

Notes:

- Step 8 is not needed by the model program (greedy decoding does not execute `RND`); it can come last or be left out.
- In step 2 the two memories X and W become two read ports of one memory, so a stream multiplies D[P0] by D[P1]. The
  built `DataMemories` already registers its read data for one clock, which is what block RAM does.
- `MAC #0` stays the classic `MAC`: one product, 4 clocks.
- Request bytes for step 9: two bytes per token id (0x80 | id >> 7, then id & 0x7F), then 0xFF. For `def square(`:
  `tools/board_ask.py artifacts/board "square" --simulate` prints the text; `lsllm.ls.gen.frame` gives the bytes.
- Milestone 2 is step 9 in simulation, milestone 3 is step 9 on the board.
