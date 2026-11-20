# The model and the course CPU: what agrees and what is missing

Written by `tools/cpu_alignment.py artifacts/board "../LS CPU"`. The VHDL is read, never changed. Nothing here was synthesised or run on
the board: cycle counts come from the cycle-exact simulator of the instruction set (`lsllm/ls/iss.py`).

## 1. Level 0, the CPU that is built

The eight programs in the program ROM of the built CPU were run in the model's simulator, with the X and W memories
of the VHDL, for all four images: **32 of 32 runs give the cycle count and the result byte that the VHDL testbench
expects** (`docs/L0_CROSSCHECK.md`). The two sides agree on the encoding and the cost of every instruction those programs
use: 4 cycles each, and n + 4 for `MAC #n`. The model's cycle counts are built on exactly these two rules.

The VHDL itself, simulated with nvc 1.23.1 on 2026-11-20: `tb_LSCore` prints ALL TESTS PASSED (all eight programs in run mode, with random clock-enable pauses and stepped by instruction), `tb_TopLevel` passes.

## 2. The program the board has to run

| | |
|---|---|
| model | `artifacts/board`: 350,880 trained weights; the integer model stores 350,080 int8 matrix entries, the 800 gains of the norms are folded into the matrices and the norm tables |
| program | 1,036 words of 16 bits (prompts over the UART, syntax filter, greedy) |
| data memory image | 450,432 bytes: weights, key/value cache, lookup tables, filter tables, scratch |
| block RAMs by our count, before any synthesis | 110 for the data image (4,096 bytes each) + 1 for the program = **111 of 135** |
| program ROM of the built CPU | 256 words (the program counter already has 12 bits) |
| data memories of the built CPU | X and W, 256 bytes each, pointers of 12 bits |

## 3. Instructions of that program

Static: words in the program. Dynamic: clock cycles in one complete request (`def hello_world(`, 15 tokens sent,
10,777,428 cycles). "As built" means the instruction is in the level-0 table with the same encoding and meaning.

| instruction | encoding | words in the program | share of the cycles | against the built CPU |
|---|---|---|---|---|
| `MAC` | opcode 8 | 34 | 75.13% | same encoding, wider meaning: multiplies D[P0] by D[P1] |
| `DJNZ` | opcode F | 52 | 3.84% | as built |
| `ST2` | opcode 5, two = 1 | 56 | 3.48% | new: `ST` through the store pointer P2 (bit 7 of the word) |
| `LDA` | opcode 7, fn = 1 | 48 | 3.36% | as built |
| `RQ` | opcode 9 | 40 | 3.07% | as built |
| `SETP0` | opcode A | 103 | 2.17% | same encoding, wider meaning: sets the low 12 bits of a 20-bit pointer |
| `LD` | opcode 4 | 121 | 2.11% | same encoding, wider meaning: reads the one data memory D |
| `MACL` | opcode 8, len = 1 | 8 | 1.21% | new: stream of LEN multiply-adds (bit 8 of the `MAC` word) |
| `LUT` | opcode 0, fn = 4 | 26 | 0.90% | new |
| `SETP1` | opcode B | 26 | 0.69% | same encoding, wider meaning: sets the low 12 bits of a 20-bit pointer |
| `BZ` | opcode D | 22 | 0.58% | as built |
| `LD2` | opcode 4, two = 1 | 6 | 0.57% | new: `LD` through the store pointer P2 (bit 7 of the word) |
| `LDI` | opcode 1 | 209 | 0.53% | as built |
| `SUB` | opcode 2, fn = 1 | 15 | 0.46% | as built |
| `MVP` | opcode 3 | 43 | 0.40% | same encoding, wider meaning: also selects the store pointer P2 |
| `MVH` | opcode 3, one = 1 | 74 | 0.33% | new |
| `RQV` | opcode 9, var = 1 | 8 | 0.29% | new: requantise with the shift from a register (bit 8 of the `RQ` word) |
| `MUL` | opcode 6, fn = 0 | 6 | 0.29% | as built |
| `ADDP` | opcode 7, fn = 0 | 6 | 0.29% | as built |
| `ST` | opcode 5 | 39 | 0.13% | same encoding, wider meaning: writes D through P0 |
| `ADD` | opcode 2, fn = 0 | 21 | 0.12% | as built |
| `CPP20` | opcode 3, fn = 14 | 18 | 0.02% | new |
| `MOV` | opcode 2, fn = 5 | 22 | 0.02% | as built |
| `JMP` | opcode C | 8 | 0.01% | as built |
| `NRM` | opcode 0, fn = 7 | 2 | 0.01% | new |
| `BNZ` | opcode E | 6 | 0.00% | as built |
| `AND` | opcode 2, fn = 2 | 3 | 0.00% | as built |
| `TX` | opcode 0, fn = 5 | 3 | 0.00% | new |
| `MVL` | opcode 3, fn = 3 | 1 | 0.00% | new |
| `OR` | opcode 2, fn = 3 | 3 | 0.00% | as built |
| `RX` | opcode 0, fn = 8 | 2 | 0.00% | new |
| `XOR` | opcode 2, fn = 4 | 2 | 0.00% | as built |
| `SHL` | opcode 2, fn = 6 | 1 | 0.00% | as built |
| `SHR` | opcode 2, fn = 7 | 2 | 0.00% | as built |

In the level-1 table and not used by this program: `CPP01`, `CPP02`, `CPP10`, `CPP12`, `CPP21`, `HALT`, `IN`, `NOP`, `OUT`, `RND`, `SAR`.
The two stream instructions take 76.3% of all cycles: the multiply-accumulate unit of the built CPU is where the
time goes, and it is used as it is.

## 4. What the CPU still needs (level 1)

The full list with estimated cost is `docs/HARDWARE_CHANGES.md`, section 1; the exact encodings are `TABLE_L1` in
`lsllm/ls/isa.py` and Part A of `docs/SPEC.md`. In the order the program depends on them:

1. Program memory of at least 1,036 words (now 256). In block RAM the read is synchronous, so FETCH has to allow for it.
2. One data memory D of 450,432 bytes with two read ports instead of X and W of 256 bytes; P0 and P1 with 20 bits (now 12).
   The built data memories already register their read data for one clock, which is how block RAM behaves.
3. The store pointer P2 with `MVP`, `MVH`, `MVL` and `CPP`, and `LD` and `ST` through P2 (`LD2`, `ST2`).
4. The LEN register and `MACL` (a stream whose length is only known at run time).
5. `LUT` (64 tables of 256 bytes inside D), `NRM` and `RQV` (the division in the attention).
6. `TX` (115,200 baud, 8N1; two bytes per token) and `RX` (the request from the PC).
7. `RND` is in the table for sampling. This program decodes greedily and does not execute it.

Rules that the cycle counts assume and that must hold in the VHDL: every instruction takes 4 clocks; a stream of n
multiply-adds takes n + 4; `RX` waits while no byte has arrived (waiting clocks are not in the counts).

## 5. What to compare when the extended CPU simulates

`export/board_uart/` has the memory images (`mem/data.hex`, `mem/program.hex`, also as `.coe`), the VHDL package with
the constants, and `vectors/` with every intermediate value of the first tokens. The bytes sent by `TX` must equal
those of the simulator. Short requests to start with (from `results/board_C2bp.json`; cycles as counted when the
requests are sent one after the other in suite order, so a request sent alone can differ by a few hundred cycles):

| request | prompt tokens | tokens sent | cycles | time at 100 MHz |
|---|---|---|---|---|
| `def square(` | 7 | 8 | 6,122,860 | 0.06 s |
| `def sum_list(` | 6 | 9 | 6,237,732 | 0.06 s |
| `def lowercase(` | 7 | 9 | 6,617,924 | 0.07 s |
| `def is_odd(` | 6 | 10 | 6,732,800 | 0.07 s |
| `def is_even(` | 6 | 10 | 6,732,904 | 0.07 s |
| `def bubble_sort(` | 10 | 81 | 46,611,752 | 0.47 s |

## 6. Numbers for the documents, with their source

| statement | value | source |
|---|---|---|
| weights | 350,880 trained weights, all carried by int8 values: 350,080 int8 matrix entries, and the 800 norm gains folded into the matrices and the norm tables | `artifacts/board/model.json`, `model.npz` |
| memory by our count, before synthesis | 110 block RAMs for the data image plus 1 for the program: 111 of 135 | this file, section 2 |
| predicted speed at 100 MHz over the 401 test requests | 167.9 tokens per second (29,135 tokens in 17,347,450,624 cycles) | `results/board_C2bp.json` |
| the CPU program sends the tokens of the integer reference | in 401 of 401 requests | `results/board_C2bp.json` |
| programs that pass the tests of their request | 400 of 401 | `results/board_C2bp.json` |
| what kind of result this is | recall of programs the model was trained on, not a benchmark; it does not write arbitrary Python | `REPORT.md`, section 3 |
| board results | none yet | - |
