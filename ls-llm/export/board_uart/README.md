# Export of the integer model `runs/qat/C2bp` for the LS CPU

Written by `.venv/bin/python -m lsllm.export`. Do not edit by hand. The program in `mem/` is the complete
program: it takes a prompt, generates until the text ends and sends every token over the UART; stop rule on,
syntax filter on, greedy.
Form of the prompt: **prompts typed over the UART (needs the proposed instruction `RX`)**; after every answer the program sends the byte 0xFF and waits for the next prompt (`UART.md`).

| file | content |
|---|---|
| `mem/data.hex`, `mem/data.coe` | data memory at reset, one byte per line; the `.coe` is for the Vivado block memory generator |
| `mem/map.json` | the regions of the data memory, for `tools/set_prompt.py` and for test benches |
| `mem/program.hex`, `mem/program.coe`, `mem/program.lst` | program ROM, one 16-bit word per line; listing |
| `MEMORY_MAP.md` | every region of the data memory, how to change the prompt, block RAM count, ROM size |
| `PROGRAM.md` | what the program does per token, ROM words and cycles per step, tokens per second |
| `weights/`, `weights/INDEX.md` | every array on its own, one byte per line |
| `ls_tables_pkg.vhd` | VHDL-2008 package: small tables, shifts, preloads |
| `vectors/`, `vectors/README.md` | test vectors: what the program must send, every intermediate value of position 0 and of the first generated tokens |
| `UART.md` | baud rate, frame, pins, receiver on the laptop |

## Round-trip check at the end of the export

- weights: 22 .hex files read back, 368,448 bytes, all equal to the arrays
- memory: data.hex and data.coe (450,432 bytes, no prompt inside), program.hex and program.coe (1,036 words) and map.json equal to the Build
- simulator on data.hex and program.hex, prompts over the UART (stop rule on, syntax filter on, greedy): the prompt of the vectors (21 tokens) and `def add(` (5 tokens) sent in one run, switches 0; answers [4, 50, 67, 164, 481, 2] and [434, 25, 449, 27, 5, 53, 126, 97, 330, 431, 12, 434, 25, 449, 27, 5, 71, 449, 430, 223, 460, 20, 5, 50, 434, 6, 50, 126, 97, 330, 431, 2], each followed by the end byte 0xFF, then the CPU waits; 78 UART bytes equal to the integer reference; no accumulator overflow; 28,028,972 cycles = 0.28 s at 100 MHz
- vectors: 8 text files read back, equal to vectors.npz
- ls_tables_pkg.vhd: analysed without errors by `nvc --std=2008 -a`; 29 constants parsed back by Python, all equal to the model
