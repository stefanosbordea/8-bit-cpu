# Export of the integer model `runs/qat/C2bp` for the LS CPU

Written by `.venv/bin/python -m lsllm.export`. Do not edit by hand. The program in `mem/` is the complete
program: it takes a prompt, generates until the text ends and sends every token over the UART; stop rule on,
syntax filter on, greedy.
Form of the prompt: **a table of prompts in the memory image, chosen by the switches** (`PROMPTS.md` lists them); the program halts after the answer.

| file | content |
|---|---|
| `mem/data.hex`, `mem/data.coe` | data memory at reset with the prompt table in it, one byte per line; the `.coe` is for the Vivado block memory generator |
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
- memory: data.hex and data.coe (452,014 bytes, the table of 64 prompts inside), program.hex and program.coe (1,008 words) and map.json equal to the Build
- simulator on data.hex and program.hex from reset to HALT (stop rule on, syntax filter on, greedy), switches 0 (prompt `def bubble_sort(`): 81 tokens sent [399, 27, 5, 33, 47, 145, 12, 399, 13, 4, 83, 285, 72, 256, 12, 433, 27, 5, 83, 471, 430, 72, 256, 12, 460, 25, 33, 241, 285, 241, 458, 27, 5, 71, 399, 14, 471, 15, 430, 130, 399, 14, 471, 430, 164, 458, 15, 20, 5, 399, 14, 471, 15, 25, 399, 14, 471, 430, 164, 458, 15, 430, 47, 12, 399, 14, 471, 430, 164, 458, 15, 25, 399, 14, 471, 15, 13, 8, 50, 399, 2], the last one is the end mark; 162 UART bytes equal to the integer reference; no accumulator overflow; 46,611,004 cycles = 0.47 s at 100 MHz for 10 prompt tokens and 81 generated ones
- simulator on data.hex and program.hex from reset to HALT (stop rule on, syntax filter on, greedy), switches 32 (prompt `def count_set_bits(`): 76 tokens sent [177, 20, 430, 158, 13, 430, 176, 158, 20, 5, 71, 344, 492, 430, 460, 20, 5, 387, 491, 44, 87, 261, 12, 18, 57, 414, 85, 324, 49, 445, 440, 226, 143, 355, 153, 448, 131, 121, 18, 13, 6, 330, 49, 47, 460, 4, 386, 151, 30, 177, 20, 5, 71, 344, 369, 277, 223, 458, 20, 5, 330, 49, 481, 47, 458, 6, 344, 469, 469, 47, 458, 6, 50, 330, 431, 2], the last one is the end mark; 152 UART bytes equal to the integer reference; no accumulator overflow; 43,724,416 cycles = 0.44 s at 100 MHz for 10 prompt tokens and 76 generated ones
- simulator on data.hex and program.hex from reset to HALT (stop rule on, syntax filter on, greedy), switches 63 (prompt `def flatten(`): 16 tokens sent [52, 474, 435, 27, 5, 50, 303, 12, 52, 474, 435, 25, 14, 15, 13, 2], the last one is the end mark; 32 UART bytes equal to the integer reference; no accumulator overflow; 10,113,840 cycles = 0.10 s at 100 MHz for 7 prompt tokens and 16 generated ones
- vectors: 10 text files read back, equal to vectors.npz
- ls_tables_pkg.vhd: analysed without errors by `nvc --std=2008 -a`; 29 constants parsed back by Python, all equal to the model
