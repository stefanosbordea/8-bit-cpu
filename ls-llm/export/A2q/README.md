# Export of the integer model `runs/qat/A2q` for the LS CPU

Written by `.venv/bin/python -m lsllm.export`. Do not edit by hand. The program in `mem/` is the complete
program: it takes a prompt, generates until the text ends and sends every token over the UART; stop rule on,
syntax filter on, sampling at temperature 0.8.
Form of the prompt: **one prompt, written in the memory image**; the program halts after the answer.

| file | content |
|---|---|
| `mem/data.hex`, `mem/data.coe` | data memory at reset with the prompt in it, one byte per line; the `.coe` is for the Vivado block memory generator |
| `mem/map.json` | the regions of the data memory, for `tools/set_prompt.py` and for test benches |
| `mem/program.hex`, `mem/program.coe`, `mem/program.lst` | program ROM, one 16-bit word per line; listing |
| `MEMORY_MAP.md` | every region of the data memory, how to change the prompt, block RAM count, ROM size |
| `PROGRAM.md` | what the program does per token, ROM words and cycles per step, tokens per second |
| `weights/`, `weights/INDEX.md` | every array on its own, one byte per line |
| `ls_tables_pkg.vhd` | VHDL-2008 package: small tables, shifts, preloads |
| `vectors/`, `vectors/README.md` | test vectors: what the program must send, every intermediate value of position 0 and of the first generated tokens |
| `UART.md` | baud rate, frame, pins, receiver on the laptop |

## Round-trip check at the end of the export

- weights: 23 .hex files read back, 368,704 bytes, all equal to the arrays
- memory: data.hex and data.coe (450,432 bytes, the 21 prompt tokens of the test vectors inside), program.hex and program.coe (994 words) and map.json equal to the Build
- simulator on data.hex and program.hex from reset to HALT (stop rule on, syntax filter on, temperature 0.8), switches 0: 29 tokens sent [4, 238, 25, 449, 25, 449, 25, 449, 25, 288, 30, 47, 77, 452, 128, 439, 12, 449, 25, 449, 13, 4, 50, 12, 238, 25, 449, 13, 2], the last one is the end mark; 58 UART bytes equal to the integer reference; no accumulator overflow; 22,922,232 cycles = 0.23 s at 100 MHz for 21 prompt tokens and 29 generated ones
- vectors: 10 text files read back, equal to vectors.npz
- ls_tables_pkg.vhd: analysed without errors by `nvc --std=2008 -a`; 30 constants parsed back by Python, all equal to the model
