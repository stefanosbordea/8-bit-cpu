`export/<model>/` is written by `python -m lsllm.export` (`lsllm/export/`). The export ends with a round-trip check:
the written memory files are loaded into the cycle-exact simulator of the LS CPU, run from reset to HALT, and the UART
bytes must equal those of the integer reference; every weight file is read back and compared. Nothing was synthesised
in Vivado or run on the board.

| file | content |
|---|---|
| `weights/<name>.hex`, `weights/INDEX.md` | every weight matrix, lookup table, RoPE table and filter table, one byte per line (two hex digits, two's complement), with shapes and value ranges |
| `mem/data.hex`, `mem/data.coe` | the complete data memory image, prompt included, as the program expects it (Xilinx `.coe` for block RAM initialisation) |
| `mem/program.hex`, `mem/program.coe`, `mem/program.lst` | the LS program, one 16-bit word per line, and its listing |
| `ls_tables_pkg.vhd` | VHDL-2008 package with the small tables, shifts and preloads as constants; analysed with `nvc --std=2008` |
| `MEMORY_MAP.md` | every region of the data memory with address and size, the block RAM count, where the prompt bytes are |
| `PROGRAM.md` | outline of the program with ROM words and cycles per component at context 16, 64 and 256 |
| `vectors/` | test vectors: the fixed prompt, the trace of position 0, and for the first 8 generated tokens every intermediate int8 value and accumulator value, the logits, the chosen token and the UART bytes |
| `UART.md` | 115,200 baud, 8N1, two bytes per token (`0x80 | (id >> 7)`, then `id & 0x7F`), the end mark, FPGA pin D4; decoded on the laptop by `tools/uart_receive.py` |

Another prompt is written into an existing export with `tools/set_prompt.py`.

**Three forms of the program** (`gen.build`, `docs/SPEC.md` Part D; all three run in the simulator and give the tokens of the
integer reference):

| form | how the prompt gets in | new hardware | data memory image | program |
|---|---|---|---|---|
| one prompt in the image (default) | written into the memory file (`tools/set_prompt.py`) | none | 450,432 bytes: 110 block RAMs | 977 words |
| table of up to 64 prompts (`--asks N`, `--prompts FILE`) | switches SW5..0 choose one at reset | none | 452,014 bytes with 64 request prompts: 111 block RAMs, one more than the Tier A budget | 1,008 words |
| prompts over the UART (`--uart-prompt`) | the PC sends the tokens of `def <name>(`; the program answers one request after the other | the proposed `RX` instruction | 450,432 bytes: 110 block RAMs | 1,036 words |

The board model (programs on request, section 3) is exported in the second and third form with `--no-docstring` (the filter
then never opens a docstring). `tools/board_ask.py MODEL_DIR "request" --port ...` is the PC side of the third form and
`--simulate` runs the same exchange in the simulator; `--list` prints the requests the model knows.
