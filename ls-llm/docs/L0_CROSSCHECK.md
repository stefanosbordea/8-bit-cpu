# The course CPU's eight programs in the model's simulator

Written by `tools/l0_crosscheck.py` from the VHDL in `../LS CPU` (read, not changed): the program ROM, the X and W
memories, the start addresses and the numbers the VHDL testbench expects. The simulator is `lsllm.ls.iss.LS`, level L0:
the one the model's cycle counts are built on (every instruction 4 cycles, `MAC #n` n + 4).

32 of 32 runs agree (8 programs, 4 images each).
Instructions in the eight programs, by mnemonic of the model's table: ADDP 1, AND 8, DJNZ 3, HALT 8, IN 8, LD 2, LDA 8, LDI 27, MAC 101, MUL 1, MVP 8, NOP 57, OUT 8, RQ 8, SETP1 8.
Cycles spent per mnemonic over all runs: ADDP 1024, AND 128, DJNZ 2560, HALT 128, IN 128, LD 2048, LDA 128, LDI 432, MAC 3536, MUL 1024, MVP 128, OUT 128, RQ 128, SETP1 128.

| program | start | image | cycles: VHDL testbench | cycles: simulator | result: VHDL testbench | result: simulator | agree |
|---|---|---|---|---|---|---|---|
| S64 | 0x000 | 0 | 112 | 112 | 0x66 | 0x66 | yes |
| S64 | 0x000 | 1 | 112 | 112 | 0x37 | 0x37 | yes |
| S64 | 0x000 | 2 | 112 | 112 | 0x02 | 0x02 | yes |
| S64 | 0x000 | 3 | 112 | 112 | 0x01 | 0x01 | yes |
| U64 | 0x00C | 0 | 300 | 300 | 0x66 | 0x66 | yes |
| U64 | 0x00C | 1 | 300 | 300 | 0x37 | 0x37 | yes |
| U64 | 0x00C | 2 | 300 | 300 | 0x02 | 0x02 | yes |
| U64 | 0x00C | 3 | 300 | 300 | 0x01 | 0x01 | yes |
| L64 | 0x057 | 0 | 560 | 560 | 0x66 | 0x66 | yes |
| L64 | 0x057 | 1 | 560 | 560 | 0x37 | 0x37 | yes |
| L64 | 0x057 | 2 | 560 | 560 | 0x02 | 0x02 | yes |
| L64 | 0x057 | 3 | 560 | 560 | 0x01 | 0x01 | yes |
| B64 | 0x065 | 0 | 1328 | 1328 | 0x66 | 0x66 | yes |
| B64 | 0x065 | 1 | 1328 | 1328 | 0x37 | 0x37 | yes |
| B64 | 0x065 | 2 | 1328 | 1328 | 0x02 | 0x02 | yes |
| B64 | 0x065 | 3 | 1328 | 1328 | 0x01 | 0x01 | yes |
| S32 | 0x076 | 0 | 80 | 80 | 0x76 | 0x76 | yes |
| S32 | 0x076 | 1 | 80 | 80 | 0x49 | 0x49 | yes |
| S32 | 0x076 | 2 | 80 | 80 | 0x01 | 0x01 | yes |
| S32 | 0x076 | 3 | 80 | 80 | 0x17 | 0x17 | yes |
| U32 | 0x082 | 0 | 172 | 172 | 0x76 | 0x76 | yes |
| U32 | 0x082 | 1 | 172 | 172 | 0x49 | 0x49 | yes |
| U32 | 0x082 | 2 | 172 | 172 | 0x01 | 0x01 | yes |
| U32 | 0x082 | 3 | 172 | 172 | 0x17 | 0x17 | yes |
| L32 | 0x0AD | 0 | 304 | 304 | 0x76 | 0x76 | yes |
| L32 | 0x0AD | 1 | 304 | 304 | 0x49 | 0x49 | yes |
| L32 | 0x0AD | 2 | 304 | 304 | 0x01 | 0x01 | yes |
| L32 | 0x0AD | 3 | 304 | 304 | 0x17 | 0x17 | yes |
| MAC #8 demo | 0x0BB | 0 | 56 | 56 | 0x4F | 0x4F | yes |
| MAC #8 demo | 0x0BB | 1 | 56 | 56 | 0x2D | 0x2D | yes |
| MAC #8 demo | 0x0BB | 2 | 56 | 56 | 0x12 | 0x12 | yes |
| MAC #8 demo | 0x0BB | 3 | 56 | 56 | 0x01 | 0x01 | yes |
