# The exported programs in a VHDL simulator

Written by `tools/vhdl_reference.py`. `export/vhdl_reference/ReferenceL1.vhd` is a behavioural model of the level-1
instruction set in VHDL (it is not a design of the CPU: no states, no datapath). It reads the exported hex files, executes
them and counts cycles by the two rules (4 per instruction, n more for a stream of n). Simulator: nvc 1.23.1. Nothing ran on the board.

## The ten conformance programs

| program | cycles: Python simulator | cycles: VHDL | OUT | UART bytes | registers, pointers, LEN, ACC | agree |
|---|---|---|---|---|---|---|
| `lut` | 48 | 48 | 0x0D | - | same | yes |
| `mac_stream` | 76 | 76 | 0xA3 | - | same | yes |
| `macl` | 352 | 352 | 0x17 | - | same | yes |
| `matrix_row_loop` | 236 | 236 | 0x5E | - | same | yes |
| `nrm_rqv` | 84 | 84 | 0xA6 | - | same | yes |
| `pointers` | 52 | 52 | 0x5C | - | same | yes |
| `rnd` | 52 | 52 | 0xF3 | 83 31 6E 2F | same | yes |
| `rx_echo` | 28 | 28 | 0x46 | 12 46 | same | yes |
| `store_load` | 96 | 96 | 0xE0 | - | same | yes |
| `tx` | 40 | 40 | 0xCB | 41 8A CB 45 | same | yes |

## Whole requests of the board model, from `export/board_uart/mem/`

The request goes in as UART bytes; the model runs until it waits for the next request. Cycles and every byte sent are compared with
the Python simulator (`lsllm/ls/iss.py`), which is what all numbers of the report come from.

| request | bytes sent | program text from the VHDL run | cycles: Python simulator | cycles: VHDL | bytes equal | VHDL run time on this laptop |
|---|---|---|---|---|---|---|
| `def square(` | 17 | `def square(n): /     return n * n` | 6,122,980 | 6,122,980 | yes | 0 s |
| `def hello_world(` | 31 | `def hello_world(): /     return 'Hello, World!'` | 10,777,428 | 10,777,428 | yes | 1 s |
| `def is_even(` | 21 | `def is_even(n): /     return n % 2 == 0` | 6,733,024 | 6,733,024 | yes | 0 s |
| `def bubble_sort(` | 163 | `def bubble_sort(array): /     n = len(array) /     for i in range(n): ...` | 46,611,752 | 46,611,752 | yes | 1 s |
| `def binary_search_tree(` | 83 | `def binary_search_tree(number: int) -> int: /     result = 0 /     for...` | 25,997,384 | 25,997,384 | yes | 1 s |

All 401 requests of the suite and `def binary_search_tree(` (a name outside the suite, so its text is not a trained program), each from reset: **in 402 of 402 the VHDL run sends the same bytes in the same number of cycles as the Python simulator** (17,373,496,008 cycles in all, 567 s of VHDL simulation). The table shows the first ones and any that differ.

Everything agrees.
