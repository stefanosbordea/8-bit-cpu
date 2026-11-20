# Level-1 conformance programs

Written by `tools/l1_conformance.py`. Ten small programs for the instructions that the model needs beyond the course CPU,
each run in the cycle-exact simulator (`lsllm/ls/iss.py`, class `L1`). The numbers below are what the extended CPU must show
in VHDL simulation. Nothing here ran on the board.

- `program.hex`: one 16-bit word per line, from address 0. `data.hex`: the data memory D, one byte per line from address 0.
- `listing.txt`: address, word, mnemonic and operands as written in the generator. `expected.json`: everything in the table.
- Cycles: from the first FETCH to the end of `HALT`, 4 per instruction and n more for a stream of n products. Clocks spent
  waiting in `RX` for a byte are not counted (in `rx_echo` both bytes are there before the program starts).
- Reset state: registers, pointers, LEN, ACC and flags at 0; the LFSR at 0xACE1ACE1 xor the switches (switches 0 here).
- `LUT` reads D[256 t + Rd] (table base 0).

| program | what it checks | words | cycles | OUT | TX bytes | R0 R1 R2 R3 | P0 P1 P2 | LEN | ACC |
|---|---|---|---|---|---|---|---|---|---|
| `pointers` | MVP, MVH and CPP build a 20-bit address; ST through P0 and LD through P2 meet at the same byte | 13 | 52 | 0x5C | - | 5C 34 01 5C | 01A35 34A35 01A35 | 0 | 0 |
| `store_load` | ST through P0 and through P2 with post-increment, read back through P1 and P2, combined with XOR and ADD | 24 | 96 | 0xE0 | - | E0 8F 11 7E | 00201 00202 02001 | 0 | 0 |
| `mac_stream` | MAC #16 over two rows of the one data memory, then RQ; the classic MAC (one product) after it | 15 | 76 | 0xA3 | - | 01 00 A3 7F | 00101 01201 00000 | 0 | 5337 |
| `macl` | MVL sets LEN to 300 and MACL runs a stream of that length: longer than MAC #n can say | 13 | 352 | 0x17 | - | 17 01 2C 00 | 0012C 0112C 00000 | 300 | 6130 |
| `lut` | LUT reads table 2 at the value of the register and sets Z and N; BZ and BNZ follow the flags | 14 | 48 | 0x0D | - | 00 0D 69 11 | 00000 00000 00000 | 0 | 0 |
| `nrm_rqv` | NRM counts the significant bits of ACC; RQV shifts by a register | 21 | 84 | 0xA6 | - | 00 A6 0F 18 | 00000 00000 00000 | 0 | -23131 |
| `tx` | TX sends four bytes, the last two built with the ALU | 10 | 40 | 0xCB | 41 8A CB 45 | CB 45 00 00 | 00000 00000 00000 | 0 | 0 |
| `rx_echo` | RX takes two bytes from the PC (0x12, 0x34); the program sends the first and their sum | 7 | 28 | 0x46 | 12 46 | 46 34 00 00 | 00000 00000 00000 | 0 | 0 |
| `rnd` | RND gives the next byte of the 32-bit LFSR (switches at 0); four bytes go out and are combined | 13 | 52 | 0xF3 | 83 31 6E 2F | F3 31 6E 2F | 00000 00000 00000 | 0 | 0 |
| `matrix_row_loop` | The kernel of the model in small: 4 rows of 8 weights times one input row, each result stored through P2, then summed | 24 | 236 | 0x5E | - | 00 5E 00 00 | 00048 01020 02004 | 0 | -1936 |

Instructions these programs execute: `ADD`, `ADDP`, `BZ`, `CPP12`, `CPP20`, `DJNZ`, `HALT`, `JMP`, `LD`, `LD2`, `LDA`, `LDI`, `LUT`, `MAC`, `MACL`, `MUL`, `MVH`, `MVL`, `MVP`, `NRM`, `OUT`, `RND`, `RQ`, `RQV`, `RX`, `SETP0`, `SETP1`, `SHR`, `ST`, `ST2`, `TX`, `XOR`.
Level-1 instructions not covered here: `CPP01`, `CPP02`, `CPP10`, `CPP21`.
The model program itself is the last test: `export/board_uart/` and section 5 of `docs/CPU_ALIGNMENT.md`.
