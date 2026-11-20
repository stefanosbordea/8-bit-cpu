# VHDL check: the six hand-written modules against the software model

Date: 2026-11-14. Files checked (on the Desktop, written by Loucas): `Decoder.vhd` (114 lines), `ControlFSM.vhd` (65),
`ALU.vhd` (59), `RegisterFile.vhd` (52), `ProgramCounter.vhd` (48), `ProgramMemory.vhd` (21).
Compared with: `lsllm/ls/isa.py` (table `TABLE` for L0, `TABLE_L1` for L1), `lsllm/ls/iss.py`, `docs/SPEC.md` Part A and
the design document `8-bit-cpu/docs/design` (sections 04, 05, 06, figures `formats`, `opcodes`, `fsm`).

**What was done and what was not.**
- The six VHDL files were only read. They were not changed, moved or formatted (SHA-1 of each file is the same
  before and after the check).
- Copies in a scratch directory were analysed and elaborated with `nvc 1.23.1 --std=2008`. All six pass with no
  error and no warning.
- A small testbench in the scratch directory ran the copies of three modules: the ALU (all 16 modes times all 65,536
  operand pairs), the decoder (all 65,536 instruction words) and the control FSM (30 clocks: reset, three instructions,
  a pause of the clock enable, HALT). A Python script compared the dumps with `isa.py`.
- `RegisterFile`, `ProgramCounter` and `ProgramMemory` were read, analysed and elaborated, not simulated.
- Nothing was synthesised. No timing, no Vivado. The modules are not wired together yet, so nothing was checked as a CPU.

## Summary

What matches the software model:
- **Decoder, L0.** All 26,255 valid words of the 28 instructions of `TABLE` decode correctly: opcode, the one strobe
  that goes high, `writesRegister`, and every operand field. 0 mismatches (simulated).
- **ALU.** All nine operations give the result, Z and N of `isa.py` for every pair of operands. 0 mismatches
  (simulated). SUB is Rd - Rs. MOV gives Rs and sets the flags from it. SHR shifts a 0 in. SAR keeps bit 7.
- **Control FSM, the classic loop.** FETCH, DECODE, EXECUTE, WRITEBACK: 4 enabled clocks per instruction. HALT takes
  4 clocks and then stays in HALTED. A pause of the clock enable holds the state (simulated).
- **Register file, program counter.** Sizes, reset values and the clock enable agree with the model and the design document.
- No latch, no missing case, no width error, no incomplete sensitivity list in any of the six files.

What does not match:
- **Streams.** The FSM has 5 of the 9 states. There is no STREAM, DRAIN1, DRAIN2, so `MAC #n` with n >= 1 would take
  4 clocks. The model says n + 4. Not written yet, not a mistake in what is written.
- **Program size.** The ROM has 256 words (as the design document says). The simulator allows 4,096. The L1 program
  has 2,061 words.
- **Undefined words.** The decoder looks only at the opcode (and at the function code for opcode 0). The model
  rejects a word with a reserved bit set or an unknown function code. This is the cause of every L1 collision below.
- **L1.** 4 of the 8 L1 additions collide with what the decoder does today: the opcode 3 group (`MVL`, `MVH`, `CPP`
  decode as `MVP`), `LD`/`ST` through P2 (decode as P0), `MACL` (decodes as classic `MAC`), `RQV` (decodes as `RQ`
  with a wrong shift). `LUT`, `TX`, `RND`, `NRM` do not collide: today they do nothing, like `NOP`.

Count over the six tables (59 rows): 46 the same, 3 different, 10 not written yet.

In the tables, "same?" is `yes`, `no`, or `not yet` (the VHDL has nothing for this item so far).
"(sim)" marks a row that the testbench checked. The other rows come from reading.

## Decoder.vhd

| Item | VHDL says | Software model says | Same? |
|---|---|---|---|
| Opcode | `instruction(15 downto 12)`, line 69 | bits 15..12 (`isa.py` line 3) | yes (sim) |
| First register (r1, Rd, the register of LD/ST) | bits 11..10, line 70 | bits 11..10 in formats R, I, M, Q | yes (sim) |
| Second register (r2) | bits 9..8, line 71 | bits 9..8 in format R | yes (sim) |
| Function code | bits 3..0, line 72 | bits 3..0 in format R | yes (sim) |
| imm8 of LDI, n of `MAC #n` | bits 7..0, line 73 | bits 7..0 in formats I, N | yes (sim) |
| imm12 of SETP0, SETP1, JMP, BZ, BNZ, DJNZ | bits 11..0, line 74 | bits 11..0 in format A | yes (sim) |
| Pointer select of LD, ST | bit 9, line 75 (0 = P0, 1 = P1) | bit 9 (`p`) | yes (sim) |
| Post-increment of LD, ST | bit 8, line 76 | bit 8 (`inc`) | yes (sim) |
| NOP | opcode 0, fn 0: no strobe goes high | opcode 0, fn 0, no effect | yes (sim) |
| HALT | opcode 0 and fn 1, lines 79-81 | opcode 0, fn 1 | yes (sim) |
| IN | opcode 0 and fn 2, lines 83-85 | opcode 0, fn 2, register in bits 11..10 | yes (sim) |
| OUT | opcode 0 and fn 3, lines 86-88 | opcode 0, fn 3, register in bits 11..10 | yes (sim) |
| One strobe per opcode 1..F | lines 91-105: LDI 1, ALU 2, MVP 3, LD 4, ST 5, MUL 6, ADDP/LDA 7, MAC 8, RQ 9, SETP0 A, SETP1 B, JMP C, BZ D, BNZ E, DJNZ F | the same opcodes | yes (sim) |
| Which instructions write Rd | LDI, ALU, LD, RQ, IN, lines 108-113. DJNZ is left for later (comment, line 107) | IN, LDI, the nine ALU instructions, LD, RQ write Rd. DJNZ writes R3 | yes (sim) |
| RQ: `relu` and the shift `sh` | no output of their own. `relu` is bit 9 = `pointerSelect`. `sh` is bits 4..0 = `smallNumber(4 downto 0)` | `relu` bit 9, `sh` bits 4..0, 0..23 | not yet |
| Which pointer MVP loads | not decoded. `functionCode` is passed on | fn 0 = P0, fn 1 = P1 | not yet |
| ADDP or LDA | one strobe for both, line 97. `functionCode` is passed on | fn 0 = ADDP, fn 1 = LDA | not yet |
| MAC: classic or stream | one strobe, line 98. `smallNumber` is passed on | n = 0 classic (4 cycles), n >= 1 stream (n + 4) | not yet |
| Which instructions set Z and N | no signal | only the nine ALU instructions (opcode 2). Not LDI, IN, LD, RQ, DJNZ | not yet |
| Reserved bits and unknown function codes | ignored. Example: opcode 0, fn 1 is HALT whatever bits 11..4 are | such a word is not an instruction (`decode` raises) | no (sim) |

Of the 38,914 words that neither L0 nor L1 defines, the decoder treats 3,059 as NOP. The other 35,855 raise a strobe
(for example 255 more words are HALT, 3,988 are an ALU job with a reserved bit set or an unknown function code). The assembler never
produces these words, so this only matters for a ROM typed by hand.

## ALU.vhd

`operandA` is Rd, `operandB` is Rs, `mode` is the function code.

| Item | VHDL says | Software model says | Same? |
|---|---|---|---|
| ADD, fn 0 | A + B, 8 bits, line 31 | (Rd + Rs) mod 256 | yes (sim) |
| SUB, fn 1 | A - B, 8 bits, line 33 | (Rd - Rs) mod 256 | yes (sim) |
| AND, fn 2 | A and B, line 35 | Rd & Rs | yes (sim) |
| OR, fn 3 | A or B, line 37 | Rd \| Rs | yes (sim) |
| XOR, fn 4 | A xor B, line 39 | Rd ^ Rs | yes (sim) |
| MOV, fn 5 | result = B, line 41. Flags come from the result like any other mode | result = Rs. Z and N are set from the copied value | yes (sim) |
| SHL, fn 6 | A shifted left by 1, a 0 comes in, bit 7 is lost, line 43 | (Rd << 1) mod 256 | yes (sim) |
| SHR, fn 7 | A shifted right by 1, a 0 comes in, line 45 | Rd >> 1 | yes (sim) |
| SAR, fn 8 | A shifted right by 1, bit 7 is kept, line 47 | (Rd >> 1) \| (Rd & 0x80) | yes (sim) |
| Z | 1 when the result is 0, line 56 | 1 when the result is 0 | yes (sim) |
| N | bit 7 of the result, line 58 | bit 7 of the result | yes (sim) |
| Carry, overflow | none | none | yes |
| fn 9..15 | result 0, so Z = 1, N = 0, line 49 | not an instruction | no (sim) |
| Z and N are stored | no. `isZero` and `isNegative` follow the inputs at once, lines 56-58 | Z and N are registers. Only an ALU instruction changes them (design: at the end of WRITEBACK) | not yet |

## RegisterFile.vhd

| Item | VHDL says | Software model says | Same? |
|---|---|---|---|
| Registers | 4 of 8 bits, lines 31-33 | R0..R3, 8 bits | yes |
| After reset | all 0, lines 39-41 | all 0 (`iss.py` line 44) | yes |
| Reset | synchronous, wins over the clock enable, line 39 | design document: synchronous reset (08-board) | yes |
| Write | on a rising edge when `clockEnable` and `writeEnable` are 1, lines 42-44 | one write per instruction; design: every register is gated by `ce` | yes |
| Read | two ports, no clock, lines 50-51 | an instruction reads Rd and Rs | yes |
| Write address | its own port `targetRegisterSelect`, line 23 | DJNZ writes R3 whatever bits 11..10 are | yes |

## ProgramCounter.vhd

| Item | VHDL says | Software model says | Same? |
|---|---|---|---|
| Width | 12 bits, line 26 | 12 bits (`MASK12`) | yes |
| After reset | 0, line 33 | 0 (`iss.py` line 46). The design document loads the start address of the program chosen by SW15..13 | yes |
| Next line | +1, wraps from 0xFFF to 0, line 40 | (PC + 1) & 0xFFF | yes |
| Jump | PC <- `jumpTarget` (12 bits), line 37 | PC <- imm12 | yes |
| Jump and next line together | jump wins, lines 35-38 | never together: +1 happens in FETCH, a taken branch in WRITEBACK | yes |
| Clock enable | no change while `clockEnable` is 0, line 34 | design: PC is gated by `ce` | yes |

## ProgramMemory.vhd

| Item | VHDL says | Software model says | Same? |
|---|---|---|---|
| Word | 16 bits, line 16 | 16 bits | yes |
| Size | 256 words, address of 8 bits, lines 8 and 16 | up to 4,096 words (`iss.py` line 35). Design document: 256 x 16, address PC[7:0]. L1 program: 2,061 words | no |
| Read | no clock, line 21 | design document: asynchronous read | yes |
| Empty program | every word 0x0000, line 18 | 0x0000 is NOP | yes |

## ControlFSM.vhd

| Item | VHDL says | Software model says | Same? |
|---|---|---|---|
| States | FETCH, DECODE, EXECUTE, WRITEBACK, HALTED, line 25 | nine: these five and IDLE, STREAM, DRAIN1, DRAIN2 | not yet |
| Ordinary instruction | 4 enabled clocks, lines 46-53 | 4 cycles | yes (sim) |
| HALT | FETCH, DECODE, EXECUTE, WRITEBACK, then HALTED, lines 49-51 | 4 cycles, then halted | yes (sim) |
| HALTED | stays until reset, line 55 | reset only | yes (sim) |
| `MAC` (n = 0) | 4 enabled clocks | 4 cycles | yes |
| `MAC #n`, n >= 1 | 4 enabled clocks. The FSM has no input that tells it about a stream | n + 4 cycles: FETCH, DECODE, n times STREAM, DRAIN1, DRAIN2 | not yet |
| Clock enable | the state holds while `clockEnable` is 0, line 36 | design: one state per enabled clock | yes (sim) |
| After reset | FETCH, line 35: the CPU runs as soon as reset is released | design: IDLE, waits for "go". The model has no IDLE; IDLE is not counted | not yet |
| INSTR-STEP (WRITEBACK or DRAIN2 back to IDLE) | not there | design document only; the model does not have run modes | not yet |

## L1 additions: do they collide with today's decoder?

"Decodes today as" is what the simulated decoder does with every word of that L1 instruction.

| L1 addition | Encoding | Decodes today as | Collision? | Smallest change |
|---|---|---|---|---|
| `LUT Rd, #t` | op 0, fn 4, t = IR[9:4] | nothing (like NOP), `writesRegister` = 0. All 256 words | no | One more comparison (op 0, fn 4) that also feeds `writesRegister` and the flag write. A new 6-bit output for IR[9:4]. Bits 9..8 are now part of t, so `secondRegisterSelect` is meaningless here. |
| `TX Rs` | op 0, fn 5 | nothing (like NOP) | no | One more comparison (op 0, fn 5). The register is in bits 11..10, like OUT. |
| `RND Rd` | op 0, fn 6 | nothing (like NOP), `writesRegister` = 0 | no | One more comparison (op 0, fn 6), added to `writesRegister`. |
| `NRM Rd` | op 0, fn 7 | nothing (like NOP), `writesRegister` = 0 | no | One more comparison (op 0, fn 7), added to `writesRegister`. |
| `MVL`, `MVH`, `CPP` | op 3, fn 3 / fn 4..6 / fn 9, 10, 11, 13, 14, 15 | `isMovePointer` = 1 for all 34 words, the same as MVP (line 93 looks only at the opcode) | **yes** | Make `isMovePointer` also test the function code (fn 0, 1 today; 0..2 at L1, read as a 2-bit pointer number). Give MVL (fn 3), MVH (fn[3:2] = 01) and CPP (fn[3] = 1) their own strobes. |
| `LD`/`ST` through P2 | op 4 / 5, pointer id = {IR[7], IR[9]} = 10 | LD or ST with `pointerSelect` = 0, that is P0 (line 75 ignores bit 7). All 16 words | **yes** | Make `pointerSelect` 2 bits wide: {IR[7], IR[9]}. |
| `MACL` | op 8, IR[8] = 1 | `isMac` = 1 with `smallNumber` = 0: a classic MAC, one product instead of LEN (line 98 ignores bit 8) | **yes** | A new output for IR[8] under opcode 8. The FSM goes to STREAM when n >= 1 or this bit is 1, and the counter loads LEN (12 bits) instead of n (8 bits). |
| `RQV Rd, Rs` | op 9, IR[8] = 1, Rs = IR[1:0], relu = IR[9] | `isRequantise` = 1, `writesRegister` = 1 (both right), but the shift would be IR[4:0] = the number of Rs (0..3), not its content. All 32 words | **yes** | A new output for IR[8] under opcode 9 that switches the shift amount from IR[4:0] to Rs[4:0]. Rs sits in IR[1:0], where no register select looks today, so the second read address needs one more source. |

Result: 4 of the 8 additions collide. In every case a bit that L0 calls "reserved, always 0" now has a meaning, and
the decoder ignores it. The four that do not collide are safe only because the opcode 0 strobes compare all four bits
of the function code (lines 79-88). Keep that when adding more.

Other L1 points that are not collisions: `MVP` with fn 2 (P2) already raises `isMovePointer`; `SETP0`, `SETP1` and
`MVP` keep bits 19..12 of the pointer (datapath, not decoder); `TX` may have to wait when the FIFO is full (the FSM
has no wait state; the model counts 4 cycles); the stream counter must be 12 bits for `MACL`.

## Risks and gaps

Ordered by how likely they are to cost time. None of them stops the six files from compiling.

1. **No instruction register yet.** `Decoder.vhd` line 7 takes the instruction as an input and `ProgramMemory.vhd`
   line 21 reads without a clock. If the ROM feeds the decoder directly, the PC has already moved on after FETCH and
   the decoder (and `haltRequested` in WRITEBACK, `ControlFSM.vhd` line 50) sees the next instruction. The design
   document loads IR at the end of FETCH, gated by `ce`.
2. **`writesRegister` does not know the state.** `Decoder.vhd` lines 108-113 are high during all four states. The
   register file reads without a clock (`RegisterFile.vhd` lines 50-51), so if `writeEnable` is not also tied to
   `isWriteBack`, `ADD R0, R1` adds four times and `SHL` shifts four times. The model writes once.
3. **Flags are not stored.** `ALU.vhd` lines 56-58 change with the inputs. During `BZ` the ALU mode is the low four
   bits of the branch address (`Decoder.vhd` line 72), so the raw `isZero` is meaningless there. Z and N need two
   flip-flops that load only for opcode 2 in WRITEBACK (at L1 also for `LUT`). LDI, IN, LD, RQ and DJNZ write a
   register and must leave the flags alone; `writesRegister` is therefore the wrong enable for the flags.
4. **Streams are missing in the FSM.** `ControlFSM.vhd` lines 25 and 43-57. Every `MAC #n` takes 4 clocks, and the FSM
   has no input for "this is a stream". The cycle counts of the model (n + 4) cannot be reproduced until STREAM,
   DRAIN1, DRAIN2 and the counter exist. IDLE and "go" are missing too: after reset the CPU starts at once (line 35).
5. **RQ shift amount.** The decoder has no 5-bit output for `sh`. `functionCode` (line 72) has only bits 3..0. Using
   it as the shift loses bit 4 and breaks shifts 16..23. `smallNumber(4 downto 0)` is the right source.
6. **Program address width.** `ProgramCounter.vhd` line 20 gives 12 bits, `ProgramMemory.vhd` line 8 takes 8. The top
   level must connect bits 7..0; a direct connection does not elaborate. A PC above 255 then reads a wrapped address
   without any sign of it. The simulator stops with an error when the PC leaves the program. For L1 the ROM must grow
   to 12 address bits; as a block RAM its read becomes synchronous, which moves the FETCH timing.
7. **OUT and ST read the first register field.** The comments in `Decoder.vhd` lines 10-13 call bits 11..10 "Rd" and
   bits 9..8 "Rs". For `OUT Rs` and `ST Rs` (and `TX` at L1) the source register is in bits 11..10, so the datapath
   must use `firstRegisterSelect` for them.
8. **DJNZ.** Left for later (`Decoder.vhd` line 107). The model: R3 <- R3 - 1 through the separate write address,
   jump when the new R3 is not 0, flags unchanged, R3 = 0 at the start gives 256 rounds.
9. **Undefined words do something.** `Decoder.vhd` lines 79-105 do not test reserved bits or unknown function codes.
   An ALU word with fn 9..15 clears Rd (`ALU.vhd` line 49 with `Decoder.vhd` line 109). Low risk: the assembler never
   emits such words. It becomes the L1 collisions above.
10. **Reset value of the PC.** `ProgramCounter.vhd` line 33 loads 0. The design document loads a start address from
    a table of eight, chosen by SW15..13. The model starts at 0, so this only matters for the board.
11. **Architecture name.** `RegisterFile.vhd` line 29 says `Behavioral`, the other five say `Behavioural`. It matters
    only if an instantiation names the architecture, for example `entity work.RegisterFile(Behavioural)`.
12. **Empty ROM.** `ProgramMemory.vhd` line 18 is all NOP: the CPU never reaches HALT. Fine as a placeholder.
13. **Undriven selects in simulation.** `to_integer` in `RegisterFile.vhd` lines 50-51 and `ProgramMemory.vhd` line 21
    prints a "metavalue" warning and returns 0 if a select input is still 'U' at time 0. Not seen in the run here,
    where all inputs were driven. Harmless.

Checked and fine: the two `case` statements cover every value (`ALU.vhd` line 49, `ControlFSM.vhd` lines 46-55), every
branch assigns its signal (no latch), the sensitivity lists are complete, all three clocked processes use one clock
edge with a synchronous reset that wins over the clock enable, and the start values of the signals agree with the
reset values.

## How the simulated rows were obtained

Scratch directory (not part of the project): copies of the six files, `tb_check.vhd`, `compare.py`. The testbench
writes one line per ALU case (1,048,576), per instruction word (65,536) and per FSM clock (30). The script runs the
effect functions of `isa.TABLE` on the same operands and decodes every word with `isa.decode` for `TABLE` and
`TABLE_L1`. Output: "ALU: 1048576 rows, mismatches: none"; "Decoder, L0: 26255 valid words of 28 instructions,
mismatches: 0".
