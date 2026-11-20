# 09 FPGA reference data for estimating LS CPU hardware changes on Artix-7 (LS-LLM research, 2026-11-13)

Target: Xilinx/AMD Artix-7 XC7A100T-1CSG324C on the Digilent Nexys A7-100T (speed grade -1, 100 MHz oscillator on E3).

Labels used below. [doc]: number copied from a primary document that was opened and read for this report (document,
table or page given). [derived]: our arithmetic on [doc] numbers. [estimate]: our LUT/FF estimate built from 7-series
primitive costs; it is not a synthesis result and can be off by about 50%. [cost model]: arithmetic on the cycle model
in `research/CONTEXT.md` (a row of n multiply-adds costs n + 24 cycles, element-wise work 12 to 32 cycles per element,
about 1.0 M cycles per token at T = 256); it is not a measurement. [UNVERIFIED: reason]: could not be opened.
Nothing here was synthesised or run on a board.

Fact-check pass, 2026-11-13 (second agent): [VERIFIED] marks a statement whose source was opened again and compared
number by number; source and verdict are listed in "## Verification log" at the end. Statements without a tag were
not re-checked. [derived], [estimate] and [cost model] figures remain arithmetic of this report even where their
inputs are [VERIFIED]; "re-computed" means that the arithmetic was repeated, not that the result was measured.

## 1. Answer in one paragraph

Every CPU-side hardware change under discussion is small against the device (63,400 LUTs, 126,800 flip-flops,
240 DSP48E1, 135 block RAMs of 36 Kb [VERIFIED]), so the choice should be made on cycles saved per hour of design and
verification work, not on area. The one block that is not small is the DDR2 controller of Tier B: its size for a
16-bit interface is not known, and the vendor's 72-bit figure of 9,267 LUTs would be 14.6% of the device [derived].
One DSP48E1 does a signed 8x8 multiply and a 48-bit accumulate in one slice (OPMODE 0100101), with optional
input (A, B: 0 to 2 stages), multiplier (M) and output (P) registers; the data sheet limit at speed grade -1 is
464 MHz fully pipelined, 257 MHz without the M register and 191 MHz with no pipeline register [VERIFIED], so 100 MHz
leaves a wide margin and the 24-bit accumulator is an LS design choice, not a DSP limit. Two 8x8 products can be
packed into one DSP48E1 only when they share one operand, with zero spare bits (shift 16) or one spare bit (shift 17,
upper operand limited to -127..127), which allows 1 or at most 3 to 4 accumulations before the low field overflows
[derived; arithmetic re-computed] (7 on the DSP48E2 of WP486 [VERIFIED]). Accumulating longer needs logic outside the
slice: the published DSP48E1 design of this kind (Double MAC, DATE 2017, Virtex-7) uses one guard bit, a carry
counter and a correction accumulator in fabric and requires the shared operand to be unsigned [VERIFIED]. Packing
saves nothing here because 239 DSP slices are idle: use one slice per lane. A 36 Kb block RAM holds 4,096 bytes at any
port width of 9, 18 or 36 bits, so 450,000 bytes need 110 of the 135 blocks; the ninth bit of every byte is ordinary
storage (9-bit weights for no extra blocks, or 4,608 bytes per block through a 72-bit port); the two ports may have
different widths, there are byte-wide write enables, and a read on one port of an address that the other port writes
in the same cycle returns undefined data unless the writing port is in READ_FIRST mode [VERIFIED]. For DDR2 the
vendor gives 9,267 LUTs and 6,038 flip-flops for a 72-bit controller [VERIFIED]; a figure for our 16-bit case, a read
latency in cycles and a measured sequential throughput were not found and must be measured (UG586 itself says that
read latency depends on the configuration and the traffic and that simulation should be run to analyse it
[VERIFIED]); the interface delivers 128 bits per 81.25 MHz user clock (1.3 GB/s peak) against 0.1 GB/s consumed by
one multiply-add lane, while the Digilent SRAM adapter gives one 16-bit word per 210 ns (about 9.5 MB/s; the 210 ns
is [VERIFIED] in an archived copy of Digilent's page) and is unusable as a weight stream. The FT2232H goes to
12 Mbaud; with a 100 MHz clock the exact rates are 1, 2 and 4 Mbaud; C4 is the FPGA input and D4 the FPGA output
[VERIFIED]. On the CONTEXT cost model a token is about 42% multiply-adds, 34% per-row control and 24% element-wise
work, so in-row lanes alone are capped at 1.73x. That cap does not hold for row-parallel lanes, which also share the
per-row control and reach x1.95 at 4 lanes on the same model (section 5, idea 5) [corrected by the fact-check: the
earlier text said "lanes alone are capped at 1.73x"]. Fusing the per-row control and streaming the element-wise work
cost about 150 to 300 LUTs together [estimate] and are worth more than lanes per unit of effort, and all three
together point to a ceiling of about 6x [cost model; arithmetic re-computed, the model itself is not validated by
simulation].

## 2. Proven (numbers copied from documents that were opened)

Device
- XC7A100T: 101,440 logic cells, 15,850 slices (4 LUTs and 8 flip-flops each), 1,188 Kb maximum distributed RAM,
  240 DSP48E1, 270 block RAMs of 18 Kb or 135 of 36 Kb (4,860 Kb), 6 clock management tiles (one MMCM and one PLL
  each). DS180 v2.6.1 Table 4, p. 3. https://docs.amd.com/go/en-US/ds180_7Series_Overview [VERIFIED]
- 63,400 six-input LUTs, 126,800 flip-flops, 11,100 SLICEL and 4,750 SLICEM (only SLICEM LUTs can be distributed RAM
  or shift registers). UG474 v1.9, "7 Series FPGA CLB Resources", Table 1. https://docs.amd.com/go/en-US/ug474_7Series_CLB [VERIFIED]
- The board carries the XC7A100T-1CSG324C, a 100 MHz oscillator on pin E3, 128 MiB DDR2, 16 MiB quad-SPI flash and a
  microSD slot. Nexys A7 Reference Manual (Digilent, copy dated 2018-12-25), feature list, sections 3 and 5.
  https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual (read through the mirror
  https://www.secs.oakland.edu/~llamocca/Courses/ECE2700/Boards/NexysA7_rm.pdf) [VERIFIED in the mirror and in
  the Wayback copy of the live Digilent page, snapshot 2024-11-07:
  https://web.archive.org/web/20241107031931/https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual]

DSP48E1 multiply-accumulate
- The slice is a 25 x 18 two's complement multiplier followed by a 48-bit accumulator, with a 25-bit pre-adder (D
  input) and a pattern detector. UG479 v1.10 p. 9 and p. 17. https://docs.amd.com/go/en-US/ug479_7Series_DSP48E1 [VERIFIED]
- Multiply-accumulate in one slice: X and Y multiplexers select the multiplier output M (OPMODE[3:0] = 0101) and the
  Z multiplexer selects P (OPMODE[6:4] = 010, needs PREG = 1), giving P = P + A x B. Z = 000 gives P = A x B, Z = 011
  gives P = C + A x B, Z = 001 adds the cascade input PCIN. UG479 Tables 2-7 to 2-9, p. 34. [VERIFIED] One slice can therefore
  accumulate or cascade-add in a given cycle, not both [derived].
- Pipeline registers and their settings: AREG 0/1/2, BREG 0/1/2, CREG 0/1, DREG 0/1, ADREG 0/1, MREG 0/1, PREG 0/1,
  plus OPMODEREG, ALUMODEREG, CARRYINREG, CARRYINSELREG, INMODEREG 0/1. UG479 Table 2-3, p. 26. [VERIFIED]
- "At least three pipeline registers are required for both multiply and multiply-accumulate operations to run at full
  speed"; with only one or two registers "the M register should always be used". UG479 p. 14 and p. 47. [VERIFIED: both
  quotes are on p. 14; p. 47 says "always use the M register"]
- Maximum frequency at speed grade -1: 464.25 MHz with all registers; 392.93 MHz with the pattern detector;
  257.47 MHz for a two-register multiply without MREG; 290.44 MHz with the pre-adder and no ADREG; 190.69 MHz with no
  pipeline registers (no MREG, no ADREG). DS181 v1.27.1 Table 31, p. 35 to 36.
  https://docs.amd.com/go/en-US/ds181_Artix_7_Data_Sheet [VERIFIED]
- Setup at -1: A or B input to the M register through the multiplier 3.29 ns; A or B input to the P register through
  multiplier and adder 5.48 ns; A input to AREG 0.37 ns, B input to BREG 0.45 ns; clock to P output 0.44 ns from PREG
  and 2.31 ns from MREG. DS181 Table 31, p. 33 to 35. [VERIFIED]
- Guard bits: 5 with 43-bit products, 12 if the A operand is limited to 18 bits (UG479 p. 25) [VERIFIED]. With 16-bit
  products (8 x 8) there are 32 spare bits in the 48-bit P register [derived].
- The pattern detector supports overflow, underflow and saturation detection at a chosen bit position through its mask
  (UG479 p. 45). A dual 24-bit or quad 12-bit adder mode exists, but then the multiplier must not be used
  (USE_SIMD, UG479 Table 2-3, p. 27). [VERIFIED]
- "A pipeline register should be used at the output of an SRL16 or block RAM before connecting it to the input of the
  DSP48E1 slice" (UG479 p. 53). One slice can be time-multiplexed: N x channel frequency <= maximum slice frequency
  (UG479 p. 52). [VERIFIED]

INT8 packing (WP486) and the DSP48E1
- WP486 v1.0.1 targets the DSP48E2 (27-bit pre-adder, 27 x 18 multiplier): a is shifted left by 18 bits and added to
  b in the pre-adder, then multiplied by a shared c, giving a x c and b x c in one 45-bit product. "With 2-bits
  remaining between the lower and upper product terms ... accumulation of up to 7 product terms only can be
  guaranteed"; 8 slices then do 7 x 2 multiply-adds (1.75x). WP486 p. 4.
  https://docs.amd.com/go/en-US/wp486-deep-learning-int8 [VERIFIED]
- WP486 p. 3: "At a minimum one of the inputs to the multiplier needs to be at least 24-bits and the carry accumulator
  needs to be 32-bits to perform two INT8 MACC concurrently on one DSP slice", and the upper operand must start at the
  17th bit or higher (16 + 8 = 24 bits). The technique needs the two products to share one operand ("sharing the same
  kernel weights", p. 1; "two sums of product terms in parallel with shared input", Figure 6). [VERIFIED]
- DSP48E1 port widths: 25-bit pre-adder and 25-bit multiplier input, 18-bit second input; the pre-adder has no
  saturation and operands should be limited to 24-bit sign-extended data (UG479 p. 40). Consequences [derived]:
  shift 16 fits for all int8 values (a x 65,536 + b lies inside 25 bits) but leaves 0 spare bits, so only one product
  per field can be held before the fields must be separated (16,384 <= 32,767 but 2 x 16,384 > 32,767); shift 17
  fits only if the upper operand avoids -128 (-128 x 131,072 - 128 is below -2^24) and then allows 3 accumulations
  (4 if the lower products also avoid 16,384). The DSP48E2 figure is 7. [derived; the fact-checker
  re-computed it and it holds: a x 2^16 + b and a x 2^17 + b against the 25-bit range -16,777,216 .. 16,777,215, lower
  products between -16,256 and 16,384, fields of 16 and 17 bits.] These limits apply when the two fields are simply
  accumulated; catching the carry in fabric lifts them (next item).
- A published DSP48E1 design exists: Double MAC (Nguyen, Kim, Lee, DATE 2017, doi 10.23919/DATE.2017.7927113) packs
  two 8-bit multiply-accumulates with a common operand into one DSP block of a Virtex-7 485T, as one (3n+1) x n-bit
  multiplication ("with the 25x18-bit multiplier ... n can be at most 8"). Its conditions: the common operand must be
  an unsigned integer; there is one guard bit; a carry out of the lower field is cleared at once and counted in a
  small counter; sign-correction terms are accumulated in a separate small accumulator and subtracted at the end.
  Reported cost per multiply-accumulate: 0.5 DSP, 11 LUTs, 12 flip-flops (Table I) [VERIFIED]
  https://past.date-conference.com/proceedings-archive/2017/pdf/0348.pdf . This replaces the earlier statement that
  no such source was found (section 3).
- A paper that generalises packing (Sommer et al. 2022, arXiv:2203.11028) treats the DSP48E2; its body text does not
  work out the DSP48E1 case (it only cites UG479). https://arxiv.org/abs/2203.11028 [VERIFIED]

Block RAM (RAMB36E1, RAMB18E1)
- Aspect ratios of a 36 Kb block: 32K x 1, 16K x 2, 8K x 4, 4K x 9, 2K x 18, 1K x 36; 512 x 72 only in simple
  dual-port mode; 64K x 1 by cascading two blocks. 18 Kb block: 16K x 1 to 1K x 18, and 512 x 36 in simple dual-port
  mode. UG473 v1.14 p. 11 and Tables 1-11 to 1-14, p. 29 to 30.
  https://docs.amd.com/go/en-US/ug473_7Series_Memory_Resources [VERIFIED]
- True dual port: two independent ports, each with its own clock, address, data, enable and write enable; "Each port
  can be configured in one of the available widths, independent of the other port" and read and write width of one
  port may differ (UG473 p. 11, p. 21). A narrow port addresses the wide word from the low-order bits upward
  (START = ADDR x width, Table 1-16, p. 33): byte 0 of an 8-bit port is bits 7..0 of the 32-bit word. [VERIFIED]
- Byte-wide write enables: 4 for a 36-bit true dual-port port, 8 in 72-bit simple dual-port mode, 2 for an 18 Kb block
  in 18-bit mode (UG473 Table 1-7, p. 24). [VERIFIED]
- The ninth bit per byte: "can store parity/error correction bits or serve as additional data bits. No specific
  function is performed on the ninth bit" (UG473 p. 25). Parity bits exist only for 9, 18, 36 (and 72) bit ports and
  must not be used when one port is narrower than 9 bits (UG473 p. 12 to 13, Table 1-1). [VERIFIED]
- Capacity [derived from Table 1-13]: 4,096 bytes per 36 Kb block at 9, 18 or 36-bit width when only 8 of every 9 bits
  are used; 4,608 bytes per block if the ninth bits are used (72-bit word = 9 bytes). 135 blocks = 552,960 bytes
  (540 KiB), or 622,080 bytes (607.5 KiB) with the ninth bits. [derived; re-computed]
- Output register: optional, per port (DO_REG); "Using the register mode is strongly recommended"; it adds one clock
  of latency (UG473 p. 12, p. 21). Clock to output at -1: 2.46 ns without and 0.89 ns with the output register;
  address setup 0.57 ns; maximum frequency 388.20 MHz (339.67 MHz in READ_FIRST simple dual-port mode with possible
  address overlap; 345.78 MHz cascaded). DS181 Table 30, p. 30 to 32. [VERIFIED]
- Read-during-write on the same port: WRITE_FIRST (default, output shows the written data), READ_FIRST (output shows
  the old data), NO_CHANGE (output keeps the last read value). UG473 p. 17 to 19. In simple dual-port mode WRITE_FIRST
  is mapped to NO_CHANGE in the 7 series (UG473 p. 9). [VERIFIED]
- Collision between ports with a common clock: if one port writes and the other reads the same address in the same
  cycle, the write succeeds; the read returns the old data only if the writing port is in READ_FIRST mode, otherwise
  the read data is undefined (X). Two writes to one address store undefined data. UG473 Table 1-4, p. 20. [VERIFIED]
- Contents can be initialised by the configuration bitstream (UG473 p. 11) [VERIFIED]: Tier A weights need no loader.
- Built-in FIFO in the same block: 8K x 4, 4K x 9, 2K x 18, 1K x 36, 512 x 72 for 36 Kb; independent clocks; no CLB
  logic for pointers and flags; read and write width must be equal ("Independent read and write port width selection
  is not supported in FIFO mode without the aid of external CLB logic"). EMPTY deasserts 3 (standard) or 4
  (fall-through) read clocks after a write. UG473 p. 47, Table 2-4 p. 54, Table 2-7 p. 57. Maximum 388.20 MHz at -1
  (DS181 Table 30). [VERIFIED]

CLB primitives (UG474 v1.9)
- One LUT is a 4:1 multiplexer; two LUTs make an 8:1 and four LUTs (one slice) a 16:1 multiplexer through the F7 and
  F8 multiplexers ("Multiplexers"). [VERIFIED]
- ROM: 64 x 1 in 1 LUT, 128 x 1 in 2, 256 x 1 in 4 ("Read Only Memory (ROM)", Table 1). Distributed RAM: RAM256X1S
  takes 4 LUTs; writes are synchronous, reads asynchronous ("Distributed RAM", Table 1). [VERIFIED]
- Carry logic: four bits per slice, cascadable ("Carry Logic"). LUT delay at -1: 0.13 ns; flip-flop clock to output
  0.53 ns (DS181 Table 27, p. 28). [VERIFIED]

DDR2 and the Xilinx memory interface generator (MIG)
- Board memory: "one Micron MT47H64M16HR-25:H DDR2 memory component, creating a single rank, 16-bit wide interface";
  MIG settings recommended by Digilent: maximum clock period 3,000 ps (667 Mbps), recommended 3,077 ps (650 Mbps),
  part MT47H64M16HR-25E, data width 16, internal Vref. Nexys A7 Reference Manual section 3.1, Table 3.1.1. [VERIFIED;
  the live page now writes the rates as 667 MT/s and 650 MT/s]
- Maximum DDR2 rate for Artix-7 speed grade -1: 667 Mb/s with a 4:1 controller, 620 Mb/s with a 2:1 controller.
  DS181 Table 16, p. 15. [VERIFIED]
- "Controller and user interface operate at 1/4 the memory clock frequency" (UG586, DDR3 and DDR2 chapter,
  "Features"). At 3,077 ps the memory clock is 325.0 MHz and the user clock 81.25 MHz [derived]. Read data is returned
  in request order with app_rd_data_valid (UG586 "Interfacing to the Core, User Interface, Read Path").
  https://docs.amd.com/go/en-US/ug586_7Series_MIS [VERIFIED]. UG586 "Controller Options" also documents a 2:1
  ratio: "The user interface data bus width of the 2:1 ratio is four times the width of the physical memory interface
  width, while the bus width of the 4:1 ratio is eight times ... The 2:1 ratio has lower latency" [VERIFIED].
  Digilent's Vivado board preset for the Nexys A7-100T uses the setting assumed here: `TimePeriod` 3077, `PHYRatio`
  4:1, data width 16, part MT47H64M16HR-25E [VERIFIED]
  https://github.com/Digilent/vivado-boards/blob/master/new/board_files/nexys-a7-100t/D.0/1.3/mig.prj
- The user data bus is 128 bits wide for this memory (app_rd_data(127 downto 0), app_wdf_mask(15 downto 0) in the
  Digilent wrapper `Ram2Ddr.vhd`), i.e. 16 bytes per accepted read command; 16 bytes x 81.25 MHz = 1.3 GB/s peak
  [derived]. https://github.com/Digilent/Nexys-A7-100T-OOB/blob/master/src/hdl/Ram2Ddr.vhd [VERIFIED]; UG586
  states the rule ("For nCK_PER_CLK = 4, DATA_WIDTH = NUM_DQ_PINS x 8").
- Vendor resource table: "7 series FPGAs DDR2 SDRAM: 9,267 LUTs, 6,038 flip-flops, 2 BUFG, 1 PLLE2, 1 MMCM, 0 block
  RAM", stated for a 72-bit interface; DDR3 72-bit: 14,016 LUTs, 9,019 flip-flops, 2 block RAMs. DS176 (MIG v4.2),
  "Resource Utilization", Table 1. https://docs.amd.com/r/en-US/ds176_7Series_MIS/Resource-Utilization [VERIFIED].
  The same table lists LPDDR2 for a 32-bit interface at 3,952 LUTs and 3,285 flip-flops: another memory standard, but
  the narrowest interface in the vendor table [VERIFIED].
- Calibration points from other controllers: Xylon logiMEM, 16-bit DDR3 on XC7A100T-1: 1,279 LUTs, 920 flip-flops,
  0 block RAM, AXI clock 100 MHz, DDR3 clock 400 MHz (logiMEM data sheet v1.1, Table 1) [VERIFIED].
  https://www.logicbricks.com/Documentation/Datasheets/IP/logiMEM_hds.pdf . The README of
  ultraembedded/core_ddr3_controller states that on an XC7A35T (20,800 LUTs) "the same project using the Xilinx MIG
  DDR3 controller takes 33% of the FPGA LUTs (vs 9% with this core)", i.e. about 6,900 LUTs for the whole MIG-based
  project [derived]. https://github.com/ultraembedded/core_ddr3_controller [VERIFIED: the quote, and
  20,800 LUTs for the XC7A35T in UG474 Table 1]
- Digilent DDR-to-SRAM adapter: "instantiates the memory controller and uses an asynchronous SRAM bus ... It trades
  memory bandwidth for simplicity" (Reference Manual section 3.1). The source shows one transaction per chip-enable
  pulse, 27-bit address, 16-bit data, byte lanes, a 200 MHz input clock, and that each read fetches a whole 128-bit
  burst and passes on one 16-bit word of it. In Digilent's own demo the cycle constant is `RW_CYCLE_NS := 1200`
  (`AudioDemo.vhd`).
  https://github.com/Digilent/Nexys-A7-100T-OOB/tree/master/src/hdl [VERIFIED]
- Which MIG sits behind the adapter (added by the fact-check): the 100T repository holds only `ddr.xci`, which names
  a `mig_a.prj` that is not in the repository. The `mig_a.prj` of the sister repository Nexys-A7-50T-OOB (target
  xc7a100t-csg324/-1) has 3,333 ps, 4:1, native interface and strict ordering, which fits the 128-bit ports of
  `Ram2Ddr.vhd` (user clock 75 MHz [derived]). Digilent's page for the component describes 3,333 ps with a 2:1 ratio
  instead, so page and repository are not the same version [VERIFIED]
  https://github.com/Digilent/Nexys-A7-50T-OOB/blob/master/src/other/mig_a.prj

UART through the FT2232HQ
- "The FT2232H supports a data transfer rate up to 12 Mbaud when configured as an RS232/RS422/RS485 UART ... does not
  support the baud rates of 7 Mbaud 9 Mbaud, 10 Mbaud and 11 Mbaud." The baud generator divides a 120 MHz reference
  (x16 or x10 clock, 14-bit prescaler plus fractional bits), "programmable from 183 baud to 12 million baud"; 4 kbyte
  transmit and receive buffers per channel; driver latency timer 16 ms by default. FT2232H data sheet v2.6
  (FT_000061), sections 4.1 and 4.2, p. 26 to 27. https://ftdichip.com/wp-content/uploads/2020/07/DS_FT2232H.pdf
  (read through the mirror https://www.fdi.ucm.es/profesor/mendias/das/docs/DS_FT2232H-1621240.pdf) [VERIFIED in
  the mirror and in the Wayback copy of the FTDI URL, both v2.6]
- Pins: C4 = `UART_TXD_IN` (FPGA input, data from the PC), D4 = `UART_RXD_OUT` (FPGA output, data to the PC),
  D3 = CTS, E5 = RTS; "Signal names that imply direction are from the point-of-view of the DTE ... the PC". Reference
  Manual section 6 and https://github.com/Digilent/digilent-xdc/blob/master/Nexys-A7-100T-Master.xdc . Digilent's
  GPIO demo confirms it: `UART_TXD : out STD_LOGIC` is constrained to D4.
  https://github.com/Digilent/Nexys-A7-100T-GPIO [VERIFIED]
- The reference manual states no baud limit. [VERIFIED: the word "baud" does not occur in either copy]

Calibration: published sizes of small processors and multipliers on 7-series
- PicoBlaze KCPSM6 (8-bit): "as small as 26 slices", "up to 240 MHz".
  https://www.amd.com/en/products/adaptive-socs-and-fpgas/intellectual-property/picoblaze.html [VERIFIED]
- SERV (bit-serial RV32I): 125 LUTs, 164 flip-flops on Artix-7. https://github.com/olofk/serv [VERIFIED]
- PicoRV32 (Vivado 2017.3, area-optimised): small 761 LUTs / 442 registers, regular 917 / 583, large (multiply,
  divide, barrel shifter, compressed, interrupts) 2,019 / 1,085; about 4 cycles per instruction.
  https://github.com/YosysHQ/picorv32 [VERIFIED]
- VexRiscv on Artix 7: smallest RV32I 504 LUTs / 505 flip-flops at 243 MHz; RV32IM without caches 1,418 / 949 at
  216 MHz. https://github.com/SpinalHDL/VexRiscv [VERIFIED; the README says "on the fastest speed grade"]
- MicroBlaze 32-bit on XC7A200T-3: minimum area 617 LUTs / 223 flip-flops at 235 MHz; microcontroller preset
  1,009 / 501 at 212 MHz. UG984 "Device Utilization - Artix 7". https://docs.amd.com/go/en-US/ug984-vivado-microblaze-ref [VERIFIED]
- An 8 x 8 multiplier written with `*` and built from LUTs costs 71 LUTs (Virtex-7, Vivado default options); Ullah,
  Nguyen, Kumar, IEEE Embedded Systems Letters 13(2), 2021.
  https://grk2767.tu-dresden.de/files/Images/people/chair-pd/Papers/ESL_acc_multiplier.pdf [VERIFIED]
- Xilinx `uart_tx6` (UART transmitter with a 16-byte FIFO, baud tick supplied from outside) instantiates 15 LUTs,
  8 SRL16E and 24 flip-flops (counted by us in the source file).
  https://github.com/grg-haas/fpga_miner_rework/blob/master/src/vhdl/uart_tx6.vhd [VERIFIED: recounted]
- Maximum-length LFSR taps with XNOR feedback: 16 bits: 16, 15, 13, 4; 32 bits: 32, 22, 2, 1; 31 bits: 31, 28.
  XAPP052 v1.1, Table 3. https://docs.amd.com/go/en-US/xapp052 [VERIFIED]

## 3. Uncertain or contested

- MIG resource use for our configuration (16-bit DDR2, native interface). For DDR2 only the 72-bit vendor figure was
  found (9,267 LUTs, 6,038 flip-flops) [VERIFIED]. The 16-bit number is not published in DS176 or in the DDR3/DDR2
  chapter of UG586 [VERIFIED: both were searched again] and was not found in our search; one more search by the
  fact-checker found none either. Bracket [estimate]: above the 1,279 LUTs of a size-optimised commercial 16-bit DDR3
  core, below 9,267; the vendor's 32-bit LPDDR2 row (3,952 LUTs) and the roughly 6,900 LUTs of a whole MIG DDR3
  project on an XC7A35T lie inside this bracket. Generating the core in Vivado and running
  `report_utilization -hierarchical` settles it in minutes.
- MIG read latency. UG586 gives timing diagrams but no cycle count from accepted command to app_rd_data_valid
  [VERIFIED: the DDR3/DDR2 chapter was searched for "latency"]. It says instead: "The overall read latency of the MIG
  7 series DDR3/DDR2 core is dependent on how the Memory Controller is configured, but most critically on the target
  traffic/access pattern and the number of commands already in the pipeline before the read command is issued. ...
  Simulation should be run to analyze read latency" (UG586, "Precharge Policy") [VERIFIED]. It has to be measured in
  simulation of the generated example design. (A clause about differing forum reports was removed by the fact-check:
  no forum post was cited.)
- Realistic sequential read throughput. No measurement on this board was found. 1.3 GB/s is the pin-rate peak
  (650 Mb/s x 16 pins). The delivered rate depends on how often app_rdy drops (refresh, row changes, controller
  queues). Because one multiply-add lane consumes only 0.1 GB/s, the answer matters only above about 8 lanes.
- Access time of the Digilent adapter. Digilent's "SRAM to DDR component" page gives a minimum read cycle of 210 ns
  and a minimum write cycle of 260 ns ("The only change that must be made is to the read cycle and write cycle wait
  times. They have increased to 210 and 260 ns, respectively"; Bus Timing table: tRC 210 ns, tWC 260 ns) [VERIFIED in
  the Wayback copy of 2024-10-11; the live page is behind a bot challenge]
  https://web.archive.org/web/20241011091426/https://digilent.com/reference/learn/programmable-logic/tutorials/nexys-4-ddr-sram-to-ddr-component/start .
  The page describes a MIG at 3,333 ps with a 2:1 ratio, the repository version is 4:1 (section 2), so the two
  figures belong to the documented version and were not measured for the repository version. Digilent's demo source
  uses 1,200 ns [VERIFIED]. Either value rules the adapter out for streaming.
- Whether 12 Mbaud is usable on this board. The chip supports it; the board manual is silent; with a 100 MHz clock
  there is no integer divisor (8.33 clocks per bit). FTDI application note AN_120 v1.2 could not be opened
  on the live site (403) but was read by the fact-checker in a Wayback copy; section 6.4 now follows it [VERIFIED].
  Rates above 3 Mbaud, which includes 4 Mbaud, use the 12 MHz reference and need driver version 2.4.20 or later;
  whether the host in use accepts 4 Mbaud was not tested.
- All LUT and flip-flop figures marked [estimate]. They come from primitive costs (one LUT per adder bit, one LUT per
  4:1 multiplexer bit, two compared bits per LUT). Synthesis results depend on coding style and on what Vivado shares.
- Read multiplexer of a large memory. A 450 KB memory is 110 blocks; whether Vivado builds it as bit slices (few LUTs,
  all blocks enabled) or as byte-wide blocks behind a wide multiplexer (several hundred LUTs, one block enabled) was
  not checked (UG901 not read). Either way a pipeline stage after the block RAM is advisable.
- The 71-LUT multiplier figure is for Virtex-7 and default options and the text does not say signed or unsigned.
  [VERIFIED: the sentence reads "an accurate 8 x 8 multiplier ... consumes 71 LUTs"]
- How fast the LS CPU itself can be clocked. Published processor frequencies (212 to 243 MHz for small cores on
  Artix-7) are for other designs and partly faster speed grades. The slack of the present LS build is the only
  relevant number.
- The split 42% / 34% / 24% of a token is our reconstruction of the CONTEXT cost model (423,168 multiply-adds,
  14,072 rows at 24 cycles, about 13,300 element-wise operations at about 18 cycles). It matches the counts in
  `04_tiny_hardware.md` [VERIFIED: 423,168 multiply-adds and 14,072 = 3,512 + 10,240 + 320 rows were re-computed from
  the baseline shape; the 13,300 x 18 element-wise term is a reconstruction and was not re-derived];
  `05_speed_techniques.md` gets 1.10 M cycles with a two-pass softmax [VERIFIED: 1,100,668 cycles there].
  Percentages, not absolute tokens per second, are the reliable part.
- The 2018 copy of the Digilent manual (the mirror that was read) says "1,188 Kbits of fast block RAM" in its
  feature list. DS180 gives 4,860 Kb of block RAM and 1,188 Kb of distributed RAM, so that copy mislabels the
  distributed RAM figure. The live page (Wayback copy of 2024-11-07) says "4,860 Kbits of fast block RAM", so the
  error exists in the old copy only [corrected by the fact-check; the earlier text blamed "the manual" in general].
- Packing on the DSP48E1: the limits in section 2 are our derivation from port widths. The earlier statement here,
  that no source implementing such packing on a DSP48E1 was found, is withdrawn: Double MAC (Nguyen, Kim, Lee, DATE
  2017) packs two 8-bit multiply-accumulates with a common operand into the 25 x 18 multiplier of a Virtex-7 DSP
  block [corrected by the fact-check; VERIFIED]. It is not our case as it stands: its common operand must be
  unsigned, and long accumulation relies on a carry counter and a correction accumulator in fabric. Whether it
  carries over to signed int8 x int8 streams was not worked out.

## 4. Not tried at our scale (about 260k to 400k int8 parameters, 8-bit multiply-accumulate CPU, 450 KB)

- No published design was found that streams an int8 transformer of this size from block RAM through 1 to 4 DSP48E1
  lanes under a small multi-cycle CPU on an Artix-7; the lane, control and element-wise savings below are our
  arithmetic.
- Operand packing is reported for convolution accelerators (WP486 on DSP48E2; Double MAC on the DSP48E1 of a
  Virtex-7 485T [VERIFIED]), not for sequential signed int8 dot products of 8 to 512 terms under a small CPU
  [corrected by the fact-check: the earlier text said "not on DSP48E1"].
- Using the ninth block RAM bit for a language model (9-bit weights, or 9 bytes per 72-bit word): not found.
- MIG used purely as a sequential weight stream at 100 to 400 MB/s on the Nexys A7: no measurement found.
- LUT and flip-flop counts for an argmax unit over 256 to 512 classes on 7-series: not found in our search; papers
  that contain an argmax stage report whole-design totals only.
- The published processor sizes are for 32-bit RISC-V cores, MicroBlaze and PicoBlaze; none includes a streamed
  multiply-accumulate unit, so they calibrate the scale (100 to 2,000 LUTs) and not the LS datapath.
- Read-only weights in LUT ROM next to block RAM (4 LUTs per 256 bits): the per-LUT cost is documented, a 100 KB
  class ROM with its read multiplexer at 100 MHz on this device is not.

## 5. Ranked ideas for LS-LLM

Cycle effects are [cost model] at T = 256 against about 1.0 M cycles per token (423k multiply-adds, 338k per-row
control, 239k element-wise). Resource figures are [estimate] from section 6.3. Score is unaffected unless stated; all
changes must stay bit-exact with the integer reference model.

| # | Idea | Expected effect on score / speed / memory | Cost to try | Risk | Keep-or-drop measurement | CPU change |
|---|------|-------------------------------------------|-------------|------|--------------------------|------------|
| 1 | Read the timing slack of the present build; if it allows, raise the clock with the MMCM (125 or 150 MHz) | Speed: x1.25 or x1.5 on every cycle count. Memory: none | One implementation run per target frequency | Unknown slack; long routes to 110 block RAMs at grade -1; UART divisor changes | Post-route worst slack >= 0 at the target for 3 implementation strategies, then a token-exact board run | No (clocking only) |
| 2 | Fuse clear, bias load, requantise and store: clear through the DSP Z multiplexer on the opening beat of a stream (P = A x B, or P = C + A x B for a bias), one instruction for requantise + store + pointer step | Speed: 14,072 rows x 8 cycles = 113k cycles (-11%, x1.13) | About 30 to 60 LUTs, 20 to 30 flip-flops; one opcode or mode bit; assembler support | Low; control only | Cycle-exact simulator count per token; token-exact output | Yes, small |
| 3 | Whole matrix-vector instruction: the FSM repeats stream, requantise, store, reload X pointer for a row count (per-row control 24 -> about 4 cycles); same idea as item 5 of `04_tiny_hardware.md` | Speed: 14,072 x 20 = 281k cycles (-28%, x1.39); includes idea 2 | About 60 to 120 LUTs, 50 to 80 flip-flops | Medium: bigger FSM, more states to verify | Same as 2, plus post-route slack | Yes |
| 4 | Element-wise streams: table-lookup stream (X -> 256-entry table -> X) and multiply-requantise-store stream, one element per clock | Speed: 239k -> about 14k cycles (-23%, x1.29); with idea 3: x2.03 | About 55 to 100 LUTs, 35 to 60 flip-flops, half a 36 Kb block for 8 tables | Medium: second X pointer, read and write of X in the same cycle (keep the write port in READ_FIRST or forbid overlap) | Same as 2 | Yes |
| 5 | Row-parallel lanes: 2 or 4 rows per pass, each lane its own DSP48E1 and accumulator, all lanes share the X byte; weights read 16 or 32 bits wide (rows interleaved by the exporter, or two rows through the two block RAM ports) | Speed: 4 lanes with one shared requantiser: 423k -> 106k multiply cycles and 24 -> 12 control cycles per row: -49% (x1.95); 2 lanes: -32% (x1.48) | +1 or +3 DSP, about 30 to 100 LUTs, 0 block RAM; exporter layout change | Medium: row counts not divisible by the lane count, attention and value-mixing layouts, more cases to verify; changes the one-multiplier character of the CPU (report separately) | Cycle-exact simulation; token-exact output; utilisation and slack reports | Yes, larger |
| 6 | In-row lanes: 2 or 4 bytes of X and of W per clock, lane products summed | Speed: -21% (2 lanes), -32% (4 lanes); limit -42% (x1.73) however many lanes, because control and element-wise work stay | +1 or +3 DSP, about 45 to 200 LUTs; X memory needs a wide read port beside byte writes; tail masking | Medium: alignment of every vector to the lane count; rows of 8 gain little | As 5 | Yes, larger |
| 7 | Use the ninth block RAM bit: (a) 9-bit weights at the same block count, or (b) 9 bytes per 72-bit word (+12.5% bytes: 110 blocks hold 506,880 instead of 450,560) | Score: unknown, to be tested by QAT. Memory: +1 bit per weight or +12.5% bytes. Speed: none | (a) weight path 9 bits wide, exporter; (b) 72-bit read port, byte select 0..8 (about 40 LUTs), two-part pointer | (a) leaves the int8-only rule; the 24-bit accumulator is then safe only up to 255 terms; (b) no power-of-two addressing | Practice-set score of three QAT runs at equal block count: int8, int9, int8 with +12.5% parameters | Yes |
| 8 | UART at 1 or 2 Mbaud (100 or 50 clocks per bit, exact on both sides [VERIFIED, AN_120]), or a 16-byte SRL FIFO in the transmitter | Speed: avoids waiting 8,680 cycles per byte at 115,200 baud (5 bytes per token = 43k cycles, 4%) | A constant, or 8 LUTs | Low; check the host port setting | Cycle count per token with and without output | No (peripheral) |
| 9 | Hardware argmax: 24-bit running maximum and index, updated as each logit is produced | Speed: 512 x about 18 cycles = 9k (-0.9%) | About 15 to 30 LUTs, 34 to 44 flip-flops | Low; define the tie rule (lowest index) in the reference model | Token-exact output | Yes, small |
| 10 | Tier B: DDR2 weight stream through the MIG native interface and a prefetch FIFO | Memory: 128 MiB. Speed: set by the lanes (0.1 GB/s each), not by DDR2 (1.3 GB/s peak) | MIG (size not known for 16 bits, at most 9,267 LUTs / 6,038 flip-flops) + about 100 to 200 LUTs, 100 to 150 flip-flops, 2 block RAM sites as FIFOs; a loader (5.6 minutes for 128 MiB at 4 Mbaud) | High: calibration, second clock domain, stall logic, long verification | Words per second measured by a counter on the board; cycles per weight byte; token-exact output | Yes (stall input for the stream) |
| 11 | Wider stream counter (8 -> 10 or 16 bits) | Speed: about 1.3k to 2.6k cycles (-0.1 to -0.3%) from unsplit T-long streams | About 10 to 25 LUTs, 8 to 16 flip-flops | Streams above 511 terms are no longer overflow-safe in 24 bits in the worst case | Cycle count; accumulator range assertion in the reference model | Yes, small |
| 12 | Read-only tables or extra weights in LUT ROM | Memory: 128 LUTs per KB plus the read multiplexer; 20,000 LUTs (32% of the device) = 156 KiB | Synthesis time; pipeline the read multiplexer | Congestion and timing; not tried | Utilisation, slack, build time | Yes (third memory region) |
| 13 | Double-pumped stream datapath (block RAM and DSP at 200 MHz, CPU at 100 MHz) | Speed: same as 2 lanes | 2x clock from the same MMCM; all block RAM and DSP registers on | 5 ns paths at grade -1; not worth it while DSP slices and block RAMs are idle | Post-route slack | Yes |
| - | Do not pursue: two multiplies packed in one DSP48E1 | None: needs a shared operand and logic outside the slice (Double MAC: unsigned shared operand, 11 LUTs and 12 flip-flops per multiply-accumulate [VERIFIED]), and 239 DSP slices are unused | - | - | - | - |
| - | Do not pursue: Digilent Ram2Ddr adapter as the weight stream | About 9.5 MB/s or less (one 16-bit word per 210 ns [VERIFIED]), a tenth of one lane's demand | - | - | - | - |

Ideas 2 + 3 + 4 + 5 (4 lanes) together give about 162k cycles per token [cost model], about 6x the baseline, for
about 200 to 400 LUTs, 3 DSP slices and half a block RAM for the tables [estimate]. This is a ceiling from the cost
model, not a forecast: it has to be reproduced on the cycle-accurate simulator before it is quoted anywhere.
Fact-check note: the sum was re-computed (106k multiply + 42k control + 14k element-wise = 162k; 1,000k / 162k = 6.2).
It assumes that all 14,072 rows, including the 10,240 attention-score dot products and the 320 value-mixing streams,
can be laid out four to a pass with about 3 control cycles per row; that layout has not been designed.

## 6. Tables asked for in the topic brief

### 6.1 Reference numbers

Short names: UG479 = https://docs.amd.com/go/en-US/ug479_7Series_DSP48E1 ; UG473 =
https://docs.amd.com/go/en-US/ug473_7Series_Memory_Resources ; DS181 =
https://docs.amd.com/go/en-US/ds181_Artix_7_Data_Sheet ; DS180 = https://docs.amd.com/go/en-US/ds180_7Series_Overview ;
WP486 = https://docs.amd.com/go/en-US/wp486-deep-learning-int8 ; UG474 = https://docs.amd.com/go/en-US/ug474_7Series_CLB ;
UG586 = https://docs.amd.com/go/en-US/ug586_7Series_MIS ; DS176 =
https://docs.amd.com/r/en-US/ds176_7Series_MIS/Resource-Utilization ; NexysRM =
https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual ; FT2232H =
https://ftdichip.com/wp-content/uploads/2020/07/DS_FT2232H.pdf . The URL column gives the short name.

| Item | Number | Conditions | Document and section | URL |
|------|--------|------------|----------------------|-----|
| LUTs / flip-flops / slices | 63,400 / 126,800 / 15,850 [VERIFIED] | XC7A100T | UG474 "7 Series FPGA CLB Resources" Table 1; DS180 Table 4 p. 3 | UG474, DS180 |
| SLICEM count | 4,750 (19,000 LUTs usable as RAM or SRL) [VERIFIED] | XC7A100T | UG474 Table 1 | UG474 |
| DSP48E1 slices | 240 (3 columns of 80) [VERIFIED] | XC7A100T | UG479 Table 2-1 p. 19; DS180 Table 4 | UG479 |
| Block RAM | 135 x 36 Kb = 4,860 Kb (4 columns) [VERIFIED] | XC7A100T | UG473 Table 1-2 p. 14; DS180 Table 4 | UG473 |
| Distributed RAM maximum | 1,188 Kb [VERIFIED] | XC7A100T | DS180 Table 4; UG474 Table 1 | DS180 |
| Clock management tiles | 6 (1 MMCM + 1 PLL each) [VERIFIED] | XC7A100T | DS180 Table 4, note 4 | DS180 |
| Global clock tree limit | 464 MHz [VERIFIED] | grade -1 | DS181 Table 32 p. 36 | DS181 |
| DSP multiplier and accumulator | 25 x 18 signed, 48-bit P [VERIFIED] | - | UG479 p. 9, p. 18 | UG479 |
| MACC in one slice | OPMODE 0100101 (X = M, Y = M, Z = P) [VERIFIED] | PREG = 1 | UG479 Tables 2-7 to 2-9 p. 34 | UG479 |
| DSP registers | A 0-2, B 0-2, C, D, AD, M, P 0-1 each [VERIFIED] | attributes AREG ... PREG | UG479 Table 2-3 p. 26 | UG479 |
| DSP Fmax, all registers | 464.25 MHz [VERIFIED] | grade -1 | DS181 Table 31 p. 35 | DS181 |
| DSP Fmax with pattern detector | 392.93 MHz [VERIFIED] | grade -1 | DS181 Table 31 p. 35 | DS181 |
| DSP Fmax, two registers, no MREG | 257.47 MHz [VERIFIED] | grade -1 | DS181 Table 31 p. 35 | DS181 |
| DSP Fmax, pre-adder, no ADREG | 290.44 MHz [VERIFIED] | grade -1 | DS181 Table 31 p. 35 | DS181 |
| DSP Fmax, no pipeline registers | 190.69 MHz [VERIFIED] | grade -1, no MREG, no ADREG | DS181 Table 31 p. 36 | DS181 |
| A or B pin to M register setup | 3.29 ns [VERIFIED] | grade -1, through multiplier | DS181 Table 31 p. 33 | DS181 |
| A or B pin to P register setup | 5.48 ns [VERIFIED] | grade -1, multiplier + adder, no MREG | DS181 Table 31 p. 33 | DS181 |
| A to AREG / B to BREG setup | 0.37 / 0.45 ns [VERIFIED] | grade -1 | DS181 Table 31 p. 33 | DS181 |
| Clock to P from PREG / from MREG | 0.44 / 2.31 ns [VERIFIED] | grade -1 | DS181 Table 31 p. 35 | DS181 |
| Pre-adder | 25 bits, no saturation, keep operands to 24 bits [VERIFIED] | - | UG479 p. 40 | UG479 |
| Dual 24-bit adder mode | available, multiplier must be off [VERIFIED] | USE_SIMD = TWO24 | UG479 Table 2-3 p. 27 | UG479 |
| INT8 packing, DSP48E2 | shift 18, 2 spare bits, 7 accumulations, 1.75x [VERIFIED] | 27 x 18 multiplier, shared operand | WP486 p. 3 to 4 | WP486 |
| INT8 packing, minimum | one multiplier input >= 24 bits, accumulator >= 32 bits [VERIFIED] | two concurrent INT8 MACC | WP486 p. 3 | WP486 |
| INT8 packing, DSP48E1 | shift 16: 0 spare bits, 1 product; shift 17: 1 spare bit, 3 products, upper operand -127..127 [derived, re-computed] | [derived] from 25-bit port | UG479 p. 40 + WP486 p. 3 | UG479, WP486 |
| Block RAM aspect ratios | 32K x 1, 16K x 2, 8K x 4, 4K x 9, 2K x 18, 1K x 36; 512 x 72 simple dual port [VERIFIED] | RAMB36E1 | UG473 p. 11; Tables 1-13, 1-14 p. 30 | UG473 |
| 18 Kb aspect ratios | 16K x 1 to 1K x 18; 512 x 36 simple dual port [VERIFIED] | RAMB18E1 | UG473 Tables 1-11, 1-12 p. 29 | UG473 |
| Bytes per 36 Kb block | 4,096 (8-bit bytes) or 4,608 (ninth bits used) [derived, re-computed] | widths 9, 18, 36, 72 | [derived] UG473 Table 1-13; ninth bit p. 25 | UG473 |
| Blocks for N bytes | ceil(N / 4,096): 110 for 450,000; 113 for 460,800; with ninth bits ceil(N / 4,608): 98 and 100 [derived, re-computed] | same at 8, 16 and 32-bit data width | [derived] | UG473 |
| Byte write enables | 4 (36-bit port), 8 (72-bit simple dual port), 2 (18 Kb, 18-bit) [VERIFIED] | - | UG473 Table 1-7 p. 24 | UG473 |
| Different port widths | allowed per port and per direction [VERIFIED] | true dual port | UG473 p. 11, p. 21; mapping Table 1-16 p. 33 | UG473 |
| Read during write, same port | WRITE_FIRST, READ_FIRST, NO_CHANGE [VERIFIED] | per port attribute | UG473 p. 17 to 19 | UG473 |
| Write on one port, read of the same address on the other | old data if writer is READ_FIRST, otherwise undefined [VERIFIED] | common clock | UG473 Table 1-4 p. 20 | UG473 |
| Block RAM clock to output | 2.46 ns / 0.89 ns [VERIFIED] | grade -1, without / with output register | DS181 Table 30 p. 30 | DS181 |
| Block RAM address setup / hold | 0.57 / 0.36 ns [VERIFIED] | grade -1 | DS181 Table 30 p. 30 | DS181 |
| Block RAM Fmax | 388.20 MHz (339.67 READ_FIRST simple dual port with overlap; 345.78 cascade) [VERIFIED] | grade -1 | DS181 Table 30 p. 31 to 32 | DS181 |
| Built-in FIFO | 512 x 72 ... 8K x 4; 388.20 MHz; EMPTY deassert 3 or 4 read clocks [VERIFIED] | FIFO36E1, independent clocks | UG473 p. 47, Table 2-4 p. 54; DS181 Table 30 | UG473 |
| LUT as multiplexer | 4:1 in 1 LUT, 8:1 in 2, 16:1 in 4 [VERIFIED] | - | UG474 "Multiplexers" | UG474 |
| ROM in LUTs | 256 x 1 in 4 LUTs (64 bits per LUT) [VERIFIED] | asynchronous read | UG474 "Read Only Memory (ROM)" Table 1 | UG474 |
| Distributed RAM | RAM256X1S = 4 LUTs [VERIFIED] | SLICEM, asynchronous read | UG474 "Distributed RAM" Table 1 | UG474 |
| LUT delay / flip-flop clock to output | 0.13 / 0.53 ns [VERIFIED] | grade -1 | DS181 Table 27 p. 28 | DS181 |
| Distributed RAM clock to output | 1.32 ns [VERIFIED] | grade -1 | DS181 Table 28 p. 29 | DS181 |
| DDR2 device | MT47H64M16HR-25:H, 1 Gib, 16 bits, single rank [VERIFIED] | Nexys A7 | NexysRM section 3.1 | NexysRM |
| DDR2 clock period for MIG | 3,000 ps maximum rate (667 Mbps); 3,077 ps recommended (650 Mbps) [VERIFIED] | Digilent settings | NexysRM Table 3.1.1 | NexysRM |
| DDR2 PHY rate limit | 667 Mb/s (4:1), 620 Mb/s (2:1) [VERIFIED] | Artix-7 grade -1 | DS181 Table 16 p. 15 | DS181 |
| MIG user clock | memory clock / 4 = 81.25 MHz [VERIFIED] | 4:1, 3,077 ps | UG586 "Features" + [derived] | UG586 |
| MIG user data width | 128 bits (16 bytes) per read command [VERIFIED] | 16-bit DDR2, 4:1 | `Ram2Ddr.vhd` component ports | https://github.com/Digilent/Nexys-A7-100T-OOB |
| DDR2 peak data rate | 1.3 GB/s [derived, re-computed] | 650 Mb/s x 16 pins | [derived] | NexysRM |
| MIG DDR2 resources | 9,267 LUTs, 6,038 flip-flops, 2 BUFG, 1 PLL, 1 MMCM, 0 block RAM [VERIFIED] | 72-bit interface | DS176 "Resource Utilization" Table 1 | DS176 |
| MIG DDR2 resources, 16-bit | not found | - | - | - |
| MIG read latency in cycles | not found [VERIFIED: UG586 gives no number and says to simulate] | - | UG586 topics read give no number | UG586 |
| Lean 16-bit DDR3 controller | 1,279 LUTs, 920 flip-flops, 0 block RAM [VERIFIED] | XC7A100T-1, Xylon logiMEM | logiMEM data sheet v1.1 Table 1 | https://www.logicbricks.com/Documentation/Datasheets/IP/logiMEM_hds.pdf |
| Digilent adapter cycle | 210 ns read, 260 ns write (minimum cycle times) [VERIFIED in the Wayback copy of 2024-10-11]; 1,200 ns in the demo source [VERIFIED] | Ram2Ddr | Digilent "SRAM to DDR component", Bus Timing table; `AudioDemo.vhd` | https://web.archive.org/web/20241011091426/https://digilent.com/reference/learn/programmable-logic/tutorials/nexys-4-ddr-sram-to-ddr-component/start ; https://github.com/Digilent/Nexys-A7-100T-OOB |
| Flash | S25FL128S, 16 MiB; boards of PCB revision D or later that carry a sticker may hold an S25FL127S instead (live manual, Table 3.2.1) [corrected by the fact-check: the 2018 copy names only the S25FL128S]; bitstream 30,606,304 bits, "about 77%" of the flash free [VERIFIED] | Nexys A7 | NexysRM sections 2 and 3.2 | NexysRM |
| UART bridge | FT2232HQ; C4 = FPGA input, D4 = FPGA output, D3 CTS, E5 RTS [VERIFIED] | Nexys A7 | NexysRM section 6; master XDC | https://github.com/Digilent/digilent-xdc |
| UART rate range | 183 baud to 12 Mbaud; 7, 9, 10, 11 Mbaud not supported [VERIFIED] | FT2232H | FT2232H data sheet v2.6 sections 4.1, 4.2 | FT2232H |
| FT2232H buffers / latency timer | 4 kbytes each way / 16 ms default [VERIFIED] | per channel | FT2232H data sheet section 4.1 | FT2232H |
| PicoBlaze KCPSM6 | 26 slices, up to 240 MHz [VERIFIED] | smallest case, device dependent | AMD PicoBlaze page | https://www.amd.com/en/products/adaptive-socs-and-fpgas/intellectual-property/picoblaze.html |
| SERV | 125 LUTs, 164 flip-flops [VERIFIED] | Artix-7 | README table | https://github.com/olofk/serv |
| PicoRV32 | 761 / 917 / 2,019 LUTs; 442 / 583 / 1,085 registers [VERIFIED] | small / regular / large, Vivado 2017.3 | README "Utilization on Xilinx 7-Series FPGAs" | https://github.com/YosysHQ/picorv32 |
| VexRiscv | 504 LUTs, 505 flip-flops, 243 MHz; 1,418 / 949, 216 MHz [VERIFIED] | Artix 7; smallest RV32I; RV32IM no cache | README | https://github.com/SpinalHDL/VexRiscv |
| MicroBlaze 32-bit | 617 LUTs, 223 flip-flops, 235 MHz; 1,009 / 501, 212 MHz [VERIFIED] | XC7A200T-3; minimum area; microcontroller preset | UG984 "Device Utilization - Artix 7" Table 1 | https://docs.amd.com/go/en-US/ug984-vivado-microblaze-ref |
| 8 x 8 multiplier in LUTs | 71 LUTs [VERIFIED] | Virtex-7, Vivado default, `*` operator | Ullah et al. 2021, introduction | https://grk2767.tu-dresden.de/files/Images/people/chair-pd/Papers/ESL_acc_multiplier.pdf |
| UART transmitter with 16-byte FIFO | 15 LUTs + 8 SRL16E + 24 flip-flops [VERIFIED] | Xilinx `uart_tx6`, baud tick external; counted in source | source file | https://github.com/grg-haas/fpga_miner_rework/blob/master/src/vhdl/uart_tx6.vhd |
| LFSR taps | 16: 16,15,13,4; 31: 31,28; 32: 32,22,2,1 [VERIFIED] | XNOR feedback, maximum length | XAPP052 Table 3 | https://docs.amd.com/go/en-US/xapp052 |
| Multiplier as shifter | one-hot multiplier input 2^k shifts by k [VERIFIED] | principle; written for Virtex-II | XAPP195 p. 1 | https://docs.amd.com/go/en-US/xapp195 |
| Argmax unit, published LUT count | not found | - | - | - |

### 6.2 Small blocks: cost and basis

| Block | LUT | Flip-flops | Other | Basis |
|-------|-----|------------|-------|-------|
| UART transmitter, 8N1, 16-byte FIFO | 15 + 8 SRL16E | 24 | - | [doc, VERIFIED] counted in Xilinx `uart_tx6.vhd` |
| Baud tick counter | 7 to 10 | 7 to 10 | - | [estimate] 100 clocks per bit (1 Mbaud) = 7 bits; 868 (115,200) = 10 bits |
| UART transmitter without FIFO (Digilent `UART_TX_CTRL`, 9,600 baud) | about 25 to 35 | about 31 | - | [estimate] from the registers in the source: 14-bit timer, 10-bit shift register, bit index, state |
| LFSR, 16 or 32 bits, one bit per clock | 1 | 16 or 32 | - | XAPP052 taps; one 4-input XNOR |
| LFSR, 32 bits, one byte per clock | about 8 | 32 | - | [estimate] eight look-ahead XNOR terms |
| 24-bit arithmetic right shift by 0..23, saturate to int8, optional ReLU | about 60 to 100 | 0 to 8 | - | [estimate] 4:1 multiplexer per LUT: 24 bits x 2.5 levels = 60; overflow test 5 to 15; clamp 8 |
| Same function in a DSP48E1 (multiply by 2^(16-s), pattern detector for saturation) | about 20 to 30 | 0 | 1 DSP | [estimate]; shift range 0..16; principle XAPP195, UG479 p. 45 |
| 8-bit comparator with running maximum and 9-bit index | about 6 to 10 | 17 | - | [estimate] two compared bits per LUT on the carry chain |
| 24-bit comparator with running maximum and 10-bit index | about 15 to 20 | 34 | - | [estimate] |
| n-bit counter or auto-increment pointer with load | n to 2n | n | - | [estimate] one LUT per bit on the carry chain (UG474) |
| FSM additions | 1 to 3 per added state or control output | 1 per one-hot state | - | [estimate] |
| 256 x 8 table as LUT ROM | 32 | 0 | asynchronous read | [doc, VERIFIED] UG474: 256 x 1 = 4 LUTs |
| 256 x 8 table as distributed RAM (writable) | 32, in 8 SLICEMs | 0 | asynchronous read | [doc, VERIFIED] UG474: RAM256X1S = 4 LUTs |
| 256 x 8 table in block RAM | 0 | 0 | 1/8 of an 18 Kb block; synchronous read | [doc, VERIFIED] UG473 Table 1-11: 2K x 9 holds 8 tables |
| 8 x 8 multiplier in LUTs | 71 | - | - | [doc, VERIFIED] Ullah et al. 2021 |
| 24-bit adder or accumulator | 24 | 24 | - | [estimate] |
| 24-bit 2:1 / 4:1 multiplexer | 12 to 24 / 24 | - | - | [estimate] UG474 multiplexer costs |

### 6.3 Resource estimate per plausible LS CPU change

All rows [estimate]. "DSP" counts additional DSP48E1 slices, "BRAM" additional 36 Kb blocks. The 48-bit P register of
each lane is inside its DSP slice and costs no fabric flip-flops.

| Change | LUT | FF | DSP | BRAM | Reasoning |
|--------|-----|----|-----|------|-----------|
| 2 row-parallel lanes (2 rows per pass, shared X byte, W read 16 bits wide) | 30 to 55 | 5 to 15 | +1 | 0 | Second MACC slice; 24-bit 2:1 selector in front of the one requantiser (12 to 24 LUTs); second requantise/store step and lane-valid flag (15 to 30 LUTs). W memory keeps its block count (2K x 18 per block) |
| 4 row-parallel lanes | 50 to 100 | 10 to 30 | +3 | 0 | Three more slices; 24-bit 4:1 selector (24 LUTs, one per bit); control 20 to 40 LUTs. W read 36 bits wide, or two 18-bit ports of the same block RAM. Option: 4 requantisers and one 32-bit store with 4 byte enables: +180 to 300 LUTs |
| 2 in-row lanes (2 bytes of X and W per clock) | 45 to 80 | 25 to 50 | +1 | 0 | Second slice; one 24-bit adder for the lane sum (24 LUTs) or a cascade add inside the DSP column; tail mask for odd lengths (8 LUTs); control 10 to 20 LUTs; X memory read 16 bits wide on its second port |
| 4 in-row lanes | 120 to 200 | 50 to 100 | +3 | 0 | Three adders (72 LUTs) or DSP cascade (P = M + PCIN) with one more slice; tail masks 24 LUTs; control 20 to 40 LUTs |
| Wide read port on a memory made of many blocks (applies to both lane types) | 0 to about 34 per added data bit | 0 to 1 per bit | 0 | 0 | 0 if the memory is built in lane or bit slices; a 128:1 multiplexer costs about 34 LUTs per bit (8 x 16:1 at 4 LUTs + 8:1 at 2 LUTs); add an output register stage |
| Hardware argmax over the output layer | 15 to 30 | 34 to 44 | 0 | 0 | 24-bit compare 12 LUTs, enable and clear 3 to 6 LUTs, value register 24, index register 10; +10 LUTs and +10 flip-flops if no existing counter can supply the index |
| Fused clear / requantise / store per row | 30 to 60 | 20 to 30 | 0 | 0 | Clear and bias load through the OPMODE Z field (1 to 2 LUTs, no cycle); destination pointer 17 bits with auto-increment (17 LUTs, 17 flip-flops) if not present yet; FSM and decode 10 to 20 LUTs; requantiser unchanged |
| Whole matrix-vector instruction (row loop) | 60 to 120 | 50 to 80 | 0 | 0 | Row counter 10 bits; X base register 17 bits with reload multiplexer; length register 8 to 10 bits; 5 to 8 more states |
| Wider stream counter (16 bits) | 10 to 25 | 8 to 16 | 0 | 0 | 8 more counter bits and zero detect; path to load a second length byte. Above 511 terms the accumulator must also be read wider (the P register already has 48 bits; the requantiser's shifter and overflow test grow by about 2.5 LUTs per added accumulator bit) |
| Lookup-table stream | 35 to 60 | 25 to 40 | 0 | 0.5 for 8 tables | Destination pointer (shared with the fused store), table select 3 bits, pipeline valid bits, write-data multiplexer 8 LUTs, FSM 10 to 15 LUTs. Tables: one 18 Kb block, or 32 LUTs per table as LUT ROM |
| Multiply-requantise-store stream (element-wise) | 20 to 40 | 10 to 20 | 0 | 0 | Control only: requantise and store on every beat instead of at the end; uses the existing MACC, requantiser and the destination pointer |
| DDR2 weight stream: MIG | 1,300 to 9,300 (not known) | 900 to 6,000 (not known) | 0 | 0 | Vendor figure for 72 bits is the upper bound, a lean 16-bit DDR3 core the lower mark; uses 1 PLL, 1 MMCM, 2 BUFG and a 200 MHz reference clock |
| DDR2 weight stream: prefetcher | 100 to 200 | 100 to 150 | 0 | 2 (as FIFO36E1) | 27-bit address counter; command issue with credit counter; two 512 x 72 dual-clock FIFOs hold the 128-bit words (8 KB of look-ahead); read side 16:1 byte select (32 LUTs) or 4:1 word select for 4 lanes (32 LUTs); stall of the stream pipeline on empty (5 to 10 LUTs) |
| UART transmitter at 1 Mbaud with 16-byte FIFO | 30 to 35 | 31 | 0 | 0 | `uart_tx6` count plus a 7-bit tick counter |
| LFSR random source, 32 bits, one byte per clock | about 8 | 32 | 0 | 0 | Section 6.2 |

Sum of the changes in ideas 2 to 5 (4 row-parallel lanes), 8 and 9: about 220 to 450 LUTs (0.3 to 0.7% of the device),
3 DSP slices (1.3%), half a block RAM. [estimate; sums and percentages re-computed]

### 6.4 UART rates with a 100 MHz clock

Up to 3 Mbaud the FT2232H divides a 3 MHz reference by a divisor n plus a fraction in steps of one eighth; the
special divisors 0 and 1 give 3 and 2 Mbaud. For rates between 3 and 12 Mbaud it divides a 12 MHz reference
(120 MHz / 10) in the same way: the special divisors 0 and 1 give 12 and 8 Mbaud, the divisors 2 and 3 give 6 and
4 Mbaud, and driver version 2.4.20 or later is needed. So 1 Mbaud (3 MHz / 3), 2 Mbaud (special case) and 4 Mbaud
(12 MHz / 3) are exact on the FTDI side. The driver uses the closest divisor if the result is within +/- 3% of the
requested rate, and "as long as the actual Baud rate used is within +/- 3% of the required Baud rate, then the link
should function without errors". AN_120 v1.2, sections 3 and 4.4 [VERIFIED in a Wayback copy; corrected by the
fact-check: the earlier text listed 12 Mbaud as an integer divisor of 12 MHz and left out 8 Mbaud]
https://web.archive.org/web/20251026022456/https://ftdichip.com/wp-content/uploads/2020/08/AN_120_Aliasing_VCP_Baud_Rates.pdf .
FPGA side: the number of clocks per bit must be an integer at 100 MHz [derived; the table below was re-computed].

| Rate | Clocks per bit at 100 MHz | Error | Bytes per second | Time to send 128 MiB |
|------|---------------------------|-------|------------------|----------------------|
| 115,200 | 868 | +0.006% | 11,520 | 3.2 hours |
| 921,600 | 108 or 109 | +0.47% or -0.45% | 92,160 | 24 minutes |
| 1 Mbaud | 100 | exact | 100,000 | 22 minutes |
| 2 Mbaud | 50 | exact | 200,000 | 11 minutes |
| 3 Mbaud | 33 | +1.0% | 300,000 | 7.5 minutes |
| 4 Mbaud | 25 | exact | 400,000 | 5.6 minutes |
| 6 Mbaud | 17 | -2.0% | 600,000 | 3.7 minutes |
| 12 Mbaud | 8 | +4.2% (beyond the usual 2 to 3% margin; needs a 96 or 120 MHz clock) | 1,200,000 | 1.9 minutes |

Token output needs under 1,000 bytes per second, so 115,200 baud is enough for the text; a faster rate only matters for
logit dumps, for loading DDR2, and for not stalling a transmitter without FIFO.

### 6.5 Timing budget of the stream pipeline at speed grade -1 [derived from DS181 Tables 30 and 31]

Fact-check: the five DS181 inputs (2.46, 0.89, 3.29, 5.48 and 0.45 ns) are [VERIFIED]; the sums were re-computed.

| Path | Fixed delay | Left for routing at 100 MHz | Comment |
|------|-------------|-----------------------------|---------|
| Block RAM (no output register) -> DSP input -> M register | 2.46 + 3.29 = 5.75 ns | 4.25 ns | The three-stage pipeline of CONTEXT.md (address, multiply, accumulate); comfortable |
| Block RAM (no output register) -> DSP input -> P register, no MREG | 2.46 + 5.48 = 7.94 ns | 2.06 ns | Tight; avoid |
| Block RAM with output register -> DSP input -> P register, no MREG | 0.89 + 5.48 = 6.37 ns | 3.63 ns | Acceptable; one more cycle of read latency |
| Block RAM with output register -> AREG/BREG | 0.89 + 0.45 = 1.34 ns | 8.66 ns (3.66 ns at 200 MHz) | Needed for a 200 MHz stream datapath; block RAM limit 388 MHz, DSP limit 464 MHz |

### 6.6 What a sequential prefetch FIFO for a DDR2 weight stream needs

- Command side (MIG user clock, 81.25 MHz): 27-bit address counter stepping by one burst; issue a read (app_cmd = 001,
  app_en) only when the FIFO has room for every command still in flight (credit counter or the programmable
  almost-full flag set to the read latency plus margin; the latency must be measured).
- Storage and clock crossing: two FIFO36E1 in 512 x 72 mode side by side take the 128-bit words; the built-in FIFO
  cannot change width, so the 128 -> 8, 16 or 32 bit step is done in fabric on the read side.
- Read side (CPU clock): byte or word selector and a 4-bit position counter; pop every 16, 8 or 4 clocks for 1, 2 or
  4 lanes.
- Stall: the stream state of the CPU needs a clock enable that freezes the pipeline while the FIFO is empty.
- Seek (pointer set to a different address): reset the FIFOs (reset held for at least five clocks of the slower
  clock, DS181 Table 30 note 11 [VERIFIED]) and restart; costs one read latency. If weights are stored in the order of use, a
  token pass has one seek per embedding lookup and one per wrap.
- Supply against demand: 1.3 GB/s peak against 100 MB/s per lane; a full pass over 128 MiB takes 1.34 s at one lane.
- Simplifications to consider: run the CPU on the MIG user clock (81.25 MHz, costs 19% speed, removes the clock
  crossing), or a 2:1 controller at a 5 ns memory clock so that the user clock is 100 MHz and the user data bus
  64 bits wide (0.8 GB/s peak [derived]). UG586 documents the 2:1 ratio, DS181 allows 2:1 DDR2 up to 620 Mb/s at
  grade -1, and Digilent's page for the SRAM adapter lists a 2:1 configuration at 3,333 ps for this memory part
  [VERIFIED]. [UNVERIFIED: that the MIG wizard accepts a 5,000 ps period for this part was not checked in the tool.]

## 7. Sources

| Title | Authors | Year | URL | Licence (code, model, dataset) |
|-------|---------|------|-----|--------------------------------|
| 7 Series DSP48E1 Slice User Guide, UG479 v1.10 | Xilinx | 2018 | https://docs.amd.com/go/en-US/ug479_7Series_DSP48E1 | documentation, no code used |
| 7 Series FPGAs Memory Resources User Guide, UG473 v1.14 | Xilinx | 2019 | https://docs.amd.com/go/en-US/ug473_7Series_Memory_Resources | documentation |
| Artix 7 FPGAs Data Sheet: DC and AC Switching Characteristics, DS181 v1.27.1 | AMD | 2024 | https://docs.amd.com/go/en-US/ds181_Artix_7_Data_Sheet | documentation |
| 7 Series FPGAs Data Sheet: Overview, DS180 v2.6.1 | Xilinx | 2020 | https://docs.amd.com/go/en-US/ds180_7Series_Overview | documentation |
| Deep Learning with INT8 Optimization on Xilinx Devices, WP486 v1.0.1 | Y. Fu, E. Wu, A. Sirasao, S. Attia, K. Khan, R. Wittig | 2017 | https://docs.amd.com/go/en-US/wp486-deep-learning-int8 | documentation |
| 7 Series FPGAs Configurable Logic Block User Guide, UG474 v1.9 | AMD | web edition 2025 | https://docs.amd.com/go/en-US/ug474_7Series_CLB | documentation |
| Zynq 7000 SoC and 7 Series Devices Memory Interface Solutions User Guide, UG586 (MIG v4.2) | AMD | web edition 2024 | https://docs.amd.com/go/en-US/ug586_7Series_MIS | documentation |
| Memory Interface Solutions LogiCORE IP Data Sheet, DS176 (MIG v4.2) | AMD | web edition 2024 | https://docs.amd.com/r/en-US/ds176_7Series_MIS/Resource-Utilization | documentation |
| MicroBlaze Processor Reference Guide, UG984 | AMD | web edition 2026 | https://docs.amd.com/go/en-US/ug984-vivado-microblaze-ref | documentation |
| Implementing Barrel Shifters Using Multipliers, XAPP195 v1.1 | P. Gigliotti | 2004 | https://docs.amd.com/go/en-US/xapp195 | documentation |
| Efficient Shift Registers, LFSR Counters, and Long Pseudo-Random Sequence Generators, XAPP052 v1.1 | P. Alfke | 1996 | https://docs.amd.com/go/en-US/xapp052 | documentation |
| Nexys A7 Reference Manual | Digilent | copy dated 2018-12-25; live page as archived 2024-11-07 | https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual (mirror read: https://www.secs.oakland.edu/~llamocca/Courses/ECE2700/Boards/NexysA7_rm.pdf ; the fact-check also read https://web.archive.org/web/20241107031931/https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual) | documentation |
| digilent-xdc, `Nexys-A7-100T-Master.xdc` | Digilent | repository active 2026 | https://github.com/Digilent/digilent-xdc | MIT [VERIFIED] |
| Nexys-A7-100T-OOB (`Ram2Ddr.vhd`, `RamCntrl.vhd`, `AudioDemo.vhd`) | Digilent (M. Nagy) | 2014, repository 2020 | https://github.com/Digilent/Nexys-A7-100T-OOB | no licence file; file headers "COPYRIGHT (C) 2014, Digilent RO. All rights reserved"; read only, not used [VERIFIED] |
| Nexys-A7-100T-GPIO (`UART_TX_CTRL.vhd`, `top.vhd`, XDC) | Digilent (S. Bobrowicz) | 2011, repository 2020 | https://github.com/Digilent/Nexys-A7-100T-GPIO | no licence file; header "Copyright 2011 Digilent, Inc."; read only, not used [VERIFIED] |
| SRAM to DDR component (tutorial page) | Digilent | not known | https://digilent.com/reference/learn/programmable-logic/tutorials/nexys-4-ddr-sram-to-ddr-component/start (live page behind a bot challenge; read by the fact-check in the Wayback copy https://web.archive.org/web/20241011091426/https://digilent.com/reference/learn/programmable-logic/tutorials/nexys-4-ddr-sram-to-ddr-component/start) | documentation [VERIFIED] |
| FT2232H Dual High Speed USB to Multipurpose UART/FIFO IC Datasheet v2.6, FT_000061 | FTDI | version 2.6 | https://ftdichip.com/wp-content/uploads/2020/07/DS_FT2232H.pdf (mirror read: https://www.fdi.ucm.es/profesor/mendias/das/docs/DS_FT2232H-1621240.pdf ; the fact-check also read the Wayback copy of the FTDI URL, snapshot 2025-08-23) | documentation |
| PicoBlaze 8-bit Microcontroller (KCPSM6 Rev 9) product page | AMD | 2014 release | https://www.amd.com/en/products/adaptive-socs-and-fpgas/intellectual-property/picoblaze.html | reference design under AMD terms; not used |
| `uart_tx6.vhd` (PicoBlaze UART macro), copy in a third-party repository | K. Chapman, Xilinx | 2011 | https://github.com/grg-haas/fpga_miner_rework/blob/master/src/vhdl/uart_tx6.vhd | Xilinx proprietary header; repository has no licence; primitives counted only [VERIFIED] |
| PicoRV32 | YosysHQ (repository owner) | README, Vivado 2017.3 figures | https://github.com/YosysHQ/picorv32 | ISC [VERIFIED] |
| SERV | olofk (repository owner) | README | https://github.com/olofk/serv | ISC [VERIFIED] |
| VexRiscv | SpinalHDL | README | https://github.com/SpinalHDL/VexRiscv | MIT [VERIFIED] |
| logiMEM DDR3 SDRAM Memory Controller data sheet v1.1 | Xylon | 2022 | https://www.logicbricks.com/Documentation/Datasheets/IP/logiMEM_hds.pdf | commercial core; data sheet only |
| core_ddr3_controller (README) | ultraembedded | repository | https://github.com/ultraembedded/core_ddr3_controller | Apache-2.0 according to the source file headers (`src_v/ddr3_core.v`: "License: Apache 2.0"); the repository has no LICENSE file, so the GitHub API reports none [corrected by the fact-check]; README read only |
| Energy-Efficient Low-Latency Signed Multiplier for FPGA-Based Hardware Accelerators, IEEE Embedded Systems Letters 13(2), doi 10.1109/LES.2020.2995053 | S. Ullah, T. D. A. Nguyen, A. Kumar | 2021 | https://grk2767.tu-dresden.de/files/Images/people/chair-pd/Papers/ESL_acc_multiplier.pdf | paper |
| DSP-Packing: Squeezing Low-precision Arithmetic into FPGA DSP Blocks, arXiv:2203.11028 | J. Sommer, M. A. Özkan, O. Keszocze, J. Teich | 2022 | https://arxiv.org/abs/2203.11028 | paper |
| Double MAC: Doubling the Performance of Convolutional Neural Networks on Modern FPGAs, DATE 2017, pp. 890-893, doi 10.23919/DATE.2017.7927113 (added by the fact-check) | D. Nguyen, D. Kim, J. Lee | 2017 | https://past.date-conference.com/proceedings-archive/2017/pdf/0348.pdf | paper |
| Aliasing VCP Baud Rates, AN_120 v1.2, FT_000161 (added by the fact-check) | FTDI | 2016 | https://ftdichip.com/wp-content/uploads/2020/08/AN_120_Aliasing_VCP_Baud_Rates.pdf (read in the Wayback copy https://web.archive.org/web/20251026022456/https://ftdichip.com/wp-content/uploads/2020/08/AN_120_Aliasing_VCP_Baud_Rates.pdf) | documentation |
| vivado-boards, `new/board_files/nexys-a7-100t/D.0/1.3/mig.prj` (added by the fact-check) | Digilent | file last changed 2022 | https://github.com/Digilent/vivado-boards | MIT; read only |
| Nexys-A7-50T-OOB, `src/other/mig_a.prj` (added by the fact-check) | Digilent | file last changed 2018 | https://github.com/Digilent/Nexys-A7-50T-OOB | no licence file; read only |

Searches that returned nothing usable: MIG utilisation for a 16-bit DDR2 or DDR3 interface on Nexys or Arty boards;
MIG read latency in user-clock cycles; LUT counts of argmax units on 7-series. The search for INT8 packing on DSP48E1
had also returned nothing; the fact-check found Double MAC (DATE 2017, row above), so that entry was removed here.
No external code, model or dataset was used. `SOURCES.md` was not edited by this agent (outside its brief); the table
above has the fields for rows if the orchestrator wants them. The fact-check did not edit `SOURCES.md` either; it
added four rows to the table above.

## Verification log

Fact-check pass of 2026-11-13 by a second agent. Sources were opened again: AMD documents through the docs.amd.com
document API (the PDF, or the web edition where the PDF is above 20 MB or does not exist), GitHub files through the
GitHub API, papers as PDF, Digilent and FTDI pages through Wayback Machine copies because the live sites answer with
a bot challenge (HTTP 403). Nothing was synthesised, simulated or measured. Format of each line: claim; verdict; URL.

1. XC7A100T resources: 63,400 LUTs, 126,800 flip-flops, 15,850 slices (11,100 SLICEL, 4,750 SLICEM), 240 DSP48E1 in 3 columns of 80, 135 block RAMs of 36 Kb in 4 columns, 1,188 Kb distributed RAM, 6 clock management tiles; confirmed (DS180 v2.6.1 Table 4, UG474 v1.9 Table 1, UG479 Table 2-1, UG473 Table 1-2); https://docs.amd.com/go/en-US/ds180_7Series_Overview and https://docs.amd.com/go/en-US/ug474_7Series_CLB
2. Timing at speed grade -1: DSP48E1 Fmax 464.25 / 392.93 / 257.47 / 290.44 / 190.69 MHz, setup 3.29, 5.48, 0.37 and 0.45 ns, clock to P 0.44 and 2.31 ns; block RAM clock to output 2.46 and 0.89 ns, address setup and hold 0.57 and 0.36 ns, Fmax 388.20 / 339.67 / 345.78 MHz, FIFO reset for five clock edges; LUT 0.13 ns, flip-flop 0.53 ns, distributed RAM 1.32 ns; BUFG 464 MHz; confirmed (DS181 v1.27.1 Tables 27, 28, 30, 31, 32, page numbers as cited); https://docs.amd.com/go/en-US/ds181_Artix_7_Data_Sheet
3. Multiply-accumulate in one DSP48E1: OPMODE 0100101 (X = M, Y = M, Z = P), Z options 000 / 001 / 011, register attributes, "At least three pipeline registers ..." and "the M register should always be used" (both p. 14), guard bits 5 and 12 (p. 25), pre-adder without saturation (p. 40), USE_SIMD excludes the multiplier (p. 27), pattern detector (p. 45), quotes of p. 52 and p. 53; confirmed (UG479 v1.10); https://docs.amd.com/go/en-US/ug479_7Series_DSP48E1
4. WP486: shift by 18 bits on the DSP48E2, "accumulation of up to 7 product terms only can be guaranteed", 8 slices for 7 x 2 multiply-adds (1.75x), "At a minimum one of the inputs to the multiplier needs to be at least 24-bits and the carry accumulator needs to be 32-bits", shared operand; confirmed (WP486 v1.0.1 p. 1, 3, 4, 7); https://docs.amd.com/go/en-US/wp486-deep-learning-int8
5. DSP48E1 packing limits [derived]: shift 16 fits for all int8 values and holds 1 product per field, shift 17 needs the upper operand in -127..127 and holds 3 products (4 without the product 16,384); confirmed by re-computation, with the added note that carry handling in fabric lifts the limit; https://docs.amd.com/go/en-US/ug479_7Series_DSP48E1
6. "No source was found that implements WP486-style INT8 packing on a DSP48E1" (sections 3, 4 and 7); corrected: not found -> Double MAC (Nguyen, Kim, Lee, DATE 2017) packs two 8-bit multiply-accumulates with a common unsigned operand into one DSP block of a Virtex-7 485T at 0.5 DSP + 11 LUTs + 12 flip-flops per multiply-accumulate; https://past.date-conference.com/proceedings-archive/2017/pdf/0348.pdf
7. Citation Sommer, Özkan, Keszocze, Teich, "DSP-Packing: Squeezing Low-precision Arithmetic into FPGA DSP Blocks", arXiv:2203.11028, 2022: treats the DSP48E2 and only cites UG479 for the DSP48E1; confirmed (arXiv metadata and PDF; FPL 2022); https://arxiv.org/abs/2203.11028
8. Block RAM capacity: aspect ratios 32K x 1 to 1K x 36 and 512 x 72 (simple dual port), 4,096 bytes per 36 Kb block (4,608 with the ninth bits), 110 blocks for 450,000 bytes, 135 blocks = 552,960 bytes; confirmed (UG473 v1.14 p. 11, Tables 1-11 to 1-14; arithmetic re-computed); https://docs.amd.com/go/en-US/ug473_7Series_Memory_Resources
9. Block RAM port behaviour: ninth-bit quote (p. 25), parity only for 9/18/36-bit ports (p. 12, Table 1-1), byte-wide write enables 4 / 8 / 2 (Table 1-7), independent port widths (p. 11), address mapping (Table 1-16), collision table (Table 1-4: old data only if the writing port is READ_FIRST, otherwise X), WRITE_FIRST mapped to NO_CHANGE in simple dual-port mode (p. 9), FIFO widths and EMPTY latency 3 / 4 (p. 47, Table 2-4); confirmed; https://docs.amd.com/go/en-US/ug473_7Series_Memory_Resources
10. Board DDR2: Micron MT47H64M16HR-25:H, 16-bit, single rank; Digilent MIG settings 3,000 ps (667) and 3,077 ps (650), part MT47H64M16HR-25E, data width 16, internal Vref; PHY limit for DDR2 at grade -1: 667 Mb/s (4:1) and 620 Mb/s (2:1); confirmed (Nexys A7 manual section 3.1 in the 2018 mirror and in the Wayback copy of the live page, which writes MT/s; DS181 Table 16); https://web.archive.org/web/20241107031931/https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual and https://docs.amd.com/go/en-US/ds181_Artix_7_Data_Sheet
11. MIG user interface: "Controller and user interface operate at 1/4 the memory clock frequency", 128-bit user data for a 16-bit memory at 4:1, read data in request order, no read-latency figure in UG586; confirmed; added the UG586 note "Simulation should be run to analyze read latency" and the Digilent board preset (3,077 ps, 4:1); removed the unsupported clause "forum reports differ"; https://docs.amd.com/go/en-US/ug586_7Series_MIS and https://github.com/Digilent/vivado-boards/blob/master/new/board_files/nexys-a7-100t/D.0/1.3/mig.prj
12. MIG resources: DDR2 9,267 LUTs, 6,038 flip-flops, 2 BUFG, 1 PLLE2, 1 MMCM, 0 block RAM for 72 bits; DDR3 14,016 / 9,019 / 2 block RAMs; no 16-bit figure published; confirmed (DS176 Table 1 in the v4.1 PDF and in the v4.2 web edition; the LPDDR2 32-bit row 3,952 / 3,285 was added); https://docs.amd.com/r/en-US/ds176_7Series_MIS/Resource-Utilization
13. Xylon logiMEM, 16-bit DDR3 on XC7A100T-1: 1,279 LUTs, 920 flip-flops, 0 block RAM, 100 MHz AXI clock, 400 MHz DDR3 clock; confirmed (data sheet v1.1 of 2022-04-22, Table 1); https://www.logicbricks.com/Documentation/Datasheets/IP/logiMEM_hds.pdf
14. core_ddr3_controller: README quote "the same project using the Xilinx MIG DDR3 controller takes 33% of the FPGA LUTs (vs 9% with this core)" and licence "none reported"; quote confirmed, licence corrected: no licence reported -> Apache-2.0 in the source file headers (the repository has no LICENSE file); https://github.com/ultraembedded/core_ddr3_controller
15. Digilent adapter sources: `Ram2Ddr.vhd` has a 200 MHz clock input, 27-bit address, 16-bit data, 128-bit app_rd_data and 16-bit app_wdf_mask, and passes one 16-bit word per read; `AudioDemo.vhd` sets RW_CYCLE_NS := 1200; header "COPYRIGHT (C) 2014, Digilent RO. All rights reserved", no licence file; confirmed (the MIG configuration of the adapter was added from the 50T repository); https://github.com/Digilent/Nexys-A7-100T-OOB/tree/master/src/hdl
16. Adapter cycle times 210 ns read and 260 ns write, tagged [UNVERIFIED] by the research agent; confirmed in an archived copy of Digilent's page, tag changed to [VERIFIED]; https://web.archive.org/web/20241011091426/https://digilent.com/reference/learn/programmable-logic/tutorials/nexys-4-ddr-sram-to-ddr-component/start
17. FT2232H: "data transfer rate up to 12 Mbaud", "does not support the baud rates of 7 Mbaud 9 Mbaud, 10 Mbaud and 11 Mbaud", 120 MHz reference with x16 or x10 clock and 14-bit prescaler, "183 baud to 12 million baud", 4 kbyte buffers, latency timer 16 ms; confirmed (data sheet v2.6, sections 4.1 and 4.2, in the mirror and in the Wayback copy of the FTDI URL); https://web.archive.org/web/20250823184258/https://ftdichip.com/wp-content/uploads/2020/07/DS_FT2232H.pdf
18. Section 6.4, FT2232H divisors ("integer divisors give, for example, 12, 6, 4, 3, 2.4, 2, 1.5, 1.2 and 1 Mbaud", AN_120 not opened) and exact rates 1, 2 and 4 Mbaud; corrected: 12 Mbaud listed as an integer divisor and 8 Mbaud missing -> the special divisors 0 and 1 give 12 and 8 Mbaud on the 12 MHz reference (3 and 2 Mbaud on the 3 MHz reference); 1, 2 and 4 Mbaud exact confirmed; rate table re-computed (AN_120 v1.2, sections 3 and 4.4); https://web.archive.org/web/20251026022456/https://ftdichip.com/wp-content/uploads/2020/08/AN_120_Aliasing_VCP_Baud_Rates.pdf
19. UART pins: C4 = UART_TXD_IN (FPGA input), D4 = UART_RXD_OUT (FPGA output), D3 = CTS, E5 = RTS, names from the point of view of the PC; no baud limit in the manual; digilent-xdc licence MIT; confirmed (master XDC, GPIO demo `top.vhd` and its XDC, manual section 6); https://github.com/Digilent/digilent-xdc/blob/master/Nexys-A7-100T-Master.xdc
20. Published processor sizes: SERV 125 LUTs / 164 flip-flops (ISC); PicoRV32 761 / 917 / 2,019 LUTs and 442 / 583 / 1,085 registers, Vivado 2017.3 (ISC); VexRiscv 504 LUTs / 505 flip-flops at 243 MHz and 1,418 / 949 at 216 MHz (MIT); MicroBlaze 617 / 223 at 235 MHz and 1,009 / 501 at 212 MHz on XC7A200T-3; PicoBlaze "as small as 26 slices", "Up to 240 MHz"; confirmed, licences confirmed; https://github.com/olofk/serv , https://github.com/YosysHQ/picorv32 , https://github.com/SpinalHDL/VexRiscv , https://docs.amd.com/go/en-US/ug984-vivado-microblaze-ref , https://www.amd.com/en/products/adaptive-socs-and-fpgas/intellectual-property/picoblaze.html
21. Small-block references behind the estimates: 8 x 8 multiplier 71 LUTs (Ullah, Nguyen, Kumar, IEEE ESL 13(2), 2021, pp. 41-44); `uart_tx6` 15 LUTs + 8 SRL16E + 24 flip-flops (recounted); LFSR taps 16: 16,15,13,4, 31: 31,28, 32: 32,22,2,1 (XAPP052 Table 3); multiplier as shifter (XAPP195 p. 1); 4:1 multiplexer per LUT, 16:1 per slice, 256 x 1 ROM and RAM256X1S in 4 LUTs (UG474); confirmed; https://grk2767.tu-dresden.de/files/Images/people/chair-pd/Papers/ESL_acc_multiplier.pdf , https://github.com/grg-haas/fpga_miner_rework/blob/master/src/vhdl/uart_tx6.vhd , https://docs.amd.com/go/en-US/xapp052 , https://docs.amd.com/go/en-US/xapp195 , https://docs.amd.com/go/en-US/ug474_7Series_CLB
22. Section 1, cost-model summary (token = 42% multiply-adds, 34% per-row control, 24% element-wise; "lanes alone are capped at 1.73x"; about 6x for ideas 2 + 3 + 4 + 5); corrected: "lanes alone are capped at 1.73x" -> "in-row lanes alone are capped at 1.73x; row-parallel lanes reach x1.95 at 4 lanes on the same model" (the file's own idea 5); the shares, 423,168 multiply-adds, 14,072 rows and the 162k / 6.2x sum were re-computed and are consistent; sources are research/CONTEXT.md and research/04_tiny_hardware.md (no external URL)
23. Section 3, "The Digilent manual's feature list says 1,188 Kbits of fast block RAM"; corrected: true for the 2018 copy only -> the live page says "4,860 Kbits of fast block RAM"; https://web.archive.org/web/20241107031931/https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual
24. Flash "S25FL128S, 16 MiB", bitstream 30,606,304 bits, "about 77%" of the flash free; corrected: S25FL128S -> S25FL128S or, on PCB revision D or later with a sticker, S25FL127S (live manual, Table 3.2.1); the other two numbers confirmed; https://web.archive.org/web/20241107031931/https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual
25. Section 1, "Every hardware change under discussion is small against the device"; corrected (conclusion narrowed): every change -> every CPU-side change (220 to 450 LUTs, 0.3 to 0.7% of the device); the Tier B DDR2 controller has an unknown size and takes up to 9,267 LUTs = 14.6% of the LUTs by the vendor's 72-bit figure; https://docs.amd.com/r/en-US/ds176_7Series_MIS/Resource-Utilization

Totals: 25 claims checked, 18 confirmed, 7 corrected, none left unsupported; one unsupported clause was removed
("forum reports differ", MIG read latency). Still open after this pass: the MIG size for 16-bit DDR2, its read
latency and the real sequential throughput (each needs a Vivado run or a simulation); every [estimate] and
[cost model] figure (no synthesis, no cycle-accurate simulation); whether the host driver accepts 4 Mbaud; the cycle
time of the repository version of the Digilent adapter; the "not found in our search" statements of section 4, which
are negative search results and cannot be proven.
