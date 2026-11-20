# Design notes (orchestrator's working notes; the binding version becomes docs/SPEC.md)

Written 2026-11-13 before research results. Purpose: keep the reasoning so far.

## Three hardware levels
- L0: the course CPU as designed (16 opcodes, 256 x 8 X/W, 12-bit pointers).
- L1: "LLM-capable LS" = L0 + planned extensions + the minimum additions without which the model cannot run.
  All main numbers use L1. Every ordinary instruction still costs 4 cycles, MAC #n costs n + 4.
- L2: optional speed-up changes, listed separately with estimated gains and FPGA cost. Never in the main numbers.

## L1 machine (draft)
- One byte-addressed data memory in block RAM (up to 512 KB), true dual port: port A read by P0 (register XR),
  port B read by P1 (register WR). Both MAC operands may be runtime data (needed for attention: q x K, w x V).
- Pointers P0, P1 widened to 20 bits; new write pointer P2 (ST Rs,[P2]+), so outputs are stored without
  disturbing the two stream pointers. SETP0/SETP1 imm12 set the low 12 bits; MVH sets bits 19..12 from a register;
  CPP copies a pointer; MVP also works for P2.
- LUT Rd,#t: Rd <- table t [Rd], 16 tables of 256 bytes in a fixed region of the data memory. 4 cycles.
- MAC with the count taken from a register (attention streams have a length that depends on the position).
- TX Rs (UART with a small FIFO, non-blocking), RND Rd (32-bit LFSR, 8 bits per call).
- Program ROM up to 4,096 words (12-bit PC). Layers are unrolled in the program so all addresses are constants.
- No signed-compare branch in L1 (BN does not exist). Running max / argmax uses the requantiser:
  d = RQ(256 * (cand - best), 8, relu) is an exact saturating max(0, cand - best).

## Row loop at L1 (the unit of cost)
`SETP0 x ; LDA R1,R2 ; MAC #n ; RQ R0,sh ; ST R0,[P2]+ ; DJNZ row` = n + 24 cycles per matrix row.
LDA preloads the accumulator with a 16-bit constant: rounding-to-nearest (2^(sh-1)) is free.
Residual rows: `LD R1,[P2]` then `LDA R1,R2` puts 256 * x_j + 128 in ACC, the stream adds W h, `RQ R0,8` gives the
new x_j with exact saturation, stored in place: n + 28 cycles. This fixes the shift of the residual-writing matrices
(Wo, W2) to 8, i.e. f_W = f_x + 8 - f_h. No saturating 8-bit add is needed anywhere.

## Where the cycles go (stories260K shape, L1, first estimate)
- T = 256: about 425k multiply-adds but about 1.0 M cycles per token: more than half is control overhead.
- Attention per layer at T = 256 with 8 heads: scores 8 x 256 x 36 = 74k, value mix 64 x 284 = 18k.
  Exact softmax needs the row maximum: about 22 cycles per score with instructions, plus a saturating subtract
  (about 20) and a second pass: 90 cycles per score instead of 36-40. That alone would make a token about 1.6x slower.
- Fewer, wider heads cut the number of scores (2 heads x 32 dims instead of 8 x 8): 4x fewer score rows.
- Element-wise work costs 16-36 cycles per element (LUT map 16, scale or gate 20-36).

## Candidate architecture choices to settle by float experiments, then QAT
- Normalisation: RMSNorm (sum of squares stream + 1/sqrt table with two ranges + per-element scaling, about 1.9k
  cycles per norm) versus Dynamic Tanh (one table lookup per element, about 1.0k cycles, exact in int8).
- Attention weights: exact softmax (expensive at L1), "cap-referenced" softmax (exp table referenced to the top of
  the int8 range, applied inside the score loop, then exact normalisation by a reciprocal table once per head,
  about 300 cycles per head), or sigmoid attention (no normalisation at all).
- Feed-forward: SwiGLU (table + per-element product, about 4k cycles extra per layer) versus ReLU (free inside RQ).
- Positions: RoPE (two-term streams per pair, about 2.4k cycles per layer, table per position) is the default;
  ALiBi folded into the key (2 extra multiply-adds per score) or learned absolute positions are fallbacks.
- Heads: 2, 4 or 8 query heads; KV heads 1, 2, 4.
- Context: full 256, sliding window, or "fixed prefix + window" (keep the function header visible).

## Integer rules (draft)
- Power-of-two scale per tensor (not per row: per-row shifts need per-row code; QAT at 8 bits does not need them).
- Residual stream int8 with one global scale; a 16-bit residual was considered and rejected for now
  (the requantiser cannot emit a low byte, and every norm and residual row would cost more).
- Any dot product of length n <= 511 cannot overflow the 24-bit accumulator: 511 x 128 x 128 = 8,372,224 < 8,388,607.
- QAT forward is exact integer arithmetic carried in float32 (all values below 2^24 are exact), so the PyTorch
  model, the numpy reference and the CPU simulator can be compared bit for bit.
- Sampling: Gumbel-max with a 256-entry Gumbel table indexed by LFSR bits, added to the int8 logits, then the same
  argmax as greedy decoding. No division, no cumulative sums.
- UART: token ids, not text; the laptop script decodes. A 2-byte self-synchronising frame for 512-token vocabularies.

## L2 candidates (to be costed with the same program generator)
Fused row instruction (rewind, clear, stream, requantise, store, loop inside the FSM: about n + 3 per row), K parallel
MAC lanes fed by a wide block RAM port, running max/argmax register on RQ results, stream-LUT, wider stream counter,
signed-compare branch, 3-cycle ordinary instructions, DDR2 weight stream with a prefetch FIFO (Tier B).
