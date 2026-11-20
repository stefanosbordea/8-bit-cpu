# SPEC: the machine and the arithmetic (binding for QAT, the integer reference, the CPU simulator and the exports)

Status: v0.6 (2026-11-14 01:30; v0.6 adds the shift `sh_e` of the embedding lookup, C.1). v0.5 (2026-11-13 23:55). New since v0.4: the LFSR is 32 bits (A.2, D-arch-4), an optional attention window
(C.6), the limit of 15 for shifts that round (Part C), and the clarifications that the three implementations asked
for (NRM, RQV, RECIP, the RELU table, Part E keys). Attention is final since v0.4 (C.6): unsigned 7-bit score codes, the exact row maximum,
a 7-bit exp table and division by a reciprocal table. Measured on the trained float model without any retraining:
+0.002 nats/token against float softmax (D-arch-3). Rejected by measurement: attention without normalisation
(does not train, D-arch-2) and a table referenced to a fixed code instead of the row maximum (+0.7 nats, D-arch-3).
The division needs two small instructions, `NRM` and `RQV` (A.2). Anything marked PROPOSAL is a change to the LS CPU
that the hardware team must accept; all of them are listed again in `docs/HARDWARE_CHANGES.md`.

## Part A. The LS machine at level L1 ("LLM-capable")

L0 is the course CPU of the design document (16 opcodes, four registers R0..R3, flags Z and N, 12-bit PC,
9-state FSM, every instruction 4 cycles, `MAC #n` = n + 4 cycles). L1 = L0 + the planned extensions + the minimum
additions needed to run a transformer. All headline numbers are for L1. Faster variants (L2) are separate.

### A.1 State
| Item | Width | Notes |
|---|---|---|
| R0..R3 | 8 | R3 is the DJNZ counter (a start value of 0 gives 256 iterations) |
| Z, N | 1 | set by ALU instructions and by LUT |
| PC | 12 | program ROM of up to 4,096 words of 16 bits |
| P0, P1, P2 | 20 | PROPOSAL: P0, P1 widened from 12 bits; P2 is new (store pointer) |
| LEN | 12 | PROPOSAL: stream length register for `MACL` |
| PROD | 16 | signed product |
| ACC | 24 | signed, wraps on overflow (the model is built so that it never overflows) |
| LFSR | 32 | PROPOSAL (planned): random bits |
| D | up to 2^19 bytes | PROPOSAL (planned + unified): one data memory in block RAM, two read ports |

Data memory D replaces the separate X and W memories. Port A is addressed by P0 (read register XR; also the
address of LUT reads and of P2 accesses). Port B is addressed by P1 (read register WR). A stream multiplies
D[P0] by D[P1]; because both ports see the same memory, either operand may be a weight or a run-time value.
Reads are synchronous, as in a block RAM.

### A.2 Instructions
Baseline instructions keep their encodings and meanings (design document, section 5): NOP, HALT, IN, OUT, LDI,
ADD, SUB, AND, OR, XOR, MOV, SHL, SHR, SAR, MVP, LD, ST, MUL, ADDP, LDA, MAC, MAC #n, RQ, SETP0, SETP1, JMP, BZ,
BNZ, DJNZ. Every instruction costs 4 cycles except streams.

Changes and additions (all PROPOSALS; encodings use only reserved bits and free function codes):

| Instruction | Encoding | Effect | Cycles |
|---|---|---|---|
| `SETP0 imm12`, `SETP1 imm12` | unchanged (op A, B) | Pp[11:0] <- imm12; Pp[19:12] unchanged | 4 |
| `MVP Pp, Ra, Rb` | op 3, fn = p (0, 1, 2) | Pp[11:0] <- Ra[3:0] & Rb; Pp[19:12] unchanged | 4 |
| `MVH Pp, Rb` | op 3, fn = 4 + p, r2 = Rb | Pp[19:12] <- Rb | 4 |
| `MVL Ra, Rb` | op 3, fn = 3 | LEN <- Ra[3:0] & Rb | 4 |
| `CPP Pd, Ps` | op 3, fn = 8 + 3*d + s (d, s in 0..2, d != s: fn 9, 10, 11, 13, 14, 15) | Pd <- Ps (20 bits) | 4 |
| `LD Rd, [Pp]{+}` | op 4, pointer id = {IR[7], IR[9]}: 00 P0, 01 P1, 10 P2; IR[8] = post-increment | Rd <- D[Pp] | 4 |
| `ST Rs, [Pp]{+}` | op 5, same pointer field, p in {P0, P2} | D[Pp] <- Rs | 4 |
| `LUT Rd, #t` | op 0, fn = 4, table id t = IR[9:4] (64 tables; Rd = IR[11:10]) | Rd <- D[LUT_BASE + 256 t + (Rd as unsigned byte)]; Z, N from the result | 4 |
| `TX Rs` | op 0, fn = 5 | put Rs into the UART transmit FIFO (waits only if the FIFO is full) | 4 |
| `RND Rd` | op 0, fn = 6 | LFSR advances 8 steps; Rd <- LFSR[31:24] | 4 |
| `RX Rd` | op 0, fn = 8 | PROPOSAL, optional (only for prompts typed over the UART; the model runs without it): Rd <- the next byte of the UART receive FIFO; the CPU waits while the FIFO is empty | 4 once a byte is there |
| `MACL` | op 8, IR[8] = 1 | stream of LEN multiply-adds (LEN >= 1) | LEN + 4 |
| `NRM Rd` | op 0, fn = 7 | Rd <- number of significant bits of ACC (0 for ACC = 0, 15 for 16384..32767). Defined for ACC >= 0; the program never uses it otherwise (the simulator returns 24 for a negative ACC) | 4 |
| `RQV Rd, Rs` | op 9, IR[8] = 1, Rs = IR[1:0], relu = IR[9] | as `RQ`, but the shift amount is Rs[4:0] (0..31; the program uses 0..15) | 4 |
| `MAC #n` | unchanged, n = 1..255 | stream of n multiply-adds | n + 4 |
| `MAC` | unchanged (n = 0) | one multiply-add, P0 and P1 advance | 4 |

`a & b` joins bit fields. Pointers increment as full 20-bit counters. `RQ Rd, sh, relu`: t = ACC >> sh (arithmetic,
rounds toward minus infinity); if relu and t < 0 then t = 0; Rd <- clamp(t, -128, 127); sh = 0..23.
`LDA Ra, Rb`: ACC <- 256 * Ra + (Rb as unsigned byte), i.e. the signed 16-bit value Ra:Rb.
LFSR step (32 bits, polynomial x^32 + x^22 + x^2 + x + 1, period 2^32 - 1): bit = s[0] xor s[10] xor s[30] xor s[31];
s <- (s >> 1) | (bit << 31); reset value 0xACE1ACE1 with the 16 switches xored into the lower half (never 0).
A 16-bit LFSR is too short: one sampled token at V = 512 uses 4,096 steps, so its noise would repeat, shifted by one bit, every 16 tokens.

There is no signed comparison and no branch on N. Comparisons of values in 0..127 use `SUB` followed by
`LUT` with the table RELU[v] = max(v, 0) and, where a decision is needed, `BZ`. The lookup tables start at
address 0 of D (LUT_BASE = 0).

### A.3 What L1 does not have (kept for the L2 list)
Fused row instructions, parallel MAC lanes, a hardware maximum or argmax, saturating 8-bit addition, a signed
compare, call and return, pointer arithmetic other than post-increment.

## Part B. Integer arithmetic

### B.1 Numbers
Every tensor is int8 with a power-of-two scale: real value = integer * 2^(-f), where f (fractional bits) is one
constant per tensor. Weights are int8 in [-127, 127]; activations are int8 in [-128, 127]. Non-negative activations
(attention weights, logits, hidden units after ReLU) are in [0, 127].

### B.2 Primitives (the only operations that exist)
- `acc = c + sum_k a[k] * b[k]`: a stream of length n <= 511 in one accumulation (one or two stream
  instructions). `c` is a preload: a signed 16-bit constant, or 256 * v + 128 for an int8 run-time value v.
- `rq(acc, sh, relu) = clamp(relu?(floor(acc / 2^sh)), -128, 127)`.
- `lut(t, v) = TABLE_t[v mod 256]`, tables hold int8 values.
- 8-bit ALU operations on values that provably do not overflow (only used for bookkeeping, comparisons of
  values in 0..127 and adding table noise to 7-bit logits).

Rounding: a requantisation with shift sh >= 1 uses the preload c = 2^(sh-1) (round half up), written
`rqr(acc, sh) = rq(acc + 2^(sh-1), sh)`. Where a loop has no register left for the constant the spec says `rq`
(floor) explicitly. The integer reference, the QAT forward pass and the CPU program implement exactly the stated one.

### B.3 Derived operations
- Linear: `y[j] = rqr(sum_k W[j][k] x[k], sh)`, sh = f_W + f_x - f_y >= 0. Optional ReLU in the same `RQ`.
- Residual linear (in place): `x[j] <- rq(256 * x[j] + 128 + sum_k W[j][k] h[k], 8)`. This needs
  f_W + f_h = f_x + 8. The 8-bit sum saturates correctly because it happens in the accumulator.
- Element-wise product: `y[k] = rqr(a[k] * b[k], sh)`.
- Table map: `y[k] = lut(t, x[k])`.
- Scaled dot with bias (attention scores, logits): `s = rq(c + sum_k a[k] b[k], sh, relu?)` with a 16-bit preload c
  that contains the rounding constant and, where stated, a bias.

### B.4 The accumulator never overflows
|a[k] * b[k]| <= 128 * 128 = 16,384, so for a stream of length n: |acc| <= |c| + 16,384 n. With |c| <= 32,896 and
n <= 509 this is at most 8,372,352 < 8,388,607. Every stream in the model has n <= 256, where the bound is
4,227,200, about half of the range. The export tool also computes, per tensor, the exact worst case
max_j (|c_j| + 128 * sum_k |W[j][k]|) and the largest value seen on real data, and writes them to the report.

### B.5 Power-of-two scales
One scale per weight tensor, chosen once from the trained float weights (smallest f such that the largest weight
magnitude fits in 127 after rounding, or a percentile if QAT prefers clipping) and then fixed. One scale per
activation tensor, chosen by calibration on the float model and then fixed. Per-row scales are not used: each
different shift would need its own `RQ` instruction, and QAT at 8 bits adapts the weights to a shared scale.

## Part C. Model graph (integer)

Shape: D = dim, L = layers, H = query heads, G = key/value heads (share = H / G), d = D / H (even), F = hidden,
V = vocabulary, T = context. Options: `norm` in {dyt, rms}, `ffn` in {swiglu, relu}.
Everything below is int8 unless it says "accumulator". `rq`, `rqr`, `lut` are the primitives of Part B.
Every shift `sh_*`, preload `c_*` and table is a constant fixed at export time (one per layer or per site as listed
in Part E). A model is valid only if every preload fits in a signed 16-bit value, the shifts `sh_s`, `sh_l` and `sa`
are in 0..23, and every shift of a rounded requantisation (`sh_q`, `sh_k`, `sh_v`, `sh_o`, `sh_1`, `sh_3`, `sh_g`, `sb`)
and `sh_e` is in 0..15, because its rounding constant 2^(sh - 1) is itself a 16-bit preload.

C.1 Embedding. `x[k] = rqr(E[tok][k], sh_e)` for k < D; with `sh_e = 0` this is a plain copy. The same int8 matrix E is
the output matrix in C.9, so E keeps the scale at which its largest weight fits int8 (f_E = f_x + sh_e) and the
lookup shifts a row down to the scale of the stream. Without the shift E had to use the scale of the stream, which
left it 4 to 5 bits and cost 0.03 nats per token on the final shape (D-arch-5). On the CPU a row is shifted through a
lookup table that holds rqr(v, sh_e) for every byte v (4 D cycles per token more than a plain copy).

C.2 Norm at site n (two per layer plus the final one, n = 0 .. 2L).
- `dyt`: `y[k] = lut(NORM_n, x[k])`. The table holds `clamp(round(2^f_n * tanh(alpha_n * v * 2^-f_x)), -127, 127)`.
- `rms`: `S = sum_k x[k] * x[k]` (accumulator, preload 0); `m = rq(S, sa_n)` (0..127); `r = lut(NORM_n, m)`;
  `y[k] = rqr(x[k] * r, sb_n)`. The table holds `clamp(round(2^(sb_n + f_n) * sqrt(D / ((m + 0.5) * 2^sa_n))), 1, 127)`.

C.3 Projections. `q = rqr(Wq y, sh_q)` (D values), `k = rqr(Wk y, sh_k)` (G d values), `v = rqr(Wv y, sh_v)` (G d values).

C.4 RoPE at position p, for every head of q and of k, for every pair j < d / 2 with (a, b) = (head[2j], head[2j+1]):
`a' = rqr(a * ROPE_C[p][j] - b * ROPE_S[p][j], 7)`, `b' = rqr(a * ROPE_S[p][j] + b * ROPE_C[p][j], 7)`, with
`ROPE_C = round(127 cos(p * theta_j))`, `ROPE_S = round(127 sin(p * theta_j))`, `theta_j = 10000^(-2j/d)`.
The rotated head is stored as `[a'_0 .. a'_(d/2-1), b'_0 .. b'_(d/2-1)]` (same layout for q and k, so their dot
product is unchanged).

C.5 KV cache at position p. `K[l][g][p][0..d-1]` = rotated key of key head g. `V[l][g][j][p]` = `v[g d + j]`
(one row of T bytes per value component, so that value mixing is a single stream over positions).

C.6 Attention of query head h at position p, with g = h // share and keys t = 0 .. p (with an attention window W,
config field `window`, 0 = none: keys t = max(0, p + 1 - W) .. p; the CPU then keeps the KV cache as a ring of W
positions, position p in slot p mod W, and the cost of a token stops growing at context W):
1. Score codes: `code[t] = rq(c_s[l][h] + sum_j q'[h][j] * K[l][g][t][j], sh_s[l], relu)`, so 0..127. The 16-bit
   preload `c_s[l][h]` holds the rounding constant and a calibrated offset per head that puts the typical row maximum
   near code 80. The resolution is kappa_l codes per nat (about 4; any real number, it is absorbed by the table).
2. Row maximum: `m = max_t code[t]` (on the CPU without a branch: `m <- m + RELU[code - m]` with the table
   RELU[v] = max(v, 0); the codes are 0..127, so differences fit int8).
3. Weights: `e[t] = lut(EXP_l, (code[t] - m) mod 256)` with `EXP_l[0] = 127`, `EXP_l[256 - d] = round(127 * exp(-d / kappa_l))`
   for d = 1..127, and 0 elsewhere. The largest code gets weight 127; weights below 1/254 of it are 0.
4. Sums: `Dsum = sum_t e[t]` (127 .. 32,512; a stream of e against a row of ones) and
   `A[j] = sum_t e[t] * V[l][g][j][t]` (accumulator, preload 0).
5. Division by a reciprocal table: `k = bitlen(Dsum)` (number of significant bits, 7..15); `mant = Dsum >> (k - 7)`
   (64..127); `r = lut(RECIP, mant)` with `RECIP[v] = min(127, round(8192 / v))` for v = 64..127 (the other entries are never read
   and are exported as 127); `a = rq(A[j], k)` (-128..127, no
   saturation possible because |A[j]| <= 128 Dsum); `o[h d + j] = rqr(a * r, sh_o[l])`, sh_o = 6 + f_v - f_o.
   On the CPU: `NRM` gives k, `RQV` does the two shifts by k - 7 and k.

C.7 Residual update. `x[i] <- rq(256 * x[i] + 128 + sum_k Wo[i][k] * o[k], 8)` for i < D.

C.8 Feed-forward. `y = norm(x)` at the second site of the layer, then
- `swiglu`: `h1 = rqr(W1 y, sh_1)`, `g1 = lut(SILU, h1)`, `h3 = rqr(W3 y, sh_3)`, `g[k] = rqr(g1[k] * h3[k], sh_g)` for k < F.
  `SILU[v] = clamp(round(2^f_g1 * silu(v * 2^-f_h1)), -127, 127)`.
- `relu`: `g[k] = rq(2^(sh_1 - 1) + sum W1[k] y, sh_1, relu)` (0..127).
Then `x[i] <- rq(256 * x[i] + 128 + sum_k W2[i][k] * g[k], 8)`.

C.9 Output. `y = norm(x)` at the final site; `u[j] = rq(c_l + sum_k E[j][k] * y[k], sh_l, relu)` for j < V, so
logits are unsigned 7-bit values 0..127 (`c_l` = offset + rounding). Greedy decoding: the smallest j with the largest u[j].

## Part D. Decoding
- Greedy: argmax over u (C.9). On the CPU: `SUB`, the RELU table, `BZ` (values are 0..127, so the difference fits int8).
- Sampling (Gumbel-max): `cand[j] = (u[j] >> 1) + lut(GUMBEL, RND)`, then the same argmax. `GUMBEL[r] =
  clamp(round(g0 + tau * 2^(f_l - 1) * (-ln(-ln((r + 0.5) / 256)))), 0, 63)` for the chosen temperature tau, with
  `g0 = round(tau * 2^(f_l - 1) * ln(ln 512))`, which makes the smallest entry 0. The table depends on tau and is not
  part of the exported model; `lsllm.intref.model.gumbel_table` makes it.
- Syntax filter (`docs/FILTER.md`): a class byte per token and one row of ALLOW per abstract state; the row of the
  current state is copied to a `LUT` table and the argmax skips the tokens whose class is forbidden (a row that
  allows no token of the vocabulary allows all). `RND` is drawn for every token id in id order, also for skipped ids.
- UART: token ids, two bytes per token: `0x80 | (id >> 7)` then `id & 0x7F`; 115200 baud, 8N1. The token that ends
  the text (end of text, or a line break back to level 0) is sent too, then the program halts.
- Prompt: with a prompt table in the image the program reads the switches once after reset (`IN`) and copies prompt
  number SW5..0 into its prompt buffer (a number beyond the table gives prompt 0). With `RX` it reads the prompt
  from the UART after reset and after every answer instead: token ids in the framing of the output, ended by the
  byte `0xFF`; every answer then ends with the byte `0xFF`, and the program never halts.

## Part E. Exported integer model (the contract between QAT, the integer reference, the CPU program and the exports)
A directory with `model.json` and `model.npz`.
- `model.json`: `{"config": ModelConfig fields, "options": {"norm", "ffn"}, "frac": {tensor name: fractional
  bits; `frac["s"][l]` = log2 of the score codes per nat of layer l and `frac["u"]` = fractional bits of the logits are
  used by the tables and by sampling}, "shifts": {name: int or list}, "preloads": {name: int or list}, "tokenizer": path,
  optional "tables": {alpha, gain} and "notes": calibration warnings}`.
  Shift names: `sh_e` (optional, 0 if absent), `sh_q[l]`, `sh_k[l]`, `sh_v[l]`, `sh_s[l]`, `sh_o[l]`, `sh_1[l]`, `sh_3[l]`, `sh_g[l]`, `sh_l`, and for
  `rms` norms `sa[n]`, `sb[n]`. Preload names: `c_s[l][h]`, `c_l`.
- `model.npz` (all int8): `E` (V, D); per layer l: `Wq_l` (D, D), `Wk_l` (G d, D), `Wv_l` (G d, D), `Wo_l` (D, D),
  `W1_l` (F, D), `W3_l` (F, D; swiglu only), `W2_l` (D, F); tables `NORM` (2L + 1, 256), `SILU` (256), `EXP` (L, 256), `RECIP` (256),
  `ROPE_C` (T, d / 2), `ROPE_S` (T, d / 2). Tables are indexed by the unsigned byte of the int8 value
  (index 0..127 = values 0..127, index 128..255 = values -128..-1).
The integer reference (`lsllm/intref`) loads this directory and computes `u` for one token at a time, keeping the
KV cache; QAT writes it; the CPU program generator reads it and lays it out in data memory.
