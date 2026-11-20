# Research context (shared by all research agents)

## What we are building
A language model that writes Python, small enough to be run by "LS", a student-designed 8-bit CPU on an FPGA
(Digilent Nexys A7-100T, Xilinx Artix-7 XC7A100T, 100 MHz). Two equal targets: (1) benchmark score on Python coding
benchmarks, (2) tokens per second on the LS CPU. Everything is trained on a MacBook M1 (8 GB RAM, MPS) with PyTorch,
then quantized to int8 and exported as memory images. Results must be honest and defensible number by number.

## The LS CPU
- 8-bit data, 16-bit instructions, 16 opcodes, four 8-bit registers, multi-cycle FSM: every ordinary instruction
  takes 4 clock cycles.
- MAC unit: signed 8x8 multiplier (one DSP48E1) -> 16-bit product -> signed 24-bit accumulator ACC.
- Two auto-incrementing pointers P0 (input memory X) and P1 (weight memory W), synchronous (registered) reads.
- Classic `MAC`: one multiply-add in 4 cycles. Streamed `MAC #n` (the core idea): n multiply-adds in n + 4 cycles
  (fetch/decode once, then address, multiply, accumulate overlap as a 3-stage pipeline). n <= 255 per instruction.
- `RQ` requantiser: ACC arithmetic-shift-right by a constant, optional ReLU, saturate to int8.
- Planned extensions: X and W in block RAM with wider pointers, a store instruction `ST`, a 256-entry lookup-table
  instruction for nonlinearities, a UART transmitter, an LFSR random number generator. More hardware changes may be
  proposed, but they are listed separately and not assumed in the main numbers.

## Hard constraints on the model
- Operands are signed int8 only. Products are int16, sums live in a signed 24-bit accumulator
  (-8,388,608 .. 8,388,607), which must never overflow.
- Requantisation = arithmetic shift right by a constant, then clamp to int8 (optional ReLU). All scales are powers
  of two (per tensor or per output row). No floating point, no integer division at inference.
- Nonlinearities only as 256-entry lookup tables indexed by an int8 value (exp for softmax, SiLU for SwiGLU,
  1/sqrt for RMSNorm, or anything else that fits a table). RoPE sin/cos are precomputed tables.
- Tier A (main): weights + KV cache + lookup tables in about 450 KB of on-chip block RAM (so roughly 250k-400k
  int8 parameters). Tier B: weights in 128 MiB DDR2, speed limited by one multiply-add per clock (100 M MAC/s)
  and DDR2 bandwidth (about 1.3 GB/s peak).
- Baseline shape: Karpathy's llama2.c `stories260K` (Llama-2 style: 5 layers, dim 64, 8 heads, 4 KV heads,
  SwiGLU hidden 172, RoPE, RMSNorm, tied embeddings, vocab 512). Vocabulary will be 256 or 512 SentencePiece tokens.

## What costs time on this CPU (first estimate, to be refined)
- A matrix row of length n costs about n + 24 cycles (n multiply-adds + about six 4-cycle control instructions:
  set pointer, clear/load ACC, stream, requantise, store, loop). Short rows are dominated by overhead.
- Per token for the baseline at context length T: about 425k multiply-adds at T = 256, about 1.0 M cycles in total
  (about 100 tokens/s). More than half of the cycles are control overhead, not multiplies.
- Attention grows with T: scores (heads x T short dot products), exp lookups (heads x T), value mixing (T-long streams).
  At T = 256 attention is about 60% of a layer. The output layer over the vocabulary is about 5-9% of a token.
- Anything element-wise that is not a dot product (norm scaling, gating, lookups, residual adds) costs 12-32 cycles
  per element, so element-wise operations are expensive compared with their float cost.

## What we need from research
For each topic: what is proven (numbers + sources), what is uncertain, what has not been tried at our scale, and a
ranked list of ideas we could try with expected gain, cost and risk. We will train float baselines, search shapes on
the score-versus-speed trade-off, do quantization-aware training that matches the hardware bit for bit, add
grammar-constrained decoding implementable as a small table/state machine, and finally edit int8 weights directly
(no gradients) to fix weaknesses. Benchmarks: HumanEval, MBPP, plus an easier ladder (syntax validity, runs without
error, 200 very simple custom tasks).
