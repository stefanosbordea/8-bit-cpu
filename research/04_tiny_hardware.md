# 04. Transformers on microcontrollers, retro and 8-bit machines, and FPGAs

Research date: 2026-11-13. Every number below was read at the given URL during this session, unless it carries an
`[UNVERIFIED: reason]` tag. "Our arithmetic" marks numbers we computed from verified inputs. Speeds are what each
author reported; nobody re-ran them, and models, context lengths and timing scopes differ between rows.

Fact-check pass (2026-11-13, second agent): `[VERIFIED]` marks a claim that was opened again at its source and
matched; `[UNVERIFIED: reason]` marks a claim without support; a claim with no tag was read once by the research
agent and not opened again. Corrections and additions from that pass are listed in the Verification log at the end.

## 1. Answer in one paragraph

Running Karpathy's 260K-parameter `stories260K` on very small hardware is well documented, but almost always in
float32 (hardware or software floating point): 19 to 33 tokens/s on an ESP32-S3, about 87 tokens/s (peak) on an
STM32H7, 39.31 tokens/s on a Pentium II, 2.08 tokens/s on a 486DX2-66, and about one token every 8 minutes on a
Commodore 64 [VERIFIED].
The Commodore 64 port (llama2.c64) has **no fixed-point scheme**: it keeps the float32 weights unchanged in a 2 MB RAM
expansion unit and uses the compiler's software floating point; its README answers the question about a quantized
model by inviting a pull request for an int8 version [VERIFIED].
The integer-only runs we found on 8-bit CPUs and small FPGAs all date from 2026 (on an MCU-class chip, Deeploy ran
int8 matrix products with an integer softmax approximation in 2024): a Game Boy Color port of stories260K (int8
weights, Q12 activations in 16/32 bits, 0.0059 tokens/s computed from an elapsed time its README calls estimated,
context 16), small int8 transformers on a stock Commodore 64 that the research pass had missed (Soul Player C64,
about 25K parameters, about 60 s/token; 8bitGPT, about 18K parameters, about 30 s/character), an NES port of a
different ternary model with 4-bit activations trained with quantization-aware training (0.63 s/token, measured in
emulators), and FPGA designs that run stories260K as plain RTL (a Virtex UltraScale+ with 235,704 LUTs and no
published speed; a Lattice ECP5-45 with int8 values, power-of-two shifts and 256-entry tables at 14 tokens/s) or a
4.72M-parameter model (4.20M ternary weights, 0.52M int8) on a six-instruction vector processor in an
8,064-logic-element MAX 10 (45 to 56 tokens/s at 60 MHz, int8 activations, 24-bit accumulators) [VERIFIED]. A
transformer language model on a custom or homebrew 8-bit CPU with a multiply-accumulate instruction was **not found
in our search** (queries in Table 6.5; a second, independent search in the fact-check pass did not find one either);
the nearest things are those programmable FPGA engines and the 8-bit retro CPUs that have no multiplier. The LS-LLM
estimate of about 100 tokens/s for a 260K model would sit above every stories260K figure that Table 6.1 lists for
microcontrollers, handheld consoles and PCs up to the Pentium III, except the 555 MHz PS Vita (about 120 tokens/s);
the DOS source also reports 172 to 386 tokens/s on laptops from 2004 to 2020, which Table 6.1 leaves out. The
estimate therefore has to be backed by cycle-exact counts. The two
fastest small programmable designs (the MAX 10 vector engine and the gateGPT sequencer) do things the LS CPU does
not do yet: they loop over matrix rows in hardware, they read several weights per clock, and the MAX 10 engine runs
element-wise steps at one element per clock.

## 2. Proven

**stories260K and llama2.c (the baseline shape)**
- Configuration read from the 28-byte file header: dim 64, hidden_dim 172, n_layers 5, n_heads 8, n_kv_heads 4,
  vocab_size 512, max_seq_len 512; the positive vocab_size means the classifier shares the embedding table.
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin (bytes 0..27, HTTP range request) [VERIFIED]
- Parameter count 260,032 = 32,768 (embedding 512 x 64, tied with the output layer) + 5 x 45,440 + 64 (final norm gains).
  Our arithmetic; it reproduces the file size exactly: 28 + 4 x (260,032 + 4,096 RoPE table floats) = 1,056,540 bytes.
  Of the 260,032 parameters, 259,328 take part in matrix products and 704 are RMSNorm gains.
  File sizes: https://huggingface.co/api/models/karpathy/tinyllamas/tree/main/stories260K [VERIFIED]
- Validation loss 1.2968 (model card) / 1.297 (README table); trained with `--vocab_source=custom --vocab_size=512
  --dim=64 --n_layers=5 --n_heads=8 --n_kv_heads=4 --multiple_of=4 --max_seq_len=512 --batch_size=128
  --learning_rate=1e-3 --dropout=0.05 --weight_decay=0.01 --max_iters=100000 --beta2=0.99 --warmup_iters=1000`;
  the card says "~10 minutes (?)" on an A100. Full command in Table 6.2.
  https://huggingface.co/karpathy/tinyllamas/blob/main/stories260K/readme.md and
  https://github.com/karpathy/llama2.c#models [VERIFIED]
- Licence MIT for both the code and the checkpoints (GitHub LICENSE file; Hugging Face card metadata `license: mit`).
  https://github.com/karpathy/llama2.c/blob/master/LICENSE and https://huggingface.co/api/models/karpathy/tinyllamas [VERIFIED]
- `runq.c` is **not** integer-only. Only the weights of matrix products are int8 (Q8_0: symmetric, range -127..127,
  one float32 scale per group of 64 weights). RMSNorm gains, the residual stream, the KV cache, RoPE (`cosf`, `sinf`,
  `powf`), softmax (`expf`), SiLU, and the logits stay float32; activations are re-quantized to int8 at run time
  before every matrix product (max-abs per group, float division, rounding) and each group sum is turned back into
  float. Details in Table 6.3. https://github.com/karpathy/llama2.c/blob/master/runq.c and
  https://github.com/karpathy/llama2.c/blob/master/export.py [VERIFIED]
- llama2.c README, Llama-2 7B on 64 threads: float32 4.6 tok/s, int8 (`runq`) 14 tok/s, checkpoint 4x smaller.
  https://github.com/karpathy/llama2.c#int8-quantization [VERIFIED]

**8-bit CPUs**
- Commodore 64 (llama2.c64): needs a RAM expansion unit of at least 2 MB; `weights.reu` holds "model weights
  (unchanged float32)"; "one output token approximately every 8 minutes"; the opening word appears after about
  16 minutes. The author's log gives 62 tokens in 9:25:43, which is 547 s per token (our arithmetic), and 37 to 40 s
  per token in the VICE emulator's warp mode. Sine, cosine and exp use polynomials copied from the C64 BASIC ROM.
  A branch that multiplied floats through a 64 KB 8x8-bit times table was 2x slower. Output equals llama2.c at
  temperature 0. https://github.com/ytmytm/llama2.c64 and https://github.com/ytmytm/llama2.c64/blob/main/STORY.txt [VERIFIED]
- Game Boy Color (gbc-transformer), real hardware: 16 forward passes in about 45 minutes (the README labels this
  "Estimated elapsed time"), "about 0.0059 tokens/sec, or one token every 2 minutes 49 seconds"; context capped at
  16 tokens; greedy only. Scheme (read in the source):
  row-wise int8 weights with one Q16 scale per row, activations in Q12 held in int32 and clamped to int16 for the
  product, int32 accumulators, bitwise integer square root, integer division, exp as 1/(1 + x + x^2/2 + x^3/6 + x^4/24).
  https://github.com/maddiedreese/gbc-transformer and
  https://github.com/maddiedreese/gbc-transformer/blob/main/src/gbc_q8_runtime.c [VERIFIED]
- NES (elya-nes), measured in emulators (MAME 0.277, repeated in Mesen 2.1.1): model with vocabulary 64, width 64,
  3 layers, 2 heads, feed-forward 128, 102,400 ternary weights, 4-bit activations, context 20; 1,117,248 CPU cycles
  per token = 0.6243 s at 1,789,772 Hz; the ternary kernel costs 10.688 cycles per multiply-add, the attention kernels
  8.00. The trainer's forward pass equals the 6502's forward pass (1,216 of 1,216 tokens identical).
  https://github.com/Scottcjn/elya-nes [VERIFIED]
- Same NES project, quality numbers (validation, nats per character, two seeds each): exact softmax normaliser
  instead of a power-of-two shift normaliser -0.029; 8 feed-forward weight sets selected by the current token id
  -0.185 at +0.54% cycles; context 85 instead of 20 +0.019 on the two-seed mean (+0.021 and +0.017 per seed; worse)
  at +52% cycles in the 60,000-step runs, and +0.047 to +0.049 in a separate 12,000-step experiment; attention is
  7.3% of a token at context 20 and 52.9% at context 85 (share at the last position).
  https://github.com/Scottcjn/elya-nes [VERIFIED]
- Added in the fact-check pass (missed by the research pass): small integer-only transformers on stock 8-bit
  machines. Soul Player C64: 2 layers, width 32, 4 heads of 8, feed-forward 64 (ReLU), vocabulary 128, context 20,
  about 25,000 int8 parameters with per-tensor power-of-two shifts, Q8.8 activations in int16, a 128-entry exp
  table, integer square root and division in RMSNorm, shift-and-add multiplication in 6502 assembly, trained with
  quantization-aware training; "About 60 seconds per token on real hardware"; no loss figures. Its Z80 ports: ZX
  Spectrum (the README expects about 17 s/token from the clock ratio, not a measurement) and Amstrad CPC.
  8bitGPT (Commodore 64): character-level, about 18,000 parameters, 2 layers, width 32, context 32, int8 weights
  with per-tensor integer rescale factors, Q8.8, "each character takes ~30 s" at 1 MHz. ti84-llm-math: a
  llama2.c-style character model for two-digit addition and subtraction (width 32, 4 layers, 54K parameters,
  Q16.16), about 1 minute per forward pass on an emulated TI-84 Plus; on the physical calculator it installs but
  crashes at launch. https://github.com/gizmo64k/soulplayer-c64 , https://github.com/tunechr/soulplayer-spectrum ,
  https://github.com/G1D30N/soulplayer-cpc , https://github.com/Timothy-Sikes/8bitgpt ,
  https://github.com/RileyGreiff/ti84-llm-math [VERIFIED]
- Z80 (Z80-uLM) and its TI-84 Plus CE fork (eZ80-ai) run small neural text models with integer arithmetic, but
  they are perceptrons over hashed character trigrams, not transformers. The Atari 800 model (JAM) describes a
  "context attention" step in its README but no transformer layers. None of the three gives tokens/s.
  https://github.com/HarryR/z80ai , https://github.com/marspa73/atarijam , https://github.com/tornadus/ez80ai

**Microcontrollers and old PCs (float32 unless stated)**
- ESP32-S3 (240 MHz, 2 MB PSRAM): stories260K at 19.13 tok/s using both cores and the vendor's SIMD dot product;
  a fork reports about 32.83 tok/s. https://github.com/DaveBben/esp32-llm , https://github.com/eric-humane/esp32-llm [VERIFIED]
- Four ESP32 boards, same 260K model: 2.33 tok/s reading weights from flash (no PSRAM), 11.12, 21.41 and 22.02 tok/s
  with PSRAM (article dated 2026-09-23). https://circuitdigest.com/articles/running-260K-tiny-llm-on-esp32 [VERIFIED]
- STM32H7A3 at 280 MHz: stories260K, float32, "up to ~87 tok/s" including UART output (the README table heads this
  column "Peak tok/s"). https://github.com/schuhandreas/embedllama-stm32h7a3 [VERIFIED]
- DOS (dosllam2): 486DX2-66 with 32 MB 2.08 tok/s; Pentium MMX 200 15.32 tok/s; Pentium III 667 80.04 tok/s (all 260K).
  The same table also gives 171.89 to 386.36 tok/s for the 260K model on four laptops from 2004 to 2020, which are
  not in Table 6.1. https://yeokhengmeng.com/2025/04/llama2-llm-on-dos/ [VERIFIED]
- Windows 98 Pentium II with 128 MB: 260K at 39.31 tok/s, 15M at 1.03 tok/s. https://blog.exolabs.net/day-4/ [VERIFIED]
  A post by an EXO founder quoted in the press gives 35.9 tok/s for the same 260K model (section 3).
- PSP-2000: 260K, not quantized, 35.11 to 36.06 tok/s over three runs (short paper). Corrected in the fact-check
  pass: the paper does not state the CPU clock used in the test; it only gives the PSP range of 1 to 333 MHz.
  https://sol.sbc.org.br/index.php/eramiars/article/download/39470/39242 [VERIFIED]
- PS Vita overclocked to 555 MHz: 260K at about 120 tok/s, 15M at about 1.8 tok/s.
  https://github.com/callbacked/psvita-llm [VERIFIED]
- An MCU-class chip with int8 matrix products (Deeploy on Siracusa, 8 RISC-V cores at 360 MHz plus an accelerator):
  340 tokens/s and 490 microjoules per token for a TinyStories Llama with width 64, 8 layers, 16 heads, feed-forward
  256; softmax uses the I-BERT integer approximation, Swish is replaced by Hardswish, RMSNorm divides with 32-bit
  operands; with the KV cache a 256-step run is 23x faster (0.75 s against 17.6 s) and takes 26x less energy than
  recomputing the context (Table II; the paper's introduction says 26x for throughput).
  https://arxiv.org/abs/2408.04413 [VERIFIED]
- ESP32-S3 with a 28.9M-parameter model (559K dense core, 3.1M head, 25M per-layer token-embedding table kept in
  flash, 4-bit weights): 9.88 tok/s. The table buys +0.098 nats at vocabulary 32,768 but only +0.025 nats at
  vocabulary 4,096; doubling the dense core buys +0.170 nats. https://github.com/slvDev/esp32-ai and
  https://github.com/slvDev/esp32-ai/blob/main/RESULTS.md

**Soft-core CPUs on FPGAs**
- LoongArch32R soft core (OpenLA500, 5-stage) on an Artix-7 XC7A200T, stories260K in float32: about 0.50 tok/s in
  software (1,989 ms for token 1); with a float32 multiply-accumulate coprocessor 4.78 tok/s at token 1 falling to
  1.90 tok/s at token 32 (numbers read from the two screenshots in the README).
  https://github.com/WizLiang/LLaMA2-on-OpenLA500
- VexRiscv RV32IM at 50 MHz on a Cyclone V (PocketTPU), a 1M-parameter TinyStories GPT-2 (width 64, 6 layers), Q8_0
  weights: about 24 tokens/min in software, about 255 tokens/min with a hardware dot-product unit fed by DMA.
  https://github.com/thinkelastic/PocketTPU
- Rocket core at 50 MHz plus an 8x8 int8 systolic array (Gemmini) on a Nexys Video, stories15M: about 3 tok/s greedy.
  Matrix products became 75x faster but the whole run only about 10x; afterwards 52% of the time is float attention,
  SiLU and RoPE on the CPU, 40% is serial output, 8% is sampling.
  https://mikutyan4.github.io/chipyard-linux-nexys/ch07-llm-inference.html
- A RISC-V core extended with a vector dot-product instruction (Nanhu-vdot, FPGA-tested): dot products over 4x faster
  than scalar code, GPT-2 inference about 30% faster. https://arxiv.org/abs/2409.00661 [VERIFIED]

**FPGA datapaths for small language models**
- stories260K as RTL on a Xilinx XCVU9P: int8 weights and activations with a run-time quantizer (group size 4), a
  Q12.20 residual stream, UQ8.24 scales, 12-bit tables for exp, reciprocal, 1/sqrt and SiLU; 235,704 LUTs,
  142,128 flip-flops, 435 block-RAM tiles, 188 DSPs; tokenizer on the PC; no tokens/s published.
  https://github.com/wupd1122/llama2.c-260k-rtl-fpga [VERIFIED]
- stories260K on a Lattice ECP5-45F (Colorlight i9), "int8 + power-of-2 shift" values, four 256-entry Q15 tables
  (exp, reciprocal, 1/sqrt, SiLU), a sequential 18x18 multiply-accumulate unit: 14.02 tok/s decode at 31.25 MHz with context at
  most 32; 26,874 of 43,848 LUT4, 102 of 108 block RAMs (90 of them a cache for the feed-forward weights), 17 of 72
  DSPs. Steps: 6.8 tok/s at 25 MHz, 8.53 at 31.25 MHz, 10.96 after pipelining the dot product from 256 to 68 cycles
  per 64-element row, 14.02 after moving feed-forward weights from SDRAM into block RAM.
  https://github.com/skyline624/ecp5-inference [VERIFIED]
- Same ECP5 design: 1.70 M cycles per token (28.90 M for 17 tokens) for 279,168 weight bytes = 6.1 cycles per byte,
  against a floor of 1 for a sequential multiply-accumulate; the author's conclusion (table dated 30/07) is that
  memory fetch from SDRAM, not the DSP, is the limit: the 66% of weight bytes served from the block-RAM cache cost
  about 1.5 cycles per byte and the SDRAM part about 15. Corrected in the fact-check pass: this is the state before
  four memory-read changes dated 31/07 in the same document, after which the simulated count for the 260K model is
  15.28 M cycles for 17 tokens, which is 0.90 M per token and 3.2 cycles per byte (our arithmetic); board timings
  after the changes are given only for the author's width-256 profile. The README's 14.02 tok/s was not updated,
  and 31.25 MHz / 1.70 M would be 18.4 tok/s (our arithmetic), so the author does not reconcile the board figure
  with the cycle counts. Its output equals the float model for 9 tokens and then differs.
  https://github.com/skyline624/ecp5-inference/blob/main/docs/CAPACITE_FPGA.md and
  https://github.com/skyline624/ecp5-inference/blob/main/docs/PLAN_GEN_SEQUENCER.md [VERIFIED]
- A 16-layer, width-128 ternary transformer (4.72M parameters, of which 4.20M ternary and 0.52M int8 in the
  embedding and output layer; vocabulary 2,048, context 512) on a MAX 10 10M08
  (8,064 logic elements, about 48 KB of on-chip memory, 8 MB SDRAM) at 60 MHz: 55.6 tok/s for a 64-token story,
  50.3 at 256, 45.5 at 480; 5,973 logic elements, all 42 memory blocks, 22 of 48 9-bit multipliers. It is a
  programmable vector processor with six instruction types (load, store, ternary matrix-vector product, dot product,
  element-wise pass, scalar step), int8 vectors, 24-bit accumulators, about 4 clocks of overhead per instruction,
  one element per clock in the element-wise pipeline, 48 instructions per layer.
  The speeds cover the whole run including prompt tokenization and the UART. The repository was created on
  2026-11-12, one day before this research (section 3).
  https://github.com/MRazvan/fpga-tinystories and
  https://github.com/MRazvan/fpga-tinystories/blob/main/docs/VPU_PROGRAMMING.md [VERIFIED]
- Same MAX 10 project, cost of going integer-only (int8 activations, 32-entry base-2 exp table plus shift, integer
  reciprocal, 255-entry SiLU table, integer square root): 0.622 to 0.643 bits per character, 1.647 to 1.704 nats per
  token, on 1,186 held-out TinyStories stories. https://github.com/MRazvan/fpga-tinystories [VERIFIED]
- TinyStories-1M backbone (8 layers, width 64) with 4-bit weights and 8-bit activations on a Cyclone IV EP4CE10
  (10,095 of 10,320 logic elements); embedding, final norm and output layer stay on the PC; RTL not published.
  https://github.com/jianmin5xin/TinyTransformer-FPGA
- High-level-synthesis accelerators use the `runq.c` scheme on much larger parts: HLSTransform, 110M model on a
  VU9P, 57.11 tok/s (from system simulation), perplexity 2.9667 to 2.9679 after quantization; LlamaF, TinyLlama 1.1B
  on a ZCU102, 1.478 tok/s against 0.0935 tok/s on the board's ARM cores, perplexity 7.05 to 7.09.
  https://arxiv.org/abs/2405.00738 and https://arxiv.org/abs/2409.11424 [VERIFIED]
- A microcode sequencer with macro-operations (gateGPT, Virtex-5 at 80 MHz, 16-bit Q5.11, a 1-block model with
  width 24): 32,872 cycles per token when recomputing the context, 10,192 with a KV cache, 2,757 with a 24-lane
  multiply-accumulate, 1,156 in the final design (69,204 tok/s). https://github.com/fguzman82/gateGPT [VERIFIED]

**TinyML transformer papers (mostly not text generation)**
- MCUFormer: 73.62% ImageNet top-1 with 320 KB of memory on an STM32F746. https://arxiv.org/abs/2310.16898 [VERIFIED]
- TinyFormer: 96.1% on CIFAR-10 within 1 MB storage and 320 KB memory; sparse inference up to 12.2x faster than
  CMSIS-NN. https://arxiv.org/abs/2311.01759 [VERIFIED]
- MCUBERT: BERT-tiny and BERT-mini parameters reduced 5.7x and 3.0x; more than 512 tokens within 256 KB of memory.
  https://arxiv.org/abs/2410.17957 [VERIFIED]
- Tiny transformer kernels for MCUs: 4.79x and 2.0x lower latency than CMSIS-NN and PULP-NN; fusing projection
  weights cuts run time 1.53x and parameters 25%. https://arxiv.org/abs/2404.02945 [VERIFIED]

## 3. Uncertain or contested

- **Does shift-only (power-of-two) requantisation cost quality at 260K parameters?** Indirect and mixed evidence.
  The ECP5 author notes that a power-of-two scale "loses up to 1 bit" and that the error accumulated over the 35
  chained matrix products of stories260K reaches "~30-40% on the worst significant element"; that design quantizes
  after training, its reference code picks each shift from the data at run time, and its text leaves the float model's text at token 9 (the
  author blames the simplified attention, not the norm and SiLU tables). The NES project trains with fixed
  power-of-two shifts and matches its integer reference exactly, but found the shift-normalised softmax 0.029
  nats/char worse than an exact division. The MAX 10 project avoids the question with multiply-then-shift
  `(a*M) >> S` [VERIFIED]. Soul Player C64 (added in the fact-check pass) is a working precedent for int8 weights
  with per-tensor power-of-two shifts and quantization-aware training at about 25K parameters, but it keeps int16
  activations and reports no loss [VERIFIED]. A measurement of fixed shifts plus quantization-aware training with
  int8 weights and int8 activations at this size was not found in our search.
  https://github.com/skyline624/ecp5-inference/blob/main/host/v4_quant.py , https://github.com/Scottcjn/elya-nes ,
  https://github.com/MRazvan/fpga-tinystories , https://github.com/gizmo64k/soulplayer-c64
- **Is an int8 residual stream enough?** The MAX 10 design keeps the residual stream in int8 (width 128, 16 layers)
  and loses 0.057 nats/token in total for all integer conversions. The Game Boy and XCVU9P designs keep a wider
  residual stream (Q12 in 32 bits, Q12.20). The NES design uses 4 bits with a clamp and no normalisation layer at
  all. No data for width 64, 5 layers, int8 weights. (URLs as in section 2.)
- **Emulator-only and self-corrected numbers.** The NES figures are cycle counts from emulators (two emulators
  agree on the cycle counts, a third on the tokens); no run on a console is documented. The same author's project page lists earlier claims that were withdrawn
  (an N64 counter that showed about 60 tok/s where 1.23 tok/s was real; a Genesis speedup of 11.3x that was 1.674x).
  We use these repositories for cycle counts and training ablations only.
  https://hackaday.io/project/206401-transformers-on-retro-game-consoles [VERIFIED]
- **Simulated timing.** HLSTransform takes its timing "from the system simulations", and reports the same 57.11
  tok/s for 256 and 1,024 tokens. https://arxiv.org/abs/2405.00738 [VERIFIED]
- **Speeds are not comparable across rows.** Timing scope differs (with or without serial output, prompt or
  generation, which position in the context). Within one system the number moves a lot: 4.78 to 1.90 tok/s over 32
  tokens on the OpenLA500 system, 55.6 to 45.5 tok/s between 64 and 480 tokens on the MAX 10.
- **Is a long context worth its cycles at this size?** On children's stories the NES model got worse with context 85
  than with 20. Python code needs consistency over many lines and benchmark prompts are long, so this result may not
  transfer. A measurement for code was not found in our search.
- **Extra parameters selected by token id.** One project reports a large gain (NES, 8 feed-forward sets, -0.185
  nats/char at vocabulary 64); another reports a small gain at small vocabularies (per-layer embeddings, +0.025 nats
  at vocabulary 4,096 against +0.098 at 32,768). Different mechanisms, one author each, stories only.
- **The 32.83 tok/s ESP32 fork.** Its README also describes the model as "260K parameters, 6 layers, 288 dimensions",
  which is the 15M model's shape; the number is not confirmed by a second source.
  https://github.com/eric-humane/esp32-llm [VERIFIED]
- **Quality after quantization is rarely reported.** The llama2.c64, ESP32, STM32, DOS and PSP ports are float. The
  Game Boy port shows one sample sentence; Soul Player C64 and 8bitGPT show sample text only.
  The ECP5 design reports 9 tokens equal to the float model; the XCVU9P design
  ships one 175-character fixed-point greedy sample that equals the start of the float sample on the model card.
  Deeploy gives no perplexity. Only the MAX 10 project, the NES project, HLSTransform and LlamaF give loss or
  perplexity before and after. ti84-llm-math (added in the fact-check pass) gives exact-match accuracy on two-digit
  sums for its 54K model: 99% after training and about 95% with the fixed-point code [VERIFIED].
- **`runq.c` with its default group size on stories260K.** Our reading of the code, not executed: the export uses
  group size 64 because 64 divides dim; `quantize()` and `matmul()` process only whole groups, so for the hidden
  vector of 172 values only 128 would be used, and the per-group weight scales would not line up with rows of 172.
  The XCVU9P project uses group size 4 (172 = 4 x 43). Do not take default `runq` output as an int8 reference for
  this model without checking. https://github.com/karpathy/llama2.c/blob/master/runq.c
- **Our own estimate of about 1.0 M cycles per token.** The ECP5 design also spends 68 cycles per 64-element row,
  yet needed 1.70 M cycles per token at a context of at most 17, because it waited for SDRAM (6.1 cycles per weight
  byte). Corrected in the fact-check pass: after four memory-read changes the same document gives 15.28 M simulated
  cycles for 17 tokens, which is 0.90 M per token and 3.2 cycles per byte (our arithmetic) [VERIFIED]. A working
  260K design therefore needs about 0.9 M cycles per token at a context of at most 17, where attention is short,
  while the LS estimate is 1.0 M at context 256; the LS number is the more optimistic of the two. It holds only
  with every weight in block RAM and one-cycle reads; the Tier B path
  through DDR2 is unproven. https://github.com/skyline624/ecp5-inference/blob/main/docs/CAPACITE_FPGA.md
- **Pentium II clock and a second speed figure.** The EXO Labs post and README say "Pentium II" and "128MB RAM" and
  give 39.31 tok/s; they state no clock. Tom's Hardware (2024-12-29) describes the machine in EXO's video as "an
  ancient Elonex Pentium II @ 350 MHz" and quotes a post by an EXO founder, "35.9 tok/sec on Windows 98", for the
  same 260K model. So 350 MHz rests on press coverage of a video, and two speeds (39.31 and 35.9 tok/s) exist for
  this machine. [VERIFIED]
  https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-language-model-runs-on-a-windows-98-system-with-pentium-ii-and-128mb-of-ram-open-source-ai-flagbearers-demonstrate-llama-2-llm-in-extreme-conditions
- **Young, single-author sources carry the hardware ideas** (added in the fact-check pass). The MAX 10 repository
  was created on 2026-11-12, one day before this research, and had 1 star and no forks when checked; ecp5-inference
  and dsp_inference had 0 stars. Their numbers were read from READMEs and design notes, and we found no independent
  reproduction. Ideas 5, 6, 8, 12 and 13 lean mainly on the MAX 10 project. (GitHub API, 2026-11-13.) [VERIFIED]

## 4. Not tried at our scale (about 260k to 400k int8 parameters, 8-bit multiply-accumulate CPU, 450 KB)

- A transformer language model on a custom or homebrew 8-bit CPU with a multiply-accumulate instruction:
  not found in our search (Table 6.5). No entry for AVR, 8051, PicoBlaze, Gigatron or breadboard/TTL machines either.
  [VERIFIED] A second, independent search in the fact-check pass (web and GitHub, last rows of Table 6.5) did not
  find one either. It did find int8 transformers on stock 6502 and Z80 machines that the research pass had missed
  (section 2); they run on unmodified CPUs without a multiplier, so they do not change this answer.
- A stories260K-class Llama model with int8 activations everywhere, fixed power-of-two shifts, 256-entry tables only,
  no division, trained with quantization-aware training, with a reported loss: not found. Nearest: the ECP5 design
  (post-training, run-time shifts, context 32, no loss numbers), the MAX 10 design (4.72M parameters, ternary
  matrices, multiplier rescale, integer division and square root) and Soul Player C64 (about 25K parameters, int8
  weights with per-tensor power-of-two shifts, quantization-aware training, but int16 activations, integer division
  and square root, no loss numbers; added in the fact-check pass).
- Any of these tiny systems writing program code, or reporting HumanEval or MBPP: not found. All of them generate
  children's stories, short chat replies, names, verse or small sums.
- int8 x int8 streams under a 24-bit accumulator: the MAX 10 design has 24-bit accumulators but ternary weights; the
  others use 32 bits (or 8 bits in blocks of 16 on the NES). Our arithmetic for LS: a stream of up to 511 products of
  two int8 values cannot overflow a signed 24-bit accumulator (511 x 128 x 128 = 8,372,224, limit 8,388,607); 512
  products overflow only if every product is (-128) x (-128); with operands clamped to +-127 the bound is 520 products
  (520 x 16,129 = 8,387,080) [VERIFIED: arithmetic recomputed]. The baseline's longest streams are 172
  (feed-forward output rows) and T (value mixing).
- Full context on a tiny integer-only machine with a 260K model: the Game Boy port stops at 16 tokens, the Cyclone IV
  design at 16, the NES at 20 (85 on a slower path), Soul Player C64 at 20, the ECP5 at 32. Only the MAX 10 design
  (larger model, SDRAM) runs 512.
- stories260K in integer code on a soft core without floating point: only float runs found (0.50 tok/s on OpenLA500).
- Token-selected feed-forward sets or per-layer token embeddings with a 256 to 512 token BPE vocabulary and code data:
  not found.
- Memory budget check for the baseline at int8 (our arithmetic): weights 260,032 bytes; KV cache 320 bytes per
  position (5 layers x (32 key + 32 value)), so 81,920 bytes at T = 256 and 163,840 at T = 512; RoPE table
  2 x 4 x T bytes; three 256-byte tables. At T = 256 this is about 345 KB of the 450 KB.

## 5. Ranked ideas for LS-LLM

Cycle figures in this table are our arithmetic on the CONTEXT.md cost model (a row of n multiply-adds costs n + 24
cycles, element-wise work 12 to 32 cycles per element, baseline stories260K shape, context 256). They are estimates,
not measurements. The baseline per token has 3,512 matrix rows, 10,240 attention-score dot products of length 8,
10,240 exp lookups, 320 value-mixing streams and 423,168 multiply-adds [VERIFIED: counts recomputed].

| # | Idea | Expected effect on score / speed / memory | Cost to try | Risk | Keep-or-drop test | CPU change |
|---|---|---|---|---|---|---|
| 1 | Use 2 to 4 attention heads of size 16 to 32 instead of 8 heads of size 8 (short dot products are mostly overhead). Precedent: NES 2 x 32, MAX 10 4 x 32. | Speed: score dot products 327,680 to 143,360 cycles and exp lookups 10,240 to 2,560 with 2 heads of 32, about 0.28 to 0.43 M of about 1.0 M cycles saved; 0.18 to 0.29 M with 4 heads of 16. Score: unknown, expected small. Memory: none. | One float training run per shape, two seeds. | Fewer heads may hurt at width 64; no measurement at this size was found in our search. | Validation loss within seed noise of the 8-head model, cycles per token from the simulator. | No |
| 2 | Make a bit-exact integer Python model the specification; the QAT forward pass must equal it; gate every hardware or simulator build on token-exact output over a whole context window. Precedent: NES (1,216 of 1,216 tokens), MAX 10, gateGPT; the ECP5 build passed short tests and failed at token 9. | Score: protects the benchmark numbers from silent mismatches. Speed and memory: none. | Already planned; add a test that runs at least T tokens on 3 prompts and a check that exported weights decode back with zero difference. | Slow simulation of long windows. | Any token mismatch means the build is rejected. | No |
| 3 | Report speed as cycles per token at positions 0, T/2 and T-1, the mean over a full window, and cycles per multiply-add; state timing scope. | Makes the tokens/s claim checkable against 6.1 cycles per weight byte before and 3.2 after its memory-read changes (ECP5; the latter simulated) and 10.7 cycles per multiply-add (NES). | One counter in the simulator and one on the board. | None. | Board counter equals simulator count. | No (a cycle counter register helps) |
| 4 | Decide the context length by measurement: tokenise the benchmark prompts with the chosen 256 or 512 vocabulary, then search T in {128, 256, 512}. | Memory: KV cache is 320 x T bytes (81,920 at 256, 163,840 at 512). Speed: attention share grows with T (NES: 7.3% at 20, 52.9% at 85). Score: prompts that do not fit score zero whatever the model. | One script, then the shape search already planned. | A short T may make HumanEval prompts impossible. | Share of practice-set prompts that fit; score versus cycles per token. | No |
| 5 | One instruction for a whole matrix-vector product (clear, stream, requantise, store, next row) so the per-row overhead drops from about 24 to about 4 cycles. Precedent: MAX 10 engine (about 4 clocks per instruction, matrix-level instructions), gateGPT macro-operations, ECP5 68 cycles per 64-element row. | Speed: 14,072 rows x 20 cycles = about 0.28 M of about 1.0 M cycles saved. Score and memory: none. | VHDL control FSM and assembler support. | Bigger FSM, more to verify; keep bit-exact with idea 2. | Cycle-exact simulation before and after; timing still met at 100 MHz. | Yes |
| 6 | Streaming element-wise instruction: table lookup, shift, add or store over n elements at about one element per clock (an extension of the planned lookup-table instruction). Precedent: MAX 10 element pipeline, n + 11 clocks per pass. | Speed: exp lookups cost 0.12 to 0.33 M cycles per token at T = 256, about 0.02 M when streamed; about 0.06 M more from norms, SwiGLU, residual adds and RoPE. | VHDL plus a destination pointer. | More pipeline registers; pointer conflicts. | Cycle-exact simulation; bit-exact with idea 2. | Yes |
| 7 | Softmax variants trained with QAT: exp from a 256-entry table indexed by the clamped distance to the maximum, or directly by the int8 score (no maximum search); normaliser by exact reciprocal (reference only), power-of-two shift, 256-entry reciprocal table, or none. Precedent: ATTN/11 table of exp(-i/32), MAX 10 32-entry base-2 table plus shift, NES normaliser result (0.029 nats/char); Soul Player C64 uses a 128-entry exp table, and its README reports attention close to uniform until the score shift in front of the table was changed from 17 to 14 bits, so the scale of the score entering the table has to be searched too. | Score: up to a few hundredths of a nat. Speed: skipping the maximum search saves one compare per score. | Four short QAT runs. | All table outputs zero for some inputs; needs a defined fallback (the Game Boy code attends to the last position). | Validation loss per variant; keep the cheapest within seed noise. | No for table-only variants; a maximum or leading-zero register is a small change |
| 8 | Take the cache layout from working designs: keys stored row-wise, values stored transposed so value mixing is head-size streams of length T, and the cache readable through the weight pointer so that activation-by-activation products (query x key, probability x value, sum of squares for RMSNorm) are possible. Precedent: MAX 10 `DOT` over a transposed V cache. | Speed: 64 long streams per layer instead of 64 x T scalar updates. Needed because the unit described in CONTEXT.md multiplies X memory by W memory. | Design decision; uses the planned `ST` and wide pointers. | Stride stores; two memories must both hold activations. | Simulator: attention block equals the reference. | Only the planned ones, provided both pointers can reach the cache |
| 9 | Keep the headline configuration entirely in block RAM. For Tier B build a streaming DDR2 read path and report cycles per weight byte. Evidence: ECP5 6.1 cycles per weight byte overall before its memory-read changes (about 1.5 for the block-RAM part, about 15 for the SDRAM part) and 3.2 after (simulated), 10.96 to 14.02 tok/s after moving feed-forward weights on-chip; ESP32 boards 2.33 tok/s from flash against 22.02 with PSRAM (different boards); openTPU decodes at 82 to 94% of memory peak. | Speed: avoids a several-fold loss in Tier B. | None for Tier A; FIFO and controller work for Tier B. | DDR2 timing and controller complexity. | Cycles per weight byte at most about 1.5 in simulation and on the board. | Tier B only |
| 10 | Clamp all int8 values to +-127 in training, export and requantisation (the Q8_0 range of llama2.c; the MAX 10 design does the same). | Guarantees no 24-bit overflow for streams up to 520 products; symmetric range. Costs one code level. | Trivial. | None known. | Assertion in the reference model over the practice set: accumulator never leaves 24 bits. | No, if `RQ` clamps to -127 or QAT never produces -128 |
| 11 | For Tier B only: several feed-forward weight sets selected by the current token id, or per-layer token embeddings, to add stored parameters at almost the same multiply-adds per token. Evidence: NES 8 sets -0.185 nats/char at +0.54% cycles; ESP32 per-layer embeddings +0.025 to +0.098 nats. | Score: possibly large, unproven for code. Speed: about unchanged. Memory: +165,120 bytes per extra feed-forward set of the baseline, so it does not fit Tier A. | Training and export code; a pointer table by token id. | Single-author evidence, stories only, small vocabularies. | Validation loss and practice-set score at equal cycles per token for 1, 4, 8 sets. | No |
| 12 | Multi-lane multiply-accumulate: 2 to 4 weights per clock from a wider block-RAM word, one DSP and accumulator per lane. Evidence: gateGPT 10,192 to 2,757 cycles per token with 24 lanes; MAX 10 reads 8 ternary weights per clock. | Speed: the 259,328 matrix-product cycles shrink by the lane count (0.13 to 0.19 M saved); worth more after ideas 5 and 6. | Large CPU change. | Changes the one-multiplier character of the CPU; report separately from the main numbers. | Cycle-exact simulation; resource and timing report. | Yes |
| 13 | 2-bit (ternary) or 4-bit weights to hold 2 to 4 times more parameters in 450 KB. Evidence: the MAX 10 ternary model (4.72M parameters in 1.5 MB) reaches 0.622 bits/char in float activations, close to TinyStories-3M (0.626). | Score: more parameters per byte. Speed: more multiply-adds per token, so slower unless lanes are added. Memory: 2 to 4 times denser. | Ternary QAT plus an unpack path in the CPU. | Works against the speed target. | Score per cycle against the int8 model at equal memory. | Yes |

Ideas 1, 5 and 6 together would bring the baseline at context 256 to roughly 0.5 M cycles per token (about 200
tokens/s at 100 MHz) by the same arithmetic; this needs confirmation in the cycle-exact simulator.
[UNVERIFIED: an estimate on the CONTEXT.md cost model, not simulated or measured; the fact-check pass recomputed the
sum as about 0.46 M cycles under the same assumptions.]

## 6. Tables asked for in the topic brief

### 6.1 Documented systems

Tokens/s are the authors' figures. "260K" is stories260K (260,032 parameters). "float32" means the unmodified
llama2.c arithmetic. n.d. = no date recorded by us.

**A. 8-bit CPUs**

| System | Year | CPU and clock | Memory | Model and parameters | Quantization / arithmetic | Tokens/s | Notes | Source |
|---|---|---|---|---|---|---|---|---|
| Commodore 64 + RAM expansion unit (llama2.c64) | 2025 | MOS 6510, about 1 MHz [UNVERIFIED: clock not stated in the README, standard C64 value] | 64 KB + 2 MB REU holding the weights | 260K | None: float32 weights unchanged, software float from the oscar64 compiler, sin/cos/exp polynomials from the BASIC ROM | about 0.002 (one token per about 8 min; log: 62 tokens in 9:25:43 = 547 s each) | Same output as llama2.c at temperature 0; 37 to 40 s per token in emulator warp mode; float multiply through an 8x8 table was 2x slower; int8 version not written | https://github.com/ytmytm/llama2.c64 [VERIFIED] |
| Game Boy Color (gbc-transformer) | 2026 | 8-bit Game Boy CPU without multiply or divide instruction, clock not stated | Weights in bank-switched cartridge ROM, KV cache in cartridge SRAM, under 8 KB work RAM | 260K | int8 weights with a Q16 scale per row; Q12 activations (int32, clamped to int16 for products); int32 sums; integer sqrt and division; rational exp | 0.0059 (169 s per token), real hardware; from an elapsed time the README calls estimated | Context 16, greedy; C code without assembly kernels; sample "Ares was a big, farmer. He was very happy" | https://github.com/maddiedreese/gbc-transformer [VERIFIED] |
| NES (elya-nes) | 2026 | Ricoh 2A03 (6502), 1,789,772 Hz | 2 KB RAM, 32 KB cartridge RAM for the KV cache, banked ROM | Own model: vocabulary 64, width 64, 3 layers, 2 heads, feed-forward 128; 102,400 ternary weights (446,464 with 8 token-selected feed-forward sets) | Ternary weights, 4-bit activations, 8-bit sums in blocks of 16; QAT against an exact integer reference | 1.58 to 1.60 (0.6338 to 0.6243 s per token), context 20 | Emulator cycle counts (MAME, Mesen), not a console run; 10.688 cycles per multiply-add | https://github.com/Scottcjn/elya-nes [VERIFIED] |
| Commodore 64, stock (Soul Player C64); row added in the fact-check pass | 2026 | MOS 6510, 1 MHz | 25.3 KB weights and 5.8 KB activation buffers inside the 64 KB | Own chat model: 2 layers, width 32, 4 heads of 8, feed-forward 64, vocabulary 128, context 20; about 25,000 parameters | int8 weights with per-tensor power-of-two shifts, Q8.8 activations (int16), 128-entry exp table, integer sqrt and division, QAT | about 0.017 ("About 60 seconds per token on real hardware") | 6502 assembly, shift-and-add multiply, greedy; no loss figures; Z80 ports for ZX Spectrum and Amstrad CPC | https://github.com/gizmo64k/soulplayer-c64 [VERIFIED] |
| Commodore 64, stock (8bitGPT); row added in the fact-check pass | 2026 | MOS 6502 family, 1 MHz | not stated | Character-level: 2 layers, width 32, context 32; about 18,000 parameters | int8 weights with per-tensor integer rescale factors, Q8.8, integer only | about 0.03 characters/s ("each character takes ~30 s") | C (cc65) with an assembly quarter-square matrix-vector kernel; KV cache | https://github.com/Timothy-Sikes/8bitgpt [VERIFIED] |
| TI-84 Plus (ti84-llm-math); row added in the fact-check pass | 2026 | Z80, about 15 MHz | model paged from flash through a 16 KB window; KV cache of about 16 KB | Character-level arithmetic model: width 32, 4 layers, 54K parameters, vocabulary 14 | Q16.16 fixed point, int8 group-quantized weights, integer only | about 0.017 forward passes/s (about 1 minute each) | Emulator only (Wabbitemu); installs on a physical TI-84 Plus C Silver Edition but crashes at launch | https://github.com/RileyGreiff/ti84-llm-math [VERIFIED] |
| C64 running a RISC-V emulator (semu-c64) | 2025 | 6510 emulating RV32 | 2 MB REU | "240k 8bit model" with the `runq` code path | Q8 (`runq`) | about 25 min for 2 tokens at about 3000% emulator warp | Emulator only | https://github.com/trholding/semu-c64 |
| Z80, CP/M and ZX Spectrum (Z80-uLM) | 2025 | Z80, 4 MHz | 64 KB; program about 40 KB | Perceptron over 128 + 128 trigram buckets (for example 256, 192, 128 hidden), not a transformer | 2-bit weights, 16-bit integer sums, QAT | not stated | Character-by-character replies | https://github.com/HarryR/z80ai |
| Atari 800 / 130XE (JAM) | 2026 | 6502, clock not stated | Models of about 1.6 KB to 76 KB | Neural text model with a "context attention" step, not a standard transformer | Not documented in the README | not stated | | https://github.com/marspa73/atarijam |
| TI-84 Plus CE (eZ80-ai) | 2026 | eZ80 | Weights in about 1.5 MB of archived flash | Fork of Z80-uLM (its attention step was removed) | Integer, QAT | not stated | | https://github.com/tornadus/ez80ai |

**B. Old PCs, consoles and minicomputers (16- and 32-bit)**

| System | Year | CPU and clock | Memory | Model and parameters | Quantization / arithmetic | Tokens/s | Notes | Source |
|---|---|---|---|---|---|---|---|---|
| 486 PC, MS-DOS 6.22 (dosllam2) | 2025 | 486DX2, 66 MHz | 32 MB | 260K | float32 | 2.08 | Open Watcom, DOS/32A extender | https://yeokhengmeng.com/2025/04/llama2-llm-on-dos/ [VERIFIED] |
| Toshiba Satellite 315CDT, DOS (dosllam2) | 2025 | Pentium MMX, 200 MHz | 96 MB | 260K / 15M | float32 | 15.32 / 0.43 | | same page [VERIFIED] |
| Pentium III PC, DOS (dosllam2) | 2025 | Pentium III, 667 MHz | 512 MB | 260K / 15M / 42M | float32 | 80.04 / 2.35 / 0.89 | | same page [VERIFIED] |
| Windows 98 PC (llama98.c) | 2024 | Pentium II (350 MHz in Tom's Hardware's description of EXO's video; the EXO pages state no clock) | 128 MB | 260K / 15M | float32 | 39.31 / 1.03 (35.9 for 260K in a founder's post quoted by Tom's Hardware) | Compiled with Borland C++ 5.02 | https://blog.exolabs.net/day-4/ [VERIFIED] |
| Pentium OverDrive PC, DOS (vellm) | 2026 | Pentium OverDrive, 83 MHz | 24 to 48 MB | stories15M | Q8_0 weights (`runq` scheme), int8 KV cache with float scale per head | 0.27 | | https://github.com/ecliptik/vellm |
| PlayStation Portable 2000 (llama2-psp) | 2025 | MIPS, clock during the test not stated (PSP range 1 to 333 MHz) | 64 MB | 260K | float32 | 35.11 to 36.06 | Short paper with a table of three runs | https://sol.sbc.org.br/index.php/eramiars/article/download/39470/39242 [VERIFIED] |
| PS Vita PCH-1000 (psvita-llm) | 2025 | overclocked to 555 MHz | not stated | 260K / 15M | float32 (llama2.c default; not stated otherwise) | about 120 / about 1.8 | | https://github.com/callbacked/psvita-llm [VERIFIED] |
| Dreamcast (llama2c_Dreamcast) | 2025 | not stated | 16 MB | TinyStories "262K" | not stated | no measured figure in the README ("15M would run at 1 tokens/s", 262K "like 30x faster") | Emulator only according to the curated list in section 7 | https://github.com/gameblabla/llama2c_Dreamcast |
| PDP-11/34A (ATTN/11) | 2026 | 16-bit minicomputer | fits in 32 KB | 1 layer, 1 head, width 16, 1,216 parameters, digit reversal; not a language model | Fixed point: Q8 forward, Q15 backward, Q16 sums; 256-entry exp(-i/32) table | training: 350 steps in 5.5 min | Trained on the machine, in assembly | https://github.com/dbrll/ATTN-11 |

**C. Microcontrollers**

| System | Year | CPU and clock | Memory | Model and parameters | Quantization / arithmetic | Tokens/s | Notes | Source |
|---|---|---|---|---|---|---|---|---|
| ESP32-S3FH4R2 (esp32-llm) | 2024 | dual core, 240 MHz | 2 MB PSRAM (model needs about 1 MB) | 260K | float32, vendor SIMD dot product, both cores | 19.13 | | https://github.com/DaveBben/esp32-llm [VERIFIED] |
| Same chip, fork | 2025 | same | same | 260K (README text inconsistent) | float32 | about 32.83 | Not confirmed by a second source | https://github.com/eric-humane/esp32-llm [VERIFIED] |
| Four ESP32 boards | 2026 | ESP32 and ESP32-S3 | 0, 4, 8, 8 MB PSRAM | 260K | float32 | 2.33 / 11.12 / 21.41 / 22.02 | Slowest reads weights from flash | https://circuitdigest.com/articles/running-260K-tiny-llm-on-esp32 [VERIFIED] |
| ESP32-S3 N16R8 (esp32-ai) | 2026 | 240 MHz | 512 KB SRAM, 8 MB PSRAM, 16 MB flash | 28.9M stored: 559K dense core, 3.1M head, 25M per-layer token-embedding table in flash | 4-bit weights, int8 head and activations for the head | 9.88 end to end (9.72 compute only) | Table rows read from flash cost about 0.12 ms per token | https://github.com/slvDev/esp32-ai |
| STM32H7A3 Nucleo (EmbedLlama) | 2026 | STM32H7A3, 280 MHz | about 1.4 MB RAM, 2 MB flash (weights read in place) | 260K | float32 | up to about 87 (peak), including UART | | https://github.com/schuhandreas/embedllama-stm32h7a3 [VERIFIED] |
| Raspberry Pi Pico (picollama260k) | 2026 | RP2040 | 264 KB SRAM, weights in flash | 260K | float32 math unchanged | not published | Context capped at 128 so the KV cache fits | https://github.com/tarelerulz/pico-experiments |
| RP2040 board (Starmind-Pico) | 2025 | RP2040 | vocabulary limited to 256 by memory fragmentation | Transformers of 1K to 28K parameters | not stated | 2 to 32 | 176 variants tried | https://comfyspace.tech/starmind/experiments/starmind-pico |
| RP2350 + PSRAM (PicoLlama) | 2025 | RP2350 | 8 MB PSRAM, 16 MB flash | bundled 14 MB model | float32 | about 1.2 (850 ms per token) | About 4x slower on cores without a floating-point unit | https://github.com/earlephilhower/PicoLlama |
| Siracusa MCU (Deeploy) | 2024 | 8 RISC-V cores at 360 MHz plus accelerator | on-chip memory only | TinyStories Llama: width 64, 8 layers, 16 heads, feed-forward 256 | int8 weights and activations in matrix products, I-BERT softmax, Hardswish, 32-bit divisions in RMSNorm; post-training | 340 | 490 microjoules per token; quality after quantization not reported | https://arxiv.org/abs/2408.04413 [VERIFIED] |

**D. Soft-core CPUs on FPGAs**

| System | Year | CPU and clock | Memory | Model and parameters | Quantization / arithmetic | Tokens/s | Notes | Source |
|---|---|---|---|---|---|---|---|---|
| OpenLA500 on Artix-7 XC7A200T | 2025 | LoongArch32R, 5-stage, clock not stated | DDR | 260K | float32; coprocessor with float multiply-accumulate lanes and DMA | 0.50 software; 4.78 at token 1 to 1.90 at token 32 with the coprocessor | Numbers read from README screenshots | https://github.com/WizLiang/LLaMA2-on-OpenLA500 |
| VexRiscv on Cyclone V 5CEBA4 (PocketTPU) | 2026 | RV32IM, 50 MHz | 64 KB block RAM, SDRAM, PSRAM | TinyStories GPT-2: width 64, 6 layers, 1M parameters | Q8_0 weights, dequantized in hardware during DMA | 0.4 software; 4.25 with dot-product unit (24 and 255 tokens/min) | | https://github.com/thinkelastic/PocketTPU |
| Rocket + Gemmini on Nexys Video | n.d. | Rocket core, 50 MHz, 8x8 int8 systolic array | not recorded | stories15M | int8 matrix products on the array, float32 elsewhere | about 3 greedy, about 2 with top-p | Matrix products 75x faster, whole run about 10x | https://mikutyan4.github.io/chipyard-linux-nexys/ch07-llm-inference.html |
| XiangShan Nanhu with a vector dot-product instruction (Nanhu-vdot) | 2024 | RISC-V, FPGA test | not stated | GPT-2 | not stated | about 30% faster than software | Dot product itself over 4x faster | https://arxiv.org/abs/2409.00661 [VERIFIED] |

**E. FPGA datapaths and accelerators**

| System | Year | Device and clock | Memory | Model and parameters | Quantization / arithmetic | Tokens/s | Notes | Source |
|---|---|---|---|---|---|---|---|---|
| llama2.c-260k-rtl-fpga | 2026 | Xilinx XCVU9P, 50 MHz input clock | 435 block-RAM tiles | 260K | int8 weights and activations with run-time quantizer (group 4), Q12.20 residual, UQ8.24 scales, 12-bit tables | not published | 235,704 LUT, 188 DSP; tokenizer on PC | https://github.com/wupd1122/llama2.c-260k-rtl-fpga [VERIFIED] |
| dsp_inference | 2026 | Gowin GW2AR-18 (Tang Nano 20K) | SDRAM | 260K | int8 + power-of-two shift, table operators | about 0.33 (about 3 s per token, PC-orchestrated over UART) | The all-in-one version does not route on this device | https://github.com/skyline624/dsp_inference |
| ecp5-inference | 2026 | Lattice ECP5-45F, 31.25 MHz | 102 of 108 block RAMs, SDRAM, SPI flash boot | 260K | int8 + power-of-two shift, four 256-entry Q15 tables | 14.02 decode, context up to 32 | 26,874 LUT4, 17 DSP; 1.70 M cycles per token before its memory-read changes, 0.90 M after (simulated; our arithmetic from 15.28 M for 17 tokens) | https://github.com/skyline624/ecp5-inference [VERIFIED] |
| fpga-tinystories | 2026 | Intel MAX 10 10M08, 60 MHz | about 48 KB on chip, 8 MB SDRAM, 8 MB flash | Own model: 16 layers, width 128, 4 heads, 4.72M parameters (4.20M ternary, 0.52M int8), vocabulary 2,048, context 512 | Ternary weights (QAT), int8 embedding and head, int8 activations, 24-bit sums, base-2 exp table, SiLU table, integer sqrt and reciprocal | 55.6 at 64 tokens, 50.3 at 256, 45.5 at 480 | Six-instruction vector processor, 5,973 logic elements; tokenizer on chip; speeds include prompt tokenization and UART; repository created 2026-11-12 | https://github.com/MRazvan/fpga-tinystories [VERIFIED] |
| TinyTransformer-FPGA | 2026 | Cyclone IV EP4CE10, 50 MHz | 46 M9K blocks | TinyStories-1M backbone: 8 layers, width 64 | 4-bit weights, 8-bit activations | not published | Embedding and head on PC; RTL not public; sequence length 16 | https://github.com/jianmin5xin/TinyTransformer-FPGA |
| openTPU port | 2026 | Gowin GW2AR-18, 48 MHz | 8 MiB SDRAM + SD card | stories42M | int8 matrix products, float32 vector operations | 0.22 | Custom accelerator instruction set; every token re-reads the image from the card | https://github.com/LakoMoor/openTPU-tangnano20k |
| openTPU | 2026 | Kintex-7 PCIe card, 133.33 MHz | DDR | LFM2.5-230M and larger | int8 or 4-bit | 59.0 (230M, int8) | Decode runs at 82 to 94% of memory peak rate | https://github.com/FeSens/openTPU |
| HLSTransform | 2024 | Xilinx VU9P | on-chip and external | stories110M | Q8_0 (`runq` scheme) | 57.11 (system simulation) | 9 W average | https://arxiv.org/abs/2405.00738 [VERIFIED] |
| LlamaF | 2024 | Xilinx ZCU102 | DDR | TinyLlama 1.1B | int8 weights and activations, groups of 256, post-training | 1.478 / 1.424 / 1.328 for 64 / 128 / 256 steps | 0.0935 on the ARM cores alone | https://arxiv.org/abs/2409.11424 [VERIFIED] |
| TALOS-V2 | 2026 | Cyclone V (DE1-SoC), 56 MHz [UNVERIFIED: clock and parameter count are from a third-party report, not the README] | ROM | microGPT, 4,192 parameters | Q4.12 | "50k+" | Character-level names | https://github.com/Luthiraa/TALOS-V2 |
| gateGPT | 2026 | Virtex-5 XC5VLX110T, 80 MHz | block RAM | 1 block, width 24, 4 heads, context 16, vocabulary 27 | 16-bit Q5.11, 17-entry exp table with interpolation | 53,800 to 69,204 | Microcode sequencer; 16,548 LUT, 62 DSP | https://github.com/fguzman82/gateGPT [VERIFIED] |
| kev-gpt | 2026 | Xilinx KV260, 166.7 to 200 MHz | weights in block RAM and UltraRAM | 2 to 4M-parameter model | 4-bit weights (QAT) x 8-bit activations | 19,242 single stream; 59,965 with 16 streams | | https://github.com/MichaelAyles/kev-gpt |

**F. A machine built from logic gates in a game**

| System | Year | CPU and clock | Memory | Model and parameters | Quantization / arithmetic | Tokens/s | Notes | Source |
|---|---|---|---|---|---|---|---|---|
| CraftGPT (Minecraft redstone) | 2025 | redstone logic, simulated | not stated | 5,087,280 parameters, vocabulary 1,920, 6 layers, context 64 | not opened by us | hours per reply even on an accelerated redstone server | Parameter count from The Register | https://github.com/sammyuri/craftgpt and https://www.theregister.com/2025/10/02/minecraft_ai_chatbot/ |

Listed in a curated catalogue but not opened by us [UNVERIFIED: numbers copied from the list, primary pages not
read]: Nintendo Wii, stories15M int8, 61.9 tok/s (GPU engine) and 12.7 tok/s (CPU); Nintendo 3DS (2014 revision),
stories15M Q8_0 / Q4, 18.46 / 23.28 tok/s; Xilinx KV260, LLaMA2-7B 4-bit, about 5 tok/s decode.
https://github.com/emrehangorgec/awesome-llm-on-anything

For scale: the XCVU9P design above uses 235,704 LUTs; the Artix-7 XC7A100T on the Nexys A7 has 63,400 (15,850
slices with four LUTs each; the same table lists 135 block RAMs of 36 Kb, 4,860 Kb in total, and 240 DSP48E1 slices)
[VERIFIED]. Xilinx DS180 v1.17, Table 2 and its note 1, read from a university mirror because the vendor site did
not serve the file to our client: https://eng.auburn.edu/~nelson/courses/elec4200/FPGA/ds180_7Series_Overview.pdf

### 6.2 stories260K facts

| Item | Value | Source |
|---|---|---|
| dim | 64 | file header (bytes 0..27 of stories260K.bin) |
| n_layers | 5 | file header |
| n_heads / head size | 8 / 8 | file header; 64 / 8 |
| n_kv_heads | 4 ("two heads at a time share their key,value projections") | file header; model card |
| hidden_dim (SwiGLU) | 172 = 4 x ceil(170 / 4), with 170 = int(2 x 4 x 64 / 3) and `--multiple_of=4` | file header; model.py |
| max_seq_len | 512 | file header |
| vocab_size | 512, output layer tied to the embedding | file header (positive value); model.py |
| Other structure | RMSNorm (eps 1e-5) before attention and before the feed-forward block, RoPE on queries and keys, no biases | model.py |
| Parameters | 260,032 = 32,768 + 5 x 45,440 + 64; per layer: wq 4,096, wk 2,048, wv 2,048, wo 4,096, w1 11,008, w2 11,008, w3 11,008, two norms 128 | our arithmetic, equals the file size |
| stories260K.bin | 1,056,540 bytes (28-byte header + 264,128 float32, which include 4,096 RoPE table values) | Hugging Face file listing |
| stories260K.pt | 1,061,090 bytes | Hugging Face file listing |
| Tokenizer | `tok512`: SentencePiece BPE with 512 tokens trained on TinyStories (`--vocab_source=custom --vocab_size=512`); tok512.model 7,645 bytes, tok512.bin 6,227 bytes, longest token 7 bytes | model card; file listing; tok512.bin header |
| Validation loss | 1.2968 (card), 1.297 (README) | model card; README |
| Training command | `python train.py --out_dir="outmini" --batch_size=128 --max_seq_len=512 --gradient_accumulation_steps=1 --vocab_source="custom" --vocab_size=512 --dim=64 --n_layers=5 --n_heads=8 --n_kv_heads=4 --multiple_of=4 --learning_rate=1e-3 --dropout=0.05 --weight_decay=0.01 --max_iters=100000 --beta2=0.99 --warmup_iters=1000 --eval_interval=2000 --eval_iters=100 --compile=True` | model card |
| Training time | "~10 minutes (?)" on an A100 | model card |
| Training data | TinyStories (dataset licence CDLA-Sharing-1.0) | model card; Hugging Face dataset card |
| Greedy sample | "Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. ..." | model card |
| Licence | MIT (code and checkpoints) | LICENSE file; Hugging Face metadata |

Sources: https://huggingface.co/karpathy/tinyllamas/blob/main/stories260K/readme.md ,
https://huggingface.co/api/models/karpathy/tinyllamas/tree/main/stories260K ,
https://github.com/karpathy/llama2.c/blob/master/model.py , https://github.com/karpathy/llama2.c#models ,
https://huggingface.co/api/datasets/roneneldan/TinyStories
[VERIFIED] The header bytes, file listing, tokenizer header, model card, README row and the three licences were read
again in the fact-check pass, and the parameter arithmetic was recomputed.

### 6.3 What `runq.c` quantizes

| Part | In `runq.c` |
|---|---|
| wq, wk, wv, wo, w1, w2, w3, classifier | int8, Q8_0: symmetric, range -127..127, one float32 scale per group of GS consecutive weights (export default GS = 64, halved until it divides dim) |
| Token embedding table | stored as int8 with scales, converted to float32 once at load |
| RMSNorm gains | float32 |
| Vector entering a matrix product | float32, quantized at run time: per group, scale = max-abs / 127 (float), value = round(x / scale) as int8 |
| Matrix product inner loop | int8 x int8 summed in int32 per group, then multiplied by the weight scale and the activation scale in float; the result is float32 |
| Residual stream, query, key, value, KV cache, attention scores, logits | float32 |
| RoPE | float (`powf`, `cosf`, `sinf` at every call) |
| Softmax, SiLU, RMSNorm | float (`expf`, `sqrtf`, divisions) |
| Speed claim | Llama-2 7B, 64 threads: 4.6 tok/s float32, 14 tok/s int8 |

Caveat from our reading of the code (not executed): `quantize()` and `matmul()` handle whole groups only, so the
default group size 64 does not fit the hidden size 172 of stories260K; a group size that divides 172 (4) is needed.
Sources: https://github.com/karpathy/llama2.c/blob/master/runq.c ,
https://github.com/karpathy/llama2.c/blob/master/export.py , https://github.com/karpathy/llama2.c#int8-quantization [VERIFIED]

### 6.4 Cycles per weight in integer designs

| Design | Clock | Cycles per token | Weights touched per token | Cycles per weight | Source |
|---|---|---|---|---|---|
| LS-LLM estimate, baseline at context 256 | 100 MHz | about 1.0 M | 423,168 multiply-adds (our arithmetic) | about 2.4 | research/CONTEXT.md |
| ecp5-inference, stories260K, context up to 17, before its memory-read changes (table dated 30/07) | 31.25 MHz | 1.70 M | 279,168 weight bytes read | 6.1 | https://github.com/skyline624/ecp5-inference/blob/main/docs/CAPACITE_FPGA.md [VERIFIED] |
| ecp5-inference, same model, after four memory-read changes (dated 31/07), simulation | 31.25 MHz | 0.90 M (15.28 M for 17 tokens; our arithmetic) | 279,168 weight bytes read | 3.2 (our arithmetic) | same document [VERIFIED] |
| fpga-tinystories, 4.72M-parameter model (4.20M ternary), short context | 60 MHz | "about 1 million" (1.08 M from 55.6 tok/s) | 4.20 M ternary weights + 262,144 int8 head weights | about 0.24 (our arithmetic) | https://github.com/MRazvan/fpga-tinystories [VERIFIED] |
| elya-nes, dense model, context 20 | 1.79 MHz | 1,117,248 | 102,400 ternary weights | 10.9 (kernel: 10.688) | https://github.com/Scottcjn/elya-nes [VERIFIED] |
| gateGPT, final design | 80 MHz | 1,156 (1,488 at the longest context) | not counted | not computed | https://github.com/fguzman82/gateGPT [VERIFIED] |

### 6.5 Key question: searches run

Question: has anyone run a transformer language model on a custom or homebrew 8-bit CPU that has a multiply-accumulate
instruction, or on any 8-bit datapath in an FPGA?

Answer: a custom or homebrew 8-bit CPU running a transformer language model was not found in our search, and not in
the second search of the fact-check pass either [VERIFIED]. Transformers on stock 8-bit CPUs without a multiplier
exist (llama2.c64, gbc-transformer, elya-nes and, added in the fact-check pass, Soul Player C64 with its Z80 ports,
8bitGPT and ti84-llm-math). FPGA
designs with 8-bit operands do exist (ecp5-inference: int8 x int8 with power-of-two shifts; fpga-tinystories: int8
activations, ternary weights, 24-bit sums; TinyTransformer-FPGA: 8-bit activations, 4-bit weights), but they are
dedicated engines or vector processors with wider internal registers, multipliers for rescaling or dividers, not
general 8-bit CPUs. Custom instruction sets for language models exist at larger word sizes (fpga-tinystories vector
engine, gateGPT microcode, openTPU, the Nanhu-vdot dot-product instruction).

| Where | Query | Result |
|---|---|---|
| Web search | homebrew 8-bit CPU breadboard computer runs language model transformer inference custom CPU LLM | Nothing relevant (Z80-uLM, breadboard CPUs without ML) |
| Web search | transformer language model on custom 8-bit soft-core CPU FPGA "multiply-accumulate" instruction int8 inference student-designed processor | Nothing relevant (TATAA, ELiTeFormer, an integer accelerator beside a RISC-V core) |
| Web search | 8051 OR "PicoBlaze" OR "Gigatron" OR "TTL computer" OR "Ben Eater" neural language model transformer inference tokens | Nothing relevant (ESP32 projects only) |
| Web search | Arduino AVR ATmega328 transformer language model inference tinystories tokens per second 8-bit microcontroller | Nothing for AVR |
| Web search | Z80 CP/M language model transformer inference "Z80" LLM 8-bit integer quantized github | Z80-uLM (perceptron, not a transformer). Missed by this query: the Z80 ports of Soul Player C64, found in the fact-check pass |
| Web search | TI-84 Plus CE calculator runs transformer language model locally eZ80 llama2.c tinystories | eZ80-ai (not a transformer). Missed by this query: ti84-llm-math, a transformer on an emulated TI-84 Plus, found in the fact-check pass |
| Web search | language model running on Game Boy TinyStories transformer tokens per second cartridge | gbc-transformer, elya ports |
| Web search | llama2.c RISC-V soft core FPGA PicoRV32 VexRiscv NEORV32 tinystories inference tokens per second | Chipyard and Gemmini tutorial; PocketTPU found through GitHub |
| GitHub repository search | "llama2.c" (top 60 by match), "stories260K", "stories260K in:readme fpga", "tinystories fpga in:readme", "llama2.c picorv32 in:readme", "llama2.c vexriscv in:readme", "llama2.c neorv32 in:readme", "llama2.c 8-bit cpu in:readme", "transformer language model 8-bit cpu", "llm 8051", "llama2.c avr", "llm homebrew cpu", "llama2.c 6502", "microgpt fpga" | The FPGA and soft-core projects in Table 6.1; no 8-bit custom CPU, no AVR, no 8051 |
| arXiv API | abs:"8-bit" AND abs:"microcontroller" AND abs:"language model" AND abs:transformer | 0 results |
| arXiv API | abs:"soft processor" AND abs:transformer AND abs:"language model" | 0 results |
| arXiv API | abs:TinyStories AND abs:FPGA | 0 results |
| arXiv API | abs:"Artix-7" AND abs:transformer AND (abs:"language model" OR abs:LLM) | 0 results |
| arXiv API | abs:"custom instruction" AND abs:"RISC-V" AND abs:transformer AND abs:"language model" AND abs:FPGA | 2 results: Nanhu-vdot (2409.00661) and an instruction-set-extension framework for BERT-class models (2610.08688) |
| Curated list | awesome-llm-on-anything | No homebrew CPU entry; it still lists Game Boy and graphing calculators as wanted |
| Fact-check pass, web search | transformer language model running on homebrew OR "custom-designed" 8-bit CPU FPGA "multiply-accumulate" OR "MAC instruction" TinyStories tokens per second | Nothing relevant (FPGA accelerators, microGPT on an FPGA, the Gemmini tutorial) |
| Fact-check pass, web search | "GPT" OR "transformer" OR "LLM" inference on "8-bit" microcontroller AVR ATmega328 Arduino Uno "tokens per second" int8 attention 2025 OR 2026 | Nothing for AVR (ESP32 projects only) |
| Fact-check pass, web search | language model on my own CPU design: "designed my own CPU" OR "homemade CPU" OR "TTL computer" OR "breadboard computer" runs GPT transformer text generation | Nothing relevant |
| Fact-check pass, web search | microgpt port 6502 OR Z80 OR AVR OR "Arduino Uno" OR "Apple II" OR "ZX Spectrum" OR "Atari" transformer attention assembly 8-bit | Soul Player C64 (int8 transformer on a stock C64) and snes-gpt (16-bit console); no custom CPU |
| Fact-check pass, web search | student-designed OR "custom" "8-bit" processor FPGA VHDL runs "language model" OR "transformer" OR "llama2.c" inference int8 "MAC" accumulator thesis OR project 2025 OR 2026 | Nothing relevant (student image-classification accelerators; an int8 accelerator paper seen in the results, not opened) |
| Fact-check pass, web search | GPT OR transformer OR "language model" running on nand2tetris "Hack computer" OR "Turing Complete" game CPU OR Logisim CPU OR "Minecraft CPU" OR "custom ISA" hobby CPU tokens | Nothing relevant |
| Fact-check pass, GitHub repository search | 22 name and description queries (for example "homebrew cpu llm", "8-bit cpu gpt", "transformer inference picoblaze", "llama2.c avr", "gpt atmega328", "llm ttl cpu", "microgpt 6502") and 17 README queries (for example: transformer "8-bit cpu" language model; llm "homebrew cpu"; gpt "8-bit" 6502 transformer attention; transformer z80 attention "int8"; tinystories "own cpu" OR "custom processor" fpga; gpt nand2tetris; llm logisim cpu) | No custom or homebrew CPU. Found and added: soulplayer-c64, soulplayer-spectrum, soulplayer-cpc, 8bitgpt, ti84-llm-math |

## 7. Sources

Titles are quoted as published; words such as "new" inside a title are the authors' wording. Licences were read
from the GitHub API (`license.spdx_id`), a LICENSE file, a README, or the Hugging Face card metadata. "None detected"
means the GitHub API reports no licence; that code may not be reused without asking.

Fact-check pass: the 34 GitHub repositories that the research pass lists below were queried again through the GitHub
API on 2026-11-13; all exist and every licence entry matches [VERIFIED]. The API marks semu-c64, dosllam2, llama98.c,
PicoLlama, the eric-humane esp32-llm repository and soulplayer-cpc as forks. All ten arXiv identifiers resolve to the
titles and authors given [VERIFIED]. Rows marked "added" come from the fact-check pass.

**Baseline model and code**

| Title | Authors | Year | URL | Licence |
|---|---|---|---|---|
| llama2.c: "Inference Llama 2 in one file of pure C" (README, model.py, export.py, runq.c) | Andrej Karpathy | 2023 | https://github.com/karpathy/llama2.c | MIT |
| tinyllamas checkpoints, stories260K model card | Andrej Karpathy | 2023 | https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K | MIT |
| TinyStories dataset | Ronen Eldan, Yuanzhi Li | 2023 | https://huggingface.co/datasets/roneneldan/TinyStories | CDLA-Sharing-1.0 |

**8-bit and retro machines**

| Title | Authors | Year | URL | Licence |
|---|---|---|---|---|
| llama2.c64 (README, STORY.txt, nnet64.c) | Maciej Witkowiak | 2025 | https://github.com/ytmytm/llama2.c64 | README says MIT; none detected by the API |
| gbc-transformer | maddiedreese (repository owner) | 2026 | https://github.com/maddiedreese/gbc-transformer | MIT |
| elya-nes (README, DESIGN.md, FINDINGS.md) | Scott Boudreaux | 2026 | https://github.com/Scottcjn/elya-nes | Apache-2.0 |
| Transformers on Retro Game Consoles (project page) | Scott Boudreaux | 2026 | https://hackaday.io/project/206401-transformers-on-retro-game-consoles | not stated |
| semu-c64: "L2E llama2.c on Commodore C-64" | trholding | 2025 | https://github.com/trholding/semu-c64 | MIT |
| Z80-uLM: "A Retrocomputing Micro Language Model" | HarryR | 2025 | https://github.com/HarryR/z80ai | none detected |
| JAM (Just Atari Model) | Marek Spanel | 2026 | https://github.com/marspa73/atarijam | MIT |
| eZ80-ai | tornadus | 2026 | https://github.com/tornadus/ez80ai | MIT |
| "Llama 2 LLM on DOS" (blog) and dosllam2 | Yeo Kheng Meng | 2025 | https://yeokhengmeng.com/2025/04/llama2-llm-on-dos/ and https://github.com/yeokm1/dosllam2 | MIT |
| "Running Llama on Windows 98" (blog) and llama98.c | EXO Labs | 2024 | https://blog.exolabs.net/day-4/ and https://github.com/exo-explore/llama98.c | MIT |
| vellm | ecliptik | 2026 | https://github.com/ecliptik/vellm | MIT |
| "Language Models are the new Doom" (short paper) | Caio Madeira, Mauricio Cecilio Magnaguagno, Dalvan Griebler | 2025 | https://sol.sbc.org.br/index.php/eramiars/article/download/39470/39242 | paper |
| psvita-llm | callbacked | 2025 | https://github.com/callbacked/psvita-llm | none detected |
| llama2c_Dreamcast | gameblabla | 2025 | https://github.com/gameblabla/llama2c_Dreamcast | MIT |
| ATTN/11: "Paper Tape is All You Need" | dbrll | 2026 | https://github.com/dbrll/ATTN-11 | none detected |
| CraftGPT repository and a news article about it (2025-10-02) | sammyuri; The Register | 2025 | https://github.com/sammyuri/craftgpt and https://www.theregister.com/2025/10/02/minecraft_ai_chatbot/ | MIT (repository) |
| Awesome LLM on Anything (curated list) | emrehangorgec | 2026 | https://github.com/emrehangorgec/awesome-llm-on-anything | CC0-1.0 |
| Soul Player C64 (added) | gizmo64k | 2026 | https://github.com/gizmo64k/soulplayer-c64 | GPL-3.0 |
| soulplayer-spectrum (added) | tunechr | 2026 | https://github.com/tunechr/soulplayer-spectrum | none detected |
| soulplayer-cpc (added) | G1D30N | 2026 | https://github.com/G1D30N/soulplayer-cpc | GPL-3.0 |
| 8bitGPT (added) | Timothy-Sikes | 2026 | https://github.com/Timothy-Sikes/8bitgpt | not classified by the API; the README says the integer math is MIT-licensed |
| ti84-llm-math (added) | RileyGreiff | 2026 | https://github.com/RileyGreiff/ti84-llm-math | none detected |
| "AI language model runs on a Windows 98 system with Pentium II and 128MB of RAM" (news article, added) | Tom's Hardware | 2024 | https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-language-model-runs-on-a-windows-98-system-with-pentium-ii-and-128mb-of-ram-open-source-ai-flagbearers-demonstrate-llama-2-llm-in-extreme-conditions | article |

**Microcontrollers**

| Title | Authors | Year | URL | Licence |
|---|---|---|---|---|
| esp32-llm: "Running a LLM on the ESP32" | DaveBben | 2024 | https://github.com/DaveBben/esp32-llm | none detected |
| esp32-llm (fork) | eric-humane | 2025 | https://github.com/eric-humane/esp32-llm | none detected |
| "Running 260K Tiny-LLM on ESP32" | Circuit Digest | 2026 | https://circuitdigest.com/articles/running-260K-tiny-llm-on-esp32 | article |
| esp32-ai: "Running a 28.9M parameter LLM on a microcontroller" (README, RESULTS.md) | slvDev | 2026 | https://github.com/slvDev/esp32-ai | MIT |
| EmbedLlama: llama2.c on STM32H7A3 | schuhandreas | 2026 | https://github.com/schuhandreas/embedllama-stm32h7a3 | MIT |
| pico-experiments (picollama260k) | tarelerulz | 2026 | https://github.com/tarelerulz/pico-experiments | not classified by the API |
| Starmind-Pico: "Training a language model to run on RP2040" | ThomasVuNguyen | 2025 | https://comfyspace.tech/starmind/experiments/starmind-pico | not stated |
| PicoLlama | earlephilhower | 2025 | https://github.com/earlephilhower/PicoLlama | MIT |
| Deeploy: Enabling Energy-Efficient Deployment of Small Language Models On Heterogeneous Microcontrollers | Moritz Scherer, Luka Macan, Victor Jung, Philip Wiese, et al. | 2024 | https://arxiv.org/abs/2408.04413 | paper CC BY-SA 4.0; model Maykeye/TinyLLama-v0 Apache-2.0 |
| MCUFormer: Deploying Vision Transformers on Microcontrollers with Limited Memory | Yinan Liang, Ziwei Wang, Xiuwei Xu, Yansong Tang, et al. | 2023 | https://arxiv.org/abs/2310.16898 | paper; code none detected |
| TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices | Jianlei Yang, Jiacheng Liao, Fanding Lei, Meichen Liu, et al. | 2023 | https://arxiv.org/abs/2311.01759 | paper |
| MCUBERT: Memory-Efficient BERT Inference on Commodity Microcontrollers | Zebin Yang, Renze Chen, Taiqiang Wu, Ngai Wong, et al. | 2024 | https://arxiv.org/abs/2410.17957 | paper |
| Optimizing the Deployment of Tiny Transformers on Low-Power MCUs | Victor J. B. Jung, Alessio Burrello, Moritz Scherer, Francesco Conti, et al. | 2024 | https://arxiv.org/abs/2404.02945 | paper |

**Soft cores and FPGAs**

| Title | Authors | Year | URL | Licence |
|---|---|---|---|---|
| LLaMA2-on-OpenLA500 | WizLiang and team "Circuit Breakers" | 2025 | https://github.com/WizLiang/LLaMA2-on-OpenLA500 | none detected |
| PocketTPU | thinkelastic | 2026 | https://github.com/thinkelastic/PocketTPU | none detected |
| Chipyard RISC-V Tutorial, chapter 7: Running LLM Inference on FPGA with Gemmini | mikutyan4 | n.d. | https://mikutyan4.github.io/chipyard-linux-nexys/ch07-llm-inference.html | web page |
| Research on LLM Acceleration Using the High-Performance RISC-V Processor "Xiangshan" (Nanhu Version) Based on the Open-Source Matrix Instruction Set Extension (Vector Dot Product) | Xu-Hao Chen, Si-Peng Hu, Hong-Chao Liu, Bo-Ran Liu, et al. | 2024 | https://arxiv.org/abs/2409.00661 | paper |
| A Framework for Accelerating Transformer Inference on RISC-V for Edge AI | Ajay Kumar M, Vishnu PS, Yike Li, Shreejith Shanker, et al. | 2026 | https://arxiv.org/abs/2610.08688 | paper |
| llama2.c-260k-rtl-fpga (README, docs/ARCHITECTURE_MAP.md) | wupd1122 | 2026 | https://github.com/wupd1122/llama2.c-260k-rtl-fpga | none detected |
| dsp_inference | skyline624 | 2026 | https://github.com/skyline624/dsp_inference | none detected |
| ecp5-inference (README, FPGA_RESSOURCES.md, docs/CAPACITE_FPGA.md, docs/PLAN_GEN_SEQUENCER.md, host/v4_quant.py) | skyline624 | 2026 | https://github.com/skyline624/ecp5-inference | none detected |
| fpga-tinystories: "A language model that writes stories on a small FPGA board" (README, docs/VPU_PROGRAMMING.md) | MRazvan | 2026 | https://github.com/MRazvan/fpga-tinystories | MIT |
| TinyTransformer-FPGA | jianmin5xin | 2026 | https://github.com/jianmin5xin/TinyTransformer-FPGA | not classified by the API; RTL not published |
| openTPU on Tang Nano 20K | LakoMoor | 2026 | https://github.com/LakoMoor/openTPU-tangnano20k | Apache-2.0 |
| openTPU | FeSens | 2026 | https://github.com/FeSens/openTPU | Apache-2.0 |
| HLSTransform: Energy-Efficient Llama 2 Inference on FPGAs Via High Level Synthesis | Andy He, Darren Key, Mason Bulling, Andrew Chang, Skyler Shapiro, Everett Lee | 2024 | https://arxiv.org/abs/2405.00738 | paper; code https://github.com/HLSTransform/submission MIT |
| LlamaF: An Efficient Llama2 Architecture Accelerator on Embedded FPGAs | Han Xu, Yutong Li, Shihao Ji | 2024 | https://arxiv.org/abs/2409.11424 | paper |
| TALOS-V2 | Luthiraa (repository owner) | 2026 | https://github.com/Luthiraa/TALOS-V2 | Apache-2.0 |
| gateGPT | fguzman82 | 2026 | https://github.com/fguzman82/gateGPT | none detected |
| kev-gpt | MichaelAyles | 2026 | https://github.com/MichaelAyles/kev-gpt | MIT |
| ELiTeFormer: An Efficient Transformer for FPGAs (seen in search, abstract only) | Victor Agostinelli, Nicolas Bohm Agostini, Antonino Tumeo | 2026 | https://arxiv.org/abs/2607.03652 | paper |
| 7 Series FPGAs Data Sheet: Overview, DS180 v1.17 (added; read from a university mirror) | Xilinx | 2015 | https://eng.auburn.edu/~nelson/courses/elec4200/FPGA/ds180_7Series_Overview.pdf | data sheet |

## Verification log

Fact-check pass, 2026-11-13, second agent. 25 claims were opened again at their sources: 23 confirmed (several with
added precision, noted in the line) and 2 corrected (claims 13 and 18). Format: claim; verdict; URL.

1. stories260K header (dim 64, hidden_dim 172, 5 layers, 8 heads, 4 KV heads, vocabulary 512, max_seq_len 512); confirmed by decoding bytes 0..27 again; https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
2. stories260K has 260,032 parameters, stories260K.bin is 1,056,540 bytes, the .pt file 1,061,090, tok512.bin 6,227 (longest token 7), validation loss 1.2968 / 1.297, training command; confirmed (arithmetic recomputed, including the 24-bit accumulator bounds and the per-token operation counts of section 5); https://huggingface.co/api/models/karpathy/tinyllamas/tree/main/stories260K and https://huggingface.co/karpathy/tinyllamas/blob/main/stories260K/readme.md
3. Licences: llama2.c MIT, tinyllamas checkpoints MIT, TinyStories CDLA-Sharing-1.0; confirmed (GitHub API and LICENSE file, Hugging Face API metadata); https://github.com/karpathy/llama2.c/blob/master/LICENSE , https://huggingface.co/api/models/karpathy/tinyllamas , https://huggingface.co/api/datasets/roneneldan/TinyStories
4. runq.c is not integer-only (float RMSNorm, RoPE, softmax, SiLU, KV cache and scales; int8 only inside the matrix product; export group size 64, halved until it divides dim) and the README speeds of 4.6 and 14 tok/s for Llama-2 7B with 64 threads; confirmed; https://github.com/karpathy/llama2.c/blob/master/runq.c and https://github.com/karpathy/llama2.c#int8-quantization
5. llama2.c64: REU of at least 2 MB, "model weights (unchanged float32)", about 8 minutes per token, 62 tokens in 9:25:43, 37 to 40 s per token in emulator warp mode, int8 version left to contributors, README licence MIT; confirmed; https://github.com/ytmytm/llama2.c64 and https://github.com/ytmytm/llama2.c64/blob/main/STORY.txt
6. gbc-transformer: 16 forward passes in about 45 minutes, 0.0059 tokens/s, context 16, int8 weights with Q12 activations, 2026; confirmed, with one addition: the README labels the 45 minutes "Estimated elapsed time"; https://github.com/maddiedreese/gbc-transformer
7. elya-nes: 102,400 ternary weights, 4-bit activations, 1,117,248 cycles = 0.6243 s per token in MAME and Mesen, model shape, normaliser -0.029, 8 feed-forward sets -0.185 at +0.54% cycles, attention share 7.3% / 52.9%, and the withdrawn N64 and Genesis figures on the project page; confirmed; the context-85 penalty was restated: "+0.019 to +0.021" -> +0.019 mean (+0.021 and +0.017 per seed), plus +0.047 to +0.049 in a 12,000-step experiment; https://github.com/Scottcjn/elya-nes and https://hackaday.io/project/206401-transformers-on-retro-game-consoles
8. ESP32-S3 esp32-llm 19.13 tok/s; fork about 32.83 tok/s with the inconsistent "6 layers, 288 dimensions" text; confirmed; https://github.com/DaveBben/esp32-llm and https://github.com/eric-humane/esp32-llm
9. Four ESP32 boards: 2.33 / 11.12 / 21.41 / 22.02 tok/s with 0 / 4 / 8 / 8 MB PSRAM; confirmed; year "n.d." -> 2026 (published 2026-09-23); https://circuitdigest.com/articles/running-260K-tiny-llm-on-esp32
10. STM32H7A3 at 280 MHz, float32, up to about 87 tok/s including UART; confirmed (the README table calls it peak tok/s); https://github.com/schuhandreas/embedllama-stm32h7a3
11. DOS: 486DX2-66 2.08, Pentium MMX 200 15.32 (15M: 0.43), Pentium III 667 80.04 / 2.35 / 0.89 tok/s; confirmed; added the blog title and the 171.89 to 386.36 tok/s laptop rows that the file had left out; https://yeokhengmeng.com/2025/04/llama2-llm-on-dos/
12. Windows 98 Pentium II, 128 MB: 39.31 tok/s (260K), 1.03 (15M); confirmed; the 350 MHz tag "[UNVERIFIED: search summary]" -> confirmed as a press statement, and a second figure of 35.9 tok/s was added; https://blog.exolabs.net/day-4/ and https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-language-model-runs-on-a-windows-98-system-with-pentium-ii-and-128mb-of-ram-open-source-ai-flagbearers-demonstrate-llama-2-llm-in-extreme-conditions
13. PSP-2000: 260K, not quantized, 35.11 to 36.06 tok/s; numbers confirmed; corrected "at 333 MHz" -> clock during the test not stated (the paper only gives the PSP range of 1 to 333 MHz); year tag resolved to 2025 (citation date 2025-11-12); https://sol.sbc.org.br/index.php/eramiars/article/download/39470/39242 and https://sol.sbc.org.br/index.php/eramiars/article/view/39470
14. PS Vita PCH-1000 at 555 MHz: about 120 tok/s (260K), about 1.8 (15M); confirmed; https://github.com/callbacked/psvita-llm
15. Deeploy: 340 tokens/s and 490 microjoules per token, width 64, 8 layers, 16 heads, feed-forward 256, I-BERT softmax, Hardswish, 32-bit RMSNorm divisions, 360 MHz, 23x from the KV cache, paper licence CC BY-SA 4.0; confirmed (abstract and full text); https://arxiv.org/abs/2408.04413
16. HLSTransform 57.11 tok/s from system simulation, perplexity 2.9667 to 2.9679, 9 W; LlamaF 1.478 / 1.424 / 1.328 tok/s against 0.0935, perplexity 7.05 to 7.09, group size 256; confirmed (full texts); https://arxiv.org/abs/2405.00738 and https://arxiv.org/abs/2409.11424
17. ecp5-inference: 14.02 tok/s decode at 31.25 MHz, steps 6.8 / 8.53 / 10.96 / 14.02, 26,874 of 43,848 LUT4, 102 of 108 block RAMs, 17 of 72 DSPs, four 256-entry Q15 tables, int8 with power-of-two shifts, output equal to the float model for 9 tokens; confirmed; https://github.com/skyline624/ecp5-inference
18. ecp5-inference needs 1.70 M cycles per token = 6.1 cycles per weight byte; corrected: given as the figure of the design -> it is the state before the memory-read changes dated 31/07; after them 15.28 M cycles for 17 tokens = 0.90 M per token and 3.2 cycles per byte (simulation, our arithmetic); https://github.com/skyline624/ecp5-inference/blob/main/docs/CAPACITE_FPGA.md
19. fpga-tinystories (MAX 10 10M08): 55.6 / 50.3 / 45.5 tok/s at 60 MHz, 5,973 of 8,064 logic elements, six instruction types, int8 vectors, 24-bit accumulators, n + 11 clocks per element-wise pass, 48 instructions per layer, 0.622 -> 0.643 bits per character; confirmed; wording "4.7M-parameter ternary model" -> 4.72M parameters of which 4.20M ternary and 0.52M int8; added that the repository was one day old; https://github.com/MRazvan/fpga-tinystories and https://github.com/MRazvan/fpga-tinystories/blob/main/docs/VPU_PROGRAMMING.md
20. llama2.c-260k-rtl-fpga (XCVU9P): 235,704 LUTs, 142,128 flip-flops, 435 block-RAM tiles, 188 DSPs, group size 4, Q12.20, UQ8.24, 12-bit tables, no speed published; confirmed; https://github.com/wupd1122/llama2.c-260k-rtl-fpga
21. gateGPT: 32,872 / 10,192 / 2,757 / 1,156 cycles per token, 69,204 tok/s at 80 MHz, Q5.11, 16,548 LUTs, 62 DSPs; confirmed; https://github.com/fguzman82/gateGPT
22. arXiv citations: MCUFormer 73.62% with 320 KB; TinyFormer 96.1%, 1 MB, 320 KB, 12.2x; MCUBERT 5.7x and 3.0x, 512 tokens within 256 KB; tiny transformer kernels 4.79x, 2.0x, 1.53x, 25%; Nanhu-vdot over 4x and about 30%; identifiers 2610.08688 and 2607.03652 exist with the titles and authors given; confirmed (arXiv API); https://export.arxiv.org/api/query?id_list=2310.16898,2311.01759,2410.17957,2404.02945,2409.00661,2610.08688,2607.03652
23. The 34 GitHub repositories cited by the research pass exist and carry the licences stated in section 7; confirmed (GitHub API, license.spdx_id, one call per repository); https://api.github.com/repos/MRazvan/fpga-tinystories (example)
24. A transformer language model on a custom or homebrew 8-bit CPU with a multiply-accumulate instruction was not found; confirmed by a second search (6 web queries, 39 GitHub queries); coverage corrected: Soul Player C64, its ZX Spectrum and Amstrad CPC ports, 8bitGPT and ti84-llm-math (int8 or fixed-point transformers on stock 8-bit CPUs) had been missed and were added; https://github.com/gizmo64k/soulplayer-c64 , https://github.com/Timothy-Sikes/8bitgpt , https://github.com/RileyGreiff/ti84-llm-math
25. The XC7A100T has 63,400 LUTs; tag "[UNVERIFIED: device table not opened]" -> confirmed (15,850 slices x 4 LUTs, DS180 v1.17 Table 2); https://eng.auburn.edu/~nelson/courses/elec4200/FPGA/ds180_7Series_Overview.pdf

Not opened again in this pass (no tag in the text): Z80-uLM, JAM, eZ80-ai, semu-c64, vellm, the Dreamcast port,
ATTN/11, CraftGPT, esp32-ai, the three Pico projects, OpenLA500, PocketTPU, the Gemmini tutorial, dsp_inference,
TinyTransformer-FPGA, openTPU and its Tang Nano port, TALOS-V2 and kev-gpt. For those only existence and licence were
checked (claim 23). Nothing in the file was found to be invented. One unsupported number was removed (the PSP test
clock, claim 13), and three absolute statements ("Nobody measured ..." twice, "not measured at this size anywhere")
were reworded to "not found in our search".
