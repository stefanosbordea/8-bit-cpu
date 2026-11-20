# LS-LLM research summary

Compiled 2026-11-13 from the nine fact-checked topic files in `research/` (01 to 09). Nothing here goes beyond those
files: every number is copied from them, with the link they give. Tags are kept as written there: [VERIFIED] = re-read
at the source or recomputed by the fact-check; [UNVERIFIED: reason] = not supported; [cost model] / [estimate] /
[computed here] = our own arithmetic on the `research/CONTEXT.md` cost model (a row of n multiply-adds costs n + 24
cycles, element-wise work 12 to 32 cycles per element, about 1.0 M cycles per token), not simulated and not measured.
Section 11 (the ranking) is an editorial ordering of the ideas in the topic files, not a result. Disagreements between
topic files are listed in section 11.4. Details, tables and verification logs: the topic file named in each heading.

## 0. Executive summary

What the research changes about our plan:

1. Expect zero on HumanEval pass@1 in Tier A. The smallest published models with a score have 12 to 13 M parameters
   (2.00% and 1.5%); our extrapolation to 0.3 M gives 0.1 to 0.25% pass@1 (under one problem of 164) and 1.2 to 1.7%
   pass@100. Report pass@1/10/100, show zeros, add the easier ladder.
2. Speed is in rows and element-wise passes, not multiplies: a token is about 42% multiply-adds, 34% per-row control,
   24% element-wise work [cost model]. Fewer, wider heads (-34%), a one-pass table softmax (-19%), per-layer windows
   (-32 to -42%) and a prefill skip (-17% per task, exact) come before any multiplier lane.
3. No source runs our exact arithmetic (int8 residual stream, shift-only requantisation, 256-entry tables only). QAT
   must be bit-exact with the CPU; "+0.02 to +0.05 nats" is an assumption. Rounding by an accumulator preload is
   free and removes a -0.5 LSB bias per requantisation.
4. Context and tokenizer decide which tasks can score: at 2.5 bytes per token T = 256 holds about 62% of HumanEval
   tasks, T = 512 about 95%. Use vocabulary 512 or 1024 without byte fallback, never 256.
5. A grammar filter removes syntax errors, not wrong programs (6.8% to 6.9% pass@1 at 350 M). In our canonical
   layout it is a 512-byte table; report scores with and without it.
6. "Sparks of Weight-Space Programming" exists (a blog post). Its edit removed one dominant failure and added no
   ability: our editing phase becomes a repair stage with a paired, controlled protocol.
7. Training data: licence-gated human code, a 10-gram decontamination filter, no Claude or OpenAI outputs.
8. Hardware changes are small against the device (63,400 LUTs); judge them by cycles saved per hour of work.

## 1. Tiny code models: sizes and scores (`research/01_tiny_code_models.md`)

### Proven
- Codex size ladder on HumanEval, pass@1 / @10 / @100 (n = 200, nucleus p = 0.95): 12M 2.00 / 3.62 / 8.58; 25M 3.21 /
  7.1 / 12.89; 42M 5.06 / 8.8 / 15.55; 85M 8.22 / 12.81 / 22.4; 300M 13.17 / 20.37 / 36.27; 679M 16.22 / 25.7 / 40.95.
  Trained on 159 GB of Python from 54 M repositories, 100 B tokens; T = 0.2 is named optimal for pass@1 and T = 0.8 for
  pass@100; whitespace-run tokens give "approximately 30% fewer tokens". https://arxiv.org/abs/2107.03374 [VERIFIED]
- "12M" is a non-embedding count (section 3.3 of the Codex paper; labels 85M, 300M, 679M match 12 x layers x width^2 =
  84.9 M, 302.0 M, 679.5 M; GPT-3 lists the same shapes as 125M, 350M, 760M in total,
  https://arxiv.org/abs/2005.14165) [VERIFIED]. That the 12M total is larger by its token table is our inference
  [UNVERIFIED: our inference; the paper gives no layer shapes or total counts for the 12M, 25M and 42M models]. Which
  of Codex-12M and AlphaCode-13M is smaller in total is not established [UNVERIFIED: neither paper gives total
  parameter counts or layer shapes for these two models].
- AlphaCode decoder-only (10.7 + 43.6 GB of Python, SentencePiece 8,000, 1000 samples, highest over T 0.0 to 0.8):
  13M 1.5 / 3.6 / 8.6; 29M 3.4 / 5.8 / 11.2; 55M 4.2 / 8.2 / 16.9; 89M 4.3 / 12.2 / 20.0; 302M 11.6 / 18.8 / 31.8.
  https://arxiv.org/abs/2203.07814 [VERIFIED]
- pass@100 is 4.3 times pass@1 for Codex-12M and 5.7 times for AlphaCode-13M (taken at different temperatures)
  [VERIFIED].
- Same Codex table: GPT-Neo-125M 0.75 / 1.88 / 2.97; TabNine free 2.58 / 4.35 / 7.59; "GPT-3 solves 0%" [VERIFIED].
  CodeT5 (220M), CodeGPT-Adapted (124M), CodeClippy (125M) "all get 0%" at pass@1, @10, @100.
  https://arxiv.org/abs/2206.06888 [VERIFIED]
- Smallest public models with published scores: codeparrot-small (110M) 3.80 / 6.57 / 12.78
  (https://huggingface.co/codeparrot/codeparrot-small) [VERIFIED]; PyCodeGPT-110M 8.33 / 13.36 / 19.13
  (https://arxiv.org/abs/2206.06888) [VERIFIED]; PolyCoder-160M 2.13 / 3.35 / 4.88
  (https://arxiv.org/abs/2202.13169) [VERIFIED]; tiny_starcoder_py (164M) pass@1 7.84%, card "verified: false"
  (https://huggingface.co/bigcode/tiny_starcoder_py) [VERIFIED]; TinyCodeLM-150M pretrained 6.1% HumanEval and 5.4%
  MBPP(+), tuned 12.8 ± 2.6 / 20.6 ± 1.1 and 13.6 ± 2.1 / 24.4 ± 0.8; TinyCodeLM-400M 6.7 and 6.8, tuned 13.4 / 20.9
  and 19.4 / 29.9; SmolLM-135M-Instruct 7.7 ± 0.8 / 14.5 ± 1.0 and MBPP(+) 10.1 ± 1.8 / 14.6 ± 0.5
  (https://arxiv.org/abs/2410.02749) [VERIFIED].
- Data quality at 350 M: raw deduplicated Python 12.19% HumanEval after about 200 B tokens; classifier-filtered subset
  17.68% after 36k steps; plus synthetic textbooks 20.12%; fine-tuned on CodeExercises (under 200 M tokens)
  phi-1-small 45%. phi-1 (1.3 B) 50.6% HumanEval, 55.5% MBPP with under 7 B tokens.
  https://arxiv.org/abs/2306.11644 [VERIFIED]
- At or below 10 M: about 1 M-parameter nanoGPT models (6 layers, 6 heads, embedding 120, block 256, 41 characters) on
  grammar-generated TinyPy (1.02 M training snippets, 95.9 M character tokens): token accuracy 81.23%, line exact
  match 41.74%, edit similarity 74.15, output-prediction accuracy 74.58%; with a hybrid curriculum 81.13% / 42.04% /
  74.65 and 79.23% (82.84% easy, 80.79% medium, 74.04% hard). Checkpoints 4,385,396 or 4,386,378 bytes, Apache-2.0.
  https://arxiv.org/abs/2407.10194 [VERIFIED]
- Pico-1.3-Coding-Large (6,917,376 parameters, 13.8 MB): 68% on 50 simple tasks of the training kind (48 of 50 share a
  function name with training data), 10% (2 of 20) on unseen names; card: "expect about 0 %" on HumanEval.
  https://huggingface.co/Convergent-ch/Pico-1.3-Coding-Large [VERIFIED]
- Other hub cards (not re-checked): CHISPA-20M line exact match 1.2%, leading identifier 17.1%, 11.0% undefined names
  (https://huggingface.co/esraderey/chispa-20m-python); GoldWorm v8 (85.2 M, byte vocabulary 256) 7.08% pass@1 and
  16.67% pass@16 on its own 30 tasks, 92.92% valid, 0.42% on a harder set
  (https://huggingface.co/Uuuuuuniiiiiii/goldworm-v8-coder-85m); Karpathy's 10 M character LSTMs on 474 MB of C make
  "very few syntactic errors" (https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- Benchmark sizes in bytes (our computation, recomputed [VERIFIED]): HumanEval prompt median 396 (115 to 1,360), mean
  451.1; solution body median 149; prompt + solution median 576. 31 of 164 prompts fit 256 bytes, 118 fit 512; prompt
  + solution fits 512 bytes for 64 of 164, 768 bytes for 122, 1,024 for 146. 37 canonical bodies are one non-blank
  line. MBPP test: text + 3 asserts median 226; text + asserts + code fits 512 bytes for 372 of 500 and 768 for 470.
  Files: https://github.com/openai/human-eval (MIT), https://github.com/google-research/google-research/tree/master/mbpp
  (CC BY 4.0).
- MBPP few-shot with 244 M to 137 B models: "even the smallest models can write syntactically correct Python code
  around 80% of the time". https://arxiv.org/abs/2108.07732 [VERIFIED]
- Non-neural baselines (token prediction): Java n-gram 3.94 / 2.64 / 2.32 bits per token (static / dynamic /
  maintenance), cache 2.45 / 1.89 / 1.73, nested cache - / 1.70 / 1.41, LSTM/300 3.22 / 1.84, LSTM/650 3.03 / 1.91
  (https://web.cs.ucdavis.edu/~devanbu/isDLgood.pdf) [VERIFIED]. Python small set: n-gram 5.30 bits, MRR 43.63%;
  nested cache 2.89 bits, 65.97%; GRU-512 on BPE 3.91 bits / 81.66% static, 2.72 / 86.28% dynamic; n-gram models
  need 150 to 500 MB (small set) or 6 to 8.5 GB of disk and 5 GB to 50-60 GB of RAM; neural models 15 to 45 MB
  (https://arxiv.org/abs/2003.07914) [VERIFIED]. Neural reranker: 6 MB RAM, 8 ms, 90% top-5
  (https://arxiv.org/abs/2004.13651) [VERIFIED]. Post-training 8-bit LSTM: 152 MB to 38 MB, top-5 92% to 89%
  (https://arxiv.org/abs/1912.00742) [VERIFIED]. PY150 token accuracy: LSTM 58.00%, Transformer 73.26%, GPT-2 74.22%,
  CodeGPT 74.93%, CodeParrot 77.22%, PyCodeGPT 80.24% (https://arxiv.org/abs/2206.06888) [VERIFIED].
- At our memory scale, none a benchmarked code generator: stories260K (validation loss 1.2968, 1,056,540 bytes, MIT,
  https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) [VERIFIED]; Pico-1 (254,400 parameters, 513,984
  bytes) and Pico-1.2 (460,608 parameters, 929,376 bytes, token accuracy 43.8%, "says that it cannot write code",
  https://huggingface.co/Convergent-ch/Pico-1.2) [VERIFIED]; ternary.c64 (688,128 ternary weights, validation loss
  1.439, weights 242,324 bytes, cartridge 361,216 bytes, MIT, https://huggingface.co/gdm32569/ternary-c64-688k)
  [VERIFIED].
- Our extrapolation of the Codex and AlphaCode curves to 0.3 M parameters (40 times below the smallest point):
  pass@1 0.13% to 0.24%, pass@10 0.34% to 0.38%, pass@100 1.47% to 1.73% (logit-scale fit 0.12 to 0.21% and 1.17 to
  1.25%); one HumanEval problem is 0.61%. Recomputed [VERIFIED]; an extrapolation, not a measurement.

### Uncertain
- PyBlissa-Coder-40M claims 14.6% HumanEval (24/164, T = 0.25) and 4.4% MBPP (22/500); the 50M sibling 10.4%. Not
  reproduced; HumanEval is 3.3 times MBPP while the largest ratio among paper models is 2.2; training data is
  LLM-generated; no overlap check. https://huggingface.co/Rohanify/PyBlissa-Coder-40M-GGUF [VERIFIED as card text]
- theOG-50M v3 claims 42/164 (25.6%) HumanEval and 91/500 (18.2%) MBPP test
  [UNVERIFIED: self-reported hub card, not reproduced by anyone we found] https://huggingface.co/DrunkkToys/theOG-50M
- Hub cards can be wrong: a "128 Billion Parameter" card with 78.4% HumanEval has 206,208 parameters in hub metadata
  (https://huggingface.co/codefusionduo/suku-ai-128b-titan); a 29 M model reports 99.07% token accuracy
  (https://huggingface.co/MatterWaveAI/grt-30m-code). Every hub number is a lead, not a fact.
- A HumanEval zero is expected for Tier A; for MBPP the same is assumed by analogy [UNVERIFIED: no MBPP size curve
  was fitted; the smallest paper MBPP numbers in section 6.2 are at 135 M to 150 M parameters].
- The extrapolated value: Codex and AlphaCode used 54 to 159 GB of Python, long contexts and large vocabularies; a
  vocabulary-512, context-256 model is not on the same curve.
- Whether curated or synthetic instruction data helps at 0.3 M as at 350 M: evidence only at 350 M, 150 M, 40 M
  (self-reported) and 6.9 M (in-distribution).
- Curriculum at 1 M raised output prediction by 4.65 points and did not move completion (81.23% to 81.13%)
  [VERIFIED].
- Metric variants are not comparable (n, temperature rule, harness; "MBPP" = 500-task test split, sanitized 427/426,
  or MBPP+). In the standard MBPP prompt the three asserts shown are the scoring asserts, so sample-and-select is
  pass@k [VERIFIED]. PyCodeGPT-110M is 8.33 / 13.36 / 19.13 in the paper and 8.32 / 13.53 / 18.3 on its card
  [VERIFIED].
- flytech/python-codes-25k is MIT according to the PyBlissa card [UNVERIFIED: the dataset card metadata has no
  licence field, https://huggingface.co/datasets/flytech/python-codes-25k].
- 100 tokens/s used for the on-board sampling time [UNVERIFIED: 100 tokens/s is a design estimate in
  research/CONTEXT.md, not a measurement].

### Not tried at our scale
- No HumanEval or MBPP result below 12 M parameters in papers; smallest self-reported hub score at 39.9 M [VERIFIED:
  a second search found none].
- No Python generator at or below 1 M parameters trained on natural code and scored on functional correctness.
- No code model with int8 weights in 450 KB or less. Nearest sizes: 242 KB (ternary, stories), 514 KB (Pico-1, chat),
  1.06 MB (stories260K float32), 4.39 MB (1 M TinyPy float32).
- No n-gram or cache model scored on HumanEval or MBPP, none near 450 KB (published: 150 MB to 8.5 GB).
- No benchmarked code model with a sub-word vocabulary of 256 or 512; no HumanEval or MBPP run with a 256-token
  context (shortest found: 512, self-reported) [VERIFIED].
- Sample-and-select (k of 10 to 100) is documented only down to 12 M. Curriculum followed by natural short tasks is
  not reported at 1 M.

## 2. Integer-only inference (`research/02_integer_inference.md`)

### Proven
- Every published pipeline that was opened keeps something wider than int8 or divides: I-BERT (INT8 matmul, INT32
  nonlinearities, 22-bit residual sum into LayerNorm, integer division in softmax and square root; GLUE RoBERTa-Base
  86.0 to 86.3, Large 89.0 to 89.5, https://arxiv.org/abs/2101.01321) [VERIFIED]; I-ViT (16-bit residual adds, one
  integer division per softmax row and per LayerNorm row; DeiT-S 79.85 to 80.12, ViT-S 81.39 to 81.27, Swin-S 83.20
  to 83.01; L1 LayerNorm costs 2.49 and 3.32 points, https://arxiv.org/abs/2207.01405) [VERIFIED]; HAWQ-V3 (dyadic
  scales, INT32 residual add, ResNet50 77.72 to 77.58, https://arxiv.org/abs/2011.10680) [VERIFIED]; TFLite and
  CMSIS-NN int8 softmax (32-bit, count leading zeros, Newton reciprocal) [VERIFIED]; Deeploy's dim-64 TinyStories
  Llama (8-bit matmuls, I-BERT softmax, Hardswish, 32-bit divisions in RMSNorm, 340 tokens/s, 490 uJ/token, no
  accuracy number, https://arxiv.org/abs/2408.04413) [VERIFIED].
- Jacob et al.: ResNet-50/100/150 76.4/78.0/78.8% float against 74.9/76.6/76.7% integer; a tie-up rounding shift
  gives "an overall upward bias, which has been observed to cause significant loss of end-to-end accuracy".
  https://arxiv.org/abs/1712.05877 [VERIFIED]
- FQ-ViT: LayerNorm input channel ranges exceed 40x the median; a per-channel power-of-two factor (shift 0 to 3) lifts
  ViT-B from 23.64% to 83.31% (float 84.53%); 4-bit log2 attention 82.68%. https://arxiv.org/abs/2111.13824 [VERIFIED]
- I-LLM: static I-BERT-style W8A8 "can lead to catastrophic outcomes" on LLMs; LLaMA-7B perplexity 5.68 FP16, 5.84
  W6A6, 9.10 W4A4; softmax window c = 20 / 15 / 10: 9.15 / 9.16 / 9.23, no clipping 7,360,945.
  https://arxiv.org/abs/2405.17849 [VERIFIED]
- Table softmax without retraining. IntAttention: 32-entry uint8 exp table, window 6.6 nats, uint8 probabilities,
  exact row normalisation: Llama-3.2-1B perplexity 12.663 FP16, 12.784 table softmax alone, 13.070 whole pipeline
  (13.701 Quant-Only); stable for at least 16 entries and c in [5.5, 7.7]; HumanEval 32.93 / 32.32 / 31.10 and MBPP
  33.00 / 31.40 / 34.20 (FP16 / Quant-Only / pipeline); signed int8 P raises relative L1 error from 0.041 to 0.077.
  https://arxiv.org/abs/2511.21513 [VERIFIED]. Vasyltsov and Chang: WMT14 BLEU 26.98 float, 26.76 and 26.80 at
  uint8, 26.26 and 26.68 at uint4, 24.42 and 25.29 at uint2 (https://arxiv.org/abs/2111.10770) [VERIFIED]. SOLE:
  DeiT-T 72.21 to 71.33, DeiT-S 79.85 to 79.27, BERT-Base SQuAD 88.17 to 87.57 (https://arxiv.org/abs/2510.17189)
  [VERIFIED]. Kang and Wang: LLaMA2-7B W8A8 perplexity 5.4946 against 5.4943 (exp table), 5.4971 against 5.4940
  (log-domain division), 5.4956 (RMSNorm 1/sqrt by exponent table times 256-entry mantissa table)
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC12843610/) [VERIFIED].
- Softmax with fine-tuning: Softermax (base 2, 8-bit in and out, 16-bit internals) BERT-Base SQuAD 86.28 to 85.86,
  host accelerator has a 24-bit accumulator (https://arxiv.org/abs/2103.09301) [VERIFIED]; ITA uses 8 power-of-two
  levels, probability error 0.46% (I-BERT 0.35%), 24-bit accumulator "to allow up to 256-element dot products"
  (https://arxiv.org/abs/2307.03493) [VERIFIED].
- NN-LUT: an equally spaced 1/sqrt table destroys RoBERTa (CoLA 62.1 to 4.6, MRPC 87.5 to 57.5); learned breakpoints
  with a power-of-two pre-scale keep it (60.4, 86.5); table calibration lifts INT32 GLUE 84.1 to 85.1 (float 85.4).
  https://arxiv.org/abs/2112.02191 [VERIFIED]
- Attention without normalisation, in float: ConSmax (6 layers, 6 heads, width 384, context 256): perplexity 2.3%
  higher early, under 0.9% after 10K iterations, converged about 20K (https://arxiv.org/abs/2402.10930) [VERIFIED];
  sigmoid attention with b = -log(n) at 1 B: 49.5 against 49.4; at n = 4096 48.2 against 49.2 without an extra norm
  (https://arxiv.org/abs/2409.04431) [VERIFIED]; Integer Transformer: polynomial attention and L1 LayerNorm lose
  nothing in float (31.28 to 31.20, 31.18), full INT8 31.61 against 31.18, about 0.3 BLEU, over 1 BLEU on En-Fr
  (https://arxiv.org/abs/2009.08034) [VERIFIED].
- Norm variants: Dynamic Tanh keeps LLaMA 7B to 70B training loss (1.59 to 1.60, 1.53 to 1.54, 1.50, 1.45), ViT-B
  82.3 to 82.5, hardtanh 82.2, fails on ResNet-50 BatchNorm (76.2 to 68.9) (https://arxiv.org/abs/2503.10622)
  [VERIFIED]; MobileBERT NoNorm GLUE 78.5 to 77.7, latency 192 ms to 62 ms, 8-bit post-training then free
  (https://arxiv.org/abs/2004.02984) [VERIFIED]; FlashNorm: RMSNorm weights fold into the next linear layer
  (https://arxiv.org/abs/2407.09577) [VERIFIED against the abstract].
- Power-of-two scales: TQT matches float on MobileNet v1 (71.1 to 71.1) and v2 (71.7 to 71.8) with trained
  thresholds; without training 0.6%; weights only 67.0 (https://arxiv.org/abs/1903.08066) [VERIFIED]. Legacy
  CMSIS-NN: shift-only requantisation, CIFAR-10 CNN 80.3% to 79.9%, accumulator starts at
  `(bias << bias_shift) + NN_ROUND(out_shift)` (https://arxiv.org/abs/1801.06601) [VERIFIED].
- Residual-stream precision, BERT-base GLUE: 83.06 float; 71.03 with 8-bit post-training; 83.23 weights only; 82.43
  with 16 bits on the FFN residual sum; 83.26 with 8-bit QAT (https://arxiv.org/abs/2109.12948) [VERIFIED]. Clipped
  softmax cuts BERT's largest activation from 735 to 21.5 and W8A8 perplexity from 1294 to 4.52 (float 4.49)
  (https://arxiv.org/abs/2306.12929) [VERIFIED].
- A2Q overflow bound: no overflow if sum|x||w| <= 2^(P-1) - 1 (https://arxiv.org/abs/2308.13504) [VERIFIED].
- Our calculations [computed here] [VERIFIED: script re-run]: floor shift mean error -0.4965 LSB at shift 7, round
  half up +0.0043 LSB, both standard deviation 0.289; after 11 floor roundings a -5.50 LSB offset. 24-bit
  accumulator: at most 511 products of int8 operands, 516 if weights avoid -128, 520 if both do; sum of squares of 64
  int8 values at most 2^20; a 256-key value mix with weights 0..127 is at most 4,161,536, with 0..255 8,355,840
  (below 8,388,607). Softmax reciprocal: bare power of two 29% rms (19% after folding the constant 1.337), one
  mantissa bit 10.3%, 4-bit mantissa 2.6%, 8-bit mantissa with 7-bit entries 0.38% rms and 0.9% worst. 1/sqrt of a
  20-bit sum: leading-one detection with even shift 0.85% worst (7-bit entries), 0.56% (8-bit); one fixed shift 1.5%
  worst over 4x of RMS, 5.7% over 8x. int8 residual stream after 10 additions 30.4 dB against 40.9 dB for one
  rounding (about 1.7 bits), 66.6 dB with 16-bit storage.
- Hobby projects closest to the LS scheme (single author, no independent reproduction): MAX 10 TinyStories (16
  layers, width 128, 4.72M parameters, int8 stream, integer divisions, Q15 softmax weights, multiply-then-shift):
  0.622 to 0.643 bits per character for all integer conversions (https://github.com/MRazvan/fpga-tinystories)
  [VERIFIED]; NES elya-nes (ternary weights, 4-bit activations, QAT with the integer forward pass): exact normaliser
  against power-of-two shift 1.3766 against 1.4058 nats per character (https://github.com/Scottcjn/elya-nes)
  [VERIFIED]; ECP5 stories260K (int8 times power of two, four 256-entry tables with 16-bit entries): output equals
  float for 9 tokens, then differs, no loss (https://github.com/skyline624/ecp5-inference) [VERIFIED].

### Uncertain
- Whether an 8-bit residual stream is enough: no for BERT post-training (71.03), yes with QAT (83.26); the wide-stream
  papers show no ablation that 8 bits fails under QAT; all of it post-LN or vision. One pre-norm data point: the
  MAX 10 project (0.021 bits per character in total) [VERIFIED].
- Dynamic Tanh outside the original paper: +27.3% better at 1M training tokens, 18.8% worse at 118M, about 33%
  per-seed collapse with SwiGLU (https://arxiv.org/abs/2604.23434) [VERIFIED]; "remain fragile at depth"
  (https://arxiv.org/abs/2601.09719) [VERIFIED against the abstract]; Derf 0.31 nats behind RMSNorm in one study
  (https://arxiv.org/abs/2604.01563) [VERIFIED against the abstract]. Smallest LLaMA width with a stated initial alpha
  is 1024.
- Clipped softmax hurts a decoder: OPT-125m W8A8 perplexity 21.18 vanilla, 37.20 clipped, 16.02 gated [VERIFIED]. An
  exp table with exact zeros behaves like a clipped softmax.
- L1 norm: harmless from scratch (31.28 to 31.18), 2.5 to 3.3 points worse when swapped in and fine-tuned [VERIFIED].
- Per-tensor power-of-two scales on the residual stream of a from-scratch QAT transformer: open. The ECP5 code notes
  that such a scale "loses up to 1 bit of precision" and "~30-40%" cumulative error over 35 chained matmuls
  (post-training) [VERIFIED].
- Floor against round under bit-exact training: that Jacob et al.'s damage came from a train/inference mismatch is
  our reading [UNVERIFIED: the paper describes neither the experiment nor the training setup]. The baseline has no
  bias parameters, so the offset can only be absorbed by weights. The design-question table of the topic file
  repeats it: that inference differed from the training simulation there is our reading [UNVERIFIED: not stated
  in the paper].
- Softmax precision: our random-logit test gives 1.3% (1 nat), 4.2% (2 nat), 2.7% (4 nat) total-variation distance for
  exp outputs 0..127 with 256 keys, about 0.5% with a two-byte exp value [computed here].
- ConSmax and sigmoid attention in causal generation: the number of keys grows from 1 to T, which a constant
  normaliser does not follow.
- IntAttention's loss depends on the model: Qwen3-1.7B 23.01 to 27.75 perplexity (32.95 with float softmax on int8
  Q/K/V; 22.36 with the table softmax alone) [VERIFIED].
- A power-of-two-only LayerNorm for CLIP-ViT on FPGA (DATE 2026) [UNVERIFIED: the PDF URL returned HTTP 404 on
  2026-11-13, again in the fact-check; the claim comes from search snippets only, and one snippet attributes the
  power-of-two terms to the GELU slopes and a log-based approximation to the reciprocal square root, so
  "power-of-two-only LayerNorm" may be inexact].
- TinyStories-class models on ESP32 with int8 weights and float activations [UNVERIFIED: blog and model-card snippets
  only, not opened for this file; the ESP32 ports were opened for `research/04_tiny_hardware.md`].

### Not tried at our scale
- An int8 residual stream at width 64 with int8 weights and QAT of the activations: not found.
- Shift-only requantisation in a transformer language model with accuracy numbers: not found (CNN evidence only).
- Softmax normalisation with only int8 intermediates and a table reciprocal: not found.
- Dynamic Tanh, Derf, sigmoid attention, ReLU attention, ConSmax under 1M parameters: not found (smallest: ConSmax
  6 x 384; sigmoid attention from d_model 64 and 2 layers, 2.2M non-embedding; DyT language model at 64M).
- A fixed-shift 1/sqrt table with training that holds the RMS in the table window: not found.
- Overflow study for language-model rows longer than 511 at 24 bits: not found beyond the A2Q bounds.

## 3. Quantization-aware training (`research/03_qat_small_models.md`)

### Proven
- Cost of 8-bit weights + 8-bit activations in papers that tune their QAT: -0.005 to +0.03 nats per token on language
  models, within about +/-0.3 GLUE points [VERIFIED]. Prato et al. (activation quantizers in the loop; PyTorch
  example model, 2 layers, hidden 200): WikiText-2 loss 5.65 to 5.64, WikiText-103 5.91 to 5.94 (+0.023 nats); BLEU
  26.46 to 26.38 and 38.34 to 38.41 at 8 bits, 18.32 / 1.59 at 4 bits; its 4-bit language model scores the same, so
  it is weak evidence (https://arxiv.org/abs/1910.10485) [VERIFIED]. DiffQ Table 2 (16 layers, 942 MB): PPL 18.1
  float, 18.3 "8/8" (+0.011), 19.7 "QAT 8/8" (+0.085); QAT weights with float activations 18.2 (+0.006)
  (https://arxiv.org/abs/2104.09987) [VERIFIED]. Quant-Noise: 18.3 float, 19.6 int8 after training (+0.069), 21.0
  plain QAT (+0.138), 18.7 Quant-Noise (+0.022) (https://arxiv.org/abs/2004.07320) [VERIFIED]. In both 942 MB results
  the activations are quantized after training.
- BERT-base GLUE: float 83.06, W8A8 QAT 83.26, W8A8 post-training 71.03, W4A8 QAT 82.64, with 2-bit token embeddings
  82.29; weights only 83.23, activations only 70.25 (https://arxiv.org/abs/2109.12948) [VERIFIED]. Leave-one-out: the
  residual sum after the feed-forward block is the sensitive tensor (81.57 / 82.56 / 89.73 / 67.15 against all
  quantized 62.64 / 42.67 / 50.74 / 48.74); fixes: 16 bits there (82.43), 6 per-embedding-group ranges (82.45), or
  QAT (83.26) [VERIFIED].
- Q8BERT: CoLA 58.48 to 58.48, RTE 69.7 to 68.78, SST-2 92.36 to 92.24 (https://arxiv.org/abs/1910.06188)
  [VERIFIED]. TernaryBERT 8-8-8: MNLI 84.2/84.7 against 84.5/84.9 (https://arxiv.org/abs/2009.12812) [VERIFIED].
  LLM-QAT, LLaMA-7B: float16 66.2; 8-8-8 round-to-nearest 65.6, LLM-QAT 66.0; 8-8-4: 47.4 and 61.6; 4-8-4: 41.2 and
  60.7 (KV cache at 4 bits is not harmless) (https://arxiv.org/abs/2305.17888) [VERIFIED]. MobileLLM-125M W8A8
  post-training 45.0 to 44.8 (https://arxiv.org/abs/2402.14905) [VERIFIED].
- Model cards (self-reported): ESP32-S3 28.9 M model, int8 activations +0.0003 nats (2.4793 to 2.4796)
  (https://huggingface.co/slvDev/esp32-ai-tinystories) [VERIFIED]; 11.16 M model, int8 embedding +0.0006 nats, ternary
  embedding +0.1347; float32 2.0553, ternary QAT 2.1760 (+0.12), ternary after training 5.0229
  (https://huggingface.co/euclidstellar/tinystories-1bit-llm) [VERIFIED]; 999,328-parameter int8 model
  "near-lossless", 0.625 bits per byte against 0.707 for TinyStories-1M, distilled, no run without distillation
  (https://huggingface.co/sprapp/eeny-tinystories-999k) [VERIFIED].
- llama2.c `runq.c`: int8 weights for matmuls only, RMSNorm in float32 "because these layers are very sensitive"
  (https://github.com/karpathy/llama2.c/blob/master/README.md) [VERIFIED].
- TQT recipe: log2 thresholds trained with the straight-through estimator; Adam, learning rate 1e-2 for thresholds
  and 1e-6 for weights; weight thresholds at 3 standard deviations; thresholds frozen one by one because they
  oscillate around an integer log2 level; round-half-to-even; MobileNet v1 67.0 with fixed thresholds against 71.1
  with trained ones in 2.1 epochs (https://arxiv.org/abs/1903.08066) [VERIFIED]. LSQ: 8-bit networks fine-tuned for
  1 epoch at learning rate 0.001 (float 0.1), input and output layers always 8 bits
  (https://arxiv.org/abs/1902.08153) [VERIFIED]. White paper: start from a float model, learn ranges, Adam
  (https://arxiv.org/abs/2106.08295) [VERIFIED].
- Start step, Transformer base EN-DE BLEU: never 26.46; step 100: 26.98; 10,000: 26.63; 50,000: 26.84; 80,000:
  26.41; after training 25.50; the authors: "quantizing the model early on seems preferable"; spread 0.57 BLEU, about
  the run-to-run deviation (0.09 to 0.51) [VERIFIED].
- Oscillation: damage at 8 bits 0.10 points (MobileNetV2, 71.79 against 71.89), 2.02 at 4 bits, 4.53 at 3 bits; no
  transformer experiment (https://arxiv.org/abs/2203.11086) [VERIFIED].
- Fewer bits: small BitNet language models (6 M to 48 M) need about twice the hidden size
  (https://arxiv.org/abs/2407.09527) [VERIFIED]; scaling law over 465 runs: compute-optimal precision about 7 to 8
  bits, below 4 bits needs over 4x the parameters, post-training degradation grows with tokens per parameter
  (https://arxiv.org/abs/2411.04330) [VERIFIED]; smaller and longer-trained Pythia models lose more from low-bit
  post-training quantization (https://arxiv.org/abs/2411.17691) [VERIFIED].
- Distillation into the quantized student from its float model: LSQ ResNet-18 at 2 / 3 / 4 / 8 bits 67.9 / 70.6 /
  71.2 / 71.1 against float 70.5 [VERIFIED]; TernaryBERT MNLI 83.3 with, 80.8 without [VERIFIED]; LLM-QAT: logit
  distillation beats label training, hidden-state distillation hurts [VERIFIED].
- Distillation from a larger teacher: Baby Llama (58 M student beats both teachers on 10 M words; 0.5 CE + 0.5 KL,
  temperature 2; https://arxiv.org/abs/2308.02019) [VERIFIED]; MobileLLM: 43.9 to 43.8 at 29 h to 93 h (125 M), 49.1
  to 48.8 at 42 h to 109 h (350 M) [VERIFIED]; DistilGPT2 perplexity 21.1 against 16.3 for GPT-2
  (https://huggingface.co/distilbert/distilgpt2) [VERIFIED].
- Data side: TinyStories (models "below 10 million total parameters" write fluent text,
  https://arxiv.org/abs/2305.07759) [VERIFIED]; deduplication gives "fewer train steps to achieve the same or better
  accuracy" (https://arxiv.org/abs/2107.06499) [VERIFIED]; duplicated code inflates metrics by up to 100%
  (https://arxiv.org/abs/1812.06469) [VERIFIED]; curriculum "largely unsuccessful" in BabyLM 2023
  (https://arxiv.org/abs/2504.08165) [VERIFIED].
- Our arithmetic [VERIFIED: recomputed]: 511 x 16384 = 8,372,224 <= 8,388,607 (512 overflows by one; 516 with
  weights in [-127, 127]). A float32 matrix product of int8-valued tensors equals the integer result exactly for row
  lengths 64, 172, 256, 511 (numpy and torch 2.15.0 on MPS), so the bit-exact forward pass can run on the laptop GPU.
  Floor shift: mean -0.50 LSB, RMS 0.577; with a 2^(k-1) preload mean +0.001, RMS 0.289; error power 0.333 to 0.083
  LSB^2 (4x). Power-of-two step on Gaussian data: MSE-optimal clip 3.92 standard deviations (SQNR 40.54 dB); the
  restriction costs 1.62 dB on average, 3.90 dB worst (0.27 and 0.65 bit). Effective parameters at 8 bits (Kumar et
  al. form): 0.942 for weights alone, 0.836 or 0.924 depending on the constants; illustration only.

### Uncertain
- Whether "8 bits is free" transfers to 260k to 400k parameters: every language-model number comes from 11 M
  parameters or more. For LS-LLM +0.02 to +0.05 nats is a planning assumption [UNVERIFIED: no source measures a loss
  at this scale or with static power-of-two scales and int8-indexed tables; the true cost may be larger]. The
  results table of the topic file carries the same row: LS-LLM, 260k to 400k, not trained yet, planning
  assumption +0.02 to +0.05 nats [UNVERIFIED: no source at this scale].
- PACT (learned activation clipping level, https://arxiv.org/abs/1805.06085): its hyper-parameters, the initial
  alpha and its regulariser [UNVERIFIED: paper body not opened, abstract only].
- Whether plain straight-through QAT helps at 8 bits: the two negative baselines train only the weights; our reading
  is that they argue for activations in the training loop.
- Whether oscillation matters at 8 bits in a language model: nothing measured.
- Learned ranges against statistics; symmetric against asymmetric activations; when to switch quantization on
  (early start "seems preferable" in Prato et al.; LSQ, TQT and the white paper start from float; no 8-bit language
  model result); weight decay during QAT: sources point in different directions.
- Cost of the power-of-two restriction after training: zero on MobileNets, 0.27 bit on Gaussian data before
  retraining, not measured for a transformer language model.
- Usable softmax range with an 8-bit table output: about 5 nats (ln 127 = 4.8 to ln 254 = 5.5); whether a tiny model
  needs more is unknown.
- Distillation from a larger teacher: positive when data-limited (BabyLM), no gain in MobileLLM; our setting is the
  MobileLLM side. Curriculum: mostly negative.

### Not tried at our scale
- A QAT study of an autoregressive language model at or below 1 M parameters with W8A8: not found. Nearest: Deeploy
  (4,621,392 parameters, about 0.53 M outside the embedding tables, post-training TQT, no loss) [VERIFIED]; Z80-uLM
  (2-bit weights, not a transformer, no loss; https://github.com/HarryR/z80ai).
- Static power-of-two scales end to end in a transformer language model; an 8-bit-only path for exp, 1/sqrt or SiLU
  tables; an 8-bit residual stream with one fixed scale through all layers: not found.
- Oscillation in any language model; controlled distillation below 1 M parameters; a tied input/output embedding
  under one int8 table (no source discusses the conflict of scales); controlled study of canonical code formatting
  or identifier normalisation below 1 M: not found.

## 4. Tiny hardware (`research/04_tiny_hardware.md`)

### Proven
- stories260K: dim 64, hidden 172, 5 layers, 8 heads, 4 KV heads, vocabulary 512, max_seq_len 512, tied classifier;
  260,032 parameters = 32,768 + 5 x 45,440 + 64; 259,328 take part in matrix products; file 1,056,540 bytes;
  validation loss 1.2968; MIT. https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K [VERIFIED]
- `runq.c` is not integer-only: only matmul weights are int8 (Q8_0, one float32 scale per 64 weights); norms,
  residual stream, KV cache, RoPE, softmax, SiLU stay float32. Llama-2 7B: 4.6 tok/s float32, 14 tok/s int8.
  https://github.com/karpathy/llama2.c/blob/master/runq.c [VERIFIED]
- stories260K speeds in float32: ESP32-S3 19.13 tok/s (fork: about 32.83), four ESP32 boards 2.33 / 11.12 / 21.41 /
  22.02, STM32H7A3 "up to ~87 tok/s", 486DX2-66 2.08, Pentium MMX 200 15.32, Pentium III 667 80.04, Pentium II 39.31,
  PSP-2000 35.11 to 36.06, PS Vita at 555 MHz about 120, Commodore 64 about one token every 8 minutes (547 s per
  token from the author's log) [VERIFIED]. Sources in table B (section 10).
- Integer-only runs on 8-bit CPUs (all 2026): Game Boy Color 0.0059 tok/s (context 16); Soul Player C64 about 25,000
  int8 parameters, per-tensor power-of-two shifts, QAT, about 60 s/token, no loss; 8bitGPT about 30 s/character; NES
  elya-nes 1,117,248 CPU cycles per token = 0.6243 s, 10.688 cycles per multiply-add, trainer equals the 6502 on
  1,216 of 1,216 tokens (emulator counts) [VERIFIED].
- NES ablations (nats per character, two seeds): exact softmax normaliser -0.029; 8 token-selected feed-forward sets
  -0.185 at +0.54% cycles; context 85 instead of 20 +0.019 (worse) at +52% cycles; attention 7.3% of a token at
  context 20, 52.9% at 85. https://github.com/Scottcjn/elya-nes [VERIFIED]
- Deeploy on Siracusa: 340 tokens/s, 490 uJ per token; with the KV cache a 256-step run is 23x faster (0.75 s
  against 17.6 s). https://arxiv.org/abs/2408.04413 [VERIFIED]
- FPGA designs: stories260K as RTL on an XCVU9P uses 235,704 LUTs, 142,128 flip-flops, 435 block-RAM tiles, 188
  DSPs, no speed published (https://github.com/wupd1122/llama2.c-260k-rtl-fpga) [VERIFIED]. ECP5-45F: 14.02 tok/s at
  31.25 MHz, context at most 32; steps 6.8, 8.53, 10.96 (dot product pipelined from 256 to 68 cycles per 64-element
  row), 14.02 (feed-forward weights into block RAM); 1.70 M cycles per token = 6.1 cycles per weight byte (about 1.5
  from block RAM, about 15 from SDRAM), then 0.90 M and 3.2 after memory-read changes (simulated)
  (https://github.com/skyline624/ecp5-inference) [VERIFIED]. MAX 10 vector processor (six instruction types, int8
  vectors, 24-bit accumulators, about 4 clocks of overhead per instruction, one element per clock, 48 instructions
  per layer; 5,973 logic elements): 55.6 tok/s at 64 tokens, 50.3 at 256, 45.5 at 480 at 60 MHz; integer conversion
  0.622 to 0.643 bits per character = 1.647 to 1.704 nats per token (https://github.com/MRazvan/fpga-tinystories)
  [VERIFIED]. gateGPT: 32,872 cycles per token recomputing, 10,192 with a KV cache, 2,757 with 24 lanes, 1,156 final
  (https://github.com/fguzman82/gateGPT) [VERIFIED].
- Soft cores: OpenLA500 on XC7A200T about 0.50 tok/s in software, 4.78 falling to 1.90 with a float coprocessor
  (https://github.com/WizLiang/LLaMA2-on-OpenLA500); Rocket + Gemmini: matrix products 75x faster, whole run about
  10x (https://mikutyan4.github.io/chipyard-linux-nexys/ch07-llm-inference.html); Nanhu-vdot: dot products over 4x,
  GPT-2 about 30% faster (https://arxiv.org/abs/2409.00661) [VERIFIED].
- Baseline per token at context 256: 3,512 matrix rows, 10,240 attention-score dot products of length 8, 10,240 exp
  lookups, 320 value-mixing streams, 423,168 multiply-adds [VERIFIED: counts recomputed].
- Memory at int8 (our arithmetic): weights 260,032 bytes; KV cache 320 bytes per position (81,920 at T = 256,
  163,840 at T = 512); about 345 KB of 450 KB at T = 256.
- Device for scale: XC7A100T has 63,400 LUTs, 135 block RAMs of 36 Kb, 240 DSP48E1 [VERIFIED].

### Uncertain
- Whether shift-only requantisation costs quality at 260K: indirect and mixed (ECP5 "loses up to 1 bit", "~30-40%"
  cumulative error, post-training; NES shift normaliser 0.029 nats/char worse; Soul Player C64 works with int16
  activations and reports no loss) [VERIFIED].
- Whether an int8 residual stream is enough: MAX 10 loses 0.057 nats/token in total; no data for width 64, 5 layers,
  int8 weights.
- NES figures are emulator cycle counts; the same author withdrew earlier claims (N64 "60 tok/s" was 1.23; Genesis
  11.3x was 1.674x) [VERIFIED]. HLSTransform timing is from simulation [VERIFIED]. Speeds are not comparable across
  rows (timing scope differs).
- Whether a long context pays at this size: on stories the NES model got worse at context 85; may not transfer to
  code.
- The 32.83 tok/s ESP32 fork is not confirmed by a second source [VERIFIED]. The Pentium II clock (350 MHz) rests on
  press coverage; two speeds exist (39.31 and 35.9 tok/s) [VERIFIED].
- `runq.c` default group size 64 does not divide the hidden size 172 (our reading, not executed): do not take
  default `runq` output as an int8 reference.
- Our 1.0 M cycles per token: the ECP5 design needs about 0.9 M at a context of at most 17; the LS number is the
  more optimistic and holds only with every weight in block RAM and one-cycle reads; Tier B through DDR2 is unproven.
- The hardware ideas lean on young single-author repositories (MAX 10 repository created 2026-11-12) [VERIFIED].
- Commodore 64 clock in table B [UNVERIFIED: clock not stated in the README, standard C64 value]. TALOS-V2
  [UNVERIFIED: clock and parameter count are from a third-party report, not the README]. Catalogue entries (Wii
  61.9 and 12.7 tok/s, 3DS 18.46 / 23.28, KV260 about 5) [UNVERIFIED: numbers copied from the list, primary pages
  not read] https://github.com/emrehangorgec/awesome-llm-on-anything
- Ideas 1 + 5 + 6 of that file together give roughly 0.5 M cycles per token (about 200 tokens/s at 100 MHz)
  [UNVERIFIED: an estimate on the CONTEXT.md cost model, not simulated or measured; the fact-check pass recomputed
  the sum as about 0.46 M cycles under the same assumptions.]

### Not tried at our scale
- A transformer language model on a custom or homebrew 8-bit CPU with a multiply-accumulate instruction: not found
  in two independent searches [VERIFIED].
- A stories260K-class model with int8 activations everywhere, fixed power-of-two shifts, 256-entry tables only, no
  division, QAT, and a reported loss: not found.
- Any of these tiny systems writing program code or reporting HumanEval or MBPP: not found.
- int8 x int8 streams under a 24-bit accumulator (the MAX 10 design has ternary weights); full context on a tiny
  integer-only machine with a 260K model (16, 16, 20, 20, 32 tokens in the designs found); stories260K in integer
  code on a soft core; token-selected feed-forward sets with a 256 to 512 BPE vocabulary and code data: not found.

## 5. Speed techniques (`research/05_speed_techniques.md`)

### Proven
- Cost model, baseline at T = 256 [cost model; VERIFIED as arithmetic]: 56% of every layer is attention scores and
  the softmax pass, 23% the SwiGLU block; the output layer is 5% of a token. Single changes: 8 x 8 heads to 2 x 32
  -34% cycles per token (-43% at T = 512), 4 x 16 -23% (-28%); exp lookup fused into the score loop -19% (-23%);
  windows of 32 in the lower 3 or 4 layers -32% to -42% (-42% to -57%); d = 96 x 3 layers or d = 128 x 2 layers -36%
  to -41% at 354k to 361k parameters; prefill skip -23% per prompt token and -17% per median HumanEval-sized task
  (1.56 s to 1.30 s; prompt processing is 1.00 s of the 1.56 s); MQA -2% time but -60 KiB (T = 256) or -120 KiB
  (T = 512) of KV cache; ReGLU / ReLU block / hidden 128: -1% / -4% / -6%; greedy argmax about 4k cycles against
  about 10k for sampling and about 74k for sort-based top-p; one row instruction (24 to 8 cycles per row) -21%.
  Stacked heads + one-pass softmax + windows: 498k / 546k / 642k cycles per token at T = 128 / 256 / 512 = 201 / 183
  / 156 tokens per second against 130 / 91 / 57 for the baseline. None has quality evidence at our size.
- Constant-cost recurrent mixers (minGRU, SRU, GRU): 0.40 to 0.62 M cycles at any T, no KV cache [cost model].
  Mamba about 253k cycles per block, xLSTM about 197k plus a division, RWKV-4 about 97k per layer plus a division
  per channel [rough cost model]: do not pursue.
- Heads: 65 M Transformer, 1 / 4 / 8 / 16 / 32 heads: PPL 5.29 / 5.00 / 4.92 / 4.91 / 5.01, BLEU 24.9 / 25.5 / 25.8 /
  25.8 / 25.4 (https://arxiv.org/abs/1706.03762) [VERIFIED]. TinyStories width 768, 2 layers with 2 / 4 / 8 heads:
  1.38 / 1.34 / 1.33; 1 layer 1.58 / 1.56 / 1.54 (https://arxiv.org/abs/2305.07759) [VERIFIED]. MobileLLM 125 M, 8 /
  16 / 32 heads: 43.8 / 44.6 / 44.3 (https://arxiv.org/abs/2402.14905) [VERIFIED]. Pruning: 20% (WMT) and 40% (BERT)
  of heads removable (https://arxiv.org/abs/1905.10650). An 818k hobby model: 4 heads 1.93, 1 head 1.91
  (https://huggingface.co/delcenjo/char-transformer-shakespeare) [VERIFIED].
- MQA/GQA: 211 M translation, multi-head 1.424 / 26.7 against multi-query 1.439 / 26.5; shrinking heads instead is
  worse (1.480 to 1.518) (https://arxiv.org/abs/1911.02150) [VERIFIED]; T5-XXL MHA 47.2, MQA 46.6, GQA-8 47.1
  (https://arxiv.org/abs/2305.13245) [VERIFIED]; MobileLLM 16 heads with 16 / 8 / 4 / 2 / 1 KV heads: 44.6 / 44.3 /
  44.7 / 43.5 / 43.7 (125 M) [VERIFIED].
- Vocabulary [measured here; VERIFIED within 3%]: SentencePiece BPE on MBPP code: vocabulary 256 2.10 bytes per
  token; 512 with byte fallback 2.04; 512 without 2.53; 1024 2.82 (re-run: 2.04 / 1.99 / 2.47 / 2.74). With the task
  text: 2.18 / 2.12 / 2.74 / 3.18. Per task 256 is about 24% slower than 512, 1024 about 8% faster, bytes about 4
  times slower. HumanEval prompt + solution median 574 bytes, p90 1053, p95 1254: T = 256 holds about 62% of tasks at
  2.5 bytes per token, T = 512 about 95% (57% to 71% and 94.5% to 97.6% for 2.4 to 2.7 bytes per token).
- ByT5: byte level is 1.5 to 2.6 times slower (transliteration), 3.7 to 6.4 (XSum), 6.4 to 9.5 (XNLI)
  (https://arxiv.org/abs/2105.13626) [VERIFIED]. Code BPE with 2k / 5k / 10k merges: 4.90 / 4.78 / 4.77 bits per
  token on a small set (https://arxiv.org/abs/2003.07914) [VERIFIED].
- Tied embeddings: 4.65 M to 2.65 M parameters with perplexity 114.5 to 112.4
  (https://arxiv.org/abs/1608.05859) [VERIFIED]; MobileLLM 44.8 to 44.6 for -11.8% parameters [VERIFIED].
- Windows: decoder window of 32 at 211 M: 1.427 against 1.424, BLEU 26.6 against 26.7 [VERIFIED]; Longformer windows
  growing 32 to 512 with depth 1.21 bpc against fixed 1.23 and shrinking 1.24 (https://arxiv.org/abs/2004.05150)
  [VERIFIED]; a 4-layer LSTM with one single-head attention layer 1.076 bpc against 1.330 without and 1.06 for
  Transformer-XL (https://arxiv.org/abs/1911.11423) [VERIFIED]; code n-gram entropy falls from 4.81 to 2.89 bits with
  a nested cache on Python [VERIFIED].
- Early exit: CALM oracle uses 1.53 of 8 layers (https://arxiv.org/abs/2207.07061) [VERIFIED]; LayerSkip HumanEval
  15.9 at the last layer, 4.88 at the middle layer (baseline 13.4 and 0); Llama3 8B 37.8 baseline against 28.7 with
  LayerSkip training (https://arxiv.org/abs/2404.16710) [VERIFIED].
- Depth and width, TinyStories eval loss (2 / 4 / 8 / 12 layers): hidden 64: 2.38 / 2.26 / 2.08 / 2.02; hidden 128:
  1.92 / 1.78 / 1.65 / 1.62; hidden 256: 1.60 / 1.47 / 1.38 / 1.34; consistency 64 x 8 4.45, 128 x 2 4.75, 128 x 4
  6.60 [VERIFIED]. MobileLLM at 134 to 135 M: 12 x 768 43.9, 30 x 512 44.8, 62 x 384 44.7 [VERIFIED]. Running every
  block twice with shared weights: 44.6 to 45.0 at twice the compute [VERIFIED].
- Feed-forward: T5-base log-perplexity ReLU 1.677, ReGLU 1.645, SwiGLU 1.636
  (https://arxiv.org/abs/2002.05202) [VERIFIED]; MobileLLM ReLU 42.6 to SwiGLU 43.9 [VERIFIED].
- Recurrent: minGRU 1.548, minLSTM 1.555, Mamba 1.575, Transformer 1.547 on Shakespeare at about 5 M
  (https://arxiv.org/abs/2410.01201) [VERIFIED]; TinyStories at 10.94 M: Transformer 6.93, recurrent-only 7.26,
  recurrent plus attention 5.95 (https://arxiv.org/abs/2610.39854) [VERIFIED against the abstract]; integer LSTMs
  keep the cell state at 16 bits (https://arxiv.org/abs/2101.05453) [VERIFIED].
- Decoding: Codex optimal temperature 0.2 for pass@1, 0.8 for pass@100 [VERIFIED].

### Uncertain
- Depth against width at small size (MobileLLM says deep and thin at 125 M and more; TinyStories 128 x 2 beats 64 x 8
  on loss but is weak on consistency and has twice the embedding parameters).
- Heads: the baseline head dimension of 8 is below anything found tested (smallest 16, worse than 32 to 64 at 65 M).
- What MQA costs: 0.2 BLEU at 211 M, 0.6 points at T5-XXL, 0.9 to 1.7 points at 125 M to 350 M (table not monotonic).
  Sharing K/V is nearly free but query heads cost the cycles on LS.
- SwiGLU against ReGLU: the 0.009 gap is of the order of the run-to-run spread.
- Recurrent models at small size: parity on Shakespeare is weak evidence; behind at 10.94 M on TinyStories.
- Our cost model: the two-pass softmax puts attention at 74% of a layer against "about 60%" in CONTEXT.md; rankings
  are robust, absolute tokens per second are not until the cycle-accurate simulator exists.
- Our tokenizer measurement: the original script was not saved; tokenizers trained on 8.8 MB of standard-library
  code give only 1.6 to 2.1 bytes per token on MBPP; without carriage returns 1.97 / 2.39 / 2.65.

### Not tried at our scale
- Any ablation of heads, KV heads, feed-forward type, depth against width, windows or weight sharing at 260k to 400k
  parameters; head dimension 8; any of these under int8 with power-of-two scales.
- Vocabulary 256 to 1024 for source code (smallest found: 2k merges).
- Per-layer windows for code where the task statement sits at the start of the context.
- Early exit in a decoder of 5 or fewer layers; recurrent or hybrid models below 1 M on code or with an int8 state;
  output rows only for grammar-legal tokens.
- A cost model where a row costs n + 24: published speed-ups do not transfer, only the quality numbers do.

## 6. Grammar-constrained decoding (`research/06_constrained_decoding.md`)

### Proven
- SynCode, Python: syntax errors standard to constrained on HumanEval 271 to 15 (CodeGen-350M), 36 to 3
  (WizardCoder-1B), 291 to 2 (LLaMA-7B); MBXP 78 to 4, 28 to 2, 148 to 5; average reduction over Python and Go
  96.07%; the remaining errors are outputs that did not finish inside the token limit. pass@1 6.8 to 6.9, 20.0 to
  20.0, 11.2 to 11.5; pass@10 10.6 to 10.6, 27.6 to 28.4, 17.1 to 18.9. Indentation is 22% to 35% of syntax errors.
  https://arxiv.org/html/2403.01632v4 [VERIFIED]
- Type-constrained decoding (TypeScript): an ideal syntax-only filter would remove 9.0% (HumanEval) and 4.8% (MBPP)
  of compiler errors; type constraints 74.8% and 56.0%; pass@1 +3.5% relative (Gemma 2 2B 29.1 to 30.2, 40.4 to
  42.4); run time +39.1% and +52.1%; mixed protocol. https://arxiv.org/abs/2504.09246 [VERIFIED]
- Where the unconstrained model breaks structure often, gains are large: PICARD T5-Base 57.2 / 57.9 to 65.8 / 68.4,
  T5-Large 65.3 / 67.2 to 69.1 / 72.9, T5-3B 69.9 / 71.4 to 74.1 / 76.3 (https://arxiv.org/abs/2109.05093)
  [VERIFIED]; Synchromesh SQL validity 43% to 66%, accuracy 16% to 20% (https://arxiv.org/abs/2201.11227)
  [VERIFIED]; TOML with Gemma-2-2B +53% relative (about 40% of outputs had errors), Qwen-2.5-32B +2.5%
  (https://arxiv.org/html/2606.21619v1) [VERIFIED]; Bash, 13 models: 62.5% to 75.2%, Qwen3-0.6B 16.7% to 59.2%,
  SmolLM2-360M 29.4% to 57.2%
  (https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/)
  [VERIFIED].
- When constraints hurt: token misalignment cuts GSM8K from 0.415 to 0.308 (Mistral-7B) and 0.155 to 0.0
  (Llama-2-13B) (https://arxiv.org/html/2403.06988v1) [VERIFIED]; a re-run of type constraints: 32.0% against 35.7%
  (HumanEval) and 44.9% against 50.4% (MBPP), token-limit hits 3.4% against 0.1%; forbidding the optional space
  before `=` in TOML: 48.1% to 28.6% and 62.5% to 1.9% (97% loss) (https://arxiv.org/html/2606.21619v1) [VERIFIED].
- Table sizes: SynCode's Python mask store 1.87 GB (51,200 tokens), 1.83 GB (49,153), 1.17 GB (32,000); XGrammar
  160 MB to 0.46 MB for JSON (https://arxiv.org/html/2411.15100v3) [VERIFIED]. Scaled linearly to 512 tokens about
  19 MB [UNVERIFIED: our linear extrapolation; no source builds such a store for a small vocabulary, and the other
  engines of section 6.1 were not sized]. The size table of the topic file gives about 18.7 MB at 512 tokens and
  9.4 MB at 256 [UNVERIFIED: no source builds this store for a small vocabulary].
- Masking in training (Bunel et al., LSTM, 52 tokens): 71.91 to 72.07 on full data, 12.58 to 9.81 with 10,000
  examples; a larger unmasked model 73.67 and 13.14. https://arxiv.org/abs/1805.04276 [VERIFIED]
- Python facts: indentation needs a stack of widths; no NEWLINE inside brackets; CPython limits MAXINDENT 100,
  MAXLEVEL 200; `ast.parse` accepts 7 error classes that `compile()` rejects [VERIFIED].
- Measured on the proxy corpus (9,469 standard-library functions, `ast.unparse` layout) [VERIFIED, re-run]: 100,183
  code lines, 0 violations of the layout invariants, 38,262 block headers (38.2%); bracket depth at most 7, indent
  level at most 10. Free BPE learns 28 bridge pieces of 512 (6.6% of tokens), 15 at 256 (4.8%). A structural
  vocabulary (line end = one token with the indent change `NL-k` / `OPEN`, 6 brackets and 4 quotes as single tokens)
  gives 510,645 tokens against 511,352 for llama2.c-style BPE at 512 (-0.1%; -0.8% at 256) and +2.2% / -0.3% against
  free BPE with pieces spanning spaces; lossless on 9,469 of 9,469 functions. The token filter rejected 0 of 9,469
  functions over 1,973,985 tokens; only 17 tokens can ever be forbidden, 14.5 on average, at most 16 at once.
  Truncation repair on 3,000 functions cut at 3 points each: compiles as cut 33.5%; brackets closed in place 75.0%;
  drop the unfinished line and add `pass` 95.3%; plus `except Exception: pass` 99.98% (2 failures in 9,000).
- Fixed-size stacks raise the maskable tokens from 17 to 20 [UNVERIFIED: the filter was never run with fixed-size
  stacks]; 2 of 9,469 functions have more than 8 open blocks and are rejected unless that stack is packed or
  enlarged [UNVERIFIED: fixed-size stacks are not implemented in the reference `Filter`; the fact check only
  counted the 2 functions]. Proposed filter state: 21 B (mode, escape flag, bracket depth, 8 B bracket stack,
  indent level, line flags, 8 B block-kind stack), 533 B with the class table, about 0.12% of 450 KB
  [UNVERIFIED: the 21-byte layout was never run; the reference filter uses unbounded lists].

### Uncertain
- Whether constrained decoding raises functional correctness at all: Mündler et al. +3.5% relative, Biagiola et al.
  find unconstrained ahead in every configuration; protocols differ [VERIFIED]. Syntax-only on models that already
  write valid code: +0.0 to +0.3 points.
- Whether weak models gain more: yes for SQL, Bash, TOML; no for Python syntax on CodeGen-350M (+0.1). A Python
  syntax grammar carries no task knowledge.
- The hypothesis that the filter lifts "syntax validity" and "runs without error" and leaves HumanEval and MBPP
  about where they are [UNVERIFIED: no result exists for a model of our size; the trigram stand-in of section 4
  stayed below 2% compiling with the filter on]. The "Expected effect" column of that file's idea table is a
  forecast [UNVERIFIED: no model of our size has been run with or without the filter].
- Distortion from a minimal filter: the trigram stand-in put 5.8% to 14.5% of its probability on masked tokens.
- Two different 6% figures must not be merged: 6% of unconstrained Python samples have a syntax error (SynCode); 6%
  of TypeScript compile errors are syntax errors (Mündler et al.).
- The proxy corpus is longer (mean 208 tokens) and deeper than our target tasks; invariants must be re-checked on
  the real corpus.

### Not tried at our scale
- No with-and-without comparison of a token filter for a text-token transformer under 10M parameters writing Python,
  at a 256 or 512 vocabulary, or on a microcontroller, FPGA or 8-bit CPU: not found (six searches).
- Constrained decoding with int8 logits; a full-Python mask store at 512 tokens; LR tables for a Python subset in
  tens of kilobytes; a tiny transformer trained with a structural mask in the loss: not found.
- Our only data point is a trigram stand-in (2,000 samples per row, budget 200 tokens) [VERIFIED, re-run]: compiles
  after repair 0.9% (no filter), 3.2% (filter), 4.7% (filter + header rule) at temperature 1.0; 1.9%, 10.0%, 17.2%
  at temperature 0.5. Finished inside the budget: 76.2% / 52.6% / 15.8% and 78.7% / 61.1% / 11.7%. Bracket and string
  errors disappear, statement-level errors take their place, non-termination rises.

## 7. Weight-space editing (`research/07_weight_space_editing.md`)

### Proven
- The source exists: "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights",
  Takehi, Teoh, Zhang, Khattab (MIT CSAIL), published November 2, 2026, a blog post (`note = {Blog post}`), not on
  arXiv (two API queries returned 0) [VERIFIED]. https://rikiyat.github.io/blog/weight-space-programming/
- Experiment 8: OpenAI's `gpt-6-astra`, gradients and training forbidden by the harness, changed 131,072 of the 0.8
  billion weights of Qwen3.5-0.8B (0.016%); MATH-500 7.9% to 48.1% [VERIFIED]. The edit is a difference-of-means
  "repetition" direction from four hand-written text pairs, removed from 64 down-projection columns in layers 15 and
  19 with W = W0 - 6 u (u^T W0).
- Budget ablation (MATH-500 / GSM8K): 0 passes no edit; 5 passes no edit submitted (its candidate 8.6 / 17.1); 10
  passes 12.3 / 25.7; 25 passes 14.1 / 25.1; 50 passes 48.1 / 67.1 (158 turns, 80 Python runs, 20.3 hours).
  IFEval 21.0 to 62.3, GPQA Diamond 22.7 to 36.6, MMLU-Pro 40.3 to 45.5 [VERIFIED].
- Why it worked: 93.0% of base answers hit the 32,768-token limit in a thinking loop; 87.2%, 86.4%, 33.1% after 10,
  25, 50 passes. No ability was added: accuracy on originally finished outputs unchanged (114 problems, 93.9% before
  and after). Side effects on 1,152 practice problems: finished with a wrong answer 6 to 187, finished without an
  answer 1 to 58, cut-offs ending in whitespace 0 to 51 [VERIFIED].
- Gemma 3 270M (8.5% of answers reach its limit): MATH-500 6.8 base, 7.0 / 7.6 / 7.7 / 7.0 after 5 / 10 / 25 / 50
  passes (lenient grading; strict 2.8 base, 3.5 / 3.1 / 3.8 / 2.6) [VERIFIED]. Authors: "Editing weights could remove
  an obstacle that hid an ability, but it could not add an ability the model lacked."
- Outside re-measurements of the edited checkpoint: 51.0% (31.6% at the token limit) and 51.7%
  (https://arxiv.org/abs/2611.14210); Hu et al. report 8.35% for the unedited model
  (https://arxiv.org/abs/2611.09382) [VERIFIED].
- Cost: about 6.3 million input and 0.16 million output tokens for the 50-pass run (our estimate from all `run.json`
  files); forward passes on a GPU node. Released: every prompt, reply and code run, each edit as an exact diff; logs
  CC BY 4.0, scripts MIT, model card apache-2.0 [VERIFIED]. The bit-for-bit rebuild is stated by the README
  [UNVERIFIED by us: the bit-for-bit rebuild is a statement of the README and of the model card, which reports a hash
  check; `apply_edit.py` was not run.]. The Experiment 8 agent harness is not released; the editor is a closed model.
- Experiment 4 (171,664 weights shown as integers times one scale, one reply): 51 edit statements changed 6,073
  weights (3.54%); 0/256 to 1,024/1,024 test episodes [VERIFIED]. Experiment 7: "from the weights alone, Astra was
  unable to tell" [VERIFIED].
- Established methods and what they need (table 6.1 of the topic file): direction removal W - r r^T W needs no
  gradients (https://arxiv.org/abs/2406.11717); ROME and MEMIT need gradients only to find the target vector
  (https://arxiv.org/abs/2202.05262, https://arxiv.org/abs/2210.07229); AdaRound and BRECQ need gradients
  (https://arxiv.org/abs/2004.10568, https://arxiv.org/abs/2102.05426); OBC, GPTQ/GPTAQ and COMQ need layer inputs and
  linear algebra only (https://arxiv.org/abs/2208.11580, https://arxiv.org/abs/2210.17323,
  https://arxiv.org/abs/2403.07134); bias correction and cross-layer equalisation need no gradients
  (https://arxiv.org/abs/1906.04721); zeroth-order and evolutionary search, also in the quantised domain
  (https://arxiv.org/abs/2305.17333, https://arxiv.org/abs/2602.03120); task vectors need fine-tuned checkpoints
  (https://arxiv.org/abs/2212.04089). GPTQ at 4 bits: OPT-175B 8.37 against 10.54 round-to-nearest and 8.34 at 16
  bits; OPT-125M 31.12 against 37.28 and 27.65.
- Our synthetic check (random 64 x 172 matrices, int8, power-of-two scales) [VERIFIED: re-run]: strength 1 on all
  columns changes 67.4% of the weights, mean step 2.7 LSB, cosine 0.994, 2.0% of the write along u left, Frobenius
  change 12.4%; Laplace weights 50.8%, 1.7 LSB, 0.981, 4.5%. Strength 6 on 16 columns: cosine 0.998 to 1.000, 15 to
  40 LSB, up to 0.15% clip, Frobenius change 48 to 49%. Arithmetic on random matrices, not a model result.

### Uncertain
- A capability of the model or a known playbook: Goedecke calls it "basically the Heretic abliteration playbook"
  (https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/); Willison
  reports Honnibal "called the paper misleading" (https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/)
  [UNVERIFIED: Honnibal's own text not opened; quoted through Willison]; the source list of the topic file marks
  Honnibal's comment the same way [UNVERIFIED: not opened; quoted through Willison].
- The gain depends on the base setting (thinking on, 32,768-token limit).
- Reliability: one editing run per budget; 6 of 16 follow-up trajectories ended without an edit; three independent
  Math searches gave +29.45, +5.00 and +27.30 points [VERIFIED].
- Status of the source: a blog post 11 days old, no peer review.
- For LS-LLM three unknowns decide the stage: (a) whether one failure class dominates once constrained decoding is
  in place; (b) whether at width 64 a failure is carried by one direction; (c) whether anything is left to repair
  after bit-exact QAT.
- Layer-wise refits at 8 bits after QAT: no source. Forward-only search on a few hundred practice tasks will find
  noise (selection bias). Many sequential edits degrade a model (https://arxiv.org/abs/2401.07453).
- Named but not opened: the original Optimal Brain Surgeon paper [UNVERIFIED: not opened; the OBC abstract refers to
  it as "the classical Optimal Brain Surgeon (OBS) framework of [LeCun, Denker, and Solla, 1990]"]; the original
  "logit lens" write-up [UNVERIFIED: not opened; known here only through the tuned-lens abstract, which calls it "the
  earlier 'logit lens' technique"].

### Not tried at our scale
- An LLM-agent edit of a language model below 270M parameters: not found.
- Direction removal folded into int8 weights with power-of-two scales and int8 activations; direction removal at
  residual width 64 (Arditi et al.: 1.8B to 72B): not found.
- OBC / GPTQ / COMQ on a transformer below 1M parameters at 8 bits after QAT; zeroth-order search on int8 weights
  with tasks passed as fitness; bias correction through an accumulator preload on a bias-free model; ROME or MEMIT
  below GPT-2 size; head or hidden-unit removal as a speed measure at 260k to 400k parameters; a hand-written circuit
  grafted into a trained language model of this size: not found.

## 8. Datasets and licences (`research/08_datasets_licences.md`)

### Proven
- Nine hand-picked permissive repositories (TheAlgorithms/Python, keon/algorithms, exercism/python, more-itertools,
  toolz, boltons, funcy, pydash, pygorithm; MIT or BSD-3-Clause, LICENSE files read): 13.9 MB of tarballs, 2,918 `.py`
  files, 10.1 MB of Python, 8,036 top-level functions, 4,426 with a docstring, 3,255 with a doctest, 3,639 kept
  (2.30 MB) [VERIFIED]. TheAlgorithms alone: 1,812 kept (1.26 MB), 2,112 doctests. exercism/python: 142 practice
  solutions, 163 unit-test files.
- CodeSearchNet Python: 412,178 / 23,107 / 22,176 functions; the original 940,909,997-byte zip is readable by HTTP
  range requests without login
  (https://huggingface.co/datasets/code-search-net/code_search_net); validation shard: 23,107 functions from 679
  repositories, 7,069 kept plain functions (4.63 MB); train_0: 30,000 functions from 85 repositories, 11,781 kept
  (8.15 MB). Licence table sample (748 repositories): 614 (82.1%) permissive only by root licence files, 125 (16.7%)
  with a GPL, LGPL or AGPL text, so a filter is required [VERIFIED]. Yu et al.: 91.88% permissive, 6.37% strong
  copyleft for the whole dataset (https://proceedings.mlr.press/v202/yu23g.html) [VERIFIED].
- codeparrot/codeparrot-clean: not gated, per-file `license` field, 54 files (12.8 GB); 6 MB sample: 2,413 files,
  1,435 permissive (59.5%), 1,407 kept functions (0.77 MB), about 235 kept functions per compressed MB [VERIFIED];
  Yu et al.: 64.98% permissive, 29.94% strong copyleft.
  https://huggingface.co/datasets/codeparrot/codeparrot-clean
- The virtualenv has no parquet reader; tarballs, zip, `.json.gz` and `.jsonl.gz` need only the standard library
  [VERIFIED].
- Licence label against provenance: APPS (label MIT, problems from Codewars, AtCoder, Kattis, Codeforces; Codeforces
  terms restrict sublicensing), CodeContests ("We make no representations here with respect to rights"), CoNaLa
  (label MIT, Stack Overflow content is CC BY-SA) [VERIFIED].
- Terms for training on AI outputs (clauses read at the source) [VERIFIED]: Anthropic Usage Policy forbids to "Use
  outputs to train an AI model ... without prior authorization from Anthropic" (https://www.anthropic.com/legal/aup);
  the help-centre article lists "Models designed for open-ended text generation" and "Using Outputs as training
  targets for models" as prohibited
  (https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model). OpenAI: no "Output to
  develop models that compete with OpenAI", no programmatic extraction. Llama 2, Llama 3, CodeLlama: outputs may not
  improve another LLM. Llama 3.1 / 3.3: allowed, the model name must begin with "Llama". Qwen2.5-Coder 0.5B, 1.5B,
  7B, 14B, 32B Instruct: Apache-2.0, no clause about outputs; the 3B is non-commercial. DeepSeek-Coder and StarCoder:
  allowed, but their use restrictions attach to our model. nampdn-ai/tiny-codes, flytech/python-codes-25k,
  iamtarun/python_code_instructions_18k_alpaca: generator not stated.
- Benchmarks: HumanEval 44,877 bytes, 164 problems, MIT, test-only. MBPP 563,743 bytes, 974 problems, CC-BY-4.0;
  test = ids 11-510, prompt 1-10, validation 511-600, train 601-974 (10 / 500 / 90 / 374); train split has 66,715
  bytes of code. Sanitized MBPP 427 problems [VERIFIED].
- Published contamination: 12.2% of HumanEval and 3.6% of MBPP solutions seen in the Pile, 18.9% and 20.8% in the
  Stack; StarCoderBase-15.5B 41.6 on MBPP, 33.8 after removing the 20.8% (https://arxiv.org/abs/2403.04811)
  [VERIFIED]; DeepSeek-Coder's 10-gram rule (https://arxiv.org/abs/2401.14196) [VERIFIED]; phi-1 still outperforms
  StarCoder after pruning over 40% of CodeExercises [VERIFIED]; a synthetic set added +9% HumanEval, +2% MBPP,
  unchanged on an unseen benchmark (https://arxiv.org/abs/2407.07565) [VERIFIED].
- Our 10-gram overlap check (21,693 distinct benchmark 10-grams) [VERIFIED]: at most 5 MBPP-test solutions
  (pygorithm), 1 HumanEval docstring phrase and 1 HumanEval solution phrase in any one scanned source; no MBPP test
  task text anywhere; hits are textbook code and common idioms.

### Uncertain
- Only 1 of 16 CodeSearchNet shards and 6 MB of 12.8 GB of codeparrot-clean were scanned for overlap; copies in the
  unscanned part cannot be excluded [UNVERIFIED: only 0.05% of codeparrot-clean was scanned; its crawl date is not on
  the card].
- The licence mix of CodeSearchNet Python is from 4.5% of the table; the 60% floor used for planning is the
  codeparrot-clean sample [UNVERIFIED: the full 32.67 MB table was not read].
- Corpus size: downloads total about 215 MB (0.9 + 13.9 + 99.7 + 100) [VERIFIED]; expected kept text before
  deduplication 2.3 MB (core) + about 22 MB (CodeSearchNet; 16 MB at a 60% permissive share, 25 MB at 92%) + about
  13 MB (codeparrot-clean) = about 37 MB (31 to 40 MB) [UNVERIFIED: extrapolation; the permissive share per
  function was not measured].
- Why 13% of permissive codeparrot files (188 of 1,435) do not parse [UNVERIFIED: the cause, probably Python 2
  syntax, was not measured].
- Whether int8 weights trained on MIT/BSD code are a copy: not researched; an attribution file is planned.
  Repository licence is not file licence.
- Anthropic clause, open point: tasks and tests written by Claude and used only for evaluation are not training
  data, but if gradient-free weight editing uses pass/fail on such tasks as its signal, that is arguably training on
  outputs. Conservative choice: editing and tuning use only human-written or licensed tasks (MBPP train and
  validation, exercism tests, doctests). AI-written template text is settled by the `CLAUDE.md` rule of 2026-11-13
  19:05: none enters the training data.
- OpenAI "compete" is not defined: not "clearly allowed". Llama 3.1 naming clause for a report-only project: unclear.
- Yields are from samples (kept share 30.6% and 39.3% in two shards). Not opened: copyright exceptions, LeetCode
  terms, Rosetta Code, CodeNet's data licence, codeparrot-clean's origin [UNVERIFIED: pages not opened].
- Free disk moved between 293 MB and 4.6 GB during the work; check before every download.

### Not tried at our scale
- A model of this size trained on licence-filtered human Python, reading a natural-language prompt, scored on
  HumanEval or MBPP: not found.
- A TinyStories-style Python corpus from a generator whose licence permits training, for a model below 1 M: not found
  (nearest: TinyPy at 1 M, https://arxiv.org/abs/2407.10194; phi-1 at 1.3 B) [VERIFIED].
- Whether 2 MB of short tested functions trains this size as well as 30 MB of mixed GitHub functions; whether
  decontamination changes a 300k-parameter model's scores; identifier filtering for a 256/512 vocabulary; MBPP train
  as the only in-format data; execution-filtered corpora: not measured or not found.

## 9. FPGA costs (`research/09_fpga_costs.md`)

### Proven
- XC7A100T: 63,400 LUTs, 126,800 flip-flops, 15,850 slices, 240 DSP48E1, 135 block RAMs of 36 Kb (4,860 Kb), 6 clock
  management tiles (https://docs.amd.com/go/en-US/ds180_7Series_Overview,
  https://docs.amd.com/go/en-US/ug474_7Series_CLB) [VERIFIED]. Board: XC7A100T-1CSG324C, 100 MHz oscillator, 128 MiB
  DDR2 (https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual) [VERIFIED].
- DSP48E1: 25 x 18 multiplier and 48-bit accumulator; P = P + A x B in one slice (OPMODE 0100101); maximum frequency
  at speed grade -1: 464.25 MHz fully pipelined, 257.47 MHz without MREG, 190.69 MHz with no pipeline register; with
  16-bit products 32 spare bits in P (https://docs.amd.com/go/en-US/ug479_7Series_DSP48E1,
  https://docs.amd.com/go/en-US/ds181_Artix_7_Data_Sheet) [VERIFIED]. The 24-bit accumulator is an LS design choice.
- INT8 packing: WP486 (DSP48E2) guarantees 7 product terms (https://docs.amd.com/go/en-US/wp486-deep-learning-int8)
  [VERIFIED]; on the DSP48E1 shift 16 allows 1 and shift 17 allows 3 to 4 accumulations [derived]; Double MAC (DATE
  2017) needs an unsigned shared operand and fabric logic, 0.5 DSP, 11 LUTs, 12 flip-flops per multiply-accumulate
  (https://past.date-conference.com/proceedings-archive/2017/pdf/0348.pdf) [VERIFIED]. 239 DSP slices are idle: use
  one slice per lane.
- Block RAM: 4,096 bytes per 36 Kb block at 9, 18 or 36-bit width (4,608 with the ninth bits); 450,000 bytes need
  110 of 135 blocks; 135 blocks = 552,960 bytes (622,080 with ninth bits) [derived]; ports may differ in width;
  byte-wide write enables; a read of an address written by the other port in the same cycle is undefined unless the
  writing port is READ_FIRST; output register adds one clock (2.46 ns to 0.89 ns clock to output); maximum 388.20 MHz;
  contents initialised by the bitstream (https://docs.amd.com/go/en-US/ug473_7Series_Memory_Resources) [VERIFIED].
- DDR2: Micron MT47H64M16HR-25:H, 16-bit; 667 Mb/s maximum with a 4:1 controller; user clock 81.25 MHz, 128-bit user
  bus, 16 bytes x 81.25 MHz = 1.3 GB/s peak [derived] against 0.1 GB/s for one multiply-add lane. Vendor size for a
  72-bit DDR2 controller: 9,267 LUTs, 6,038 flip-flops (14.6% of the device); 32-bit LPDDR2 3,952 LUTs; a 16-bit DDR3
  third-party core 1,279 LUTs (https://docs.amd.com/r/en-US/ds176_7Series_MIS/Resource-Utilization,
  https://docs.amd.com/go/en-US/ug586_7Series_MIS) [VERIFIED]. The Digilent SRAM adapter gives one 16-bit word per
  210 ns (about 9.5 MB/s): unusable as a weight stream [VERIFIED].
- UART: FT2232H up to 12 Mbaud; exact rates with a 100 MHz clock are 1, 2 and 4 Mbaud; C4 = FPGA input, D4 = FPGA
  output (https://ftdichip.com/wp-content/uploads/2020/07/DS_FT2232H.pdf) [VERIFIED].
- Calibration sizes: PicoBlaze 26 slices; SERV 125 LUTs; PicoRV32 761 to 2,019 LUTs; VexRiscv 504 to 1,418 LUTs;
  MicroBlaze 617 to 1,009 LUTs; an 8 x 8 LUT multiplier 71 LUTs; `uart_tx6` 15 LUTs, 8 SRL16E, 24 flip-flops
  [VERIFIED].
- Cost model split of a token [cost model]: about 42% multiply-adds (423k), 34% per-row control (338k, 14,072 rows at
  24 cycles), 24% element-wise (239k). In-row lanes alone cap at 1.73x; row-parallel lanes reach x1.95 at 4 lanes.
  Fused clear/requantise/store -11% (x1.13); whole matrix-vector instruction -28% (x1.39); element-wise streams -23%
  (x1.29), x2.03 with the matrix-vector instruction; all with 4 lanes about 162k cycles per token, about 6x, for
  about 200 to 400 LUTs, 3 DSP slices and half a block RAM [estimate]. A ceiling from the cost model, not a forecast.

### Uncertain
- MIG size for 16-bit DDR2: not published; bracket above 1,279 and below 9,267 LUTs [estimate]; settled by generating
  the core. MIG read latency: UG586 says "Simulation should be run to analyze read latency". Sequential read
  throughput on this board: no measurement found.
- Whether 4 Mbaud works with the host in use: not tested (needs driver 2.4.20 or later).
- All LUT and flip-flop figures marked [estimate]; how Vivado builds the read multiplexer of a 110-block memory; how
  fast the LS CPU itself can be clocked (the slack of the present build is the only relevant number).
- The 42% / 34% / 24% split is a reconstruction; `05_speed_techniques.md` gets 1.10 M cycles with a two-pass softmax.
- The 6x ceiling assumes all 14,072 rows can be laid out four to a pass with about 3 control cycles per row; that
  layout has not been designed.

### Not tried at our scale
- A design that streams an int8 transformer of this size from block RAM through 1 to 4 DSP48E1 lanes under a small
  multi-cycle CPU on an Artix-7: not found.
- Operand packing for sequential signed int8 dot products of 8 to 512 terms; the ninth block RAM bit for a language
  model; MIG as a pure sequential weight stream on the Nexys A7; LUT counts for an argmax unit over 256 to 512
  classes; a 100 KB class LUT ROM at 100 MHz: not found.

## 10. Comparables

Both tables are copied cell by cell from the topic files (tags included); nothing was added.

### Table A. Code models (`research/01_tiny_code_models.md`, tables 6.1 and 6.2)

HumanEval is pass@1 / pass@10 / pass@100 in percent unless stated. Last column, as defined in topic 01: "Yes" = weights public and the weight file is at most 100 MB (the project download limit); "Over limit" = public and small enough for 8 GB RAM but above the 100 MB rule; "No (file size)" = does not fit the disk. All need a loader (`transformers`/`tokenizers` are not in the venv).

| Name | Parameters | Bytes on disk | HumanEval | MBPP | Licence | Weights public | Runnable on the 8 GB laptop | Source |
|---|---|---|---|---|---|---|---|---|
| Codex-12M | 12 M (label; read as a non-embedding count, section 2) | not released | 2.00 / 3.62 / 8.58 (n = 200; temperatures in section 2) | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| AlphaCode decoder-only 13M | 13 M | not released | 1.5 / 3.6 / 8.6 (1000 samples) | none | n/a | No | No | https://arxiv.org/abs/2203.07814 [VERIFIED] |
| Codex-25M / 42M / 85M | 25 / 42 / 85 M (labels, as Codex-12M) | not released | 3.21 / 7.1 / 12.89; 5.06 / 8.8 / 15.55; 8.22 / 12.81 / 22.4 | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| AlphaCode decoder-only 29M / 55M / 89M | 29 / 55 / 89 M | not released | 3.4 / 5.8 / 11.2; 4.2 / 8.2 / 16.9; 4.3 / 12.2 / 20.0 | none | n/a | No | No | https://arxiv.org/abs/2203.07814 [VERIFIED] |
| TabNine free model | unknown | not released | 2.58 / 4.35 / 7.59 | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| 1 M TinyPy models (Naïr et al.) | about 1 M | 4,385,396 or 4,386,378 | not evaluated (token accuracy 81.23%, output accuracy 74.58-79.23% on synthetic programs) | not evaluated | Apache-2.0 | Yes (GitHub) | Yes, but cannot read natural prompts | https://github.com/kamel-yamani/CurriculumLearningForSmallCodeLanguageModels [VERIFIED] |
| nilq/python-mistral-1L-mini | 4,133,056 (1 layer, dim 64; embedding tables dominate) | 16,533,496 | none | none | none on card | Yes | Yes | https://huggingface.co/nilq/python-mistral-1L-mini [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| CinnabarLM-4M-Python-Base | 4,229,568 | 16,924,408 | none | none | Apache-2.0 | Yes | Yes | https://huggingface.co/MihaiPopa-1/CinnabarLM-4M-Python-Base [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| Pico-1.3-Coding-Large | 6,917,376 | 13,843,136 | not tested (card: "expect about 0 %") | not tested; own sets: 68% (50 seen-kind), 10% (2/20 unseen names) | Apache-2.0 | Yes | Yes | https://huggingface.co/Convergent-ch/Pico-1.3-Coding-Large [VERIFIED] |
| Mxode/minicoder-7M-base | 8,617,344 | 34,473,600 | none | none | none on card | Yes | Yes | https://huggingface.co/Mxode/minicoder-7M-base [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| BananaMind-2.1-NanoCoder | 9,895,690 (card); 12,254,978 (hub metadata) | 49,037,296 | none | none | Apache-2.0 | Yes | Yes | https://huggingface.co/Banaxi-Tech/BananaMind-2.1-NanoCoder [VERIFIED] |
| jfrick/scholarly-code-modern-gpt-10m-5b | 9,849,536 (card); 16,140,992 (hub metadata) | 64,573,288 | none (Python validation perplexity 8.47) | none | none on card | Yes | Yes | https://huggingface.co/jfrick/scholarly-code-modern-gpt-10m-5b [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| Indenta-13M-Python | about 13 M per card (weight file is named indenta-9m.gguf) | 27,117,376 (GGUF) | none | none | CC BY-NC 4.0 | Yes | Yes | https://huggingface.co/Rohanify/Indenta-13M-Python [VERIFIED] (13,258,752 parameters in the GGUF metadata, bytes, licence field via hub API; card text not re-read) |
| Rashik24/tinycoder-15M | 15,191,712 | 60,772,984 | none | none | none on card | Yes | Yes | https://huggingface.co/Rashik24/tinycoder-15M [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| CHISPA-20M-python | 20,119,360 | 80,489,416 | none (line exact match 1.2%, leading identifier 17.1%) | none | MIT | Yes | Yes | https://huggingface.co/esraderey/chispa-20m-python [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| stentor_python_30m | 30,419,712 | 121,699,864 | none | unusable (MBPP in training); own categories 33-100% | Apache-2.0 | Yes | Over limit | https://huggingface.co/CastIronMind/stentor_python_30m [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| PyBlissa-Coder-40M | 39.9 M | 160,463,648 (GGUF F32) | 14.6% pass@1 (24/164, T = 0.25), self-reported | 4.4% (22/500, tests in prompt, T = 0.05), self-reported | Apache-2.0 | Yes | Over limit | https://huggingface.co/Rohanify/PyBlissa-Coder-40M-GGUF [VERIFIED] (card text; scores self-reported) |
| PyBlissa-Coder-50M | 50.2 M | 201,685,248 (GGUF F32) | 10.4% pass@1, self-reported | none | MIT | Yes | Over limit | https://huggingface.co/Rohanify/PyBlissa-Coder-50M [VERIFIED] (50,284,800 parameters in the GGUF metadata, bytes, licence field via hub API; card text not re-read) |
| GoldWorm v8 coder | 85,169,664 | 340,690,184 | none (own 30 tasks: 7.08% pass@1, 16.67% pass@16, 92.92% valid) | none | MIT | Yes | Over limit | https://huggingface.co/Uuuuuuniiiiiii/goldworm-v8-coder-85m [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| LittleKedi-medium-wide-sparse-instruct (general chat model) | 305.4 M total, 31.7 M active per token | 1,221,460,040 | none | 3.0% of 500 held-out, self-reported (moderate variant, 173.4 M total / 19.4 M active: 1.8%) | Apache-2.0 | Yes | Over limit (q8 GGUF is 336,139,232 bytes) | https://huggingface.co/edededdy/LittleKedi-medium-wide-sparse-instruct [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| stories260K (not code) | about 260 k | 1,056,540 | not evaluated | not evaluated | MIT | Yes | Yes (reference shape, expected 0) | https://huggingface.co/karpathy/tinyllamas [VERIFIED] |
| Pico-1 / Pico-1.2 (not code) | 254,400 / 460,608 | 513,984 / 929,376 | not evaluated | not evaluated | Apache-2.0 | Yes | Yes | https://huggingface.co/Convergent-ch/Pico-1.2 [VERIFIED] (the Pico-1 byte count was not re-checked) |
| ternary.c64 (not code) | 688,128 ternary weights (820,352 parameters in hub metadata) | 242,324 (packed weights; cartridge image 361,216) | not evaluated | not evaluated | MIT | Yes | Yes | https://huggingface.co/gdm32569/ternary-c64-688k [VERIFIED] |
| codeparrot-small | 110 M | 456,677,609 | 3.80 / 6.57 / 12.78 | none | Apache-2.0 | Yes | Over limit | https://huggingface.co/codeparrot/codeparrot-small [VERIFIED] |
| PyCodeGPT-110M | 110 M | 495,109,521 | 8.33 / 13.36 / 19.13 | none | AFL-3.0 (card); MIT (GitHub repo) | Yes | Over limit | https://arxiv.org/abs/2206.06888 [VERIFIED] |
| CodeGPT-small-py | 124 M | 509,592,250 | 0 / 0 / 0 for CodeGPT-Adapted (CERT paper) | none | none on card | Yes | Over limit | https://huggingface.co/microsoft/CodeGPT-small-py [VERIFIED] (bytes, licence field, CERT zero) |
| GPT-Neo-125M | 125 M | 525,979,192 | 0.75 / 1.88 / 2.97 | none | MIT | Yes | Over limit | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| SmolLM-135M-Instruct | 134,515,008 | 269,060,552 | 7.7 ± 0.8 / 14.5 ± 1.0 (LintSeq authors' run) | MBPP(+) 10.1 ± 1.8 / 14.6 ± 0.5 | Apache-2.0 | Yes | Over limit | https://arxiv.org/abs/2410.02749 [VERIFIED] (scores; the parameter and byte cells were not re-checked for the -Instruct repository) |
| SmolLM2-135M | 134,515,008 | 269,060,552 | not in paper or card | none | Apache-2.0 | Yes | Over limit | https://huggingface.co/HuggingFaceTB/SmolLM2-135M [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| TinyCodeLM-150M | 150 M | 607,539,201 | 6.1 (T = 0, pretrained); 12.8 / 20.6 (LintSeqInstruct) | MBPP(+) 5.4 pretrained; 13.6 / 24.4 tuned | Apache-2.0 | Yes (needs the OLMo model code) | Over limit | https://huggingface.co/upiter/TinyCodeLM-150M [VERIFIED] |
| PolyCoder-160M | 160 M | 699,645,960 | 2.13 / 3.35 / 4.88 | none | none on card; MIT (GitHub repo) | Yes | Over limit | https://arxiv.org/abs/2202.13169 [VERIFIED] (scores as listed in CERT Table 6, bytes, licence field) |
| tiny_starcoder_py | 164,144,128 | 656,601,304 | 7.84 pass@1 | none | BigCode OpenRAIL-M | Yes | Over limit | https://huggingface.co/bigcode/tiny_starcoder_py [VERIFIED] |
| CodeT5+ 220M | 220 M | 445,800,957 | 12.0 / 20.7 / 31.6 | none | BSD-3-Clause | Yes | Over limit | https://arxiv.org/abs/2305.07922 [VERIFIED] (12.0 / 20.7 via LintSeq Table 1, bytes, licence field; 31.6 not re-checked) |
| Codex-300M | 300 M (label; matches the GPT-3 Medium shape without embeddings, section 2) | not released | 13.17 / 20.37 / 36.27 | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| AlphaCode decoder-only 302M | 302 M | not released | 11.6 / 18.8 / 31.8 | none | n/a | No | No | https://arxiv.org/abs/2203.07814 [VERIFIED] |
| codegen-350M-mono | 350 M | 797,367,631 | 12.76 / 23.11 / 35.19 (highest over T of 0.2, 0.6, 0.8) | sanitized MBPP, n = 100, T = 0.8: 14.59 / 41.49 / 63.00 (paper Table 11); MBPP(+) 9.4 / 15.2 (LintSeq run) | BSD-3-Clause | Yes | Over limit | https://arxiv.org/abs/2203.13474 [VERIFIED] |
| codegen-350M-multi / -nl | 350 M | not checked | 6.67 / 10.61 / 16.84; 2.12 / 4.10 / 7.38 | sanitized MBPP, n = 100, T = 0.8: 7.46 / 24.18 / 46.37; 0.96 / 6.37 / 19.91 | BSD-3-Clause | Yes | not checked | https://arxiv.org/abs/2203.13474 [VERIFIED] |
| phi-1-small | 350 M | not released | 45 pass@1 | not stated | n/a | No | No | https://arxiv.org/abs/2306.11644 [VERIFIED] |
| SmolLM-360M-Instruct | 361,821,120 | not checked | 11.3 (authors) / 19.3 ± 1.1 (LintSeq run) | MBPP(+) 19.4 ± 2.4 / 23.1 ± 0.5 | Apache-2.0 | Yes | Over limit | https://arxiv.org/abs/2410.02749 [VERIFIED] (scores) |
| TinyCodeLM-400M | 400 M | 1,816,698,449 | 6.7 pretrained; 13.4 / 20.9 tuned | MBPP(+) 6.8; 19.4 / 29.9 | Apache-2.0 | Yes | No (file larger than free disk) | https://huggingface.co/upiter/TinyCodeLM-400M [VERIFIED] |
| Qwen2.5-Coder-0.5B | 494,032,768 | 988,097,824 | 28.0 (HumanEval+ 23.8), EvalPlus harness | 52.9 (MBPP+ 47.1) | Apache-2.0 | Yes | Over limit | https://arxiv.org/abs/2409.12186 [VERIFIED] (parameters, bytes, licence field via hub API; scores not re-checked) |
| Codex-679M | 679 M (label; matches the GPT-3 Large shape without embeddings, section 2) | not released | 16.22 / 25.7 / 40.95 | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| CodeT5+ 770M | 770 M | not checked | 15.5 / 27.2 / 42.7 | none | BSD-3-Clause | Yes | not checked | https://arxiv.org/abs/2305.07922 |
| SantaCoder | 1.1 B | 4,600,328,417 | 14.0 pass@1 (as listed in phi-1 Table 1) | 35.0 (same table) | BigCode OpenRAIL-M | Yes | No (file size) | https://arxiv.org/abs/2306.11644 [VERIFIED] (scores and token count as listed in phi-1 Table 1) |
| StarCoderBase-1B | 1,137,207,296 | 4,548,859,752 | 15.17 pass@1 (hub model-index, self-reported, "verified: false") | none | BigCode OpenRAIL-M | Gated | No (file size) | https://huggingface.co/bigcode/starcoderbase-1b [VERIFIED] |
| InCoder 1.3B | 1.3 B | not checked | 8.9 / 16.7 / 25.6 (as listed in CodeT5+ Table 2) | none | not checked | Yes | not checked | https://arxiv.org/abs/2305.07922 |
| GPT-Neo 1.3B | 1.3 B | not checked | 4.79 / 7.47 / 16.30 | none | MIT | Yes | not checked | https://arxiv.org/abs/2107.03374 [VERIFIED] (scores) |
| DeepSeek-Coder-1.3B-base | 1.3 B | 2,693,014,838 | 34.8 (HumanEval+ 26.8), as listed in the Qwen2.5-Coder report | 55.6 (MBPP+ 46.9) | "other" (DeepSeek licence) | Yes | No (file size) | https://arxiv.org/abs/2409.12186 |
| phi-1 | 1,418,270,720 | 2,836,578,696 | 50.6 pass@1 | 55.5 pass@1 | MIT | Yes | No (file size) | https://arxiv.org/abs/2306.11644 [VERIFIED] |
| Qwen2.5-Coder-1.5B | 1.5 B | not checked | 43.9 (HumanEval+ 36.6) | 69.2 (MBPP+ 58.6) | Apache-2.0 | Yes | No (file size) | https://arxiv.org/abs/2409.12186 |
| SmolLM2-1.7B | 1.7 B | not checked | 22.6 pass@1 | none | Apache-2.0 | Yes | No (file size) | https://arxiv.org/abs/2502.02737 |

### Table B. Language models on tiny hardware (`research/04_tiny_hardware.md`, table 6.1)

Tokens/s are the authors' figures; "260K" is stories260K (260,032 parameters); timing scope differs between rows, so speeds are not comparable across rows.

| System | CPU (or device) and clock | Model | Quantization / arithmetic | Tokens/s | Source |
|---|---|---|---|---|---|
| Commodore 64 + RAM expansion unit (llama2.c64) | MOS 6510, about 1 MHz [UNVERIFIED: clock not stated in the README, standard C64 value] | 260K | None: float32 weights unchanged, software float from the oscar64 compiler, sin/cos/exp polynomials from the BASIC ROM | about 0.002 (one token per about 8 min; log: 62 tokens in 9:25:43 = 547 s each) | https://github.com/ytmytm/llama2.c64 [VERIFIED] |
| Game Boy Color (gbc-transformer) | 8-bit Game Boy CPU without multiply or divide instruction, clock not stated | 260K | int8 weights with a Q16 scale per row; Q12 activations (int32, clamped to int16 for products); int32 sums; integer sqrt and division; rational exp | 0.0059 (169 s per token), real hardware; from an elapsed time the README calls estimated | https://github.com/maddiedreese/gbc-transformer [VERIFIED] |
| NES (elya-nes) | Ricoh 2A03 (6502), 1,789,772 Hz | Own model: vocabulary 64, width 64, 3 layers, 2 heads, feed-forward 128; 102,400 ternary weights (446,464 with 8 token-selected feed-forward sets) | Ternary weights, 4-bit activations, 8-bit sums in blocks of 16; QAT against an exact integer reference | 1.58 to 1.60 (0.6338 to 0.6243 s per token), context 20 | https://github.com/Scottcjn/elya-nes [VERIFIED] |
| Commodore 64, stock (Soul Player C64); row added in the fact-check pass | MOS 6510, 1 MHz | Own chat model: 2 layers, width 32, 4 heads of 8, feed-forward 64, vocabulary 128, context 20; about 25,000 parameters | int8 weights with per-tensor power-of-two shifts, Q8.8 activations (int16), 128-entry exp table, integer sqrt and division, QAT | about 0.017 ("About 60 seconds per token on real hardware") | https://github.com/gizmo64k/soulplayer-c64 [VERIFIED] |
| Commodore 64, stock (8bitGPT); row added in the fact-check pass | MOS 6502 family, 1 MHz | Character-level: 2 layers, width 32, context 32; about 18,000 parameters | int8 weights with per-tensor integer rescale factors, Q8.8, integer only | about 0.03 characters/s ("each character takes ~30 s") | https://github.com/Timothy-Sikes/8bitgpt [VERIFIED] |
| TI-84 Plus (ti84-llm-math); row added in the fact-check pass | Z80, about 15 MHz | Character-level arithmetic model: width 32, 4 layers, 54K parameters, vocabulary 14 | Q16.16 fixed point, int8 group-quantized weights, integer only | about 0.017 forward passes/s (about 1 minute each) | https://github.com/RileyGreiff/ti84-llm-math [VERIFIED] |
| C64 running a RISC-V emulator (semu-c64) | 6510 emulating RV32 | "240k 8bit model" with the `runq` code path | Q8 (`runq`) | about 25 min for 2 tokens at about 3000% emulator warp | https://github.com/trholding/semu-c64 |
| Z80, CP/M and ZX Spectrum (Z80-uLM) | Z80, 4 MHz | Perceptron over 128 + 128 trigram buckets (for example 256, 192, 128 hidden), not a transformer | 2-bit weights, 16-bit integer sums, QAT | not stated | https://github.com/HarryR/z80ai |
| Atari 800 / 130XE (JAM) | 6502, clock not stated | Neural text model with a "context attention" step, not a standard transformer | Not documented in the README | not stated | https://github.com/marspa73/atarijam |
| TI-84 Plus CE (eZ80-ai) | eZ80 | Fork of Z80-uLM (its attention step was removed) | Integer, QAT | not stated | https://github.com/tornadus/ez80ai |
| 486 PC, MS-DOS 6.22 (dosllam2) | 486DX2, 66 MHz | 260K | float32 | 2.08 | https://yeokhengmeng.com/2025/04/llama2-llm-on-dos/ [VERIFIED] |
| Toshiba Satellite 315CDT, DOS (dosllam2) | Pentium MMX, 200 MHz | 260K / 15M | float32 | 15.32 / 0.43 | same page [VERIFIED] |
| Pentium III PC, DOS (dosllam2) | Pentium III, 667 MHz | 260K / 15M / 42M | float32 | 80.04 / 2.35 / 0.89 | same page [VERIFIED] |
| Windows 98 PC (llama98.c) | Pentium II (350 MHz in Tom's Hardware's description of EXO's video; the EXO pages state no clock) | 260K / 15M | float32 | 39.31 / 1.03 (35.9 for 260K in a founder's post quoted by Tom's Hardware) | https://blog.exolabs.net/day-4/ [VERIFIED] |
| Pentium OverDrive PC, DOS (vellm) | Pentium OverDrive, 83 MHz | stories15M | Q8_0 weights (`runq` scheme), int8 KV cache with float scale per head | 0.27 | https://github.com/ecliptik/vellm |
| PlayStation Portable 2000 (llama2-psp) | MIPS, clock during the test not stated (PSP range 1 to 333 MHz) | 260K | float32 | 35.11 to 36.06 | https://sol.sbc.org.br/index.php/eramiars/article/download/39470/39242 [VERIFIED] |
| PS Vita PCH-1000 (psvita-llm) | overclocked to 555 MHz | 260K / 15M | float32 (llama2.c default; not stated otherwise) | about 120 / about 1.8 | https://github.com/callbacked/psvita-llm [VERIFIED] |
| Dreamcast (llama2c_Dreamcast) | not stated | TinyStories "262K" | not stated | no measured figure in the README ("15M would run at 1 tokens/s", 262K "like 30x faster") | https://github.com/gameblabla/llama2c_Dreamcast |
| PDP-11/34A (ATTN/11) | 16-bit minicomputer | 1 layer, 1 head, width 16, 1,216 parameters, digit reversal; not a language model | Fixed point: Q8 forward, Q15 backward, Q16 sums; 256-entry exp(-i/32) table | training: 350 steps in 5.5 min | https://github.com/dbrll/ATTN-11 |
| ESP32-S3FH4R2 (esp32-llm) | dual core, 240 MHz | 260K | float32, vendor SIMD dot product, both cores | 19.13 | https://github.com/DaveBben/esp32-llm [VERIFIED] |
| Same chip, fork | same | 260K (README text inconsistent) | float32 | about 32.83 | https://github.com/eric-humane/esp32-llm [VERIFIED] |
| Four ESP32 boards | ESP32 and ESP32-S3 | 260K | float32 | 2.33 / 11.12 / 21.41 / 22.02 | https://circuitdigest.com/articles/running-260K-tiny-llm-on-esp32 [VERIFIED] |
| ESP32-S3 N16R8 (esp32-ai) | 240 MHz | 28.9M stored: 559K dense core, 3.1M head, 25M per-layer token-embedding table in flash | 4-bit weights, int8 head and activations for the head | 9.88 end to end (9.72 compute only) | https://github.com/slvDev/esp32-ai |
| STM32H7A3 Nucleo (EmbedLlama) | STM32H7A3, 280 MHz | 260K | float32 | up to about 87 (peak), including UART | https://github.com/schuhandreas/embedllama-stm32h7a3 [VERIFIED] |
| Raspberry Pi Pico (picollama260k) | RP2040 | 260K | float32 math unchanged | not published | https://github.com/tarelerulz/pico-experiments |
| RP2040 board (Starmind-Pico) | RP2040 | Transformers of 1K to 28K parameters | not stated | 2 to 32 | https://comfyspace.tech/starmind/experiments/starmind-pico |
| RP2350 + PSRAM (PicoLlama) | RP2350 | bundled 14 MB model | float32 | about 1.2 (850 ms per token) | https://github.com/earlephilhower/PicoLlama |
| Siracusa MCU (Deeploy) | 8 RISC-V cores at 360 MHz plus accelerator | TinyStories Llama: width 64, 8 layers, 16 heads, feed-forward 256 | int8 weights and activations in matrix products, I-BERT softmax, Hardswish, 32-bit divisions in RMSNorm; post-training | 340 | https://arxiv.org/abs/2408.04413 [VERIFIED] |
| OpenLA500 on Artix-7 XC7A200T | LoongArch32R, 5-stage, clock not stated | 260K | float32; coprocessor with float multiply-accumulate lanes and DMA | 0.50 software; 4.78 at token 1 to 1.90 at token 32 with the coprocessor | https://github.com/WizLiang/LLaMA2-on-OpenLA500 |
| VexRiscv on Cyclone V 5CEBA4 (PocketTPU) | RV32IM, 50 MHz | TinyStories GPT-2: width 64, 6 layers, 1M parameters | Q8_0 weights, dequantized in hardware during DMA | 0.4 software; 4.25 with dot-product unit (24 and 255 tokens/min) | https://github.com/thinkelastic/PocketTPU |
| Rocket + Gemmini on Nexys Video | Rocket core, 50 MHz, 8x8 int8 systolic array | stories15M | int8 matrix products on the array, float32 elsewhere | about 3 greedy, about 2 with top-p | https://mikutyan4.github.io/chipyard-linux-nexys/ch07-llm-inference.html |
| XiangShan Nanhu with a vector dot-product instruction (Nanhu-vdot) | RISC-V, FPGA test | GPT-2 | not stated | about 30% faster than software | https://arxiv.org/abs/2409.00661 [VERIFIED] |
| llama2.c-260k-rtl-fpga | Xilinx XCVU9P, 50 MHz input clock | 260K | int8 weights and activations with run-time quantizer (group 4), Q12.20 residual, UQ8.24 scales, 12-bit tables | not published | https://github.com/wupd1122/llama2.c-260k-rtl-fpga [VERIFIED] |
| dsp_inference | Gowin GW2AR-18 (Tang Nano 20K) | 260K | int8 + power-of-two shift, table operators | about 0.33 (about 3 s per token, PC-orchestrated over UART) | https://github.com/skyline624/dsp_inference |
| ecp5-inference | Lattice ECP5-45F, 31.25 MHz | 260K | int8 + power-of-two shift, four 256-entry Q15 tables | 14.02 decode, context up to 32 | https://github.com/skyline624/ecp5-inference [VERIFIED] |
| fpga-tinystories | Intel MAX 10 10M08, 60 MHz | Own model: 16 layers, width 128, 4 heads, 4.72M parameters (4.20M ternary, 0.52M int8), vocabulary 2,048, context 512 | Ternary weights (QAT), int8 embedding and head, int8 activations, 24-bit sums, base-2 exp table, SiLU table, integer sqrt and reciprocal | 55.6 at 64 tokens, 50.3 at 256, 45.5 at 480 | https://github.com/MRazvan/fpga-tinystories [VERIFIED] |
| TinyTransformer-FPGA | Cyclone IV EP4CE10, 50 MHz | TinyStories-1M backbone: 8 layers, width 64 | 4-bit weights, 8-bit activations | not published | https://github.com/jianmin5xin/TinyTransformer-FPGA |
| openTPU port | Gowin GW2AR-18, 48 MHz | stories42M | int8 matrix products, float32 vector operations | 0.22 | https://github.com/LakoMoor/openTPU-tangnano20k |
| openTPU | Kintex-7 PCIe card, 133.33 MHz | LFM2.5-230M and larger | int8 or 4-bit | 59.0 (230M, int8) | https://github.com/FeSens/openTPU |
| HLSTransform | Xilinx VU9P | stories110M | Q8_0 (`runq` scheme) | 57.11 (system simulation) | https://arxiv.org/abs/2405.00738 [VERIFIED] |
| LlamaF | Xilinx ZCU102 | TinyLlama 1.1B | int8 weights and activations, groups of 256, post-training | 1.478 / 1.424 / 1.328 for 64 / 128 / 256 steps | https://arxiv.org/abs/2409.11424 [VERIFIED] |
| TALOS-V2 | Cyclone V (DE1-SoC), 56 MHz [UNVERIFIED: clock and parameter count are from a third-party report, not the README] | microGPT, 4,192 parameters | Q4.12 | "50k+" | https://github.com/Luthiraa/TALOS-V2 |
| gateGPT | Virtex-5 XC5VLX110T, 80 MHz | 1 block, width 24, 4 heads, context 16, vocabulary 27 | 16-bit Q5.11, 17-entry exp table with interpolation | 53,800 to 69,204 | https://github.com/fguzman82/gateGPT [VERIFIED] |
| kev-gpt | Xilinx KV260, 166.7 to 200 MHz | 2 to 4M-parameter model | 4-bit weights (QAT) x 8-bit activations | 19,242 single stream; 59,965 with 16 streams | https://github.com/MichaelAyles/kev-gpt |
| CraftGPT (Minecraft redstone) | redstone logic, simulated | 5,087,280 parameters, vocabulary 1,920, 6 layers, context 64 | not opened by us | hours per reply even on an accelerated redstone server | https://github.com/sammyuri/craftgpt and https://www.theregister.com/2025/10/02/minecraft_ai_chatbot/ |

## 11. Ranked plan

Ordering rule: expected gain times probability of success divided by cost, for the two equal targets (benchmark
score, tokens per second) under the memory budget (about 450 KB in Tier A). The ordering is editorial; the gains,
costs and keep-or-drop tests are those of the topic files (idea numbers in brackets, for example [05-2] = idea 2 of
topic 05). Cycle figures are cost-model estimates until the cycle-exact simulator exists. Keep-or-drop tests use
validation loss and practice sets only, never HumanEval, the MBPP test split or the custom-200 test split.

### 11.1 Foundations (no gain by themselves; everything else depends on them)

| # | Item | What it enables | How to check it | Depends on | CPU change |
|---|---|---|---|---|---|
| F1 | Bit-exact integer reference model in Python as the specification; QAT forward pass equal to it (float32 tensors holding integers are exact for rows up to 511); token-exact gate for every simulator or board build [03-1, 04-2] | Every score and every edit is measured on what the CPU computes | Token-for-token equality with the emulator on at least 10k tokens and over whole context windows on 3 prompts | nothing | No |
| F2 | Cycle-exact simulator and speed protocol: cycles per token at positions 0, T/2, T-1, cycles per multiply-add, seconds per task and bytes per second, timing scope stated [04-3, 05 yardstick] | Replaces the cost model; makes the tokens/s claim checkable | Board counter equals simulator count | F1 | No (a cycle counter register helps) |
| F3 | Reporting protocol: pass@1 / @10 / @100 with the unbiased estimator (n = 200), zeros reported, the prior expectation of section 1, ladder metrics (next-token accuracy, `compile()` validity, undefined-name rate, seen- against unseen-name pass rate), every score with filter off and on [01-2, 01-7, 06-4] | A zero becomes defensible; progress is visible where pass@1 is zero | Estimator reproduces a hand-computed case | nothing | No |
| F4 | Corpus manifest with licence gate, 10-gram decontamination filter fixed before any benchmark run, hard exclusions [08-1, 08-3, 08-11] | Every data claim and benchmark number can be defended | A test fails on a row without an allowed licence; functions dropped per source reported | nothing | No |
| F5 | Export checks: weights in [-127, 127], every accumulated row at most 511 products (or 128 x sum abs(w) + abs(preload) <= 8,388,607), preload inside the headroom [02-7, 03-15, 04-10] | No 24-bit overflow, guaranteed without a training constraint | Check passes on every exported image; largest abs(ACC) in the simulator under the bound | F1 | No, if `RQ` clamps to -127 or QAT never produces -128 |

### 11.2 Ranked ideas

| Rank | Idea | Should improve | Keep-or-drop measurement | Depends on | CPU change |
|---|---|---|---|---|---|
| 1 | Round to nearest by preloading the accumulator with 2^(s-1), simulated bit for bit in QAT [02-1, 03-2] | Score: removes -0.5 LSB per rounding (-5.5 LSB after 11 roundings; error power 0.333 to 0.083 LSB^2). Speed: 0 cycles if "load ACC" replaces "clear ACC". Cost: hours | QAT with floor against round, integer validation loss; keep round unless floor is equal | F1 | No, if ACC can be loaded with a constant; otherwise a rounding bit in `RQ` |
| 2 | Tokenizer and context chosen by measurement: no byte fallback, pieces may span spaces, vocabulary 512 against 1024 judged per byte, not 256; tokenise practice prompts and pick T in {128, 256, 512}; truncated tasks counted as failures [05-6, 01-4, 04-4] | Speed: 19% fewer tokens at 512 (2.04 to 2.53 bytes per token); 1024 is -8% seconds per task for +32.8k parameters. Score: avoids forced zeros (T = 256 holds about 62% of HumanEval tasks, T = 512 about 95%). Memory: KV cache 320 x T bytes | Bits per byte (never loss per token), seconds per task, share of practice prompts that fit | F2, F4 | No |
| 3 | Prefill skip: for prompt tokens run the last layer only up to K and V, skip the output layer [05-1] | Speed: -23% per prompt token, -17% per median HumanEval-sized task; output identical. Cost: runtime code only | Same tokens as the full run on the practice set; cycles per task | F1, F2 | No |
| 4 | Fewer, wider heads: 8 x 8 to 4 x 16 to 2 x 32 [04-1, 05-2] | Speed: -23% / -34% at T = 256 (-28% / -43% at T = 512). Score: +0.01 to +0.05 nats at larger scale, unknown here | Bits per byte and practice pass rate; keep the fewest heads within seed spread of 8 heads | float training, F2 | No |
| 5 | Data that fits the task: core corpus of the nine repositories (3,639 functions, 3,255 with doctests), bulk from CodeSearchNet and codeparrot-clean through the licence filter, then the self-contained filter checked by execution in `lsllm.sandbox`, simplest N MB; MBPP train for format, MBPP validation for tuning [08-2, 08-4, 08-5, 08-6, 08-7] | Score: the largest lever in the sources at fixed size (350 M: 12.19% to 20.12% to 45%); a hypothesis at our size | Filtered against unfiltered corpus at equal steps on the practice sets; practice pass rate on unseen function names | F4 | No |
| 6 | QAT recipe: TQT-style log2 thresholds per activation tensor (frozen one by one), short fine-tune from the float checkpoint with 0.5 CE + 0.5 KL to its own float logits, exact 256-entry tables in the forward pass with smooth derivatives, equalisation on free power-of-two pairs [03-3, 03-4, 03-6, 02-6] | Score: closes the float-to-int8 gap (TQT: 67.0 to 71.1 with trained thresholds). Planning assumption +0.02 to +0.05 nats [UNVERIFIED: assumption, see section 1] | Integer loss minus float loss; share of saturated values per tensor; loss change when each table is switched on | F1, a float checkpoint of the chosen shape | No (planned table instruction) |
| 7 | One-pass table softmax: exp table on the int8 score inside the score loop, no maximum search, one normalisation of the mixed output by a reciprocal table [02-2, 05-3, 03-8, 04-7] | Speed: -19% (-23% at T = 512); output-side normalisation about 4k to 7k cycles per layer instead of 25k to 66k. Score: unknown; a bare power-of-two normaliser cost 0.029 nats/char in the NES model | QAT loss against the float-softmax model; count of positions with zero weight sum; histogram of (127 - row maximum) | 6 | No (planned lookup instruction) |
| 8 | Residual-stream precision measured in the QAT simulator before any assembly: float, int8, int8 with per-layer power-of-two rescale, 16-bit; one shared scale with an RMS band [02-3, 03-5] | Decides whether int8 is enough (expected about 1.7 bits of read precision lost at the last layer) | Validation-loss gap between int8 and 16-bit stream; keep int8 if inside seed noise | F1 | No |
| 9 | Structural vocabulary, structural filter and budget repair [06-1, 06-2, 06-3] | Score: bracket, string and line-structure errors removed; "syntax validity" and "runs without error" rungs; near 0 on HumanEval/MBPP. Speed: under 0.1% of a token. Memory: 512 B class table + under 32 B state | 0 rejections over the corpus and practice reference solutions; practice "compiles" and "runs" rates with and without; share of runs hitting the budget | 2 (same tokenizer work) | No (table read by token id and a conditional branch) |
| 10 | Per-layer windows: lower 3 or 4 layers see the last 32 tokens [05-4] | Speed: -32% to -42% (-42% to -57% at T = 512). Memory: KV cache 80 to 38 KiB. Score: indirect support only | As rank 4, plus pass rate split by prompt length | float training, F2 | No |
| 11 | Shape search on the score against seconds-per-task front: d = 96 x 3, d = 128 x 2, MQA or 2 KV heads with the memory spent on context, ReGLU / ReLU block, norm variants (fixed-shift RMSNorm table, Dynamic Tanh table, folded BatchNorm) [05-5, 05-7, 05-8, 02-4, 03-7] | Speed: -36% to -41% (wide shapes), -1% to -6% (feed-forward), RMSNorm is 0.9 to 2.4% of a token. Memory: MQA frees 60 to 120 KiB | Position on the front at equal total memory (weights + KV); float loss at equal steps, 3 seeds | 2, 4 | (a) RMSNorm needs both multiplier inputs from X memory or a copy; others no |
| 12 | Hardware, listed separately from the main numbers: read the timing slack and raise the clock (x1.25 or x1.5); fuse clear / requantise / store (-11%); whole matrix-vector instruction (-28%); element-wise streams (-23%); UART at 1 or 2 Mbaud (4%); hardware argmax (-0.9%); then row-parallel lanes (-32% at 2, -49% at 4) [09-1 to 09-5, 09-8, 09-9, 04-5, 04-6, 05-18, 02-A] | Speed only; score unaffected if bit-exact. About 150 to 300 LUTs for control fusion and element streams | Cycle-exact simulator count; token-exact output; post-route slack >= 0 | F1, F2 | Yes (clock and UART: no) |
| 13 | Weight-editing stage as repair: edit harness and failure census on the int8 simulator; accumulator preload as an editable bias (bias correction, 512-value logit bias); layer-wise integer refit against the float teacher; direction removal only if one failure class dominates; forward-only search on small parameter groups [07-1 to 07-5] | Score: unknown; source evidence +40 points where 93% of answers looped, at most +0.9 where 8.5% did. Speed: 0 cycles; fewer tokens if loops end. Memory: up to 10.5 KB for per-row preloads | Paired rule: (b - c) / n at least 2 SE on the "confirm" part; random-direction control within 2 SE of 0; validation loss not worse | F1, 6, 9; human-written or licensed practice tasks | No, if the accumulator can be loaded per row |
| 14 | Many samples per task on the board (LFSR, UART), reported as pass@k; greedy as the default decoder [01-9, 05-10] | Score: pass@100 about 4 to 6 times pass@1 at 12-13 M. Speed: k times the tokens | pass@k curve for k = 1, 10, 100 on the practice set; greedy against temperature 0.2 | F3 | No (planned LFSR and UART) |
| 15 | Reference rows: reproduce one published number with a public model (codeparrot-small 3.80%, or SmolLM-135M-Instruct 7.7 ± 0.8); equal-memory n-gram baseline in 450 KB [01-3, 01-8] | Score: none; validates the harness and shows what 450 KB buys without a network | Reproduced value within sampling error; bits per byte against the transformer at equal bytes | F3; needs `transformers` or a hand-written loader and a download above 100 MB | No |
| 16 | Later or only on evidence: hybrid or recurrent mixers (-51%, high risk) [05-11, 05-12]; training with the mask in the loss [06-5]; header rules and LR table [06-6, 06-8]; 16-bit stream in two limbs (only if rank 8 shows a loss) [02-9]; two-byte exp weights [02-10]; larger-teacher distillation [03-12]; synthetic corpus from Apache-2.0 Qwen2.5-Coder [08-9]; agent session [07-7]; Tier B DDR2 stream [09-10, 04-9]; token-selected feed-forward sets [04-11]; ninth block RAM bit [09-7] | See the topic files | See the topic files | the items above | DDR2, ninth bit: yes |

Not recommended in the topic files: Mamba, xLSTM, RWKV-4 and self-speculative decoding on this CPU; early exit
(code tasks fail in the literature); adaptive softmax (at most 3 to 4% of a token); packing two multiplies in one
DSP48E1; the Digilent Ram2Ddr adapter as a weight stream; skipping output rows of masked tokens (about 0.14% of a
token); curriculum by difficulty; an accumulator-aware weight penalty while rows stay at or below 511 products.

### 11.3 What can run in parallel

- Track A (data): F4, then rank 5. Independent of the model code.
- Track B (reference and simulator): F1, F2, F5, then ranks 1 and 3. Independent of the data.
- Track C (tokenizer and decoding): rank 2 and rank 9 share the tokenizer work and should be done together; they
  need only a corpus sample.
- Track D (float shape runs): ranks 4, 10 and 11 are float trainings; the machine allows one training at a time,
  so they queue, but they do not wait for QAT.
- Track E (QAT): ranks 6, 7 and 8 need F1 and a float checkpoint; rank 8 is a measurement that can start as soon
  as F1 exists.
- Track F (VHDL): rank 12 depends only on F1/F2 for its acceptance test and can proceed beside all software work.
- Rank 13 comes last: it needs the int8 model, the filter of rank 9 (to see which failures remain) and the split
  practice set.

### 11.4 Disagreements between topic files

- Parameter range of Tier A: topic 01 corrected "260k to 400k" to "250k to 400k" (the CONTEXT.md value); the
  headings of topics 02 to 09 still say 260k to 400k.
- Cycles per token and attention share: CONTEXT.md says about 1.0 M cycles and attention "about 60%" of a layer;
  topic 05 gets 1.10 M (1,100,668) with a two-pass softmax and attention at 74% of a layer (56% for scores plus the
  softmax pass in its summary); topic 09 uses 423k + 338k + 239k = 1.0 M. Baseline speed is "about 100 tokens/s" in
  CONTEXT.md and 91 tokens/s at T = 256 in topic 05.
- Per-row control after a row instruction: 24 to about 4 cycles in topics 04 and 09 (0.28 M cycles, -28%), 24 to 8
  in topic 05 (-21%); topic 09's smaller fusion step saves 8 cycles per row (-11%).
- Stacked speed estimates are not the same thing: topic 04 (heads + row instruction + element stream) about 0.5 M
  cycles, recomputed 0.46 M; topic 05 (heads + one-pass softmax + windows, no CPU change) 546k at T = 256; topic 09
  (hardware only, 4 lanes) 162k. None is simulated.
- 16-bit residual stream: topic 02 keeps it as a conditional option (about +2 to 4% cycles, idea 9); topic 03 lists
  it under "not recommended" (doubles the cost of every stream that reads it).
- Softmax normaliser: topic 02 recommends a reciprocal table after the value mix and quotes a 0.029 nats/char cost
  for a bare power-of-two normaliser; topics 03 and 04 keep "power-of-two shift" and "none" as variants to test.
- Grammar speed shortcuts: topic 05 (idea 9) expects -23% on forced tokens and up to -4.1% from skipped output
  rows; topic 06 measures 14.5 maskable rows (about 0.14% of a token) and expects no gain from fast-forward because
  forced sequences are already single tokens.
- Tokenizer flags: topic 05 recommends free BPE without byte fallback, pieces spanning spaces; topic 06's structural
  vocabulary is +2.2% longer than exactly that baseline at 512 tokens (and -0.1% against llama2.c-style BPE).
- Context length: topic 05 argues for T = 512 (95% of HumanEval tasks fit); topic 04 reports a model that got worse
  with context 85 than 20 on stories and KV cache of 163,840 bytes at T = 512; topic 05 notes that wide shapes at
  T = 512 exceed 450 KB without windows or MQA.
- Fewer bits per weight: topic 04 (idea 13) lists ternary or 4-bit weights; topic 03 does not recommend them (twice
  the hidden size for ternary; the multiplier is already 8 x 8); topic 09 (idea 7) proposes 9-bit weights, which
  leaves the int8-only rule and is overflow-safe only to 255 terms.
- Curriculum and generated data: topic 01 (ideas 1 and 6) proposes own generated tasks, a grammar-based generator
  and a filtered OpenCodeInstruct subset; topic 03 finds curriculum by difficulty mostly negative; topic 08 and the
  `CLAUDE.md` rule of 2026-11-13 19:05 forbid AI-written template text and any synthetic set without a named
  generator whose licence allows training. Topic 01 itself notes that OpenCodeInstruct is LLM-generated; topic 08
  did not examine that dataset, so it is not usable until it passes the topic 08 rule. Topic 03 idea 13 ("corpus
  written or filtered by a strong code model") is likewise limited to generators that topic 08 clears
  (Apache-2.0 Qwen2.5-Coder).
- Evaluation tasks and weight editing: topic 07 scores edits by pass/fail on practice tasks; topic 08 notes that
  doing so on Claude-written tasks is arguably training on outputs, so the editing stage is limited to human-written
  or licensed tasks.
- Decoding: topic 05 makes greedy the default; topic 01 proposes many samples per task on the board. Both hold if
  the result is labelled pass@k.
- Cost of going integer in the MAX 10 project: 0.021 bits per character (topic 02) and 0.057 nats per token
  (topic 04) are the same measurement in two units (0.622 to 0.643 bits/char; 1.647 to 1.704 nats/token).

## 12. Topic 7 status

"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights" was found: it is a blog
post by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL), published November 2, 2026 at
https://rikiyat.github.io/blog/weight-space-programming/, read in full together with its released logs
(GitHub RikiyaT/lm_understand-logs) and the edited model card on Hugging Face [VERIFIED].

Evidence:
- The numbers we were given are in the post: 131,072 of the 0.8 billion weights of Qwen3.5-0.8B changed, MATH-500
  7.9% to 48.1%, gradients and training forbidden by the harness [VERIFIED].
- It is not on arXiv: the API queries `all:"Sparks of Weight-Space Programming"` and `au:Takehi AND au:Khattab`
  returned 0 results, repeated by the fact-check; the citation block says `note = {Blog post}`; no peer review
  [VERIFIED].
- Two outside parties re-measured the edited checkpoint at 51.0% and 51.7% (https://arxiv.org/abs/2611.14210);
  Hu et al. measure the unedited model at 8.35% (https://arxiv.org/abs/2611.09382) [VERIFIED].
- The base model card warns of the thinking loops the edit removes
  (https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/README.md#L856) [VERIFIED].
- Not verified by us: the bit-for-bit rebuild of the edit (`apply_edit.py` was not run); the Experiment 8 agent
  harness is not released and the editing model is a closed commercial model.

What it means for our weight-editing phase:
1. Plan it as a repair stage for failures we can measure (generation loops and missing end of output, mean shifts
   after quantisation, saturation), not as a source of coding skill. The edit worked where 93.0% of answers looped
   (to 33.1%) and gave at most +0.9 points on Gemma 3 270M, where 8.5% did; accuracy on originally finished outputs
   was unchanged.
2. Copy the measurement protocol: original and edited int8 model on the same practice tasks, dose-response, a
   random-direction control, freeze the edit, confirm on untouched practice tasks (the paired rule of section 11).
   With greedy decoding the int8 simulator is deterministic, so the pairing is exact.
3. Start with the failure census. Whether one failure class dominates after grammar-constrained decoding, whether a
   failure is one direction at width 64, and whether anything is left after bit-exact QAT are all unknown.
4. Use the forward-only tools with the longest record for the int8 stage: accumulator preload as a bias, layer-wise
   integer refits (OBC, GPTQ, COMQ), direction removal folded into the weights (our synthetic check: cosine 0.98 or
   higher after int8 rounding at width 64; a strength-6 edit on 16 columns changes about 48% of the matrix norm, so
   start at strength 1), zeroth-order search over small integer groups. Their record is at 2 to 4 bits on far larger
   models; none is tested at our scale.
5. An agent session in the style of Experiment 8 is a replication at about 1/3,000 of the source model's size with a
   high chance of no gain (6 of 16 follow-up trajectories ended without an edit); a null result is reported as such.
6. Fitness for any edit comes from human-written or licensed practice tasks only (topic 08), never from test sets.

## 13. Sources

One entry per URL, merged from the source tables (section 7) of the nine topic files; the number in brackets is the topic file that cites it. Licences are those recorded in the topic files for code, models and datasets; for a paper row the topic file's licence cell (licence of its code or model, where given) is kept. Entries named "artefact of" are repositories or models that a topic file names inside the licence cell of a paper. Sources cited only in running text of a topic file are in that file.

### Papers (232)

- A Comprehensive Study of Knowledge Editing for Large Language Models (survey). https://arxiv.org/abs/2401.01286. [07]
- A Framework for Accelerating Transformer Inference on RISC-V for Edge AI. https://arxiv.org/abs/2610.08688. [04]
- A Multimodal Automated Interpretability Agent (MAIA). https://arxiv.org/abs/2404.14394. Licence: code https://github.com/multimodal-interpretability/maia: none detected. [07]
- A Simple and Effective Pruning Approach for Large Language Models (Wanda). https://arxiv.org/abs/2306.11695. Licence: code https://github.com/locuslab/wanda: MIT. [07]
- A Study on ReLU and Softmax in Transformer. https://arxiv.org/abs/2302.06461. [02]
- A Syntactic Neural Model for General-Purpose Code Generation (configuration paragraph read; row added in the fact check). https://arxiv.org/abs/1704.01696. [06]
- A Systematic Evaluation of Large Language Models of Code. https://arxiv.org/abs/2202.13169. Licence: code MIT (https://github.com/VHellendoorn/Code-LMs); no licence on the hub card [VERIFIED] (both licence fields). [01]
- A Tiny Transformer for Low-Power Arrhythmia Classification on Microcontrollers. https://arxiv.org/abs/2402.10748. [03]
- A White Paper on Neural Network Quantization. https://arxiv.org/abs/2106.08295. [03]
- A2Q+: Improving Accumulator-Aware Weight Quantization. https://arxiv.org/abs/2401.10432. [03]
- A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance. https://arxiv.org/abs/2308.13504. Licence: code in https://github.com/Xilinx/brevitas : BSD 3-clause style text plus third-party notices in LICENSE (GitHub reports NOASSERTION). [02]
- Accumulator-Aware Post-Training Quantization for Large Language Models. https://arxiv.org/abs/2409.17092. [02]
- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. https://arxiv.org/abs/1909.11942. [05]
- An Empirical Study of Mamba-based Language Models. https://arxiv.org/abs/2406.07887. [05]
- Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. https://arxiv.org/abs/1905.09418. [07]
- Analyzing Transformers in Embedding Space. https://arxiv.org/abs/2209.02535. [07]
- Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy. https://arxiv.org/abs/1711.05852. [03]
- Are Deep Neural Networks the Best Choice for Modeling Source Code?. https://web.cs.ucdavis.edu/~devanbu/isDLgood.pdf. Licence: SLP-Core: MIT (https://github.com/SLP-team/SLP-Core) [VERIFIED] (SLP-Core licence). [01]
- Are Sixteen Heads Really Better than One?. https://arxiv.org/abs/1905.10650. [05, 07]
- Attention Is All You Need. https://arxiv.org/abs/1706.03762. [05]
- Attention, Dynamic Wave and Hybrid Sequence Mixers in Language Models. https://arxiv.org/abs/2610.39854. [05]
- Automatic Generation of Python Programs Using Context-Free Grammars. https://arxiv.org/abs/2403.06503. Licence: https://github.com/MarwaNair/TinyPy-Generator has no licence file [VERIFIED] (repository has no licence). [01]
- Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. https://arxiv.org/abs/2308.02019. Licence: code https://github.com/timinar/BabyLlama : MIT. [03]
- BabyLlama-2: Ensemble-Distilled Models Consistently Outperform Teachers With Limited Data. https://arxiv.org/abs/2409.17312. [03]
- Beam Search, Self-Consistency, and the Limits of Inference-Time Scaling for Grammar-Constrained Text-to-SQL in Small Language Models (abstract only). https://arxiv.org/abs/2608.25761. [06]
- Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code. https://arxiv.org/abs/2003.07914. Licence: not checked. [01, 05]
- Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. https://arxiv.org/abs/1602.02830. [02, 03]
- BinaryBERT: Pushing the Limit of BERT Quantization. https://arxiv.org/abs/2012.15701. [03]
- BitNet b1.58 2B4T Technical Report. https://arxiv.org/abs/2504.12285. [03]
- BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks. https://arxiv.org/abs/2407.09527. Licence: code https://github.com/schneiderkamplab/bitlinear : MIT. [03]
- BitNet: Scaling 1-bit Transformers for Large Language Models. https://arxiv.org/abs/2310.11453. [03]
- Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models. https://arxiv.org/abs/2601.09719. Licence: code https://github.com/MLAI-Yonsei/BHyT : Apache-2.0. [02]
- BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction. https://arxiv.org/abs/2102.05426. Licence: code https://github.com/yhhhli/BRECQ: MIT. [07]
- ByT5: Towards a token-free future with pre-trained byte-to-byte models. https://arxiv.org/abs/2105.13626. [05]
- Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids?. https://arxiv.org/abs/2611.08070. [07]
- CERT: Continual Pre-Training on Sketches for Library-Oriented Code Generation. https://arxiv.org/abs/2206.06888. Licence: PyCodeGPT: AFL-3.0 on the hub card, MIT on https://github.com/microsoft/PyCodeGPT [VERIFIED] (both licence fields). [01]
- CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs. https://arxiv.org/abs/1801.06601. [02]
- CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis. https://arxiv.org/abs/2203.13474. Licence: models BSD-3-Clause (https://huggingface.co/Salesforce/codegen-350M-mono). [01]
- CodeIPPrompt: Intellectual Property Infringement Assessment of Code Language Models. https://proceedings.mlr.press/v202/yu23g.html. [08]
- CodeSearchNet Challenge: Evaluating the State of Semantic Code Search. https://arxiv.org/abs/1909.09436. [08]
- CodeT5+: Open Code Large Language Models for Code Understanding and Generation. https://arxiv.org/abs/2305.07922. Licence: BSD-3-Clause. [01]
- Competition-Level Code Generation with AlphaCode. https://arxiv.org/abs/2203.07814. Licence: models not released. [01, 08]
- Compiling Known Rules into Compact Neural Agents: Weight-Space Programming on Unseen Games. https://arxiv.org/abs/2611.16981. [07]
- Compressing Pre-trained Models of Code into 3 MB. https://arxiv.org/abs/2208.07120. [03]
- COMQ: A Backpropagation-Free Algorithm for Post-Training Quantization. https://arxiv.org/abs/2403.07134. [07]
- Confident Adaptive Language Modeling. https://arxiv.org/abs/2207.07061. [05]
- ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters. https://arxiv.org/abs/2402.10930. Licence: code https://github.com/ReaLLMASIC/ConSmax : Apache-2.0. [02]
- Constrained Decoding Eliminates Structural Failures in Small LLMs but Reveals a Scale-Dependent Semantic Gap (abstract only). https://arxiv.org/abs/2609.23742. [06]
- Constrained Decoding for Fill-in-the-Middle Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars. https://arxiv.org/abs/2402.17988. [06]
- Continual Quantization-Aware Pre-Training: When to transition from 16-bit to 1.58-bit pre-training for BitNet language models?. https://arxiv.org/abs/2502.11895. [03]
- Curriculum learning for language modeling. https://arxiv.org/abs/2108.02170. [03]
- Curriculum Learning for Small Code Language Models. https://arxiv.org/abs/2407.10194. Licence: code and 1 M checkpoints Apache-2.0 (https://github.com/kamel-yamani/CurriculumLearningForSmallCodeLanguageModels); data on Kaggle, licence not checked [VERIFIED] (repository licence). [01, 08]
- Data-Free Quantization Through Weight Equalization and Bias Correction. https://arxiv.org/abs/1906.04721. Licence: implementation in https://github.com/qualcomm/aimet: licence file present, not classified by the GitHub API. [07]
- Deduplicating Training Data Makes Language Models Better. https://arxiv.org/abs/2107.06499. Licence: code https://github.com/google-research/deduplicate-text-datasets : Apache-2.0, archived. [03]
- DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference. https://arxiv.org/abs/2004.12993. [05]
- Deeploy: Enabling Energy-Efficient Deployment of Small Language Models On Heterogeneous Microcontrollers. https://arxiv.org/abs/2408.04413. Licence: code https://github.com/pulp-platform/Deeploy : Apache-2.0 for most source files and MIT for some scripts according to the README; `LICENSES/` holds Apache-2.0.txt, MIT.txt and CC-BY-ND-4.0.txt; the GitHub API reports no single licence [VERIFIED]. [02, 03, 04]
- DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence. https://arxiv.org/abs/2401.14196. [08]
- Differentiable Model Compression via Pseudo Quantization Noise. https://arxiv.org/abs/2104.09987. Licence: code https://github.com/facebookresearch/diffq : CC BY-NC 4.0 (licence file), archived. [03]
- Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks. https://arxiv.org/abs/1908.05033. [03]
- Diffusion On Syntax Trees For Program Synthesis (architecture section read; row added in the fact check). https://arxiv.org/abs/2405.20519. [06]
- DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. https://arxiv.org/abs/1910.01108. [03]
- Does Localization Inform Editing?. https://arxiv.org/abs/2301.04213. [07]
- Does Your Optimizer Care How You Normalize? Normalization-Optimizer Coupling in LLM Training. https://arxiv.org/abs/2604.01563. [02]
- DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients. https://arxiv.org/abs/1606.06160. [03]
- Double MAC: Doubling the Performance of Convolutional Neural Networks on Modern FPGAs, DATE 2017, pp. 890-893, doi 10.23919/DATE.2017.7927113 (added by the fact-check). https://past.date-conference.com/proceedings-archive/2017/pdf/0348.pdf. [09]
- DSP-Packing: Squeezing Low-precision Arithmetic into FPGA DSP Blocks, arXiv:2203.11028. https://arxiv.org/abs/2203.11028. [09]
- Editing Large Language Models: Problems, Methods, and Opportunities (survey). https://arxiv.org/abs/2305.13172. Licence: framework https://github.com/zjunlp/EasyEdit: MIT. [07]
- Editing Models with Task Arithmetic. https://arxiv.org/abs/2212.04089. Licence: code https://github.com/mlfoundations/task_vectors: none detected. [07]
- Efficient Grammar-Constrained Decoding via Parser Stack Classification (abstract only). https://arxiv.org/abs/2608.03065. [06]
- Efficient Guided Generation for Large Language Models (Outlines). https://arxiv.org/abs/2307.09702. [06]
- Efficient Softmax Approximation for Deep Neural Networks with Attention Mechanism. https://arxiv.org/abs/2111.10770. [02]
- Efficient softmax approximation for GPUs. https://arxiv.org/abs/1609.04309. [05]
- Eliciting Latent Predictions from Transformers with the Tuned Lens. https://arxiv.org/abs/2303.08112. Licence: code https://github.com/AlignmentResearch/tuned-lens: MIT. [07]
- ELiTeFormer: An Efficient Transformer for FPGAs (seen in search, abstract only). https://arxiv.org/abs/2607.03652. [04]
- EmbBERT: Attention Under 2 MB Memory. https://arxiv.org/abs/2502.10001. [03]
- Energy-Efficient Low-Latency Signed Multiplier for FPGA-Based Hardware Accelerators, IEEE Embedded Systems Letters 13(2), doi 10.1109/LES.2020.2995053. https://grk2767.tu-dresden.de/files/Images/people/chair-pd/Papers/ESL_acc_multiplier.pdf. [09]
- Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. https://arxiv.org/abs/1308.3432. [03]
- Evaluating Large Language Models Trained on Code. https://arxiv.org/abs/2107.03374. Licence: HumanEval: MIT (https://github.com/openai/human-eval); Codex weights not released [VERIFIED] (HumanEval licence). [01, 05, 08]
- Evolution Strategies as a Scalable Alternative to Reinforcement Learning. https://arxiv.org/abs/1703.03864. [07]
- Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning. https://arxiv.org/abs/2509.24372. [07]
- Evolutionary fine tuning of quantized convolution-based deep learning models. https://arxiv.org/abs/2605.05228. [07]
- Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations. https://arxiv.org/abs/2611.09382. Licence: code https://github.com/WSP-Scope/WSP-Scope-Project: none detected. [07]
- Fast and Memory-Efficient Neural Code Completion. https://arxiv.org/abs/2004.13651. Licence: not checked. [01]
- Fast Transformer Decoding: One Write-Head is All You Need. https://arxiv.org/abs/1911.02150. [05]
- FIND: A Function Description Benchmark for Evaluating Interpretability Methods. https://arxiv.org/abs/2309.03886. [07]
- Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora. https://arxiv.org/abs/2504.08165. [03]
- Findings of the Second BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora. https://arxiv.org/abs/2412.05149. [03]
- Fine-Tuning Language Models with Just Forward Passes (MeZO). https://arxiv.org/abs/2305.17333. Licence: code https://github.com/princeton-nlp/MeZO: MIT. [07]
- Fine-tuning Quantized Neural Networks with Zeroth-order Optimization (QZO). https://arxiv.org/abs/2505.13430. [07]
- FlashNorm: Fast Normalization for Transformers. https://arxiv.org/abs/2407.09577. [02]
- FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer. https://arxiv.org/abs/2111.13824. Licence: code https://github.com/megvii-research/FQ-ViT : Apache-2.0. [02, 03]
- Fully Quantized Transformer for Machine Translation. https://arxiv.org/abs/1910.10485. [03]
- GLU Variants Improve Transformer. https://arxiv.org/abs/2002.05202. [05]
- GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration. https://arxiv.org/abs/2504.02692. [07]
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. https://arxiv.org/abs/2210.17323. Licence: code https://github.com/IST-DASLab/gptq: Apache-2.0. [07]
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. https://arxiv.org/abs/2305.13245. [05]
- Grammar-Aligned Decoding. https://arxiv.org/abs/2405.21047. [06]
- Grammar-Based Code Representation: Is It a Worthy Pursuit for LLMs? (abstract only). https://arxiv.org/abs/2503.05507. [06]
- Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning. https://arxiv.org/abs/2305.13971. [06]
- Grammar-Constrained Decoding Makes Large Language Models Better Logical Parsers (abstract only; row added in the fact check). https://aclanthology.org/2025.acl-industry.34/. [06]
- Guiding Language Models of Code with Global Context using Monitors. https://arxiv.org/abs/2306.10763. [06]
- Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation (Domino). https://arxiv.org/abs/2403.06988. [06]
- HAAN: A Holistic Approach for Accelerating Normalization Operations in Large Language Models. https://arxiv.org/abs/2502.11832. [02]
- Hardware-Oriented Approximations of Softmax and RMSNorm for Efficient Transformer Inference (Micromachines 17(1):84). https://pmc.ncbi.nlm.nih.gov/articles/PMC12843610/. Licence: article CC BY. [02]
- HAWQ-V3: Dyadic Neural Network Quantization. https://arxiv.org/abs/2011.10680. Licence: code https://github.com/Zhen-Dong/HAWQ : MIT. [02]
- HLSTransform: Energy-Efficient Llama 2 Inference on FPGAs Via High Level Synthesis. https://arxiv.org/abs/2405.00738. Licence: paper; code https://github.com/HLSTransform/submission MIT. [04]
- I-BERT: Integer-only BERT Quantization. https://arxiv.org/abs/2101.01321. Licence: code https://github.com/kssteven418/I-BERT : MIT. [02, 03]
- I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models. https://arxiv.org/abs/2405.17849. [02, 03]
- I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference. https://arxiv.org/abs/2207.01405. Licence: code https://github.com/zkkli/I-ViT : Apache-2.0. [02, 03]
- Improved Knowledge Distillation via Teacher Assistant. https://arxiv.org/abs/1902.03393. [03]
- Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights. https://arxiv.org/abs/1702.03044. [03]
- IntAttention: A Fully Integer Attention Pipeline for Efficient Edge Inference. https://arxiv.org/abs/2511.21513. [02]
- ITA: An Energy-Efficient Attention and Softmax Accelerator for Quantized Transformers. https://arxiv.org/abs/2307.03493. [02]
- IWIBench: Evaluating Gradient-Free Model Driven Intra-weight Injection. https://arxiv.org/abs/2611.08647. [07]
- KDLSQ-BERT: A Quantized Bert Combining Knowledge Distillation with Learned Step Size Quantization. https://arxiv.org/abs/2101.05938. [03]
- Language Models are Few-Shot Learners (row added in the fact-check; used only for the model shapes in its Table 2.1). https://arxiv.org/abs/2005.14165. Licence: paper only [VERIFIED]. [01]
- "Language Models are the new Doom" (short paper). https://sol.sbc.org.br/index.php/eramiars/article/download/39470/39242. [04]
- LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. https://arxiv.org/abs/2404.16710. Licence: code facebookresearch/LayerSkip: Creative Commons Attribution-NonCommercial 4.0 text in LICENSE (GitHub reports no SPDX identifier). [05]
- LEACE: Perfect linear concept erasure in closed form. https://arxiv.org/abs/2306.03819. Licence: code https://github.com/EleutherAI/concept-erasure: MIT. [07]
- Learned Step Size Quantization. https://arxiv.org/abs/1902.08153. [03]
- Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow (CoNaLa). https://arxiv.org/abs/1805.08949. [08]
- Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models (abstract only). https://arxiv.org/abs/2408.02442. [06]
- Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis. https://arxiv.org/abs/1805.04276. [06]
- LlamaF: An Efficient Llama2 Architecture Accelerator on Embedded FPGAs. https://arxiv.org/abs/2409.11424. [04]
- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models. https://arxiv.org/abs/2305.17888. Licence: code https://github.com/facebookresearch/LLM-QAT : CC BY-NC 4.0 (licence file). [03]
- Locating and Editing Factual Associations in GPT (ROME). https://arxiv.org/abs/2202.05262. Licence: code https://github.com/kmeng01/rome: MIT. [07]
- Longformer: The Long-Document Transformer. https://arxiv.org/abs/2004.05150. [05]
- Lost in Space: Finding the Right Tokens for Structured Output (abstract only; leading-whitespace tokens help small models most). https://arxiv.org/abs/2502.14969. [06]
- Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens. https://arxiv.org/abs/2411.17691. [03]
- LSQ+: Improving low-bit quantization through learnable offsets and better initialization. https://arxiv.org/abs/2004.09576. [03]
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces. https://arxiv.org/abs/2312.00752. Licence: code state-spaces/mamba: Apache-2.0 (not used). [05]
- Mass-Editing Memory in a Transformer (MEMIT). https://arxiv.org/abs/2210.07229. Licence: code https://github.com/kmeng01/memit: MIT. [07]
- MCUBERT: Memory-Efficient BERT Inference on Commodity Microcontrollers. https://arxiv.org/abs/2410.17957. [03, 04]
- MCUFormer: Deploying Vision Transformers on Microcontrollers with Limited Memory. https://arxiv.org/abs/2310.16898. Licence: paper; code none detected. [04]
- Measuring Coding Challenge Competence With APPS. https://arxiv.org/abs/2105.09938. [08]
- Mixed Signals: Weight-Space Programming for EEG and ECG. https://arxiv.org/abs/2611.13758. [07]
- MobiEdit: Resource-efficient Knowledge Editing for Personalized On-device LLMs [fact-check: added]. https://arxiv.org/abs/2506.13772. [07]
- MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. https://arxiv.org/abs/2004.02984. [02, 03]
- MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases. https://arxiv.org/abs/2402.14905. [03, 05]
- Model Editing at Scale leads to Gradual and Catastrophic Forgetting. https://arxiv.org/abs/2401.07453. [07]
- Model Editing for LLMs4Code: How Far are We?. https://arxiv.org/abs/2411.06638. [07]
- Network Quantization with Element-wise Gradient Scaling. https://arxiv.org/abs/2104.00903. [03]
- Neuron Patching: Semantic-based Neuron-level Language Model Repair for Code Generation (MINT). https://arxiv.org/abs/2312.05356. [07]
- NN-LUT: Neural Approximation of Non-Linear Operations for Efficient Transformer Inference. https://arxiv.org/abs/2112.02191. [02]
- On Leakage of Code Generation Evaluation Datasets. https://arxiv.org/abs/2407.07565. [08]
- On the Mathematical Relationship Between Layer Normalization and Dynamic Activation Functions. https://arxiv.org/abs/2503.21708. [02]
- On the Naturalness of Software. https://people.inf.ethz.ch/suz/publications/natural.pdf. [01]
- On the quantization of recurrent neural networks. https://arxiv.org/abs/2101.05453. [05]
- On Using Very Large Target Vocabulary for Neural Machine Translation. https://arxiv.org/abs/1412.2007. [05]
- One Size Does Not Fit All: Setting Inference Depth from the Questions a Deployment Actually Asks. https://arxiv.org/abs/2609.14144. [05]
- OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs (metadata and dataset card only, paper not read). https://arxiv.org/abs/2504.04030. Licence: dataset CC BY 4.0. [01]
- Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning (OBC). https://arxiv.org/abs/2208.11580. Licence: code https://github.com/IST-DASLab/OBC: none detected. [07]
- Optimizing the Deployment of Tiny Transformers on Low-Power MCUs. https://arxiv.org/abs/2404.02945. [04]
- Oscillation-free Quantization for Low-bit Vision Transformers. https://arxiv.org/abs/2302.02210. [03]
- Overcoming Oscillations in Quantization-Aware Training. https://arxiv.org/abs/2203.11086. Licence: code https://github.com/qualcomm-ai-research/oscillations-qat : BSD-3-Clause-Clear. [03]
- PACT: Parameterized Clipping Activation for Quantized Neural Networks. https://arxiv.org/abs/1805.06085. [03]
- Pareto-Optimal Quantized ResNet Is Mostly 4-bit. https://arxiv.org/abs/2105.03536. [03]
- PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models. https://arxiv.org/abs/2109.05093. [06]
- Program Synthesis with Large Language Models. https://arxiv.org/abs/2108.07732. Licence: MBPP: CC BY 4.0 (https://huggingface.co/datasets/google-research-datasets/mbpp); models not released [VERIFIED] (MBPP licence). [01, 08]
- Progress measures for grokking via mechanistic interpretability [fact-check: added]. https://arxiv.org/abs/2301.05217. [07]
- Pythia: AI-assisted Code Completion System. https://arxiv.org/abs/1912.00742. Licence: not released. [01]
- Q8BERT: Quantized 8Bit BERT. https://arxiv.org/abs/1910.06188. [03]
- QKD: Quantization-aware Knowledge Distillation. https://arxiv.org/abs/1911.12491. [03]
- Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models. https://arxiv.org/abs/2403.04811. [08]
- Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing. https://arxiv.org/abs/2306.12929. Licence: code https://github.com/Qualcomm-AI-research/outlier-free-transformers : BSD-3-Clause-Clear. [02, 03]
- Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. https://arxiv.org/abs/1712.05877. [02, 03]
- Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost (QES). https://arxiv.org/abs/2602.03120. Licence: code https://github.com/dibbla/Quantized-Evolution-Strategies: none detected. [07]
- Quantized Neural Networks for Low-Precision Accumulation with Guaranteed Overflow Avoidance. https://arxiv.org/abs/2301.13376. [03]
- Quantizing deep convolutional networks for efficient inference: A whitepaper. https://arxiv.org/abs/1806.08342. [03]
- Quasi-Recurrent Neural Networks. https://arxiv.org/abs/1611.01576. [05]
- QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models. https://arxiv.org/abs/2502.12346. [07]
- Qwen2.5-Coder Technical Report. https://arxiv.org/abs/2409.12186. Licence: Qwen2.5-Coder-0.5B: Apache-2.0. [01]
- Refusal in Language Models Is Mediated by a Single Direction. https://arxiv.org/abs/2406.11717. Licence: code https://github.com/andyrdt/refusal_direction: Apache-2.0. [07]
- REPAIR: REnormalizing Permuted Activations for Interpolation Repair. https://arxiv.org/abs/2211.08403. [07]
- Replacing softmax with ReLU in Vision Transformers. https://arxiv.org/abs/2309.08586. [02, 03, 05]
- Research on LLM Acceleration Using the High-Performance RISC-V Processor "Xiangshan" (Nanhu Version) Based on the Open-Source Matrix Instruction Set Extension (Vector Dot Product). https://arxiv.org/abs/2409.00661. [04]
- RWKV: Reinventing RNNs for the Transformer Era. https://arxiv.org/abs/2305.13048. Licence: code BlinkDL/RWKV-LM: Apache-2.0 (not used). [05]
- Scaling Laws for Precision. https://arxiv.org/abs/2411.04330. [03]
- Sequence-Level Knowledge Distillation. https://arxiv.org/abs/1606.07947. [03]
- Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context. https://arxiv.org/abs/1805.04623. [05]
- Simple Recurrent Units for Highly Parallelizable Recurrence. https://arxiv.org/abs/1709.02755. Licence: code taolei87/sru: MIT (not used). [05]
- Single Headed Attention RNN: Stop Thinking With Your Head. https://arxiv.org/abs/1911.11423. [05]
- SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization. https://arxiv.org/abs/2405.11582. Licence: code https://github.com/xinghaochen/SLAB : no licence detected by the GitHub API. [02]
- SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model. https://arxiv.org/abs/2502.02737. Licence: Apache-2.0; smollm-corpus ODC-BY. [01]
- Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers. https://arxiv.org/abs/2103.09301. [02, 03]
- SOLE: Hardware-Software Co-design of Softmax and LayerNorm for Efficient Transformer Inference. https://arxiv.org/abs/2510.17189. [02]
- Sparse Attention with Linear Units. https://arxiv.org/abs/2104.07012. [02]
- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. https://arxiv.org/abs/2301.00774. Licence: code https://github.com/IST-DASLab/sparsegpt: Apache-2.0. [07]
- StarCoder: may the source be with you!. https://arxiv.org/abs/2305.06161. [08]
- Steering Language Models With Activation Engineering. https://arxiv.org/abs/2308.10248. [07]
- Steering Llama 2 via Contrastive Activation Addition. https://arxiv.org/abs/2312.06681. Licence: code https://github.com/nrimsky/CAA: MIT. [07]
- Steering Under Compression: Dose-Response, Capability Cost, and Failure Asymmetry in Quantized LLMs [fact-check: added]. https://arxiv.org/abs/2609.06473. [07]
- Stronger Normalization-Free Transformers. https://arxiv.org/abs/2512.10938. [02]
- Synchromesh: Reliable code generation from pre-trained language models. https://arxiv.org/abs/2201.11227. [06]
- SynCode: LLM Generation with Grammar Augmentation. https://arxiv.org/abs/2403.01632. [06]
- TernaryBERT: Distillation-aware Ultra-low Bit BERT. https://arxiv.org/abs/2009.12812. [03]
- Test-Time Model Adaptation with Only Forward Passes (FOA). https://arxiv.org/abs/2404.01650. [07]
- Textbooks Are All You Need. https://arxiv.org/abs/2306.11644. Licence: phi-1: MIT; phi-1-small and CodeExercises not released. [01, 03, 08]
- The Adverse Effects of Code Duplication in Machine Learning Models of Code. https://arxiv.org/abs/1812.06469. [03]
- The Alignment Problem in Constrained Code Generation. https://arxiv.org/abs/2606.21619. [06]
- The Depth-to-Width Interplay in Self-Attention. https://arxiv.org/abs/2006.12467. [05]
- The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. https://arxiv.org/abs/2402.17764. [03]
- The Stack: 3 TB of permissively licensed source code. https://arxiv.org/abs/2211.15533. [08]
- The Vault: A Comprehensive Multilingual Dataset for Advancing Code Understanding and Generation. https://arxiv.org/abs/2305.06156. [08]
- Theory, Analysis, and Best Practices for Sigmoid Self-Attention. https://arxiv.org/abs/2409.04431. Licence: code https://github.com/apple-aiml-research/ml-sigmoid-attention : custom Apple licence text (GitHub reports NOASSERTION). [02, 03, 05]
- TinyBERT: Distilling BERT for Natural Language Understanding. https://arxiv.org/abs/1909.10351. [03]
- TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices. https://arxiv.org/abs/2311.01759. [04]
- TinyStories: How Small Can Language Models Be and Still Speak Coherent English?. https://arxiv.org/abs/2305.07759. Licence: dataset licence not checked here. [03, 05]
- Towards Fully 8-bit Integer Inference for the Transformer Model. https://arxiv.org/abs/2009.08034. [02]
- Towards the Law of Capacity Gap in Distilling Language Models. https://arxiv.org/abs/2311.07052. [03]
- Tracking Universal Features Through Fine-Tuning and Model Merging (metadata only, paper not read). https://arxiv.org/abs/2410.12391. Licence: nilq/baby-python: MIT. [01]
- Tracr: Compiled Transformers as a Laboratory for Interpretability. https://arxiv.org/abs/2301.05062. Licence: code https://github.com/google-deepmind/tracr: Apache-2.0. [07]
- Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks. https://arxiv.org/abs/1903.08066. Licence: code https://github.com/Xilinx/graffitist : BSD 3-Clause text in LICENSE (GitHub reports NOASSERTION). [02, 03]
- Training Language Models on Synthetic Edit Sequences Improves Code Synthesis. https://arxiv.org/abs/2410.02749. Licence: TinyCodeLM: Apache-2.0. [01]
- Training with Quantization Noise for Extreme Model Compression. https://arxiv.org/abs/2004.07320. [03]
- Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. https://arxiv.org/abs/2203.14680. [07]
- Transformers without Normalization. https://arxiv.org/abs/2503.10622. Licence: code https://github.com/jiachenzhu/DyT : MIT. [02, 03]
- Tree-Ising: Repairing Language Models with Bit-Level Ising Edits [fact-check: added]. https://arxiv.org/abs/2611.14210. Licence: code https://github.com/TheQuantumGOAT/Tree-Ising: none detected. [07]
- Type-Constrained Code Generation with Language Models. https://arxiv.org/abs/2504.09246. [06]
- Understanding and Overcoming the Challenges of Efficient Transformer Quantization. https://arxiv.org/abs/2109.12948. [02, 03]
- Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets. https://arxiv.org/abs/1903.05662. [03]
- Unsupervised Translation of Programming Languages (TransCoder; abstract and tokenizer source read). https://arxiv.org/abs/2006.03511. [06]
- Up or Down? Adaptive Rounding for Post-Training Quantization (AdaRound). https://arxiv.org/abs/2004.10568. [07]
- Using the Output Embedding to Improve Language Models. https://arxiv.org/abs/1608.05859. [05]
- Weight Oracles: Reading Neural Network Weights with Language Models. https://arxiv.org/abs/2610.07334. [07]
- Weight-space programming for auditing clinical decision networks. https://arxiv.org/abs/2611.05341. [07]
- Weight-Space Programming of Variational Quantum Circuits. https://arxiv.org/abs/2611.11537. [07]
- Well-Read Students Learn Better: On the Importance of Pre-training Compact Models. https://arxiv.org/abs/1908.08962. [03]
- Were RNNs All We Needed?. https://arxiv.org/abs/2410.01201. Licence: third-party code lucidrains/minGRU-pytorch: MIT (not used). [05]
- When Does Removing LayerNorm Help? Activation Bounding as a Regime-Dependent Implicit Regularizer. https://arxiv.org/abs/2604.23434. [02]
- XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models. https://arxiv.org/abs/2411.15100. [06]
- xLSTM: Extended Long Short-Term Memory. https://arxiv.org/abs/2405.04517. Licence: code NX-AI/xlstm: Apache-2.0 per GitHub (not used). [05]

### Models (46)

- amd/AMD-Llama-135m-code (draft model, no standalone score). https://huggingface.co/amd/AMD-Llama-135m-code. Licence: Apache-2.0. [01]
- artefact of: CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis. https://huggingface.co/Salesforce/codegen-350M-mono. Licence: models BSD-3-Clause (https://huggingface.co/Salesforce/codegen-350M-mono). [01]
- Banaxi-Tech/BananaMind-2.1-NanoCoder. https://huggingface.co/Banaxi-Tech/BananaMind-2.1-NanoCoder. Licence: Apache-2.0. [01]
- BEE-spoke-data/smol_llama-101M-GQA-python (no score on card). https://huggingface.co/BEE-spoke-data/smol_llama-101M-GQA-python. Licence: Apache-2.0. [01]
- bigcode/tiny_starcoder_py. https://huggingface.co/bigcode/tiny_starcoder_py. Licence: BigCode OpenRAIL-M. [01]
- CastIronMind/stentor_python_30m. https://huggingface.co/CastIronMind/stentor_python_30m. Licence: Apache-2.0. [01]
- char-transformer-shakespeare (model card). https://huggingface.co/delcenjo/char-transformer-shakespeare. Licence: MIT. [05]
- codefusionduo/suku-ai-128b-titan (cautionary example only). https://huggingface.co/codefusionduo/suku-ai-128b-titan. Licence: MIT. [01]
- CodeLlama. https://github.com/meta-llama/codellama. Licence: Llama 2 Community License. [08]
- codeparrot/codeparrot-small. https://huggingface.co/codeparrot/codeparrot-small. Licence: Apache-2.0. [01]
- Convergent-ch/Pico-1.3-Coding-Large, Pico-1, Pico-1.2. https://huggingface.co/Convergent-ch/Pico-1.3-Coding-Large. Licence: Apache-2.0. [01]
- Daoguang/PyCodeGPT. https://huggingface.co/Daoguang/PyCodeGPT. Licence: AFL-3.0. [01]
- DeepSeek-Coder, DeepSeek-Coder-V2. https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL. Licence: DeepSeek License Agreement 1.0. [08]
- DistilGPT2 model card. https://huggingface.co/distilbert/distilgpt2. Licence: model: Apache-2.0. [03]
- DrunkkToys/theOG-50M (lead added in the fact-check, section 3). https://huggingface.co/DrunkkToys/theOG-50M. Licence: Apache-2.0. [01]
- edededdy/LittleKedi-*-instruct. https://huggingface.co/edededdy/LittleKedi-medium-wide-sparse-instruct. Licence: Apache-2.0. [01]
- eeny-tinystories-999k (model card). https://huggingface.co/sprapp/eeny-tinystories-999k. Licence: model: Apache-2.0. [03]
- EleutherAI/gpt-neo-125m. https://huggingface.co/EleutherAI/gpt-neo-125m. Licence: MIT. [01]
- esp32-ai-tinystories (model card). https://huggingface.co/slvDev/esp32-ai-tinystories. Licence: model: MIT; code https://github.com/slvDev/esp32-ai : MIT. [03]
- esraderey/chispa-20m-python. https://huggingface.co/esraderey/chispa-20m-python. Licence: MIT. [01]
- gdm32569/ternary-c64-688k. https://huggingface.co/gdm32569/ternary-c64-688k. Licence: MIT. [01]
- HuggingFaceTB/SmolLM-135M, SmolLM2-135M, SmolLM2-360M. https://huggingface.co/HuggingFaceTB/SmolLM2-135M. Licence: Apache-2.0. [01]
- jfrick/scholarly-code-modern-gpt-10m-5b. https://huggingface.co/jfrick/scholarly-code-modern-gpt-10m-5b. Licence: none on card. [01]
- karpathy/tinyllamas (stories260K). https://huggingface.co/karpathy/tinyllamas. Licence: MIT. [01]
- MatterWaveAI/grt-30m-code (cautionary example only). https://huggingface.co/MatterWaveAI/grt-30m-code. Licence: Apache-2.0. [01]
- microsoft/CodeGPT-small-py. https://huggingface.co/microsoft/CodeGPT-small-py. Licence: none on card. [01]
- microsoft/phi-1. https://huggingface.co/microsoft/phi-1. Licence: MIT. [01]
- MihaiPopa-1/CinnabarLM-4M-Python-Base. https://huggingface.co/MihaiPopa-1/CinnabarLM-4M-Python-Base. Licence: Apache-2.0. [01]
- Mxode/minicoder-7M-base. https://huggingface.co/Mxode/minicoder-7M-base. Licence: none on card. [01]
- nilq/python-mistral-1L-mini. https://huggingface.co/nilq/python-mistral-1L-mini. Licence: none on card. [01]
- NinedayWang/PolyCoder-160M. https://huggingface.co/NinedayWang/PolyCoder-160M. Licence: none on card. [01]
- Qwen/Qwen2.5-Coder-0.5B. https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B. Licence: Apache-2.0. [01]
- Qwen2.5-Coder 0.5B / 1.5B / 7B / 14B / 32B Instruct. https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct. Licence: Apache-2.0. [08]
- Qwen2.5-Coder-3B-Instruct. https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct. Licence: Qwen Research Licence (non-commercial). [08]
- Qwen3.5-0.8B (base model card). https://huggingface.co/Qwen/Qwen3.5-0.8B. Licence: Apache-2.0. [07]
- Qwen3.5-0.8B-weight-space-programming (edited model, with an outside re-run). https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming. Licence: Apache-2.0. [07]
- Rashik24/tinycoder-15M. https://huggingface.co/Rashik24/tinycoder-15M. Licence: none on card. [01]
- Rohanify/PyBlissa-Coder-40M-GGUF, PyBlissa-Coder-50M, Indenta-13M-Python. https://huggingface.co/Rohanify/PyBlissa-Coder-40M-GGUF. Licence: Apache-2.0, MIT, CC BY-NC 4.0. [01]
- Salesforce/codet5p-220m. https://huggingface.co/Salesforce/codet5p-220m. Licence: BSD-3-Clause. [01]
- StarCoder, StarCoder2. https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement. Licence: BigCode OpenRAIL-M v1. [08]
- TinyLLama-v0 (model used by Deeploy). https://huggingface.co/Maykeye/TinyLLama-v0. Licence: Apache-2.0 (model card tag). [02, 03]
- tinyllamas checkpoints, stories260K model card. https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K. Licence: MIT. [04, 05]
- TinyStories 1-bit LLM (model card). https://huggingface.co/euclidstellar/tinystories-1bit-llm. Licence: model: MIT. [03]
- TinyStories-1M (model card and config.json; added by the fact-check). https://huggingface.co/roneneldan/TinyStories-1M. Licence: model: no licence in the card metadata. [03]
- upiter/TinyCodeLM-150M and -400M. https://huggingface.co/upiter/TinyCodeLM-150M. Licence: Apache-2.0. [01]
- Uuuuuuniiiiiii/goldworm-v8-coder-85m. https://huggingface.co/Uuuuuuniiiiiii/goldworm-v8-coder-85m. Licence: MIT. [01]

### Datasets (19)

- APPS. https://github.com/hendrycks/apps. Licence: MIT label; scraped sources. [08]
- artefact of: Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. https://github.com/timinar/BabyLlama. Licence: code https://github.com/timinar/BabyLlama : MIT. [03]
- artefact of: Program Synthesis with Large Language Models. https://huggingface.co/datasets/google-research-datasets/mbpp. Licence: MBPP: CC BY 4.0 (https://huggingface.co/datasets/google-research-datasets/mbpp); models not released [VERIFIED] (MBPP licence). [01]
- bigcode/the-stack (and -dedup, -smol, -smol-xs). https://huggingface.co/datasets/bigcode/the-stack. Licence: tag `other`; gated except -smol-xs. [08]
- CodeContests. https://github.com/google-deepmind/code_contests. Licence: CC-BY-4.0 data, Apache-2.0 code; third-party parts under own terms. [08]
- codeparrot/codeparrot-clean. https://huggingface.co/datasets/codeparrot/codeparrot-clean. Licence: none on card (per-file licences). [01, 08]
- codeparrot/github-code. https://huggingface.co/datasets/codeparrot/github-code. Licence: tag `other`; per-file `license` field. [08]
- CodeSearchNet. https://github.com/github/CodeSearchNet. Licence: tooling MIT; data per repository (table in `python_licenses.pkl`). [08]
- CoNaLa. https://conala-corpus.github.io/. Licence: MIT label; Stack Overflow content CC BY-SA. [08]
- flytech/python-codes-25k. https://huggingface.co/datasets/flytech/python-codes-25k. Licence: [UNVERIFIED: no licence field on card]. [01, 08]
- Fsoft-AIC/the-vault-function. https://huggingface.co/datasets/Fsoft-AIC/the-vault-function. Licence: MIT; per-row licence list. [08]
- HuggingFaceTB/smollm-corpus (Python-Edu). https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus. Licence: ODC-BY. [01]
- HumanEval+ / MBPP+ (EvalPlus). https://huggingface.co/datasets/evalplus/humanevalplus. Licence: Apache-2.0. [01, 08]
- iamtarun/python_code_instructions_18k_alpaca. https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca. Licence: none stated. [08]
- MBPP data file (byte lengths; train and validation splits for the tokenizer measurement). https://github.com/google-research/google-research/tree/master/mbpp. Licence: data CC-BY-4.0 (repository README: "All datasets in this repository are released under the CC BY 4.0 International license"; the Hugging Face dataset card says the same); source files of the repository Apache-2.0. [05, 08]
- nampdn-ai/tiny-codes. https://huggingface.co/datasets/nampdn-ai/tiny-codes. Licence: MIT (gated). [01, 08]
- nilq/baby-python. https://huggingface.co/datasets/nilq/baby-python. Licence: MIT. [01]
- nvidia/OpenCodeInstruct. https://huggingface.co/datasets/nvidia/OpenCodeInstruct. Licence: CC BY 4.0. [01]
- TinyStories dataset. https://huggingface.co/datasets/roneneldan/TinyStories. Licence: CDLA-Sharing-1.0. [04]

### Code repositories (134)

- 8bitGPT (added). https://github.com/Timothy-Sikes/8bitgpt. Licence: not classified by the API; the README says the integer math is MIT-licensed. [04]
- ActiveState/code. https://github.com/ActiveState/code. Licence: MIT by default; legacy recipes per recipe. [08]
- artefact of: A Multimodal Automated Interpretability Agent (MAIA). https://github.com/multimodal-interpretability/maia. Licence: code https://github.com/multimodal-interpretability/maia: none detected. [07]
- artefact of: A Simple and Effective Pruning Approach for Large Language Models (Wanda). https://github.com/locuslab/wanda. Licence: code https://github.com/locuslab/wanda: MIT. [07]
- artefact of: A Systematic Evaluation of Large Language Models of Code. https://github.com/VHellendoorn/Code-LMs. Licence: code MIT (https://github.com/VHellendoorn/Code-LMs); no licence on the hub card [VERIFIED] (both licence fields). [01]
- artefact of: A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance. https://github.com/Xilinx/brevitas. Licence: code in https://github.com/Xilinx/brevitas : BSD 3-clause style text plus third-party notices in LICENSE (GitHub reports NOASSERTION). [02]
- artefact of: Are Deep Neural Networks the Best Choice for Modeling Source Code?. https://github.com/SLP-team/SLP-Core. Licence: SLP-Core: MIT (https://github.com/SLP-team/SLP-Core) [VERIFIED] (SLP-Core licence). [01]
- artefact of: Automatic Generation of Python Programs Using Context-Free Grammars. https://github.com/MarwaNair/TinyPy-Generator. Licence: https://github.com/MarwaNair/TinyPy-Generator has no licence file [VERIFIED] (repository has no licence). [01]
- artefact of: BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks. https://github.com/schneiderkamplab/bitlinear. Licence: code https://github.com/schneiderkamplab/bitlinear : MIT. [03]
- artefact of: Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models. https://github.com/MLAI-Yonsei/BHyT. Licence: code https://github.com/MLAI-Yonsei/BHyT : Apache-2.0. [02]
- artefact of: BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction. https://github.com/yhhhli/BRECQ. Licence: code https://github.com/yhhhli/BRECQ: MIT. [07]
- artefact of: CERT: Continual Pre-Training on Sketches for Library-Oriented Code Generation. https://github.com/microsoft/PyCodeGPT. Licence: PyCodeGPT: AFL-3.0 on the hub card, MIT on https://github.com/microsoft/PyCodeGPT [VERIFIED] (both licence fields). [01]
- artefact of: ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters. https://github.com/ReaLLMASIC/ConSmax. Licence: code https://github.com/ReaLLMASIC/ConSmax : Apache-2.0. [02]
- artefact of: Curriculum Learning for Small Code Language Models. https://github.com/kamel-yamani/CurriculumLearningForSmallCodeLanguageModels. Licence: code and 1 M checkpoints Apache-2.0 (https://github.com/kamel-yamani/CurriculumLearningForSmallCodeLanguageModels); data on Kaggle, licence not checked [VERIFIED] (repository licence). [01]
- artefact of: Data-Free Quantization Through Weight Equalization and Bias Correction. https://github.com/qualcomm/aimet. Licence: implementation in https://github.com/qualcomm/aimet: licence file present, not classified by the GitHub API. [07]
- artefact of: Deduplicating Training Data Makes Language Models Better. https://github.com/google-research/deduplicate-text-datasets. Licence: code https://github.com/google-research/deduplicate-text-datasets : Apache-2.0, archived. [03]
- artefact of: Deeploy: Enabling Energy-Efficient Deployment of Small Language Models On Heterogeneous Microcontrollers. https://github.com/pulp-platform/Deeploy. Licence: code https://github.com/pulp-platform/Deeploy : Apache-2.0 for most source files and MIT for some scripts according to the README; `LICENSES/` holds Apache-2.0.txt, MIT.txt and CC-BY-ND-4.0.txt; the GitHub API reports no single licence [VERIFIED]. [02, 03]
- artefact of: Differentiable Model Compression via Pseudo Quantization Noise. https://github.com/facebookresearch/diffq. Licence: code https://github.com/facebookresearch/diffq : CC BY-NC 4.0 (licence file), archived. [03]
- artefact of: Editing Large Language Models: Problems, Methods, and Opportunities (survey). https://github.com/zjunlp/EasyEdit. Licence: framework https://github.com/zjunlp/EasyEdit: MIT. [07]
- artefact of: Editing Models with Task Arithmetic. https://github.com/mlfoundations/task_vectors. Licence: code https://github.com/mlfoundations/task_vectors: none detected. [07]
- artefact of: Eliciting Latent Predictions from Transformers with the Tuned Lens. https://github.com/AlignmentResearch/tuned-lens. Licence: code https://github.com/AlignmentResearch/tuned-lens: MIT. [07]
- artefact of: esp32-ai-tinystories (model card). https://github.com/slvDev/esp32-ai. Licence: model: MIT; code https://github.com/slvDev/esp32-ai : MIT. [03, 04]
- artefact of: Evaluating Large Language Models Trained on Code. https://github.com/openai/human-eval. Licence: HumanEval: MIT (https://github.com/openai/human-eval); Codex weights not released [VERIFIED] (HumanEval licence). [01, 05, 08]
- artefact of: Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations. https://github.com/WSP-Scope/WSP-Scope-Project. Licence: code https://github.com/WSP-Scope/WSP-Scope-Project: none detected. [07]
- artefact of: Fine-Tuning Language Models with Just Forward Passes (MeZO). https://github.com/princeton-nlp/MeZO. Licence: code https://github.com/princeton-nlp/MeZO: MIT. [07]
- artefact of: FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer. https://github.com/megvii-research/FQ-ViT. Licence: code https://github.com/megvii-research/FQ-ViT : Apache-2.0. [02, 03]
- artefact of: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. https://github.com/IST-DASLab/gptq. Licence: code https://github.com/IST-DASLab/gptq: Apache-2.0. [07]
- artefact of: HAWQ-V3: Dyadic Neural Network Quantization. https://github.com/Zhen-Dong/HAWQ. Licence: code https://github.com/Zhen-Dong/HAWQ : MIT. [02]
- artefact of: HLSTransform: Energy-Efficient Llama 2 Inference on FPGAs Via High Level Synthesis. https://github.com/HLSTransform/submission. Licence: paper; code https://github.com/HLSTransform/submission MIT. [04]
- artefact of: I-BERT: Integer-only BERT Quantization. https://github.com/kssteven418/I-BERT. Licence: code https://github.com/kssteven418/I-BERT : MIT. [02, 03]
- artefact of: I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference. https://github.com/zkkli/I-ViT. Licence: code https://github.com/zkkli/I-ViT : Apache-2.0. [02]
- artefact of: LEACE: Perfect linear concept erasure in closed form. https://github.com/EleutherAI/concept-erasure. Licence: code https://github.com/EleutherAI/concept-erasure: MIT. [07]
- artefact of: LLM-QAT: Data-Free Quantization Aware Training for Large Language Models. https://github.com/facebookresearch/LLM-QAT. Licence: code https://github.com/facebookresearch/LLM-QAT : CC BY-NC 4.0 (licence file). [03]
- artefact of: Locating and Editing Factual Associations in GPT (ROME). https://github.com/kmeng01/rome. Licence: code https://github.com/kmeng01/rome: MIT. [07]
- artefact of: Mass-Editing Memory in a Transformer (MEMIT). https://github.com/kmeng01/memit. Licence: code https://github.com/kmeng01/memit: MIT. [07]
- artefact of: Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning (OBC). https://github.com/IST-DASLab/OBC. Licence: code https://github.com/IST-DASLab/OBC: none detected. [07]
- artefact of: Overcoming Oscillations in Quantization-Aware Training. https://github.com/qualcomm-ai-research/oscillations-qat. Licence: code https://github.com/qualcomm-ai-research/oscillations-qat : BSD-3-Clause-Clear. [03]
- artefact of: Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing. https://github.com/Qualcomm-AI-research/outlier-free-transformers. Licence: code https://github.com/Qualcomm-AI-research/outlier-free-transformers : BSD-3-Clause-Clear. [02]
- artefact of: Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost (QES). https://github.com/dibbla/Quantized-Evolution-Strategies. Licence: code https://github.com/dibbla/Quantized-Evolution-Strategies: none detected. [07]
- artefact of: Refusal in Language Models Is Mediated by a Single Direction. https://github.com/andyrdt/refusal_direction. Licence: code https://github.com/andyrdt/refusal_direction: Apache-2.0. [07]
- artefact of: SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization. https://github.com/xinghaochen/SLAB. Licence: code https://github.com/xinghaochen/SLAB : no licence detected by the GitHub API. [02]
- artefact of: SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. https://github.com/IST-DASLab/sparsegpt. Licence: code https://github.com/IST-DASLab/sparsegpt: Apache-2.0. [07]
- artefact of: Steering Llama 2 via Contrastive Activation Addition. https://github.com/nrimsky/CAA. Licence: code https://github.com/nrimsky/CAA: MIT. [07]
- artefact of: Theory, Analysis, and Best Practices for Sigmoid Self-Attention. https://github.com/apple-aiml-research/ml-sigmoid-attention. Licence: code https://github.com/apple-aiml-research/ml-sigmoid-attention : custom Apple licence text (GitHub reports NOASSERTION). [02]
- artefact of: Tracr: Compiled Transformers as a Laboratory for Interpretability. https://github.com/google-deepmind/tracr. Licence: code https://github.com/google-deepmind/tracr: Apache-2.0. [07]
- artefact of: Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks. https://github.com/Xilinx/graffitist. Licence: code https://github.com/Xilinx/graffitist : BSD 3-Clause text in LICENSE (GitHub reports NOASSERTION). [02, 03]
- artefact of: Transformers without Normalization. https://github.com/jiachenzhu/DyT. Licence: code https://github.com/jiachenzhu/DyT : MIT. [02]
- artefact of: Tree-Ising: Repairing Language Models with Bit-Level Ising Edits [fact-check: added]. https://github.com/TheQuantumGOAT/Tree-Ising. Licence: code https://github.com/TheQuantumGOAT/Tree-Ising: none detected. [07]
- artefact of: Understanding and Overcoming the Challenges of Efficient Transformer Quantization. https://github.com/qualcomm-ai-research/transformer-quantization. Licence: code https://github.com/qualcomm-ai-research/transformer-quantization : BSD-3-Clause-Clear. [03]
- artefact of: Weight-Space Programming from scratch on simple tasks (report). https://github.com/csinva/weight-space-programming-autoresearch. Licence: code https://github.com/csinva/weight-space-programming-autoresearch: MIT. [07]
- ATTN/11: "Paper Tape is All You Need". https://github.com/dbrll/ATTN-11. Licence: none detected. [04]
- Awesome LLM on Anything (curated list). https://github.com/emrehangorgec/awesome-llm-on-anything. Licence: CC0-1.0. [04]
- bigcode-evaluation-harness. https://github.com/bigcode-project/bigcode-evaluation-harness. Licence: Apache-2.0. [01]
- Chalarangelo/30-seconds-of-python. https://github.com/Chalarangelo/30-seconds-of-python. Licence: CC-BY-4.0. [08]
- CMSIS-NN v6.0.0: `arm_softmax_s8.c`. https://github.com/ARM-software/CMSIS-NN/blob/v6.0.0/Source/SoftmaxFunctions/arm_softmax_s8.c. Licence: Apache-2.0. [02]
- CMSIS_5 5.9.0: `arm_softmax_q7.c`, `arm_fully_connected_q7.c`, `arm_nnsupportfunctions.h`. https://github.com/ARM-software/CMSIS_5/tree/5.9.0/CMSIS/NN. Licence: Apache-2.0. [02]
- core_ddr3_controller (README). https://github.com/ultraembedded/core_ddr3_controller. Licence: Apache-2.0 according to the source file headers (`src_v/ddr3_core.v`: "License: Apache 2.0"); the repository has no LICENSE file, so the GitHub API reports none [corrected by the fact-check]; README read only. [09]
- CraftGPT repository and a news article about it (2025-10-02). https://github.com/sammyuri/craftgpt. Licence: MIT (repository). [04]
- dgilland/pydash. https://github.com/dgilland/pydash. Licence: MIT. [08]
- digilent-xdc, `Nexys-A7-100T-Master.xdc`. https://github.com/Digilent/digilent-xdc. Licence: MIT [VERIFIED]. [09]
- doocs/leetcode. https://github.com/doocs/leetcode. Licence: CC-BY-SA-4.0. [08]
- dottxt-ai/outlines. https://github.com/dottxt-ai/outlines. Licence: Apache-2.0. [06]
- dsp_inference. https://github.com/skyline624/dsp_inference. Licence: none detected. [04]
- ebmoon/transformers-GAD. https://github.com/ebmoon/transformers-GAD. Licence: MIT. [06]
- ecp5-inference (stories260K on a Lattice ECP5; added in the fact-check). https://github.com/skyline624/ecp5-inference. Licence: no licence detected by the GitHub API. [02, 04]
- elya-nes (ternary transformer on the NES; added in the fact-check). https://github.com/Scottcjn/elya-nes. Licence: Apache-2.0. [02, 04]
- EmbedLlama: llama2.c on STM32H7A3. https://github.com/schuhandreas/embedllama-stm32h7a3. Licence: MIT. [04]
- epfl-dlab/GCD. https://github.com/epfl-dlab/GCD. Licence: MIT. [06]
- esp32-llm (fork). https://github.com/eric-humane/esp32-llm. Licence: none detected. [04]
- esp32-llm: "Running a LLM on the ESP32". https://github.com/DaveBben/esp32-llm. Licence: none detected. [04]
- eth-sri/domino. https://github.com/eth-sri/domino. Licence: none: the repository is empty (size 0, no files, checked 2026-11-13). [06]
- eth-sri/type-constrained-code-generation. https://github.com/eth-sri/type-constrained-code-generation. Licence: MIT. [06]
- exercism/python. https://github.com/exercism/python. Licence: MIT. [08]
- eZ80-ai. https://github.com/tornadus/ez80ai. Licence: MIT. [04]
- facebookresearch/CodeGen (TransCoder). https://github.com/facebookresearch/CodeGen. Licence: MIT. [06]
- fpga-tinystories (ternary TinyStories model on a MAX 10; added in the fact-check). https://github.com/MRazvan/fpga-tinystories. Licence: MIT. [02, 04]
- gateGPT. https://github.com/fguzman82/gateGPT. Licence: none detected. [04]
- gbc-transformer. https://github.com/maddiedreese/gbc-transformer. Licence: MIT. [04]
- ggml-org/llama.cpp. https://github.com/ggml-org/llama.cpp. Licence: MIT. [06]
- guidance-ai/guidance. https://github.com/guidance-ai/guidance. Licence: MIT. [06]
- guidance-ai/llguidance. https://github.com/guidance-ai/llguidance. Licence: MIT. [06]
- Heretic (tool for automated direction removal). https://github.com/p-e-w/heretic. Licence: AGPL-3.0. [07]
- Hugging Face Transformers, `modeling_ibert.py` (v4.44.0). https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/ibert/modeling_ibert.py. Licence: Apache-2.0. [02]
- JAM (Just Atari Model). https://github.com/marspa73/atarijam. Licence: MIT. [04]
- kamyu104/LeetCode-Solutions, walkccc/LeetCode, neetcode-gh/leetcode, qiyuangong/leetcode. https://github.com/kamyu104/LeetCode-Solutions. Licence: MIT (each). [08]
- keon/algorithms. https://github.com/keon/algorithms. Licence: MIT. [08]
- kev-gpt. https://github.com/MichaelAyles/kev-gpt. Licence: MIT. [04]
- lark-parser/lark. https://github.com/lark-parser/lark. Licence: MIT. [06]
- Llama 2, 3, 3.1, 3.2, 3.3, 4. https://github.com/meta-llama/llama-models/tree/main/models. Licence: Llama Community Licences (per version). [08]
- LLaMA2-on-OpenLA500. https://github.com/WizLiang/LLaMA2-on-OpenLA500. Licence: none detected. [04]
- llama2.c. https://github.com/karpathy/llama2.c. Licence: MIT. [01, 02, 03, 04, 05, 06]
- llama2.c-260k-rtl-fpga (README, docs/ARCHITECTURE_MAP.md). https://github.com/wupd1122/llama2.c-260k-rtl-fpga. Licence: none detected. [04]
- llama2.c64 (llama2.c on a Commodore 64). https://github.com/ytmytm/llama2.c64. Licence: no licence file in the repository root. [02, 04]
- llama2c_Dreamcast. https://github.com/gameblabla/llama2c_Dreamcast. Licence: MIT. [04]
- lm_understand-logs (logs, networks, weight diffs, evaluation scripts of the post). https://github.com/RikiyaT/lm_understand-logs. Licence: logs, data, networks CC BY 4.0; scripts MIT; MinAtar source quoted in Experiment 5 prompts GPL-3.0. [07]
- mahmoud/boltons. https://github.com/mahmoud/boltons. Licence: BSD-3-Clause. [08]
- mamba_small_bench (README). https://github.com/apapiu/mamba_small_bench. Licence: no licence file. [05]
- microsoft/monitors4codegen (archived). https://github.com/microsoft/monitors4codegen. Licence: MIT. [06]
- mlc-ai/xgrammar. https://github.com/mlc-ai/xgrammar. Licence: Apache-2.0. [06]
- more-itertools. https://github.com/more-itertools/more-itertools. Licence: MIT. [08]
- nanoGPT (README; source of the 1.4697 that mamba_small_bench quotes). https://github.com/karpathy/nanoGPT. Licence: MIT. [05]
- networkx. https://github.com/networkx/networkx. Licence: BSD-3-Clause. [08]
- Nexys-A7-100T-GPIO (`UART_TX_CTRL.vhd`, `top.vhd`, XDC). https://github.com/Digilent/Nexys-A7-100T-GPIO. Licence: no licence file; header "Copyright 2011 Digilent, Inc."; read only, not used [VERIFIED]. [09]
- Nexys-A7-100T-OOB (`Ram2Ddr.vhd`, `RamCntrl.vhd`, `AudioDemo.vhd`). https://github.com/Digilent/Nexys-A7-100T-OOB. Licence: no licence file; file headers "COPYRIGHT (C) 2014, Digilent RO. All rights reserved"; read only, not used [VERIFIED]. [09]
- Nexys-A7-50T-OOB, `src/other/mig_a.prj` (added by the fact-check). https://github.com/Digilent/Nexys-A7-50T-OOB. Licence: no licence file; read only. [09]
- OmkarPathak/pygorithm. https://github.com/OmkarPathak/pygorithm. Licence: MIT. [08]
- openTPU. https://github.com/FeSens/openTPU. Licence: Apache-2.0. [04]
- openTPU on Tang Nano 20K. https://github.com/LakoMoor/openTPU-tangnano20k. Licence: Apache-2.0. [04]
- pico-experiments (picollama260k). https://github.com/tarelerulz/pico-experiments. Licence: not classified by the API. [04]
- PicoLlama. https://github.com/earlephilhower/PicoLlama. Licence: MIT. [04]
- PicoRV32. https://github.com/YosysHQ/picorv32. Licence: ISC [VERIFIED]. [09]
- PocketTPU. https://github.com/thinkelastic/PocketTPU. Licence: none detected. [04]
- psvita-llm. https://github.com/callbacked/psvita-llm. Licence: none detected. [04]
- python/cpython. https://github.com/python/cpython. Licence: Python Software Foundation License Version 2 (LICENSE file of the 3.12 branch, read in the fact check; the GitHub API reports "Other") [VERIFIED]. [06, 08]
- pytoolz/toolz. https://github.com/pytoolz/toolz. Licence: BSD-3-Clause. [08]
- semu-c64: "L2E llama2.c on Commodore C-64". https://github.com/trholding/semu-c64. Licence: MIT. [04]
- SentencePiece (tool used for the measurement). https://github.com/google/sentencepiece. Licence: Apache-2.0. [05, 06]
- SERV. https://github.com/olofk/serv. Licence: ISC [VERIFIED]. [09]
- ServiceNow/picard. https://github.com/ServiceNow/picard. Licence: Apache-2.0. [06]
- Soul Player C64 (int8 transformer on a Commodore 64; added in the fact-check). https://github.com/gizmo64k/soulplayer-c64. Licence: GPL-3.0. [02, 04]
- soulplayer-cpc (added). https://github.com/G1D30N/soulplayer-cpc. Licence: GPL-3.0. [04]
- soulplayer-spectrum (added). https://github.com/tunechr/soulplayer-spectrum. Licence: none detected. [04]
- structuredllm/syncode (was uiuc-focal-lab/syncode). https://github.com/structuredllm/syncode. Licence: MIT. [06]
- Suor/funcy. https://github.com/Suor/funcy. Licence: BSD-3-Clause. [08]
- TALOS-V2. https://github.com/Luthiraa/TALOS-V2. Licence: Apache-2.0. [04]
- TensorFlow Lite v2.16.1: `reference/softmax.h`, `common.h`, `optimized/optimized_ops.h`. https://github.com/tensorflow/tensorflow/tree/v2.16.1/tensorflow/lite/kernels/internal. Licence: Apache-2.0. [02]
- TheAlgorithms/Python. https://github.com/TheAlgorithms/Python. Licence: MIT. [08]
- ti84-llm-math (added). https://github.com/RileyGreiff/ti84-llm-math. Licence: none detected. [04]
- TinyTransformer-FPGA. https://github.com/jianmin5xin/TinyTransformer-FPGA. Licence: not classified by the API; RTL not published. [04]
- `uart_tx6.vhd` (PicoBlaze UART macro), copy in a third-party repository. https://github.com/grg-haas/fpga_miner_rework/blob/master/src/vhdl/uart_tx6.vhd. Licence: Xilinx proprietary header; repository has no licence; primitives counted only [VERIFIED]. [09]
- vellm. https://github.com/ecliptik/vellm. Licence: MIT. [04]
- VexRiscv. https://github.com/SpinalHDL/VexRiscv. Licence: MIT [VERIFIED]. [09]
- vivado-boards, `new/board_files/nexys-a7-100t/D.0/1.3/mig.prj` (added by the fact-check). https://github.com/Digilent/vivado-boards. Licence: MIT; read only. [09]
- Z80-uLM (README; added by the fact-check). https://github.com/HarryR/z80ai. Licence: code: "MIT or Apache-2.0 as you see fit" (README; the GitHub API detects no licence file). [03, 04]

### Documentation, web pages, terms and policies (43)

- 7 Series DSP48E1 Slice User Guide, UG479 v1.10. https://docs.amd.com/go/en-US/ug479_7Series_DSP48E1. Licence: documentation, no code used. [09]
- 7 Series FPGAs Configurable Logic Block User Guide, UG474 v1.9. https://docs.amd.com/go/en-US/ug474_7Series_CLB. Licence: documentation. [09]
- 7 Series FPGAs Data Sheet: Overview, DS180 v1.17 (added; read from a university mirror). https://eng.auburn.edu/~nelson/courses/elec4200/FPGA/ds180_7Series_Overview.pdf. Licence: data sheet. [04]
- 7 Series FPGAs Data Sheet: Overview, DS180 v2.6.1. https://docs.amd.com/go/en-US/ds180_7Series_Overview. Licence: documentation. [09]
- 7 Series FPGAs Memory Resources User Guide, UG473 v1.14. https://docs.amd.com/go/en-US/ug473_7Series_Memory_Resources. Licence: documentation. [09]
- Accelerating Zero-Entropy LLM Output to 100k Tokens per Second (blog). https://michaelorenstein.com/blog/zero-entropy-tokens. [05]
- "AI language model runs on a Windows 98 system with Pentium II and 128MB of RAM" (news article, added). https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-language-model-runs-on-a-windows-98-system-with-pentium-ii-and-128mb-of-ram-open-source-ai-flagbearers-demonstrate-llama-2-llm-in-extreme-conditions. [04]
- Aliasing VCP Baud Rates, AN_120 v1.2, FT_000161 (added by the fact-check). https://ftdichip.com/wp-content/uploads/2020/08/AN_120_Aliasing_VCP_Baud_Rates.pdf. Licence: documentation. [09]
- Artix 7 FPGAs Data Sheet: DC and AC Switching Characteristics, DS181 v1.27.1. https://docs.amd.com/go/en-US/ds181_Artix_7_Data_Sheet. Licence: documentation. [09]
- Can I use my Outputs to train an AI model?. https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model. [08]
- Chipyard RISC-V Tutorial, chapter 7: Running LLM Inference on FPGA with Gemmini. https://mikutyan4.github.io/chipyard-linux-nexys/ch07-llm-inference.html. [04]
- Commercial Terms of Service. https://www.anthropic.com/legal/commercial-terms. [08]
- Consumer Terms of Service. https://www.anthropic.com/legal/consumer-terms. [08]
- Deep Learning with INT8 Optimization on Xilinx Devices, WP486 v1.0.1. https://docs.amd.com/go/en-US/wp486-deep-learning-int8. Licence: documentation. [09]
- Efficient Shift Registers, LFSR Counters, and Long Pseudo-Random Sequence Generators, XAPP052 v1.1. https://docs.amd.com/go/en-US/xapp052. Licence: documentation. [09]
- Europe Terms of Use (EEA, Switzerland, UK). https://web.archive.org/web/20261028153406/https://openai.com/policies/eu-terms-of-use/. [08]
- FT2232H Dual High Speed USB to Multipurpose UART/FIFO IC Datasheet v2.6, FT_000061. https://ftdichip.com/wp-content/uploads/2020/07/DS_FT2232H.pdf. Licence: documentation. [09]
- Implementing Barrel Shifters Using Multipliers, XAPP195 v1.1. https://docs.amd.com/go/en-US/xapp195. Licence: documentation. [09]
- Improving Bash Generation in Small Language Models with Grammar-Constrained Decoding (blog). https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/. [06]
- "Llama 2 LLM on DOS" (blog) and dosllam2. https://yeokhengmeng.com/2025/04/llama2-llm-on-dos/. Licence: MIT. [04]
- LLMs are not superhuman at reading and writing neural network weights. https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/. [07]
- logiMEM DDR3 SDRAM Memory Controller data sheet v1.1. https://www.logicbricks.com/Documentation/Datasheets/IP/logiMEM_hds.pdf. Licence: commercial core; data sheet only. [09]
- Memory Interface Solutions LogiCORE IP Data Sheet, DS176 (MIG v4.2). https://docs.amd.com/r/en-US/ds176_7Series_MIS/Resource-Utilization. Licence: documentation. [09]
- MicroBlaze Processor Reference Guide, UG984. https://docs.amd.com/go/en-US/ug984-vivado-microblaze-ref. Licence: documentation. [09]
- Nexys A7 Reference Manual. https://digilent.com/reference/programmable-logic/nexys-a7/reference-manual. Licence: documentation. [09]
- PicoBlaze 8-bit Microcontroller (KCPSM6 Rev 9) product page. https://www.amd.com/en/products/adaptive-socs-and-fpgas/intellectual-property/picoblaze.html. Licence: reference design under AMD terms; not used. [09]
- Qwen 3.8 27B on my laptop can interpret model weights, too. https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/. [07]
- "Running 260K Tiny-LLM on ESP32". https://circuitdigest.com/articles/running-260K-tiny-llm-on-esp32. [04]
- "Running Llama on Windows 98" (blog) and llama98.c. https://blog.exolabs.net/day-4/. Licence: MIT. [04]
- Services Agreement. https://web.archive.org/web/20261110000152/https://openai.com/policies/services-agreement/. [08]
- SmolLM - blazingly fast and remarkably powerful (blog). https://huggingface.co/blog/smollm. [01]
- Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights (blog post). https://rikiyat.github.io/blog/weight-space-programming/. Licence: none stated on the page. [07]
- SRAM to DDR component (tutorial page). https://digilent.com/reference/learn/programmable-logic/tutorials/nexys-4-ddr-sram-to-ddr-component/start. Licence: documentation [VERIFIED]. [09]
- Starmind-Pico: "Training a language model to run on RP2040". https://comfyspace.tech/starmind/experiments/starmind-pico. Licence: not stated. [04]
- Terms and Conditions. https://codeforces.com/terms. [08]
- Terms of Use (rest of world). https://web.archive.org/web/20261109060701/https://openai.com/policies/row-terms-of-use/. [08]
- The Python Language Reference: Lexical analysis; Full Grammar specification. https://docs.python.org/3/reference/lexical_analysis.html. [06]
- The Unreasonable Effectiveness of Recurrent Neural Networks (blog). https://karpathy.github.io/2015/05/21/rnn-effectiveness/. [01]
- Transformers on Retro Game Consoles (project page). https://hackaday.io/project/206401-transformers-on-retro-game-consoles. Licence: not stated. [04]
- Usage Policy. https://www.anthropic.com/legal/aup. [08]
- Weight-Space Programming from scratch on simple tasks (report). https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html. Licence: code https://github.com/csinva/weight-space-programming-autoresearch: MIT. [07]
- What is the license for the content I post?. https://stackoverflow.com/help/licensing. [08]
- Zynq 7000 SoC and 7 Series Devices Memory Interface Solutions User Guide, UG586 (MIG v4.2). https://docs.amd.com/go/en-US/ug586_7Series_MIS. Licence: documentation. [09]
