# 03. Quantization-aware training (QAT) for very small models

Research agent 3 of 9, written 2026-11-13. Scope: QAT methods, the loss to expect from int8 weights and int8
activations, which components are sensitive, distillation into tiny language models, cheap data-side methods.
Hardware constraints are taken from `research/CONTEXT.md`.

Fact-check pass (2026-11-13, second agent): claims marked [VERIFIED] were re-read at the primary source (paper
PDF through pdftotext, arXiv API, hub API and raw cards, GitHub API) or recomputed with the project virtualenv.
[UNVERIFIED: reason] marks claims that no source supports. Text introduced by "Fact-check" was added or changed
in that pass. Claims without a tag were not re-checked. Tags sit at the earliest mention of a claim (sections 1
and 2); the tables of sections 5 and 6 repeat those claims. One line per checked claim is in the "Verification
log" at the end.

How to read this file
- Papers were read through ar5iv / arXiv HTML with targeted text extraction, or through the arXiv API abstract
  only (marked "abstract" in section 7). Every number is followed by the URL of the paper it was read in.
- Community model cards are self-reported and not peer reviewed. They are labelled "model card" wherever used.
- A claim that could not be checked at its primary source carries `[UNVERIFIED: reason]`.
- Section 2J is our own arithmetic, run on this machine with the project virtualenv. It is not literature.
- Loss differences are in nats per token. A perplexity ratio r equals ln(r) nats.
- Not checked: bodies of the papers marked "abstract", dataset licences. (The integer-type constants of Kumar et
  al. were read during the fact-check, see 2J.) The research agent and the fact-check agent wrote only this
  file; nothing was appended to SOURCES.md, DECISIONS.md or LOG.md.

## 1. Answer in one paragraph

The literature supports this route for a 260k to 400k parameter Llama-style model on the LS CPU: train a float
model whose operations already have the shape of the hardware, fine-tune it briefly with quantization in the loop
(clipping ranges learned in the log2 domain so that every scale is a power of two, as in TQT; straight-through
gradients; logit distillation from the model's own float checkpoint), and finish with a forward pass that is
bit-exact with the CPU. In the papers that tune their QAT, 8-bit weights plus 8-bit activations cost between
-0.005 and +0.03 nats per token on language models (WikiText-2/103; models of 243 MB and 942 MB in float32) and
stay within about +/-0.3 points on GLUE [VERIFIED]. Fact-check caveat: among these language-model results only
Prato et al. train with the activation quantizers in the loop, on a weak 2-layer example model whose 4-bit
version scores the same; in the two 942 MB results the activations are quantized after training. Two papers
report a baseline at +0.08 and +0.14 nats in which only the weights are trained with the straight-through
estimator and the activations are quantized after training [VERIFIED numbers; Fact-check correction: the
earlier text called this a "plain straight-through baseline", which hid that the activations were not in the
training loop; with float activations the same baseline costs +0.006 nats, DiffQ Table B.6]. Weights alone cost
nothing measurable; the activations carry the risk, roughly in this order (our reading of section 2B): the
residual-stream sum after the feed-forward block, the inputs of the normalisation, the attention map, the
softmax input, the KV cache, the embeddings. No source at or below 1 M parameters with integer-only inference,
static power-of-two scales and lookup tables indexed by an int8 value was found. The nearest item (Fact-check
addition, missing from the earlier text) is Deeploy, which runs a 4.6 M parameter TinyStories Llama (about
0.5 M parameters outside the embedding tables) with 8-bit integer inference on a microcontroller after
post-training quantization with TQT, and reports speed and energy but no loss [VERIFIED]. Smaller,
longer-trained models are reported to be more sensitive to post-training quantization of the weights, so for
LS-LLM a total of +0.02 to +0.05 nats is a planning assumption and not a result [UNVERIFIED: no source measures
a loss at this scale or with static power-of-two scales and int8-indexed tables; the true cost may be
larger]. Oscillation fixes, ternary (BitNet) weights and curriculum learning have
no support at 8 bits and our scale (the one 8-bit oscillation measurement, in a CNN, shows a 0.10 point
effect). Distillation from a larger teacher is contested (it helps in the data-limited BabyLM setting and gave
no gain in MobileLLM) and has no controlled result below 1 M parameters. Among cheap data-side methods,
restricting the data to a simple distribution (TinyStories) and deduplication have the strongest evidence. Two
zero-cost items come from our own arithmetic: a row of up to 511 int8 products cannot overflow the 24-bit
accumulator, and folding a rounding constant into the accumulator preload cuts the requantisation error power
(bias included) by 4x compared with the bare arithmetic shift [VERIFIED: recomputed].

## 2. Proven (numbers and source URLs)

### 2A. Reported cost of 8-bit weights + 8-bit activations on language models

- Fully quantized Transformer trained with quantization in the loop (Prato et al. 2019). Quantized: all weight
  matrices (one scale per output row), the embedding sum, the Q/K/V inputs, the softmax numerator, denominator
  and output, the LayerNorm numerator, denominator, quotient and output. Biases and scale constants stay float.
  Language model (243.04 MB in float32): WikiText-2 loss 5.65 -> 5.64 (PPL 284.15 -> 282.67), WikiText-103 loss
  5.91 -> 5.94 (PPL 369.20 -> 377.79, ln ratio +0.023). Translation, base model, BLEU: 26.46 -> 26.38 (EN-DE)
  and 38.34 -> 38.41 (EN-FR) at 8 bits; 26.98 / 37.07 at 6 bits; 18.32 / 1.59 at 4 bits. [VERIFIED]
  Fact-check caveat: the language model is PyTorch's example model (2 encoder layers, embedding and hidden size
  200, 2 heads, 10 epochs), and its 4-bit version scores 5.65 / 5.94, the same as 8 bits within 0.01. This setup
  barely separates bit widths and is weak evidence that 8 bits is free.
  https://arxiv.org/abs/1910.10485
- 16-layer Transformer on WikiText-103 (942 MB in float32), DiffQ Table 2: uncompressed PPL 18.1; row "8 bits /
  8 bits" 18.3 (+0.011 nats); row "QAT 8 bits / 8 bits" 19.7 (+0.085 nats); LSQ 4-bit weights 18.9; DiffQ at
  130 MB 18.1. [VERIFIED] Fact-check clarification: in all rows of that table with 8-bit activations, the
  activations are quantized after training by the framework (per-channel scaling); "QAT" means that only the
  weights were trained with the straight-through estimator. Appendix Table B.6 separates the two: QAT 8-bit
  weights with float activations 18.2 (+0.006 nats); with a histogram activation quantizer instead, 19.5 for
  weights rounded after training and 26.0 for QAT weights. The +0.085 therefore appears only when activations
  are quantized after training on top of QAT weights; it does not measure QAT of weights and activations
  together. https://arxiv.org/abs/2104.09987
- Same model family, Quant-Noise Table 1: uncompressed PPL 18.3; int8 applied after training 19.6 (+0.069 nats);
  int8 trained with plain QAT 21.0 (+0.138 nats); int8 trained with Quant-Noise 18.7 (+0.022 nats). int4: 39.4 /
  34.1 / 21.8. [VERIFIED] Fact-check clarification: here too QAT and Quant-Noise act on the weights during
  training; the activation scales are calibrated with a histogram method "ahead of the evaluation and then
  fixed" (their section 7.7). With per-channel ranges (their Table 10) int8 after training gives 18.5 and with
  Quant-Noise 18.3, equal to float. https://arxiv.org/abs/2004.07320
- BERT-base, GLUE average (Bondarenko et al. 2021): float 83.06; W8A8 QAT with one scale per tensor 83.26; W8A8
  post-training 71.03; W4A8 QAT 82.64; W4A8 QAT with 2-bit token embeddings 82.29. [VERIFIED]
  https://arxiv.org/abs/2109.12948
- Integer-only RoBERTa-Base (I-BERT): GLUE average 86.0 -> 86.3. INT8 multiplications, INT32 accumulation, and
  the nonlinear operations (GELU, softmax, LayerNorm) are computed on the INT32 values before requantising to
  INT8. [VERIFIED] https://arxiv.org/abs/2101.01321
- Q8BERT (8-bit QAT of all fully connected and embedding layers; softmax, LayerNorm and GELU stay float32): error
  below 1% relative except RTE. CoLA 58.48 -> 58.48 (dynamic post-training quantization 56.74), RTE 69.7 -> 68.78
  (63.32), SST-2 92.36 -> 92.24 (91.04). [VERIFIED] https://arxiv.org/abs/1910.06188
- TernaryBERT Table 1, 8-8-8 rows (weights-embedding-activations, trained with distillation): BERT MNLI-m/mm
  84.2/84.7 against float 84.5/84.9; TinyBERT 84.4/84.6 against float TinyBERT 84.5/84.5. [VERIFIED]
  https://arxiv.org/abs/2009.12812
- LLaMA-7B, zero-shot common-sense average (LLM-QAT Table 1; bits are weights-activations-KV cache): float16 66.2;
  8-8-8 round-to-nearest 65.6, SmoothQuant 66.3, LLM-QAT 66.0. 8-8-4: round-to-nearest 47.4, LLM-QAT 61.6. 4-8-4:
  41.2 / 60.7. [VERIFIED] https://arxiv.org/abs/2305.17888
- MobileLLM-125M, W8A8 post-training with per-token min-max: average 45.0 -> 44.8 (gap 0.2; all tested sizes
  within 0.5). [VERIFIED] https://arxiv.org/abs/2402.14905
- Model cards (self-reported, single author, not peer reviewed; [VERIFIED] on these three means that the card
  exists and states the numbers, not that anyone reproduced them):
  - 28.9 M parameter TinyStories model on an ESP32-S3 (556k dense core, int4 weights in groups of 128): int8
    activations per matrix-vector product cost +0.0003 nats (2.4793 -> 2.4796). RMSNorm stays float. [VERIFIED]
    https://huggingface.co/slvDev/esp32-ai-tinystories
  - 11.16 M parameter TinyStories model: embedding table at int8 instead of fp16 costs +0.0006 nats (2.1789 ->
    2.1795); ternary embedding +0.1347 nats. [VERIFIED] The rest of that model is ternary with int8 activations
    scaled per token, so this isolates the embedding table only.
    https://huggingface.co/euclidstellar/tinystories-1bit-llm
  - 999,328 parameter TinyStories model, int8 with one scale per row: "near-lossless", no number given.
    [VERIFIED] https://huggingface.co/sprapp/eeny-tinystories-999k
- llama2.c `runq.c` (Karpathy): int8 symmetric weights for matmuls only, activations quantized dynamically at run
  time, RMSNorm scale kept in float32 "because these layers are very sensitive"; quality effect described as
  "silently very slightly lower", no number. The float `stories260K` checkpoint has validation loss 1.297.
  [VERIFIED] https://github.com/karpathy/llama2.c/blob/master/README.md

### 2B. Which component is sensitive

- Weights are not the problem, activations are. BERT-base post-training, GLUE average: float 83.06; weights only
  at 8 bits (W8A32) 83.23; activations only (W32A8) 70.25; both 71.03. [VERIFIED]
  https://arxiv.org/abs/2109.12948
- Leave-one-out on the activation quantizers of BERT-base (weights float; scores STS-B / MNLI / QNLI / RTE):
  all quantized 62.64 / 42.67 / 50.74 / 48.74. Leaving one group unquantized: softmax input 70.92 / 42.54 /
  51.84 / 48.74; softmax output 72.83 / 50.35 / 50.23 / 49.46; self-attention output 70.47 / 46.57 / 50.98 /
  50.90; sum of embeddings 67.57 / 46.82 / 51.22 / 51.26 (row added by the fact-check); residual sum after the
  feed-forward block 81.57 / 82.56 / 89.73 / 67.15 (float 89.09 / 84.91 / 91.58 / 70.40). Cause given: mismatch
  between the dynamic ranges of the feed-forward input, output and their sum, with structured outliers in a few
  dimensions. Fixes that recover it: 16-bit activations for the feed-forward
  input, output and their sum (GLUE 82.43), per-embedding-group ranges with K = 6 groups (82.45), or QAT
  (83.26). [VERIFIED]
  https://arxiv.org/abs/2109.12948
- The outliers come from attention heads that learn a "no-op": to get exact zeros in the attention matrix the
  softmax input is pushed larger and larger during training. Clipped softmax and gated attention reduce them and
  allow full INT8 activation quantization of BERT, OPT and ViT (abstract; numbers not extracted). [VERIFIED]
  https://arxiv.org/abs/2306.12929
- Normalisation inputs: in vision transformers some channel ranges of the LayerNorm input exceed 40x the median
  channel range; a per-channel power-of-two factor on top of one layer scale fixes it. Attention maps are
  extremely non-uniform (most values between 0 and 0.01); 4-bit log2 quantization of the attention map with
  shifts is sufficient (post-training, about 1% accuracy drop). [VERIFIED] https://arxiv.org/abs/2111.13824
- Integer-only inference of large language models: the main obstacle named is the large fluctuation of
  activations across channels and across tokens, in linear and nonlinear operations (abstract). [VERIFIED]
  https://arxiv.org/abs/2405.17849
- Softmax with 8-bit input and 8-bit output (Softermax): base 2 instead of e, input in fixed point Q(6,2), output
  Q(1,7), running maximum, exact fixed-point forward pass and straight-through backward pass during fine-tuning.
  Against an 8-bit quantized baseline: BERT-Base SQuAD 86.28 -> 85.86, MNLI 83.27 -> 83.80; BERT-Large SQuAD
  89.40 -> 89.46; largest drop below 0.5%. The internal values are wider than 8 bits: unnormalised power Q(1,15),
  power sum Q(10,6), reciprocal Q(1,7). [VERIFIED] https://arxiv.org/abs/2103.09301
- KV cache: 8 bits is harmless, 4 bits is not (LLaMA-7B 8-8-8 against 8-8-4 above).
  https://arxiv.org/abs/2305.17888
- Embeddings and output layer: 8 bits is harmless in every source that reports it (Q8BERT, the 11 M model card,
  Bondarenko et al. even at 2 bits lose 0.35 GLUE points, https://arxiv.org/abs/2109.12948). TernaryBERT reports
  that a scale per row helps the
  word embedding (https://arxiv.org/abs/2009.12812). LSQ keeps the input and output layers at 8 bits whatever
  the rest uses (https://arxiv.org/abs/1902.08153). BitNet leaves the input/output embedding at high precision
  "because the language models have to use high-precision probabilities to perform sampling"
  (https://arxiv.org/abs/2310.11453). [VERIFIED]

### 2C. Power-of-two scales, learned ranges, rounding

- TQT trains the log2 of each clipping threshold with the straight-through estimator; the scale is
  2^ceil(log2 t) / 2^(b-1), one per tensor, symmetric, no zero point. ImageNet top-1 with 8-bit weights and
  activations: MobileNet v1 float 71.1 -> 71.1; calibration only 0.6; retraining the weights with fixed thresholds
  67.0; retraining weights and thresholds 71.1 (2.1 epochs). MobileNet v2 float 71.7 -> 71.8 (calibration only
  0.3, weights only 68.2). Inception v4, calibration only: 80.2 -> 79.4. Settings: Adam (0.9, 0.999); learning
  rate 1e-2 for the log2 thresholds and 1e-6 for the weights; threshold rate halved every 1000 x (24/N) steps;
  weight thresholds initialised at 3 standard deviations, activation thresholds by a KL-type calibration on 50
  images; thresholds frozen incrementally because they oscillate around an integer log2 level; data augmentation
  and dropout switched off; rounding is round-half-to-even to avoid a bias. [VERIFIED] The float values are
  checkpoints retrained with the same procedure; the downloaded checkpoints score 71.0 (v1) and 70.1 (v2).
  https://arxiv.org/abs/1903.08066
- The same paper (its Table 1) quotes Google's QAT with real-valued scales for comparison: MobileNet v1 70.7
  (per channel, symmetric) and 70.0 (per tensor, asymmetric) against float 70.9
  [VERIFIED: Table 4 of the original whitepaper arXiv:1806.08342 was opened by the fact-check and gives 0.707,
  0.70 and 0.709].
- White paper (Nagel et al. 2021): a power-of-two scale makes scaling a bit shift, "however, the restricted
  expressiveness of the scale factor can complicate the trade-off between rounding and clipping error". Its QAT
  pipeline: start from a pre-trained float model, initialise the ranges with post-training steps, learn the
  quantization parameters, use Adam (with an adaptive optimiser the quantizer parameters need no separate
  learning rate). For networks without depthwise-separable convolutions W8A8 and W4A8 QAT are on par with float.
  [VERIFIED] https://arxiv.org/abs/2106.08295
- LSQ: one learned step size per layer, initialised to 2<|v|>/sqrt(Qp), step-size gradient scaled by
  1/sqrt(N x Qp); always initialised from a trained float model; 8-bit networks were fine-tuned for 1 epoch at
  learning rate 0.001 (float training used 0.1, 2 to 4 bit networks 0.01 for 90 epochs); weight decay was halved
  for 3 bits and quartered for 2 bits; input and output layers always 8 bits. In most cases no accuracy gain from
  4 to 8 bits. [VERIFIED] https://arxiv.org/abs/1902.08153
- LSQ+: activations with negative values (Swish) lose accuracy under unsigned or naive signed quantization
  (EfficientNet-B0 W4A4: LSQ loses 4.1%); a learnable offset recovers 1.8% at W4A4 and up to 5.6% at W2A2; range
  initialisation by MSE minimisation gives lower variance over 5 runs than min-max or the LSQ rule. [VERIFIED]
  https://arxiv.org/abs/2004.09576
- Jacob et al. 2018: activation ranges by exponential moving average; activation quantization switched off for
  the initial 50 thousand to 2 million steps; post-training quantization "leads to significant accuracy drops for
  small models", with two named causes: weight ranges that differ by more than 100x between output channels, and
  outlier weights. [VERIFIED] https://arxiv.org/abs/1712.05877
- Shift-only normalisation exists as prior art for BatchNorm in binarized networks: the "shift based batch
  normalizing transform" replaces the multiplications by shifts with AP2(x) = sign(x) x 2^round(log2|x|)
  (MNIST, CIFAR-10, SVHN). https://arxiv.org/abs/1602.02830
- Accumulator-aware QAT (A2Q, A2Q+): bounding the L1 norm of each integer weight row guarantees that an
  accumulator of a chosen width cannot overflow; with 8-bit weights and activations and 16-bit accumulators the
  benchmark models keep 99.2% of float performance at 98.2% weight sparsity (abstract). [VERIFIED]
  https://arxiv.org/abs/2301.13376 , https://arxiv.org/abs/2401.10432

### 2D. From scratch or from a float checkpoint; staging

- LSQ, TQT, the white paper, Q8BERT, LLM-QAT all start from a trained float model (URLs above).
- Start step of quantization, Transformer base, EN-DE BLEU (Prato et al. Table 5): never quantized 26.46; start at
  step 100: 26.98; 10,000: 26.63; 50,000: 26.84; 80,000: 26.41; quantized after training 25.50. [VERIFIED]
  Fact-check correction: the earlier text said "no trend with the start step at 8 bits". The authors conclude
  the opposite, "quantizing the model early on seems preferable", and the latest start is the lowest on both
  language pairs (EN-FR: 38.55, 38.62, 38.50, 38.43). The spread is 0.57 BLEU (EN-DE) and 0.19 (EN-FR), about
  the size of the run-to-run standard deviation the paper gives for its main table (0.09 to 0.51), so the
  ordering is weak evidence. Their Table 2 gives 26.44, not 25.50, for 8-bit quantization after training; the
  paper does not explain the difference. https://arxiv.org/abs/1910.10485
- For ternary weights, 16-bit training followed by 1.58-bit training is preferable to 1.58-bit training
  throughout (11 downstream tasks); keeping the optimiser state and phasing the quantization in reduce the loss
  spike at the switch, but further training compensates for the spike anyway. [VERIFIED: abstract]
  https://arxiv.org/abs/2502.11895
- BinaryBERT: binary weights are reached by training a half-width ternary model and splitting it; performance
  drops slowly from float to ternary and sharply at binary. https://arxiv.org/abs/2012.15701
- TernaryBERT (their Table 7, row "-Initialization"): initialising from the fine-tuned float model instead of
  the pre-trained one gives CoLA 55.7 against 46.0 and RTE 72.9 against 66.4. [VERIFIED]
  https://arxiv.org/abs/2009.12812
- QKD runs three phases: the quantized student alone, teacher and student together, then distillation, because
  "the regularization effect of KD further diminishes the already reduced representation power of a quantized
  model" (abstract). [VERIFIED] https://arxiv.org/abs/1911.12491

### 2E. Weight oscillation

- Mechanism and scope (Nagel et al. 2022): with the straight-through estimator a latent weight whose optimum
  lies between two grid points keeps flipping between them; a lower learning rate reduces the amplitude but not
  the frequency; other gradient estimators show it too. The damage reported is at 3 and 4 bits in networks with
  depthwise-separable layers, partly through wrong BatchNorm statistics. ImageNet top-1: MobileNetV2 3/3 bits LSQ
  65.3, with dampening 67.8, with iterative freezing 67.6 (float 71.7 by the stated differences); 4/4 bits 70.5 /
  70.6. MobileNetV3-Small (float 65.1): 3/3 bits LSQ 52.0 -> 59.0 / 58.9; 4/4 bits 63.7 / 63.6. The input and
  output layers were kept at 8 bits. [VERIFIED] Fact-check correction: the earlier text said that the reported
  experiments are at 3 and 4 bits. The method comparisons are, but the paper's Table 2 has one 8-bit row
  (MobileNetV2, weights quantized, 20 epochs, 3 seeds): 71.79 before and 71.89 after re-estimating the BatchNorm
  statistics, against 68.99 / 71.01 at 4 bits and 64.97 / 69.50 at 3 bits. Measured this way the oscillation
  damage is 0.10 points at 8 bits, 2.02 at 4 bits and 4.53 at 3 bits. The text has no transformer experiment.
  https://arxiv.org/abs/2203.11086
- Vision transformers at low bit width (Liu et al. 2023, abstract): a learnable scale factor aggravates weight
  oscillation; query and key weights oscillate together; fixes are a statistics-based weight scale, freezing
  high-confidence weights, and a query-key reparameterisation. [VERIFIED] https://arxiv.org/abs/2302.02210
- Power-of-two thresholds oscillate too, between neighbouring integer log2 levels; TQT freezes them one after
  another. https://arxiv.org/abs/1903.08066

### 2F. Noise injection instead of rounding

- DiffQ adds independent uniform pseudo-quantization noise to the weights and uses no straight-through estimator;
  WikiText-103 PPL 18.1 at 130 MB (float 18.1 at 942 MB). Activations in that table are quantized to 8 bits by the
  framework afterwards. [VERIFIED] https://arxiv.org/abs/2104.09987
- Quant-Noise quantizes a random fraction p of the weights in each forward pass. With int8 on EfficientNet,
  top-1 by p: 80.66 (p = 0), 80.83, 80.82, 80.88, 80.92 (p = 0.8), 80.64 (p = 1). [VERIFIED]
  https://arxiv.org/abs/2004.07320

### 2G. Fewer than 8 bits at small scale

- BitNet b1.58 on small models: language models of 6 M to 48 M parameters (4 layers, hidden 32 to 256) need about
  twice the hidden size to match 16-bit training; small vision models (100k to 2.2 M) match at equal size.
  Learning rate 1e-3 (16-bit) against 1e-3 or 1e-2 (ternary); weight decay hurts with the high rate and helps with
  the low one. [VERIFIED] https://arxiv.org/abs/2407.09527
- Model card, 11.16 M parameters on TinyStories, same architecture, data and seed: float32 2.0553 nats; ternary
  QAT with fp16 embedding 2.1760 (+0.12); ternary applied after training 5.0229; fp16 model of equal memory
  (2.1 M parameters) 2.3607. The model is under-trained (1.79 tokens per parameter). [VERIFIED: card]
  https://huggingface.co/euclidstellar/tinystories-1bit-llm
- Precision scaling law fitted on 465 runs, 30 M to 220 M non-embedding parameters, 3 to 16 bits: quantized
  training behaves like a smaller model, N_eff = N (1 - e^(-Pw/gw)) (1 - e^(-Pa/ga)) (1 - e^(-Pkv/gkv)); the
  compute-optimal precision is about 7 to 8 bits; going below 4 bits needs more than 4x the parameters; the
  degradation from post-training quantization grows with the tokens-per-parameter ratio. The authors state that
  their fitted constants are unlikely to transfer to other setups. [VERIFIED] https://arxiv.org/abs/2411.04330
- Over 1500 quantized checkpoints of the Pythia suite (160 M to 12 B parameters, quantized after training to 2,
  3 and 4 bits, GPTQ in the main figures; no 8-bit data): smaller models and models trained on more tokens lose
  more from low-bit post-training quantization. [VERIFIED; Fact-check correction: the earlier text said that
  the appendix names Pythia, it is section 3.1 of the main text, and the bit widths were missing]
  https://arxiv.org/abs/2411.17691
- ResNets on ImageNet: at equal cost, models quantized mostly to 4 bits dominate the 8-bit and bfloat16 curves
  (abstract). [VERIFIED] https://arxiv.org/abs/2105.03536

### 2H. Distillation

Into a quantized student (teacher = the float model):
- LSQ with same-architecture distillation (temperature 1, equal weight on both losses): ResNet-18 top-1 at 2 / 3 /
  4 / 8 bits 67.9 / 70.6 / 71.2 / 71.1 against float 70.5. [VERIFIED] https://arxiv.org/abs/1902.08153
- TernaryBERT ablation at 2 bits: with distillation on layers and logits MNLI 83.3, CoLA 55.7, RTE 72.9; without
  the layer term 82.9 / 52.7 / 69.0; without any distillation 80.8 / 45.4 / 56.3. [VERIFIED]
  https://arxiv.org/abs/2009.12812
- LLM-QAT: logit distillation on text sampled from the float model beats training on the sampled next token as a
  label; adding attention or hidden-state distillation hurts. AdamW, weight decay 0, learning rate 2e-5, cosine.
  [VERIFIED] https://arxiv.org/abs/2305.17888

From a larger teacher into a small language model:
- Baby Llama: on 10 M words, a 58 M student distilled from an ensemble (GPT-2 705 M + LLaMA 360 M) beats both
  teachers and the same student trained without distillation. Loss = 0.5 CE + 0.5 KL, temperature 2, 6 epochs,
  learning rate 3e-4; two teachers worked better than four or six. [VERIFIED] https://arxiv.org/abs/2308.02019
  BabyLlama-2 repeats it with a 345 M student and reports that the gain is not explained by badly tuned teachers
  (abstract). [VERIFIED] https://arxiv.org/abs/2409.17312
- BabyLM 2023 findings: "Knowledge distillation using auxiliary models was often a successful approach".
  [VERIFIED] https://arxiv.org/abs/2504.08165
- MobileLLM: distilling LLaMA-v2 7B into 125 M and 350 M students slowed training by 2.6 to 3.2x and gave
  "comparable or inferior accuracy to label-based training" (their Table 16: average 43.9 -> 43.8 at 29 h ->
  93 h for 125 M; 49.1 -> 48.8 at 42 h -> 109 h for 350 M). [VERIFIED] https://arxiv.org/abs/2402.14905
- DistilGPT2: 82 M parameters distilled from the 124 M GPT-2; WikiText-103 test perplexity 21.1 against 16.3.
  The card gives no student trained without distillation. [VERIFIED] https://huggingface.co/distilbert/distilgpt2
- Sequence level against token level (Kim and Rush 2016, translation): token-level distillation helps; training
  the student on the teacher's decoded outputs helps further; their strongest student is 4.2 / 1.7 BLEU above a
  student without distillation (greedy / beam) and runs 10x faster than the teacher (abstract). [VERIFIED]
  https://arxiv.org/abs/1606.07947
- Capacity gap: a student gets worse when the teacher is too much larger (Mirzadeh et al., abstract,
  https://arxiv.org/abs/1902.03393); for language models below 3 B parameters the optimal teacher size grows
  linearly with the student size (Zhang et al., abstract, https://arxiv.org/abs/2311.07052). [VERIFIED]
- Encoder students (abstracts only): DistilBERT is 40% smaller and 60% faster than BERT and keeps 97% of its
  language-understanding scores, trained with a language-modelling, a distillation and a cosine loss
  (https://arxiv.org/abs/1910.01108); TinyBERT distils transformer layers at the pre-training and the task
  stage (https://arxiv.org/abs/1909.10351); MobileBERT is 4.3x smaller and 5.5x faster than BERT-base, trained
  from a specially designed teacher (https://arxiv.org/abs/2004.02984); pre-training stays important for compact
  models and distillation adds to it (Turc et al., https://arxiv.org/abs/1908.08962). An 8-bit TinyBERT is in 2A.
- Code models: CodeBERT and GraphCodeBERT compressed to 3 MB (160x smaller) by architecture search plus
  distillation on unlabelled data, 4.31x / 4.15x faster; the tasks are vulnerability prediction and clone
  detection, not generation (abstract). [VERIFIED] https://arxiv.org/abs/2208.07120
- Below 1 M parameters the only item found is a model card: a 999,328 parameter model "trained by knowledge
  distillation from a small full-precision teacher" scores 0.625 bits per byte on TinyStories validation against
  0.707 for TinyStories-1M [VERIFIED: the card states this; self-reported, not reproduced]. The card describes
  the comparison model as "~3M (1M non-embedding)". Fact-check correction of that size: the TinyStories-1M card
  gives no count, and its config.json (GPT-Neo, hidden size 64, 8 layers, vocabulary 50,257, 2,048 positions)
  works out to about 3.75 M parameters, about 0.40 M of them outside the two embedding tables (our arithmetic).
  The 999k model has about 0.64 M outside its embedding table, so the comparison model is about 3.7 times larger
  in total but smaller in its transformer body. The card has no run without distillation.
  https://huggingface.co/sprapp/eeny-tinystories-999k , https://huggingface.co/roneneldan/TinyStories-1M

### 2I. Cheap data-side methods

- Restricting the distribution: with stories limited to the vocabulary of 3 to 4 year olds, models "below 10
  million total parameters" or with one transformer block still write fluent, consistent text (abstract).
  [VERIFIED] https://arxiv.org/abs/2305.07759
- Data quality for code: phi-1 (1.3 B parameters, 6 B tokens of selected "textbook quality" code plus 1 B
  synthetic tokens) reaches 50.6% HumanEval pass@1 and 55.5% MBPP; the 350 M variant reaches 45% HumanEval
  (abstract). [VERIFIED: pass@1 for both benchmarks, as the abstract states] The smallest model there is 1000x
  our size. https://arxiv.org/abs/2306.11644
- Deduplication: removing near-duplicates lets models "require fewer train steps to achieve the same or better
  accuracy" and emit memorised text ten times less often; over 4% of standard validation sets overlapped with
  training data (abstract). [VERIFIED] https://arxiv.org/abs/2107.06499
- Duplicated code inflates reported metrics by up to 100% on some corpora (abstract). [VERIFIED]
  https://arxiv.org/abs/1812.06469
- Curriculum learning: in BabyLM 2023 it was the most common approach and "largely unsuccessful, though some
  showed modest improvements" (https://arxiv.org/abs/2504.08165); a separate study with several linguistically
  motivated curricula found no compelling evidence of a gain (abstract, https://arxiv.org/abs/2108.02170).
  [VERIFIED]
- Formatting of training samples: in BabyLM 2023, "when controlled comparisons were performed, these
  preprocessing steps often led to improvements"; short sequences or single sentences as samples were among the
  successful ones. [VERIFIED] https://arxiv.org/abs/2504.08165
- BabyLM 2024: results correlate strongly with training FLOPs; the top submissions changed the training data,
  the objective or the architecture. [VERIFIED] https://arxiv.org/abs/2412.05149
- Canonical formatting or identifier normalisation of code for small models: no controlled study found in our
  search.

### 2J. Our own arithmetic (computed here, not from the literature)

- Accumulator. Largest product of two int8 values is 16384. 511 x 16384 = 8,372,224 <= 8,388,607, and 512 x 16384
  = 8,388,608 overflows by one. A row of up to 511 products can never overflow the signed 24-bit accumulator;
  with weights restricted to [-127, 127] the limit is 516. Baseline rows (64, 172, 256 and two chained 255-long
  streams) are inside the limit. A bias or rounding preload uses part of the remaining headroom.
  [VERIFIED: recomputed; the 516 holds for weights in [-127, 127] with inputs in [-128, 127]]
- Exact simulation. Because every partial sum stays below 2^24, a float32 matrix product of int8-valued tensors
  equals the integer result exactly. Checked for row lengths 64, 172, 256 and 511 including worst-case rows:
  numpy float32 and torch 2.15.0 on MPS both match int64, and floor(acc / 2^k) followed by a clamp matches the
  integer shift and saturation. The bit-exact forward pass can therefore run on the laptop GPU.
  [VERIFIED: re-run by the fact-check with numpy float32 and torch 2.15.0 on MPS, same four row lengths]
- Rounding. An arithmetic shift right is a floor: mean error -0.50 LSB, RMS 0.577 LSB (shift 10, 1 M random
  accumulator values). Adding 2^(k-1) before the shift gives mean error +0.001 LSB, RMS 0.289 LSB. The error
  power, bias included, drops from 0.333 to 0.083 LSB^2 (4x, equal to one bit of resolution). Three quarters
  of the floor error is the constant -0.5 LSB bias, which exact QAT could partly learn around, so the
  measured gain may be smaller than 4x. [VERIFIED: re-run; floor mean -0.4995 and RMS 0.577, rounded mean
  +0.0008 and RMS 0.289]
- Power-of-two step, before any retraining. For Gaussian data and an 8-bit symmetric uniform quantizer the
  MSE-optimal step clips at 3.92 standard deviations (SQNR 40.54 dB). Restricting the step to a power of two
  costs 1.62 dB on average and 3.90 dB in the worst alignment (0.27 and 0.65 bit). [Fact-check correction,
  recomputed in closed form for levels -127..127: the earlier text had 40.45 dB, 1.53 dB, 3.77 dB, 0.25 and
  0.63 bit; the clip point 3.92 was confirmed]
- Effective-parameter illustration. With the form of Kumar et al. and their floating-point fit gw = 2.8111,
  8 bits gives N_eff / N = 0.942 for the weights alone and 0.942^3 = 0.836 if activations and KV cache had the
  same sensitivity [VERIFIED: constant and arithmetic]. Fact-check addition: their Appendix K lists the
  integer-type constants gw = 2.6745, ga = 2.2102 (printed as gamma_i, taken to be the activation constant) and
  gkv = 0.9578, each fitted with an offset (0.3037, 1.4072, 2.4185) whose sign convention we did not resolve;
  without the offsets 8 bits gives 0.950 x 0.973 x 1.000 = 0.924. This only illustrates the size of the
  effect; the authors write that their "numerical constants are unlikely to be useful" in another setup.

## 3. Uncertain or contested

- Whether "8 bits is free" transfers to 260k to 400k parameters. Why uncertain: every language-model number in 2A
  comes from a model of 11 M parameters or more (the one card below that, at 999k, gives no number, and the
  integer TinyStories model of Deeploy in section 4 reports no loss). Jacob et al. report that simple
  post-training quantization fails on small models; Ouyang et al. (2 to 4 bits) and Kumar et al. (GPTQ on the
  weights) report that smaller models and higher tokens-per-parameter ratios lose more from post-training
  quantization. If we train a few hundred thousand parameters on the order of a billion tokens, our ratio lies
  outside the range Kumar et al. fitted (30 M to 220 M parameters, up to about 1000 tokens per parameter), and
  the checkpoints of Ouyang et al. are far larger than our model.
- Whether plain straight-through QAT helps at 8 bits. Two papers that promote alternatives show it worse than
  quantizing after training on WikiText-103 (PPL 21.0 against 19.6 in Fan et al.; 19.7 against 18.3 in Defossez
  et al.). Prato et al., Bondarenko et al. and Q8BERT show QAT within noise of float (Prato et al. -0.08 and
  +0.07 BLEU; Bondarenko et al. +0.20 GLUE points; Q8BERT within 1% relative except RTE; Fact-check: the earlier
  text said "equal to or above float"). Fact-check correction: the earlier text put the difference down to
  range handling, start point and learning rate, "which none of the papers isolates". The papers do show one
  difference. In Fan et al. and Defossez et al. the QAT baseline trains only the weights and the activations
  are quantized after training; Prato et al., Bondarenko et al. and Q8BERT train with the activation quantizers
  in the loop. Defossez et al. Table B.6 gives 18.2 for QAT weights with float activations (+0.006 nats) and
  19.7 or 26.0 once the activations are quantized afterwards, depending on the activation quantizer. Our
  reading: the two baselines argue for having the activations in the training loop (idea 1), not against the
  straight-through estimator.
- Whether weight oscillation matters at 8 bits. The method evidence is at 4 bits or fewer, in depthwise CNNs and
  vision transformers, and part of the damage there goes through BatchNorm statistics, which we do not have. The
  one 8-bit measurement (Nagel et al. Table 2, MobileNetV2, Fact-check addition) shows 0.10 points of such
  damage against 2.02 at 4 bits. The mechanism does not depend on the bit width, but one flip is 16x smaller
  relative to the range at 8 bits than at 4. Nothing measured for a language model where each weight carries
  more.
- Learned ranges or ranges from statistics. LSQ and TQT learn them; Liu et al. report that a learnable scale
  aggravates oscillation (low-bit ViT); TQT reports that learning the scale directly needs careful tuning and
  moves to the log2 domain; LLM-QAT uses plain min-max without clipping because clipping outliers of pretrained
  LLaMA gave perplexity above 10000. The last point concerns large pretrained models with outliers; a tiny model
  trained under saturation from the start may not develop them.
- Symmetric or asymmetric activations. TernaryBERT finds min-max (asymmetric) better because BERT hidden states
  are skewed negative; LLM-QAT finds symmetric adequate for models with gated linear units. LS is symmetric
  unless a zero point is folded into the bias preload (the term z x sum(w) is a constant per row).
- When to switch quantization on. At 8 bits Prato et al. conclude that an early start "seems preferable"
  (Fact-check correction: the earlier text said "no trend"): their latest start, at step 80,000, scored 0.57
  (EN-DE) and 0.12 (EN-FR) BLEU below the start at step 100, which is weak evidence against a late switch. LSQ,
  TQT and the white paper always start from a float model; for ternary weights float-then-quantized beats
  quantized throughout (Nielsen et al.). The sources therefore point in different directions and none of them
  is a language model at 8 bits. Not tested: whether table-shaped nonlinearities must be present from step 0.
- Weight decay during QAT. LSQ lowers it with precision; LLM-QAT uses 0; BitNet b1.58 2B4T uses a peak of 0.1 and
  then 0 in its second stage (https://arxiv.org/abs/2504.12285); the small-model BitNet study finds it harmful
  with a high learning rate and helpful with a low one. No ablation at 8 bits on a language model was found.
- How much the power-of-two restriction costs after training. Zero on MobileNets at 8 bits (TQT); the white paper
  warns of a harder rounding/clipping trade-off; our Gaussian estimate is 0.27 bit on average before retraining.
  Not measured for a transformer language model.
- How much softmax input precision is needed. In BERT post-training the softmax input is not the bottleneck
  (2B); Softermax works with 2 fractional bits; but outliers grow from softmax inputs being pushed up to produce
  exact zeros. With an 8-bit table output the usable range below the table's top is about 5 nats (ln 127 = 4.8
  to ln 254 = 5.5, depending on how the entries are rounded); whether a tiny model needs more is unknown.
- Distillation from a larger teacher. Positive when data is the limit (BabyLM, 10 M to 100 M words), no gain when
  it is not (MobileLLM, 0.25 T tokens, teacher 20 to 56 times the student). Capacity-gap results say a far larger
  teacher can be worse than a moderately larger one. Our setting has plenty of data per parameter, which is the
  MobileLLM side of this disagreement.
- Curriculum learning. Mostly negative (BabyLM 2023, Campos 2021), with some modest positive reports. Ordering
  by data quality in stages, as opposed to difficulty, is not covered by those results.
- Fewer bits with more parameters. Kumar et al.: 7 to 8 bits is compute-optimal (30 M to 220 M parameters).
  Abdolrashidi et al.: 4 bits is Pareto-optimal for ResNets. Small-model BitNet: ternary needs about twice the
  hidden size, which is about 4x the multiply-adds. For LS the multiplier is 8x8 and speed is a target, so int8
  stays; the memory-only argument for 4-bit weights is not settled by these sources.

## 4. Not tried at our scale (about 260k to 400k int8 parameters, 8-bit multiply-accumulate CPU, 450 KB)

Not found in our search:
- A QAT study of an autoregressive language model at or below 1 M parameters with 8-bit weights and 8-bit
  activations. Nearest items: an encoder for classification that fits 2 MB and drops to 781 kB at 8 bits
  (EmbBERT, abstract, https://arxiv.org/abs/2502.10001); a 6k-parameter ECG transformer with 98.97% accuracy
  under 8-bit integer inference (abstract, https://arxiv.org/abs/2402.10748); BERT-tiny / BERT-mini on
  microcontrollers within 256 KB of working memory (abstract, https://arxiv.org/abs/2410.17957); the three model
  cards in 2A. [VERIFIED: the three abstracts]
  Fact-check addition, two nearer items that the earlier text missed. (a) Deeploy
  (https://arxiv.org/abs/2408.04413) runs a TinyStories Llama (`Maykeye/TinyLLama-v0`: hidden size 64, 8 layers,
  16 heads, feed-forward 256, vocabulary 32,000; 4,621,392 parameters, about 0.53 M of them outside the two
  embedding tables) on a RISC-V microcontroller at 340 tokens/s with all matmul weights and activations as
  8-bit integers. The model is quantized after training (QuantLib with TQT "for PTQ"), uses the I-BERT
  polynomial softmax, a Hardswish approximation and 32-bit divisions in RMSNorm, and the paper reports no loss
  or perplexity for the quantized model [VERIFIED]. (b) Z80-uLM (https://github.com/HarryR/z80ai, README only):
  a character-level generator for the Z80 with 2-bit weights trained with QAT, 16-bit integer arithmetic and an
  arithmetic shift right by 2 after each layer; it is not a transformer and reports no loss.
- Static power-of-two scales end to end in a transformer language model. The TQT paper's own experiments are CNN
  only. Deeploy (above) applies TQT to a TinyStories Llama after training, but it stores requantisation
  parameters for every layer and does not say that its scales are pure shifts (Fact-check correction: the
  earlier text said "TQT is CNN only"). I-ViT (shift-based softmax and GELU, abstract,
  https://arxiv.org/abs/2207.01405) and FQ-ViT are vision. I-LLM is integer-only for large models but quantizes
  inputs and outputs dynamically at run time.
- Nonlinearities with an 8-bit-only path: an exp, 1/sqrt or SiLU table indexed by an int8 value with an int8
  result. Softermax has 8-bit input and output but 16-bit intermediates and a reciprocal unit. I-BERT feeds its
  nonlinearities 32-bit integers.
- An 8-bit residual stream kept through all layers with one fixed scale. Sources either leave the residual sum at
  16 bits, use per-group or per-channel factors, or rescale per token.
- Oscillation measurements in any language model. At 8 bits only the single MobileNetV2 row of Nagel et al.
  (2E) was found (Fact-check correction: the earlier text listed 8 bits as not found at all).
- A controlled distillation result (same student with and without a teacher) below 1 M parameters.
- Normalisation-free or softmax-free variants below 1 M parameters: element-wise tanh in place of normalisation
  (abstract, https://arxiv.org/abs/2503.10622), sigmoid attention (abstract, https://arxiv.org/abs/2409.04431),
  ReLU attention divided by sequence length (vision, abstract, https://arxiv.org/abs/2309.08586). [VERIFIED: the
  three abstracts]
- Accumulator-aware training for a 24-bit accumulator in a language model. By 2J it is not needed while rows
  stay at or below 511 products.
- Tied input/output embedding under one int8 table. As input the rows are activations at the residual-stream
  scale; as output weights they would prefer to fill the int8 range. No source discusses this conflict.
- A controlled study of canonical code formatting, identifier normalisation or a TinyStories-style simplified
  code corpus for models below 1 M parameters (other agents may cover data sources).
- MobileBERT with int8 quantization: no result was opened in this search, so nothing is claimed about it.

## 5. Ranked ideas for LS-LLM

Measurement rule used in the table: validation loss in nats per token computed with the bit-exact integer forward
pass, plus pass rates on the practice sets only; two seeds; keep an idea when its effect is larger than twice the
spread between seeds. Cycle figures use the CONTEXT.md estimates (row of n costs n + 24 cycles, element-wise
operation 12 to 32 cycles per element, about 1.0 M cycles per token) and are estimates, not measurements.

| # | Idea | Expected effect on score / speed / memory | Cost to try | Risk | Keep-or-drop measurement | CPU change |
|---|---|---|---|---|---|---|
| 1 | Bit-exact integer forward pass in PyTorch (float32 tensors holding integers, exact by 2J) as the last QAT stage, straight-through backward; equality test against the CPU emulator | Score: removes the train/deploy mismatch (in sources, quantizing without training ranges from +0.0003 nats to collapse). Speed, memory: none | 2 to 4 days | Gradients vanish where values saturate; training slower than float | Integer loss minus float loss; token-for-token equality with the emulator on at least 10k tokens | No |
| 2 | Round to nearest by adding 2^(k-1) to the accumulator preload before the shift | Requantisation error power, bias included, 0.333 -> 0.083 LSB^2 (2J); the gain after QAT may be smaller because training can absorb a constant bias. Speed: 0 cycles if the row already loads a bias. Memory: 0 | 1 hour in emulator and exporter | None known | Integer loss with floor against with rounding | No, if "load ACC with a constant" exists; otherwise a rounding option in RQ |
| 3 | Clipping ranges as learned log2 thresholds (TQT) for every activation tensor; weight scales set from statistics (3 standard deviations or a percentile, snapped to a power of two), one per tensor, one per row only where a tensor needs it | Closes the gap between fixed and trained ranges (MobileNet v1: 67.0 -> 71.1 in TQT). Speed, memory: none | 2 to 3 days | Thresholds hop between integer levels: freeze them one by one | Integer loss; share of saturated values per tensor | No |
| 4 | Short QAT fine-tune from the float checkpoint: weights at about 1/100 of the float peak learning rate, thresholds at 1e-2 with Adam, weight decay 0, dropout off, loss 0.5 CE + 0.5 KL to the float model's own logits at temperature 1 | Planning assumption: total increase stays within +0.02 to +0.05 nats [UNVERIFIED: assumption, see section 1]. One extra float forward pass per step | 1 to 2 days plus a few percent of the float training time | Too short a schedule; settings come from CNN papers and need re-tuning | Loss against steps; with and without the distillation term | No |
| 5 | One shared power-of-two scale for the whole residual stream, with a clamp or a penalty in float training that keeps its RMS inside a fixed band | Protects the tensor that dominates the loss in BERT (2B). Speed: residual adds need no rescaling | 2 days | The clamp may cost float loss | Float loss with and without the clamp; integer loss | No |
| 6 | Table-shaped nonlinearities in training: exact 256-entry table in the forward pass, derivative of the smooth function in the backward pass (the Softermax recipe); tables generated per scale, so any factor that is not a power of two is absorbed into a table | Removes the mismatch at softmax, norm and SiLU. Memory: 256 bytes per table | 3 to 5 days | The 8-bit-only path is untested (section 4) | Loss change when each table is switched on separately | No (uses the planned table instruction) |
| 7 | Normalisation ablation in float: (a) RMSNorm with a 1/sqrt table that is accurate only inside a 2x RMS band, (b) element-wise tanh table in place of normalisation, (c) normalise by the nearest power of two | (b) removes the sum of squares and the reciprocal; element-wise work stays about equal, so the gain is precision, not speed. (c) turns the scaling into a shift | 2 to 3 days | (b) and (c) untested below 1 M parameters | Float loss at equal steps; cycles per token | (a) no, (b) no, (c) yes: shift amount from a register |
| 8 | Attention ablation in float: exp table anchored at the top of the int8 range so that no maximum pass is needed (saturating scores), with or without normalisation; sigmoid attention as the variant without normalisation | A maximum pass would cost about 256 x 8 element-wise steps per layer (25k to 66k cycles); anchoring avoids it. Dropping normalisation removes the reciprocal, the hardest integer step | 2 to 3 days | Score loss; sigmoid attention needs its initial attention norms controlled (paper abstract) | Float loss at equal steps; cycles per token | No |
| 9 | Fold the residual add into the last matmul of each branch: for each output element preload the accumulator with the stream value shifted left, then stream the products, then requantise once | Saves 640 element-wise adds per token (7.7k to 20.5k cycles, 0.8 to 2.0%) and one rounding step per element | 1 day in emulator and training code | Needs the instruction | Cycles per token; integer loss | Yes: load ACC from data memory with a left shift |
| 10 | Uniform noise of +/- half a step on the weights during the last part of float training (DiffQ style) before QAT starts | Cheap robustness; in DiffQ the noise-trained model matched float perplexity at 130 MB against 942 MB. Speed, memory: none | Half a day | No gain at 8 bits | Loss right after rounding, with and without the noise phase | No |
| 11 | Oscillation guard at the end: freeze weights whose integer value keeps flipping; choose among the last checkpoints on the practice set | Small or zero at 8 bits (one CNN measurement: 0.10 points, 2E; nothing for language models) | Half a day | None | Flip counts; spread of the loss over the last checkpoints | No |
| 12 | Logit distillation from a larger teacher with the same tokenizer (10 to 50 times the parameters, trained on the same data; 0.5/0.5 loss, temperature 2 as in Baby Llama) | Unknown: positive when data-limited, none in MobileLLM, no controlled result below 1 M | Teacher training (hours to days on the laptop) and a slower student step (2.6 to 3.2x in MobileLLM, with a far larger teacher) | No gain | Practice-set pass rates, two seeds, same number of student steps | No |
| 13 | Sequence-level distillation: a simple, clean Python corpus written or filtered by a strong code model (the TinyStories pattern); independent of our tokenizer | The data-side approach with the most direct evidence for tiny models (TinyStories); effect size unknown for code | Generation budget; licence check of the generator and its outputs | Corpus too narrow for HumanEval and MBPP prompts | Practice-set pass rates at an equal number of training tokens | No |
| 14 | Data hygiene: remove exact and near duplicates; remove any overlap between training data and every practice or test set; one canonical code format | Fewer steps to the same loss (Lee et al.); honest scores (Allamanis). Canonical format: unproven | 1 to 2 days | Benchmark prompts are not in the canonical format | Loss per byte; count of leaked items; practice-set pass rates | No |
| 15 | Static export check: weights within [-127, 127], every accumulated row (chained streams included) at most 511 products, preload inside the headroom | Guarantees no accumulator overflow without any training constraint | 1 hour | None | Check passes on every exported image | No |

Not recommended now, with the reason:
- Ternary or 4-bit weights: for ternary weights about twice the hidden size is needed for equal quality at small
  scale (2G). For 4-bit weights the sources show small losses on large models only (BERT W4A8 QAT 82.64 against
  83.06, LSQ 4-bit PPL 18.9 against 18.1) and nothing at our scale (Fact-check correction: the earlier text
  applied the "twice the hidden size" result to 4 bits as well). Either way the LS multiplier is already 8x8,
  and tokens per second is a target.
- Per-token dynamic activation scales (LLM-QAT, BitNet, llama2.c `runq`): LS has a constant shift only.
- 16-bit residual stream: doubles the cost of every stream that reads it on an 8-bit data path.
- Curriculum by difficulty: mostly negative results.
- An accumulator-aware weight penalty (A2Q): unnecessary while rows stay at or below 511 products.

Design notes that follow from the constraints (our reasoning, to be checked by experiment):
- A lookup table can hold any real factor. The power-of-two restriction therefore only binds on paths without a
  table: matmul outputs and the residual add. Normalisation gains can be folded into the next weight matrix
  before quantization; QAT has to run on the folded graph so that training and inference compute the same thing
  (the same rule as BatchNorm folding in Jacob et al. and TQT).
- A 1/sqrt table indexed by a sum of squares that was shifted by a constant is coarse at small indices: the
  relative step is about 1/(2v) at index v (1.6% at 32, 6% at 8). Indices 32 to 127 span a factor 1.99 in RMS.
  A fixed-shift RMSNorm is accurate only if training keeps the stream RMS inside a 2x band (idea 5).
- An exp table with an 8-bit result (largest entry 127) has exact zeros for scores more than 4.8 nats (entries
  floored, ln 127) or 5.5 nats (entries rounded, ln 254) below its top. Exact zeros at a finite score gap are
  what the clipped softmax of Bondarenko et al. 2023 is designed to allow (Fact-check: "the same mechanism"
  softened, their method was read as an abstract only); whether the table also prevents outliers in a tiny
  model is untested.
- SiLU has a minimum of -0.278, so its int8 output uses few negative levels. LSQ+ gains from an offset were shown
  at 4 bits or fewer; at 8 bits this is expected to be minor and an offset cannot be folded through the SwiGLU
  product anyway.
- ReLU is exact under requantisation (RQ already has it). Comparing a ReLU
  feed-forward block with SwiGLU at equal cycles belongs to the architecture search; from the QAT side ReLU has
  no table error and no element-wise product.

## 6. Tables asked for in the topic brief

### 6.1 Methods (required table)

"Smallest scale" means the smallest model for which a number was read in the source, not necessarily the smallest
in the paper. arXiv identifiers resolve as https://arxiv.org/abs/<id>.

| Method | Key idea | Hyper-parameters that matter | Reported result at the smallest scale read | Risk for our case (int8, power-of-two scales, 24-bit accumulator, table nonlinearities) | Source |
|---|---|---|---|---|---|
| Straight-through estimator (STE) | Round in the forward pass, treat rounding as identity in the backward pass; zero gradient outside the clip range | Clip range; learning rate (a lower rate shrinks the flip amplitude, not its frequency) | Theory for a two-layer network with binarized ReLU: a properly chosen STE gives a descent direction (abstract) | Biased gradient; latent weights near a rounding boundary keep flipping; values that saturate in RQ get no gradient | 1308.3432 (metadata), 1903.05662, 2203.11086 |
| STE variants: EWGS, DSQ | Scale each gradient element by its rounding error (EWGS); replace rounding by a soft function that sharpens during training (DSQ) | Scaling factor; sharpness schedule | Numbers not extracted (abstracts). Nagel et al. report that oscillation persists under several STE variants | No evidence of a benefit at 8 bits; extra hyper-parameters | 2104.00903, 1908.05033, 2203.11086 |
| DoReFa-Net | Low-bit weights, activations and gradients | Bit widths of weights / activations / gradients | AlexNet with 1-bit weights, 2-bit activations, 6-bit gradients: 46.1% ImageNet top-1 (abstract) | Aimed at 1 to 4 bits; no learned clipping, so no handle on power-of-two ranges | 1606.06160 |
| PACT | Learn the clipping level alpha of an activation by gradient descent | Initial alpha and its regulariser [UNVERIFIED: paper body not opened, abstract only] | 4-bit weights and activations "comparable to full precision" on several models (abstract; smallest model not extracted) | One-sided clipping for ReLU, our stream is signed; alpha is real-valued and must be snapped to a power of two, which is what TQT does | 1805.06085 |
| LSQ | Learn the step size per layer with a scaled step gradient | Init 2<\|v\|>/sqrt(Qp); gradient scale 1/sqrt(N x Qp); start from float; at 8 bits 1 epoch at 1/100 of the float learning rate; less weight decay at low precision | ResNet-18, 8 bits with same-architecture distillation: 71.1 against float 70.5 | Step is a free real number and needs snapping to a power of two; learned scales reported to aggravate oscillation at low bit widths | 1902.08153 |
| LSQ+ | Add a learned offset; initialise ranges by MSE | Initialisation rule (variance over 5 runs) | EfficientNet-B0: +1.8% over LSQ at W4A4, up to +5.6% at W2A2 | No zero point in the MAC path unless folded into the bias; shown at 4 bits or fewer | 2004.09576 |
| TQT | Learn log2 of the clipping threshold; scale forced to a power of two; one scale per tensor; symmetric | Adam; rate 1e-2 for thresholds against 1e-6 for weights; threshold init (3 standard deviations, KL-type calibration); incremental freezing; no dropout or augmentation | MobileNet v1: float 71.1 -> INT8 71.1; fixed thresholds 67.0; calibration only 0.6 | Closest match to LS. Accuracy evidence is CNN only (Deeploy applies it after training to a TinyStories Llama without reporting a loss); it rounds half to even, RQ floors unless a rounding constant is preloaded | 1903.08066 |
| QAT of Jacob et al. | Simulated quantization in training; integer-only inference with a real-valued multiplier plus shift; ranges by moving average | Smoothing close to 1; activation quantization delayed by 50k to 2M steps | COCO detection with MobileNet: mAP 21.7 at 8 bits against 22.1 float | Real multipliers and zero points do not exist on LS; ranges are not powers of two | 1712.05877 |
| Oscillation dampening, iterative freezing | Pull latent weights towards bin centres; freeze weights that flip too often | Dampening weight and its schedule; flip-frequency threshold (annealed 0.04 -> 0.01 gave 70.33 on 3-bit MobileNetV2, the highest value read in their Table 5) | MobileNetV3-Small: 3/3 bits 52.0 -> 59.0 / 58.9; 4/4 bits 63.7 / 63.6 (float 65.1) | Method evidence at 3 to 4 bits in CNNs only (their one 8-bit row shows 0.10 points of damage); the BatchNorm symptom does not apply to us; cheap to add at the end | 2203.11086 |
| Statistics-based scale, confidence-guided freezing, query-key reparameterisation | Do not learn the weight scale; freeze confident weights; decouple Q and K | Not extracted (abstract) | Low-bit ViT; numbers not extracted | Low-bit vision only; "scale from statistics" fits power-of-two scales per tensor | 2302.02210 |
| DiffQ | Uniform pseudo-quantization noise on weights, bits per group learned, no STE | Size penalty lambda; group size | WikiText-103, 16 layers: PPL 18.1 at 130 MB (float 18.1 at 942 MB) | Weights only; the noisy forward pass is not the hardware forward pass, so a final exact stage is still needed; code licence CC BY-NC 4.0 | 2104.09987 |
| Quant-Noise | Quantize a random fraction p of the weights in each step | p (0.8 gave the highest int8 accuracy in their Table 9) | WikiText-103: int8 PPL 18.7 against float 18.3 | Large models only; same remark about a final exact stage | 2004.07320 |
| Q8BERT | 8-bit symmetric fake quantization of linear and embedding layers; ranges from max\|W\| and a moving average of max\|x\| | Moving-average rate | BERT-Base: within 1% relative on GLUE except RTE | Softmax, LayerNorm and GELU stay float; real-valued scales | 1910.06188 |
| FullyQT | Quantize all weights and nearly all activations, including the inside of softmax and LayerNorm; one weight scale per output row | Start step (authors: an early start "seems preferable", differences at most 0.57 BLEU); per-row scales on or off | Transformer language model: WikiText-103 loss 5.91 -> 5.94 | Scales and biases stay float; the paper replaces exp by a step function at inference | 1910.10485 |
| I-BERT | Integer-only: polynomial GELU and exp, integer square root | Polynomial coefficients (fixed in the paper) | RoBERTa-Base GLUE 86.0 -> 86.3 | Its nonlinearities take 32-bit inputs; LS offers an int8 index and a 24-bit accumulator | 2101.01321 |
| Softermax | Base-2 softmax, 8-bit input and output, running maximum, exact fixed-point forward pass during fine-tuning | Input format Q(6,2); output format Q(1,7) | BERT-Base: SQuAD 86.28 -> 85.86, MNLI 83.27 -> 83.80 | 16-bit intermediates and a reciprocal unit; otherwise the closest published match to a table softmax | 2103.09301 |
| FQ-ViT | Per-channel power-of-two factor for normalisation inputs; 4-bit log2 attention map | Not extracted | ViT, DeiT, Swin without training: about 1% drop (abstract) | Vision, no training; per-channel shifts are element-wise work on LS | 2111.13824 |
| Mixed precision, per-embedding-group ranges, or QAT for BERT | 16-bit residual sum; a few range groups over the embedding dimension; or QAT | Number of groups K (3 to 6) | BERT-base W8A8 QAT 83.26 against float 83.06 | A 16-bit stream doubles its cost; groups need per-element shifts | 2109.12948 |
| Clipped softmax, gated attention | Let heads produce exact zeros without extreme logits | Not extracted (abstract) | BERT, OPT, ViT with full INT8 activations (abstract; numbers not extracted) | A truncated exp table already gives exact zeros (our inference) | 2306.12929 |
| TernaryBERT | Ternary weights, 8-bit min-max activations, distillation on layers and logits | Scale granularity (per row for embeddings); loss terms | 8-8-8 TinyBERT: MNLI 84.4/84.6 against 84.5/84.5 | Asymmetric activations; BERT scale | 2009.12812 |
| BinaryBERT | Train a half-width ternary model, split it into a binary one, fine-tune | Not extracted | Binary BERT, 24x smaller, slight drop (abstract) | Not needed at int8; shows that staging helps with a hard target | 2012.15701 |
| LLM-QAT | Distil logits on text sampled from the float model; quantize the KV cache; symmetric min-max ranges | AdamW, weight decay 0, rate 2e-5, cosine | LLaMA-7B at 8-8-8: 66.0 against 66.2 | Per-token dynamic activation scale; 7 B parameters and more; code licence CC BY-NC 4.0 | 2305.17888 |
| BitNet b1.58 on small models | Ternary weights from scratch, 8-bit activations | Learning rate 1e-3 or 1e-2; weight decay; mean or median scale | Language models of 6 M to 48 M parameters need about 2x the hidden size | An 8x8 multiplier gains no speed from ternary weights; about 4x the multiply-adds | 2407.09527, 2402.17764, 2310.11453 |
| 16-bit then 1.58-bit pre-training | Switch precision during training; keep the optimiser state; phase in | Switch point | Preferable to ternary throughout on 11 tasks | Ternary evidence; supports float-then-quantize | 2502.11895 |
| A2Q, A2Q+ | Bound the L1 norm of each integer weight row so that a chosen accumulator width cannot overflow | Accumulator width | 8-bit models with 16-bit accumulators keep 99.2% of float performance (abstract) | Not needed while rows have at most 511 products | 2301.13376, 2401.10432 |
| Distillation into the quantized student (Apprentice, LSQ, QKD, KDLSQ-BERT) | The float model teaches its quantized copy | Temperature 1 and equal loss weights (LSQ); phases (QKD) | ResNet-18: 3 bits 70.6, 8 bits 71.1, float 70.5 (LSQ) | Low; one extra float forward pass per step | 1711.05852, 1902.08153, 1911.12491, 2101.05938 |
| Incremental network quantization | Quantize a growing share of the weights, retrain the rest | Partition schedule | CNNs with power-of-two weights (abstract) | Constrains weights, not scales; only the staging idea carries over | 1702.03044 |
| Shift-based normalisation | Replace the multiplications of a normalisation by shifts to the nearest power of two | None beyond the usual | Binarized networks on MNIST, CIFAR-10, SVHN | BatchNorm only; as a per-token norm it needs a variable shift; untested for language models | 1602.02830 |

### 6.2 Reported change for 8-bit weights and 8-bit activations

| Model (size) | Metric | Float | 8-bit W and A | Change | How | Source |
|---|---|---|---|---|---|---|
| Transformer LM (243 MB float32) | WikiText-2 loss | 5.65 | 5.64 | -0.01 nats | QAT, softmax and norm internals included | 1910.10485 |
| same | WikiText-103 loss | 5.91 | 5.94 | +0.03 nats (ln PPL ratio +0.023) | same | 1910.10485 |
| 16-layer Transformer (942 MB) | WikiText-103 PPL | 18.1 | 18.3 | +0.011 nats | row "8 bits / 8 bits" | 2104.09987 |
| same | WikiText-103 PPL | 18.1 | 19.7 | +0.085 nats | weights trained with plain QAT, activations quantized after training | 2104.09987 |
| same family | WikiText-103 PPL | 18.3 | 19.6 | +0.069 nats | int8 after training | 2004.07320 |
| same family | WikiText-103 PPL | 18.3 | 21.0 | +0.138 nats | weights trained with plain QAT, activation ranges calibrated after training | 2004.07320 |
| same family | WikiText-103 PPL | 18.3 | 18.7 | +0.022 nats | Quant-Noise | 2004.07320 |
| 28.9 M TinyStories (model card) | validation loss | 2.4793 | 2.4796 | +0.0003 nats | int8 activations only, dynamic scale; weights already int4 | huggingface.co/slvDev/esp32-ai-tinystories |
| 11.16 M TinyStories (model card) | validation loss | 2.1789 | 2.1795 | +0.0006 nats | int8 embedding table only (rest of the model ternary, int8 activations) | huggingface.co/euclidstellar/tinystories-1bit-llm |
| BERT-base | GLUE average | 83.06 | 83.26 | +0.20 points | QAT, one scale per tensor | 2109.12948 |
| BERT-base | GLUE average | 83.06 | 71.03 | -12.03 points | after training, static, one scale per tensor | 2109.12948 |
| RoBERTa-Base | GLUE average | 86.0 | 86.3 | +0.3 points | integer-only QAT | 2101.01321 |
| LLaMA-7B | common-sense average | 66.2 | 66.0 (65.6 without training) | -0.2 points | LLM-QAT, KV cache also 8 bits | 2305.17888 |
| MobileLLM-125M | common-sense average | 45.0 | 44.8 | -0.2 points | after training, per-token min-max | 2402.14905 |
| LS-LLM, 260k to 400k | validation loss | not trained yet (float `stories260K`: 1.297) | not measured | planning assumption +0.02 to +0.05 nats [UNVERIFIED: no source at this scale] | staged QAT of section 5 | assumption, to be replaced by a measurement |

### 6.3 Component sensitivity

| Component | Evidence | Consequence for LS-LLM |
|---|---|---|
| Residual sum after the feed-forward block | Leaving it unquantized lifts BERT MNLI from 42.67 to 82.56 (2109.12948) | One fixed scale for the stream; keep its RMS in a band; quantize it last and watch the share of saturated values |
| Normalisation input and 1/sqrt | Channel ranges above 40x the median in ViT (2111.13824); I-BERT feeds 32-bit values; llama2.c keeps RMSNorm in float; Deeploy divides with 32-bit operands (2408.04413) | The table norm is the least supported step: run the ablation of idea 7 |
| Attention map (softmax output) | Second-largest single recovery on STS-B and MNLI in the leave-one-out (2109.12948); 4-bit log2 maps suffice in ViT (2111.13824) | An 8-bit table result is adequate in sources; exact zeros come for free |
| Softmax input | Small effect in the leave-one-out (2109.12948); Q(6,2) input works after fine-tuning (2103.09301) | Two fractional bits per base-2 unit is a published working point |
| KV cache | 8 bits fine, 4 bits not, on LLaMA-7B (2305.17888) | int8 cache as planned |
| Weights | W8A32 83.23 against float 83.06 (2109.12948) | int8 with one power-of-two scale per tensor; per-row scales only if a tensor needs them |
| Embedding and output layer | int8 costs +0.0006 nats (model card); 2 bits cost 0.35 GLUE points (2109.12948) | int8 is enough; the open point is the scale of the tied table |

### 6.4 Starting settings for the QAT stage (taken from the sources, to be re-tuned on the practice loss)

| Setting | Starting value | Taken from |
|---|---|---|
| Start point | Converged float checkpoint with hardware-shaped operations | LSQ, TQT, white paper |
| Length | 1 to 5% of the float schedule | LSQ (1 epoch at 8 bits), TQT (under 5 epochs) |
| Weight learning rate | About 1/100 of the float peak, cosine to zero | LSQ |
| Threshold learning rate | 1e-2 on the log2 thresholds with Adam, halved on a schedule, then frozen one by one | TQT |
| Weight decay, dropout | 0, off | LLM-QAT and BitNet b1.58 2B4T (second stage); TQT |
| Loss | 0.5 CE + 0.5 KL to the float model's logits at temperature 1; no hidden-state terms | LSQ; LLM-QAT |
| Order | Weights, then activations, then tables, then the exact integer pass | Jacob et al. (delayed activation quantization); our staging |
| End | Freeze weights that keep flipping; choose among the last checkpoints on the practice set | Nagel et al. 2022 |

## 7. Sources

"Read" column: text = body read through ar5iv or arXiv HTML with targeted extraction; abstract = arXiv API
abstract only; metadata = title and authors only; card = README or model card. Licences are given for code,
models and datasets and were read at the source (GitHub API or licence file, Hugging Face card metadata). None
of the code or models listed here is used by LS-LLM yet.

| # | Title | Authors | Year | URL | Read | Licence (code, model, dataset) |
|---|---|---|---|---|---|---|
| 1 | Learned Step Size Quantization | Esser, McKinstry, Bablani, Appuswamy, Modha | 2019 | https://arxiv.org/abs/1902.08153 | text | paper |
| 2 | LSQ+: Improving low-bit quantization through learnable offsets and better initialization | Bhalgat, Lee, Nagel, Blankevoort, Kwak | 2020 | https://arxiv.org/abs/2004.09576 | text | paper |
| 3 | PACT: Parameterized Clipping Activation for Quantized Neural Networks | Choi, Wang, Venkataramani, Chuang, Srinivasan, Gopalakrishnan | 2018 | https://arxiv.org/abs/1805.06085 | abstract | paper |
| 4 | DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients | Zhou, Wu, Ni, Zhou, Wen, Zou | 2016 | https://arxiv.org/abs/1606.06160 | abstract | paper |
| 5 | Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks | Jain, Gural, Wu, Dick | 2019 | https://arxiv.org/abs/1903.08066 | text | code https://github.com/Xilinx/graffitist : BSD 3-Clause (licence file) |
| 6 | Overcoming Oscillations in Quantization-Aware Training | Nagel, Fournarakis, Bondarenko, Blankevoort | 2022 | https://arxiv.org/abs/2203.11086 | text | code https://github.com/qualcomm-ai-research/oscillations-qat : BSD-3-Clause-Clear |
| 7 | A White Paper on Neural Network Quantization | Nagel, Fournarakis, Amjad, Bondarenko, van Baalen, Blankevoort | 2021 | https://arxiv.org/abs/2106.08295 | text | paper |
| 8 | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | Jacob, Kligys, Chen, Zhu, Tang, Howard, et al. | 2017 | https://arxiv.org/abs/1712.05877 | text | paper |
| 9 | Quantizing deep convolutional networks for efficient inference: A whitepaper | Krishnamoorthi | 2018 | https://arxiv.org/abs/1806.08342 | Table 4 (opened by the fact-check) | paper |
| 10 | Differentiable Model Compression via Pseudo Quantization Noise | Défossez, Adi, Synnaeve | 2021 | https://arxiv.org/abs/2104.09987 | text | code https://github.com/facebookresearch/diffq : CC BY-NC 4.0 (licence file), archived |
| 11 | Training with Quantization Noise for Extreme Model Compression | Fan, Stock, Graham, Grave, Gribonval, Jegou, et al. | 2020 | https://arxiv.org/abs/2004.07320 | text | paper |
| 12 | BitNet: Scaling 1-bit Transformers for Large Language Models | Wang, Ma, Dong, Huang, Wang, Ma, et al. | 2023 | https://arxiv.org/abs/2310.11453 | text | paper |
| 13 | The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits | Ma, Wang, Ma, Wang, Wang, Huang, et al. | 2024 | https://arxiv.org/abs/2402.17764 | abstract | paper |
| 14 | BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks | Nielsen, Schneider-Kamp | 2024 | https://arxiv.org/abs/2407.09527 | text | code https://github.com/schneiderkamplab/bitlinear : MIT |
| 15 | Continual Quantization-Aware Pre-Training: When to transition from 16-bit to 1.58-bit pre-training for BitNet language models? | Nielsen, Schneider-Kamp, Galke | 2025 | https://arxiv.org/abs/2502.11895 | text | paper |
| 16 | BitNet b1.58 2B4T Technical Report | Ma, Wang, Huang, Zhang, Hu, Song, et al. | 2025 | https://arxiv.org/abs/2504.12285 | text | paper |
| 17 | I-BERT: Integer-only BERT Quantization | Kim, Gholami, Yao, Mahoney, Keutzer | 2021 | https://arxiv.org/abs/2101.01321 | text | code https://github.com/kssteven418/I-BERT : MIT |
| 18 | Q8BERT: Quantized 8Bit BERT | Zafrir, Boudoukh, Izsak, Wasserblat | 2019 | https://arxiv.org/abs/1910.06188 | text | paper |
| 19 | TernaryBERT: Distillation-aware Ultra-low Bit BERT | Zhang, Hou, Yin, Shang, Chen, Jiang, et al. | 2020 | https://arxiv.org/abs/2009.12812 | text | paper |
| 20 | BinaryBERT: Pushing the Limit of BERT Quantization | Bai, Zhang, Hou, Shang, Jin, Jiang, et al. | 2020 | https://arxiv.org/abs/2012.15701 | text | paper |
| 21 | LLM-QAT: Data-Free Quantization Aware Training for Large Language Models | Liu, Oguz, Zhao, Chang, Stock, Mehdad, et al. | 2023 | https://arxiv.org/abs/2305.17888 | text | code https://github.com/facebookresearch/LLM-QAT : CC BY-NC 4.0 (licence file) |
| 22 | Fully Quantized Transformer for Machine Translation | Prato, Charlaix, Rezagholizadeh | 2019 | https://arxiv.org/abs/1910.10485 | text | paper |
| 23 | Understanding and Overcoming the Challenges of Efficient Transformer Quantization | Bondarenko, Nagel, Blankevoort | 2021 | https://arxiv.org/abs/2109.12948 | text | code https://github.com/qualcomm-ai-research/transformer-quantization : BSD-3-Clause-Clear |
| 24 | Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing | Bondarenko, Nagel, Blankevoort | 2023 | https://arxiv.org/abs/2306.12929 | abstract | paper |
| 25 | FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer | Lin, Zhang, Sun, Li, Zhou | 2021 | https://arxiv.org/abs/2111.13824 | text | code https://github.com/megvii-research/FQ-ViT : Apache-2.0 |
| 26 | Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers | Stevens, Venkatesan, Dai, Khailany, Raghunathan | 2021 | https://arxiv.org/abs/2103.09301 | text | paper |
| 27 | Scaling Laws for Precision | Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, et al. | 2024 | https://arxiv.org/abs/2411.04330 | text | paper |
| 28 | Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens | Ouyang, Ge, Hartvigsen, Zhang, Mi, Yu | 2024 | https://arxiv.org/abs/2411.17691 | text (abstract and introduction) | paper |
| 29 | MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases | Liu, Zhao, Iandola, Lai, Tian, Fedorov, et al. | 2024 | https://arxiv.org/abs/2402.14905 | text | paper |
| 30 | Pareto-Optimal Quantized ResNet Is Mostly 4-bit | Abdolrashidi, Wang, Agrawal, Malmaud, Rybakov, Leichner, et al. | 2021 | https://arxiv.org/abs/2105.03536 | abstract | paper |
| 31 | Oscillation-free Quantization for Low-bit Vision Transformers | Liu, Liu, Cheng | 2023 | https://arxiv.org/abs/2302.02210 | abstract | paper |
| 32 | Quantized Neural Networks for Low-Precision Accumulation with Guaranteed Overflow Avoidance | Colbert, Pappalardo, Petri-Koenig | 2023 | https://arxiv.org/abs/2301.13376 | abstract | paper |
| 33 | A2Q+: Improving Accumulator-Aware Weight Quantization | Colbert, Pappalardo, Petri-Koenig, Umuroglu | 2024 | https://arxiv.org/abs/2401.10432 | abstract | paper |
| 34 | I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference | Li, Gu | 2022 | https://arxiv.org/abs/2207.01405 | abstract | paper |
| 35 | I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models | Hu, Cheng, Yang, Yuan, Yu, et al. | 2024 | https://arxiv.org/abs/2405.17849 | abstract | paper |
| 36 | Transformers without Normalization | Zhu, Chen, He, LeCun, Liu | 2025 | https://arxiv.org/abs/2503.10622 | abstract | paper |
| 37 | Theory, Analysis, and Best Practices for Sigmoid Self-Attention | Ramapuram, Danieli, Dhekane, Weers, Busbridge, et al. | 2024 | https://arxiv.org/abs/2409.04431 | abstract | paper |
| 38 | Replacing softmax with ReLU in Vision Transformers | Wortsman, Lee, Gilmer, Kornblith | 2023 | https://arxiv.org/abs/2309.08586 | abstract | paper |
| 39 | Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1 | Courbariaux, Hubara, Soudry, El-Yaniv, Bengio | 2016 | https://arxiv.org/abs/1602.02830 | text (shift-based normalisation algorithm) | paper |
| 40 | Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets | Yin, Lyu, Zhang, Osher, Qi, et al. | 2019 | https://arxiv.org/abs/1903.05662 | abstract | paper |
| 41 | Network Quantization with Element-wise Gradient Scaling | Lee, Kim, Ham | 2021 | https://arxiv.org/abs/2104.00903 | abstract | paper |
| 42 | Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks | Gong, Liu, Jiang, Li, Hu, et al. | 2019 | https://arxiv.org/abs/1908.05033 | abstract | paper |
| 43 | Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation | Bengio, Léonard, Courville | 2013 | https://arxiv.org/abs/1308.3432 | metadata | paper |
| 44 | Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights | Zhou, Yao, Guo, Xu, Chen | 2017 | https://arxiv.org/abs/1702.03044 | abstract | paper |
| 45 | Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy | Mishra, Marr | 2017 | https://arxiv.org/abs/1711.05852 | abstract | paper |
| 46 | QKD: Quantization-aware Knowledge Distillation | Kim, Bhalgat, Lee, Patel, Kwak | 2019 | https://arxiv.org/abs/1911.12491 | abstract | paper |
| 47 | KDLSQ-BERT: A Quantized Bert Combining Knowledge Distillation with Learned Step Size Quantization | Jin, Liang, Wu, Zou, Gan | 2021 | https://arxiv.org/abs/2101.05938 | abstract | paper |
| 48 | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | Sanh, Debut, Chaumond, Wolf | 2019 | https://arxiv.org/abs/1910.01108 | abstract | paper |
| 49 | DistilGPT2 model card | Hugging Face | not dated on the card | https://huggingface.co/distilbert/distilgpt2 | card | model: Apache-2.0 |
| 50 | TinyBERT: Distilling BERT for Natural Language Understanding | Jiao, Yin, Shang, Jiang, Chen, Li, et al. | 2019 | https://arxiv.org/abs/1909.10351 | abstract | paper |
| 51 | MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices | Sun, Yu, Song, Liu, Yang, Zhou | 2020 | https://arxiv.org/abs/2004.02984 | abstract | paper |
| 52 | Sequence-Level Knowledge Distillation | Kim, Rush | 2016 | https://arxiv.org/abs/1606.07947 | abstract | paper |
| 53 | Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty | Timiryasov, Tastet | 2023 | https://arxiv.org/abs/2308.02019 | text | code https://github.com/timinar/BabyLlama : MIT |
| 54 | BabyLlama-2: Ensemble-Distilled Models Consistently Outperform Teachers With Limited Data | Tastet, Timiryasov | 2024 | https://arxiv.org/abs/2409.17312 | abstract | paper |
| 55 | Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora | Warstadt, Mueller, Choshen, Wilcox, Zhuang, Ciro, et al. | 2023 | https://arxiv.org/abs/2504.08165 | text | paper |
| 56 | Findings of the Second BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora | Hu, Mueller, Ross, Williams, Linzen, Zhuang, et al. | 2024 | https://arxiv.org/abs/2412.05149 | text | paper |
| 57 | TinyStories: How Small Can Language Models Be and Still Speak Coherent English? | Eldan, Li | 2023 | https://arxiv.org/abs/2305.07759 | abstract | dataset licence not checked here |
| 58 | Well-Read Students Learn Better: On the Importance of Pre-training Compact Models | Turc, Chang, Lee, Toutanova | 2019 | https://arxiv.org/abs/1908.08962 | abstract | paper |
| 59 | Improved Knowledge Distillation via Teacher Assistant | Mirzadeh, Farajtabar, Li, Levine, Matsukawa, Ghasemzadeh | 2019 | https://arxiv.org/abs/1902.03393 | abstract | paper |
| 60 | Towards the Law of Capacity Gap in Distilling Language Models | Zhang, Li, Song, Ye, Gao, Hu | 2023 | https://arxiv.org/abs/2311.07052 | abstract | paper |
| 61 | Compressing Pre-trained Models of Code into 3 MB | Shi, Yang, Xu, Kang, Lo | 2022 | https://arxiv.org/abs/2208.07120 | abstract | paper |
| 62 | Deduplicating Training Data Makes Language Models Better | Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, et al. | 2021 | https://arxiv.org/abs/2107.06499 | abstract | code https://github.com/google-research/deduplicate-text-datasets : Apache-2.0, archived |
| 63 | The Adverse Effects of Code Duplication in Machine Learning Models of Code | Allamanis | 2018 | https://arxiv.org/abs/1812.06469 | abstract | paper |
| 64 | Curriculum learning for language modeling | Campos | 2021 | https://arxiv.org/abs/2108.02170 | abstract | paper |
| 65 | Textbooks Are All You Need | Gunasekar, Zhang, Aneja, Mendes, Del Giorno, et al. | 2023 | https://arxiv.org/abs/2306.11644 | abstract | paper |
| 66 | EmbBERT: Attention Under 2 MB Memory | Bravin, Pavan, Shalby, Pittorino, Roveri | 2025 | https://arxiv.org/abs/2502.10001 | abstract | paper |
| 67 | A Tiny Transformer for Low-Power Arrhythmia Classification on Microcontrollers | Busia, Scrugli, Jung, Benini, et al. | 2024 | https://arxiv.org/abs/2402.10748 | abstract | paper |
| 68 | MCUBERT: Memory-Efficient BERT Inference on Commodity Microcontrollers | Yang, Chen, Wu, Wong, Liang, et al. | 2024 | https://arxiv.org/abs/2410.17957 | abstract | paper |
| 69 | llama2.c (README, `runq.c` section, model table) | Karpathy | 2023 to 2024 | https://github.com/karpathy/llama2.c | card | code: MIT |
| 70 | esp32-ai-tinystories (model card) | slvDev | 2026 | https://huggingface.co/slvDev/esp32-ai-tinystories | card | model: MIT; code https://github.com/slvDev/esp32-ai : MIT |
| 71 | TinyStories 1-bit LLM (model card) | euclidstellar | 2026 | https://huggingface.co/euclidstellar/tinystories-1bit-llm | card | model: MIT |
| 72 | eeny-tinystories-999k (model card) | sprapp | 2026 | https://huggingface.co/sprapp/eeny-tinystories-999k | card | model: Apache-2.0 |
| 73 | Deeploy: Enabling Energy-Efficient Deployment of Small Language Models On Heterogeneous Microcontrollers (added by the fact-check) | Scherer, Macan, Jung, Wiese, Bompani, Burrello, Conti, Benini | 2024 | https://arxiv.org/abs/2408.04413 | text (sections IV and VI) | code https://github.com/pulp-platform/Deeploy : Apache-2.0 for most sources (README; its LICENSES folder also holds MIT and CC-BY-ND-4.0); model https://huggingface.co/Maykeye/TinyLLama-v0 : Apache-2.0 (card metadata) |
| 74 | Z80-uLM (README; added by the fact-check) | HarryR | 2025 to 2026 | https://github.com/HarryR/z80ai | card | code: "MIT or Apache-2.0 as you see fit" (README; the GitHub API detects no licence file) |
| 75 | TinyStories-1M (model card and config.json; added by the fact-check) | Eldan | 2023 | https://huggingface.co/roneneldan/TinyStories-1M | card, config | model: no licence in the card metadata |

## Verification log

Fact-check of 2026-11-13 by a second agent: 25 claims checked (one line groups the numbers taken from one source
or one kind of source), 16 confirmed as written, 8 corrected, 1 unsupported and tagged. No invented citation and
no invented number was found; nothing had to be deleted. Paper PDFs were read through pdftotext, abstracts and
metadata through the arXiv API, model cards and configs through the hub API and raw files, licences through the
GitHub API and licence files; section 2J was re-run with the project virtualenv. Format: claim: verdict. URL.

- 1. All 67 arXiv identifiers cited by the research pass resolve to the stated title and authors: confirmed. https://export.arxiv.org/api/query (id_list of the 67 identifiers)
- 2. Prato et al. language model (243.04 MB; WikiText-2 5.65 -> 5.64, PPL 284.15 -> 282.67; WikiText-103 5.91 -> 5.94, PPL 369.20 -> 377.79), BLEU rows at 8, 6 and 4 bits, list of quantized tensors, exp replaced by a step function: confirmed; caveat added (2-layer example model whose 4-bit version scores the same). https://arxiv.org/abs/1910.10485
- 3. Prato et al. start step (Table 5: 26.46 / 26.98 / 26.63 / 26.84 / 26.41 / 25.50): values confirmed; corrected "no trend with the start step at 8 bits" -> the authors conclude "quantizing the model early on seems preferable", on differences of at most 0.57 BLEU (sections 2D, 3, 6.1). https://arxiv.org/abs/1910.10485
- 4. DiffQ Table 2 (18.1 at 942 MB; 18.3; 19.7; LSQ 18.9; DiffQ 18.1 at 130 MB): values confirmed; corrected "plain straight-through baseline" -> only the weights are trained with QAT and the activations are quantized after training; QAT weights with float activations give 18.2, +0.006 nats, in Table B.6 (sections 1, 2A, 3, 6.2). https://arxiv.org/abs/2104.09987
- 5. Quant-Noise Table 1 (18.3; 19.6; 21.0; 18.7; int4 39.4 / 34.1 / 21.8) and the rate ablation 80.66 to 80.92 to 80.64: confirmed; clarification added (QAT acts on the weights, activation ranges are calibrated before evaluation). https://arxiv.org/abs/2004.07320
- 6. Component sensitivity: Bondarenko et al. GLUE averages (83.06, 83.26, 71.03, 82.64, 82.29, 83.23, 70.25, 82.43, 82.45) and the leave-one-out rows; FQ-ViT 40x the median, 0 to 0.01, 4-bit attention maps: confirmed. https://arxiv.org/abs/2109.12948 and https://arxiv.org/abs/2111.13824
- 7. 8-bit BERT-family results: I-BERT 86.0 -> 86.3 with nonlinearities on INT32 values; Q8BERT CoLA, RTE and SST-2 rows; TernaryBERT 8-8-8 rows and its Table 6 and Table 7 rows: confirmed. https://arxiv.org/abs/2101.01321 , https://arxiv.org/abs/1910.06188 , https://arxiv.org/abs/2009.12812
- 8. Large-model sources: LLM-QAT Table 1 (66.2; 65.6 / 66.3 / 66.0; 47.4 / 61.6; 41.2 / 60.7) and its settings; MobileLLM W8A8 45.0 -> 44.8, distillation slowdown 2.6 to 3.2x and quotation: confirmed. https://arxiv.org/abs/2305.17888 and https://arxiv.org/abs/2402.14905
- 9. Planning assumption "+0.02 to +0.05 nats in total for LS-LLM": unsupported (no source measures a loss at or below 1 M parameters, or with static power-of-two scales and int8-indexed tables); kept as an assumption and tagged [UNVERIFIED] in sections 1, 5 and 6.2. No URL.
- 10. Range and scale papers: TQT accuracies (71.1 -> 71.1, 0.6, 67.0, 2.1 epochs; 71.7 -> 71.8, 0.3, 68.2; 80.2 -> 79.4) and settings; Google QAT 70.7 / 70.0 / 70.9, now read in Table 4 of the original whitepaper (the old [UNVERIFIED] tag is resolved); white-paper quotation; Jacob et al. delay of 50 thousand to 2 million steps and statement on small models: confirmed. https://arxiv.org/abs/1903.08066 , https://arxiv.org/abs/1806.08342 , https://arxiv.org/abs/2106.08295 , https://arxiv.org/abs/1712.05877
- 11. LSQ (67.9 / 70.6 / 71.2 / 71.1 against 70.5; 1 epoch at 0.001 for 8 bits; weight decay; 8-bit input and output layers) and LSQ+ (4.1%, 1.8%, 5.6%, 5 runs): confirmed. https://arxiv.org/abs/1902.08153 and https://arxiv.org/abs/2004.09576
- 12. Nagel et al. oscillation numbers (65.3 / 67.8 / 67.6; 70.5 / 70.6; 52.0 -> 59.0 / 58.9; 63.7 / 63.6; 65.1; 70.33): confirmed; corrected "the reported experiments are at 3 and 4 bits" and the section 4 entry that listed oscillation measurements at 8 bits as not found -> Table 2 of the paper has an 8-bit MobileNetV2 row, 71.79 -> 71.89, 0.10 points (sections 1, 2E, 3, 4, 5, 6.1). https://arxiv.org/abs/2203.11086
- 13. Model card esp32-ai-tinystories exists; 28,867,968 parameters, 556,416 in the dense core, int4 in groups of 128, +0.0003 nats (2.4793 -> 2.4796), MIT: confirmed (self-reported). https://huggingface.co/slvDev/esp32-ai-tinystories
- 14. Model card tinystories-1bit-llm exists; 11,159,360 parameters, 2.1789 -> 2.1795, +0.1347, 2.0553 / 2.1760 / 5.0229 / 2.3607, 1.79 tokens per parameter, MIT: confirmed (self-reported). https://huggingface.co/euclidstellar/tinystories-1bit-llm
- 15. Model card eeny-tinystories-999k exists; 999,328 parameters, 0.625 against 0.707 bits per byte, distillation, int8 per row, Apache-2.0: confirmed (self-reported); corrected the size of the comparison model, "about 3 M total, 1 M non-embedding" (card wording, was [UNVERIFIED]) -> about 3.75 M total and about 0.40 M outside the embedding tables, computed from the TinyStories-1M config. https://huggingface.co/sprapp/eeny-tinystories-999k and https://huggingface.co/roneneldan/TinyStories-1M
- 16. llama2.c README: both quotations, int8 for matmul weights with dynamic activation quantization, stories260K validation loss 1.297, MIT: confirmed. https://github.com/karpathy/llama2.c/blob/master/README.md
- 17. Licences in section 7 (graffitist BSD 3-Clause; oscillations-qat and transformer-quantization BSD-3-Clause-Clear; diffq and LLM-QAT CC BY-NC 4.0; bitlinear, I-BERT, BabyLlama, llama2.c, esp32-ai MIT; FQ-ViT and deduplicate-text-datasets Apache-2.0; both archived flags; the three model cards; DistilGPT2 Apache-2.0): confirmed. https://api.github.com/repos/OWNER/REPO and /license for each repository, hub API for the cards
- 18. Kumar et al. (465 runs, 30 M to 220 M non-embedding parameters, 3 to 16 bits, N_eff form, 7 to 8 bits, more than 4x below 4 bits, about 1000 tokens per parameter, gw = 2.8111, warning about the constants): confirmed; integer-type constants added to 2J. https://arxiv.org/abs/2411.04330
- 19. Ouyang et al. (over 1500 checkpoints; smaller and longer-trained models lose more): confirmed; corrected "(the appendix names the Pythia suite)" -> Pythia is the subject of section 3.1 of the main text, 160 M to 12 B parameters, 2 to 4 bits only. https://arxiv.org/abs/2411.17691
- 20. Low-bit sources: BitNet b1.58 Reloaded (6 M to 48 M, 4 layers, hidden 32 to 256, about double the hidden size, vision models of 100k to 2.2 M, learning rates, weight decay), BitNet embedding quotation, 2B4T weight decay 0.1 then 0: confirmed; corrected the "not recommended" note of section 5, which applied "about twice the hidden size" to 4-bit weights too -> ternary only. https://arxiv.org/abs/2407.09527 , https://arxiv.org/abs/2310.11453 , https://arxiv.org/abs/2504.12285
- 21. Distillation and BabyLM sources: Baby Llama (58 M student, 705 M + 360 M teachers, 0.5 / 0.5, temperature 2, 6 epochs, 3e-4, two teachers against four or six), three BabyLM 2023 quotations, BabyLM 2024 statement on FLOPs, DistilGPT2 82 M / 124 M / 21.1 / 16.3: confirmed. https://arxiv.org/abs/2308.02019 , https://arxiv.org/abs/2504.08165 , https://arxiv.org/abs/2412.05149 , https://huggingface.co/distilbert/distilgpt2
- 22. Softermax (Q(6,2) input, Q(1,7) output, intermediates Q(1,15) / Q(10,6) / Q(1,7); 86.28 -> 85.86, 83.27 -> 83.80, 89.40 -> 89.46; worst drop under 0.5%): confirmed. https://arxiv.org/abs/2103.09301
- 23. Abstract-level statements, re-read through the arXiv API: A2Q 99.2% / 98.2%; phi-1 50.6% / 55.5% (both pass@1) and 45%; TinyStories; deduplication; code duplication up to 100%; Kim and Rush 4.2 / 1.7 BLEU and 10x; both capacity-gap papers; BabyLlama-2 345 M; 16-bit then 1.58-bit; QKD quotation; clipped softmax; I-LLM; Liu et al.; Abdolrashidi et al.; Shi et al. 3 MB / 160x / 4.31x / 4.15x; EmbBERT 2 MB / 781 kB; Busia et al. 6k / 98.97%; MCUBERT 256 KB; Campos; tanh in place of normalisation, sigmoid attention, ReLU attention: confirmed. https://export.arxiv.org/api/query
- 24. Own arithmetic of 2J, re-run: 511-product bound and the limit 516, float32 exactness in numpy and in torch 2.15.0 on MPS, floor against rounding (0.333 -> 0.083 LSB^2), 0.942 and 0.836: confirmed; corrected the Gaussian quantizer numbers 40.45 dB -> 40.54 dB, 1.53 dB -> 1.62 dB, 3.77 dB -> 3.90 dB, 0.25 / 0.63 bit -> 0.27 / 0.65 bit (clip point 3.92 confirmed; 0.25 -> 0.27 also in section 3). No URL (computed here).
- 25. "Not found" claims of sections 1 and 4 (integer-only language model near our scale; "TQT is CNN only"): corrected -> Deeploy runs a TinyStories Llama of 4,621,392 parameters (about 0.53 M outside the embedding tables) with 8-bit integer inference after post-training quantization with TQT and reports no loss; Z80-uLM added as a second near item. The narrower statements (no loss measured at or below 1 M parameters; no static power-of-two scales together with int8-indexed tables) still stand as "not found in our search". https://arxiv.org/abs/2408.04413 , https://huggingface.co/Maykeye/TinyLLama-v0 , https://github.com/HarryR/z80ai

Not re-checked by the fact-check: the shift-based BatchNorm source (1602.02830), BinaryBERT, the DistilBERT,
TinyBERT, MobileBERT and Turc et al. abstracts, the abstract-level rows of table 6.1 (EWGS, DSQ, DoReFa, PACT,
incremental network quantization, Apprentice, KDLSQ-BERT), the cycle estimates taken from `research/CONTEXT.md`,
and the design notes at the end of section 5 beyond their arithmetic. Still open from the research pass: none of
the sources of section 7 has a row in SOURCES.md yet.
