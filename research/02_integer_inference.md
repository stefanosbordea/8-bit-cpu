# 02 Integer-only and int8 transformer inference, power-of-two scaling, lookup-table nonlinearities

Research date: 2026-11-13. Numbers tagged **[computed here]** come from
`research/02_integer_inference_checks.py` (numpy, seeded, a few seconds). Cycle counts tagged **[estimate]** use the
cost model in `research/CONTEXT.md` (a row of length n costs n + 24 cycles, an element-wise operation 12 to 32 cycles)
and the baseline shape (5 layers, dim 64, 8 heads, head dim 8, T = 256). Every other number carries the URL it was
read from. "Abstract only" means the body of the paper was not opened.

Fact-check pass (2026-11-13): **[VERIFIED]** marks a statement that was compared again with its source,
**[UNVERIFIED: reason]** marks one that could not be supported, and statements without a tag were not re-checked.
Corrections are listed in the Verification log at the end of the file.

## 1. Answer in one paragraph

Integer-only transformer inference at 8-bit weights and 8-bit matrix inputs is well established, but every published
pipeline we opened keeps something wider than int8 or performs an integer division somewhere: I-BERT computes
softmax, GELU and LayerNorm in INT32 and keeps a 22-bit residual sum; I-ViT keeps a 16-bit residual stream and still
uses integer divisions in its softmax (one per row) and its LayerNorm; the TFLite and CMSIS-NN int8 softmax kernels
use 32-bit fixed point with count-leading-zeros and a Newton reciprocal; Deeploy's dim-64 TinyStories Llama on a
microcontroller uses 32-bit divisions in RMSNorm [VERIFIED]. In the papers we found no language model that runs with
an int8 residual stream, shift-only requantisation and 256-entry tables only. Three hobby projects from 2026, missed
by the research pass of this file and documented in `research/04_tiny_hardware.md`, come closer (section 4): a MAX 10
design keeps an int8 residual stream through 16 layers and goes from 0.622 to 0.643 bits per character for all its
integer conversions together; an ECP5 design runs stories260K with int8 values, power-of-two scales and four
256-entry tables but reports no loss; an NES model with ternary weights is trained with the integer forward pass in
the loop. None of them is the LS scheme (the MAX 10 design has integer divisions, 15-bit softmax weights and
multiply-then-shift rescales; the ECP5 tables have 16-bit entries; the NES model has ternary weights and 4-bit
activations) [VERIFIED], so the exact LS arithmetic is untested, but each part has support. (1) Softmax: an 8-bit exp
table over a window of 5.5 to 7.7 nats was enough for a 1B-parameter model without retraining (IntAttention:
Llama-3.2-1B perplexity 12.66 to 12.78 with a 32-entry uint8 table, uint8 probabilities and an exact row
normalisation) [VERIFIED]; divide the value-mixed output rather than each weight (the Integer Transformer does
this), and replace the division by a reciprocal table on a normalised mantissa (the TFLite int16 softmax does this)
[VERIFIED]; the error is below 1% with an 8-bit mantissa (128 table entries), 2.6% rms with a 4-bit mantissa (8
entries) and, for a bare power of two, 29% rms, or 19% rms once the constant 1.337 is folded into the next weights
[computed here].
Dropping normalisation altogether (ConSmax, sigmoid attention) matched softmax in float at 6-layer-GPT and
1B-parameter scale [VERIFIED] and is untested below that. (2) Norm: a 1/sqrt table needs the roughly 20-bit sum of
squares reduced by leading-one detection with an even shift (error at most 0.9%); one fixed shift is accurate to
1.5% only over a 4x range of RMS, two fixed shifts cover 16x [computed here]. Dynamic Tanh replaces the whole norm by
one table lookup per element and matched RMSNorm on LLaMA 7B to 70B [VERIFIED], but is unverified for small models
and one preprint reports seed-dependent collapse with SwiGLU [VERIFIED]. (3) Residual stream: I-BERT, I-ViT and
HAWQ-V3 keep the residual sum at 16 to 32 bits; 8-bit sums appear with per-channel shifts (FQ-ViT), run-time scales
(Integer Transformer), quantisation-aware training (Bondarenko et al.) and in the MAX 10 project [VERIFIED]; an int8
stream after 10 additions has about 30 dB signal-to-noise, 1.7 bits worse than a single 8-bit rounding [computed
here], which is plausible with quantisation-aware training but must be measured. (4) Any int8 dot product of up to
511 terms cannot overflow 24 bits; longer rows need the per-row check sum abs(w) <= 65,535. (5) Floor shifts bias
every output by -0.5 LSB; preloading the accumulator with 2^(s-1), as the legacy CMSIS-NN kernels do [VERIFIED],
removes the bias, at no cycle cost if the accumulator can be loaded with a constant instead of cleared. (6) Tables
are trained through by running the exact integer operation in the forward pass with a straight-through gradient
(I-ViT, Softermax); with 8-bit quantisation-aware fine-tuning the reported average scores ended between 0.2 points
below and 0.5 points above float (I-BERT +0.3 and +0.5 GLUE, I-ViT -0.19 to +0.27 top-1, Bondarenko et al. +0.20
GLUE) [VERIFIED; the earlier wording was "within about 0.3 points"], always starting from a trained float model; in
I-ViT's own comparison the I-BERT operators lose 0.74 to 1.34 points on the same vision transformers [VERIFIED].

## 2. Proven (numbers and sources)

**Integer-only pipelines and their bit widths**
- Jacob et al. 2018: uint8 weights and activations, int32 accumulator and bias, requantisation by a fixed-point
  multiplier M0 in [0.5, 1) plus a rounding right shift. ImageNet ResNet-50/100/150: 76.4/78.0/78.8% float,
  74.9/76.6/76.7% integer. Weights never take -128, so each product is below 2^14 in magnitude.
  https://arxiv.org/abs/1712.05877 [VERIFIED]
- HAWQ-V3: scales are dyadic numbers b/2^c (integer multiply then shift, no integer division anywhere); the residual
  addition is done in INT32. ResNet50 77.72% float, 77.58% W8A8. https://arxiv.org/abs/2011.10680 [VERIFIED]
- I-BERT: MatMul and Embedding in INT8, non-linear operations in INT32. i-exp (second-order polynomial plus shift)
  has a largest gap of 1.9e-3 to exp. Softmax ends in an integer division `qexp / sum(qexp)`. Integer square root by
  Newton iteration converges within four iterations for any INT32 input, each with one integer division. GLUE average
  RoBERTa-Base 86.0 to 86.3, RoBERTa-Large 89.0 to 89.5 after quantisation-aware fine-tuning.
  https://arxiv.org/abs/2101.01321 . In the Hugging Face implementation: `act_bit = 8`, `ln_input_bit = 22`
  (the residual sum entering LayerNorm), `ln_output_bit = 32`, `embedding_act_bit = 16`, `bias_bit = 32`.
  https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/ibert/modeling_ibert.py
  [VERIFIED]
- I-ViT: Shiftmax computes e^x as 2^(x*log2 e) with log2 e = (1.0111)b, i.e. `x + (x>>1) - (x>>4)`, and
  2^r = 1 + r/2 on (-1, 0]; ShiftGELU uses 1.702 = (1.1011)b; I-LayerNorm uses 10 fixed integer Newton iterations.
  DeiT-S 79.85% float, 80.12% INT8; ViT-S 81.39 to 81.27; Swin-S 83.20 to 83.01. Replacing Shiftmax by the I-BERT
  polynomial costs 0.12 (DeiT-B) and 0.22 (Swin-S); replacing I-LayerNorm by L1 LayerNorm costs 2.49 and 3.32 points.
  Training is quantisation-aware fine-tuning with a plain straight-through estimator.
  https://arxiv.org/abs/2207.01405 . In the code both residual additions are `QuantAct(16)`, norm outputs are
  `QuantAct()` (8 bit), the softmax is `IntSoftmax(16)`. Integer divisions remain: Shiftmax has one per row,
  `factor = floor((2**31-1) / exp_int_sum)`, and divides each element by the constant `x0_int = floor(-1/scale)` to
  split the exponent; I-LayerNorm has one per row, `factor = floor((2**31-1) / std_int)`, plus one per square-root
  iteration (`var_int / k`).
  https://github.com/zkkli/I-ViT/blob/main/models/vit_quant.py ,
  https://github.com/zkkli/I-ViT/blob/main/models/quantization_utils/quant_modules.py [VERIFIED]
- FQ-ViT (post-training): LayerNorm inputs have channel ranges that exceed 40x the median. A per-channel
  Power-of-Two Factor (a left shift of 0 to K = 3 bits per channel) lifts fully quantised ViT-B from 23.64% to 83.31%
  (float 84.53%). Log-Int-Softmax stores attention as 4-bit log2 values and replaces the multiply by a shift:
  82.68%. DeiT-S: 79.85 float, 79.17 at 8/8/8, 78.40 with 4-bit attention. https://arxiv.org/abs/2111.13824 [VERIFIED]
- I-LLM (post-training, LLaMA family): static I-BERT-style W8A8 "can lead to catastrophic outcomes" on LLMs. Their
  fix is a per-token dynamic dyadic scale m/2^k (m and k 8-bit, k found from the most significant bit), a softmax
  input clipped to a window c, a shift-only exponent with linear interpolation, and a bitwise integer square root.
  LLaMA-7B WikiText2 perplexity: 5.68 FP16, 5.84 W6A6, 9.10 W4A4. Window ablation at W4A4: c = 20: 9.15, c = 15: 9.16,
  c = 10: 9.23, no clipping: 7,360,945. https://arxiv.org/abs/2405.17849 [VERIFIED]
- Deeploy: a Llama2-style TinyStories model with hidden size 64, 16 heads, 8 layers, FFN size 256
  (https://huggingface.co/Maykeye/TinyLLama-v0 ; 4,621,392 parameters in the model card metadata, about 4.1M of
  them in the untied 32,000-token embedding and output matrices by our arithmetic from its config.json; the paper
  lists 512 kB of linear-layer weights for the 8 layers) runs integer-only on the Siracusa microcontroller at 340
  tokens/s and 490 uJ/token, averaged over 256 autoregressive steps. Matmul weights and activations are 8-bit (TQT,
  post-training); softmax uses the I-BERT approximation; SiLU is replaced by Hardswish; "all divisions in RMSNorm"
  use 32-bit numerators and denominators. No perplexity or accuracy number for the quantised model appears in the
  text. https://arxiv.org/abs/2408.04413 [VERIFIED; parameter count added in the fact-check]

**Softmax kernels in deployed libraries**
- TFLite reference int8/uint8 softmax: exp input in Q5.26, sum in Q12.19, reciprocal by `GetReciprocal`
  (count leading zeros, then a Newton 1/(1+x) on the normalised mantissa), multiply, rounding shift; inputs more than
  `diff_min` below the maximum are skipped. The int16 variant uses two tables: exp over [-10, 0] and 1/(1+x) on the
  mantissa after `CountLeadingZeros`, then a right shift by `31 - headroom_plus_one`.
  https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/lite/kernels/internal/reference/softmax.h
  [VERIFIED]
- TFLite tables: "For uint8 and int8 inputs a simple 256 entries LUT is used"; int16 uses 513 entries with 9 index
  bits and 7 interpolation bits.
  https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/lite/kernels/internal/common.h
  [VERIFIED; quote corrected to the wording of the source comment]
  The optimised uint8 softmax indexes a 256-entry exp table by `input + (255 - max)`; a 16-bit exp value is stored as
  two byte tables (`uint8_table1`, `uint8_table2`).
  https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/lite/kernels/internal/optimized/optimized_ops.h
- CMSIS-NN `arm_softmax_s8`: same scheme in 32-bit arithmetic (`CLZ`, `ONE_OVER1`, `MUL_SAT`, `DIV_POW2`).
  https://github.com/ARM-software/CMSIS-NN/blob/v6.0.0/Source/SoftmaxFunctions/arm_softmax_s8.c [VERIFIED]
- Legacy CMSIS-NN `arm_softmax_q7`: base-2 softmax, y_i = 2^x_i / sum 2^x_j, one octave per input LSB, window of 8
  octaves below the maximum, exponent and normalisation by shifts, one integer division `(1 << 20) / sum` per vector.
  https://github.com/ARM-software/CMSIS_5/blob/5.9.0/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_q7.c [VERIFIED]

**Softmax approximations with reported accuracy**
- Softermax: base 2, 8-bit input Q(6,2) (4 steps per octave, so the fractional power needs a 4-entry table), online
  integer maximum, 16-bit internals, reciprocal by a piecewise-linear unit and an integer multiplier, 8-bit output.
  BERT-Base SQuAD 86.28 to 85.86 (the 86.28 baseline is itself 8-bit quantised, with the standard softmax), worst
  drop under 0.5%, after Softermax-aware fine-tuning with straight-through gradients. The host accelerator uses
  8-bit weights and activations with a 24-bit accumulator. https://arxiv.org/abs/2103.09301 [VERIFIED]
- ITA: softmax directly on int8 scores with scale 1/32 octave per LSB; only the top 3 bits of (max - x) are used as a
  shift, so weights take 8 power-of-two levels; the denominator is accumulated in 15 bits and inverted once per row
  by a serial divider. Mean absolute error of the probabilities 0.46% (I-BERT softmax 0.35%). The accumulator is 24
  bits "to allow up to 256-element dot products". https://arxiv.org/abs/2307.03493 [VERIFIED]
- SOLE (no retraining): E2Softmax quantises exp outputs to 4-bit log2 values and divides by leading-one detection
  plus one mantissa bit, 2^-(ky+ks+1) * (1.636 - q(s)), i.e. factors 0.818 or 0.568. AILayerNorm compresses 8-bit
  inputs to 4 bits for the squares (16-entry table) and reads x^-0.5 from a table. DeiT-T 72.21 to 71.33, DeiT-S 79.85
  to 79.27, BERT-Base SQuAD 88.17 to 87.57. https://arxiv.org/abs/2510.17189 [VERIFIED]
- Vasyltsov and Chang (no retraining): table-only softmax (exp table plus a reciprocal table or a 2D table, at most
  about 700 bytes). Transformer-base WMT14 BLEU: 26.98 float, 26.76 (2D table) and 26.80 (reciprocal table) at uint8,
  26.26 and 26.68 at uint4, 24.42 and 25.29 at uint2. https://arxiv.org/abs/2111.10770 [VERIFIED]
- IntAttention (no retraining): row maximum subtracted in integers, difference clipped at c = 6.6 nats, 32-entry
  uint8 exp table, integer row sum, P = round(255 * E / rowSum(E)) as uint8. Llama-3.2-1B WikiText perplexity: 12.663
  FP16, 12.784 with the table softmax alone, 13.070 for the whole integer pipeline (13.701 for "Quant-Only": int8
  Q/K/V with a float softmax). Stable for at least 16 entries and c in [5.5, 7.7]. On Llama-3.2-1B-Instruct, FP16 /
  Quant-Only / whole integer pipeline give HumanEval 32.93 / 32.32 / 31.10 and MBPP 33.00 / 31.40 / 34.20 (pass
  rates from lm-evaluation-harness; the paper does not state k); the table softmax alone gives 31.10 and 34.20, the
  same two values as the whole pipeline. The authors count HumanEval among "tasks that are highly sensitive to
  probability distortion". Storing P as signed int8 instead of uint8 raises its relative L1 error from 0.041 to
  0.077 (their Table 9), which matters for LS weights limited to 0..127. https://arxiv.org/abs/2511.21513
  [VERIFIED; corrected: the three HumanEval and MBPP values were attributed to the wrong settings]
- Kang and Wang 2026 (model not retrained, LLaMA2-7B, WikiText-2, W8A8): division-free softmax through ln(sum) from
  leading-one detection and small tables; truncating the mantissa of the sum to 4 bits gives a 3 to 6% scale
  variation, which they argue the following RMSNorm absorbs. Exp table: perplexity 5.4946 with their compensated
  two-table version (512 entries) against 5.4943 with a plain 2048-entry table (their Table 3; 5.4949 without
  compensation, and 5.4955 for the same configuration with 16-bit entries in their Table 2). Whole softmax with the
  log-domain division: 5.4971 against 5.4940 for quantisation alone (their Table 7). Reciprocal square root for
  RMSNorm as 2^(-k/2) * 1/sqrt(1+s): an exponent table (32 entries in the text, 34 x 21 bits in their Table 4; it
  also handles odd k) times a 256-entry mantissa table; perplexity 5.4956 (5.5010 for the two-table baseline they
  compare with). https://pmc.ncbi.nlm.nih.gov/articles/PMC12843610/
  [VERIFIED; corrected: exp-table perplexity 5.4955 -> 5.4946, exponent table 34 -> 32 entries in the text]
- NN-LUT: 16-entry piecewise-linear tables with learned breakpoints for GELU, exp, division and 1/sqrt. A table with
  equally spaced breakpoints for the LayerNorm 1/sqrt destroys RoBERTa (CoLA 62.1 to 4.6, MRPC 87.5 to 57.5) while
  learned breakpoints with a power-of-two input pre-scale keep it (60.4, 86.5). Calibrating the table entries on
  unlabeled data lifts the INT32 GLUE average from 84.1 to 85.1 (float 85.4). https://arxiv.org/abs/2112.02191 [VERIFIED]

**Attention without softmax normalisation**
- ConSmax: exp(s - beta) / gamma with learnable beta and gamma, merged into one constant at inference, so no maximum,
  no sum and no division. GPT with 6 layers, 6 heads, width 384, context 256 on WikiText-103: perplexity 2.3% higher
  early, less than 0.9% higher after 10K iterations, converged with softmax after about 20K.
  https://arxiv.org/abs/2402.10930 [VERIFIED]
- Sigmoid attention: element-wise sigmoid(score + b) with b = -log(n). 1B-parameter LLM, n = 2048: average 49.5
  against 49.4 for softmax; at n = 4096 the bias alone is not enough (48.2 against 49.2) and an extra norm on the
  attention output is needed (50.2 against 49.4). Language modelling is sensitive to the initial attention norm; QK
  norm is needed for learning-rate robustness; ReLU "pose instabilities and underperform" on sequence tasks.
  https://arxiv.org/abs/2409.04431 [VERIFIED]
- ReLU attention divided by sequence length approaches or matches softmax on ViT S/32 to L/16 with qk-layernorm;
  vision only. https://arxiv.org/abs/2309.08586
- Integer Transformer (Lin et al.): softmax replaced by [ReLU(x + b)]^n and LayerNorm by an L1 version; the value
  mix is computed before the division by the sum of weights. In float this loses nothing (WikiText-103 test
  perplexity 31.28 to 31.20 with polynomial attention, 31.18 with L1 LayerNorm); the full INT8 model reaches 31.61
  against a 31.18 baseline (the paper's float baseline is 31.28 in its ablation table and 31.18 in its result
  table; its integer design evaluated in float reaches 30.79) and loses about 0.3 BLEU on translation, more than 1
  BLEU on En-Fr, attributed to the last residual connection and LayerNorm. n = 1 is clearly worse than n > 1.
  https://arxiv.org/abs/2009.08034 [VERIFIED]

**Normalisation variants**
- Dynamic Tanh: gamma * tanh(alpha * x) + beta instead of LayerNorm or RMSNorm. LLaMA training loss 1.59 to 1.60 (7B),
  1.53 to 1.54 (13B), 1.50 (34B) and 1.45 (70B) unchanged; ViT-B 82.3 to 82.5. LLMs need a learnable scalar after the
  embedding and a width-dependent initial alpha (width 1024: 1.0/1.0, 4096: 0.8/0.2, 8192: 0.2/0.05). Hardtanh
  instead of tanh: ViT-B 82.2 against 82.5. Replacing BatchNorm in ResNet-50 fails (76.2 to 68.9).
  https://arxiv.org/abs/2503.10622 [VERIFIED]
- MobileBERT NoNorm: LayerNorm replaced by a per-channel affine gamma*h + beta (foldable into the next layer). GLUE
  78.5 to 77.7 together with ReLU instead of GELU; latency 192 ms to 62 ms; trained with a teacher. TFLite 8-bit
  post-training quantisation then costs nothing (MNLI-m 83.9 to 83.9, SQuAD 90.0 to 90.0).
  https://arxiv.org/abs/2004.02984 [VERIFIED]
- SLAB (abstract only): LayerNorm is progressively replaced by a re-parameterised BatchNorm during training, which
  has no statistics at inference; "comparable performance and lower latency" on language modelling.
  https://arxiv.org/abs/2405.11582 [VERIFIED against the abstract]
- FlashNorm (abstract only): the RMSNorm weights fold exactly into the following linear layer.
  https://arxiv.org/abs/2407.09577 [VERIFIED against the abstract]
- HAAN: the logarithm of the inverse standard deviation is close to linear in the layer index in later layers, so
  some norm statistics can be predicted instead of computed; ten norm layers skipped in GPT-2 and inputs subsampled by
  half with at most 0.5% accuracy loss. https://arxiv.org/abs/2502.11832

**Power-of-two scales, rounding and accumulators**
- TQT: per-tensor, symmetric, power-of-two scales with trained thresholds match float on MobileNet v1 (71.1 to 71.1)
  and v2 (71.7 to 71.8) and are above Google's per-channel real-scale QAT (70.7, 71.1). Without training the same
  constraint gives 0.6% on MobileNet v1; training weights only gives 67.0. Rounding is round-half-to-even, which
  "prevents an overall upward or downward bias". https://arxiv.org/abs/1903.08066 [VERIFIED; quote corrected]
- CMSIS-NN (legacy q7 format): power-of-two scaling only, requantisation is a bit shift; CIFAR-10 CNN 80.3% float,
  79.9% 8-bit. Sigmoid and tanh by table, optionally two tables with different granularity.
  https://arxiv.org/abs/1801.06601 . The kernels start the accumulator at
  `(bias << bias_shift) + NN_ROUND(out_shift)` with `NN_ROUND(s) = (1 << s) >> 1`, then `__SSAT(sum >> out_shift, 8)`.
  https://github.com/ARM-software/CMSIS_5/blob/5.9.0/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7.c ,
  https://github.com/ARM-software/CMSIS_5/blob/5.9.0/CMSIS/NN/Include/arm_nnsupportfunctions.h [VERIFIED]
- Jacob et al., Appendix B: a rounding shift that breaks ties upward gives "an overall upward bias, which has been
  observed to cause significant loss of end-to-end accuracy"; bias vectors are kept at 32 bits because their error
  "tends to act as an overall bias". https://arxiv.org/abs/1712.05877 [VERIFIED]
- A2Q: a P-bit signed accumulator cannot overflow, including every partial sum, if sum|x_i||w_i| <= 2^(P-1) - 1,
  which gives the weight bound ||w||_1 <= (2^(P-1) - 1) * 2^(1signed(x) - N) for N-bit inputs.
  https://arxiv.org/abs/2308.13504 . For LLMs (abstract only): Llama3 8B with a 16-bit multi-stage accumulator keeps
  up to 98% of FP16 perplexity. https://arxiv.org/abs/2409.17092 [VERIFIED; the second paper against its abstract]
- Residual-stream precision. BERT-base GLUE: 83.06 float; 71.03 with 8-bit post-training quantisation; 83.23 when
  only weights are 8-bit; 82.43 with 16-bit on the FFN residual sum, FFN input/output and final output; 83.26 with
  8-bit quantisation-aware training. https://arxiv.org/abs/2109.12948 [VERIFIED]
- Clipped softmax (exact zeros in attention) cuts the largest activation in BERT from 735 to 21.5 and the W8A8
  post-training perplexity from 1294 to 4.52 (float 4.49). https://arxiv.org/abs/2306.12929 [VERIFIED]

**Our calculations [computed here]** [VERIFIED: the script was run again on 2026-11-13 and prints every number in
this list]
- Floor shift: mean error -0.4965 LSB at shift 7 (-0.5 + 2^-(s+1)); round-half-up: +0.0043 LSB; both with standard
  deviation 0.289 LSB. After 11 floor roundings an int8 residual stream carries a -5.50 LSB offset.
- 24-bit accumulator: at most 511 products with both operands in [-128, 127], 516 if weights avoid -128, 520 if both
  avoid -128. Sum of squares of 64 int8 values is at most 2^20. A 256-key value mix with weights 0..127 is at most
  4,161,536; with weights 0..255 it is 8,355,840, still below 2^23 - 1 = 8,388,607.
- Reciprocal of the softmax sum: bare power of two 29% rms error (ratio 0.50 to 1.00, mean 0.75), and 19% rms
  (ratio 0.67 to 1.34) after multiplying by the constant 1.337 that brings the mean ratio to 1;
  SOLE's one mantissa bit 10.3% rms (0.82 to 1.23); table on a 4-bit mantissa 2.6% rms; 8-bit mantissa with 7-bit
  entries 0.38% rms, 0.9% worst.
- 1/sqrt of a 20-bit sum of squares: leading-one detection with an even shift and a mantissa of 64..255: 0.85% worst
  with 7-bit entries, 0.56% with 8-bit entries; one fixed shift: 1.5% worst over a 16x range of the sum (4x of RMS),
  5.7% over 64x (8x of RMS).
- int8 residual stream after 10 branch additions (stream standard deviation 32 LSB): 30.4 dB against 40.9 dB for a
  single rounding; with 6 extra fraction bits (16-bit storage): 66.6 dB.

## 3. Uncertain or contested

- **Is an 8-bit residual stream enough?** Post-training results say no for BERT (71.03 against 83.06) but 8-bit QAT
  recovers fully (83.26) https://arxiv.org/abs/2109.12948 [VERIFIED]. I-BERT (22 bits), I-ViT (16 bits) and HAWQ-V3
  (INT32) keep the residual wide by design, and we found no ablation in those papers showing that 8 bits fails under
  QAT.
  The Integer Transformer blames its largest INT8 loss on the last residual connection and LayerNorm and reports
  that "adding a few bits can recover most of the performance" https://arxiv.org/abs/2009.08034 [VERIFIED]. All of
  this is on post-LN or vision models; a pre-LN stream that is never renormalised is a harder case and is not covered
  by these papers. One pre-norm data point from outside the papers: the MAX 10 TinyStories project (section 4) keeps
  the stream in int8 through 16 layers, rescaling stream and branch with a multiplier and a shift at each addition,
  and its whole post-training integer conversion costs 0.021 bits per character (0.622 to 0.643)
  https://github.com/MRazvan/fpga-tinystories [VERIFIED].
- **Dynamic Tanh outside the original paper.** A single-author preprint on GPT-2-family models from 64M parameters
  reports that DyT improves validation loss by 27.3% with 1M training tokens and worsens it by 18.8% with 118M, and
  that Llama-style models with DyT have a roughly 33% per-seed collapse mode localised to SwiGLU gating; runs are
  5K steps and compute-limited https://arxiv.org/abs/2604.23434 [VERIFIED]. Another preprint (abstract only) states
  that DyT "remain fragile at depth" https://arxiv.org/abs/2601.09719 [VERIFIED against the abstract]. The Derf
  follow-up claims erf(alpha x + s) beats LayerNorm, RMSNorm and DyT on vision, speech and DNA tasks (abstract only;
  language modelling is not in that list) https://arxiv.org/abs/2512.10938 [VERIFIED against the abstract], while a
  1B-parameter, 1000-step study (abstract only) finds Derf 0.31 nats behind RMSNorm under AdamW
  https://arxiv.org/abs/2604.01563 [VERIFIED against the abstract]. For LLaMA the optimal initial alpha depends on
  width, and the smallest width in that table of the original paper is 1024 (its other models use 0.5), so nothing
  tells us the value for a width-64 language model [VERIFIED; corrected: the earlier wording made 1024 the smallest
  width of the whole study, which also has ViT models down to width 384].
- **Clipped softmax and decoder models.** It helps BERT and ViT but hurts OPT-125m: W8A8 perplexity 21.18 for
  vanilla, 37.20 with clipped softmax, 16.02 with gated attention https://arxiv.org/abs/2306.12929 [VERIFIED]. An
  exp table with exact zeros behaves like a clipped softmax, so this could cut either way for a decoder.
- **L1 instead of L2 normalisation.** Harmless when trained in from scratch (31.28 to 31.18 perplexity)
  https://arxiv.org/abs/2009.08034 [VERIFIED], but 2.5 to 3.3 points worse when swapped into pretrained ViTs and
  fine-tuned https://arxiv.org/abs/2207.01405 [VERIFIED]. The difference is probably training from scratch against
  fine-tuning; not tested directly.
- **Per-tensor power-of-two scales.** Equal to float with trained thresholds on CNNs, collapse without training
  (0.6%) https://arxiv.org/abs/1903.08066 [VERIFIED]. FQ-ViT needs per-channel power-of-two factors on LayerNorm
  inputs in the post-training setting https://arxiv.org/abs/2111.13824 [VERIFIED]. Whether a from-scratch QAT
  transformer needs per-channel scales on the residual stream is open. Two hobby projects touch the question without
  settling it (section 4): the reference code of the ECP5 stories260K design (post-training, one power-of-two scale
  per tensor) notes that such a scale "loses up to 1 bit of precision" and that over 35 chained matmuls "the
  cumulative error reaches ~30-40% on the worst significant element"; Soul Player C64 trains about 25,000 int8
  parameters for per-tensor power-of-two shifts and reports no loss [VERIFIED].
- **Floor against round under bit-exact training.** Jacob et al. report the damage from a rounding bias in one
  sentence about inference [VERIFIED]; that it arose because inference differed from the training simulation is our
  reading [UNVERIFIED: the paper describes neither the experiment nor the training setup]. If training simulates the
  floor exactly, the model may absorb the bias; we found no test of this. Our baseline has no bias parameters, so the
  offset can only be absorbed by weights.
- **How much softmax precision attention needs.** No-retraining results tolerate 4-bit probabilities (26.98 to 26.26
  or 26.68 BLEU) https://arxiv.org/abs/2111.10770 [VERIFIED]. ITA uses 8 power-of-two levels but reports only the
  error of the probabilities, not a task score [VERIFIED; corrected: the earlier wording counted ITA as a result that
  tolerates this]. Our random-logit test is harsher: with 256 keys and Gaussian logits an exp table with outputs
  0..127 gives a total-variation distance of 1.3% (logit standard deviation 1 nat), 4.2% (2 nat) and 2.7% (4 nat)
  from the float softmax even with exact division, because mid-sized weights are only a few LSB; a two-byte exp value
  brings this to about 0.5% [computed here]. Real attention rows may be easier or harder than Gaussian logits. One
  trained data point: in the NES ternary model (3-bit attention weights, quantisation-aware training with the integer
  forward pass) an exact normaliser instead of a power-of-two shift improved validation loss from 1.4058 to 1.3766
  nats per character, two seeds per arm; the exact arm also changes one shift constant
  https://github.com/Scottcjn/elya-nes [VERIFIED].
- **Attention without normalisation in causal language models.** ConSmax has one experiment (6-layer GPT) and reports
  perplexity curves only. Sigmoid attention needs a bias tied to the sequence length plus stabilisers and still
  needed an extra norm at context 4096. During generation the number of keys grows from 1 to T, which a constant
  normaliser does not follow.
- **Polynomial against shift-based exponent.** I-ViT reports its Shiftmax ahead of the I-BERT polynomial by 0.12 to
  0.22 points and ShiftGELU ahead by about 0.9 [VERIFIED]; this is one group's comparison on vision models. For LS
  both are moot because a 256-entry table is exact up to output rounding.
- **IntAttention's pipeline loss depends on the model**: Qwen3-1.7B goes from 23.01 to 27.75 perplexity (32.95 with
  float softmax on int8 Q/K/V), so the int8 quantisation of Q, K and V costs more there than the table softmax
  (22.36 with the table softmax alone) https://arxiv.org/abs/2511.21513 [VERIFIED].

## 4. Not tried at our scale (about 260k to 400k int8 parameters, 8-bit multiply-accumulate CPU, 450 KB)

Corrected in the fact-check: the three items at the top of this list said "not found". Projects outside the papers,
listed in `research/04_tiny_hardware.md`, answer part of them; their READMEs and reference code were opened again for
this file. They are single-author repositories without independent reproduction (the MAX 10 repository was created on
2026-11-12).

- An integer-only language model below 1M parameters with reported accuracy: found outside the papers, with ternary
  weights. The NES project elya-nes streams 102,400 ternary weights per token, uses 4-bit activations and a
  64-symbol vocabulary, is trained on TinyStories "with quantisation-aware training so the forward pass the trainer
  sees is the forward pass the 6502 executes", and reports validation loss (dense model, two seeds: 1.4058 nats per
  character with a power-of-two softmax normaliser, 1.3766 with an exact one)
  https://github.com/Scottcjn/elya-nes [VERIFIED]. With int8 weights we found models without a loss figure only:
  Soul Player C64 (about 25,000 int8 parameters, per-tensor power-of-two shifts, Q8.8 activations in int16, a
  128-entry exp table, integer square root and division in RMSNorm, quantisation-aware training)
  https://github.com/gizmo64k/soulplayer-c64 [VERIFIED] and the ECP5 design below. In papers the closest is Deeploy
  (dim 64, 8 layers, 4.6M parameters including embeddings, 8-bit matmuls, 32-bit softmax and norm divisions, no
  accuracy number) https://arxiv.org/abs/2408.04413 [VERIFIED]. llama2.c64 runs the float32 `stories260K` weights on
  a Commodore 64 with a 2 MB RAM expansion at about one token every 8 minutes, so it is not an integer model
  https://github.com/ytmytm/llama2.c64 [VERIFIED].
- A language model with an int8 residual stream: not found in papers (the Integer Transformer is post-LN, uses
  run-time scales and integer division), found in one project. The MAX 10 TinyStories design (16 layers, width 128,
  4.72M parameters; ternary matrices trained from scratch with a straight-through estimator; activations converted
  to int8 after training from calibration ranges) keeps the stream in int8 and combines it with each branch as
  `x' = clamp((x*A + acc*B + 2^(S-1)) >> S, -127, 127)`. On 1,186 held-out TinyStories stories it reports 0.622
  bits per character for the float model and 0.643 for the integer model. It still uses integer divisions (softmax
  reciprocal, RMSNorm), Q15 softmax weights and multiply-then-shift rescaling.
  https://github.com/MRazvan/fpga-tinystories ,
  https://github.com/MRazvan/fpga-tinystories/blob/main/ref/int_model.py [VERIFIED]. An int8 stream at width 64 with
  int8 weights, trained with quantisation-aware training of the activations, was not found.
- Shift-only requantisation (all scales powers of two) in a transformer language model with accuracy numbers: not
  found. Evidence with accuracy numbers is from CNNs (TQT, CMSIS-NN). Without accuracy numbers: the ECP5 stories260K
  design stores every activation tensor as int8 times a power of two (its reference code picks the shift from the
  data) and uses four 256-entry tables with 16-bit entries (exp, reciprocal, 1/sqrt, SiLU); the board output equals
  the float model for 9 tokens and then differs https://github.com/skyline624/ecp5-inference ,
  https://github.com/skyline624/ecp5-inference/blob/main/host/v4_quant.py [VERIFIED]. Soul Player C64 (above) trains
  for per-tensor power-of-two shifts and gives no loss. Deeploy's flow uses TQT for thresholds and inserts
  requantisation layers; the text does not say that every scale is a power of two [VERIFIED].
- Softmax normalisation with only int8 intermediates and a table reciprocal: not found. TFLite uses 16 or 32 bits,
  SOLE 4-bit log values on models of 5M parameters and more without retraining; the ECP5 design has a 256-entry
  reciprocal table with 16-bit entries; the NES model normalises 3-bit weights by a shift or by an exact division.
- Dynamic Tanh, Derf, sigmoid attention, ReLU attention and ConSmax in language models under 1M parameters: not
  found. Smallest settings seen: ConSmax 6 layers x 384 [VERIFIED]; sigmoid attention learning-rate ablations on
  models from d_model 64 and 2 layers upward (reported as 2.2M non-embedding parameters)
  https://arxiv.org/abs/2409.04431 [VERIFIED]; DyT in a language model at 64M https://arxiv.org/abs/2604.23434
  [VERIFIED].
- A fixed-shift 1/sqrt table with training that holds the RMS inside the table's accurate window: not found.
  Leading-one RMSNorm was tested on LLaMA2-7B only, with 16- and 21-bit table entries [VERIFIED].
- A power-of-two-only LayerNorm for CLIP-ViT on FPGA (DATE 2026, "training-free") appeared in a search result
  [UNVERIFIED: the PDF URL returned HTTP 404 on 2026-11-13, again in the fact-check; the claim comes from search
  snippets only, and one snippet attributes the power-of-two terms to the GELU slopes and a log-based approximation
  to the reciprocal square root, so "power-of-two-only LayerNorm" may be inexact].
- Shift-based normalisation in training exists for batch norm in binarised CNNs ("did not observe accuracy loss")
  https://arxiv.org/abs/1602.02830 , not for layer-wise norms in transformers.
- 24-bit accumulators with 8-bit operands are used by two accelerators: ITA, which sizes the accumulator "to allow up
  to 256-element dot products", and the MAGNet setup in Softermax, which lists 24-bit accumulation without a term
  limit [VERIFIED; corrected: the earlier wording gave the 256-term limit for both]. No overflow study for
  language-model rows longer than 511 at this width was found apart from the general A2Q bounds.
- TinyStories-class models on ESP32 microcontrollers with int8 weights and float activations were mentioned in
  search results [UNVERIFIED: blog and model-card snippets only, not opened for this file; the ESP32 ports were
  opened for `research/04_tiny_hardware.md`].

## 5. Ranked ideas for LS-LLM

Ranked by expected value for the effort. "Token" means one generated token at T = 256, about 1.0 M cycles in the
CONTEXT.md estimate. Keep-or-drop tests use validation loss and practice sets only (never HumanEval, MBPP test or the
custom-200 test split), with 3 seeds where a training run is involved.

| # | Idea | Expected effect on score / speed / memory | Cost to try | Risk | Keep-or-drop measurement | CPU change |
|---|------|-------------------------------------------|-------------|------|--------------------------|------------|
| 1 | Round-half-up in every requantisation by preloading the accumulator with 2^(s-1) (plus any bias); simulate it bit for bit in QAT | Score: removes a -0.5 LSB bias per rounding (-5.5 LSB in an int8 residual stream after 11 roundings; -1.37 against +0.04 LSB in the softmax normaliser chain) [computed here]. Speed: 0 cycles if "load ACC" replaces "clear ACC". Memory: 0 | Hours | Low | QAT with floor against round, validation loss; keep round unless floor is equal | No, if ACC can be loaded with a constant; otherwise a rounding bit in RQ |
| 2 | Softmax = exp table, one extra "sum" row, value mix on raw table outputs, then divide the 8 outputs per head by a reciprocal table (fixed two-octave bands, section 6.2 Q1); top code as reference with a trained per-head offset instead of a maximum search | Speed: output-side normalisation about 4k to 7k cycles per layer instead of 25k to 66k for per-key normalisation (9 to 31% of a token), and no maximum search (about 25k per layer) [estimate]. Score: normaliser error 0.8 LSB rms [computed here]; exp-table error 1.3 to 4.2% total variation on random logits, more if row maxima drift below the top code. A bare power-of-two normaliser cost 0.029 nats/char against an exact one in the NES model (section 3) [VERIFIED]. Memory: 2 tables, 512 B | 2 to 3 days (QAT module, assembly, bit-exact test) | Medium-low | Validation loss of the float-softmax model against the same model with the exact integer softmax; cycles per token in the CPU simulator; histogram of (127 - row maximum) | No (code duplicated per band). A variable-shift RQ plus a leading-one instruction would remove the duplication |
| 3 | Measure residual-stream precision in the QAT simulator before writing assembly: float stream, int8, int8 with a per-layer power-of-two rescale, 16-bit | Decides idea 9. Expected int8 penalty in 5 layers: about 1.7 bits of read precision at the last layer [computed here]; the MAX 10 project lost 0.021 bits/char in total with an int8 stream over 16 layers (section 4) [VERIFIED]. No speed or memory effect | 1 day | None (measurement) | Validation-loss gap between int8 and 16-bit stream; keep int8 if the gap is inside seed noise | No |
| 4 | Norm comparison in float, then QAT of the winner: (a) RMSNorm with a fixed-shift 1/sqrt table, the gain folded into the next weights and a penalty that holds the RMS in the table window; (b) Dynamic Tanh as one table per norm site (tanh or hardtanh, trained alpha folded into the table, gamma folded into the next weights); (c) LayerNorm progressively replaced by BatchNorm and folded at export | Speed: RMSNorm is about 0.9 to 2.4% of a token, a table norm 0.8%, a folded BatchNorm 0 [estimate]. Score: unknown at this size. Memory: 256 B per table, 11 sites = 2.8 KB. (b) and (c) remove the sum of squares and the reciprocal square root | 3 float trainings x 3 seeds, then QAT | Medium (DyT collapse with SwiGLU in one preprint; BatchNorm collapse in transformers noted by SLAB; the RMS window may not hold) | Validation loss and practice-set scores against float RMSNorm; pick the cheapest variant inside seed noise | (a) needs both multiplier inputs from X memory, or a copy of x; (b), (c) no |
| 5 | Attention without normalisation: constant normaliser (ConSmax) or a sigmoid table with bias -log2(T) | Speed: drops the sum row and the output normalisation, about 4k to 7k cycles per layer (2 to 3.5% of a token) [estimate]. Score: unknown below 6 layers x 384. Memory: -256 B | 1 to 2 days in float | Medium-high (number of keys grows from 1 to T during generation; output saturation) | Float validation loss against softmax at equal steps; then saturation rate of attention outputs in QAT | No |
| 6 | Trained power-of-two thresholds per tensor (TQT style) and cross-layer equalisation at the start of QAT on pairs where a per-channel power of two moves for free: Q with K (per RoPE pair), V into O, W3 into W2 (I-LLM's FSBR uses the same pairs with real factors) | Score: per-tensor power-of-two scales matched float on CNNs only with trained thresholds (71.1% against 0.6% static); equalisation can recover up to 1 bit per channel. Speed 0, memory 0 | 1 to 2 days | Low | Validation loss with and without; share of int8 range used per row | No (per-row RQ shifts would need unrolled code or a variable-shift RQ) |
| 7 | Overflow rule in the exporter: assert n <= 511, or 128 * sum abs(w) + abs(preload) <= 8,388,607 per row; worst-case unit test; L1 penalty (A2Q style) only for Tier B rows that fail | Guarantees no accumulator overflow. No cost at run time | Hours | None | Exporter test passes; largest abs(ACC) seen in the simulator stays under the bound | No |
| 8 | Regenerate or calibrate table entries after QAT (from trained alpha, offsets and scale exponents), later direct int8 edits | NN-LUT gained 1.0 GLUE point from table calibration in a no-retraining setting; with exact tables in QAT the gain is probably small | Hours | Low | Validation loss before and after | No |
| 9 | 15- or 16-bit residual stream in two int8 limbs, blocks read an 8-bit window (only if idea 3 shows a loss) | Score: stream noise 30.4 dB to 66.6 dB, read precision back to a single rounding [computed here]. Speed: roughly +2 to 4% cycles [rough estimate]. Memory: +64 B | 2 to 3 days of assembly | Medium (complexity) | As idea 3, on the hardware-exact simulator | No (a 16-bit accumulator load and store would make it cheap) |
| 10 | Two-byte exp weights (0..16383) if 7-bit weights limit quality | Score: softmax total variation 1.3 to 4.2% down to about 0.5% [computed here]. Speed: doubles the value-mix and sum streams, about +20k cycles per layer (10% of a token) [estimate]. Memory: +256 B | 1 to 2 days | Low | Validation-loss gain against cycles lost | No |
| A | Table lookup inside the stream: MAC #n with the X operand passed through the 256-entry table | Speed: removes heads x T element-wise exp lookups, 24.6k to 65.5k cycles per layer (12 to 33% of a token) [estimate]; one more pipeline stage | VHDL change | Medium (timing) | Cycles per token; bit-exact against the software version | Yes |
| B | Variable-shift RQ (shift amount from a register) plus a leading-one (normalise) instruction | Score: softmax reciprocal and 1/sqrt at 0.9% worst error with one table each [computed here]; enables per-row weight scales and per-token dynamic scales. Speed: removes band code duplication | VHDL change | Low | Error against the exact quotient; validation loss | Yes |
| C | Unsigned operand mode for table outputs (weights 0..255) | Score: one more bit of attention weight; a 256-key value mix still fits (8,355,840 < 8,388,607) [computed here] | VHDL change | Low | Validation loss | Yes |
| D | Self-product stream (both multiplier inputs from X memory) | Needed only for RMSNorm's sum of squares without copying x | VHDL change | Low | n/a | Yes |

## 6. Tables asked for in the topic brief

### 6.1 Methods: what they replace, inference operations, reported accuracy, tested sizes

"QAT" = accuracy after quantisation-aware fine-tuning; "PTQ" = no retraining. The numbers repeat section 2 and
carry the tags given there; three cells were corrected in the fact-check (IntAttention operations, Kang and Wang
perplexity, Softermax baseline).

| Method | Replaces | Operations at inference | Reported accuracy change | Tested model sizes | Source |
|--------|----------|-------------------------|--------------------------|--------------------|--------|
| Jacob et al. integer inference | float conv/FC | int8 multiply-add into int32, int32 fixed-point multiply by M0, rounding shift | ResNet-50 76.4 to 74.9% (QAT) | ImageNet CNNs | https://arxiv.org/abs/1712.05877 |
| HAWQ-V3 dyadic scaling | float rescale | integer multiply by b, shift by c; residual add in INT32; no division | ResNet50 77.72 to 77.58% at W8A8 (QAT) | ResNet18/50, InceptionV3 | https://arxiv.org/abs/2011.10680 |
| TQT | real-valued scales | shift only (per-tensor, symmetric, power of two) | MobileNet v1 71.1 to 71.1%, v2 71.7 to 71.8% (QAT); 0.6% static | ImageNet CNNs | https://arxiv.org/abs/1903.08066 |
| CMSIS-NN q7 | float layers | shift-only requantisation with rounding preload; table sigmoid/tanh | CIFAR-10 CNN 80.3 to 79.9% (PTQ) | small CNN on Cortex-M7 | https://arxiv.org/abs/1801.06601 |
| I-BERT i-GELU | GELU | integer multiply, add (second-order polynomial), INT32 | GLUE avg 86.0 to 86.3 (Base), 89.0 to 89.5 (Large) for the whole pipeline (QAT) | RoBERTa-Base/Large | https://arxiv.org/abs/2101.01321 |
| I-BERT i-exp + softmax | softmax | polynomial, shift, integer division per element, INT32 | as above; exp gap 1.9e-3 | as above | https://arxiv.org/abs/2101.01321 |
| I-BERT integer LayerNorm | LayerNorm | integer sqrt by Newton (at most 4 iterations, one division each), division | as above | as above | https://arxiv.org/abs/2101.01321 |
| I-ViT Shiftmax | softmax | shifts and adds for 2^x, one integer division per row, multiply, shift | DeiT-S 79.85 to 80.12% (QAT, whole pipeline); polynomial instead: -0.12 / -0.22 | ViT-S/B, DeiT-T/S/B, Swin-T/S (about 5M to 86M parameters) | https://arxiv.org/abs/2207.01405 |
| I-ViT ShiftGELU | GELU | shifts, adds, Shiftmax-style sigmoid with integer division | polynomial instead: -0.86 / -0.91 | as above | https://arxiv.org/abs/2207.01405 |
| I-ViT I-LayerNorm | LayerNorm | integer mean and squares, 10 Newton iterations with division, one division per row, multiply, shift | L1 LayerNorm instead: -2.49 / -3.32 | as above | https://arxiv.org/abs/2207.01405 |
| FQ-ViT Power-of-Two Factor | per-channel scales at LayerNorm input | per-channel left shift (0 to 3 bits) | ViT-B 23.64 to 83.31% (float 84.53) (PTQ) | DeiT, ViT-B/L, Swin | https://arxiv.org/abs/2111.13824 |
| FQ-ViT Log-Int-Softmax | softmax and the attention multiply | i-exp, integer log2 by leading bit, 4-bit log attention, shift instead of multiply | ViT-B 83.31 to 82.68% (PTQ) | as above | https://arxiv.org/abs/2111.13824 |
| I-LLM (DI-MatMul, DI-ClippedSoftmax, DI-Exp, DI-Norm) | static scales, softmax, RMSNorm, SwiGLU | per-token dyadic scale from the most significant bit, clipping, shift-only exp with linear interpolation, bitwise integer sqrt, multiply | LLaMA-7B WikiText2 perplexity 5.68 to 5.84 (W6A6), 9.10 (W4A4) (PTQ) | LLaMA 7B to 30B, LLaMA2, LLaMA3-8B, OPT | https://arxiv.org/abs/2405.17849 |
| SOLE E2Softmax | softmax | shift-add log2 of exp, 4-bit log outputs, leading-one detection plus one mantissa bit; no multiplier, no table | DeiT-T 72.21 to 71.33%; BERT-Base SQuAD 88.17 to 87.57 (PTQ) | DeiT-T/S/B, Swin-T/S/B, BERT-Base | https://arxiv.org/abs/2510.17189 |
| SOLE AILayerNorm | LayerNorm | 4-bit compression, 16-entry square table, shifts, x^-0.5 table, 8-bit multiply | included in the row above | as above | https://arxiv.org/abs/2510.17189 |
| Softermax | softmax | base-2 power by 4-entry table and shift, online integer max, piecewise-linear reciprocal, integer multiply | BERT-Base SQuAD 86.28 (8-bit baseline) to 85.86; worst drop under 0.5% (QAT) | BERT-Base/Large | https://arxiv.org/abs/2103.09301 |
| NN-LUT | GELU, exp, division, 1/sqrt | 16-entry piecewise-linear table: compare, one multiply, one add; power-of-two prescale for 1/sqrt | RoBERTa GLUE avg 85.4 to 84.1, 85.1 with calibration; equally spaced table for 1/sqrt: CoLA 62.1 to 4.6 | RoBERTa-base, MobileBERT | https://arxiv.org/abs/2112.02191 |
| TFLite / CMSIS-NN int8 softmax | softmax | 32-bit fixed-point exp, count leading zeros, Newton reciprocal on the mantissa, multiply, rounding shift | kernel, no model result | n/a | https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/lite/kernels/internal/reference/softmax.h |
| TFLite int16 softmax | softmax | 513-entry exp table, count leading zeros, 513-entry 1/(1+x) table, multiply, shift | kernel, no model result | n/a | same file |
| CMSIS-NN `arm_softmax_q7` | softmax | base 2, shifts, one integer division per vector | kernel, no model result | n/a | https://github.com/ARM-software/CMSIS_5/blob/5.9.0/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_q7.c |
| ITA softmax | softmax | subtract max, shift by top 3 bits (8 weight levels), 15-bit sum, one serial division per row, shift | probability mean absolute error 0.46%; task accuracy not given | Compact Transformer activations | https://arxiv.org/abs/2307.03493 |
| Vasyltsov and Chang (reciprocal table, 2D table) | softmax | exp table, reciprocal table and multiply, or one 2D table; at most about 700 B | WMT14 BLEU 26.98 to 26.76/26.80 (uint8), 26.26/26.68 (uint4) (PTQ) | Transformer-base, BERT-base, DETR | https://arxiv.org/abs/2111.10770 |
| IntAttention IndexSoftmax | softmax | integer max-subtract, clip at 6.6 nats, 32-entry uint8 table, integer sum, per-element scaling by 255 / rowSum ("fixed-point scaling" in the paper) | Llama-3.2-1B perplexity 12.663 to 12.784 (table softmax alone), 13.070 whole pipeline (PTQ) | Llama-3.2-1B, OPT-1.3B, Qwen3-1.7B, ViTs | https://arxiv.org/abs/2511.21513 |
| Kang and Wang | softmax division; RMSNorm 1/sqrt | leading-one detection, small tables, log-domain subtraction; exponent table x mantissa table, one multiply | LLaMA2-7B perplexity 5.4943 to 5.4946 (exp tables), 5.4940 to 5.4971 (whole division-free softmax), 5.4956 (1/sqrt) (PTQ) | LLaMA2-7B | https://pmc.ncbi.nlm.nih.gov/articles/PMC12843610/ |
| ConSmax | softmax max, sum and division | one table lookup, one constant multiply | perplexity within 0.9% after 10K iterations, converged after about 20K (trained from scratch) | GPT, 6 layers x 384, context 256 | https://arxiv.org/abs/2402.10930 |
| Sigmoid attention | softmax | element-wise sigmoid(score + b): one table, no normalisation | 1B LLM: 49.5 against 49.4 (n = 2048); needs extra output norm at n = 4096 | ViT-B, 255M ASR, 85M to 7B LLMs | https://arxiv.org/abs/2409.04431 |
| ReLU attention / L | softmax | ReLU, division by sequence length | "approach or match" softmax | ViT S/32 to L/16 (vision only) | https://arxiv.org/abs/2309.08586 |
| Polynomial attention + L1 LayerNorm (Integer Transformer) | softmax, LayerNorm | [ReLU(x+b)]^n, integer division after the value mix; abs, sum, division | float: perplexity 31.28 to 31.20 / 31.18; INT8: 31.61 against 31.18; about -0.3 BLEU, En-Fr more than -1 | Transformer-base/big, LM base/big | https://arxiv.org/abs/2009.08034 |
| Dynamic Tanh | LayerNorm, RMSNorm | element-wise tanh(alpha x), per-channel scale and shift: one table lookup per element | LLaMA 7B loss 1.59 to 1.60; 70B 1.45 to 1.45; ViT-B 82.3 to 82.5; clamp without alpha 80.7 | ViT, ConvNeXt, DiT, wav2vec, LLaMA 7B to 70B | https://arxiv.org/abs/2503.10622 |
| MobileBERT NoNorm | LayerNorm | per-channel affine, foldable | GLUE 78.5 to 77.7 (with ReLU for GELU); 8-bit PTQ then free | 25.3M, trained with a teacher | https://arxiv.org/abs/2004.02984 |
| SLAB progressive BatchNorm (abstract only) | LayerNorm | none after folding | "comparable performance" on language modelling | Swin, DeiT, a language model | https://arxiv.org/abs/2405.11582 |
| A2Q | wide accumulator | none at inference (L1 constraint in training) | bound only quoted here | CNNs; LLM variant in https://arxiv.org/abs/2409.17092 | https://arxiv.org/abs/2308.13504 |
| Clipped softmax | softmax (training-time) | clip((zeta - gamma) softmax + gamma, 0, 1) | BERT W8A8 perplexity 1294 to 4.52; OPT-125m 21.18 to 37.20 (worse) | BERT-base, OPT-125m, ViT-S | https://arxiv.org/abs/2306.12929 |
| Shift-based batch norm | batch-norm scaling | power-of-two approximation (most significant bit), shifts | "did not observe accuracy loss" | binarised nets on MNIST, CIFAR-10, SVHN | https://arxiv.org/abs/1602.02830 |

### 6.2 Design questions

Notation: `RQ(a, s)` = arithmetic shift right by the constant s, then saturate to int8. All tables have 256 entries
and int8 outputs. T = number of keys.

**Q1. Softmax without division, exp table indexed by an int8 score**

1. *Table.* With m score steps per octave, E[s] = round(127 * 2^((s - 127) / m)), set to 0 where it rounds below 1.
   Non-zero entries span log2(254) = 8.0 octaves = 5.5 nats (9.0 octaves = 6.2 nats for outputs 0..255). Windows in
   the literature: 8 octaves (CMSIS-NN q7, ITA), 6.6 nats with a stable range 5.5 to 7.7 (IntAttention), 10 nats
   (TFLite int16), 16 nats (Kang and Wang), almost no loss from 20 down to 10 nats (I-LLM: perplexity 9.15 to 9.23 at
   W4A4) [VERIFIED; corrected from "no loss"]. With m = 32 the window uses all 256 codes (ITA's scale);
   with m = 16 it uses 128 codes and leaves room for a fixed reference. Score resolution is not the limit: score
   rounding alone costs 0.26% total variation at m = 32 and 1.0% at m = 8, against 1.3 to 4.2% for the 7-bit table
   output [computed here].
2. *Reference.* Softmax does not depend on the reference point, so the maximum search can be replaced by the top
   code 127: the saturating RQ clamps larger scores. Cost: if the row maximum sits g codes below 127 the largest
   weight is 127 * 2^(-g/m) and precision falls. At m = 16 and logit standard deviation 2 nat the total variation is
   4.2% (g = 0), 7.4% (g = 16), 12.5% (g = 32), 30% (g = 64) [computed here]. So hold row maxima within about one
   octave of the top code (a trained per-head offset in the accumulator preload, plus a penalty on 127 - maximum in
   QAT), or pay about 24.6k cycles per layer for a search [estimate].
3. *Normalise the output, not the weights.* softmax(s) V = (sum_j E[s_j] v_j) / (sum_j E[s_j]). Compute
   y_d = sum_j E[s_j] v_jd (one stream of length T per output dimension; abs(y_d) <= 128 S <= 4,161,536) and
   S = sum_j E[s_j] (one more stream against a row of ones; S <= 32,512). o_d = y_d / S is a weighted average of the
   values, so abs(o_d) <= 128 and it has the scale of v. This needs head_dim quotients per head (8) instead of T (256).
   The Integer Transformer orders the operations the same way.
4. *The quotient.* Let k be the position of the leading one of S.

| Normaliser | Formula | Ratio to exact 1/S | Extra operations |
|------------|---------|--------------------|------------------|
| Exact | o = y / S | 1 | integer division (I-BERT, I-ViT, IntAttention, CMSIS q7 use one) |
| Power of two | o = y >> (k+1) | 0.50 to 1.00, mean 0.75, 29% rms; with the constant 1.337 folded into the next weights 0.67 to 1.34, 19% rms | leading-one detection, variable shift |
| Power of two and one mantissa bit (SOLE) | factor 0.818 or 0.568 | 0.82 to 1.23, 10.3% rms | plus one bit test |
| Reciprocal table on a b-bit mantissa (leading one included) | o = (y >> k') * R[mantissa] >> c | b = 3: 0.89 to 1.11, 5.1% rms; b = 4: 0.94 to 1.06, 2.6%; b = 5: 0.97 to 1.03, 1.3%; b = 8 with 7-bit entries: 0.991 to 1.009, 0.38% | leading-one detection, variable shift, table, one multiply (TFLite int16, Kang and Wang) |
| Reciprocal table in fixed two-octave bands (fits LS as it is) | choose band b with S >> b in 64..255; t = RQ(y + 2^(b+7), b+8); o = RQ(t * R[S >> b] + 16, 5); R[a] = round(2^13 / (a + 0.5)), 32..127 | output error 0.80 LSB rms, 3 LSB worst, mean +0.04 LSB; with floor shifts 1.63 rms, mean -1.37 | compares to choose the band, constant shifts, table, one int8 multiply per output |
| None | o = RQ(y, c) | not an approximation; the model is trained with it | none (ConSmax, sigmoid attention) |

   Rows 2 to 5 [computed here]. In the band version t always fits int8 because abs(y) < 128 * 2^(b+8), and t keeps at
   least 5 bits when the output is full scale. Indexing all 256 entries with a non-negative quantity needs an offset:
   preload the accumulator with -128 * 2^s so that RQ returns (value >> s) - 128.
5. *Cycles per layer [estimate].* Scores 8 x 256 x (8 + 24) = 65.5k; exp lookups 8 x 256 x (12 to 32) = 24.6k to
   65.5k; value mix 64 x (256 + 24) = 17.9k; sum rows 8 x 280 = 2.2k; output normalisation 64 x (24 to 64) = 1.5k to
   4.1k plus band selection. Normalising each weight instead would cost 8 x 256 x (12 to 32) = 24.6k to 65.5k.

**Q2. RMSNorm or LayerNorm with a 1/sqrt table when the sum of squares has about 20 bits**

- Size: S = sum x_i^2 <= 64 * 128^2 = 2^20 for dim 64 (2,818,048 for 172); it fits the accumulator. For dim 64,
  S = 64 RMS^2: RMS 4, 8, 16, 32, 64 LSB give S = 1,024, 4,096, 16,384, 65,536, 262,144. On LS the sum of squares
  needs both multiplier inputs to read the same vector (both pointers on X memory, or a copy of x in W memory).
- One table bucket covering index a (S between a 2^s and (a+1) 2^s), with the entry at the geometric centre, has a
  worst relative error of ((a+1)/a)^(1/4) - 1, about 1/(4a): 5.7% at a = 4, 3.0% at 8, 1.5% at 16, 0.77% at 32, 0.39%
  at 64 [computed here].
- *Fixed shift, one table:* a = S >> s. Error at most 1.5% over a 16x range of S (4x of RMS), at most 5.7% over 64x
  of S (8x of RMS). With only 128 usable codes the ranges halve in S. Works only if training holds the RMS of each
  norm input inside the window.
- *Two fixed shifts, two tables, one compare:* shifts 8 and 12 give at most 1.6% for S >= 4,096 and at most 6% for
  S >= 1,024 (RMS from 8, respectively 4, up to 128 LSB); shifts 6 and 12 give at most 6% for S >= 256 (RMS >= 2).
- *Leading-one detection with an even shift:* S = m * 4^e with m in 64..255, 1/sqrt(S) = R[m] * 2^(-e). Worst error
  0.85% with 7-bit entries and 0.56% with 8-bit entries over S from 64 to 2^20. The even shift avoids the sqrt(2)
  factor for odd exponents; Kang and Wang use a second small table for it. Needs a variable shift on LS.
- Effect of the error: a relative error e in 1/sqrt scales the whole normalised vector by (1 + e); queries and keys
  both carry it, so attention scores move by about 2e.
- Literature check: an equally spaced 16-entry table for 1/sqrt breaks RoBERTa and a power-of-two prescale fixes it
  (NN-LUT); SOLE's 4-bit compressed squares cost 0.2% on E(x^2) and 0.4% on the standard deviation for uniform
  inputs; I-BERT, I-ViT and I-LLM compute an integer square root and then divide.
- Options without any statistic: a Dynamic Tanh table y = round(127 tanh(alpha x 2^-f)) per element (bounded output
  at a fixed scale); a clamp alone (hardtanh without alpha) is the int8 saturation the stream already has and gave
  80.7% against 82.3% for LayerNorm on ViT-B in the DyT ablation. [VERIFIED]

**Q3. Residual stream in int8: bit widths in the literature and what breaks at 8 bits**

| Pipeline | Matmul inputs / weights | Residual stream | Internals of softmax and norm | Source |
|----------|-------------------------|-----------------|-------------------------------|--------|
| Jacob et al. | 8 / 8 | 8-bit add with rescaling | n/a, accumulator int32 | https://arxiv.org/abs/1712.05877 |
| HAWQ-V3 | 4 to 8 | addition in INT32 | n/a | https://arxiv.org/abs/2011.10680 |
| I-BERT (Hugging Face code) | 8 / 8 | 22-bit sum into LayerNorm, 32-bit LayerNorm output, then 8 | INT32 | https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/ibert/modeling_ibert.py |
| I-ViT (code) | 8 / 8 | 16 bit at both residual additions | softmax output 16 bit, divisions against 2^31 | https://github.com/zkkli/I-ViT/blob/main/models/vit_quant.py |
| FQ-ViT | 8 / 8 | 8 bit plus per-channel shift of 0 to 3 bits at LayerNorm inputs | 4-bit log attention | https://arxiv.org/abs/2111.13824 |
| Bondarenko et al. 2021 | 8 / 8 | 8 bit fails post-training (71.03); 16-bit residual sums recover (82.43); 8-bit QAT works (83.26) | float (simulated quantisation) | https://arxiv.org/abs/2109.12948 |
| Integer Transformer | 8 / 8 | 8 bit with run-time scales; "adding a few bits" recovers | L1 norm, polynomial attention, integer division | https://arxiv.org/abs/2009.08034 |
| I-LLM | 4 to 8 | per-token dynamic dyadic scales | shift-based exp, integer sqrt | https://arxiv.org/abs/2405.17849 |
| Deeploy | 8 / 8 | not stated | I-BERT softmax, 32-bit norm divisions | https://arxiv.org/abs/2408.04413 |
| MAX 10 TinyStories project (hobby, 2026) | 8 / ternary (int8 in the embedding and output layer) | int8 through 16 layers; stream and branch rescaled by multiply and shift at each addition | Q15 softmax weights with an integer reciprocal; integer square root and division in RMSNorm | https://github.com/MRazvan/fpga-tinystories/blob/main/ref/int_model.py |
| Softermax, ITA (hardware) | 8 / 8 | n/a | 15 to 16 bit; accumulator 24 bit | https://arxiv.org/abs/2103.09301 , https://arxiv.org/abs/2307.03493 |

Rows compared again with their sources in the fact-check: all except Jacob et al. [VERIFIED]; the MAX 10 row was
added.

What breaks at 8 bits, and how large it is:
- Range mismatch between a branch output and the stream it is added to, with a few outlier dimensions; this is what
  sinks post-training BERT (Bondarenko et al.) and why FQ-ViT needs per-channel shifts (ranges above 40x the median).
- The last residual sum and norm lose the most, because the stream is the sum of all earlier branches (Lin et al.).
- Rounding noise adds up: embedding plus 10 branch roundings give 11/12 LSB^2. With a stream standard deviation of
  32 LSB (clipping at 4 sigma) that is 30.4 dB against 40.9 dB for one rounding, i.e. the stream read by the last
  layer is as noisy as a 6.3-bit activation; with floor shifts add a -5.5 LSB offset [computed here].
- A branch output smaller than half an LSB disappears.
- A 16-bit stream with 6 extra fraction bits measures 66.6 dB [computed here], but a block still reads 8 bits, so
  the gain at the read is at most back to 40.9 dB (1.7 bits) in a 5-layer model.

**Q4. Overflow rules for int8 x int8 products in a signed 24-bit accumulator (limit 8,388,607)**

1. Any accumulation of at most 511 products is safe for all operand values, starting from zero
   (511 x 16,384 = 8,372,224). 516 if weights never take -128; 520 if neither operand does [computed here].
2. With a preload b (bias, rounding constant, index offset): abs(b) + 16,384 n <= 8,388,607.
3. Rows longer than 511 (Tier B): check each row once at export, 128 * sum abs(w_i) + abs(b) <= 8,388,607, i.e.
   sum abs(w_i) <= 65,535 without preload. This is the A2Q bound for 8-bit signed inputs and it also covers every
   partial sum [VERIFIED]. With a mean abs(w) of 32 a row may have up to 2,047 terms.
4. Products of two data vectors have no static weights, so rule 1 applies or a known bound on one operand:

| Accumulation in the baseline | Terms | Worst abs(ACC) | Share of the limit |
|------------------------------|-------|----------------|--------------------|
| Score q.k | 8 | 131,072 | 1.6% |
| Weight row, dim 64 (also the output layer) | 64 | 1,048,576 | 12.5% |
| Sum of squares, dim 64 | 64 | 1,048,576 | 12.5% |
| Weight row, SwiGLU hidden 172 | 172 | 2,818,048 | 33.6% |
| Sum of exp weights (0..127) against ones | 256 | 32,512 | 0.4% |
| Value mix, weights 0..127 | 256 | 4,161,536 | 49.6% |
| Value mix, weights 0..255 (unsigned mode) | 256 | 8,355,840 | 99.6% |

5. If the accumulator wraps around (two's complement) and never saturates, an intermediate overflow is harmless as
   long as the final sum is in range; a test with partial sums up to 9,677,400 ended at the exact value -83,454
   [computed here]. Do not rely on this unless the VHDL accumulator wraps.
6. RQ saturates, so nothing overflows after requantisation, but saturation is itself an error: log the clip rate of
   every RQ in the QAT simulator.

**Q5. Floor against round-to-nearest in shift requantisation**

- Floor: error mean -(2^s - 1) / 2^(s+1), about -0.5 LSB, variance (1 - 2^(-2s)) / 12. Round-half-up, i.e.
  (ACC + 2^(s-1)) >> s: mean +2^-(s+1), same variance. Measured on 64-term int8 dot products: shift 7: -0.4965
  against +0.0043; shift 4: -0.4688 against +0.0318; shift 1: -0.2506 against +0.2506 [computed here]. For shifts
  below about 4 round-half-up is itself biased; TQT uses round-half-to-even.
- Where the bias goes: (a) the residual stream collects -0.5 LSB per rounding, -5.5 LSB after 11; (b) an offset d on
  all inputs of a row moves output j by d * sum_k w_jk / 2^s; (c) chained shifts compound it: the two-shift softmax
  normaliser has mean error -1.37 LSB with floor and +0.04 with rounding [computed here].
- Implementation: start the accumulator at 2^(s-1) instead of 0 (legacy CMSIS-NN: `NN_ROUND(out_shift)` added to the
  bias; an `ARM_NN_TRUNCATE` switch turns it off). [VERIFIED]
- Literature: Jacob et al. report "significant loss of end-to-end accuracy" from a tie-breaking bias alone
  [VERIFIED]; the paper does not describe the experiment, so that inference differed from the training simulation
  there is our reading [UNVERIFIED: not stated in the paper]. I-ViT and I-BERT use floor inside their shift formulas
  and train through it (`floor_ste` in the I-ViT code [VERIFIED]). The MAX 10 project rounds half up in every
  rescale (section 4) [VERIFIED].

**Q6. How lookup tables and other integer operators are trained through**

- Exact integer operation forward, identity gradient through each floor or round backward: I-ViT wraps every floor
  and round in `floor_ste` / `round_ste`; Softermax's "forward passes faithfully implement the fixed point, low
  precision computations, while the backward passes use the straight through estimator"; A2Q states the rule as
  grad of round(x) = 1. [VERIFIED]
- Trained scales: TQT learns log2 of each threshold by gradient and rounds it up to a power of two in the forward
  pass. [VERIFIED]
- Tables fitted separately, then calibrated: NN-LUT trains a one-hidden-layer ReLU network per function, converts it
  to a table, then regresses the table against the float function on unlabeled data with the model frozen (+1.0
  GLUE average). [VERIFIED]
- Trained constants replacing a normaliser: ConSmax learns beta and gamma end to end and merges them at inference;
  initial values are picked by a short warm-up sweep. [VERIFIED]
- For a 256-entry table on LS this means: forward y = Table[index(x)] exactly as the CPU computes it; backward the
  derivative of the smooth function the table samples, i.e. y = f(x) + stop_gradient(Table[index(x)] - f(x)). For
  band or leading-one selection, treat the selected band as a constant in the backward pass. Regenerate the table
  from the trained continuous parameters at export.
- Starting point: I-BERT, I-ViT, Softermax and TQT all fine-tune a trained float model; none of them trains from
  scratch with the integer operators in the loop [VERIFIED]. Outside the papers, the NES project trains its ternary
  model from a "differentiable twin" of the integer reference ("STE on every hard op") and tests that the trainer's
  forward pass equals the reference exactly; Soul Player C64 trains with fake-quantised int8 weights and selects
  checkpoints by the integer forward pass (section 4) [VERIFIED].

## 7. Sources

Read depth: F = full text opened and searched, A = abstract only, C = source code read. Licences are given for code,
models and datasets; papers are cited, not redistributed.

| Title | Authors | Year | URL | Licence (code / model) | Read |
|-------|---------|------|-----|------------------------|------|
| Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | B. Jacob, S. Kligys, B. Chen, M. Zhu et al. | 2018 | https://arxiv.org/abs/1712.05877 | paper | F |
| HAWQ-V3: Dyadic Neural Network Quantization | Z. Yao, Z. Dong, Z. Zheng, A. Gholami et al. | 2021 | https://arxiv.org/abs/2011.10680 | code https://github.com/Zhen-Dong/HAWQ : MIT | F |
| I-BERT: Integer-only BERT Quantization | S. Kim, A. Gholami, Z. Yao, M. W. Mahoney, K. Keutzer | 2021 | https://arxiv.org/abs/2101.01321 | code https://github.com/kssteven418/I-BERT : MIT | F |
| Hugging Face Transformers, `modeling_ibert.py` (v4.44.0) | Hugging Face | 2024 | https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/ibert/modeling_ibert.py | Apache-2.0 | C |
| I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference | Z. Li, Q. Gu | 2023 | https://arxiv.org/abs/2207.01405 | code https://github.com/zkkli/I-ViT : Apache-2.0 | F, C |
| FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer | Y. Lin, T. Zhang, P. Sun, Z. Li, S. Zhou | 2022 | https://arxiv.org/abs/2111.13824 | code https://github.com/megvii-research/FQ-ViT : Apache-2.0 | F |
| I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models | X. Hu, Y. Cheng, D. Yang, Z. Yuan et al. | 2024 | https://arxiv.org/abs/2405.17849 | paper | F |
| Deeploy: Enabling Energy-Efficient Deployment of Small Language Models On Heterogeneous Microcontrollers | M. Scherer, L. Macan, V. Jung, P. Wiese et al. | 2024 | https://arxiv.org/abs/2408.04413 | code https://github.com/pulp-platform/Deeploy : Apache-2.0 for most source files and MIT for some scripts according to the README; `LICENSES/` holds Apache-2.0.txt, MIT.txt and CC-BY-ND-4.0.txt; the GitHub API reports no single licence [VERIFIED] | F |
| TinyLLama-v0 (model used by Deeploy) | Maykeye | 2023 | https://huggingface.co/Maykeye/TinyLLama-v0 | Apache-2.0 (model card tag) | model API |
| SOLE: Hardware-Software Co-design of Softmax and LayerNorm for Efficient Transformer Inference | W. Wang, S. Zhou, W. Sun et al. | 2025 (arXiv posting) | https://arxiv.org/abs/2510.17189 | paper | F |
| Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers | J. R. Stevens, R. Venkatesan, S. Dai, B. Khailany et al. | 2021 | https://arxiv.org/abs/2103.09301 | paper | F |
| NN-LUT: Neural Approximation of Non-Linear Operations for Efficient Transformer Inference | J. Yu, J. Park, S. Park, M. Kim et al. | 2021 | https://arxiv.org/abs/2112.02191 | paper | F |
| ITA: An Energy-Efficient Attention and Softmax Accelerator for Quantized Transformers | G. Islamoglu, M. Scherer, G. Paulin, T. Fischer et al. | 2023 | https://arxiv.org/abs/2307.03493 | paper | F |
| Efficient Softmax Approximation for Deep Neural Networks with Attention Mechanism | I. Vasyltsov, W. Chang | 2021 | https://arxiv.org/abs/2111.10770 | paper | F |
| IntAttention: A Fully Integer Attention Pipeline for Efficient Edge Inference | W. Zhong, H. Feng, Z. Zhou, H. Peng, S. Yu | 2025 | https://arxiv.org/abs/2511.21513 | paper | F |
| Hardware-Oriented Approximations of Softmax and RMSNorm for Efficient Transformer Inference (Micromachines 17(1):84) | Y. Kang, D. Wang | 2026 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12843610/ | article CC BY | F |
| ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters | S. Liu, G. Tao, Y. Zou, D. Chow et al. | 2024 | https://arxiv.org/abs/2402.10930 | code https://github.com/ReaLLMASIC/ConSmax : Apache-2.0 | F |
| Theory, Analysis, and Best Practices for Sigmoid Self-Attention | J. Ramapuram, F. Danieli, E. Dhekane, F. Weers et al. | 2024 | https://arxiv.org/abs/2409.04431 | code https://github.com/apple-aiml-research/ml-sigmoid-attention : custom Apple licence text (GitHub reports NOASSERTION) | F |
| Replacing softmax with ReLU in Vision Transformers | M. Wortsman, J. Lee, J. Gilmer, S. Kornblith | 2023 | https://arxiv.org/abs/2309.08586 | paper | F |
| A Study on ReLU and Softmax in Transformer | K. Shen, J. Guo, X. Tan, S. Tang, R. Wang, J. Bian | 2023 | https://arxiv.org/abs/2302.06461 | paper | A |
| Sparse Attention with Linear Units | B. Zhang, I. Titov, R. Sennrich | 2021 | https://arxiv.org/abs/2104.07012 | paper | A |
| Towards Fully 8-bit Integer Inference for the Transformer Model | Y. Lin, Y. Li, T. Liu, T. Xiao, T. Liu, J. Zhu | 2020 | https://arxiv.org/abs/2009.08034 | paper | F |
| Transformers without Normalization | J. Zhu, X. Chen, K. He, Y. LeCun, Z. Liu | 2025 | https://arxiv.org/abs/2503.10622 | code https://github.com/jiachenzhu/DyT : MIT | F |
| Stronger Normalization-Free Transformers | M. Chen, T. Lu, J. Zhu, M. Sun, Z. Liu | 2025 | https://arxiv.org/abs/2512.10938 | paper | A |
| When Does Removing LayerNorm Help? Activation Bounding as a Regime-Dependent Implicit Regularizer | L. Verma | 2026 | https://arxiv.org/abs/2604.23434 | paper | F (parts) |
| Does Your Optimizer Care How You Normalize? Normalization-Optimizer Coupling in LLM Training | A. Abouzeid | 2026 | https://arxiv.org/abs/2604.01563 | paper | A |
| Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models | H. Byun, Y. Choi, T. Kim, S. Park, K. Song | 2025 | https://arxiv.org/abs/2601.09719 | code https://github.com/MLAI-Yonsei/BHyT : Apache-2.0 | A |
| On the Mathematical Relationship Between Layer Normalization and Dynamic Activation Functions | F. Stollenwerk | 2025 | https://arxiv.org/abs/2503.21708 | paper | A |
| FlashNorm: Fast Normalization for Transformers | N. Graef, F. Makraduli, A. Wasielewski, M. Clapp | 2024 | https://arxiv.org/abs/2407.09577 | paper | A |
| MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices | Z. Sun, H. Yu, X. Song, R. Liu et al. | 2020 | https://arxiv.org/abs/2004.02984 | paper | F |
| SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization | J. Guo, X. Chen, Y. Tang, Y. Wang | 2024 | https://arxiv.org/abs/2405.11582 | code https://github.com/xinghaochen/SLAB : no licence detected by the GitHub API | A |
| HAAN: A Holistic Approach for Accelerating Normalization Operations in Large Language Models | T. Peng, J. Qin, T. Xia, S. Q. Zhang | 2025 | https://arxiv.org/abs/2502.11832 | paper | F (parts) |
| Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks | S. R. Jain, A. Gural, M. Wu, C. H. Dick | 2020 | https://arxiv.org/abs/1903.08066 | code https://github.com/Xilinx/graffitist : BSD 3-Clause text in LICENSE (GitHub reports NOASSERTION) | F |
| CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs | L. Lai, N. Suda, V. Chandra | 2018 | https://arxiv.org/abs/1801.06601 | paper | F |
| CMSIS_5 5.9.0: `arm_softmax_q7.c`, `arm_fully_connected_q7.c`, `arm_nnsupportfunctions.h` | Arm | 2020 | https://github.com/ARM-software/CMSIS_5/tree/5.9.0/CMSIS/NN | Apache-2.0 | C |
| CMSIS-NN v6.0.0: `arm_softmax_s8.c` | Arm | 2024 | https://github.com/ARM-software/CMSIS-NN/blob/v6.0.0/Source/SoftmaxFunctions/arm_softmax_s8.c | Apache-2.0 | C |
| TensorFlow Lite v2.16.1: `reference/softmax.h`, `common.h`, `optimized/optimized_ops.h` | TensorFlow authors | 2024 | https://github.com/tensorflow/tensorflow/tree/v2.16.1/tensorflow/lite/kernels/internal | Apache-2.0 | C |
| A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance | I. Colbert, A. Pappalardo, J. Petri-Koenig | 2023 | https://arxiv.org/abs/2308.13504 | code in https://github.com/Xilinx/brevitas : BSD 3-clause style text plus third-party notices in LICENSE (GitHub reports NOASSERTION) | F (parts) |
| Accumulator-Aware Post-Training Quantization for Large Language Models | I. Colbert, G. Franco, F. Grob, J. Zhang, R. Saab | 2024 | https://arxiv.org/abs/2409.17092 | paper | A |
| Understanding and Overcoming the Challenges of Efficient Transformer Quantization | Y. Bondarenko, M. Nagel, T. Blankevoort | 2021 | https://arxiv.org/abs/2109.12948 | paper | F |
| Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing | Y. Bondarenko, M. Nagel, T. Blankevoort | 2023 | https://arxiv.org/abs/2306.12929 | code https://github.com/Qualcomm-AI-research/outlier-free-transformers : BSD-3-Clause-Clear | F (parts) |
| Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1 | M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, Y. Bengio | 2016 | https://arxiv.org/abs/1602.02830 | paper | F (parts) |
| llama2.c64 (llama2.c on a Commodore 64) | ytmytm | 2025 | https://github.com/ytmytm/llama2.c64 | no licence file in the repository root | README |
| llama2.c and the `stories260K` checkpoint (baseline shape) | A. Karpathy | 2023 | https://github.com/karpathy/llama2.c , https://huggingface.co/karpathy/tinyllamas | MIT (code), MIT (model card tag) | licence check only |
| elya-nes (ternary transformer on the NES; added in the fact-check) | Scottcjn | 2026 | https://github.com/Scottcjn/elya-nes | Apache-2.0 | README |
| fpga-tinystories (ternary TinyStories model on a MAX 10; added in the fact-check) | MRazvan | 2026 | https://github.com/MRazvan/fpga-tinystories | MIT | README, C (`ref/int_model.py`) |
| ecp5-inference (stories260K on a Lattice ECP5; added in the fact-check) | skyline624 | 2026 | https://github.com/skyline624/ecp5-inference | no licence detected by the GitHub API | README, C (`host/v4_quant.py`) |
| Soul Player C64 (int8 transformer on a Commodore 64; added in the fact-check) | gizmo64k | 2026 | https://github.com/gizmo64k/soulplayer-c64 | GPL-3.0 | README |

Fact-check of this table: the titles and authors of the 36 arXiv entries match the arXiv API (the year column gives
the arXiv year or the venue year), and every licence entry matches the GitHub API (repository licence field and,
where it reports NOASSERTION, the LICENSE text) or the Hugging Face model API, all read on 2026-11-13 [VERIFIED].

Not opened: the DATE 2026 paper on power-of-two LayerNorm for CLIP-ViT
(https://past.date-conference.com/proceedings-archive/2026/DATA/436.pdf returned HTTP 404), and the ESP32 TinyStories
write-ups that appeared in search results. Both are mentioned in section 4 with an [UNVERIFIED] tag only. The DATE
URL returned HTTP 404 again in the fact-check.

Reproduce the [computed here] numbers:

```
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
.venv/bin/python research/02_integer_inference_checks.py
```

## Verification log

Fact-check pass of 2026-11-13. One line per checked claim: claim | verdict | source opened again. A line may cover
neighbouring statements of the same kind. Papers were read as full text (PDF converted with pdftotext), code and
READMEs through raw.githubusercontent.com or the GitHub API. Statements without a tag in the file were not re-checked.

1. The 36 arXiv identifiers cited in this file exist with the stated titles and authors, and the abstract-only quotes (SLAB, FlashNorm, BHyT, Derf, Abouzeid, A2Q for LLMs) match the abstracts | confirmed | https://export.arxiv.org/api/query (id_list with the 36 identifiers)
2. IntAttention: Llama-3.2-1B WikiText perplexity 12.663 / 12.784 / 13.070 / 13.701, 32-entry uint8 table, c = 6.6, plateau for at least 16 entries and c in [5.5, 7.7], Qwen3-1.7B 23.01 / 27.75 / 32.95 | confirmed | https://arxiv.org/abs/2511.21513
3. IntAttention on Llama-3.2-1B-Instruct: "the same three settings give HumanEval 32.93, 32.32, 31.10 and MBPP 33.00, 31.40, 34.20" | corrected: FP16 / table softmax alone / whole pipeline -> FP16 / Quant-Only / whole pipeline (their Table 3); the table softmax alone gives 31.10 and 34.20 (their Table 7); these are pass rates and the paper does not state k | https://arxiv.org/abs/2511.21513
4. Deeploy: hidden size 64, 16 heads, 8 layers, FFN 256; 340 tokens/s and 490 uJ/token; 8-bit matmuls with TQT; I-BERT softmax; Hardswish; 32-bit RMSNorm divisions; no accuracy number for the quantised model | confirmed; parameter count 4,621,392 added from the model card | https://arxiv.org/abs/2408.04413 , https://huggingface.co/Maykeye/TinyLLama-v0
5. I-BERT: GLUE 86.0 to 86.3 and 89.0 to 89.5, i-exp gap 1.9e-3, at most four Newton iterations; Hugging Face code act_bit 8, ln_input_bit 22, ln_output_bit 32, embedding_act_bit 16, bias_bit 32 | confirmed | https://arxiv.org/abs/2101.01321 , https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/ibert/modeling_ibert.py
6. I-ViT: DeiT-S 79.85 to 80.12, ViT-S 81.39 to 81.27, Swin-S 83.20 to 83.01; ablations -0.12 / -0.22, -0.86 / -0.91, -2.49 / -3.32; code has QuantAct(16) at both residual additions, IntSoftmax(16), divisions of 2^31 - 1 by the row sum and by the standard deviation, 10 Newton iterations | confirmed | https://arxiv.org/abs/2207.01405 , https://github.com/zkkli/I-ViT/blob/main/models/vit_quant.py , https://github.com/zkkli/I-ViT/blob/main/models/quantization_utils/quant_modules.py
7. Section 1 (6): 8-bit quantisation-aware fine-tuning "stayed within about 0.3 points of float" (I-BERT, I-ViT, Bondarenko et al.) | corrected: within about 0.3 points -> between 0.2 below and 0.5 above float (I-BERT +0.3 / +0.5, I-ViT -0.19 to +0.27, Bondarenko et al. +0.20); added that the I-BERT operators lose 0.74 to 1.34 points on vision transformers in I-ViT's Table 1 | sources of lines 5, 6 and 8
8. Bondarenko et al.: BERT-base GLUE 83.06 / 71.03 / 83.23 / 82.43 / 83.26 (2021); clipped softmax 735 to 21.5 and 1294 to 4.52, OPT-125m 21.18 / 37.20 / 16.02 (2023) | confirmed | https://arxiv.org/abs/2109.12948 , https://arxiv.org/abs/2306.12929
9. Dynamic Tanh: LLaMA loss 1.59 to 1.60, 1.53 to 1.54, 1.50, 1.45; ViT-B 82.3 to 82.5; alpha table 1.0/1.0, 0.8/0.2, 0.2/0.05; hardtanh 82.2; clamp without alpha 80.7; ResNet-50 76.2 to 68.9. MobileBERT NoNorm: GLUE 78.5 to 77.7, 192 ms to 62 ms, 8-bit quantisation 83.9 to 83.9 and 90.0 to 90.0 | confirmed; corrected in section 3: "the smallest width in the original study is 1024" -> the smallest width in its LLaMA alpha table is 1024 | https://arxiv.org/abs/2503.10622 , https://arxiv.org/abs/2004.02984
10. Verma preprint: GPT-2-family models from 64M parameters, DyT -27.3% validation loss at 1M tokens and +18.8% at 118M, about 33% per-seed collapse localised to SwiGLU, 5K steps, compute-limited | confirmed | https://arxiv.org/abs/2604.23434
11. Integer Transformer: value mix before the division; WikiText-103 perplexity 31.28 / 31.20 / 31.18 in float, INT8 31.61 against 31.18; about -0.3 BLEU, En-Fr more than -1; "adding a few bits can recover most of the performance" | confirmed; note added that the paper's float baseline is 31.28 in its Table 4 and 31.18 in its Table 3 | https://arxiv.org/abs/2009.08034
12. TQT: MobileNet v1 71.1 to 71.1, v2 71.7 to 71.8, per-channel real-scale QAT 70.7 / 71.1, static 0.6, weights only 67.0, round-half-to-even | confirmed; quote corrected: "to prevent an overall upward or downward bias" -> "prevents an overall upward or downward bias" | https://arxiv.org/abs/1903.08066
13. Jacob et al.: ResNet-50/100/150 76.4 / 78.0 / 78.8 float and 74.9 / 76.6 / 76.7 integer, M0 in [0.5, 1), weights never -128, Appendix B quotes; HAWQ-V3 ResNet50 77.72 to 77.58 with INT32 residual addition and no integer division; "the reported damage from rounding bias is for inference that differs from the training simulation" | numbers and quotes confirmed; the statement about the setting is unsupported (the paper describes no experiment) and is now tagged [UNVERIFIED] in section 3 and in 6.2 Q5 | https://arxiv.org/abs/1712.05877 , https://arxiv.org/abs/2011.10680
14. Legacy CMSIS-NN: accumulator starts at (bias << bias_shift) + NN_ROUND(out_shift), NN_ROUND(s) = (1 << s) >> 1, __SSAT(sum >> out_shift, 8), ARM_NN_TRUNCATE; arm_softmax_q7 with `(1 << 20) / sum` and an 8-octave window; paper: CIFAR-10 80.3 to 79.9 | confirmed | https://github.com/ARM-software/CMSIS_5/blob/5.9.0/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7.c , https://github.com/ARM-software/CMSIS_5/blob/5.9.0/CMSIS/NN/Include/arm_nnsupportfunctions.h , https://github.com/ARM-software/CMSIS_5/blob/5.9.0/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_q7.c , https://arxiv.org/abs/1801.06601
15. TFLite softmax: Q5.26 exp input, Q12.19 sum, GetReciprocal; int16 variant with exp and 1/(1+x) tables and shift 31 - headroom_plus_one; LUT sizes 256 and 513; CMSIS-NN arm_softmax_s8 with CLZ, ONE_OVER1, MUL_SAT, DIV_POW2 | confirmed; quote corrected: "For int8/uint8 inputs a simple 256 entries LUT is used" -> "For uint8 and int8 inputs a simple 256 entries LUT is used"; the sentence on optimized_ops.h was not opened | https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/lite/kernels/internal/reference/softmax.h , https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/lite/kernels/internal/common.h , https://github.com/ARM-software/CMSIS-NN/blob/v6.0.0/Source/SoftmaxFunctions/arm_softmax_s8.c
16. Sigmoid attention: 1B LLM 49.5 against 49.4 (n = 2048), 48.2 against 49.2 and with hybrid norm 50.2 against 49.4 (n = 4096); smallest ablation d_model 64, 2 layers, reported as 2.2M non-embedding parameters; ReLU "pose instabilities and underperform" | confirmed | https://arxiv.org/abs/2409.04431
17. ConSmax: GPT with 6 layers, 6 heads, width 384, context 256 on WikiText-103; perplexity 2.3% higher at the start, under 0.9% after 10K iterations, converged after about 20K | confirmed | https://arxiv.org/abs/2402.10930
18. I-LLM: LLaMA-7B WikiText2 perplexity 5.68 / 5.84 / 9.10; window ablation 9.15 / 9.16 / 9.23 / 7,360,945; "can lead to catastrophic outcomes" | confirmed; corrected in 6.2 Q1: "no loss from 20 down to 10 nats" -> "almost no loss (9.15 to 9.23 at W4A4)" | https://arxiv.org/abs/2405.17849
19. Kang and Wang: "perplexity 5.4955 with their compact exp tables against 5.4943"; "a 34-entry table for the exponent part"; 5.4956 against 5.5010; 3 to 6% scale variation | corrected: 5.4955 -> 5.4946 (their Table 3; 5.4955 is the uncompensated configuration with 16-bit entries in their Table 2) and 34-entry -> 32 entries in the text (34 x 21 bits in their Table 4); the other numbers confirmed; whole division-free softmax 5.4971 against 5.4940 added | https://pmc.ncbi.nlm.nih.gov/articles/PMC12843610/
20. Accumulators: Softermax (SQuAD 86.28 to 85.86, under 0.5%, Q(6,2) input, 24-bit accumulation) and ITA (mean absolute error 0.46% against 0.35%, 24 bits "to allow up to 256-element dot products"); A2Q condition sum|x||w| <= 2^(P-1) - 1 and its L1 bound | confirmed; added that 86.28 is an 8-bit quantised baseline; corrected in section 4: "both for dot products up to 256 terms" -> only ITA states that limit; corrected in section 3: ITA reports no task score | https://arxiv.org/abs/2103.09301 , https://arxiv.org/abs/2307.03493 , https://arxiv.org/abs/2308.13504
21. Post-training softmax and norm replacements: FQ-ViT 23.64 / 83.31 / 84.53 / 82.68 and DeiT-S 79.85 / 79.17 / 78.40; SOLE 72.21 to 71.33, 79.85 to 79.27, 88.17 to 87.57; Vasyltsov and Chang BLEU 26.98 / 26.76 / 26.80 / 26.26 / 26.68 / 24.42 / 25.29; NN-LUT 62.1 to 4.6, 87.5 to 57.5, 84.1 to 85.1 (float 85.4) | confirmed | https://arxiv.org/abs/2111.13824 , https://arxiv.org/abs/2510.17189 , https://arxiv.org/abs/2111.10770 , https://arxiv.org/abs/2112.02191
22. llama2.c64: unchanged float32 stories260K weights, RAM expansion of at least 2 MB, about one token every 8 minutes, no licence file in the repository root | confirmed | https://github.com/ytmytm/llama2.c64
23. Licence column of the sources table (19 GitHub repositories, 2 model cards, 1 article) | confirmed; Deeploy corrected: "[UNVERIFIED: licence id not confirmed]" -> Apache-2.0 for most source files and MIT for some scripts (README) | https://api.github.com/repos/OWNER/REPO and .../license for each repository , https://huggingface.co/api/models/Maykeye/TinyLLama-v0 , https://huggingface.co/api/models/karpathy/tinyllamas
24. Section 4 and section 1: "integer-only language model below 1M parameters with reported accuracy: not found", "the closest are Deeploy and llama2.c64", "language model with an int8 residual stream: not found", "the literature uses 16 to 32 bits" | corrected: not found -> found outside the papers: elya-nes (ternary weights, 4-bit activations, loss reported), fpga-tinystories on a MAX 10 (int8 residual stream over 16 layers, 0.622 to 0.643 bits per character), ecp5-inference (int8 with power-of-two scales, no loss), Soul Player C64 (int8 weights, power-of-two shifts, no loss) | https://github.com/Scottcjn/elya-nes , https://github.com/MRazvan/fpga-tinystories , https://github.com/MRazvan/fpga-tinystories/blob/main/ref/int_model.py , https://github.com/skyline624/ecp5-inference , https://github.com/skyline624/ecp5-inference/blob/main/host/v4_quant.py , https://github.com/gizmo64k/soulplayer-c64
25. [computed here] numbers: 511 / 516 / 520 terms, sum abs(w) <= 65,535, -0.4965 / +0.0043 LSB, -5.50 LSB after 11 roundings, 0.80 LSB rms and +0.04 mean (1.63 and -1.37 with floor), 1/sqrt errors 0.85% / 0.56% / 1.5% / 5.7%, 30.4 / 40.9 / 66.6 dB, total variation 1.3 / 4.2 / 2.7% | confirmed by running the script again; corrected in section 1: "19% rms for a bare power of two" -> 29% rms bare, 19% rms after the constant 1.337 | research/02_integer_inference_checks.py

Not re-checked: HAAN, ReLU attention in vision transformers, the binarised-network quote, the SOLE figures in 6.2 Q2,
the sentence on `optimized_ops.h`, and the cost model behind the [estimate] cycle counts (their arithmetic follows from
`research/CONTEXT.md`, which marks its numbers as an estimate to be refined). Still [UNVERIFIED]: the DATE 2026 paper
and the ESP32 snippets in section 4, and the reading of the Jacob et al. remark on rounding bias.
