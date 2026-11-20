# 05 Speed techniques for tiny transformers (LS-LLM research, 2026-11-13)

Labels used below. [paper]: number copied from the cited source, which was opened and read. [cost model]: our own
arithmetic with the CONTEXT.md constants (a row of n multiply-adds costs n + 24 cycles, plus 4 for every extra `MAC`
instruction when n > 255; multiply-type element-wise operation 24 cycles; add, lookup or copy 12 cycles; softmax 24
cycles per score when done in two passes, 4 when the table lookup is fused into the score loop; 100 MHz). These are
estimates, not measurements. [measured here]: small local measurement made for this report (tokenizer compression,
benchmark byte lengths; no model was run). With these constants the baseline costs 1.10 M cycles per token at
T = 256 (CONTEXT.md says about 1.0 M; the gap is the two-pass softmax assumption), so read percentages, not
absolute tokens per second.

Fact-check tags (added on 2026-11-13 by an independent re-check). [VERIFIED]: the numbers of that bullet were
compared again with the cited source, or re-computed or re-measured, and agree; what was compared is listed in the
Verification log at the end. [UNVERIFIED: reason]: could not be supported as written (none is left after the
re-check). Bullets and table rows without a tag were not re-checked; the log ends with their list.

## 1. Answer in one paragraph

On the LS CPU the money is in rows and element-wise passes, not in multiply-adds: at T = 256 the baseline spends
56% of every layer on attention scores and the softmax pass (8 heads x 256 dot products of only 8 multiply-adds
each, every one paying 24 cycles of control, then an exp pass) and 23% on the SwiGLU block; the output layer is 5%
of a token [cost model]. The techniques that pay are therefore the ones that delete rows: fewer and wider heads
(8 x 8 -> 2 x 32 is -34% cycles per token at T = 256 and -43% at T = 512), an exp lookup fused into the score loop (-19%), small
attention windows in the lower layers with full context only in the top one or two (-32% to -42%), fewer but wider
layers (d = 128 with 2 layers is -25% to -41% at 361k parameters), and skipping the last layer and the output layer
for prompt tokens (exact, -17% end to end on a median HumanEval-sized task) [all cost model]. Grouped-query and
multi-query attention barely change time (MQA -2%) but free 60 to 120 KiB of KV cache; tied embeddings save 32.8k
parameters and no time; layer sharing saves memory and no time. Vocabulary must be judged per byte, not per token:
we measured 2.1 bytes per token at vocabulary 256, 2.5 to 2.7 at 512 and 2.8 to 3.2 at 1024 on MBPP-style Python
(an independent re-run gave 2.0 to 2.1, 2.5 to 2.7 and 2.7 to 3.1, see 6.4),
so with attention dominating, 256 is about 24% slower per task than 512, 1024 is about 8% faster, and byte level is
about 4 times slower [measured here + cost model]; and a HumanEval prompt plus reference solution has a median of
574 bytes, so T = 256 holds only about 62% of the tasks at 2.5 bytes per token while T = 512 holds 95%
(57% to 71% at T = 256 and 94.5% to 97.6% at T = 512 if the real figure lies between 2.4 and 2.7 bytes per token)
[measured here]. Published quality evidence sits almost entirely at 65 M to 8 B parameters: going from 8 to 4 heads
costs 0.01 to 0.02 nats and from 8 to 2 heads 0.04 to 0.05 nats (TinyStories, width 768) or 1.6% perplexity (8 to 4
heads, 65 M Transformer), going to one head costs 7.5% perplexity, MQA costs 0.2 BLEU (211 M) or 0.6 to 1.7 accuracy
points (T5-XXL; 125 M to 350 M), ReLU instead of SwiGLU costs 0.04 log-perplexity (T5-base) or 1.3 points (125 M);
below 10 M parameters we found only the TinyStories depth/width table (its head table is at width 768, 15 to 22 M by
our arithmetic), tying on a 2.65 M LSTM, minGRU on character-level Shakespeare (about 5 M by our arithmetic) and one
hobby model, and nothing at 260k to 400k int8 parameters.
Constant-cost recurrent mixers (minGRU, SRU, GRU) would bring a token to about 0.40 to 0.62 M cycles at any context
length with no KV cache [cost model] and match a Transformer on Shakespeare at about 5 M parameters (1.548 vs 1.547
test loss), but there is no evidence for code at our size and an int8 recurrent state is a known quantization risk
(production integer LSTMs keep the cell state at 16 bits); Mamba, xLSTM and RWKV-4 fit this CPU badly because
their work is element-wise or needs division. Early exit and output-layer shortcuts are low value here (5 layers,
a 512-row output layer worth 5% of a token) and early exit without verification collapses on code (LayerSkip:
HumanEval 15.9 at the last layer, 4.88 at the middle layer). Greedy decoding is the cheapest decoder and close to
what Codex found optimal for pass@1 (temperature 0.2).
[VERIFIED: the cost-model figures of this paragraph were re-computed from the stated constants and reproduce; the
literature numbers were re-read at their sources; the bytes-per-token figures reproduce within 3%. Corrected here by
the re-check: the costs of fewer heads and of MQA are now given per comparison, and the TinyStories head table is no
longer counted as evidence below 10 M.]

## 2. Proven (numbers and sources)

Number of heads and head dimension
- 65 M base Transformer, WMT14 EN-DE dev, heads varied at constant total attention width (Table 3 rows A): 1 head
  PPL 5.29 / BLEU 24.9; 4 heads 5.00 / 25.5; 8 heads 4.92 / 25.8; 16 heads 4.91 / 25.8; 32 heads (head dim 16)
  5.01 / 25.4. Rows B: key dimension cut to 16 with 8 heads gives 5.16 / 25.1. [paper] [VERIFIED]
  https://arxiv.org/abs/1706.03762
- TinyStories, hidden size 768 (Figure 24): 2 layers with 2 / 4 / 8 heads -> eval loss 1.38 / 1.34 / 1.33; 1 layer
  -> 1.58 / 1.56 / 1.54. [paper] [VERIFIED] https://arxiv.org/abs/2305.07759
- Trained 12- and 16-head models: only 8 of 96 encoder heads change BLEU significantly when removed; most layers
  can be cut to one head at test time; 20% (WMT) and 40% (BERT) of heads can be pruned without noticeable loss,
  then quality drops sharply. BERT inference +17.5% faster with 50% of heads pruned. [paper] https://arxiv.org/abs/1905.10650
- MobileLLM Table 13 (125 M, 8 layers, dim 896, equal KV heads): 8 / 16 / 32 heads -> average accuracy 43.8 / 44.6 /
  44.3; 350 M: 49.3 / 49.6 / 48.6. [paper] [VERIFIED] https://arxiv.org/abs/2402.14905
- One hobby data point below 1 M: 818k-parameter character model (4 layers, dim 128, block 64, Tiny Shakespeare,
  2000 steps, one seed): 4 heads val loss 1.93, 1 head 1.91. [model card] [VERIFIED]
  https://huggingface.co/delcenjo/char-transformer-shakespeare

Grouped-query and multi-query attention
- WMT14 EN-DE, 211 M, equal parameters: multi-head ln PPL 1.424 / BLEU 26.7; multi-query (one shared K/V head)
  1.439 / 26.5. Shrinking heads instead, to the same K/V size, is worse: 1 head 1.518 / 25.8; 2 heads 1.480 / 26.2;
  4 heads 1.488 / 26.1; 8 narrow heads 1.513 / 25.8. Billion-word LM, 192 M: multi-head 29.9 PPL, multi-query 30.2,
  shrunken heads 30.9 to 31.2. Decoder step time 46 -> 3.8 microseconds per token on TPUv2 (a memory-bandwidth
  effect). [paper] [VERIFIED] https://arxiv.org/abs/1911.02150
- T5-XXL, 5% uptraining: MHA 47.2 average at 1.51 s per sample; MQA 46.6 at 0.24 s; GQA-8 47.1 at 0.28 s. [paper]
  [VERIFIED] https://arxiv.org/abs/2305.13245
- MobileLLM Table 13, 16 query heads with 16 / 8 / 4 / 2 / 1 KV heads: 44.6 / 44.3 / 44.7 / 43.5 / 43.7 (125 M) and
  49.6 / 48.5 / 49.4 / 47.5 / 47.9 (350 M). [paper] [VERIFIED] https://arxiv.org/abs/2402.14905
- stories260K (dim 64, 5 layers, 8 heads, 4 KV heads, vocab 512, context 512, tied embeddings) reaches validation
  loss 1.2968 on TinyStories: an existence proof at 260k parameters, with no ablation. [VERIFIED]
  https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K and https://github.com/karpathy/llama2.c

Vocabulary size, bytes per token, context length
- A 4096-token vocabulary trained on TinyStories gives about the same sequence length per example as the 32000
  Llama 2 vocabulary. [README] [VERIFIED] https://github.com/karpathy/llama2.c
- mT5 SentencePiece averages 4.1 bytes per token; byte-level ByT5 is 1.5 to 2.6 times slower at inference on
  transliteration, 3.7 to 6.4 times on XSum and 6.4 to 9.5 times on XNLI. [paper] [VERIFIED]
  https://arxiv.org/abs/2105.13626
- Source code, single-layer GRU (512 units), BPE with 2k / 5k / 10k merges, Java static entropy in bits per code
  token: 4.90 / 4.78 / 4.77 on the small training set, 3.59 / 3.35 / 3.15 on the large one. Small vocabularies are
  enough on little data; more data favours larger ones. Smallest vocabulary tested is 2k. [paper] [VERIFIED]
  https://arxiv.org/abs/2003.07914
- [measured here] SentencePiece BPE trained on the MBPP train split (task ids 601-974) and scored on the MBPP
  validation split (511-600), code only: vocab 256 -> 2.10 bytes per token; vocab 512 with llama2.c flags
  (`byte_fallback=True`, which spends 256 of the 512 entries on raw bytes) -> 2.04; vocab 512 without byte fallback
  -> 2.53; vocab 1024 -> 2.82. With the task description included: 2.18 / 2.12 / 2.74 / 3.18. Full table in section 6.
  [VERIFIED within 3%: an independent re-run with these settings gave 2.04 / 1.99 / 2.47 / 2.74 for code and 2.13 /
  2.08 / 2.69 / 3.13 with the description, in the same order and with the same ratios; the script of the original
  measurement is not in the repository, see section 3]
- [measured here] HumanEval (164 tasks): prompt mean 451 bytes (median 396, p90 775, max 1360); canonical solution
  mean 181 (median 148, p90 373); prompt plus solution median 574, p90 1053, p95 1254. MBPP (974 tasks): description
  mean 79 bytes, code mean 181 (median 146, p90 343). Byte lengths only; nothing was tuned or run on these sets.
  [VERIFIED: re-measured from `data/bench/HumanEval.jsonl.gz` and `data/bench/mbpp.jsonl`; means, medians and maxima
  are identical, p90 and p95 agree within 2 bytes]

Tied embeddings
- Penn Treebank small LSTM: tying input and output embeddings takes the model from 4.65 M to 2.65 M parameters and
  test perplexity from 114.5 to 112.4. [paper] [VERIFIED] https://arxiv.org/abs/1608.05859
- MobileLLM Table 1 (30 layers, dim 512): sharing removes 16 M parameters (11.8%) for 44.8 -> 44.6 average accuracy;
  spending part of the saving on 2 more layers gives 45.0 at 125 M. [paper] [VERIFIED]
  https://arxiv.org/abs/2402.14905
- ALBERT factorised embeddings, all layers shared: embedding size 64 / 128 / 256 / 768 -> 79.0 / 80.1 / 79.6 / 79.8
  average, so an embedding narrower than the model can cost nothing. [paper] https://arxiv.org/abs/1909.11942

Local and windowed attention
- Decoder self-attention limited to the current and previous 31 positions (WMT14 EN-DE, 211 M): ln PPL 1.427 vs
  1.424, BLEU 26.6 vs 26.7, decoder time 46 -> 23 microseconds per token. [paper] [VERIFIED]
  https://arxiv.org/abs/1911.02150
- Longformer, text8, 12 layers x 512 (41 M), same average window, dev bpc after 150k steps: windows growing from 32
  at the bottom to 512 at the top 1.21 bpc; fixed 230 everywhere 1.23; shrinking from 512 to 32 1.24. [paper]
  [VERIFIED] https://arxiv.org/abs/2004.05150
- The released TinyStories models use the "GPT-Neo architecture with window size 256 and context length 512"
  (footnote 2).
  [paper] [VERIFIED] https://arxiv.org/abs/2305.07759
- A 4-layer LSTM (51 to 54 M) on enwik8: no attention 1.330 bpc; one single-head attention layer 1.076; one head
  in every layer 1.068; 12-layer Transformer-XL (41 M, 160 heads) 1.06. One attention head in one layer recovers almost all
  of the gap. [paper] [VERIFIED] https://arxiv.org/abs/1911.11423
- Code is locally repetitive: n-gram entropy on Java falls from 5.54 to 2.57 bits per token with a nested cache of
  recent code, on Python from 4.81 to 2.89. [paper, Table 2, dynamic setting, small training set] [VERIFIED]
  https://arxiv.org/abs/2003.07914
- LSTM language models use about 200 tokens of context and are order-sensitive only within the nearest 50.
  [abstract only] [VERIFIED against the abstract] https://arxiv.org/abs/1805.04623

Early exit
- CALM (T5 with 8 decoder layers): an oracle that exits as soon as the top token agrees with the full model uses
  1.53 layers per token at ROUGE-L 38.24 vs 38.32; skipped layers must receive the copied hidden state (copying
  projected keys and values instead gives 23.02). Softmax-response confidence: 2.62 layers on average, 2.57 times
  faster on CNN/DM, but on CNN/DM and WMT it raises total arithmetic (FLOPs reduction factor 0.44 to 0.91, below 1);
  the speed-up is wall-clock time on a TPU, where the paper notes that the next layer can start in parallel.
  A linear classifier on the hidden state (d + 1 parameters): 4.51 layers, 2.04 times faster. [paper] [VERIFIED]
  https://arxiv.org/abs/2207.07061
- LayerSkip (Llama2 7B / 13B): HumanEval at the last layer 13.4 baseline and 15.9 with LayerSkip training; at the
  middle layer 0 and 4.88 (13B: 18.9 / 18.3 and 0 / 3.05). The next table of the paper goes the other way at the last
  layer: Llama3 8B scores 37.8 as baseline and 28.7 with LayerSkip training (middle layer 0 and 7.32), so the
  training itself can cost code accuracy. Exit at layer 6 of a 7B code model: ROUGE-2 0.0035 vs 0.0513. The reported
  1.34 to 2.16 times speed-ups (1.82 on HumanEval at 45% token acceptance) come from self-speculative decoding, where
  the remaining layers verify the draft. [paper, Tables 1, 2 and 5] [VERIFIED; the Llama3 row was added by the
  re-check] https://arxiv.org/abs/2404.16710
- DeeBERT (BERT-base classifiers, entropy threshold): SST-2 -0.6 accuracy at -40% time; MNLI -0.8 / -0.7 at -25%.
  [paper] https://arxiv.org/abs/2004.12993
- Early exit tuned for token-level agreement with the full model: 60 arithmetic word problems solved by the full
  model, 10 to 28 under early exit. [abstract only] [VERIFIED against the abstract; the paper is on arXiv, submitted
  2026-09-12] https://arxiv.org/abs/2609.14144

Depth, width, layer sharing
- TinyStories Figure 4, eval loss by hidden size and layers (2 / 4 / 8 / 12 layers): hidden 64: 2.38 / 2.26 / 2.08 /
  2.02; hidden 128: 1.92 / 1.78 / 1.65 / 1.62; hidden 256: 1.60 / 1.47 / 1.38 / 1.34; hidden 512: 1.39 / 1.27 / 1.20 /
  1.19; hidden 768: 1.31 / 1.20 / 1.18 / 1.18 and 1.54 with 1 layer. Consistency score (out of 10): 64 x 8 layers
  4.45, 128 x 2 layers 4.75, 128 x 4 layers 6.60. The authors: width matters more for knowledge, depth for
  context tracking; 1-layer models struggle to follow instructions. [paper; table read from the figure image at
  https://ar5iv.labs.arxiv.org/html/2305.07759] [VERIFIED against page 7 of the PDF] https://arxiv.org/abs/2305.07759
- MobileLLM Table 11 at 134 to 135 M parameters, except the 4-layer row at 163 M: 4 layers x 1280 -> 43.3; 12 x 768
  -> 43.9; 30 x 512 -> 44.8; 62 x 384 -> 44.7. [paper] [VERIFIED; corrected: the file said "at about 135 M" for all
  four rows] https://arxiv.org/abs/2402.14905
- Theory and experiments at depths 6 to 48, vocabulary 2000: depth helps only up to about log3(width) layers; "for
  small enough network sizes, deeper self-attention networks perform worse than shallow ones". [paper] https://arxiv.org/abs/2006.12467
- ALBERT-base, 12 layers, embedding 768: not shared 108 M -> 82.3; attention shared 83 M -> 81.6; feed-forward
  shared 57 M -> 79.5; all shared 31 M -> 79.8. Compute is unchanged. [paper] https://arxiv.org/abs/1909.11942
- MobileLLM Tables 2 and 14: running every block twice with the same weights, 125 M: 44.6 -> 45.0; 350 M: 49.6 ->
  50.2; three or four repeats add nothing clear (45.0 / 45.3; 49.4 / 50.4). Twice the compute, same memory. [paper]
  [VERIFIED] https://arxiv.org/abs/2402.14905

Feed-forward variants at equal parameters and multiply-adds
- T5-base, log-perplexity after 524k steps: ReLU 1.677, GELU 1.679, Swish 1.683, GLU 1.663, Bilinear (no
  nonlinearity) 1.648, ReGLU 1.645, SwiGLU 1.636, GEGLU 1.633; run-to-run spread 0.003 to 0.010 at 65k steps.
  Gated variants use hidden size 2/3 of the plain one. [paper] [VERIFIED] https://arxiv.org/abs/2002.05202
- MobileLLM 125 M: ReLU feed-forward 42.6 -> SwiGLU 43.9 average accuracy. [paper, Table 10] [VERIFIED]
  https://arxiv.org/abs/2402.14905

Attention-free and recurrent models
- minGRU: z = sigmoid(Linear(x)), h~ = Linear(x), h = (1 - z) * h_prev + z * h~; 2 matrices per layer, no dependence
  of the gates on the previous state. Character-level Shakespeare (nanoGPT setup, 3 layers, width 384, about 5 M
  parameters by our arithmetic; block = Conv4 -> minRNN -> MLP, expansion 2): test loss minGRU 1.548, minLSTM 1.555,
  Mamba 1.575, Transformer 1.547. [paper] [VERIFIED] https://arxiv.org/abs/2410.01201
- Hobby repeat on the same data: Mamba 1.463 validation loss against the 1.4697 quoted for nanoGPT. [README]
  [VERIFIED: the nanoGPT README gives 1.4697 as the lowest validation loss of its 6-layer, width-384 character model,
  https://github.com/karpathy/nanoGPT] https://github.com/apapiu/mamba_small_bench
- TinyStories, three 10.94 M models: Transformer perplexity 6.93, recurrent-only 7.26, recurrent plus attention
  5.95. [abstract only] [VERIFIED against the abstract; the paper is on arXiv, submitted 2026-10-29; its recurrent
  layer is called "Dynamic Wave"] https://arxiv.org/abs/2610.39854
- Smallest language models in the original papers: Mamba 130 M, RWKV 169 M, xLSTM 125 M. xLSTM Table 1 (about
  400 M, 15 B tokens): xLSTM 13.43 perplexity, Mamba 13.70, Llama 14.25, RWKV-4 15.62. [papers] [VERIFIED]
  https://arxiv.org/abs/2312.00752 https://arxiv.org/abs/2305.13048 https://arxiv.org/abs/2405.04517
- At 8 B parameters a stack with 7% attention layers, 43% Mamba-2 and 50% MLP layers beats the pure Transformer by
  2.65 points on 12 tasks, while pure Mamba lags on copying and in-context tasks. [abstract only] [VERIFIED against
  the abstract] https://arxiv.org/abs/2406.07887
- SRU (3 input matrices per layer, state recurrence purely element-wise): enwik8 1.19 bpc at 49 M against 1.30 for
  an LSTM at 46 M. QRNN (2 layers x 640, with zoneout): Penn Treebank 78.3 against 82.0 for the matching LSTM.
  [papers] [VERIFIED] https://arxiv.org/abs/1709.02755 https://arxiv.org/abs/1611.01576
- Integer-only LSTM in production use: 8-bit weights, but "16 bits is needed to preserve accuracy" for the cell
  state because it persists across steps. [paper] [VERIFIED] https://arxiv.org/abs/2101.05453

Output layer and decoding
- Text8, 44k-word vocabulary: full softmax 144 perplexity in 83 min; adaptive softmax 147 in 30 min; hierarchical
  softmax 166 (frequency tree) and 155 (similarity tree); the paper claims 2 to 10 times speed-up for vocabularies
  of 44k to 800k. [paper] https://arxiv.org/abs/1609.04309
- Decoding from a shortlist of candidate words is the standard trick for very large vocabularies. [abstract only]
  [VERIFIED against the abstract] https://arxiv.org/abs/1412.2007
- Codex, 679 M model: the optimal sampling temperature is 0.2 for pass@1 and 0.8 for pass@100. A 12 M Codex scores
  2.00% pass@1 on HumanEval. [paper] [VERIFIED] https://arxiv.org/abs/2107.03374
- llama2.c: temperature 0 takes the argmax of the logits with no softmax; top-p sampling sorts the vocabulary with
  qsort; the KV cache is (layer, position, kv_dim) with kv_dim = dim x n_kv_heads / n_heads. [code] [VERIFIED]
  https://github.com/karpathy/llama2.c/blob/master/run.c

## 3. Uncertain or contested

- Depth against width at small size. MobileLLM says deep and thin wins, but its models are 125 M and more. In
  TinyStories the 2-layer model of width 128 (loss 1.92) beats 8 layers of width 64 (2.08) with the same parameters
  in the layers, but it also has twice the embedding parameters (vocabulary 10k x width), so the comparison is not
  clean; and the same table shows 2-layer models weak on consistency. Levine et al. predict that depth stops
  helping beyond about log3(width) layers (3.8 for width 64, 4.4 for width 128; our extrapolation below their
  tested range). Which way this falls for Python at width 64 to 128 is unknown.
- Heads. Training with one head costs 0.9 BLEU at 65 M; pruning says most heads are redundant; the 818k hobby
  model shows no difference. The baseline head dimension of 8 is below anything we found tested (smallest: 16, and
  16 was worse than 32 to 64 at 65 M). Fewer, wider heads could even help at width 64. Not known.
- What MQA costs: 0.2 BLEU and 1% perplexity at 200 M (Shazeer), 0.6 points at T5-XXL, but 0.9 to 1.7 points at
  125 M to 350 M (MobileLLM; that table is not monotonic: with 16 heads at 350 M, going from 8 to 4 KV heads raises
  the score by 0.9 points, 48.5 -> 49.4, and at 125 M by 0.4, so the noise is of the same order).
  [VERIFIED; corrected: the file said "neighbouring rows move by up to 0.6 points"]
- Query heads against K/V capacity. Shazeer's data say sharing K/V is nearly free while cutting query heads hurts.
  On LS it is the query heads that cost cycles (heads x T rows), so the cheap direction for quality is the useless
  direction for speed.
- SwiGLU. Better at equal parameters in both sources, but on LS it needs 172 extra lookups and 172 extra
  multiplies per layer; a ReLU block with the same multiply-adds is 4% faster per token [cost model]. ReGLU sits
  0.009 log-perplexity behind SwiGLU at 524k steps (single runs; the spread reported at 65k steps is 0.003 to 0.010,
  so the gap is of the same order) and needs no table.
- Recurrent models at small size. Parity on Shakespeare is weak evidence (1 MB of text, every model overfits);
  the 10.94 M TinyStories study has the recurrent-only model behind; an LSTM without attention is far behind on
  enwik8 (1.33 vs 1.07 bpc). All large-scale claims of parity are at 125 M and above.
- Early exit. CALM is positive on fine-tuned summarisation and translation; LayerSkip and Kaplan show exact-answer
  and code tasks failing when layers are skipped without verification.
- Byte level against subword quality. ByT5 finds bytes competitive in small encoder-decoder models; the hobby model
  gets 3.63 nats per BPE-512 token against 1.93 per character, which favours BPE only if a token is about 2
  characters (its own estimate). No controlled result.
- Softmax replacements. Sigmoid attention matches softmax at 1 B (49.5 vs 49.4 average) but needs LayerScale,
  QK-norm and a bias of -log(n), and its speech runs with RoPE were the least stable; ReLU divided by sequence
  length approaches softmax in vision transformers only. https://arxiv.org/abs/2409.04431 https://arxiv.org/abs/2309.08586
- Our cost model. The constants are the CONTEXT.md initial estimates. The two-pass softmax (24 cycles per score)
  puts attention at 74% of a layer, against "about 60%" in CONTEXT.md. Rankings are robust to this; absolute
  tokens per second are not until the cycle-accurate simulator exists.
- Our tokenizer measurement. The in-domain tokenizers were trained on 68 to 98 KB and scored on 17 to 24 KB.
  Trained on 8.8 MB of standard-library code and applied to MBPP they give only 1.6 to 2.1 bytes per token, so
  compression depends strongly on matching the training corpus to the task style.
  Added by the re-check: the script of this measurement was not saved. An independent re-run (settings of 6.4 plus
  `allow_whitespace_only_pieces=True`; without that flag the values fall to 1.85 / 1.99 / 2.25 / 2.58) came out 1 to
  3% lower in every cell. The raw MBPP fields are 66.7 to 96.1 KB (train) and 16.4 to 23.4 KB (validation), about 2%
  less than the sizes above, so the original byte count probably included something beyond the raw fields; that
  alone would explain most of the gap. MBPP code also has Windows line endings (536
  carriage returns in the validation code) that a merge absorbs for free and that HumanEval does not have: counted
  without them the re-run gives 1.97 / 2.39 / 2.65 bytes per token at vocabulary 256 / 512 / 1024.
- Context fit. The 62% / 95% figures divide byte lengths by an assumed bytes per token; docstring English may
  compress differently from code. At 2.4 bytes per token (code without carriage returns) the shares are 57.3% and
  94.5%; at 2.7 (description plus code) they are 71.3% and 97.6%.

## 4. Not tried at our scale (260k to 400k int8 parameters, 8-bit multiply-accumulate CPU, 450 KB)

- Any ablation of heads, KV heads, feed-forward type, depth against width, windows or weight sharing at 260k to
  400k parameters. The smallest controlled ablations found are TinyStories (about 1 M and up, 10k vocabulary,
  float) and the 818k hobby model.
- Head dimension 8, as in the baseline.
- Any of these techniques under int8 weights and activations with power-of-two scales and a 24-bit accumulator.
  MobileLLM reports only 8-bit post-training quantization at 125 M and 350 M (within 0.5 points).
- Vocabulary 256 to 1024 for source code (smallest found in our search: 2k merges), and the quality effect of
  vocabulary size on a model this small.
- Local or per-layer windows for code generation where the task statement sits at the start of the context.
- Early exit in a decoder with 5 or fewer layers, or in any language model below 10 M parameters: not found in
  our search.
- Recurrent or hybrid language models below 1 M parameters on code, and any of them with an int8 recurrent state.
- Adaptive or hierarchical softmax at a vocabulary of 512; computing output rows only for grammar-legal tokens
  (not found in our search; the nearest idea found is skipping the model on grammar-forced tokens,
  https://michaelorenstein.com/blog/zero-entropy-tokens).
- A cost model where a row costs n + 24. Every published speed number above is for GPUs, TPUs or vector CPUs, where
  short rows are free and memory bandwidth dominates; on LS the opposite holds, so published speed-ups do not
  transfer and only the quality numbers do.

## 5. Ranked ideas for LS-LLM

Yardstick before the ideas: report speed three ways, (a) generated tokens per second at a stated context length,
(b) seconds per task including prompt processing, (c) bytes per second. In the baseline, prompt processing is 1.00 s
of the 1.56 s a median HumanEval-sized task takes, and tokens per second alone rewards a small vocabulary that is
slower per program [cost model, table 6.6]. All cycle figures below are cost-model estimates against the baseline
at T = 256 unless stated; "seed spread" means the loss range over at least 3 training seeds of the same shape.

| # | Idea | Expected effect on score / speed / memory | Cost to try | Risk | Keep-or-drop measurement | CPU change? |
|---|------|-------------------------------------------|-------------|------|--------------------------|-------------|
| 1 | Prefill skip: for every prompt token except the last, run the last layer only up to its K and V (norm, two projections, RoPE); skip its attention, feed-forward block and the output layer | Score: identical output. Speed: -23% cycles per prompt token, -17% seconds per median HumanEval-sized task (1.56 -> 1.30 s). Memory: 0 | Runtime code only | None for quality | Same tokens as the full run on the practice set; cycles per task in the simulator | No |
| 2 | Fewer, wider heads: 8 x 8 -> 4 x 16 -> 2 x 32, K/V width kept at 32 | Speed: -23% / -34% (T = 512: -28% / -43%). Score: +0.01 to +0.05 nats at larger scale, unknown here. Memory: 0 | 3 float runs plus seeds | Low to medium | Bits per byte and practice pass rate; keep the fewest heads within seed spread of 8 heads | No |
| 3 | One-pass softmax: exp table applied to the int8 score inside the score loop, no max search, one normalisation after mixing | Speed: -19% (T = 512: -23%). Score: unknown; needs training that uses the same table and clamps bit for bit | Part of the QAT design; 1 run against a two-pass reference | Medium: all weights can round to 0 when scores are low | QAT loss against float; count of positions with zero weight sum | Planned lookup instruction only |
| 4 | Per-layer windows: lower 3 or 4 layers see the last 32 tokens, the top 1 or 2 see everything | Speed: -32% (3 layers) to -42% (4 layers); T = 512: -42% to -57%. Memory: KV cache 80 -> 38 KiB. Score: indirect support only (window schedule at 41 M, one attention layer in a 52 M LSTM, window 32 in a 211 M translation decoder), unknown here | Attention mask in training, ring buffer in the runtime; 2 to 3 runs | Medium | As 2, plus pass rate split by prompt length | No |
| 5 | Wider, shallower shapes in the shape search: d = 96 x 3 layers, d = 128 x 2 layers, 2 heads | Speed: -36% to -41%. Parameters 354k to 361k, KV 64 to 72 KiB. Score: TinyStories has 128 x 2 at loss 1.92 against 2.26 for 64 x 4, but weak consistency scores | 4 to 6 float runs | Medium: context tracking; at T = 512 weights plus KV exceed 450 KB without windows or MQA | Position on the score against seconds-per-task front | No |
| 6 | Tokenizer: no byte-fallback pieces, pieces may span spaces; compare vocabulary 512 and 1024 per byte; do not use 256 or bytes | Speed: dropping byte fallback and letting pieces span spaces gives 19% fewer tokens at vocab 512 (2.04 -> 2.53 bytes per token; byte fallback alone 2.04 -> 2.37, 14%) at no model cost; 1024 is -8% seconds per task for +32.8k parameters; 256 is +24%. Score: unknown | Minutes per tokenizer; 2 to 3 runs | Low for the flags (non-ASCII becomes unknown tokens), medium for 1024 (12% of the parameter budget at d = 64, 18% at d = 128) | Bits per byte, never loss per token; seconds per task; practice pass rate | No |
| 7 | MQA or 2 KV heads, with the freed memory spent on context or width | Speed: -2%. Memory: -60 KiB at T = 256, -120 KiB at T = 512, which pays for T = 512 (95% of HumanEval tasks fit, against 62% at T = 256). Score: -0.2 BLEU at 211 M, -0.9 to -1.7 points at 125 to 350 M | 2 runs | Medium | Compare at equal total memory (weights + KV), not equal parameters | No |
| 8 | Feed-forward block: ReGLU, ReLU block of equal multiply-adds, hidden 128 instead of 172 | Speed: -1% / -4% / -6%, and more once attention is cheap (the block is 23% of a baseline layer). Score: +0.009 (ReGLU) and +0.041 (ReLU) log-perplexity at T5-base | 3 runs | Low | Bits per byte against cycles per token | No (ReLU is already in `RQ`) |
| 9 | Grammar shortcuts: when one token is legal, emit it and apply idea 1 to that step; when a subset is legal, compute only those output rows | Speed: -23% on forced tokens; at most -4.1% per generated token from skipped rows (the legality test must cost less than the 88-cycle row). Score: identical | Runtime plus grammar table support | Low; share of forced tokens unknown | Same tokens; cycles per task | No |
| 10 | Greedy decoding as the default: argmax fused into the output loop | Speed: about 4k cycles against about 10k for sampling and about 74k for sort-based top-p. Score: Codex optimum for pass@1 is temperature 0.2 | None | Low: repetition loops, handled by the grammar and a length cap | Practice pass@1, greedy against temperature 0.2, on the host | No (sampling needs the planned LFSR) |
| 11 | Hybrid: GRU-type or convolutional mixers in the lower layers, one single-head attention layer on top | Speed: about 537k cycles (-51%), almost flat in T. Memory: KV cache for one layer (32 KiB at T = 256). Score: supported at 52 M and 8 B, unknown here | An extra block type in training and runtime; 3 or more runs | High: int8 recurrent state, no small-scale evidence on code | Float bits per byte at equal parameters; then int8 state error against float over 256 steps | Lookup tables only |
| 12 | Fully recurrent model: minGRU or SRU mixer plus feed-forward block | Speed: 0.40 to 0.62 M cycles at any T (1.8 to 2.8 times the baseline rate at T = 256, 2.9 to 4.4 times at T = 512). Memory: no KV cache (80 to 160 KiB freed). Score: parity at 5 M on Shakespeare, behind at 10.94 M on TinyStories, unknown on code | 2 float runs decide whether to continue | High, as 11 | Same data, parameters and training tokens as the Transformer; keep only if it lands on the score against seconds-per-task front | Lookup tables only |
| 13 | Sliding window on all layers (W = 128), optionally anchored on the opening tokens of the prompt | Speed: -30% at T = 256, -57% at T = 512. Memory: KV halves. Score: the median HumanEval prompt (396 bytes, about 158 tokens) no longer fits in view; MBPP-sized tasks (median 292 bytes) do | 2 runs | High on HumanEval | Pass rate split by prompt length | No |
| 14 | Keep tied embeddings | Memory: -32,768 parameters (12.6% of 260k). Speed: 0. Score: neutral or better at 2.65 M | None | Low: one int8 scale must serve both roles | Already in the baseline | No |
| 15 | Early exit after layer 3 of 5 with a 1-row confidence classifier and hidden-state copy | Speed: -37% per exited token; check costs 88 cycles. Score: exact-answer and code tasks fail in the literature | Early-exit loss in training; 2 runs | High | Practice pass rate at equal seconds per task | No |
| 16 | Share attention projections across layers | Memory: -12,288 parameters per shared layer. Speed: 0. Score: -0.7 to +0.1 points in ALBERT | 1 run | Low | Only if memory binds before time | No |
| 17 | Adaptive or hierarchical softmax | Speed: at most 3 to 4% of a token. Score: +3 to +22 perplexity at a 44k vocabulary | Training change | Low gain | Skip | No |
| 18 | Hardware proposal, listed separately: one row instruction (clear ACC, stream n multiply-adds, requantise, store, advance, reload P0) | Speed: 24 -> 8 cycles per row is -21% on the baseline and stacks with ideas 2 to 5 | VHDL plus assembler | Hardware schedule | Simulator cycles | Yes |

Do not pursue on this CPU: Mamba (about 253k cycles per block at d = 64, of which 197k is the element-wise update
of 2 x 64 x 16 = 2,048 state values), xLSTM with matrix memory (64 x 64 element-wise update, about 197k cycles, plus
a division) and RWKV-4 (about 97k cycles per layer plus one division per channel) [rough cost model]. Their
advantage on GPUs is parallel element-wise work, which is the expensive kind here. Self-speculative decoding also
gives nothing on a sequential CPU, because verification redoes the skipped layers for every token.

Stacked estimate for ideas 2 + 3 + 4 on the baseline shape (2 heads x 32, one-pass softmax, lower 3 layers at window
32): 498k / 546k / 642k cycles per token at T = 128 / 256 / 512, that is 201 / 183 / 156 tokens per second against
130 / 91 / 57 for the baseline [cost model]. None of the three has quality evidence at our size.

[VERIFIED: the cycle counts, percentages, parameter counts, KV-cache sizes and tokens per second of this section were
re-computed independently from the constants in the header and reproduce; the rough per-block figures for Mamba,
xLSTM and RWKV-4 were reconstructed to within about 10%. They remain estimates: there is no cycle-accurate
simulation and no model run behind them. Corrected by the re-check: idea 4 had "supported at 41 M and
52 M" and idea 6 gave the whole 19% to byte fallback.]

## 6. Tables asked for in the topic brief

### 6.1 Technique, speed effect, quality effect, evidence below 10 M, risk, CPU change, source

Speed effects are cost-model estimates against the baseline at T = 256. Quality numbers are from the sources in
section 2; the scale of each is in brackets. "arXiv N" stands for https://arxiv.org/abs/N.
Fact-check: the speed column was re-computed and reproduces; the quality numbers were re-read at their sources,
except in the rows that rest on arXiv 1909.11942, 1609.04309, 2409.04431 and 2309.08586, which were not re-checked.

| Technique | Speed effect in our cost model | Quality effect (scale of evidence) | Evidence below 10 M? | Risk | CPU change? | Source |
|-----------|-------------------------------|------------------------------------|----------------------|------|-------------|--------|
| GQA: 4 KV heads instead of 8 (the baseline already has it) | -64 rows and -4,096 multiply-adds per layer, -3%; KV cache 160 -> 80 KiB | 44.6 -> 44.7 (125 M), 49.6 -> 49.4 (350 M); GQA-8 47.1 vs 47.2 (T5-XXL) | Existence only (stories260K) | Low | No | arXiv 2402.14905, 2305.13245 |
| MQA: 1 KV head | -48 rows and -3,072 multiply-adds per layer, -2%; KV cache 80 -> 20 KiB | ln PPL 1.424 -> 1.439, BLEU -0.2 (211 M); PPL 29.9 -> 30.2 (192 M); 47.2 -> 46.6 (T5-XXL); -0.9 / -1.7 points (125 / 350 M) | No | Medium | No | arXiv 1911.02150, 2305.13245, 2402.14905 |
| 4 heads x 16 instead of 8 x 8 | Score rows per token 10,240 -> 5,120; -23% | PPL 4.92 -> 5.00, BLEU -0.3 (65 M); loss +0.01 to +0.02 (TinyStories width 768, 15 to 22 M by our arithmetic) | Hobby 818k model only: 1 head 1.91 vs 4 heads 1.93 | Low to medium | No | arXiv 1706.03762, 2305.07759 |
| 2 heads x 32 | Score rows -> 2,560; -34% (T = 512: -43%) | Loss +0.04 to +0.05 for 2 vs 8 heads (TinyStories width 768) | As above | Medium | No | arXiv 2305.07759 |
| 1 head x 64 | Score rows -> 1,280; -37%; KV cache doubles to 160 KiB unless the key/value width is cut | PPL 4.92 -> 5.29, BLEU -0.9 (65 M) | As above | Medium | No | arXiv 1706.03762 |
| Vocabulary 256 (BPE) instead of 512 | -256 output rows, -2% per token; but 2.10 vs 2.53 bytes per token, so +24% seconds per task and 46% vs 62% of HumanEval tasks fit at T = 256 | No controlled number | No (compression measured here) | Medium | No | measured here |
| Vocabulary 1024 instead of 512 | +512 output rows, +4% per token; 2.82 vs 2.53 bytes per token, so -8% seconds per task; +32.8k parameters | Code, GRU: larger vocabularies win on large corpora (3.59 -> 3.15 bits from 2k to 10k), tie on small ones | No | Medium | No | measured here; arXiv 2003.07914 |
| Byte or character level | -256 output rows but 1 byte per token: 3.9 times the seconds per task; 7% of HumanEval tasks fit at T = 256, 39% at T = 512 | ByT5 competitive in quality, 1.5 to 9.5 times slower at inference | Hobby 818k model only | High (speed) | No | arXiv 2105.13626; measured here |
| Tied embeddings | 0 cycles; -32,768 parameters | PPL 114.5 -> 112.4 (2.65 M LSTM); 44.8 -> 44.6 (125 M) | Yes (2.65 M) | Low | No | arXiv 1608.05859, 2402.14905 |
| Sliding window on all layers, W = 128 | Context-dependent cost capped at W: 0% for T <= 128, -30% at T = 256, -57% at T = 512; KV cache halves | Window 32 in a translation decoder: BLEU 26.7 -> 26.6 (211 M); the released TinyStories models use window 256 | No ablation | High for prompts longer than the window | No | arXiv 1911.02150, 2305.07759 |
| Per-layer windows, 32 in the lower 3 to 4 layers | -32% to -42%; KV cache 80 -> 38 KiB | Growing windows 1.21 bpc vs fixed 1.23 (41 M); one attention layer 1.076 vs 1.068 bpc for one per layer (52 M LSTM) | No | Medium | No | arXiv 2004.05150, 1911.11423 |
| Early exit (CALM-style) | -37% per exited token when leaving after layer 3 of 5; check 88 cycles (classifier) or 49k (softmax response) | ROUGE-L 38.24 vs 38.32 at 1.53 layers (oracle, T5 8 layers); HumanEval 15.9 -> 4.88 at the middle layer (7 B); 60 -> 10 to 28 problems solved | No | High | No | arXiv 2207.07061, 2404.16710, 2609.14144 |
| Self-speculative decoding (LayerSkip) | None: verification repeats the skipped layers | 1.34 to 2.16 times on GPUs | No | Not applicable | No | arXiv 2404.16710 |
| Wide and shallow: d = 128 x 2 layers | Fewer per-layer attention taxes, longer rows: -25% with 8 heads, -41% with 2 heads; 361k parameters | TinyStories loss 1.92 (128 x 2) vs 2.26 (64 x 4) and 2.08 (64 x 8); deep and thin wins at 125 M (44.8 vs 43.3) | Yes (TinyStories, about 1 to 10 M, embedding width confounded) | Medium | No | arXiv 2305.07759, 2402.14905, 2006.12467 |
| Layer sharing | 0 cycles; -45,440 parameters per shared layer | +0.4 / +0.6 points for running each block twice, at twice the compute (125 / 350 M); all layers shared 79.8 vs 82.3 at equal depth (ALBERT) | No | Low | No | arXiv 2402.14905, 1909.11942 |
| ReLU block instead of SwiGLU at equal multiply-adds | 86 fewer rows, no lookups, no gate multiplies per layer: -4% | log-PPL 1.636 -> 1.677 (T5-base); 43.9 -> 42.6 (125 M) | No | Low | No | arXiv 2002.05202, 2402.14905 |
| ReGLU or Bilinear instead of SwiGLU | No SiLU table: -1% | log-PPL 1.636 -> 1.645 / 1.648 (T5-base) | No | Low | No | arXiv 2002.05202 |
| minGRU / minLSTM | Mixer 16k to 41k cycles per layer against 156k for attention; token 0.40 to 0.62 M cycles at any T; no KV cache | Test loss 1.548 / 1.555 vs 1.547 for the Transformer (character Shakespeare, about 5 M) | Yes, one small dataset | High: int8 state, code untested | Lookup tables only | arXiv 2410.01201 |
| SRU | Mixer about 29k per layer; token about 0.47 M | enwik8 1.19 bpc vs 1.30 for an LSTM (46 to 49 M) | No | High | Lookup tables only | arXiv 1709.02755 |
| QRNN | About the cost of a GRU (rows of length 2d) | Penn Treebank 78.3 vs 82.0 for an LSTM (18 to 20 M) | No | High | Lookup tables only | arXiv 1611.01576 |
| GRU / LSTM | Mixer about 37k / 48k per layer; token about 0.50 M (GRU) | LSTM without attention 1.330 bpc vs 1.076 with one attention head (enwik8, 52 M) | No | High | Lookup tables only | arXiv 1911.11423 |
| RWKV-4 | About 97k per layer plus 64 divisions | On par with Transformers at 169 M to 14 B (authors); 15.62 vs 14.25 PPL in the xLSTM comparison | No | High | Divider or a reciprocal routine | arXiv 2305.13048, 2405.04517 |
| xLSTM (matrix memory) | 4,096-element memory update, about 197k per layer, plus a division | 13.43 vs 14.25 PPL (about 400 M) | No | High | Divider or a reciprocal routine | arXiv 2405.04517 |
| Mamba | About 253k per block | 1.575 vs 1.547 (character Shakespeare, about 5 M); matches Transformers from 130 M | Yes, one small dataset | High | No, but slow | arXiv 2312.00752, 2410.01201 |
| Hybrid: recurrent layers plus one attention layer | 4 GRU layers + 1 single-head attention layer about 537k, -51% | 1.076 vs 1.068 bpc (52 M); 5.95 vs 6.93 PPL (10.94 M, abstract only); +2.65 points (8 B) | Near (10.94 M) | High | Lookup tables only | arXiv 1911.11423, 2610.39854, 2406.07887 |
| Adaptive softmax | Output rows 512 -> about 130 plus a tail on demand: at most 3% of a token | PPL 144 -> 147 (44k vocabulary) | No | Low gain | No | arXiv 1609.04309 |
| Hierarchical softmax | About 2 x sqrt(V) rows: at most 4% of a token | PPL 144 -> 155 to 166 (44k vocabulary) | No | Low gain | No | arXiv 1609.04309 |
| Shortlist, or output rows only for grammar-legal tokens | -88 cycles per skipped token: at most -4.1% (-5.9% at T = 128) | Exact when the list comes from the grammar | Not found in our search | Low | No | arXiv 1412.2007 (shortlists) |
| Forced-token skip | -23% on tokens the grammar forces | Exact | Not found in our search | Low | No | michaelorenstein.com/blog/zero-entropy-tokens |
| Prefill skip | -23% per prompt token, -17% per task | Exact | Not applicable | None | No | our analysis |
| KV-cache layout | Keys time-major (the weight pointer walks one head's keys without resets), values dimension-major (each mixing output is one T-long stream); a time-major value cache would need a strided pointer; with attention weights in 0..127 one accumulation is overflow-safe up to 516 products (516 x 127 x 128 < 2^23), so a 512-token context just fits | None | Not applicable | None | No (yes for time-major values) | our analysis; layout in llama2.c run.c |
| Greedy against sampling | Argmax in the output loop about 4k cycles; exp, sum and scan about 10k; sort-based top-p about 74k (7% of a token) | pass@1 optimum at temperature 0.2 (679 M) | No | Low | LFSR for sampling (planned) | arXiv 2107.03374; llama2.c run.c |
| One-pass softmax | -19% (T = 512: -23%) | None published for this integer form | No | Medium | Planned lookup instruction | our analysis |
| Sigmoid attention | Same as one-pass softmax plus no normalisation: about -20% | 49.5 vs 49.4 average (1 B); needs LayerScale, QK-norm, bias -log(n) | No | Medium to high | Planned lookup instruction | arXiv 2409.04431 |
| ReLU attention | No table at all (ReLU is in `RQ`): about -23% | Approaches softmax in vision transformers when divided by sequence length; no language-model evidence | No | High | No | arXiv 2309.08586 |
| Row overhead 24 -> 8 cycles | -21% | None | Not applicable | Hardware work | Yes | our analysis |

### 6.2 Baseline cost breakdown (cost model, one layer, T = 256)

| Part | Cycles | Share of layer |
|------|--------|----------------|
| Attention scores: 8 heads x 256 rows of 8 multiply-adds | 65,536 | 31% |
| Softmax pass (two-pass, 24 cycles per score, plus sums) | 51,424 | 24% |
| Value mixing: 64 rows of 256 multiply-adds, then normalise | 19,808 | 9% |
| q, k, v, o projections: 192 rows | 16,896 | 8% |
| RoPE on q and k | 2,496 | 1% |
| SwiGLU block: 408 rows, 172 lookups, 172 gate multiplies | 49,008 | 23% |
| Two RMSNorms | 3,272 | 2% |
| Two residual adds | 1,536 | 1% |
| Layer total | 209,976 | 100% |

Token total: 5 x 209,976 + 50,788 (final norm, 512 output rows, argmax) = 1,100,668 cycles, 91 tokens per second.
Multiply-adds per token: 424,832. Rows per token: 14,123, of which 10,240 are attention-score rows.
Baseline against context length: 600k cycles at T = 64 (167 tokens/s), 766k at 128 (130), 1,101k at 256 (91),
1,433k at 384 (70), 1,768k at 512 (57).
[VERIFIED: every cell of the table and all totals on these lines were re-computed and are exact.]

### 6.3 Cost-model effect of single changes and of shapes (cycles per token, change against the baseline)

| Configuration | T = 128 | T = 256 | T = 512 | Parameters | KV cache at T = 256 |
|---------------|---------|---------|---------|------------|---------------------|
| Baseline: d 64, 5 layers, 8 heads x 8, 4 KV heads, hidden 172, vocab 512 | 766k | 1,101k | 1,768k | 260k | 80 KiB |
| 4 heads x 16 | 640k (-16%) | 849k (-23%) | 1,265k (-28%) | 260k | 80 KiB |
| 2 heads x 32 | 577k (-25%) | 723k (-34%) | 1,014k (-43%) | 260k | 80 KiB |
| 1 head x 64 | 578k (-25%) | 693k (-37%) | 920k (-48%) | 281k | 160 KiB |
| 1 head, attention width 32 | 469k (-39%) | 542k (-51%) | 687k (-61%) | 240k | 80 KiB |
| MQA (1 KV head) | 742k (-3%) | 1,076k (-2%) | 1,743k (-1%) | 245k | 20 KiB |
| 8 KV heads | 799k (+4%) | 1,133k (+3%) | 1,800k (+2%) | 281k | 160 KiB |
| Window 128 on all layers | 766k (0%) | 766k (-30%) | 766k (-57%) | 260k | 40 KiB |
| Window 64 on all layers | 600k (-22%) | 600k (-45%) | 600k (-66%) | 260k | 20 KiB |
| Window 32 in the lower 3 layers | 617k (-20%) | 750k (-32%) | 1,017k (-42%) | 260k | 38 KiB |
| Window 32 in the lower 4 layers | 567k (-26%) | 634k (-42%) | 767k (-57%) | 260k | 24 KiB |
| One-pass softmax | 664k (-13%) | 896k (-19%) | 1,358k (-23%) | 260k | 80 KiB |
| ReLU block, hidden 258 | 726k (-5%) | 1,061k (-4%) | 1,728k (-2%) | 260k | 80 KiB |
| ReGLU, hidden 172 | 756k (-1%) | 1,090k (-1%) | 1,757k (-1%) | 260k | 80 KiB |
| Vocabulary 256 (per token) | 742k (-3%) | 1,076k (-2%) | 1,743k (-1%) | 244k | 80 KiB |
| Vocabulary 1024 (per token) | 816k (+6%) | 1,150k (+4%) | 1,817k (+3%) | 293k | 80 KiB |
| Row overhead 8 instead of 24 (CPU change) | 615k (-20%) | 867k (-21%) | 1,370k (-22%) | 260k | 80 KiB |
| 2 heads x 32 + one-pass + window 32 in the lower 3 layers | 498k (-35%) | 546k (-50%) | 642k (-64%) | 260k | 38 KiB |
| d 64, 4 layers | 623k (-19%) | 891k (-19%) | 1,424k (-19%) | 215k | 64 KiB |
| d 64, 8 layers | 1,196k (+56%) | 1,731k (+57%) | 2,798k (+58%) | 396k | 128 KiB |
| d 96, 3 layers, 2 heads x 48, hidden 256, one-pass | 608k (-21%) | 705k (-36%) | 899k (-49%) | 354k | 72 KiB |
| d 128, 2 layers, 8 heads x 16, hidden 256 | 656k (-14%) | 823k (-25%) | 1,156k (-35%) | 361k | 64 KiB |
| d 128, 2 layers, 2 heads x 64, hidden 256, one-pass | 571k (-26%) | 652k (-41%) | 814k (-54%) | 361k | 64 KiB |
| 5 layers of minGRU (expansion 1) + SwiGLU block | 399k | 399k | 399k | not sized | 0 |
| 5 layers of Conv4 + minGRU (expansion 2) + SwiGLU block | 618k | 618k | 618k | not sized | 0 |
| 5 layers of SRU + SwiGLU block | 466k | 466k | 466k | not sized | 0 |
| 5 layers of GRU + SwiGLU block | 504k | 504k | 504k | not sized | 0 |
| 4 GRU layers + 1 attention layer (1 head x 64, one-pass) | not computed | 537k (-51%) | not computed | not sized | 32 KiB |

Per-layer cost of the sequence mixer alone (d = 64): attention 39k cycles at T = 32, 56k at 64, 89k at 128, 156k at
256, 290k at 512; minGRU 16k; minGRU with expansion 2 and down-projection 41k; dense 4-tap convolution 18k; SRU 29k;
GRU 37k; LSTM 48k. Eight-head attention is cheaper than a GRU only below about 27 tokens of context and is never
cheaper than a bare minGRU.
[VERIFIED: all rows were re-computed. The Transformer rows are exact (the one-pass rows keep the per-head sum row).
The recurrent rows follow exactly from the per-block costs quoted above; a reconstruction of each block from the
header constants gives minGRU 15.9k, minGRU with expansion 41.5k, Conv4 18.2k, GRU 36.9k, LSTM 48.1k (all within 2%)
and SRU 25k to 31k depending on how its two state products are scheduled (the file uses 29k).]

### 6.4 Bytes per token for small vocabularies on Python [measured here]

SentencePiece 0.2.3, BPE, `character_coverage=1.0`, identity normalisation, `remove_extra_whitespaces=False`,
`split_digits=True`, newline and tab mapped to placeholder characters, ASCII files only. "Standard library" is 8.84 MB
of Python 3.12 standard-library source with 0.83 MB held out. MBPP train is task ids 601-974, MBPP validation is
511-600 (90 tasks, 17 KB of code, 24 KB with descriptions). The MBPP test split and HumanEval were not tokenized.

| Tokenizer | Trained on standard library, scored on its held-out part | Same, scored on MBPP validation code | Trained on MBPP train code, scored on MBPP validation code | Trained and scored on MBPP description + code |
|-----------|------|------|------|------|
| Byte level | 1.00 | 1.00 | 1.00 | 1.00 |
| Vocab 256, no byte fallback, pieces may span spaces | 1.91 | 1.65 | 2.10 | 2.18 |
| Vocab 512, `byte_fallback=True` (llama2.c flags; 256 entries are raw bytes) | 1.89 | 1.60 | 2.04 | 2.12 |
| Vocab 512, no byte fallback, pieces end at spaces | 2.27 | 1.91 | 2.37 | not run |
| Vocab 512, no byte fallback, pieces may span spaces | 2.36 | 1.90 | 2.53 | 2.74 |
| Vocab 1024, no byte fallback, pieces may span spaces | 2.80 | 2.12 | 2.82 | 3.18 |

For the standard-library tokenizers without byte fallback, the 259 tab characters in MBPP validation are unknown
(the standard-library training text has none) and come out as 138 unknown tokens, because a run of tabs collapses
into one; this flatters that column slightly.

[VERIFIED within 3%, columns 3 and 4 only: an independent re-run (SentencePiece 0.2.3, the flags above plus
`allow_whitespace_only_pieces=True`, bytes counted on the raw `code` field) gave, in table order, 2.04 / 1.99 / 2.32 /
2.47 / 2.74 for column 3 and 2.13 / 2.08 / not run / 2.69 / 3.13 for column 4. The two standard-library columns were
not re-run.]

### 6.5 Benchmark sizes and how much context they need [measured here: byte lengths only]

| Set | Field | Mean | Median | p90 | p95 | Max |
|-----|-------|------|--------|-----|-----|-----|
| HumanEval (164) | Prompt, bytes | 451 | 396 | 775 | 926 | 1,360 |
| HumanEval | Canonical solution, bytes | 181 | 148 | 373 | 437 | 864 |
| HumanEval | Prompt + solution, bytes | 632 | 574 | 1,053 | 1,254 | 1,993 |
| MBPP (974) | Description, bytes | 79 | 77 | 103 | 118 | 249 |
| MBPP | Code, bytes | 181 | 146 | 343 | 416 | 1,331 |
| MBPP | Description + one test + code, bytes | 323 | 292 | 487 | 558 | 3,390 |

Share of HumanEval tasks whose prompt plus canonical solution fits in T tokens (bytes divided by bytes per token):

| Bytes per token | T = 128 | T = 256 | T = 384 | T = 512 | T = 1024 | Prompt alone fits in 256 |
|-----------------|---------|---------|---------|---------|----------|--------------------------|
| 1.0 (bytes) | 0% | 7% | 21% | 39% | 89% | 19% |
| 1.9 | 5% | 37% | 73% | 87% | 99% | 70% |
| 2.1 (vocab 256) | 9% | 46% | 77% | 91% | 100% | 73% |
| 2.5 (vocab 512) | 13% | 62% | 87% | 95% | 100% | 83% |
| 2.8 (vocab 1024) | 18% | 73% | 91% | 98% | 100% | 87% |

[VERIFIED: both tables were re-measured from the local benchmark files. The fit table is identical. In the size table
means, medians and maxima are identical and p90 / p95 agree within 2 bytes; the MBPP "description + one test + code"
row includes two separator bytes.]

### 6.6 Time for one median HumanEval-sized task: prompt 396 bytes, solution 148 bytes [cost model]

| Configuration | Prompt tokens | Generated tokens | Prompt processing | Generation | Total | Generated tokens/s |
|---------------|---------------|------------------|-------------------|------------|-------|--------------------|
| Baseline, vocab 512 at 2.5 bytes per token | 158 | 59 | 1.00 s | 0.55 s | 1.56 s | 107 |
| Baseline + prefill skip | 158 | 59 | 0.75 s | 0.55 s | 1.30 s | 107 |
| Baseline, vocab 256 at 2.1 | 189 | 70 | 1.23 s | 0.70 s | 1.93 s | 99 |
| Baseline, vocab 1024 at 2.8 | 141 | 53 | 0.93 s | 0.50 s | 1.43 s | 107 |
| Baseline, byte level | 396 | 148 | 3.65 s | 2.43 s | 6.08 s | 61 |
| 2 heads x 32 + one-pass softmax + prefill skip | 158 | 59 | 0.58 s | 0.36 s | 0.95 s | 162 |
| d 128, 2 layers, 2 heads x 64, one-pass + prefill skip | 158 | 59 | 0.40 s | 0.36 s | 0.76 s | 162 |

[VERIFIED: re-computed with the context growing by one token per step. The totals are identical; single cells differ
by at most 0.01 s and 3 tokens per second.]

### 6.7 KV-cache size: layers x positions x 2 x kv_dim bytes

| Configuration | T = 256 | T = 512 |
|---------------|---------|---------|
| Baseline: 5 layers, kv_dim 32 | 80 KiB | 160 KiB |
| MQA: kv_dim 8 | 20 KiB | 40 KiB |
| 8 KV heads, or 1 head x 64: kv_dim 64 | 160 KiB | 320 KiB |
| d 128, 2 layers, kv_dim 64 | 64 KiB | 128 KiB |
| Window 32 in the lower 3 layers, full context in the top 2, kv_dim 32 | 38 KiB | 70 KiB |
| Window 128 on all layers | 40 KiB | 40 KiB |
| One attention layer, kv_dim 64 (hybrid) | 32 KiB | 64 KiB |
| Recurrent model (state only) | under 1 KiB | under 1 KiB |

At T = 512 the baseline cache is 36% of the 450 KB budget. Layout that needs no strided pointer: keys stored
position by position (the query in X memory is re-read for each row, the weight pointer runs straight through the
keys), values stored dimension by dimension (the attention weights sit in X memory and each of the 64 outputs is one
T-long stream). A window is a ring buffer: softmax and mixing do not care about slot order. Storing a value
dimension by dimension costs kv_dim scattered stores per layer per token (about 400 cycles).
[VERIFIED: every cell of the table follows from layers x positions x 2 x kv_dim bytes and was re-computed.]

## 7. Sources

"Read" says how much was opened: full = full text opened and the cited sections and tables read, part = only the
cited passages, abs = abstract only, data / tool = used as data or as a tool. Licences are given for code, models and
datasets and were checked at the repository or model card on 2026-11-13. Where GitHub reports no standard licence
identifier, the heading of the licence file is quoted (that code was not used). [VERIFIED: every licence in this
table except the Python one was read again through the GitHub and Hugging Face APIs; four entries were made more
precise, see the log.]

| Title | Authors | Year | URL | Read | Licence (code / model / data) |
|-------|---------|------|-----|------|-------------------------------|
| Attention Is All You Need | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin | 2017 | https://arxiv.org/abs/1706.03762 | full | paper only |
| Fast Transformer Decoding: One Write-Head is All You Need | Shazeer | 2019 | https://arxiv.org/abs/1911.02150 | full | paper only |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, et al. | 2023 | https://arxiv.org/abs/2305.13245 | full | paper only |
| Are Sixteen Heads Really Better than One? | Michel, Levy, Neubig | 2019 | https://arxiv.org/abs/1905.10650 | full | paper only |
| TinyStories: How Small Can Language Models Be and Still Speak Coherent English? | Eldan, Li | 2023 | https://arxiv.org/abs/2305.07759 (table image from https://ar5iv.labs.arxiv.org/html/2305.07759) | full | dataset roneneldan/TinyStories: CDLA-Sharing-1.0 |
| MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases | Liu, Zhao, Iandola, Lai, et al. | 2024 | https://arxiv.org/abs/2402.14905 | full | code facebookresearch/MobileLLM: FAIR Noncommercial Research License (LICENSE file; no SPDX identifier) |
| llama2.c (README, model.py, run.c, tinystories.py) | Karpathy | 2023 | https://github.com/karpathy/llama2.c | full | MIT |
| tinyllamas stories260K model card | Karpathy | 2023 | https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K | full | MIT |
| Using the Output Embedding to Improve Language Models | Press, Wolf | 2016 | https://arxiv.org/abs/1608.05859 | full | paper only |
| ByT5: Towards a token-free future with pre-trained byte-to-byte models | Xue, Barua, Constant, Al-Rfou, et al. | 2021 | https://arxiv.org/abs/2105.13626 | full | paper only |
| Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code | Karampatsis, Babii, Robbes, Sutton, Janes | 2020 | https://arxiv.org/abs/2003.07914 | full | paper only |
| Longformer: The Long-Document Transformer | Beltagy, Peters, Cohan | 2020 | https://arxiv.org/abs/2004.05150 | full | paper only |
| Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context | Khandelwal, He, Qi, Jurafsky | 2018 | https://arxiv.org/abs/1805.04623 | abs | paper only |
| Single Headed Attention RNN: Stop Thinking With Your Head | Merity | 2019 | https://arxiv.org/abs/1911.11423 | full | paper only |
| Confident Adaptive Language Modeling | Schuster, Fisch, Gupta, Dehghani, et al. | 2022 | https://arxiv.org/abs/2207.07061 | full | paper only |
| LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding | Elhoushi, Shrivastava, Liskovich, Hosmer, et al. | 2024 | https://arxiv.org/abs/2404.16710 | full | code facebookresearch/LayerSkip: Creative Commons Attribution-NonCommercial 4.0 text in LICENSE (GitHub reports no SPDX identifier) |
| DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference | Xin, Tang, Lee, Yu, Lin | 2020 | https://arxiv.org/abs/2004.12993 | full | paper only |
| One Size Does Not Fit All: Setting Inference Depth from the Questions a Deployment Actually Asks | Kaplan | 2026 | https://arxiv.org/abs/2609.14144 | abs | paper only |
| ALBERT: A Lite BERT for Self-supervised Learning of Language Representations | Lan, Chen, Goodman, Gimpel, et al. | 2019 | https://arxiv.org/abs/1909.11942 | full | paper only |
| The Depth-to-Width Interplay in Self-Attention | Levine, Wies, Sharir, Bata, Shashua | 2020 | https://arxiv.org/abs/2006.12467 | full | paper only |
| GLU Variants Improve Transformer | Shazeer | 2020 | https://arxiv.org/abs/2002.05202 | full | paper only |
| Were RNNs All We Needed? | Feng, Tung, Ahmed, Bengio, Hajimirsadeghi | 2024 | https://arxiv.org/abs/2410.01201 | full | third-party code lucidrains/minGRU-pytorch: MIT (not used) |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu, Dao | 2023 | https://arxiv.org/abs/2312.00752 | full | code state-spaces/mamba: Apache-2.0 (not used) |
| RWKV: Reinventing RNNs for the Transformer Era | Peng, Alcaide, Anthony, Albalak, et al. | 2023 | https://arxiv.org/abs/2305.13048 | full | code BlinkDL/RWKV-LM: Apache-2.0 (not used) |
| xLSTM: Extended Long Short-Term Memory | Beck, Poeppel, Spanring, Auer, et al. | 2024 | https://arxiv.org/abs/2405.04517 | full | code NX-AI/xlstm: Apache-2.0 per GitHub (not used) |
| Simple Recurrent Units for Highly Parallelizable Recurrence | Lei, Zhang, Wang, Dai, Artzi | 2017 | https://arxiv.org/abs/1709.02755 | full | code taolei87/sru: MIT (not used) |
| Quasi-Recurrent Neural Networks | Bradbury, Merity, Xiong, Socher | 2016 | https://arxiv.org/abs/1611.01576 | full | paper only |
| Attention, Dynamic Wave and Hybrid Sequence Mixers in Language Models | Babarytskyi, Babarytskyi | 2026 | https://arxiv.org/abs/2610.39854 | abs | paper only |
| An Empirical Study of Mamba-based Language Models | Waleffe, Byeon, Riach, Norick, Korthikanti, et al. | 2024 | https://arxiv.org/abs/2406.07887 | abs | paper only |
| On the quantization of recurrent neural networks | Li, Alvarez | 2021 | https://arxiv.org/abs/2101.05453 | full | paper only |
| mamba_small_bench (README) | apapiu | 2023 | https://github.com/apapiu/mamba_small_bench | full | no licence file |
| nanoGPT (README; source of the 1.4697 that mamba_small_bench quotes) | Karpathy | 2022 | https://github.com/karpathy/nanoGPT | part | MIT |
| char-transformer-shakespeare (model card) | delcenjo | 2026 | https://huggingface.co/delcenjo/char-transformer-shakespeare | full | MIT |
| Efficient softmax approximation for GPUs | Grave, Joulin, Cisse, Grangier, Jegou | 2016 | https://arxiv.org/abs/1609.04309 | full | paper only |
| On Using Very Large Target Vocabulary for Neural Machine Translation | Jean, Cho, Memisevic, Bengio | 2014 | https://arxiv.org/abs/1412.2007 | abs | paper only |
| Accelerating Zero-Entropy LLM Output to 100k Tokens per Second (blog) | Orenstein | 2025 (the page says "Written November 20, 2025") | https://michaelorenstein.com/blog/zero-entropy-tokens | part | blog |
| Evaluating Large Language Models Trained on Code | Chen, Tworek, Jun, Yuan, et al. | 2021 | https://arxiv.org/abs/2107.03374 | full | dataset openai/human-eval: MIT |
| Theory, Analysis, and Best Practices for Sigmoid Self-Attention | Ramapuram, Danieli, Dhekane, Weers, et al. | 2024 | https://arxiv.org/abs/2409.04431 | part | code apple/ml-sigmoid-attention (GitHub now redirects to apple-aiml-research/ml-sigmoid-attention): Apple licence text in LICENSE, no SPDX identifier |
| Replacing softmax with ReLU in Vision Transformers | Wortsman, Lee, Gilmer, Kornblith | 2023 | https://arxiv.org/abs/2309.08586 | part | paper only |
| HumanEval data file (byte lengths only) | OpenAI | 2021 | https://github.com/openai/human-eval | data | MIT |
| MBPP data file (byte lengths; train and validation splits for the tokenizer measurement) | Google Research | 2021 | https://github.com/google-research/google-research/tree/master/mbpp | data | data CC-BY-4.0 (repository README: "All datasets in this repository are released under the CC BY 4.0 International license"; the Hugging Face dataset card says the same); source files of the repository Apache-2.0 |
| SentencePiece (tool used for the measurement) | Google | 2018 | https://github.com/google/sentencepiece | tool | Apache-2.0 |
| Python 3.12 standard library source (proxy corpus for the measurement) | Python Software Foundation | 2023 | local install, LICENSE.txt | data | PSF License Version 2 |

Not edited by this agent: `SOURCES.md`. The four items actually used here (HumanEval and MBPP data files for byte
lengths, SentencePiece, the standard-library corpus) still need rows there if the measurement is kept.

## Verification log

Independent fact-check of this file on 2026-11-13. 25 claims were checked, grouped by source. Papers were read again
in the arXiv PDF (text extraction; the TinyStories table as a page image), repositories and model cards through the
GitHub and Hugging Face APIs, [measured here] numbers by measuring again from `data/bench/`, [cost model] numbers by an
independent re-computation from the constants in the header. No claim looked invented and none was deleted. Format:
claim | verdict | URL.

1. arXiv 2609.14144 (Kaplan, 2026) exists; its abstract gives 60 arithmetic problems solved by the full model and 10 to 28 under early exit | confirmed | https://arxiv.org/abs/2609.14144
2. arXiv 2610.39854 (Babarytskyi and Babarytskyi, 2026) exists; three 10.94 M TinyStories models with perplexity 6.93 (Transformer), 7.26 (recurrent only), 5.95 (hybrid) | confirmed | https://arxiv.org/abs/2610.39854
3. The other abstract-only citations: LSTMs use about 200 tokens of context and word order within 50; a 43% Mamba-2, 7% attention, 50% MLP stack beats the 8 B Transformer by 2.65 points on 12 tasks; shortlist decoding for large vocabularies | confirmed | https://arxiv.org/abs/1805.04623 https://arxiv.org/abs/2406.07887 https://arxiv.org/abs/1412.2007
4. Hobby and web sources: model card with 818,176 parameters, 4 layers, width 128, block 64, 2000 steps, one seed, val loss 1.93 (4 heads) and 1.91 (1 head), 3.63 for BPE-512; mamba_small_bench README 1.463 against 1.4697; the blog on zero-entropy token skipping exists | confirmed; corrected: nanoGPT figure "UNVERIFIED" -> verified (the nanoGPT README states 1.4697); blog year "not dated on the page read" -> 2025 | https://huggingface.co/delcenjo/char-transformer-shakespeare https://github.com/apapiu/mamba_small_bench https://github.com/karpathy/nanoGPT https://michaelorenstein.com/blog/zero-entropy-tokens
5. llama2.c and stories260K: dim 64, 5 layers, 8 heads, 4 KV heads, vocabulary 512, context 512, validation loss 1.2968; a 4096-token vocabulary gives about the same sequence length as the 32000 one; run.c takes the argmax at temperature 0, sorts with qsort for top-p and sizes the cache with kv_dim = dim x n_kv_heads / n_heads; tinystories.py trains with byte_fallback=True | confirmed | https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K https://github.com/karpathy/llama2.c
6. Licences in section 7 | confirmed: llama2.c MIT, tinyllamas MIT, TinyStories dataset CDLA-Sharing-1.0, human-eval MIT, SentencePiece Apache-2.0, minGRU-pytorch MIT, mamba Apache-2.0, RWKV-LM Apache-2.0, xlstm Apache-2.0, sru MIT, char-transformer-shakespeare MIT, mamba_small_bench no licence; corrected: MBPP "repository Apache-2.0; Hugging Face dataset card CC-BY-4.0" -> data CC-BY-4.0 by the repository README and the dataset card, source files Apache-2.0; "custom" -> FAIR Noncommercial Research License (MobileLLM), Creative Commons Attribution-NonCommercial 4.0 (LayerSkip), Apple licence text (ml-sigmoid-attention, repository moved to apple-aiml-research) | GitHub API `repos/<owner>/<name>` and `repos/<owner>/<name>/license` for each repository; https://huggingface.co/api/models/karpathy/tinyllamas https://huggingface.co/api/datasets/roneneldan/TinyStories https://huggingface.co/api/datasets/google-research-datasets/mbpp https://github.com/google-research/google-research/blob/master/README.md
7. Vaswani et al. Table 3, 65 M base model: 1 / 4 / 8 / 16 / 32 heads -> PPL 5.29 / 5.00 / 4.92 / 4.91 / 5.01 and BLEU 24.9 / 25.5 / 25.8 / 25.8 / 25.4; key dimension 16 -> 5.16 / 25.1 | confirmed | https://arxiv.org/abs/1706.03762
8. TinyStories: Figure 24 (2 layers 1.38 / 1.34 / 1.33; 1 layer 1.58 / 1.56 / 1.54), Figure 4 (the 21 losses and the three consistency scores quoted), footnote 2 (window 256, context 512) | confirmed; corrected in section 1: "halving heads costs 0.01 to 0.02 nats" -> 8 to 4 heads 0.01 to 0.02, 8 to 2 heads 0.04 to 0.05; "TinyStories depth/width/head tables" below 10 M -> only the depth/width table (the head table is at width 768) | https://arxiv.org/abs/2305.07759
9. MobileLLM Tables 1, 2, 10, 11, 13, 14, 15, every number quoted from them | confirmed; corrected: KV-head table "neighbouring rows move by up to 0.6 points" -> up to 0.9 points against the trend (350 M, 16 heads: 48.5 with 8 KV heads, 49.4 with 4); Table 11 "at about 135 M parameters" -> 134 to 135 M with the 4-layer row at 163 M; section 1 "MQA costs ... about 1 accuracy point" -> 0.6 to 1.7 points | https://arxiv.org/abs/2402.14905
10. Shazeer 2019 Tables 1 to 3: 211 M and 192 M models; ln PPL / BLEU of the eight rows quoted; 29.9 / 30.2 / 30.9 to 31.2; decoder 46 -> 3.8 and 46 -> 23 microseconds per token; window of the current and previous 31 positions | confirmed | https://arxiv.org/abs/1911.02150
11. GQA Table 1: MHA 47.2 at 1.51 s, MQA 46.6 at 0.24 s, GQA-8 47.1 at 0.28 s, 5% uptraining | confirmed | https://arxiv.org/abs/2305.13245
12. LayerSkip Tables 1 and 5: HumanEval 13.4 / 15.9 at the last layer and 0 / 4.88 at the middle layer (13B: 18.9 / 18.3 and 0 / 3.05); ROUGE-2 0.0035 vs 0.0513; speed-ups 1.34 to 2.16, 1.82 at 45% acceptance | confirmed; added the opposite Llama3 8B result of Table 2 (last layer 37.8 -> 28.7), which the file left out | https://arxiv.org/abs/2404.16710
13. CALM: oracle 1.53 layers at ROUGE-L 38.24 vs 38.32; 23.02 with copied keys and values; softmax response 2.62 layers, 2.57 times faster, FLOPs factor 0.44 to 0.91; classifier 4.51 layers, 2.04 times faster, d + 1 parameters | confirmed; wording "wins only because the next layer runs in parallel" -> the speed-up is TPU wall-clock time and the paper notes the next layer can start in parallel | https://arxiv.org/abs/2207.07061
14. minGRU on character-level Shakespeare: test loss 1.548 / 1.555 / 1.575 / 1.547; 3 layers, width 384, Conv4 -> minRNN -> MLP, expansion 2 | confirmed | https://arxiv.org/abs/2410.01201
15. Codex: 679 M model, optimal temperature 0.2 for pass@1 and 0.8 for pass@100; Codex-12M 2.00% pass@1 on HumanEval | confirmed | https://arxiv.org/abs/2107.03374
16. GLU variants Table 1: the eight log-perplexities at 524k steps, spread 0.003 to 0.010 at 65k steps, gated hidden size 2/3 | confirmed; corrected in section 3: ReGLU gap "inside the reported run-to-run spread" -> of the same order as the spread, which is reported at 65k steps only | https://arxiv.org/abs/2002.05202
17. Integer-only LSTM: 8-bit weights and "16 bits is needed to preserve accuracy" for the cell state | confirmed | https://arxiv.org/abs/2101.05453
18. ByT5: 4.1 bytes per SentencePiece token; 1.5 to 2.6 (Dakshina), 3.7 to 6.4 (XSum), 6.4 to 9.5 (XNLI) times slower | confirmed | https://arxiv.org/abs/2105.13626
19. Karampatsis et al. Tables 2 and 3: 4.90 / 4.78 / 4.77 and 3.59 / 3.35 / 3.15; cache 5.54 -> 2.57 (Java) and 4.81 -> 2.89 (Python); GRU with 512 units; smallest vocabulary 2k | confirmed; section 4 wording "smallest published" -> "smallest found in our search" | https://arxiv.org/abs/2003.07914
20. Press and Wolf: 4.65 M -> 2.65 M parameters and test perplexity 114.5 -> 112.4 | confirmed | https://arxiv.org/abs/1608.05859
21. Windowed and single-head attention: SHA-RNN 1.330 / 1.076 / 1.068 bpc at 51 / 52 / 54 M and Transformer-XL 1.06 at 41 M with 160 heads; Longformer 1.21 / 1.23 / 1.24 at 41 M | confirmed; corrected in idea 4: "supported at 41 M and 52 M" -> "indirect support only", because neither paper compares small lower-layer windows with full attention | https://arxiv.org/abs/1911.11423 https://arxiv.org/abs/2004.05150
22. Recurrent models: xLSTM Table 1 (13.43 / 13.70 / 14.25 / 15.62 at about 400 M, 15 B tokens); smallest models Mamba 130 M, RWKV 169 M, xLSTM 125 M; SRU 1.19 bpc at 49 M vs LSTM 1.30 at 46 M; QRNN 78.3 vs 82.0 (QRNN with zoneout) | confirmed | https://arxiv.org/abs/2405.04517 https://arxiv.org/abs/2312.00752 https://arxiv.org/abs/2305.13048 https://arxiv.org/abs/1709.02755 https://arxiv.org/abs/1611.01576
23. [measured here] HumanEval and MBPP byte lengths (6.5) and the context-fit table, with 62% of HumanEval tasks at T = 256 and 95% at T = 512 for 2.5 bytes per token | confirmed (102 of 164 and 156 of 164 tasks); added: 57.3% / 94.5% at 2.4 and 71.3% / 97.6% at 2.7 bytes per token | https://github.com/openai/human-eval https://github.com/google-research/google-research/tree/master/mbpp
24. [measured here] bytes per token 2.10 / 2.04 / 2.37 / 2.53 / 2.82 for code and 2.18 / 2.12 / 2.74 / 3.18 with the description (6.4) | confirmed within 3% only: the re-run gave 2.04 / 1.99 / 2.32 / 2.47 / 2.74 and 2.13 / 2.08 / 2.69 / 3.13, and the original script is not in the repository; corrected in idea 6: "dropping byte fallback gives 19% fewer tokens" -> 14% for byte fallback alone, 19% together with pieces that span spaces | https://github.com/google-research/google-research/tree/master/mbpp https://github.com/google/sentencepiece
25. [cost model] tables 6.2, 6.3, 6.6, 6.7 and every cost-model percentage in sections 1 and 5 (baseline 1,100,668 cycles and 91 tokens per second at T = 256; 2 heads x 32 -34%; one-pass softmax -19%; window 32 in the lower 3 or 4 layers -32% / -42%; d = 128 x 2 layers -25% / -41%; prefill skip -23% per prompt token and -17% per task; stacked 201 / 183 / 156 tokens per second) | confirmed as arithmetic: every table cell reproduces, and the header constants agree with `research/CONTEXT.md` (row of n multiply-adds n + 24 cycles, at most 255 per `MAC` instruction, 12 to 32 cycles per element-wise operation, 100 MHz, about 450 KB); these are estimates, not measurements | `research/CONTEXT.md`

Not re-checked, and therefore without a tag: Michel et al. (arXiv 1905.10650), ALBERT (1909.11942), DeeBERT
(2004.12993), Levine et al. (2006.12467), adaptive softmax (1609.04309), sigmoid attention (2409.04431), ReLU
attention (2309.08586), the two standard-library columns of table 6.4, and the licence of the Python standard library.
