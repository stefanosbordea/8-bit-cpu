# 01. The smallest language models that write code: sizes and benchmark scores

Research date: 2026-11-13. Scope: Python code generation. Every published number below was read at the URL given
next to it on that date. Sections 6.4 and 6.5 are our own computations and are labelled as such. Hub model cards
are self-reported unless stated otherwise. Nothing here was tuned on a test set; the benchmark size statistics in
section 6.4 are aggregate byte counts only (no problem text was inspected or stored).

Fact-check pass (2026-11-13, second agent): claims marked [VERIFIED] were re-read at the primary source (paper
PDF, hub API or card, dataset file) or recomputed. [UNVERIFIED: reason] marks claims that could not be supported.
Claims without a tag were not re-checked. One line per checked claim is in the "Verification log" at the end.

## 1. Answer in one paragraph

The smallest models with a non-zero HumanEval score in a primary paper, as found in our search, are Codex-12M
(pass@1 2.00%, pass@10 3.62%, pass@100 8.58%) and AlphaCode's 13 M decoder-only model (1.5% / 3.6% / 8.6%)
[VERIFIED]. "12M" is the paper's label: the Codex paper measures model size as non-embedding parameters, so
the total with the GPT-3 token-embedding table is larger, and which of the two models is smaller in total
parameters is not established [UNVERIFIED: neither paper gives total parameter counts or layer shapes for
these two models]. Neither has public weights; they were trained on 159 GB (Codex, GPT-3 BPE plus whitespace
tokens) and 54 GB (AlphaCode, SentencePiece 8,000) of Python [VERIFIED]. The smallest public models with
paper-level numbers found in our search are about 110 M to 125 M parameters (codeparrot-small 3.80%,
PyCodeGPT-110M 8.33%, GPT-Neo-125M 0.75% pass@1; weight files of 457 to 526 MB) [VERIFIED]. Below that we found
only hobby models with self-reported scores (PyBlissa-Coder-40M claims 14.6% HumanEval and 4.4% MBPP; a
6.9 M model reports 2 of 20 unseen tasks and says "expect about 0 %" on HumanEval) [VERIFIED] (the cards say
this; the scores are not reproduced). We found no HumanEval or MBPP number for any model below 12 M
parameters, no code model that fits 450 KB, and no n-gram or cache model scored on these benchmarks
[VERIFIED] (a second search also found none). The nearest things to our budget are not benchmarked code
generators: 1 M-parameter character-level models trained on grammar-generated Python (81.2% next-token
accuracy, 74.6 to 79.2% output prediction accuracy on synthetic programs), `stories260K` (1.06 MB, stories),
and a ternary story model running on a Commodore 64 (242 KB weight file, 361 KB cartridge image) [VERIFIED].
Extrapolating the published Codex and AlphaCode size curves down to 0.3 M parameters (our own fit, 40 times
below the smallest data point) gives about 0.1 to 0.25% pass@1 on HumanEval, i.e. less than one problem out of
164, and about 1.2 to 1.7% pass@100 [VERIFIED] (the fits were recomputed; they are an extrapolation, not a
measurement). A HumanEval score of zero at pass@1 is the expected outcome for Tier A; for MBPP the same is
assumed by analogy [UNVERIFIED: no MBPP size curve was fitted; the smallest paper MBPP numbers in section 6.2
are at 135 M to 150 M parameters]. The informative numbers will be pass@10/pass@100 and the easier ladder.

## 2. Proven (numbers with sources)

Low end of the published range
- Codex-12M: HumanEval pass@1 2.00%, pass@10 3.62%, pass@100 8.58%. Codex-25M 3.21 / 7.1 / 12.89; Codex-42M
  5.06 / 8.8 / 15.55; Codex-85M 8.22 / 12.81 / 22.4; Codex-300M 13.17 / 20.37 / 36.27. Estimator: unbiased
  pass@k from n = 200 samples per task, nucleus p = 0.95. The paper names T = 0.2 as optimal for pass@1 and
  T = 0.8 for pass@100 (found on the 679M model) and plots both against size at these temperatures; taking the
  highest result over temperatures 0.2, 0.4, 0.8 is stated for the GPT-Neo rows only, and the temperature of
  the Codex pass@10 column is not stated. Trained for 100 B tokens on 159 GB of Python
  from 54 M GitHub repositories, GPT-3 BPE plus whitespace-run tokens. Table 1 and sections 2.1, 3.1, 3.2, 3.3
  of https://arxiv.org/abs/2107.03374 [VERIFIED]
- Codex model size is counted without embeddings: section 3.3 fits test loss against N, "the number of
  non-embedding parameters in the model" [VERIFIED]. The labels 85M, 300M and 679M match
  12 x layers x width^2 (84.9 M, 302.0 M, 679.5 M) for the GPT-3 Small, Medium and Large shapes (12 x 768,
  24 x 1024, 24 x 1536), which the GPT-3 paper lists as 125M, 350M and 760M in total (Table 2.1 of
  https://arxiv.org/abs/2005.14165) [VERIFIED]. We therefore read "12M" as a non-embedding count; the total
  size of Codex-12M is larger by the size of its token table
  [UNVERIFIED: our inference; the paper gives no layer shapes or total counts for the 12M, 25M and 42M models].
  Whether AlphaCode's "13M" includes its 8,000-token embedding table is not stated in the passages read.
- AlphaCode decoder-only models, pretrained on the Python part of their GitHub set (10.7 + 43.6 GB), no
  fine-tuning, 1000 samples: 13M 1.5 / 3.6 / 8.6; 29M 3.4 / 5.8 / 11.2; 55M 4.2 / 8.2 / 16.9; 89M 4.3 / 12.2 / 20.0;
  302M 11.6 / 18.8 / 31.8. Each cell is the highest nucleus-sampling result over temperatures 0.0 to 0.8.
  SentencePiece vocabulary of 8,000 (stated for the AlphaCode models in general). Appendix Table A3, the
  tokenization paragraph and the Appendix B data table of https://arxiv.org/abs/2203.07814 [VERIFIED]
- In the same Codex table, GPT-Neo-125M (trained on The Pile, 8% GitHub code) scores 0.75 / 1.88 / 2.97 and the
  TabNine free model 2.58 / 4.35 / 7.59; "GPT-3 solves 0%". https://arxiv.org/abs/2107.03374 [VERIFIED]
- CodeT5 (220M), CodeGPT-Adapted (124M) and CodeClippy (125M) "all get 0% on pass@1, pass@10, and pass@100" on
  HumanEval (CERT paper, Table 1 caption and appendix A, "Results on HumanEval").
  https://arxiv.org/abs/2206.06888 [VERIFIED]
- pass@100 is 4.3 times pass@1 for Codex-12M (8.58 / 2.00) and 5.7 times for AlphaCode-13M (8.6 / 1.5), from the
  two tables above. [VERIFIED] pass@1 and pass@100 are taken at different temperatures, so the ratio mixes
  them: one sample set drawn at the pass@1 temperature gives a ratio no larger than this, one drawn at the
  pass@100 temperature a ratio no smaller.

Smallest public models with published scores (about 100 M to 165 M)
- codeparrot-small (110M, GPT-2, BPE 32,768, about 29 B training tokens): 3.80 / 6.57 / 12.78.
  https://huggingface.co/codeparrot/codeparrot-small [VERIFIED]
- PyCodeGPT-110M (GPT-Neo shape, BPE 32k trained from scratch, 96 GB / 13 M files): 8.33 / 13.36 / 19.13 in the
  paper (Table 6, n = 200, highest over temperatures 0.1 to 1.0). https://arxiv.org/abs/2206.06888 [VERIFIED]
- PolyCoder-160M (249 GB of code in 12 languages, 2.5 B Python tokens): 2.13 / 3.35 / 4.88.
  https://arxiv.org/abs/2202.13169 [VERIFIED] (the three scores, as listed in CERT Table 6; the PolyCoder paper
  itself and the corpus figures were not re-read)
- tiny_starcoder_py (164M, StarCoderData Python, 100 B tokens, vocabulary 49,152): pass@1 7.84% (card, marked
  "verified: false"). https://huggingface.co/bigcode/tiny_starcoder_py [VERIFIED]
- TinyCodeLM-150M (72 B tokens of FineWeb + The Stack Python): pretrained 6.1% HumanEval and 5.4% MBPP(+) at
  temperature 0; after LintSeq instruction tuning 12.8 ± 2.6 / 20.6 ± 1.1 (HumanEval pass@1 / pass@10) and
  13.6 ± 2.1 / 24.4 ± 0.8 (MBPP(+)). TinyCodeLM-400M: 6.7 and 6.8 pretrained; 13.4 / 20.9 and 19.4 / 29.9 tuned.
  https://huggingface.co/upiter/TinyCodeLM-150M and Table 1 of https://arxiv.org/abs/2410.02749 [VERIFIED]
- SmolLM-135M-Instruct, evaluated by the LintSeq authors: HumanEval 7.7 ± 0.8 / 14.5 ± 1.0, MBPP(+) 10.1 ± 1.8 /
  14.6 ± 0.5. Same table: AlphaCode-89M 4.3 / 12.2, Codex-85M 8.2 / 12.8, CodeT5+ 220M 12.0 / 20.7,
  codegen-350M-mono 12.8 / 23.1 (HumanEval) and 9.4 ± 1.8 / 15.2 ± 0.7 (MBPP(+)). https://arxiv.org/abs/2410.02749
  [VERIFIED]

Data quality moves the score more than size does (at 350 M)
- 350 M models trained on raw deduplicated Python from The Stack plus StackOverflow saturate at 12.19% HumanEval
  after about 200 B tokens; the classifier-filtered subset gives 17.68% after 36k steps; adding synthetic
  textbooks gives 20.12%; fine-tuning on synthetic short exercises (CodeExercises, under 200 M tokens) gives
  phi-1-small 45%. phi-1 (1.3 B) reaches 50.6% HumanEval and 55.5% MBPP with under 7 B pretraining tokens.
  Abstract, Table 1, Figure 2.1 and section 2.1 of https://arxiv.org/abs/2306.11644 [VERIFIED]
- Codex: adding tokens for whitespace runs represents code "using approximately 30% fewer tokens".
  Section 3.2 of https://arxiv.org/abs/2107.03374 [VERIFIED]

At or below about 10 M parameters
- "Curriculum Learning for Small Code Language Models" trains nanoGPT models of about 1 M parameters
  (6 layers, 6 heads, embedding 120, block size 256, character-level vocabulary of 41, no biases) on
  grammar-generated Python (TinyPy: assignments, if/elif/else, for and while loops, print; 1.2 M snippets, of
  which 1.02 M with 95.9 M character tokens are the training set). Baseline: token-level accuracy 81.23%,
  line-level exact match 41.74%, edit
  similarity 74.15, output-prediction accuracy 74.58%. With a hybrid curriculum: 81.13% / 42.04% / 74.65 and
  79.23% output accuracy (82.84% easy, 80.79% medium, 74.04% hard). Checkpoints are public (4,385,396 or
  4,386,378 bytes each, Apache-2.0). https://arxiv.org/abs/2407.10194 and
  https://github.com/kamel-yamani/CurriculumLearningForSmallCodeLanguageModels [VERIFIED]
- Pico-1.3-Coding-Large (hub, 6,917,376 parameters, 13.8 MB, Llama-style 6 layers / 256 wide, BPE 8,192,
  about 28,000 short test-passing tasks from OpenCodeInstruct plus chat data, 12 minutes on one GPU): 68% on 50
  simple tasks of the training kind (48 of the 50 share a function name with training data), 10% (2 of 20) on
  hand-written tasks with unseen function names, and the card states it "is not tested on HumanEval or similar
  benchmarks; expect about 0 % there". https://huggingface.co/Convergent-ch/Pico-1.3-Coding-Large [VERIFIED]
- Karpathy's character-level 3-layer LSTMs with "about 10 million parameters" trained on 474 MB of Linux C
  source make "very few syntactic errors" but the author does not think the output compiles; the common error is
  undefined or unused variables. https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- CHISPA-20M (hub, 20.1 M, line completion only, 521 M training tokens): full-line exact match 1.2%, exact
  match of the leading identifier 17.1%, 11.0% of completions introduce an undefined name.
  https://huggingface.co/esraderey/chispa-20m-python
- GoldWorm v8 (hub, 85.2 M, byte-level vocabulary 256, trained from scratch): pass@1 7.08% and pass@16 16.67%
  (5 of 30 tasks) on its own 30-task set, 92.92% syntactically valid; 0.42% pass@1 on a harder 30-task set.
  https://huggingface.co/Uuuuuuniiiiiii/goldworm-v8-coder-85m

Size of the benchmark inputs (computed by us from the dataset files, section 6.4)
- HumanEval: prompt median 396 bytes (115 to 1,360); 31 of 164 prompts are at most 256 bytes and 118 at most 512;
  prompt plus canonical solution fits in 512 bytes for 64 of 164 and in 768 bytes for 122. 37 canonical solution
  bodies are one non-blank line. File: https://github.com/openai/human-eval (data/HumanEval.jsonl.gz, MIT)
  [VERIFIED] (recomputed)
- MBPP test split (task ids 11 to 510): task text plus the three asserts has a median of 226 bytes; text plus
  asserts plus reference code fits in 512 bytes for 372 of 500 and in 768 bytes for 470.
  File: https://github.com/google-research/google-research/tree/master/mbpp (CC BY 4.0 per
  https://huggingface.co/datasets/google-research-datasets/mbpp) [VERIFIED] (recomputed)
- In MBPP few-shot runs with web-text models from 244 M to 137 B non-embedding parameters, "even the smallest
  models can write syntactically correct Python code around 80% of the time"; solved fraction is roughly linear
  in log model size (the paper draws 80 samples per task at temperature 0.5). Section 4.1 of
  https://arxiv.org/abs/2108.07732 [VERIFIED]

Non-neural and small-memory baselines (token prediction, not program synthesis)
- Smoothed n-gram models give 3 to 4 bits per token on Java projects, against about 10 bits (unigram) to under
  8 bits (10-gram) for English; entropy saturates around 3- to 4-grams. Hindle et al.,
  https://people.inf.ethz.ch/suz/publications/natural.pdf
- Java, bits per token, static / dynamic / maintenance: plain n-gram 3.94 / 2.64 / 2.32; cache 2.45 / 1.89 / 1.73;
  nested cache - / 1.70 / 1.41; LSTM/300 3.22 / 1.84 / -; LSTM/650 3.03 / 1.91 / -. Table 2 of Hellendoorn and
  Devanbu, https://web.cs.ucdavis.edu/~devanbu/isDLgood.pdf [VERIFIED]
- Python, small training set: n-gram 5.30 bits and MRR 43.63% (static); nested cache n-gram 2.89 bits and
  65.97% (dynamic); one-layer GRU with 512 units on BPE 3.91 bits / 81.66% (static), 2.72 bits / 86.28%
  (dynamic). Disk for the n-gram models: 150 to 500 MB (small set), 6 to 8.5 GB (large set); RAM about 5 GB to
  50-60 GB. The neural models are 15 MB (BPE 2k) to 45 MB (BPE 10k) on disk. Table 2 and section 8.3 of
  https://arxiv.org/abs/2003.07914 [VERIFIED]
- A neural reranker over static-analysis candidates needs 6 MB of RAM, 8 ms per completion, and reaches 90%
  top-5 accuracy for API-member completion. https://arxiv.org/abs/2004.13651 [VERIFIED]
- Post-training 8-bit quantisation of an LSTM completion model: 152 MB to 38 MB, top-5 accuracy 92% to 89%.
  Section 8.1 of https://arxiv.org/abs/1912.00742 [VERIFIED]
- PY150 next-token accuracy (CodeXGLUE numbers as listed in CERT Table 7): LSTM 58.00%, Transformer 73.26%,
  GPT-2 (117M) 74.22%, CodeGPT (124M) 74.93%, CodeParrot (110M) 77.22%, PyCodeGPT (110M) 80.24%.
  https://arxiv.org/abs/2206.06888 [VERIFIED]

Things at our memory scale that exist, none of them a benchmarked code generator
- `stories260K`: 5 layers, dim 64, 8 heads, 4 KV heads, vocabulary 512, validation loss 1.2968 on TinyStories,
  1,056,540-byte .bin export, MIT. https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K [VERIFIED]
- Pico-1 (254,400 parameters, 513,984 bytes) and Pico-1.2 (460,608 parameters, 929,376 bytes, token accuracy
  43.8%): chat toys; according to its card, Pico-1.2 "says that it cannot write code".
  https://huggingface.co/Convergent-ch/Pico-1.2 [VERIFIED] (the Pico-1 file size was not re-checked)
- ternary.c64: 688,128 ternary weights (820,352 parameters in total per hub metadata), int16 fixed point, runs
  on an unexpanded Commodore 64 from a cartridge, TinyStories validation loss 1.439, packed weight file
  242,324 bytes, complete cartridge image (model, tables, vocabulary, code) 361,216 bytes, MIT.
  https://huggingface.co/gdm32569/ternary-c64-688k [VERIFIED]

How the hub was searched
- Hugging Face API by model name: 32 search terms times 3 sort orders (1,000 results per query cap), 52,495
  distinct records, filtered to those with fewer than 60 M parameters and code/python in the name; plus about 15
  full-text searches of model cards (https://huggingface.co/api/search/full-text). Records without safetensors
  metadata cannot be filtered by size, so small models can have been missed.

## 3. Uncertain or contested

- PyBlissa-Coder-40M (39.9 M, hub) claims HumanEval pass@1 14.6% (24/164, temperature 0.25, "zero-shot,
  fenced-code extraction") and MBPP 4.4% (22/500); the 50M sibling claims 10.4% HumanEval. Not reproduced by
  anyone we found. Reasons for doubt: the count 24/164 implies one sampled completion per task at temperature
  0.25; HumanEval is 3.3 times the MBPP score, while in section 6.2 MBPP is the higher of the two for most
  models that report both and the largest HumanEval-to-MBPP ratio is 2.2 (codegen-350M-nl); training data is LLM-generated instruction data
  (nvidia/OpenCodeInstruct), and the card reports no overlap check against the benchmarks.
  https://huggingface.co/Rohanify/PyBlissa-Coder-40M-GGUF [VERIFIED] (the card states these numbers; they are
  self-reported). The ratio argument is indicative only: the MBPP numbers in section 6.2 are different variants
  (CodeGen: sanitized MBPP, n = 100, temperature 0.8; LintSeq runs: MBPP(+)).
- Hub cards can be wrong or invented. One card claims a "128 Billion Parameter" model with 78.4% HumanEval while
  the hub metadata for the same repository reports 206,208 parameters
  (https://huggingface.co/codefusionduo/suku-ai-128b-titan). Another 29 M model reports 99.07% held-out token
  accuracy after 200 fine-tuning steps (https://huggingface.co/MatterWaveAI/grt-30m-code). Two cards state a
  parameter count that differs from the hub metadata (BananaMind NanoCoder: card 9,895,690, metadata 12,254,978;
  jfrick 10m: card 9,849,536, metadata 16,140,992). Treat every hub number as a lead, not a fact.
  [VERIFIED] (hub metadata values 206,208, 12,254,978 and 16,140,992 and the BananaMind card total; the 78.4%,
  99.07% and jfrick card figures were not re-read). A card-versus-metadata mismatch does not show that the card
  is wrong: hub metadata counts every stored tensor, including non-weight buffers (GPT-Neo-125M: 125,198,592
  float32 plus 25,165,824 uint8 entries; PolyCoder-160M: 212,654,976 in total).
- Our extrapolation to 0.3 M parameters (section 6.5) fits four or five points between 12 M and 302 M and
  extends 40 times below them. Those models used 54 to 159 GB of Python, long contexts and large vocabularies;
  a model with vocabulary 512 and context 256 is not on the same curve. The direction is not in doubt, the
  value is.
- Whether curated or synthetic instruction data helps as much at 0.3 M as at 350 M is unknown. The evidence is
  at 350 M (phi-1-small), 150 M (TinyCodeLM), 40 M (self-reported) and 6.9 M (in-distribution only).
- Curriculum learning at 1 M parameters raised output-prediction accuracy by 4.65 points but did not move
  completion (81.23% to 81.13% token accuracy); the abstract itself calls the completion effect "less
  significant". Synthetic TinyPy programs are far from HumanEval/MBPP tasks. https://arxiv.org/abs/2407.10194
  [VERIFIED]
- Published numbers for the same model differ by source: PyCodeGPT-110M is 8.33 / 13.36 / 19.13 in the paper
  and 8.32 / 13.53 / 18.3 on its card (https://huggingface.co/Daoguang/PyCodeGPT). [VERIFIED]
- Metric variants are not comparable: Codex used n = 200 and the highest result over temperatures; AlphaCode
  1000 samples; the SmolLM blog 20 samples at temperature 0.2; TinyCodeLM "temperature-tuned"; the Qwen2.5-Coder
  report uses the EvalPlus harness (decoding settings not checked). "MBPP" means the 500-task test split, the
  sanitized set (427 tasks in the released data, 426 in the paper text), or MBPP+ depending on the paper. The
  CodeGen MBPP numbers in section 6.2 are on sanitized MBPP with n = 100 at temperature 0.8; Austin et al. draw
  80 samples per task at temperature 0.5. [VERIFIED] (Codex, AlphaCode, TinyCodeLM, CodeGen and MBPP parts; the
  SmolLM blog setting was not re-checked)
- In the standard MBPP prompt the three asserts shown to the model are the same asserts used for scoring
  (https://arxiv.org/abs/2108.07732). Sampling many programs and keeping one that passes them is therefore
  pass@k, not pass@1, and must be reported as pass@k. [VERIFIED] (the paper counts a sample as correct when it
  "passes the three given test cases")
- StarCoderBase-1B HumanEval pass@1 15.17% [VERIFIED] (model-index metadata of the gated repository, read through
  https://huggingface.co/api/models/bigcode/starcoderbase-1b; self-reported, "verified: false").
- flytech/python-codes-25k is MIT according to the PyBlissa card [UNVERIFIED: the dataset card metadata has no
  licence field, https://huggingface.co/datasets/flytech/python-codes-25k].
- The exact MBPP percentages of the 244 M model in Austin et al. are only in Figure 3 and were not extracted.
- A web search result pointed to an 85 M model with 14.6% HumanEval in arXiv 2505.10222 (ComplexFormer); the
  arXiv page marks that paper as withdrawn, so it is not used. [VERIFIED]
- Found during the fact-check, not in the original search: theOG-50M v3 (hub, 50,148,352 parameters, Apache-2.0)
  claims 42/164 on HumanEval (25.6%, greedy, EvalPlus 0.3.1 base tests) and 91/500 on the MBPP test split
  (18.2%); its own run of codegen-350M-mono on the same harness gives 22/164 (13.4%; published 12.76%).
  [UNVERIFIED: self-reported hub card, not reproduced by anyone we found]
  https://huggingface.co/DrunkkToys/theOG-50M

## 4. Not tried at our scale (about 250k to 400k int8 parameters, 8-bit multiply-accumulate CPU, 450 KB)

- No HumanEval or MBPP result for any model below 12 M parameters was found in papers. On the hub the smallest
  model with a self-reported benchmark number is 39.9 M (PyBlissa); the 6.9 M Pico card predicts about 0%.
  [VERIFIED] (a second search of hub models tagged with the HumanEval or MBPP datasets, of the arXiv API and of
  the web also found none below 39.9 M)
- No Python code generator at or below 1 M parameters trained on natural code and scored on functional
  correctness was found. The 1 M TinyPy models use a 41-character alphabet and cannot read a natural prompt.
- No code model with int8 weights in 450 KB or less was found. Nearest sizes: 242 KB (ternary, stories),
  514 KB (Pico-1, chat), 1.06 MB (stories260K float32), 4.39 MB (1 M TinyPy float32). [VERIFIED] (the 242 KB,
  1.06 MB and 4.39 MB sizes; the Pico-1 size was not re-checked)
- No n-gram, cache or other non-neural model scored on HumanEval or MBPP was found, and no published n-gram
  code model anywhere near 450 KB (published sizes are 150 MB to 8.5 GB). [VERIFIED] (a second web search also
  found none; the sizes are those of Karampatsis et al.)
- No benchmarked code model with a sub-word vocabulary of 256 or 512 was found. Seen: characters (41),
  bytes (256, at 85 M parameters), BPE 4,096 to 16,384 (hobby), 8,000 to 151,936 (papers).
- No HumanEval or MBPP evaluation with a 256-token context was found; reference models use 1,024 to 8,192.
  The shortest context with a HumanEval number found is 512 tokens (PyBlissa card, self-reported) [VERIFIED].
- Sample-and-select (pass@k with k of 10 to 100) is documented only down to 12 M parameters.
- Curriculum or grammar-generated pretraining followed by natural short tasks has not been reported at 1 M.

## 5. Ranked ideas for LS-LLM

| # | Idea | Expected effect on score / speed / memory | Cost to try | Risk | Keep-or-drop measurement | CPU change? |
|---|---|---|---|---|---|---|
| 1 | Train on short, test-verified instruction-to-function pairs (filtered OpenCodeInstruct subset, MBPP train split, own generated tasks) instead of raw GitHub files; remove anything sharing function names or long n-grams with the test sets | Score: largest lever in the sources at fixed size (350 M: 12.19% to 20.12% to 45%; 6.9 M: 68% in-distribution). Speed, memory: none | 2-3 days of data work; stream a subset under 100 MB (the full set is a 6.86 GB parquet download and needs a parquet reader, which is not in the venv) | Overlap with benchmarks inflates scores; narrow data gives 68% seen versus 10% unseen names | Practice-set pass rate on tasks with unseen function names versus a raw-code baseline at equal steps; keep if higher by more than the seed spread | No |
| 2 | Fix the reporting protocol now: HumanEval and MBPP with pass@1, pass@10, pass@100 (unbiased estimator, n = 200), zeros reported, plus the prior expectation from section 6.5 | Score: none; makes a zero defensible and shows pass@100 (4 to 6 times pass@1 at 12-13 M, the two taken at different temperatures) | 1 day; 164 x 200 samples in the float simulator | Low | Not a keep-or-drop item; check that the estimator reproduces a hand-computed case | No |
| 3 | Validate the harness by reproducing one published number with a public model (codeparrot-small 3.80% pass@1, 457 MB; or SmolLM-135M-Instruct 7.7 ± 0.8, 269 MB), and run the small public models as local reference rows | Score: none; turns the ladder into measured rows on our machine | 0.5-1 day; needs `transformers` and `tokenizers` (not in the venv) or a hand-written GPT-2 loader; 269-457 MB files exceed the 100 MB download rule | Disk (1.4 to 1.7 GB free during the research session; two `df -h /` readings during the fact-check gave 199 MiB and 3.9 GiB, so read it again before any download) | Reproduced value within the sampling error of the published one | No |
| 4 | Measure bytes per token of the 256/512 tokenizer on benchmark prompts and choose context so the prompts fit; count truncated tasks as failures and report the count | Score: avoids forced zeros (at 512 bytes of context only 64 of 164 HumanEval tasks fit prompt plus solution; MBPP test 372 of 500). Speed: longer context costs attention time | Hours | None | Number of tasks whose prompt plus reference fits; pass rate on the fitting subset reported separately | No (context length is a model choice) |
| 5 | Structural tokens for indentation and line breaks (INDENT, DEDENT, NEWLINE style) and whitespace runs | Speed: fewer tokens per program (Codex: about 30% fewer tokens from whitespace tokens); score: more program per context; memory: a few vocabulary rows | 1 day with the tokenizer work | Lexer and detokeniser bugs | Bytes per token on practice code; syntax-valid rate; tokens per solved task | No |
| 6 | Curriculum: grammar-generated correct-by-construction programs for the early steps, then short natural tasks (write our own generator; the TinyPy repository has no licence file) | Score: syntax validity and "runs without error" expected up; published gain is on output prediction (+4.65 points at 1 M), not on completion | 1-2 days | Synthetic programs may not transfer; wasted steps | Syntax-valid rate and practice pass rate at equal total steps, 3 seeds | No |
| 7 | Add cheap ladder metrics used by other tiny-model reports: next-token accuracy, syntax-valid rate, undefined-name rate, pass rate on seen-name versus unseen-name tasks | Score: none; shows progress where pass@1 is zero | 0.5 day | None | Not a keep-or-drop item | No |
| 8 | Equal-memory non-neural baseline: pruned or hashed n-gram with backoff over the same tokens in 450 KB, optionally with a small cache | Score: gives the reference row "what 450 KB buys without a network"; expected weak (published n-gram code models need 150 MB or more) | 1 day in numpy | None | Bits per byte, token accuracy, syntax-valid rate versus the transformer at equal bytes | No |
| 9 | Many samples per task on the board (LFSR sampling, UART out), reported as pass@k | Score: pass@100 about 4-6 times pass@1 in the 12-13 M data; speed: k times the tokens (100 samples of 100 tokens is about 100 s at 100 tokens/s [UNVERIFIED: 100 tokens/s is a design estimate in research/CONTEXT.md, not a measurement]) | Small once sampling works | Must be labelled pass@k; MBPP prompt asserts are the scoring asserts | pass@k curve for k = 1, 10, 100 on the practice set | No (uses the planned LFSR and UART) |
| 10 | Hashed bigram/4-gram embedding added to the residual stream (as in BananaMind NanoCoder [VERIFIED]: its card describes a causal bigram/four-gram hash memory with 1,920,002 parameters) | Unknown; no measured gain in the source. Memory: for example 1,024 x 64 int8 = 64 KB; speed: one table row and 64 adds per token | 1 day | Unproven; spends 14% of the budget | Validation loss and practice pass rate at equal total bytes | No beyond the planned block-RAM table read |

## 6. Tables

### 6.1 Models at or below about 100 M parameters

"Runnable locally?" means: weights public, and the weight file is at most 100 MB (the project download limit).
"Over limit" means public and small enough for 8 GB RAM but above the 100 MB rule. All need a loader
(`transformers`/`tokenizers` are not in the venv). HumanEval columns are pass@1 / pass@10 / pass@100 in percent
unless stated.

| Model | Parameters | Bytes on disk | Training data | Tokenizer / vocabulary | HumanEval | MBPP | Licence | Weights public? | Runnable locally? | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| Codex-12M | 12 M (label; read as a non-embedding count, section 2) | not released | 159 GB Python, 54 M GitHub repos, 100 B tokens | GPT-3 BPE + whitespace tokens | 2.00 / 3.62 / 8.58 (n = 200; temperatures in section 2) | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| AlphaCode decoder-only 13M | 13 M | not released | Python part of GitHub set (54.3 GB) | SentencePiece 8,000 | 1.5 / 3.6 / 8.6 (1000 samples) | none | n/a | No | No | https://arxiv.org/abs/2203.07814 [VERIFIED] |
| Codex-25M / 42M / 85M | 25 / 42 / 85 M (labels, as Codex-12M) | not released | as Codex-12M | as Codex-12M | 3.21 / 7.1 / 12.89; 5.06 / 8.8 / 15.55; 8.22 / 12.81 / 22.4 | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| AlphaCode decoder-only 29M / 55M / 89M | 29 / 55 / 89 M | not released | as 13M | SentencePiece 8,000 | 3.4 / 5.8 / 11.2; 4.2 / 8.2 / 16.9; 4.3 / 12.2 / 20.0 | none | n/a | No | No | https://arxiv.org/abs/2203.07814 [VERIFIED] |
| TabNine free model | unknown | not released | unknown | unknown | 2.58 / 4.35 / 7.59 | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| 1 M TinyPy models (Naïr et al.) | about 1 M | 4,385,396 or 4,386,378 | 1.2 M grammar-generated Python snippets | characters, 41 | not evaluated (token accuracy 81.23%, output accuracy 74.58-79.23% on synthetic programs) | not evaluated | Apache-2.0 | Yes (GitHub) | Yes, but cannot read natural prompts | https://github.com/kamel-yamani/CurriculumLearningForSmallCodeLanguageModels [VERIFIED] |
| nilq/python-mistral-1L-mini | 4,133,056 (1 layer, dim 64; embedding tables dominate) | 16,533,496 | not stated on card | 32,000 | none | none | none on card | Yes | Yes | https://huggingface.co/nilq/python-mistral-1L-mini [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| CinnabarLM-4M-Python-Base | 4,229,568 | 16,924,408 | OLMo-Coding/starcoder-python-instruct | BPE 4,096 | none | none | Apache-2.0 | Yes | Yes | https://huggingface.co/MihaiPopa-1/CinnabarLM-4M-Python-Base [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| Pico-1.3-Coding-Large | 6,917,376 | 13,843,136 | 28k short tasks from OpenCodeInstruct + 1,006 generated + chat | BPE 8,192 | not tested (card: "expect about 0 %") | not tested; own sets: 68% (50 seen-kind), 10% (2/20 unseen names) | Apache-2.0 | Yes | Yes | https://huggingface.co/Convergent-ch/Pico-1.3-Coding-Large [VERIFIED] |
| Mxode/minicoder-7M-base | 8,617,344 | 34,473,600 | not stated (empty card) | not stated | none | none | none on card | Yes | Yes | https://huggingface.co/Mxode/minicoder-7M-base [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| BananaMind-2.1-NanoCoder | 9,895,690 (card); 12,254,978 (hub metadata) | 49,037,296 | 30 B tokens: 75% The Stack v3, 25% FineWeb-Edu | BPE 8,192 | none | none | Apache-2.0 | Yes | Yes | https://huggingface.co/Banaxi-Tech/BananaMind-2.1-NanoCoder [VERIFIED] |
| jfrick/scholarly-code-modern-gpt-10m-5b | 9,849,536 (card); 16,140,992 (hub metadata) | 64,573,288 | 5 B token exposures, 30% Python | BPE 32,768 | none (Python validation perplexity 8.47) | none | none on card | Yes | Yes | https://huggingface.co/jfrick/scholarly-code-modern-gpt-10m-5b [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| Indenta-13M-Python | about 13 M per card (weight file is named indenta-9m.gguf) | 27,117,376 (GGUF) | flytech/python-codes-25k | custom | none | none | CC BY-NC 4.0 | Yes | Yes | https://huggingface.co/Rohanify/Indenta-13M-Python [VERIFIED] (13,258,752 parameters in the GGUF metadata, bytes, licence field via hub API; card text not re-read) |
| Rashik24/tinycoder-15M | 15,191,712 | 60,772,984 | nampdn-ai/tiny-codes | 32,000 | none | none | none on card | Yes | Yes | https://huggingface.co/Rashik24/tinycoder-15M [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| CHISPA-20M-python | 20,119,360 | 80,489,416 | codeparrot-clean, permissive licences, 521 M tokens | byte BPE 16,384, Python-lexeme pretokeniser | none (line exact match 1.2%, leading identifier 17.1%) | none | MIT | Yes | Yes | https://huggingface.co/esraderey/chispa-20m-python [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| stentor_python_30m | 30,419,712 | 121,699,864 | 872 examples that include MBPP tasks | not stated | none | unusable (MBPP in training); own categories 33-100% | Apache-2.0 | Yes | Over limit | https://huggingface.co/CastIronMind/stentor_python_30m [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| PyBlissa-Coder-40M | 39.9 M | 160,463,648 (GGUF F32) | 272 M tokens: OpenCodeInstruct + python-codes-25k | byte BPE 16,000 | 14.6% pass@1 (24/164, T = 0.25), self-reported | 4.4% (22/500, tests in prompt, T = 0.05), self-reported | Apache-2.0 | Yes | Over limit | https://huggingface.co/Rohanify/PyBlissa-Coder-40M-GGUF [VERIFIED] (card text; scores self-reported) |
| PyBlissa-Coder-50M | 50.2 M | 201,685,248 (GGUF F32) | 166 M tokens, same two sources | byte BPE 16,000 | 10.4% pass@1, self-reported | none | MIT | Yes | Over limit | https://huggingface.co/Rohanify/PyBlissa-Coder-50M [VERIFIED] (50,284,800 parameters in the GGUF metadata, bytes, licence field via hub API; card text not re-read) |
| GoldWorm v8 coder | 85,169,664 | 340,690,184 | 108 MB of Python package source + 324 distilled pairs | bytes, 256 | none (own 30 tasks: 7.08% pass@1, 16.67% pass@16, 92.92% valid) | none | MIT | Yes | Over limit | https://huggingface.co/Uuuuuuniiiiiii/goldworm-v8-coder-85m [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| LittleKedi-medium-wide-sparse-instruct (general chat model) | 305.4 M total, 31.7 M active per token | 1,221,460,040 | 20.4 B token general mix incl. github-code-clean | byte BPE 16,384 | none | 3.0% of 500 held-out, self-reported (moderate variant, 173.4 M total / 19.4 M active: 1.8%) | Apache-2.0 | Yes | Over limit (q8 GGUF is 336,139,232 bytes) | https://huggingface.co/edededdy/LittleKedi-medium-wide-sparse-instruct [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| stories260K (not code) | about 260 k | 1,056,540 | TinyStories | SentencePiece 512 | not evaluated | not evaluated | MIT | Yes | Yes (reference shape, expected 0) | https://huggingface.co/karpathy/tinyllamas [VERIFIED] |
| Pico-1 / Pico-1.2 (not code) | 254,400 / 460,608 | 513,984 / 929,376 | 108,735 / 153,974 chats | not read | not evaluated | not evaluated | Apache-2.0 | Yes | Yes | https://huggingface.co/Convergent-ch/Pico-1.2 [VERIFIED] (the Pico-1 byte count was not re-checked) |
| ternary.c64 (not code) | 688,128 ternary weights (820,352 parameters in hub metadata) | 242,324 (packed weights; cartridge image 361,216) | TinyStories | not read | not evaluated | not evaluated | MIT | Yes | Yes | https://huggingface.co/gdm32569/ternary-c64-688k [VERIFIED] |

### 6.2 Reference ladder, about 100 M to 1.7 B

| Model | Parameters | Bytes on disk | Training data | Tokenizer / vocabulary | HumanEval (pass@1 / @10 / @100) | MBPP | Licence | Weights public? | Runnable locally? | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| codeparrot-small | 110 M | 456,677,609 | codeparrot-clean GitHub Python, about 29 B tokens | BPE 32,768 | 3.80 / 6.57 / 12.78 | none | Apache-2.0 | Yes | Over limit | https://huggingface.co/codeparrot/codeparrot-small [VERIFIED] |
| PyCodeGPT-110M | 110 M | 495,109,521 | 96 GB, 13 M Python files | BPE 32,000 | 8.33 / 13.36 / 19.13 | none | AFL-3.0 (card); MIT (GitHub repo) | Yes | Over limit | https://arxiv.org/abs/2206.06888 [VERIFIED] |
| CodeGPT-small-py | 124 M | 509,592,250 | not checked | 50,001 | 0 / 0 / 0 for CodeGPT-Adapted (CERT paper) | none | none on card | Yes | Over limit | https://huggingface.co/microsoft/CodeGPT-small-py [VERIFIED] (bytes, licence field, CERT zero) |
| GPT-Neo-125M | 125 M | 525,979,192 | The Pile (8% GitHub code) | GPT-2 BPE 50,257 | 0.75 / 1.88 / 2.97 | none | MIT | Yes | Over limit | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| SmolLM-135M-Instruct | 134,515,008 | 269,060,552 | Cosmopedia v2, Python-Edu (4 B tokens), FineWeb-Edu | 49,152 | 7.7 ± 0.8 / 14.5 ± 1.0 (LintSeq authors' run) | MBPP(+) 10.1 ± 1.8 / 14.6 ± 0.5 | Apache-2.0 | Yes | Over limit | https://arxiv.org/abs/2410.02749 [VERIFIED] (scores; the parameter and byte cells were not re-checked for the -Instruct repository) |
| SmolLM2-135M | 134,515,008 | 269,060,552 | SmolLM2 data mix, 2 T tokens (paper section 6) | 49,152 | not in paper or card | none | Apache-2.0 | Yes | Over limit | https://huggingface.co/HuggingFaceTB/SmolLM2-135M [VERIFIED] (parameters, bytes, licence field via hub API; card text not re-read) |
| TinyCodeLM-150M | 150 M | 607,539,201 | 72 B tokens FineWeb + The Stack Python | GPT-NeoX BPE 50,304 | 6.1 (T = 0, pretrained); 12.8 / 20.6 (LintSeqInstruct) | MBPP(+) 5.4 pretrained; 13.6 / 24.4 tuned | Apache-2.0 | Yes (needs the OLMo model code) | Over limit | https://huggingface.co/upiter/TinyCodeLM-150M [VERIFIED] |
| PolyCoder-160M | 160 M | 699,645,960 | 249 GB code, 12 languages | GPT-2 style BPE 50,304 | 2.13 / 3.35 / 4.88 | none | none on card; MIT (GitHub repo) | Yes | Over limit | https://arxiv.org/abs/2202.13169 [VERIFIED] (scores as listed in CERT Table 6, bytes, licence field) |
| tiny_starcoder_py | 164,144,128 | 656,601,304 | StarCoderData Python, 100 B tokens | 49,152 | 7.84 pass@1 | none | BigCode OpenRAIL-M | Yes | Over limit | https://huggingface.co/bigcode/tiny_starcoder_py [VERIFIED] |
| CodeT5+ 220M | 220 M | 445,800,957 | not checked | 32,100 | 12.0 / 20.7 / 31.6 | none | BSD-3-Clause | Yes | Over limit | https://arxiv.org/abs/2305.07922 [VERIFIED] (12.0 / 20.7 via LintSeq Table 1, bytes, licence field; 31.6 not re-checked) |
| Codex-300M | 300 M (label; matches the GPT-3 Medium shape without embeddings, section 2) | not released | as Codex-12M | as Codex-12M | 13.17 / 20.37 / 36.27 | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| AlphaCode decoder-only 302M | 302 M | not released | as 13M | SentencePiece 8,000 | 11.6 / 18.8 / 31.8 | none | n/a | No | No | https://arxiv.org/abs/2203.07814 [VERIFIED] |
| codegen-350M-mono | 350 M | 797,367,631 | The Pile, then BigQuery, then BigPython | BPE 51,200 | 12.76 / 23.11 / 35.19 (highest over T of 0.2, 0.6, 0.8) | sanitized MBPP, n = 100, T = 0.8: 14.59 / 41.49 / 63.00 (paper Table 11); MBPP(+) 9.4 / 15.2 (LintSeq run) | BSD-3-Clause | Yes | Over limit | https://arxiv.org/abs/2203.13474 [VERIFIED] |
| codegen-350M-multi / -nl | 350 M | not checked | BigQuery / The Pile | not checked | 6.67 / 10.61 / 16.84; 2.12 / 4.10 / 7.38 | sanitized MBPP, n = 100, T = 0.8: 7.46 / 24.18 / 46.37; 0.96 / 6.37 / 19.91 | BSD-3-Clause | Yes | not checked | https://arxiv.org/abs/2203.13474 [VERIFIED] |
| phi-1-small | 350 M | not released | CodeTextbook + CodeExercises | codegen-350M-mono tokenizer | 45 pass@1 | not stated | n/a | No | No | https://arxiv.org/abs/2306.11644 [VERIFIED] |
| SmolLM-360M-Instruct | 361,821,120 | not checked | as SmolLM-135M | not checked | 11.3 (authors) / 19.3 ± 1.1 (LintSeq run) | MBPP(+) 19.4 ± 2.4 / 23.1 ± 0.5 | Apache-2.0 | Yes | Over limit | https://arxiv.org/abs/2410.02749 [VERIFIED] (scores) |
| TinyCodeLM-400M | 400 M | 1,816,698,449 | as TinyCodeLM-150M | GPT-NeoX BPE (same as 150M per paper) | 6.7 pretrained; 13.4 / 20.9 tuned | MBPP(+) 6.8; 19.4 / 29.9 | Apache-2.0 | Yes | No (file larger than free disk) | https://huggingface.co/upiter/TinyCodeLM-400M [VERIFIED] |
| Qwen2.5-Coder-0.5B | 494,032,768 | 988,097,824 | Qwen2.5-Coder corpus | 151,936 | 28.0 (HumanEval+ 23.8), EvalPlus harness | 52.9 (MBPP+ 47.1) | Apache-2.0 | Yes | Over limit | https://arxiv.org/abs/2409.12186 [VERIFIED] (parameters, bytes, licence field via hub API; scores not re-checked) |
| Codex-679M | 679 M (label; matches the GPT-3 Large shape without embeddings, section 2) | not released | as Codex-12M | as Codex-12M | 16.22 / 25.7 / 40.95 | none | n/a | No | No | https://arxiv.org/abs/2107.03374 [VERIFIED] |
| CodeT5+ 770M | 770 M | not checked | as 220M | not checked | 15.5 / 27.2 / 42.7 | none | BSD-3-Clause | Yes | not checked | https://arxiv.org/abs/2305.07922 |
| SantaCoder | 1.1 B | 4,600,328,417 | 236 B tokens (phi-1 Table 1) | not checked | 14.0 pass@1 (as listed in phi-1 Table 1) | 35.0 (same table) | BigCode OpenRAIL-M | Yes | No (file size) | https://arxiv.org/abs/2306.11644 [VERIFIED] (scores and token count as listed in phi-1 Table 1) |
| StarCoderBase-1B | 1,137,207,296 | 4,548,859,752 | not checked | not checked | 15.17 pass@1 (hub model-index, self-reported, "verified: false") | none | BigCode OpenRAIL-M | Gated | No (file size) | https://huggingface.co/bigcode/starcoderbase-1b [VERIFIED] |
| InCoder 1.3B | 1.3 B | not checked | not checked | not checked | 8.9 / 16.7 / 25.6 (as listed in CodeT5+ Table 2) | none | not checked | Yes | not checked | https://arxiv.org/abs/2305.07922 |
| GPT-Neo 1.3B | 1.3 B | not checked | The Pile | not checked | 4.79 / 7.47 / 16.30 | none | MIT | Yes | not checked | https://arxiv.org/abs/2107.03374 [VERIFIED] (scores) |
| DeepSeek-Coder-1.3B-base | 1.3 B | 2,693,014,838 | DeepSeek-Coder corpus | not checked | 34.8 (HumanEval+ 26.8), as listed in the Qwen2.5-Coder report | 55.6 (MBPP+ 46.9) | "other" (DeepSeek licence) | Yes | No (file size) | https://arxiv.org/abs/2409.12186 |
| phi-1 | 1,418,270,720 | 2,836,578,696 | under 7 B tokens CodeTextbook + CodeExercises | 51,200 | 50.6 pass@1 | 55.5 pass@1 | MIT | Yes | No (file size) | https://arxiv.org/abs/2306.11644 [VERIFIED] |
| Qwen2.5-Coder-1.5B | 1.5 B | not checked | Qwen2.5-Coder corpus | not checked | 43.9 (HumanEval+ 36.6) | 69.2 (MBPP+ 58.6) | Apache-2.0 | Yes | No (file size) | https://arxiv.org/abs/2409.12186 |
| SmolLM2-1.7B | 1.7 B | not checked | 11 T tokens (paper Table 8) | not checked | 22.6 pass@1 | none | Apache-2.0 | Yes | No (file size) | https://arxiv.org/abs/2502.02737 |

### 6.3 Non-neural and small-memory baselines (token prediction)

| Model | Corpus | Setting | Entropy, bits per token | Ranking accuracy | Memory | Source |
|---|---|---|---|---|---|---|
| Smoothed 3-/4-gram | 10 Java projects | self, 10-fold | 3 to 4 (English: 10 falling to under 8) | not given | not given | https://people.inf.ethz.ch/suz/publications/natural.pdf |
| Plain n-gram | Java | static / dynamic / maintenance | 3.94 / 2.64 / 2.32 | not read | not given | https://web.cs.ucdavis.edu/~devanbu/isDLgood.pdf [VERIFIED] |
| Cache n-gram | Java | same | 2.45 / 1.89 / 1.73 | not read | not given | same [VERIFIED] |
| Nested cache n-gram | Java | dynamic / maintenance | 1.70 / 1.41 | MRR 76.2% / 78.8% | not given | same [VERIFIED] (entropy; the MRR cell was not re-checked) |
| LSTM/300, LSTM/650 | Java (1% of corpus, vocabulary 74,046) | static / dynamic | 3.22 / 1.84; 3.03 / 1.91 | not read | not given | same [VERIFIED] (entropy) |
| n-gram | Python, small train | static; dynamic | 5.30; 4.81 | MRR 43.63%; 47.39% | 150-500 MB disk, about 5 GB RAM | https://arxiv.org/abs/2003.07914 [VERIFIED] |
| Cache n-gram | Python, small train | dynamic | 3.22 | MRR 62.27% | as above | same [VERIFIED] |
| Nested cache n-gram | Python, small train; large train | dynamic | 2.89; 1.45 | MRR 65.97%; 71.22% | large: 6-8.5 GB disk, 50-60 GB RAM | same [VERIFIED] |
| GRU-512 on BPE (one layer) | Python, small train; large train | static; dynamic | 3.91; 2.72 (small); 3.04; 2.14 (large) | MRR 81.66%; 86.28%; 84.31%; 87.06% | 15-45 MB disk, 250-400 MB RAM at inference | same [VERIFIED] |
| Neural reranker + static analysis | API-member completion | - | not given | top-5 90% | 6 MB RAM, 8 ms | https://arxiv.org/abs/2004.13651 [VERIFIED] |
| Pythia LSTM, float / int8 | Python API completion, 2,700 repos | - | not given | top-5 92% / 89% | 152 MB / 38 MB | https://arxiv.org/abs/1912.00742 [VERIFIED] |

### 6.4 Size of benchmark inputs in bytes (our computation from the dataset files, 2026-11-13)

Tokens = bytes divided by the bytes per token of the LS-LLM tokenizer, which has to be measured. The MBPP prompt
counts exclude the fixed instruction wrapper of the standard prompt.

| Quantity | min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|---|
| HumanEval prompt (164 tasks) | 115 | 288 | 396 | 571 | 1,360 | 451.1 |
| HumanEval canonical solution body | 16 | 78 | 149 | 247 | 864 | 180.9 |
| HumanEval prompt + solution | 132 | 421 | 576 | 781 | 1,993 | 632.0 |
| MBPP test text (500 tasks) | 37 | 63 | 77 | 89 | 191 | 78.6 |
| MBPP test text + 3 asserts | 124 | 184 | 226 | 299 | 4,293 | 269.4 |
| MBPP test reference code | 37 | 100 | 146 | 222 | 1,331 | 183.3 |

| Budget in bytes | HumanEval prompts that fit | HumanEval prompt + solution | MBPP test text + asserts | MBPP test text + asserts + code |
|---|---|---|---|---|
| 256 | 31 / 164 | 11 / 164 | 315 / 500 | 52 / 500 |
| 384 | 75 | 34 | 442 | 215 |
| 512 | 118 | 64 | 483 | 372 |
| 768 | 147 | 122 | 495 | 470 |
| 1,024 | 158 | 146 | 498 | 492 |

HumanEval canonical solution bodies with 1 non-blank line: 37; at most 2: 47; at most 3: 57; at most 5: 82. Bodies
of at most 40 bytes: 16; at most 80 bytes: 43. MBPP test reference code with at most 3 non-blank lines: 149;
at most 5: 255.

[VERIFIED] Recomputed by the fact-checker from the same two files on 2026-11-13: every number in the two tables
and in the paragraph above reproduces. The p25, median and p75 columns are index-based (sorted[n//4],
sorted[n//2], sorted[3n//4]); interpolated medians are 148.5, 574.5 and 225.5 instead of 149, 576 and 226. The
last column of the second table adds text, asserts and code with no separator between asserts and code.

### 6.5 Extrapolation of published size curves to our scale (our own fit; not a published result)

Least-squares line of log(pass@k) against log2(parameters).

| Points used | Metric | Factor per doubling of parameters | 0.3 M | 1 M | 3 M | 10 M |
|---|---|---|---|---|---|---|
| Codex 12M, 25M, 42M, 85M | pass@1 | 1.66 | 0.13% | 0.32% | 0.71% | 1.72% |
| Codex 12M to 300M | pass@1 | 1.51 | 0.24% | 0.49% | 0.95% | 1.95% |
| AlphaCode 13M, 29M, 55M, 89M | pass@1 | 1.46 | 0.22% | 0.43% | 0.78% | 1.51% |
| AlphaCode 13M to 302M | pass@1 | 1.51 | 0.18% | 0.36% | 0.70% | 1.44% |
| Codex 12M to 85M | pass@10 | 1.55 | 0.38% | 0.82% | 1.64% | 3.51% |
| AlphaCode 13M to 89M | pass@10 | 1.54 | 0.34% | 0.72% | 1.43% | 3.01% |
| Codex 12M to 85M | pass@100 | 1.40 | 1.47% | 2.63% | 4.47% | 7.98% |
| AlphaCode 13M to 89M | pass@100 | 1.37 | 1.49% | 2.58% | 4.27% | 7.40% |
| Codex 12M to 300M (row added in the fact-check) | pass@100 | 1.36 | 1.73% | 2.96% | 4.83% | 8.26% |
| AlphaCode 13M to 302M (row added in the fact-check) | pass@100 | 1.34 | 1.71% | 2.86% | 4.57% | 7.64% |

One HumanEval problem is 0.61%. At 0.3 M the fits give 0.2 to 0.4 problems at pass@1 and 2 to 3 problems at
pass@100 (a logit-scale fit gives 0.12 to 0.21% and 1.17 to 1.25%). These fits assume the training data,
vocabulary and context of Codex and AlphaCode, which Tier A does not have, so they are an optimistic guide and
not a forecast.

[VERIFIED] All eight original rows and the logit-scale values were recomputed in the fact-check from the
published points and reproduce to the printed digits. The two added rows are where the "1.7%" upper value of
section 1 comes from. The Codex paper counts model size without embeddings (section 2), so the x-axis of the
Codex rows is read as non-embedding parameters; a 0.3 M model with a 512 x 64 token table has about 0.27 M of
them, which lowers the Codex-based values by under a tenth.

## 7. Sources

Papers
| Title | Authors | Year | URL | Licence of code / model / data |
|---|---|---|---|---|
| Evaluating Large Language Models Trained on Code | Chen, Tworek, Jun, Yuan et al. (58 authors) | 2021 | https://arxiv.org/abs/2107.03374 | HumanEval: MIT (https://github.com/openai/human-eval); Codex weights not released [VERIFIED] (HumanEval licence) |
| Language Models are Few-Shot Learners (row added in the fact-check; used only for the model shapes in its Table 2.1) | Brown, Mann, Ryder, Subbiah et al. (31 authors) | 2020 | https://arxiv.org/abs/2005.14165 | paper only [VERIFIED] |
| Program Synthesis with Large Language Models | Austin, Odena, Nye, Bosma et al. (11 authors) | 2021 | https://arxiv.org/abs/2108.07732 | MBPP: CC BY 4.0 (https://huggingface.co/datasets/google-research-datasets/mbpp); models not released [VERIFIED] (MBPP licence) |
| Competition-Level Code Generation with AlphaCode | Li, Choi, Chung, Kushman et al. (26 authors) | 2022 | https://arxiv.org/abs/2203.07814 | models not released |
| CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis | Nijkamp, Pang, Hayashi, Tu et al. (8 authors) | 2022 | https://arxiv.org/abs/2203.13474 | models BSD-3-Clause (https://huggingface.co/Salesforce/codegen-350M-mono) |
| CERT: Continual Pre-Training on Sketches for Library-Oriented Code Generation | Zan, Chen, Yang, Lin et al. (9 authors) | 2022 | https://arxiv.org/abs/2206.06888 | PyCodeGPT: AFL-3.0 on the hub card, MIT on https://github.com/microsoft/PyCodeGPT [VERIFIED] (both licence fields) |
| A Systematic Evaluation of Large Language Models of Code | Xu, Alon, Neubig, Hellendoorn | 2022 | https://arxiv.org/abs/2202.13169 | code MIT (https://github.com/VHellendoorn/Code-LMs); no licence on the hub card [VERIFIED] (both licence fields) |
| Training Language Models on Synthetic Edit Sequences Improves Code Synthesis | Piterbarg, Pinto, Fergus | 2024 (ICLR 2025) | https://arxiv.org/abs/2410.02749 | TinyCodeLM: Apache-2.0 |
| Textbooks Are All You Need | Gunasekar, Zhang, Aneja, Mendes et al. (19 authors) | 2023 | https://arxiv.org/abs/2306.11644 | phi-1: MIT; phi-1-small and CodeExercises not released |
| CodeT5+: Open Code Large Language Models for Code Understanding and Generation | Wang, Le, Gotmare, Bui et al. (6 authors) | 2023 | https://arxiv.org/abs/2305.07922 | BSD-3-Clause |
| Qwen2.5-Coder Technical Report | Hui, Yang, Cui, Yang et al. (24 authors) | 2024 | https://arxiv.org/abs/2409.12186 | Qwen2.5-Coder-0.5B: Apache-2.0 |
| SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model | Ben Allal, Lozhkov, Bakouch, Martín Blázquez et al. (22 authors) | 2025 | https://arxiv.org/abs/2502.02737 | Apache-2.0; smollm-corpus ODC-BY |
| Curriculum Learning for Small Code Language Models | Naïr, Yamani, Said Lhadj, Baghdadi | 2024 (ACL SRW) | https://arxiv.org/abs/2407.10194 | code and 1 M checkpoints Apache-2.0 (https://github.com/kamel-yamani/CurriculumLearningForSmallCodeLanguageModels); data on Kaggle, licence not checked [VERIFIED] (repository licence) |
| Automatic Generation of Python Programs Using Context-Free Grammars | Yamani, Naïr, Baghdadi | 2024 | https://arxiv.org/abs/2403.06503 | https://github.com/MarwaNair/TinyPy-Generator has no licence file [VERIFIED] (repository has no licence) |
| On the Naturalness of Software | Hindle, Barr, Gabel, Su, Devanbu | 2012 (ICSE; author copy with an expanded future-work section) | https://people.inf.ethz.ch/suz/publications/natural.pdf | paper only |
| Are Deep Neural Networks the Best Choice for Modeling Source Code? | Hellendoorn, Devanbu | 2017 (FSE) | https://web.cs.ucdavis.edu/~devanbu/isDLgood.pdf | SLP-Core: MIT (https://github.com/SLP-team/SLP-Core) [VERIFIED] (SLP-Core licence) |
| Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code | Karampatsis, Babii, Robbes, Sutton, Janes | 2020 (ICSE) | https://arxiv.org/abs/2003.07914 | not checked |
| Fast and Memory-Efficient Neural Code Completion | Svyatkovskiy, Lee, Hadjitofi, Riechert, Franco, Allamanis | 2020 (MSR 2021) | https://arxiv.org/abs/2004.13651 | not checked |
| Pythia: AI-assisted Code Completion System | Svyatkovskiy, Zhao, Fu, Sundaresan | 2019 | https://arxiv.org/abs/1912.00742 | not released |
| Tracking Universal Features Through Fine-Tuning and Model Merging (metadata only, paper not read) | Horn, Elliott | 2024 | https://arxiv.org/abs/2410.12391 | nilq/baby-python: MIT |
| OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs (metadata and dataset card only, paper not read) | Ahmad, Ficek, Samadi, Huang et al. (7 authors) | 2025 | https://arxiv.org/abs/2504.04030 | dataset CC BY 4.0 |
| The Unreasonable Effectiveness of Recurrent Neural Networks (blog) | Karpathy | 2015 | https://karpathy.github.io/2015/05/21/rnn-effectiveness/ | blog |
| SmolLM - blazingly fast and remarkably powerful (blog) | Ben Allal, Lozhkov, Bakouch | 2024 | https://huggingface.co/blog/smollm | blog |

Models (hub cards read on 2026-11-13)
| Model | URL | Licence |
|---|---|---|
| codeparrot/codeparrot-small | https://huggingface.co/codeparrot/codeparrot-small | Apache-2.0 |
| Daoguang/PyCodeGPT | https://huggingface.co/Daoguang/PyCodeGPT | AFL-3.0 |
| bigcode/tiny_starcoder_py | https://huggingface.co/bigcode/tiny_starcoder_py | BigCode OpenRAIL-M |
| upiter/TinyCodeLM-150M and -400M | https://huggingface.co/upiter/TinyCodeLM-150M | Apache-2.0 |
| HuggingFaceTB/SmolLM-135M, SmolLM2-135M, SmolLM2-360M | https://huggingface.co/HuggingFaceTB/SmolLM2-135M | Apache-2.0 |
| Salesforce/codegen-350M-mono | https://huggingface.co/Salesforce/codegen-350M-mono | BSD-3-Clause |
| EleutherAI/gpt-neo-125m | https://huggingface.co/EleutherAI/gpt-neo-125m | MIT |
| microsoft/CodeGPT-small-py | https://huggingface.co/microsoft/CodeGPT-small-py | none on card |
| NinedayWang/PolyCoder-160M | https://huggingface.co/NinedayWang/PolyCoder-160M | none on card |
| Salesforce/codet5p-220m | https://huggingface.co/Salesforce/codet5p-220m | BSD-3-Clause |
| microsoft/phi-1 | https://huggingface.co/microsoft/phi-1 | MIT |
| Qwen/Qwen2.5-Coder-0.5B | https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B | Apache-2.0 |
| amd/AMD-Llama-135m-code (draft model, no standalone score) | https://huggingface.co/amd/AMD-Llama-135m-code | Apache-2.0 |
| BEE-spoke-data/smol_llama-101M-GQA-python (no score on card) | https://huggingface.co/BEE-spoke-data/smol_llama-101M-GQA-python | Apache-2.0 |
| Convergent-ch/Pico-1.3-Coding-Large, Pico-1, Pico-1.2 | https://huggingface.co/Convergent-ch/Pico-1.3-Coding-Large | Apache-2.0 |
| Rohanify/PyBlissa-Coder-40M-GGUF, PyBlissa-Coder-50M, Indenta-13M-Python | https://huggingface.co/Rohanify/PyBlissa-Coder-40M-GGUF | Apache-2.0, MIT, CC BY-NC 4.0 |
| MihaiPopa-1/CinnabarLM-4M-Python-Base | https://huggingface.co/MihaiPopa-1/CinnabarLM-4M-Python-Base | Apache-2.0 |
| nilq/python-mistral-1L-mini | https://huggingface.co/nilq/python-mistral-1L-mini | none on card |
| Mxode/minicoder-7M-base | https://huggingface.co/Mxode/minicoder-7M-base | none on card |
| Banaxi-Tech/BananaMind-2.1-NanoCoder | https://huggingface.co/Banaxi-Tech/BananaMind-2.1-NanoCoder | Apache-2.0 |
| jfrick/scholarly-code-modern-gpt-10m-5b | https://huggingface.co/jfrick/scholarly-code-modern-gpt-10m-5b | none on card |
| Rashik24/tinycoder-15M | https://huggingface.co/Rashik24/tinycoder-15M | none on card |
| esraderey/chispa-20m-python | https://huggingface.co/esraderey/chispa-20m-python | MIT |
| CastIronMind/stentor_python_30m | https://huggingface.co/CastIronMind/stentor_python_30m | Apache-2.0 |
| Uuuuuuniiiiiii/goldworm-v8-coder-85m | https://huggingface.co/Uuuuuuniiiiiii/goldworm-v8-coder-85m | MIT |
| edededdy/LittleKedi-*-instruct | https://huggingface.co/edededdy/LittleKedi-medium-wide-sparse-instruct | Apache-2.0 |
| karpathy/tinyllamas (stories260K) | https://huggingface.co/karpathy/tinyllamas | MIT |
| gdm32569/ternary-c64-688k | https://huggingface.co/gdm32569/ternary-c64-688k | MIT |
| codefusionduo/suku-ai-128b-titan (cautionary example only) | https://huggingface.co/codefusionduo/suku-ai-128b-titan | MIT |
| MatterWaveAI/grt-30m-code (cautionary example only) | https://huggingface.co/MatterWaveAI/grt-30m-code | Apache-2.0 |
| DrunkkToys/theOG-50M (lead added in the fact-check, section 3) | https://huggingface.co/DrunkkToys/theOG-50M | Apache-2.0 |

[VERIFIED] Every repository in this table exists and its licence field matches the hub API on 2026-11-13, except
HuggingFaceTB/SmolLM-135M, HuggingFaceTB/SmolLM2-360M and Convergent-ch/Pico-1, which were not queried.

Datasets and code
| Name | URL | Licence | Note |
|---|---|---|---|
| HumanEval | https://github.com/openai/human-eval | MIT | 164 tasks; test set, never train or tune on it [VERIFIED] |
| MBPP | https://huggingface.co/datasets/google-research-datasets/mbpp | CC BY 4.0 | 974 tasks; 374 train, 90 validation, 500 test, 10 prompt [VERIFIED] |
| HumanEval+ / MBPP+ (EvalPlus) | https://huggingface.co/datasets/evalplus/humanevalplus | Apache-2.0 | stricter tests |
| nvidia/OpenCodeInstruct | https://huggingface.co/datasets/nvidia/OpenCodeInstruct | CC BY 4.0 | 5,000,000 samples, 6.86 GB download, has unit tests and test status fields |
| nampdn-ai/tiny-codes | https://huggingface.co/datasets/nampdn-ai/tiny-codes | MIT (gated) | 1 M to 10 M size class |
| flytech/python-codes-25k | https://huggingface.co/datasets/flytech/python-codes-25k | [UNVERIFIED: no licence field on card] | used by PyBlissa and Indenta |
| codeparrot/codeparrot-clean | https://huggingface.co/datasets/codeparrot/codeparrot-clean | none on card (per-file licences) | GitHub Python |
| HuggingFaceTB/smollm-corpus (Python-Edu) | https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus | ODC-BY | 4 B tokens of educational Python |
| nilq/baby-python | https://huggingface.co/datasets/nilq/baby-python | MIT | BabyLM 100M + matched Python |
| TinyPy Generator | https://github.com/MarwaNair/TinyPy-Generator | no licence file | grammar-based generator; re-implement, do not copy [VERIFIED] |
| bigcode-evaluation-harness | https://github.com/bigcode-project/bigcode-evaluation-harness | Apache-2.0 | reference harness [VERIFIED] |
| llama2.c | https://github.com/karpathy/llama2.c | MIT | baseline shape [VERIFIED] |

## Verification log

Fact-check of 2026-11-13 by a second agent: 25 claims checked, 20 confirmed as written, 5 corrected, no invented
citation found. Paper PDFs were streamed through pdftotext, hub records and cards read through the hub API and raw
files, dataset files parsed for aggregate counts only; nothing was stored. Format: claim: verdict. URL.

- 1. Codex-12M HumanEval 2.00 / 3.62 / 8.58 and the 25M to 679M rows (Table 1): confirmed. https://arxiv.org/abs/2107.03374
- 2. Codex setup (159 GB, 54 M repositories, 100 B tokens, n = 200, p = 0.95, about 30% fewer tokens) and "(12 M parameters)": numbers confirmed; corrected "12 M parameters" -> "12M" is a non-embedding count, total size not stated; corrected "highest result over temperatures" -> T = 0.2 and T = 0.8 named as optimal, the highest-over-temperatures rule is stated for the GPT-Neo rows only. https://arxiv.org/abs/2107.03374 and https://arxiv.org/abs/2005.14165
- 3. AlphaCode decoder-only 13M to 302M (1.5 / 3.6 / 8.6 at 13M), 1000 samples, 10.7 + 43.6 GB of Python, SentencePiece 8,000: confirmed. https://arxiv.org/abs/2203.07814
- 4. GPT-Neo-125M 0.75 / 1.88 / 2.97, TabNine 2.58 / 4.35 / 7.59, "GPT-3 solves 0%", GPT-Neo-125M file 525,979,192 bytes, MIT: confirmed. https://arxiv.org/abs/2107.03374 and https://huggingface.co/EleutherAI/gpt-neo-125m
- 5. CodeT5 (220M), CodeGPT-Adapted (124M), CodeClippy (125M) at 0% for pass@1, pass@10, pass@100: confirmed; corrected location "section 6" -> appendix A. https://arxiv.org/abs/2206.06888
- 6. PyCodeGPT-110M 8.33 / 13.36 / 19.13 (Table 6), 8.32 / 13.53 / 18.3 (card), PY150 token accuracies (Table 7), 96 GB and 13 M files, BPE 32k, 495,109,521 bytes, AFL-3.0: confirmed. https://arxiv.org/abs/2206.06888 and https://huggingface.co/Daoguang/PyCodeGPT
- 7. codeparrot-small 110M, 3.80 / 6.57 / 12.78, vocabulary 32,768, about 29 B tokens, 456,677,609 bytes, Apache-2.0: confirmed. https://huggingface.co/codeparrot/codeparrot-small
- 8. tiny_starcoder_py 164,144,128 parameters, pass@1 7.84% ("verified: false"), 100 B tokens, 656,601,304 bytes, BigCode OpenRAIL-M: confirmed. https://huggingface.co/bigcode/tiny_starcoder_py
- 9. TinyCodeLM-150M and -400M, SmolLM-Instruct, AlphaCode-89M, Codex-85M, CodeT5+ 220M, codegen-350M-mono rows of LintSeq Table 1; pretrained 6.1 / 5.4 and 6.7 / 6.8 on the card; 72 B tokens; 607,539,201 bytes: confirmed. https://arxiv.org/abs/2410.02749 and https://huggingface.co/upiter/TinyCodeLM-150M
- 10. phi-1 data ladder at 350 M (12.19% -> 17.68% -> 20.12%, phi-1-small 45%), phi-1 50.6% / 55.5%, SantaCoder 14.0 / 35.0: confirmed. https://arxiv.org/abs/2306.11644
- 11. 1 M TinyPy models (81.23% / 41.74% / 74.15; hybrid 81.13% / 42.04% / 74.65; output accuracy 74.58% -> 79.23%; vocabulary 41; Apache-2.0): confirmed; corrected "4,385,396 bytes each" -> 4,385,396 or 4,386,378 bytes; corrected "1.2 M snippets, 95.9 M training characters" -> 1.02 M training snippets with 95.9 M character tokens. https://arxiv.org/abs/2407.10194 and https://github.com/kamel-yamani/CurriculumLearningForSmallCodeLanguageModels
- 12. Pico-1.3-Coding-Large 6,917,376 parameters, 13,843,136 bytes, 68% and 2 of 20, "expect about 0 %": confirmed. https://huggingface.co/Convergent-ch/Pico-1.3-Coding-Large
- 13. PyBlissa-Coder-40M 14.6% HumanEval (24/164, T = 0.25), 4.4% MBPP (22/500), 39.9 M, 160,463,648 bytes, Apache-2.0: confirmed as card text (self-reported, not reproduced). https://huggingface.co/Rohanify/PyBlissa-Coder-40M-GGUF
- 14. stories260K 5 layers, dim 64, 8 heads, 4 KV heads, vocabulary 512, validation loss 1.2968, 1,056,540 bytes, MIT: confirmed. https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K
- 15. ternary.c64 688,128 ternary weights, 242,324-byte weight file, validation loss 1.439, MIT: confirmed; added the 361,216-byte cartridge image and 820,352 total parameters. https://huggingface.co/gdm32569/ternary-c64-688k
- 16. Section 6.4 byte statistics of HumanEval and the MBPP test split, HumanEval MIT, MBPP CC BY 4.0, MBPP splits 374 / 90 / 500 / 10: confirmed by recomputation. https://github.com/openai/human-eval and https://huggingface.co/datasets/google-research-datasets/mbpp
- 17. Section 6.5 extrapolation (0.1 to 0.25% pass@1 and 1.2 to 1.7% pass@100 at 0.3 M): confirmed by recomputation, two pass@100 rows added; "MBPP score of zero is the expected outcome": unsupported (no MBPP fit), tagged. https://arxiv.org/abs/2107.03374 and https://arxiv.org/abs/2203.07814
- 18. Hellendoorn and Devanbu Table 2 (plain 3.94 / 2.64 / 2.32, cache 2.45 / 1.89 / 1.73, nested cache 1.70 / 1.41, LSTM rows): confirmed. https://web.cs.ucdavis.edu/~devanbu/isDLgood.pdf
- 19. Karampatsis et al. Python rows (n-gram 5.30 bits / 43.63%, nested cache 2.89 / 65.97%, GRU-512 3.91 / 81.66% and 2.72 / 86.28%), disk 150 to 500 MB and 6 to 8.5 GB, RAM up to 50-60 GB, neural 15 to 45 MB: confirmed. https://arxiv.org/abs/2003.07914
- 20. Project numbers used in sections 4 and 5: "100 tokens/s" (idea 9): unsupported as a measurement, it is a design estimate, tagged; corrected "260k to 400k int8 parameters" -> 250k to 400k; corrected "1.4 to 1.7 GB free during this session" -> readings of 199 MiB and 3.9 GiB during the fact-check. research/CONTEXT.md and `df -h /`
- 21. StarCoderBase-1B HumanEval pass@1 15.17: confirmed (hub model-index, self-reported); corrected tag "[UNVERIFIED: from memory]" -> [VERIFIED]. https://huggingface.co/api/models/bigcode/starcoderbase-1b
- 22. Austin et al.: 244 M to 137 B non-embedding parameters, "around 80% of the time" syntactically correct, scoring on the three prompt asserts, sanitized set 427 (426 in the paper text): confirmed. https://arxiv.org/abs/2108.07732
- 23. codegen-350M mono / multi / nl HumanEval rows (12.76 / 23.11 / 35.19 for mono) and MBPP rows (14.59 / 41.49 / 63.00 for mono): numbers confirmed; corrected "MBPP" -> sanitized MBPP, n = 100, temperature 0.8. https://arxiv.org/abs/2203.13474
- 24. Pythia 8-bit quantisation 152 MB -> 38 MB with top-5 92% -> 89%; neural reranker 6 MB RAM, 8 ms, 90% top-5: confirmed. https://arxiv.org/abs/1912.00742 and https://arxiv.org/abs/2004.13651
- 25. Existence claims: all 35 hub model repositories cited exist with the stated parameter counts, weight-file sizes and licence fields; arXiv 2505.10222 is withdrawn; the BananaMind card describes hashed n-gram tables; GitHub licence fields of microsoft/PyCodeGPT, VHellendoorn/Code-LMs, SLP-team/SLP-Core, karpathy/llama2.c (all MIT) and bigcode-project/bigcode-evaluation-harness (Apache-2.0) match: confirmed. "No HumanEval or MBPP number below 12 M", "no n-gram model scored on these benchmarks", "smallest self-reported hub score at 39.9 M": no counterexample in a second search (hub dataset tags, arXiv API, web), so "not found" stands; one additional self-reported lead at 50 M (theOG-50M) added to section 3. https://huggingface.co/api/models and https://export.arxiv.org/api/query?id_list=2505.10222 and https://huggingface.co/DrunkkToys/theOG-50M

Not re-checked (left without a tag): the Qwen2.5-Coder, DeepSeek-Coder, SmolLM2, InCoder and CodeT5+ 770M rows and the
CodeT5+ 220M pass@100; the PolyCoder paper itself and its corpus figures; Hindle et al.; the Karpathy LSTM note; the
score text of the CHISPA, GoldWorm, LittleKedi, stentor, suku-ai and grt-30m cards; the Pico-1 file size; the SmolLM
blog sampling setting; the nvidia/OpenCodeInstruct size; the hub search counts in "How the hub was searched".
