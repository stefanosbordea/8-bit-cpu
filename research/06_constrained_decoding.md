# 06. Grammar-constrained decoding for code

Research date: 2026-11-13. Literature numbers were read in the cited source (URL at the end of each bullet; "abstract
only" where only the abstract was read). "Local" numbers come from `research/06_local_measurements.py`, run on a proxy
corpus: 9,469 functions (4,640,832 characters, 529 files) of the CPython 3.12.13 standard library after `lsllm.canon`.
The proxy corpus is not the training corpus; every local number has to be measured again on the real corpus.

Fact check (2026-11-13, second agent). `[VERIFIED]` after a bullet or sentence: its numbers and quotations were compared
again with the cited source; "(re-run)" means that the script was run again and printed the same values.
`[UNVERIFIED: reason]`: not supported by a source or a measurement. Bullets without a tag were not re-checked. Every
change is listed in "Verification log" at the end.

```
.venv/bin/python research/06_local_measurements.py 2>/dev/null
```

Two words are used in opposite ways in the literature. SynCode calls a filter "sound" when it never removes a valid
token and "complete" when it removes every invalid one. Biagiola et al. call a filter "incomplete" when it rejects valid
programs. Below: **never-blocks-valid** (keeps every valid continuation) and **catches-all** (removes every invalid one).

## 1. Answer in one paragraph

Masking grammatically impossible tokens removes nearly all syntax errors (SynCode: 96% fewer on Python and Go, the rest
are outputs cut off by the token limit) [VERIFIED], but it does not make programs correct: for CodeGen-350M on
HumanEval, pass@1 moved from 6.8% to 6.9% [VERIFIED], although the filter removed 256 of that model's 271 syntactically
broken HumanEval samples (over HumanEval and MBXP together, 349 of its 4,154 unconstrained samples, 8.4%, had a syntax
error; the paper's average over its three models is 6%). A sample that is repaired syntactically is still almost always
functionally wrong; that reading is ours, the paper names no cause. The gain is large
only where the unconstrained model breaks structure often (TOML with a 2B model: +53% relative exact match where about
40% of outputs had errors; SQL with T5-Base: +8.6 points) [VERIFIED], and a filter that forbids something the model
would legitimately write does real damage (GSM8K 41.5% to 30.8% with misaligned tokens; up to 97% loss from one
forbidden optional space) [VERIFIED]. A SynCode-style mask store does not fit LS: the published Python stores are 1.2 to
1.9 GB [VERIFIED], and scaled linearly to 512 tokens that is still about 19 MB against a 450 KB budget [UNVERIFIED: our
linear extrapolation; no source builds such a store for a small vocabulary, and the other engines of section 6.1 were
not sized]. Our corpus, however, has one canonical layout
(`ast.unparse`: 4-space indents, one statement per line, no comments, no newline inside brackets), and in that layout
the structural part of Python is a counter machine: string mode, a bracket stack, an indent level and a header flag. If
line ends, brackets and quotes are single tokens, the filter needs one class byte per token (512 bytes) and under 32
bytes of state, can only ever forbid 17 of the 512 tokens (at most 16 at once) and rejected 0 of 9,469 proxy functions
[VERIFIED] (re-run for the 17, the 16 and the 0 rejections; the byte counts are a design figure; the reference filter
has unbounded stacks and its tokenizer was trained on the same functions, see 2.5). That vocabulary needs 0.1% fewer
tokens than llama2.c-style BPE that splits at spaces, but 2.2% more than free BPE trained with the same whitespace
setting as the structural vocabulary (row 2 of 6.2) [VERIFIED] (re-run; the second baseline was missing from this
paragraph). The hypothesis to test is that the filter lifts the "syntax validity" and
"runs without error" rungs and leaves HumanEval and MBPP about where they are [UNVERIFIED: no result exists for a model
of our size; the trigram stand-in of section 4 stayed below 2% compiling with the filter on]; statement-level errors
(`def name:` without parentheses, two expressions in a row) stay unless the model learns them or an LR table is added; a
repair rule for the token budget is mandatory because constraints make non-termination more frequent (truncate to the
last complete line, add `pass`, close `try`: 99.98% valid on truncated proxy functions) [VERIFIED] (re-run; measured on
cut corpus functions, not on model output); and the mask saves no compute worth having (14.5 of 512 output rows on
average) [VERIFIED] (re-run). Scores must always be reported with and without the filter.

## 2. Proven

### 2.1 Effect on syntax errors and on functional correctness (literature)

- **SynCode, Python (CodeGen-350M, WizardCoder-1B, LLaMA-7B).** Programs with syntax errors, standard to constrained:
  HumanEval 271 to 15, 36 to 3, 291 to 2; MBXP 78 to 4, 28 to 2, 148 to 5. About 6% of all unconstrained Python samples
  had a syntax error (the paper counts 4,154 samples per language; the 852 error programs of the 3 models give 6.8%).
  Average reduction over Python and Go: 96.07%. Every remaining error was an output that did not finish inside the
  token limit. HumanEval Python pass@1: 6.8 to 6.9 (CodeGen-350M), 20.0 to 20.0 (WizardCoder-1B), 11.2 to 11.5
  (LLaMA-7B); pass@10: 10.6 to 10.6, 27.6 to 28.4, 17.1 to 18.9. pass@k is reported for HumanEval only, not for MBXP.
  https://arxiv.org/html/2403.01632v4 (Tables 3 and 4) [VERIFIED]
- **Same paper, error types (few-shot MBXP Python, Table 7).** Syntax / indentation errors without the filter:
  CodeGen-350M 53 / 15, WizardCoder-1B 40 / 22, LLaMA-7B 110 / 40, so indentation is 22% to 35% of these errors. With
  SynCode: 0 / 3, 2 / 1, 0 / 5. https://arxiv.org/html/2403.01632v4 [VERIFIED]
- **Type-constrained decoding (TypeScript; Gemma 2 2B/9B/27B, DeepSeek Coder 33B, CodeLlama 34B, Qwen2.5 32B).** Syntax
  errors are on average 6% of compilation errors. An ideal syntax-only filter would remove 9.0% (HumanEval) and 4.8%
  (MBPP) of the compiler errors in synthesis; type constraints remove 74.8% and 56.0%. pass@1 in synthesis rises by 3.5%
  relative on average (Gemma 2 2B: 29.1 to 30.2 on HumanEval, 40.4 to 42.4 on MBPP). Run time +39.1% and +52.1%. In
  99.4% of steps the sampled token is accepted without resampling. The compile errors that remain come from
  non-termination: "generation loops, entered when generation is amended by constraints and the LLM is unable to
  recover". The "Types" scores come from a mixed protocol: a sample that compiles without constraints is reported as
  it is, and only the others are resampled under constraints (section 5.1); they are not the scores of always-on
  constrained decoding. https://arxiv.org/abs/2504.09246 (PDF v2: Tables 2 to 4, sections 5.1, 5.3 and 6) [VERIFIED]
- **PICARD (SQL, fine-tuned T5, Spider dev, exact match / execution accuracy).** T5-Base 57.2 / 57.9 to 65.8 / 68.4;
  T5-Large 65.3 / 67.2 to 69.1 / 72.9; T5-3B 69.9 / 71.4 to 74.1 / 76.3. T5-Base with PICARD (65.8) is above T5-Large
  without it (65.3). 12% of T5-3B outputs gave execution errors without PICARD; 2% were unusable with it (beam search
  ended with no valid query). 3.1 s per sample with PICARD against 2.5 s without (T5-3B, beam 4, A100).
  https://arxiv.org/abs/2109.05093 [VERIFIED]
- **Synchromesh CSD (few-shot GPT-3 and Codex), valid outputs / task accuracy.** SQL, GPT-3 13B: 43% to 66% / 16% to
  20%. SQL, Codex: 73% to 85% / 56% to 61%. Vega-Lite, Codex: 87% to 99% / 39% to 40%. SMCalFlow, Codex: 79% to 97% /
  45% to 46%. Sampling overhead 8%. Unconstrained Codex needs 3 samples to reach the SQL validity that CSD gets with
  one. https://arxiv.org/abs/2201.11227 [VERIFIED]
- **Grammar-constrained decoding (Geng et al.), closed information extraction F1.** LLaMA-7B 11.9 to 23.5, LLaMA-13B
  12.9 to 30.6, LLaMA-33B 17.5 to 36.0. Parser overhead 1 to 69 ms per token on a CPU, against 54 to 136 ms for the
  forward pass on an A100. https://arxiv.org/abs/2305.13971 (Tables 1 and 4) [VERIFIED]
- **A syntax filter that covers the whole language (TOML, 2026).** Gemma-2-2B gains 53% relative exact match (about 40%
  of its unconstrained outputs had type or syntax errors); Qwen-2.5-32B gains 2.5%; an 80B model gains nothing (0.2%
  errors without the filter). https://arxiv.org/html/2606.21619v1 (Table 4, section 5.3.3) [VERIFIED]
- **Small models, Bash commands (NVIDIA blog, 2026-05-08).** 13 models, 299 tasks, per-command grammars plus a parse
  check and retry: mean pass rate 62.5% to 75.2%; Qwen3-0.6B 16.7% to 59.2%; SmolLM2-360M 29.4% to 57.2%;
  Qwen2.5-3B 74.6% to 75.6%. Multi-line shell constructs (loops, heredocs): 69.4% to 69.0%. These grammars encode which
  commands and flags exist, which is more than syntax.
  https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/ [VERIFIED]
- **Monitor-guided decoding (Java, type-correct identifiers).** Compilation rate improves by 21.77% to 24.69% relative
  for CodeGen-350M/2B/6B and SantaCoder-1.1B; CodeGen-350M with the monitor compiles more often than text-davinci-003
  from 4 samples up. https://arxiv.org/html/2306.10763v2 [VERIFIED]
- **0.6B to 4B models, JSON (abstract only).** Schema validity 78.6-92.9% to 100% under constrained decoding; content
  errors that are not schema errors stay. https://arxiv.org/abs/2609.23742 [VERIFIED] (abstract)

### 2.2 When constraints hurt (literature)

- **Token misalignment (Domino).** A JSON filter that admits only tokens matching one grammar terminal at a time
  (lookahead k = 0) cuts GSM8K accuracy from 0.415 to 0.308 for Mistral-7B and from 0.155 to 0.0 for Llama-2-13B; with
  every bridge token admitted (k = infinity) accuracy is 0.418 and 0.157. Template-based guidance loses up to 11 points.
  https://arxiv.org/html/2403.06988v1 (Tables 2 and 4; the Llama-2 13B figures are from Table 4, Table 2 of the same
  paper lists 0.262 for unconstrained Llama-2 13B and does not explain the difference) [VERIFIED]
- **Filters that block valid programs (2026 re-run of type-constrained decoding).** TypeScript, Gemma-2-2B, temperature
  0.1: constrained 32.0% against unconstrained 35.7% functional correctness on HumanEval, 44.9% against 50.4% on MBPP;
  outputs that hit the token limit 3.4% against 0.1%. In TOML, forbidding only the optional space before `=` drops
  Gemma-2-2B exact match from 48.1% to 28.6% and an 80B model from 62.5% to 1.9% (97% loss); forbidding comments or
  dotted keys costs nothing. Fine-tuning Gemma-2-2B toward the filter (GRPO with a LoRA adapter; reward 1 when the
  output passes the incremental type checker) cut its token-limit hits from 3.4% to 0.2% (HumanEval) and from 6.7% to
  1.7% (MBPP) and its time-limit hits from 3.0% to 0.8% and from 2.8% to 1.3%; after it, constrained decoding matched
  unconstrained decoding on HumanEval (32.1% against 32.8%) and stayed behind on MBPP (46.6% against 48.5%).
  https://arxiv.org/html/2606.21619v1 (Tables 1, 3, 4) [VERIFIED]
- **Forced tokens with an unusual tokenization.** Forcing the token `order` where the model normally sees `orderId`
  "severely impacted the distribution of model output"; llguidance therefore re-tokenizes forced text and backs off up
  to 4 tokens. https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md [VERIFIED]
- **Distribution distortion (grammar-aligned decoding).** Local masking renormalises at each step and ignores how
  likely each branch is to end grammatically; in the paper's example the constrained sampler produced the intended form
  only 30% of the time. The exact remedy (ASAp) needs repeated full generations, and "even after 2,000 iterations the KL
  divergence barely improves" on some tasks. https://arxiv.org/html/2405.21047v4 [VERIFIED]
- **Format restrictions and reasoning (abstract only).** Stricter output formats lowered the reasoning accuracy of
  LLMs. https://arxiv.org/abs/2408.02442 [VERIFIED] (abstract)
- **Which token variant the grammar admits matters (abstract only).** Accuracy under structured output improved by 5%
  to 10% when the grammar admitted tokens with leading whitespace, "with smaller models benefiting the most".
  https://arxiv.org/abs/2502.14969 [VERIFIED] (abstract)

### 2.3 Mechanisms, token misalignment and cost (literature)

- **How each system handles tokens that span lexical boundaries.** Synchromesh: test each token's characters against
  a character-level completion engine, tokens kept in a trie ("a single token from the language model's vocabulary
  might close multiple parentheses at once"). Outlines: for every automaton state, store the set of tokens whose whole
  string can be read from that state (the index); lookup is O(1) on average. SynCode: keep the unfinished lexical token
  as a "remainder", ask an LR(1) parser for acceptable terminal sequences of length 1 or 2, and look up a precomputed
  table M(DFA state, next terminal) that holds a bit mask over the vocabulary. llama.cpp: push each candidate token's
  code points through the grammar's stacks. XGrammar: per automaton node, precompute which tokens are accepted or
  rejected whatever the stack holds, and check the few others at run time. Geng et al.: tokenize the grammar's terminals;
  the paper notes that `[[[` can be tokenized three ways and that forcing single-bracket tokens is "not principled".
  https://arxiv.org/abs/2201.11227, https://arxiv.org/abs/2307.09702, https://arxiv.org/html/2403.01632v4,
  https://arxiv.org/html/2411.15100v3, https://arxiv.org/abs/2305.13971 [VERIFIED]
- **Table sizes.** SynCode's Python mask store: 1.87 GB (51,200 tokens), 1.83 GB (49,153), 1.17 GB (32,000), built in
  382 to 602 s; its Python grammar has 520 rules and 94 terminals and leaves out some features (lambda). XGrammar: 160 MB
  to 0.46 MB for JSON by storing only the shorter of the accept and reject lists per node; context-dependent tokens
  1,134 to 120. Outlines: memory grows with the number of automaton states.
  https://arxiv.org/html/2403.01632v4 (Table 5, section 5), https://arxiv.org/html/2411.15100v3 (section 3) [VERIFIED]
- **Per-token cost on servers.** XGrammar under 40 µs (JSON) and under 200 µs (XML, Python DSL); llguidance about 50 µs
  for a 128k-token vocabulary; llama.cpp grammar sampling in Domino's tests 0.74x to 0.87x throughput over four grammars
  (0.80x and 0.86x on the GSM8K JSON task, 0.74x and 0.81x on a C grammar) and 0.55x to 0.69x on a fixed template;
  Domino 0.78x on the C grammar (0.85x with its second model) and up to 1.77x on schema-driven JSON, where long forced
  runs are skipped ("the C code is to hard to predict using our simple count-based model").
  https://arxiv.org/html/2411.15100v3, https://github.com/guidance-ai/llguidance,
  https://arxiv.org/html/2403.06988v1 (Tables 2 and 3) [VERIFIED]
- **Mask time independent of vocabulary size (abstract only).** PSC classifies the parser stack once per step; masks up
  to 700x faster than baselines on programming-language grammars. https://arxiv.org/abs/2608.03065 [VERIFIED] (abstract)
- **llama.cpp control flow.** Sample without the grammar and test that one token; only on rejection apply the grammar
  to all candidates and resample. End-of-generation is allowed only when a grammar stack is empty. An empty stack set
  after accepting a token raises "Unexpected empty grammar stack"; no selectable token is an assertion failure. The
  README warns of "performance gotchas".
  https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp,
  https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp,
  https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md [VERIFIED]
- **Masking during training.** Bunel et al. added a {-inf, 0} syntax mask before the softmax in training and at test
  time (Karel programs, two-layer LSTM, 52-token vocabulary): top-1 generalisation 71.91 to 72.07 on the full data set,
  12.58 to 9.81 with 10,000 examples; a jointly learned syntax model reached 69.37 and 17.02; a larger model without
  any syntax check, sized to match the learned-syntax model, reached 73.67 and 13.14.
  https://arxiv.org/abs/1805.04276 (Table 3; the first version of this bullet cited Table 5) [VERIFIED]

### 2.4 Python facts the filter relies on

- **INDENT/DEDENT.** The tokenizer keeps a stack of indentation widths that starts with 0. At each logical line: equal
  width, nothing; larger, push and emit one INDENT; smaller, it "must be one of the numbers occurring on the stack" and
  one DEDENT is emitted per popped entry. https://docs.python.org/3/reference/lexical_analysis.html (section 2.1.8) [VERIFIED]
- **Brackets suspend line structure.** "There is no NEWLINE token between implicit continuation lines" inside
  parentheses, square brackets or braces. Same page, section 2.1.6. [VERIFIED]
- **The "colon, newline, indent" rule** is one grammar rule: `block: NEWLINE INDENT statements DEDENT | simple_stmts`,
  reached after the `:` of `if`, `while`, `def` and the other compound statements.
  https://docs.python.org/3/reference/grammar.html [VERIFIED]
- **CPython's own limits.** `MAXINDENT 100`, `MAXLEVEL 200`. Local check on 3.12.13: 200 nested parentheses parse and
  201 fail ("too many nested parentheses"); 99 nested blocks parse and 100 fail ("too many levels of indentation").
  https://github.com/python/cpython/blob/3.12/Parser/tokenizer.h [VERIFIED] (the local check was run again)
- **Two variables are enough for indentation.** Lark's `Indenter` holds `paren_level` (a counter) and `indent_level`
  (a stack) and ignores line ends while `paren_level > 0`. SynCode's Python parser uses a copy of the same class and derives
  the set of acceptable next indents from the stack. Melcer et al. note that for indentation "there is always some valid
  completion—additional whitespace if the indentation level is too low, or a newline if too high".
  https://github.com/lark-parser/lark/blob/master/lark/indenter.py,
  https://github.com/structuredllm/syncode/blob/main/syncode/parsers/python_parser.py,
  https://arxiv.org/html/2402.17988v2 [VERIFIED]
- **Structure as tokens.** TransCoder tokenizes Python with the standard `tokenize` module and writes `INDENT`,
  `DEDENT` and a line-end token into the sequence. SentencePiece user-defined symbols are "guaranteed to be tokenized as
  that single token, regardless of other subword probabilities". llama2.c trains its tokenizer with `model_type="bpe"`,
  `split_digits=True`, `allow_whitespace_only_pieces=True`, `byte_fallback=True`, `normalization_rule_name="identity"`.
  https://github.com/facebookresearch/CodeGen/blob/main/codegen_sources/preprocessing/lang_processors/python_processor.py,
  https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md,
  https://github.com/karpathy/llama2.c/blob/master/tinystories.py [VERIFIED]
- **`ast.parse` is weaker than `compile` (local check, Python 3.12.13).** `ast.parse` accepts `return`, `yield`,
  `break` and `continue` outside a function or loop, `nonlocal` at module level, duplicate argument names and a bare
  `except:` that is not last; `compile()` rejects all of them with SyntaxError. Unclosed or mismatched brackets, missing
  colon, missing or unexpected indent, bad dedent and unterminated strings are rejected by both. [VERIFIED] (re-run)

### 2.5 Measured locally on the proxy corpus

- **Canonical layout invariants (100,183 code lines).** 0 lines with an indentation that is not a multiple of 4; 0
  lines where indentation grows without a colon at the end of the line before; 0 cases where the step after a colon is
  not exactly +4; 0 lines that start with a block keyword and do not end with a colon; 38,262 block headers (38.2% of
  code lines); 24 lines end with a colon without a hard block keyword in front (`match` and `case`); 0 newlines inside
  brackets and 0 comments (the character machine rejected 0 of 9,469 functions); newlines inside strings occur only in
  triple-quoted strings (18,569 lines). [VERIFIED] (re-run)
- **Stack depths.** Maximum bracket depth per code line: 0 in 45.5%, 1 in 47.1%, 2 in 6.28%, 3 in 1.00%, 4 in 0.103%,
  5 to 7 in 0.019%; largest value 7. Indent level per code line: 0 in 9.5%, 1 in 42.4%, 2 in 27.4%, 3 in 13.0%, 4 in
  5.2%, 5 or more in 2.6%; largest value 10. [VERIFIED] (re-run)
- **Bridge tokens exist at our vocabulary size.** Free BPE with llama2.c-style settings (see 6.2) learns 28 pieces (of
  512) that mix a bracket, quote or newline with other characters; they are 6.6% of all tokens. At 256 tokens: 15
  pieces, 4.8% of tokens. The most frequent ones are `:\n`, `)\n`, `):\n`, ` '`, `('`, ` (`, `',`, `()\n` (table in 6.2).
  [VERIFIED] (re-run)
- **A vocabulary with structural tokens costs between -0.1% and +2.2% in length at 512 tokens, depending on the
  baseline.** Making every line end one token that also carries the
  indentation change (`NL-k` = newline and k dedents, `OPEN` = colon, newline and one indent), plus single tokens for
  the 6 brackets and 4 quote marks, gives 510,645 tokens against 511,352 for llama2.c-style BPE at 512 tokens (-0.1%),
  and 618,343 against 623,378 at 256 tokens (-0.8%). Against BPE whose pieces may span spaces it is +2.2% and -0.3%;
  that is the like-for-like baseline, because the structural vocabularies were trained with pieces allowed to span
  spaces (`split_by_whitespace=False` in the script) and the llama2.c-style row was not.
  Forcing brackets, quotes and newline to single tokens without the structural line ends costs +5.6% and +3.9%.
  All counts are in-sample: each tokenizer was trained on the 9,469 functions and counted on every 4th of them.
  [VERIFIED] (re-run; the heading said "costs no length" before the fact check)
- **Structural line ends are lossless and small.** Encoding and decoding returned the identical text for 9,469 of 9,469
  functions; characters fall from 4,640,832 to 3,860,978 (-16.8%). Of 118,752 line ends: `NL-0` 41.7%, `OPEN` 32.2%,
  `NL-1` 21.0%, `NL-2` 4.1%, `NL-3` 0.75%, `NL-4` to `NL-6` 0.17%. [VERIFIED] (re-run)
- **The token-level filter never blocked valid code.** With that 512-token vocabulary, the filter of section 6.5
  (header rule on) rejected 0 of 9,469 functions over 1,973,985 tokens (mean 208 tokens per function). Only 17 tokens
  can ever be forbidden (3 control tokens, 3 closing brackets, 11 line-end tokens); over the corpus states it forbids
  14.5 on average (13.9 without the header rule) and never more than 16 at once. The other 495 tokens are allowed in
  every state. SentencePiece gives the control and user-defined tokens the ids 0 to 23, so the filter looks at 24 ids
  and the loop over ids 24 to 511 needs no test. [VERIFIED] (re-run; the script prints only the 0 rejections and the
  14.5, so the fact check recomputed the 13.9, the maximum of 16 and the id range with the script's own `Filter`
  class). Limits of this result: the reference `Filter` keeps its stacks in unbounded Python lists, and its tokenizer
  was trained on the same 9,469 functions. Bracket depth never exceeded 7, but 2 of the 9,469 functions have more
  than 8 open blocks (7 have more than 7), so the 8-entry block-kind stack of section 6.3 does not hold every proxy
  function. A fixed 8-entry bracket stack also needs a rule that forbids the 3 opening brackets when it is full, which
  raises the 17 maskable tokens to 20 [UNVERIFIED: the filter was never run with fixed-size stacks].
- **Repair after truncation.** 3,000 functions cut at 3 random characters each (97.4% of cuts fall inside a line,
  28.2% inside a string, 18.8% inside brackets). Compiles as cut: 33.5%. Close the string and brackets in place, then
  close blocks: 75.0%. Drop the unfinished line and add `pass` after a dangling header: 95.3%. The same plus
  `except Exception: pass` for every `try` without a handler: 99.98% (2 failures in 9,000). [VERIFIED] (re-run)

## 3. Uncertain or contested

- **Does constrained decoding raise functional correctness at all?** Contested on the same benchmark. Mündler et al.
  report +3.5% relative pass@1 for type constraints on TypeScript HumanEval/MBPP; Biagiola et al. re-run the method and
  find unconstrained decoding ahead in every configuration (not significantly in two), and explain it by the filter
  rejecting valid programs [VERIFIED]. The two protocols differ: Mündler et al. keep every sample that compiles without
  constraints and constrain only the resample of the rest; Biagiola et al. constrain every sample and count a time-out
  as a failure [VERIFIED]. For syntax-only filters on models that already write valid code the effect is close to zero
  (SynCode pass@1 +0.0 to +0.3 points). Our case differs from both: we choose the corpus layout, so a filter can be
  built that blocks nothing the corpus contains; whether our model then gains is unmeasured.
- **Do weak models gain more?** Yes in PICARD (T5-Base +8.6 exact match against +3.8 and +4.2 for the larger models),
  in the NVIDIA Bash study (+42.5 points at 0.6B, +1.0 at 3B) and on TOML (+53% at 2B, +2.5% at 32B). No for Python
  syntax on CodeGen-350M (+0.1 point). [VERIFIED] The difference is how often the unconstrained model breaks the
  constraint and whether the grammar carries task knowledge (SQL schema, command flags). A grammar for Python syntax
  carries none. A text-to-SQL study on 0.5B to 7B models reports that beam search under a grammar helps small models
  most (abstract only, https://arxiv.org/abs/2608.25761) [VERIFIED] (abstract). Raspanti et al. (ACL 2025 industry
  track, logical parsing) write that grammar constraints "can serve as an effective substitute for in-context examples,
  especially beneficial for resource-constrained applications using smaller models" (abstract only,
  https://aclanthology.org/2025.acl-industry.34/) [VERIFIED] (abstract; this was [UNVERIFIED] before the fact check
  opened the ACL Anthology page; no number is taken from it).
- **How much distortion does a minimal filter cause?** The grammar-aligned-decoding result is proven in principle, but
  its size depends on how much probability the model puts on masked tokens. A filter that forbids at most 16 of 512
  tokens should distort little; the stand-in generator of section 4 still put 6% to 15% of its probability on masked
  tokens along constrained runs [VERIFIED] (re-run: 5.8% to 14.5%). Unknown for our model.
- **Training with the mask in the loss.** Bunel et al.: +0.16 points with full data, -2.77 with little data [VERIFIED];
  in the same table a larger model without any mask scored 73.67 and 13.14, above the masked model on both data sizes.
  Biagiola et al.: fine-tuning toward the filter reduces time-outs but does not make constrained decoding win. No
  result for a tiny transformer. The expected sign for us is unclear.
- **Which errors will our model make?** Unknown until a model exists. Two different 6% figures appear in the literature
  and must not be merged: 6% of unconstrained Python samples contain a syntax error (SynCode, 350M to 7B models), and
  6% of TypeScript compile errors are syntax errors (Mündler et al., 2B to 34B models) [VERIFIED]. Neither is for a
  model near our size. If our model's errors are mostly structural (brackets, strings, indentation) the minimal filter
  removes them; if they are statement-level, it does not (section 4).
- **The proxy corpus.** Standard-library functions are longer (mean 208 tokens) and deeper than the simple tasks we
  target. Depth distributions, bridge-token shares and token counts will shift on the real corpus; the invariants of
  section 2.5 follow from `ast.unparse` and should hold, but must be re-checked.
- **What "syntax validity" means.** `ast.parse` and `compile()` disagree on the seven error classes listed in section
  2.4. The ladder needs one definition; `compile()` is the stricter and matches what happens when the code is run.
  [VERIFIED] (re-run)
- **Emitting grammar rules instead of text tokens.** Liang et al. report that grammar-based code representations still
  help billion-parameter models on HumanEval and MBPP (abstract only, https://arxiv.org/abs/2503.05507). [VERIFIED]
  (abstract) This replaces the tokenizer rather than adding a filter; not examined further here.

## 4. Not tried at our scale

- **No published with-and-without comparison of a token filter for a text-token transformer under 10M parameters that
  writes Python, at a 256 or 512 token vocabulary, or on a microcontroller, FPGA or 8-bit CPU: not found in our
  search** (two web searches on these terms and the arXiv listings read for this report; four more web searches in
  the fact check, also without a hit). The smallest transformer language models in the sources read are T5-Base
  (about 220M parameters, https://huggingface.co/google-t5/t5-base; SQL), CodeGen-350M (Python, Java) and
  SmolLM2-360M (Bash). Before the fact check this bullet said that no result exists for any code model under 10M
  parameters and named SmolLM2-360M and CodeGen-350M as the smallest models; both statements were too broad. Small
  models that decode under a grammar are in the literature: Bunel et al. (two-layer LSTM, hidden size 256, 52-token
  vocabulary, Karel, mask in training and test; section 2.3); Yin and Neubig 2017 (LSTM, state size 256, writes
  Python by emitting grammar rules instead of text tokens, https://arxiv.org/abs/1704.01696); Kapur et al. 2024
  (decoder-only transformer, 8 layers, embedding size 256, graphics languages, tokens are the grammar's terminals,
  logits masked by the grammar, no run without the mask, https://arxiv.org/abs/2405.20519). [VERIFIED] (the three
  papers were opened in the fact check; none states a total parameter count; the tag covers these facts only, a
  "not found" cannot be verified)
- **Constrained decoding with int8 logits and power-of-two scales: not found.** Whether quantisation noise changes how
  often the filter intervenes is unmeasured. (One of the fact-check searches targeted this: no hit.)
- **A state-times-vocabulary mask store for full Python at 512 tokens: not found.** SynCode's stores are for 32,000 to
  51,200 tokens; the 19 MB figure in section 6.3 is our linear extrapolation, not a measurement.
- **LR(1) or LALR tables for a Python subset on a memory budget of tens of kilobytes: not found.** SynCode builds its
  LR(1) table offline in up to 2 minutes and gives no table size. [VERIFIED]
- **Training a tiny transformer with a structural mask in the loss: not found** (Bunel et al. is an LSTM on Karel;
  Kapur et al. mask the logits of a small transformer while decoding, and the text read does not say whether the
  mask is also in the training loss).
- **Beam search or sample-and-vote under a grammar: only for 0.5B to 7B models** (abstract only,
  https://arxiv.org/abs/2608.25761) [VERIFIED] (abstract). On LS every extra beam is a full forward pass of about 1.0 M
  cycles per token.
- **Our only data point, a stand-in and not our model.** A token-trigram generator trained on the proxy corpus (it sees
  2 tokens of context, so it knows almost no syntax), 2,000 samples per row, budget 200 tokens, 512-token structural
  vocabulary. "Temperature 0.5" squares the trigram counts.

| Generator | Filter | Finished inside the budget | Compiles as generated | Compiles after truncation repair |
|---|---|---|---|---|
| temperature 1.0 | none | 76.2% | 0.3% | 0.9% |
| temperature 1.0 | filter | 52.6% | 0.7% | 3.2% |
| temperature 1.0 | filter + header rule | 15.8% | 0.5% | 4.7% |
| temperature 0.5 | none | 78.7% | 0.5% | 1.9% |
| temperature 0.5 | filter | 61.1% | 0.8% | 10.0% |
| temperature 0.5 | filter + header rule | 11.7% | 1.9% | 17.2% |

  What it shows: (a) the bracket and string errors disappear (most frequent compile messages after repair, without
  the filter, at temperature 1.0: `expected '('` 526, `unmatched ')'` 317, `unterminated string literal` 269, `invalid syntax` 169; with the
  filter: `expected '('` 960, `invalid syntax` 286, `expected ':'` 146), and statement-level errors take their place;
  (b) the share of runs that end inside the budget falls from 76% to 53% and to 16% with the header rule, the same
  non-termination effect the literature reports (unfiltered runs also "finish" through invalid early dedents);
  (c) a generator that does not know statement syntax stays invalid under a structural filter. The trigram put 5.8% to
  14.5% of its probability on masked tokens, and in 0.27% to 0.36% of steps every token it proposed was masked. A
  260k-parameter transformer should know far more local syntax than a trigram; that is a hypothesis to test, not a
  result. Every number of the table and of this paragraph was reproduced by the re-run [VERIFIED].

## 5. Ranked ideas for LS-LLM

Cycle figures are estimates from the cost model in `CONTEXT.md` (4 cycles per ordinary instruction, n + 24 cycles per
matrix row, about 1.0 M cycles per token); they are not measured. [VERIFIED] (quoted correctly from `CONTEXT.md`)

| # | Idea | Expected effect: score / speed / memory | Cost to try | Risk | Keep-or-drop measurement | CPU change? |
|---|---|---|---|---|---|---|
| 1 | Structural vocabulary: every line end is one token that carries the indent change (`NL-0` to `NL-3`, `OPEN` = colon + newline + indent); the 6 brackets and 4 quote marks are single tokens; indentation spaces are never generated | Score: indentation errors impossible by construction (22-35% of syntax errors in SynCode's few-shot runs). Speed: tokens per function -0.1% (512) and -0.8% (256) against llama2.c-style BPE on the proxy, +2.2% and -0.3% against free BPE with the same whitespace setting. Memory: none. Makes idea 2 a one-byte-per-token table | Pre- and post-processing around SentencePiece, about 60 lines (prototype: `to_structure` / `from_structure` in the script) | Benchmark prompts must pass through the same encoder; dedents deeper than 3 need a repeat rule (0.17% of line ends on the proxy); belongs to the tokenizer work | Round trip exact on 100% of the corpus; tokens per function at most +2% against free BPE on the real corpus (on the proxy it is already +2.2% against row 2 of 6.2); practice-set score not lower | No |
| 2 | Structural filter in the token-selection loop: string mode, bracket stack (8 entries), indent level; line-end tokens only at bracket depth 0 and outside one-line strings; closing bracket must match; dedent at most the current level | Score: bracket, string and line-structure errors removed; large on "syntax validity" only if our model's errors are of these kinds; near 0 on HumanEval/MBPP (SynCode: +0.1 point at 350M on HumanEval; that paper reports no MBPP pass@k). Speed: a few hundred cycles per token, under 0.1%. Memory: 512 B class table + under 32 B state + routine | Python reference exists as `Filter` in the script (about 50 lines); assembly routine; bit-exact test against the reference | Any rejection of valid code is harmful (up to 97% loss in the literature); more runs hit the budget | (a) 0 rejections over the full corpus and the practice-set reference solutions (never the test sets); (b) practice-set "compiles" and "runs" rates with and without; (c) share of runs hitting the budget; keep if (b) rises and practice pass rate does not fall | No (needs a table read by token id and a conditional branch) |
| 3 | Budget repair: send only complete lines; when the budget runs out, drop the unfinished line, add `pass` after a dangling header, add `except Exception: pass` for an open `try`, dedent to level 0 | Score: a budget overrun no longer breaks the syntax when the lines before the cut are valid (99.98% on truncated proxy functions; closing brackets in place gives only 75.0%); it does not mend statement-level errors (stand-in generator with the filter on: 3.2% to 17.2% compile after repair). Speed: none (rule-based, no forward pass). Memory: one line buffer + 8 B block-kind stack | Small; prototype is `repair` in the script | The repaired function is valid but usually wrong; must be counted and reported as "truncated" | Share of generations repaired; "compiles" rate of repaired outputs on the practice set; always keep, always report the share | No |
| 4 | Report every score twice (filter off / filter on) and define syntax validity as `compile()` success | Honest attribution: validity bought by the filter is not credited to the model; `compile()` also catches the 7 classes `ast.parse` misses | None | None | Not applicable (reporting rule) | No |
| 5 | Train with the same filter: mask forbidden tokens in the loss, decode with the filter always on | Score: uncertain (+0.16 / -2.77 points in Bunel et al.); should lower the probability mass on masked tokens and with it budget overruns. Speed and memory: none at inference | One class-forbid byte per training position, computed offline by the reference filter; one extra training run | Training and CPU filter must agree bit for bit; no evidence at our scale | Practice-set pass rate and budget-overrun share against unmasked training at equal steps; keep at +1 point or clearly fewer overruns | No |
| 6 | Header and keyword rules: a line that starts with a block keyword must end with `OPEN` and only such lines may; `break`/`continue` only inside a loop block. Needs the 12 block keywords as dedicated line-start tokens | Score: removes "missing colon" and the scope errors `compile()` reports; in the stand-in, valid-after-repair 10.0% to 17.2% but finished runs 61% to 12%. Memory: +8 B block-kind stack (shared with idea 3) | Tokenizer change (line-start keyword tokens) + 2 rules | More budget overruns; `match`/`case` are legal names, so they may open a block but must not be required to | Same as idea 2; drop if overruns rise more than validity | No |
| 7 | Class-pair table: map each token to one of about 32 lexical classes and allow only class pairs seen in the corpus (32 x 32 bits = 128 B) | Score: catches some statement-level errors (two names in a row, operator before a line end); size of effect unknown. Speed: one table read per candidate | Medium: class design, corpus statistics | Blocks valid pairs that the corpus happens not to contain; this is exactly the failure Biagiola et al. measure | 0 rejections on held-out corpus and practice-set reference solutions, else drop; then as idea 2 | No |
| 8 | LR table for a Python subset, parser stack on the CPU (catches-all for the subset) | Score: syntax valid by construction for the subset (SynCode level); pass@k gain in the literature 0 to 2 points. Memory: unknown until generated, competes with weights for the 450 KB; candidate for Tier B. Speed: one or more reductions per checked token | High: grammar subset, table generator, CPU parser, proof that the subset covers the corpus | A subset that misses corpus constructs blocks valid code; engineering time | Only start if, after ideas 1 to 3, most remaining practice-set failures are statement-level syntax; table size in bytes; 0 corpus rejections | No (table memory only) |
| 9 | Skip output-layer rows of masked tokens | Speed: at most 16 rows x 88 cycles = 1.4k cycles, about 0.14% of a token; 14.5 rows on average | Trivial once idea 2 exists | None | Cycle count in the simulator; expected to be dropped as noise | No |
| 10 | Skip the forward pass for forced tokens (fast-forward) | Speed: none expected. The literature's gains (Domino 1.77x) come from schema-driven formats; on a C grammar Domino measured 0.78x. With idea 1 the forced sequences of Python (colon, newline, indent) are already one token | None | None | Count states with exactly one allowed token on the practice set; drop if rare | No |

Fact check of this table: the literature and proxy numbers quoted in rows 1 to 3, 5, 6, 9 and 10 match their sources
and the re-run [VERIFIED]. The "Expected effect" column is a forecast [UNVERIFIED: no model of our size has been run
with or without the filter]. Rows 1 and 3 were reworded (like-for-like token baseline; what the repair does not do).

## 6. Tables asked for in the topic brief

### 6.1 Methods

| Method | What it guarantees | Mechanism | Per-token cost (as reported) | Reported effect on correctness | Reducible to a small table for an 8-bit CPU? | Source |
|---|---|---|---|---|---|---|
| PICARD (2021) | Valid SQL prefix incl. schema names, when beam search finds one | Incremental parser (Haskell attoparsec) on the detokenized text; checks only the top-k tokens of each step, rejected ones get -inf | 3.1 s per sample against 2.5 s without (T5-3B, beam 4) | Spider exact match: T5-Base 57.2 to 65.8, T5-Large 65.3 to 69.1, T5-3B 69.9 to 74.1; 2% of outputs still unusable | No (full parser). The "check only the top few candidates" idea transfers | https://arxiv.org/abs/2109.05093 |
| Synchromesh CSD (2022) | Output inside the language of a completion engine (syntax, scope, types) | Character-level completion engine; each token's string is walked against it; token trie | +8% sampling time | SQL validity 73% to 85%, execution 56% to 61% (Codex); Vega-Lite and SMCalFlow validity 97-99%, accuracy +1 point | No (engine is a parser). The offline walk of each token's string transfers | https://arxiv.org/abs/2201.11227 |
| Grammar-constrained decoding, Geng et al. (2023) | Output in a context-free language (task-specific, may depend on the input) | Grammar rewritten over tokens; incremental parser of Grammatical Framework returns allowed tokens | 1 to 69 ms on a CPU | Closed IE F1: LLaMA-7B 11.9 to 23.5, 13B 12.9 to 30.6 | No. Works cleanly only when tokens equal terminals, which is what a structural vocabulary gives | https://arxiv.org/abs/2305.13971 |
| llama.cpp GBNF | Output prefix derivable from a GBNF grammar | Set of pushdown stacks; candidate tokens walked code point by code point; sample, check one token, re-mask all on rejection | 0.74x to 0.87x throughput over four grammars, 0.80x and 0.86x on the GSM8K JSON task (Domino's tests) | GSM8K 0.415 to 0.375 (Mistral-7B, JSON output, Domino's tests) | No (unbounded stack sets). A regular grammar could be compiled to a table by hand | https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md |
| Outlines (2023) | Output matches a regular expression; CFG by extension | Finite-state machine; index from state to the set of tokens readable from it, built once per vocabulary | O(1) average lookup | None reported in the paper | Yes for regular constraints: next-state table of states x tokens bytes. Python needs a stack on top | https://arxiv.org/abs/2307.09702 |
| SynCode (2024) | Never blocks a valid token; catches all invalid ones when lookahead suffices; indentation handled outside the grammar | LR(1) parser gives acceptable terminal sequences; DFA mask store M(state, terminal) to vocabulary bit mask; remainder for unfinished tokens | Generation time about equal to unconstrained (JSON: 3.07 s against 3.58 s for Llama-2-7B-chat, 4.82 s against 4.32 s for Gemma2-2B-it) | Python/Go syntax errors -96.07%; HumanEval pass@1 +0.0 to +0.3 points | In part. The indentation mechanism (stack + bracket counter) transfers as is; the mask store does not (about 19 MB at 512 tokens by our linear scaling, not measured) | https://arxiv.org/abs/2403.01632 |
| Guidance / llguidance | Output in a context-free grammar | Earley parser on a derivative-based lexer; token trie; forced ("fast-forward") tokens re-tokenized canonically | About 50 µs for 128k tokens | Templates lose up to 11 points on GSM8K (Domino's tests); whitespace-flexible grammars recover most | No (Earley parser, lazily built automata) | https://github.com/guidance-ai/llguidance |
| Domino (2024) | Output in a CFG without excluding any bridge token (at k = infinity) | Per lexer state, a precomputed tree of terminal sequences each token can span; parser checks the tree; count-based speculation | 0.78x (C) to 1.77x (JSON schema) | GSM8K 0.415 to 0.418; 0.308 when bridge tokens are excluded | In part (the per-state trees are tables; a parser is still needed) | https://arxiv.org/abs/2403.06988 |
| XGrammar (2024) | Output in a CFG | Byte-level pushdown automaton; per-node cache of tokens accepted or rejected whatever the stack holds; persistent stack for the rest | Under 40 µs (JSON), under 200 µs (Python DSL) | Syntactic correctness only: 62% to 100% (function calling), 80% to 100% (XML), LLaMA-3.1 8B; no task accuracy | In part: the per-node lists are a compact table (0.46 MB for JSON at a large vocabulary); stack-dependent tokens need run-time parsing | https://arxiv.org/abs/2411.15100 |
| Grammar-aligned decoding, ASAp (2024) | Grammatical output whose distribution converges to the model's distribution conditioned on the grammar | Repeated sampling; remembers probability lost to masked tokens in a trie of earlier samples | Many full generations; slow convergence (2,000 iterations not enough on some tasks) | Higher likelihood under the model than plain masking; no pass@k reported | No | https://arxiv.org/abs/2405.21047 |
| Type-constrained decoding (2025) | Well-typed TypeScript prefix (subset of the language) | Prefix automata + search over inhabitable types; sample and check, resample on rejection | +39.1% and +52.1% run time | Compile errors -74.8% and -56.0%; pass@1 +3.5% relative (mixed protocol: constrained resample only for samples that fail to compile); contested by the 2026 re-run (32.0 against 35.7 for Gemma-2-2B) | No (type inference engine, 11,249 lines of Python) | https://arxiv.org/abs/2504.09246 |
| Monitor-guided decoding (2023) | Type-consistent identifiers at dereference points; no syntax guarantee | Static analysis (language server) proposes legal identifiers; other tokens masked at those points | Not extracted | Compilation rate +21.77% to +24.69% relative | No | https://arxiv.org/abs/2306.10763 |
| Quotient parsing for fill-in-the-middle, Melcer et al. (2024) | Python output that can still be completed to fit the code on the right | Earley parser on left and right quotients; lexer emits INDENT/DEDENT; checks the top 50 candidates | Not extracted | Fewer syntax errors than unconstrained generation on 2 data sets; slightly ahead of checking only at the end | No. Its budget policy transfers (see 6.4) | https://arxiv.org/abs/2402.17988 |
| Syntax mask in training and test, Bunel et al. (2018) | Karel program prefix is syntactically valid | Hand-written checker gives a {-inf, 0} mask before the softmax, also during training | Not reported | 71.91 to 72.07 (full data); 12.58 to 9.81 (10,000 examples) | Yes (the mask is a vector per step; Karel's checker is small) | https://arxiv.org/abs/1805.04276 |
| Proposed LS filter (section 6.5), measured on the proxy only | Balanced and matched brackets, terminated strings, consistent indentation, line ends only where legal; no statement-level guarantee | Class byte per token + string mode, bracket stack, indent level, header flag | A few hundred cycles, under 0.1% of a token (estimate) | No model result. 0 of 9,469 valid functions rejected | Yes: 512 B table + under 32 B state | `research/06_local_measurements.py` |

Fact check of this table: the numbers of every literature row were compared with the source [VERIFIED]. Three cells
were corrected (PICARD timing wording, llama.cpp throughput range, XGrammar correctness) and two got a qualifier
(SynCode timing, protocol of type-constrained decoding). Mechanism descriptions were only spot-checked. The last row
is local (re-run).

### 6.2 Token misalignment at 256 and 512 tokens (local, proxy corpus, SentencePiece 0.2.3 BPE)

"Bridge piece" = a piece longer than one character that contains a bracket, quote or newline together with other
characters (the triple quotes `'''` and `"""` are not counted). Token counts are for every 4th function and are
in-sample (each tokenizer was trained on all 9,469 functions). Row 1 splits pieces at spaces, rows 2 to 5 do not. All
variants use the llama2.c trainer options except `byte_fallback` and `add_dummy_prefix` (both off: the corpus is
ASCII); with byte fallback on, 256 of the vocabulary slots would go to byte tokens and every count would be higher.

| Vocabulary | Variant | Tokens | Against row 1 | Characters per token | Bridge pieces | Share of tokens that are bridge pieces |
|---|---|---|---|---|---|---|
| 512 | 1. Free BPE, split at spaces (llama2.c-style) | 511,352 | 0.0% | 2.371 | 28 | 6.6% |
| 512 | 2. Free BPE, pieces may span spaces | 499,560 | -2.3% | 2.427 | 33 | 6.7% |
| 512 | 3. Brackets, quotes, newline forced to single tokens | 539,747 | +5.6% | 2.246 | 0 | 0.0% |
| 512 | 4. Structural line ends, free BPE otherwise | 496,418 | -2.9% | 2.442 | 23 | 3.4% |
| 512 | 5. Structural line ends + brackets and quotes single | 510,645 | -0.1% | 2.374 | 0 | 0.0% |
| 256 | 1. Free BPE, split at spaces (llama2.c-style) | 623,378 | 0.0% | 1.945 | 15 | 4.8% |
| 256 | 2. Free BPE, pieces may span spaces | 620,143 | -0.5% | 1.955 | 17 | 4.8% |
| 256 | 3. Brackets, quotes, newline forced to single tokens | 647,492 | +3.9% | 1.872 | 0 | 0.0% |
| 256 | 4. Structural line ends, free BPE otherwise | 610,726 | -2.0% | 1.985 | 9 | 2.3% |
| 256 | 5. Structural line ends + brackets and quotes single | 618,343 | -0.8% | 1.960 | 0 | 0.0% |

Reading: with free BPE (rows 1, 2) one token in 15 to 21 crosses a lexical boundary, and the frequent ones (`):\n`,
`()\n`, `('`) carry two or three structural effects each. There are two ways to deal with them. (a) Keep them and give
every piece a precomputed effect list (the SynCode / Outlines approach shrunk: walk each piece's characters through the
character machine offline; about 30 pieces need such a list). (b) Remove them by vocabulary design (row 5), which
costs -0.1% in sequence length against row 1 and +2.2% against row 2, its like-for-like baseline (same whitespace
setting), and makes the per-token effect a single class code. Row 3 shows that removing them without
structural line ends is expensive, because the merged `)\n` and `:\n` pieces were saving tokens.
[VERIFIED] (re-run: all ten rows reproduced; the reading of row 5 was corrected)

### 6.3 Size of mask and state tables

| Design | Size | Basis |
|---|---|---|
| Generic bit table, S states x V tokens | S x V / 8 bytes: 16 states x 512 = 1 KB; 256 x 512 = 16 KB; 1,024 x 512 = 64 KB; 4,096 x 512 = 256 KB | arithmetic |
| Generic next-state table (Outlines index stored densely), one byte per entry | S x V bytes: 16 x 512 = 8 KB; 256 x 512 = 128 KB | arithmetic |
| SynCode Python mask store as published | 1.17 GB at 32,000 tokens; 1.87 GB at 51,200 tokens (about 36.5 KB per vocabulary token in both) | https://arxiv.org/html/2403.01632v4, Table 5 [VERIFIED] |
| The same scaled linearly to our vocabulary (assumption: size proportional to V) | about 18.7 MB at 512 tokens, 9.4 MB at 256 | our extrapolation, not measured [UNVERIFIED: no source builds this store for a small vocabulary] |
| XGrammar adaptive cache, JSON grammar | 0.46 MB (from 160 MB) | https://arxiv.org/html/2411.15100v3 [VERIFIED] |
| Proposed LS filter: class byte per token | 512 B (256 B at 256 tokens) | section 6.5 |
| Proposed LS filter: state | mode 1 B, escape flag 1 B, bracket depth 1 B, bracket stack 8 B, indent level 1 B, line flags 1 B, block-kind stack 8 B: 21 B | section 6.5; depth 7 and level 10 were the largest values on the proxy. An 8-entry block-kind stack with one byte per open block does not hold levels 9 and 10 (2 of 9,469 proxy functions): pack the kinds, enlarge the stack, or forbid `OPEN` at level 8 and accept those rejections [UNVERIFIED: the 21-byte layout was never run; the reference filter uses unbounded lists] |
| Share of the 450 KB budget | 533 B, about 0.12% | arithmetic |

Compute. The output layer at 512 tokens and dimension 64 is about 512 x 88 = 45k cycles (4.5% of 1.0 M; `CONTEXT.md`
gives 5-9%). Only 17 structural and control tokens can ever be masked, at most 16 at once, so skipping masked rows
saves at most 16 x 88 = 1.4k cycles (0.14% of a token), 14.5 rows on average. The filter itself has to examine the 24
low token ids and update the state once per generated token: a few hundred cycles (estimate). A mask is a correctness
feature here, not a speed feature. [VERIFIED] (arithmetic done again, 14.5 and 16 rows from the re-run; the cycle
costs themselves are estimates from `CONTEXT.md`, not measurements)

### 6.4 What to do when nothing valid is left, or the budget runs out

| System | Behaviour | Source |
|---|---|---|
| llama.cpp | Assertion failure if no token can be selected; exception on an empty grammar stack set | `common/sampling.cpp`, `src/llama-grammar.cpp` |
| PICARD | Beam search ends without a valid query (2% of Spider dev outputs for T5-3B) | https://arxiv.org/abs/2109.05093 |
| SynCode | No rule; outputs that reach the token limit are the remaining syntax errors | https://arxiv.org/html/2403.01632v4 |
| Type-constrained decoding | Returns the partial program at the 300 s limit; names generation loops as the open problem | https://arxiv.org/abs/2504.09246 |
| Melcer et al. | Try the top 50 candidates; if all fail, stop; if no end token came before the limit, cut the output at the valid stopping point with the highest end-token probability; else report failure | https://arxiv.org/html/2402.17988v2 |
| NVIDIA Bash study | Parse the result; on error ask for an unconstrained answer with the parse error included | NVIDIA blog |
| Proposed for LS | "Every token masked" cannot happen: 495 of 512 tokens are allowed in every state. The real failure is the budget. Rule: keep only complete lines; at the limit drop the unfinished line, add `pass` after a dangling header, add a handler for an open `try`, dedent to level 0. 99.98% valid on truncated proxy functions | section 2.5 |

Fact check of this table: the six literature rows match their sources (llama.cpp code, PICARD, SynCode, Mündler et
al., Melcer et al., NVIDIA blog) [VERIFIED].

### 6.5 The proposed filter (specification, validated only on the proxy corpus)

Vocabulary assumptions: 11 line-end tokens (`NL-0` to `NL-9`, `OPEN`; 5 would cover 99.83% of proxy line ends), 6
bracket tokens, 4 quote tokens (`'`, `"`, `'''`, `"""`), 3 control tokens, 488 ordinary pieces that contain none of
these characters. A function ends with the `NL-k` that returns to level 0, which is the stop rule already written in
`lsllm/api.py`. Before generation the prompt tokens are pushed through the same routine to set the state.

| State | Meaning | Changed by |
|---|---|---|
| `mode` | code, or inside a string opened by one of the 4 quote tokens | quote tokens |
| `esc` | the previous string character was an unpaired backslash | pieces ending in a backslash |
| `stack` | open brackets, innermost on top (8 entries) | bracket tokens in code mode |
| `level` | current block depth | `OPEN` (+1), `NL-k` (-k) in code mode |
| `header` | the line started with a block keyword (only with idea 6) | line-start keyword token, cleared at line end |

| Token class | Allowed when |
|---|---|
| Closing bracket | code mode: `stack` not empty and its top is the matching opener. String mode: always (it is text) |
| `NL-k` | code mode: `stack` empty and k <= `level` (and not `header`, with idea 6). Triple-quoted string: only `NL-0`. One-line string: never |
| `OPEN` | code mode: `stack` empty (and `header`, or the line starts with `match`/`case`, with idea 6). String mode: never |
| Control tokens | never during generation (end of text only at level 0 in code mode, if that token is used) |
| Quote tokens, opening brackets, ordinary pieces | always |

Not covered, by design: statement and expression grammar (`def name:`, `x = 1 2`, a dangling operator), names, types.
Comments need no state because the canonical layout has none; a `#` can only occur inside a string.

Open point from the fact check: the two tables above assume stacks that never fill. With the 8-entry bracket stack of
section 6.3 the rule "opening brackets: always" has to become "only while the stack has room", and `OPEN` needs the
same rule for the block-kind stack. Then 20 tokens can be forbidden, not 17, and 2 of the 9,469 proxy functions (more
than 8 open blocks) are rejected unless that stack is packed or enlarged [UNVERIFIED: fixed-size stacks are
not implemented in the reference `Filter`; the fact check only counted the 2 functions].

## 7. Sources

Papers (no licence listed: only the text was read). Titles, authors and years of every row were compared with arXiv, the
ACL Anthology or the publisher page [VERIFIED]; three rows were added and one author was corrected in the fact check.

| Title | Authors | Year | URL |
|---|---|---|---|
| PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models | Scholak, Schucher, Bahdanau | 2021 | https://arxiv.org/abs/2109.05093 |
| Synchromesh: Reliable code generation from pre-trained language models | Poesia, Polozov, Le, Tiwari, Soares, Meek, Gulwani | 2022 | https://arxiv.org/abs/2201.11227 |
| Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning | Geng, Josifoski, Peyrard, West | 2023 | https://arxiv.org/abs/2305.13971 |
| Efficient Guided Generation for Large Language Models (Outlines) | Willard, Louf | 2023 | https://arxiv.org/abs/2307.09702 |
| Guiding Language Models of Code with Global Context using Monitors | Agrawal, Kanade, Goyal, Lahiri, Rajamani | 2023 | https://arxiv.org/abs/2306.10763 |
| SynCode: LLM Generation with Grammar Augmentation | Ugare, Suresh, Kang, Misailovic, Singh | 2024 | https://arxiv.org/abs/2403.01632 |
| Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation (Domino) | Beurer-Kellner, Fischer, Vechev | 2024 | https://arxiv.org/abs/2403.06988 |
| Grammar-Aligned Decoding | Park, Wang, Berg-Kirkpatrick, Polikarpova, D'Antoni | 2024 | https://arxiv.org/abs/2405.21047 |
| XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models | Dong, Ruan, Cai, Lai, Xu, Zhao, Chen | 2024 | https://arxiv.org/abs/2411.15100 |
| Constrained Decoding for Fill-in-the-Middle Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars | Melcer, Fulton, Gouda, Qian | 2024 | https://arxiv.org/abs/2402.17988 |
| Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models (abstract only) | Tam, Wu, Tsai, Lin, Lee, Chen | 2024 | https://arxiv.org/abs/2408.02442 |
| Type-Constrained Code Generation with Language Models | Mündler, He, Wang, Sen, Song, Vechev | 2025 | https://arxiv.org/abs/2504.09246 |
| Lost in Space: Finding the Right Tokens for Structured Output (abstract only; leading-whitespace tokens help small models most) | Hamilton, Mimno | 2025 | https://arxiv.org/abs/2502.14969 |
| Grammar-Based Code Representation: Is It a Worthy Pursuit for LLMs? (abstract only) | Liang, Zhang, Sun, Lin, Luo, Xiao, Chen, Zhang, Zhang, Zhang, Chen, Xiong | 2025 | https://arxiv.org/abs/2503.05507 |
| The Alignment Problem in Constrained Code Generation | Biagiola, Cesario, Di Grazia, Zakhour, Salvaneschi | 2026 | https://arxiv.org/abs/2606.21619 |
| Efficient Grammar-Constrained Decoding via Parser Stack Classification (abstract only) | Li, Dong, Li, Li | 2026 | https://arxiv.org/abs/2608.03065 |
| Beam Search, Self-Consistency, and the Limits of Inference-Time Scaling for Grammar-Constrained Text-to-SQL in Small Language Models (abstract only) | Chermsirivatana, MacCormick | 2026 | https://arxiv.org/abs/2608.25761 |
| Constrained Decoding Eliminates Structural Failures in Small LLMs but Reveals a Scale-Dependent Semantic Gap (abstract only) | Chavan | 2026 | https://arxiv.org/abs/2609.23742 |
| Improving Bash Generation in Small Language Models with Grammar-Constrained Decoding (blog) | Joseph Lucas (NVIDIA AI Red Team) | 2026 | https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/ |
| Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis | Bunel, Hausknecht, Devlin, Singh, Kohli | 2018 | https://arxiv.org/abs/1805.04276 |
| Unsupervised Translation of Programming Languages (TransCoder; abstract and tokenizer source read) | Lachaux, Roziere, Chanussot, Lample | 2020 | https://arxiv.org/abs/2006.03511 |
| Grammar-Constrained Decoding Makes Large Language Models Better Logical Parsers (abstract only; row added in the fact check) | Raspanti, Ozcelebi, Holenderski | 2025 | https://aclanthology.org/2025.acl-industry.34/ |
| A Syntactic Neural Model for General-Purpose Code Generation (configuration paragraph read; row added in the fact check) | Yin, Neubig | 2017 | https://arxiv.org/abs/1704.01696 |
| Diffusion On Syntax Trees For Program Synthesis (architecture section read; row added in the fact check) | Kapur, Jenner, Russell | 2024 | https://arxiv.org/abs/2405.20519 |
| The Python Language Reference: Lexical analysis; Full Grammar specification | Python Software Foundation | 2026 (3.x docs) | https://docs.python.org/3/reference/lexical_analysis.html, https://docs.python.org/3/reference/grammar.html |

Code read (licence as reported by the GitHub API on 2026-11-13). None of it is copied into the project. All 17
repositories were queried again [VERIFIED]; two rows changed (python/cpython, eth-sri/domino).

| Repository | What was read | Licence | URL |
|---|---|---|---|
| structuredllm/syncode (was uiuc-focal-lab/syncode) | `syncode/parsers/python_parser.py` | MIT | https://github.com/structuredllm/syncode |
| ggml-org/llama.cpp | `grammars/README.md`, `common/sampling.cpp`, `src/llama-grammar.cpp` | MIT | https://github.com/ggml-org/llama.cpp |
| guidance-ai/llguidance | `README.md`, `docs/fast_forward.md` | MIT | https://github.com/guidance-ai/llguidance |
| guidance-ai/guidance | licence only | MIT | https://github.com/guidance-ai/guidance |
| dottxt-ai/outlines | licence only | Apache-2.0 | https://github.com/dottxt-ai/outlines |
| mlc-ai/xgrammar | licence only | Apache-2.0 | https://github.com/mlc-ai/xgrammar |
| ServiceNow/picard | licence only | Apache-2.0 | https://github.com/ServiceNow/picard |
| epfl-dlab/GCD | licence only | MIT | https://github.com/epfl-dlab/GCD |
| ebmoon/transformers-GAD | licence only | MIT | https://github.com/ebmoon/transformers-GAD |
| eth-sri/type-constrained-code-generation | licence only | MIT | https://github.com/eth-sri/type-constrained-code-generation |
| eth-sri/domino | repository metadata only | none: the repository is empty (size 0, no files, checked 2026-11-13) | https://github.com/eth-sri/domino |
| microsoft/monitors4codegen (archived) | licence only | MIT | https://github.com/microsoft/monitors4codegen |
| lark-parser/lark | `lark/indenter.py` | MIT | https://github.com/lark-parser/lark |
| facebookresearch/CodeGen (TransCoder) | `codegen_sources/preprocessing/lang_processors/python_processor.py` | MIT | https://github.com/facebookresearch/CodeGen |
| google/sentencepiece | `doc/special_symbols.md`; library 0.2.3 used for the local measurements | Apache-2.0 | https://github.com/google/sentencepiece |
| karpathy/llama2.c | `tinystories.py` (tokenizer settings) | MIT | https://github.com/karpathy/llama2.c |
| python/cpython | `Parser/tokenizer.h` (3.12 branch); standard library used as the proxy corpus | Python Software Foundation License Version 2 (LICENSE file of the 3.12 branch, read in the fact check; the GitHub API reports "Other") [VERIFIED] | https://github.com/python/cpython |

Project files used: `research/CONTEXT.md` (cost model), `lsllm/canon.py` (canonical layout), `lsllm/api.py` (stop rule),
`research/06_local_measurements.py` (written for this report).

## Verification log

Fact check of 2026-11-13 by a second agent. The cited sources were opened again (arXiv HTML or PDF, arXiv API, GitHub
raw files and API, publisher pages); local numbers were checked by running `research/06_local_measurements.py` again
and, for the few numbers it does not print, by a short script that calls its `Filter` class. 25 claims: 13 confirmed,
11 corrected, 1 unsupported. Every cited paper, post and repository exists (eth-sri/domino exists but is empty).

| # | Claim | Verdict | Source |
|---|---|---|---|
| 1 | SynCode: 96.07% fewer syntax errors on Python and Go; Table 3 counts (HumanEval 271 to 15, 36 to 3, 291 to 2; MBXP 78 to 4, 28 to 2, 148 to 5); the errors that remain are outputs cut off by the token limit | confirmed | https://arxiv.org/html/2403.01632v4 |
| 2 | SynCode HumanEval pass@1 6.8 to 6.9, 20.0 to 20.0, 11.2 to 11.5 and pass@10 10.6 to 10.6, 27.6 to 28.4, 17.1 to 18.9; section 1 explained the flat pass@1 with "because only about 6% of the unconstrained samples had a syntax error at all" | corrected (numbers confirmed): "because only about 6% ... had a syntax error" -> "although 256 of 271 broken HumanEval samples were repaired; 8.4% of that model's samples were broken; the reading is ours" | https://arxiv.org/html/2403.01632v4 (Tables 3, 4) |
| 3 | SynCode few-shot MBXP error types: syntax / indentation 53 / 15, 40 / 22, 110 / 40 without and 0 / 3, 2 / 1, 0 / 5 with the filter; indentation is 22% to 35% | confirmed | https://arxiv.org/html/2403.01632v4 (Table 7) |
| 4 | SynCode Python mask store 1.87 / 1.83 / 1.17 GB at 51,200 / 49,153 / 32,000 tokens, built in 382 to 602 s; grammar with 520 rules and 94 terminals, no lambda; LR(1) table built in up to 2 minutes, no table size given | confirmed | https://arxiv.org/html/2403.01632v4 (Table 5, sections 4.7 and 5) |
| 5 | "General grammar engines do not fit LS": scaled linearly to 512 tokens the store is still about 19 MB | unsupported as written: the 19 MB is the file's own extrapolation and only SynCode's store was sized; narrowed to "a SynCode-style mask store does not fit LS" and tagged [UNVERIFIED] | https://arxiv.org/html/2403.01632v4 (Table 5) |
| 6 | Type-constrained decoding: syntax errors are 6% of compile errors; ideal syntax filter removes 9.0% / 4.8%, types 74.8% / 56.0%; pass@1 +3.5% relative (Gemma 2 2B 29.1 to 30.2, 40.4 to 42.4); run time +39.1% / +52.1%; 99.4% accepted at once; 11,249 lines. Section 3 said "the literature's 6% syntax-error share is for 350M to 34B models" | corrected (numbers confirmed): one 6% share for 350M to 34B models -> two different 6% figures (SynCode: samples with a syntax error, 350M to 7B; Mündler et al.: share of compile errors, 2B to 34B); added the paper's protocol (constrained resample only for samples that fail to compile) | https://arxiv.org/abs/2504.09246 (PDF v2) |
| 7 | PICARD: Spider dev exact match / execution 57.2 / 57.9 to 65.8 / 68.4, 65.3 / 67.2 to 69.1 / 72.9, 69.9 / 71.4 to 74.1 / 76.3; 12% execution errors without, 2% unusable with; 2.5 s and 3.1 s per sample | confirmed (timing wording made explicit: 3.1 s with PICARD, 2.5 s without) | https://arxiv.org/abs/2109.05093 |
| 8 | Synchromesh CSD: valid outputs / accuracy for SQL (GPT-3 13B 43% to 66% / 16% to 20%; Codex 73% to 85% / 56% to 61%), Vega-Lite 87% to 99% / 39% to 40%, SMCalFlow 79% to 97% / 45% to 46%; 8% overhead; 3 samples; quotation | confirmed | https://arxiv.org/abs/2201.11227 |
| 9 | Geng et al.: closed information extraction F1 11.9 to 23.5, 12.9 to 30.6, 17.5 to 36.0; parser 1 to 69 ms per token on a CPU against 54 to 136 ms for the forward pass; the `[[[` note and "not principled" | confirmed | https://arxiv.org/abs/2305.13971 |
| 10 | Biagiola et al. exists (arXiv:2606.21619, 19 June 2026). Table 1: Gemma-2-2B 32.0 against 35.7 (HumanEval), 44.9 against 50.4 (MBPP), token limit 3.4% against 0.1%. Table 4 (TOML): +53%, +2.5%, 48.1 to 28.6, 62.5 to 1.9 (97% loss), 0.2% errors for the 80B model | confirmed | https://arxiv.org/html/2606.21619v1 |
| 11 | Same paper: "Fine-tuning the model on filter-conforming data cut time-outs by 53-73% and token-limit hits by 74-94%" | corrected: -> GRPO fine-tuning with a type-checker reward; Gemma-2-2B token-limit hits 3.4% to 0.2% and 6.7% to 1.7%, time-limit hits 3.0% to 0.8% and 2.8% to 1.3% (the old ranges were own ratios, cut instead of rounded: 54-73% and 75-94%) | https://arxiv.org/html/2606.21619v1 (Table 3, section 5.2) |
| 12 | NVIDIA blog exists, dated 8 May 2026: 13 models, 299 tasks, 62.5% to 75.2%, Qwen3-0.6B 16.7% to 59.2%, SmolLM2-360M 29.4% to 57.2%, Qwen2.5-3B 74.6% to 75.6%, shell constructs 69.4% to 69.0% | corrected (numbers confirmed): author "NVIDIA AI Red Team" -> "Joseph Lucas (NVIDIA AI Red Team)" | https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/ |
| 13 | Monitor-guided decoding: compilation rate +21.77% to +24.69% relative; CodeGen-350M with the monitor above text-davinci-003 from 4 samples up | confirmed | https://arxiv.org/html/2306.10763v2 |
| 14 | Domino: GSM8K 0.415 to 0.308 (Mistral-7B) and 0.155 to 0.0 (Llama-2-13B) at k = 0; 0.418 and 0.157 at k = infinity; templates lose up to 11 points | confirmed (Table 4; note added that Table 2 lists 0.262 for unconstrained Llama-2 13B) | https://arxiv.org/html/2403.06988v1 |
| 15 | "llama.cpp grammar sampling 0.80x to 0.86x throughput in Domino's tests"; Domino 0.78x on a C grammar and 1.77x on JSON with a schema | corrected: 0.80x to 0.86x -> 0.74x to 0.87x over four grammars (0.80x / 0.86x is the GSM8K JSON task only; C grammar 0.74x / 0.81x; fixed template 0.55x to 0.69x); the Domino figures are confirmed | https://arxiv.org/html/2403.06988v1 (Tables 2, 3) |
| 16 | XGrammar: 160 MB to 0.46 MB, context-dependent tokens 1,134 to 120, under 40 µs (JSON) and under 200 µs (XML, Python DSL); table 6.1 gave its effect on correctness as "Not measured" | corrected (numbers confirmed): "Not measured" -> syntactic correctness 62% to 100% (function calling) and 80% to 100% (XML) in its Table 4 | https://arxiv.org/html/2411.15100v3 |
| 17 | Grammar-aligned decoding: intended form only 30% of the time in the paper's example; "even after 2,000 iterations the KL divergence barely improves" | confirmed | https://arxiv.org/html/2405.21047v4 |
| 18 | Engine sources. llguidance: "severely impacted the distribution of model output", check up to 4 tokens back, about 50 µs for a 128k vocabulary. llama.cpp: sample, check one token, resample under the grammar on rejection; end of generation only with an empty stack; "Unexpected empty grammar stack"; assertion when no token is selected; README "performance gotchas" | confirmed | https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md, https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp, https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp |
| 19 | Bunel et al.: 71.91 to 72.07 (full data), 12.58 to 9.81 (10,000 examples), learned syntax 17.02, cited as "(Table 5)" | corrected (values confirmed): Table 5 -> Table 3; added the learned model's full-data score (69.37) and the size-matched model without syntax (73.67 / 13.14) | https://arxiv.org/abs/1805.04276 |
| 20 | The abstract-only papers exist and say what is quoted: 2609.23742 (schema validity 78.6-92.9% to 100%), 2608.03065 (masks up to 700x faster), 2608.25761 (0.5B to 7B, small models gain most), 2502.14969 (5%-10%, small models most), 2408.02442, 2503.05507 | confirmed (titles, authors and abstracts through the arXiv API) | https://export.arxiv.org/api/query?id_list=2609.23742,2608.03065,2608.25761,2502.14969,2408.02442,2503.05507 |
| 21 | Raspanti et al., ACL 2025 industry track: grammar constraints most useful for small models (was tagged [UNVERIFIED]) | corrected: unverified search-engine summary -> the abstract's own sentence, read at the ACL Anthology; row added to section 7 | https://aclanthology.org/2025.acl-industry.34/ |
| 22 | Licences of the 17 repositories in section 7 | corrected (15 confirmed with the GitHub API): python/cpython "Other" -> Python Software Foundation License Version 2 (LICENSE file read); eth-sri/domino "none detected" -> the repository is empty | https://api.github.com/repos/python/cpython, https://github.com/python/cpython/blob/3.12/LICENSE, https://api.github.com/repos/eth-sri/domino |
| 23 | Python and tool facts of section 2.4: MAXINDENT 100, MAXLEVEL 200; the reference's quotations and section numbers; block rule; Lark `Indenter`; SynCode `python_parser.py`; Melcer quotation; TransCoder INDENT / DEDENT tokens; SentencePiece quotation; llama2.c trainer options; local limits (200 / 201 parentheses, 99 / 100 blocks) and the 7 classes that `ast.parse` accepts | confirmed | https://github.com/python/cpython/blob/3.12/Parser/tokenizer.h, https://docs.python.org/3/reference/lexical_analysis.html, https://github.com/karpathy/llama2.c/blob/master/tinystories.py |
| 24 | "No published grammar-constrained decoding result for a code model under 10M parameters"; "the smallest models in the sources read are SmolLM2-360M and CodeGen-350M" | corrected: narrowed to text-token transformers that write Python; small-model precedents named (Bunel et al. 2018, Yin and Neubig 2017, Kapur et al. 2024); smallest transformer language model in the sources -> T5-Base, about 220M parameters | https://arxiv.org/abs/1704.01696, https://arxiv.org/abs/2405.20519, https://huggingface.co/google-t5/t5-base |
| 25 | Local numbers (script run again, 267 s, Python 3.12.13): 9,469 functions, layout invariants, depth histograms, truncation repair 99.98%, ten vocabulary rows, 0 rejections, 14.5 rows masked, stand-in table | corrected (every number reproduced): "costs no length" / "0.1% fewer tokens" -> -0.1% against llama2.c-style BPE but +2.2% against the like-for-like baseline; "turns every budget overrun into code that compiles" -> only when the lines before the cut are valid; added: 2 of 9,469 functions exceed 8 open blocks, fixed-size stacks were never run | research/06_local_measurements.py |

Not re-checked: the mechanism descriptions of section 6.1 beyond their numbers, the cycle estimates (they come from
`CONTEXT.md` and are not measurements), and anything about the real training corpus, which did not exist for this
report. `SOURCES.md` was not edited by the fact check; the three papers added to section 7 need rows there if they
are used.
