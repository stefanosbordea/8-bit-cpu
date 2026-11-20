# 07. Weight-space editing without training: "Sparks of Weight-Space Programming" and the established literature

**PAPER FOUND: https://rikiyat.github.io/blog/weight-space-programming/** (a blog post, published 2 November 2026;
complete logs at https://github.com/RikiyaT/lm_understand-logs). Research date: 2026-11-13. Everything stated about it
below was read in the post, in the log repository or on the model card; quotes are verbatim. The post has no numbered
sections, so quotes are located by heading ("Experiment 8", "Additional experiment", "Response to Comments") or by
file path in the log repository.

Fact-check (2026-11-13, a second agent): the page was fetched again (HTTP 200) and the title, the four authors, "MIT
CSAIL", "Published November 2, 2026" and `note = {Blog post}` were read on it; the log repository and the model card
resolve [VERIFIED]. Tags: [VERIFIED] = the source was opened again and number and wording match; [UNVERIFIED: reason]
= not supported by a source that was opened; "[fact-check: ...]" marks text that the fact-checker changed or added.
The section "Verification log" at the end lists every checked claim.

Short names used below:
- [POST] https://rikiyat.github.io/blog/weight-space-programming/
- [LOGS] https://github.com/RikiyaT/lm_understand-logs (paths are given relative to the repository root)
- [CARD] https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming

## 1. Answer in one paragraph

The work exists and the numbers we were given are right: "Sparks of Weight-Space Programming: LLMs can natively read &
write neural network weights" by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (all MIT CSAIL) reports that
OpenAI's `gpt-6-astra` changed 131,072 of the 0.8 billion weights of Qwen3.5-0.8B and MATH-500 went from 7.9% to 48.1%,
with gradients and training forbidden by the harness [VERIFIED]. It is a blog post (its own citation entry says "Blog
post"), not on arXiv and not peer reviewed. What the short description leaves out decides how useful it is to us. (1)
The edit is a known interpretability move composed by the model: a difference-of-means "repetition" direction taken from
four hand-written pairs of texts, removed from 64 output columns of the feed-forward block in each of two layers with
the rank-one update W = W0 - 6 u (u^T W0). (2) The one run that worked used 50 forward passes on practice problems, 158
turns, 80 Python runs and 20.3 hours [VERIFIED; fact-check: "It needed" changed to "used": the logs show what was spent,
not what is necessary, see 2.3 and 2.7]; with 0 or 5 forward passes no edit was submitted, and with 10 or 25 the result
was 12.3% and 14.1%. (3) It worked because 93.0% of the base model's answers ran into the 32,768-token limit in a
thinking loop; the edit lowered that to 33.1%. On Gemma 3 270M, which rarely loops, the same harness moved MATH-500 from
6.8% to at most 7.7% [VERIFIED; fact-check: "the same procedure" corrected: the Gemma runs used a later prompt version
and a more lenient grading rule than the Qwen runs, so the two pairs of numbers are not the same benchmark variant, see
2.4]. The authors' own conclusion: "Editing weights could remove an obstacle that hid an ability, but it could not add
an ability the model lacked." For LS-LLM this means: plan the final gradient-free stage as a repair stage for failures
we can measure (generation loops and missing end of output, mean shifts after quantisation, saturation), not as a source
of coding skill; copy the post's measurement protocol (original and edited model on the same practice problems,
dose-response, a random-direction control, freeze the edit, confirm on untouched practice problems); and use the
forward-only tools with the strongest record (a record at 2 to 4 bits on far larger models, see section 3) for the int8
stage: layer-wise integer refits against the float teacher (OBC, GPTQ, COMQ: no backpropagation), bias correction
through the accumulator preload, direction removal folded into the weights (our synthetic check: such a rank-one edit
survives int8 rounding with power-of-two scales at width 64), and zeroth-order or evolutionary search over small groups
of integer parameters. ROME and MEMIT need gradients only to find the target vector; AdaRound and BRECQ need gradients;
task vectors need fine-tuned checkpoints.

## 2. Proven

### 2.1 What the source is

- Title, authors, date: "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights",
  Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, affiliation "MIT CSAIL" for all four, "Published November 2,
  2026"; a section "Response to Comments" was "Added November 12, 2026". The citation block ends with
  `note = {Blog post}`. [POST] [VERIFIED]
- Not on arXiv on 2026-11-13: the arXiv API query `all:"Sparks of Weight-Space Programming"` returned 0 results and
  `au:Takehi AND au:Khattab` returned 0. https://export.arxiv.org/api/query?search_query=all:%22Sparks+of+Weight-Space+Programming%22
  [VERIFIED: both queries repeated by the fact-checker (0 and 0); a web search found secondary coverage only (for
  example DataCamp, The Register, iThome), no arXiv or OpenReview version.]
- The editing model: "'Astra' in the post is OpenAI's `gpt-6-astra`", run through the Codex CLI with every built-in
  tool switched off; reasoning effort `max` in Experiment 8; "All 1,125 turns in this release are tool-free"
  (meaning: no tool other than the harness's own sandbox). [LOGS] README.md [VERIFIED]
- Headline claim (TL;DR): "asked to make Qwen3.5-0.8B better at math without gradients, Astra identifies an issue in
  its weights causing reasoning loops that Qwen's own model card warns about, and fixes this by natively editing
  0.016% of the weights. With the same sampling settings, this weight correction raises MATH-500 from 7.9% to 48.1%
  and generalizes out of the box to instruction following, raising IFEval by 3x." [POST] [VERIFIED]
- The model-card warning is real: "In thinking mode, we have observed that when using the recommended sampling
  parameters, Qwen3.5-0.8B is more prone to entering thinking loops compared to other Qwen3.5 models, which may
  prevent it from terminating generation properly." https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/README.md#L856
  [VERIFIED: line 856 of the raw README.]
- An outside re-run of the released checkpoint (by Ahsen Khaliq at Hugging Face, 500 problems x 4 samples) gave 51.0% on
  MATH-500 with 31.6% of answers at the token limit; a 25-problem check of the base model gave 5.0%. [CARD] [VERIFIED]
  [fact-check: the re-run is published in the lead author's own Hugging Face repository and repeats the evaluation of
  the released checkpoint, not the editing run. A second outside evaluation of the same edit gave 51.7% on MATH-500
  (Salloum and Ali, https://arxiv.org/abs/2611.14210, section 2.7; they did not re-run the base model). For the
  unedited model, Hu et al. report 8.35% on MATH-500 in their own harness (Table 2, https://arxiv.org/abs/2611.09382),
  against 7.9% in the post.]

### 2.2 Experiment 8: procedure

What the editing model is given and may do (all from the opening message, [LOGS] exp8_math/runs/p50c/turns/t001/prompt.txt):

- The task: "make it better at mathematics by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the
  network when it fails at math problems and repair that with targeted changes." Only weight values may change ("same
  tensor names and shapes").
- Weights: not pasted as text. A persistent Python kernel (CPU, no internet) exposes `load(name)`,
  `load_original(name)`, `save_edit(name, tensor)`, `drop_edit(name)`; numpy, scipy, scikit-learn, PyTorch on CPU with
  "Gradients are disabled"; at most 300 code runs, 900 seconds and 96 GB each, output cut at 10,000 characters.
- Activations and outputs: only through a forward-pass tool with a budget of 10, 25 or 50 passes. A pass is either a
  generation pass (up to 512 generations with the evaluation's decoding; `"model": "both"` runs original and edited
  weights on the same items with the same seeds and returns "the accuracy difference edited - original over items with
  its standard error, and how many samples went from wrong to right and back"; optional mean residual-stream and MLP
  activations) or a teacher-forced pass (up to 1,024 texts; token log-probabilities, top-k, hidden states per layer,
  MLP activations, attention for at most 2 texts).
- Data: practice problems with answers, "none of them is in the evaluation": 7,469 GSM8K train, 7,275 MATH train, 714
  AIME problems.
- Forbidden: "(a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative
  fitting of weights to a loss; (b) running the model other than through the forward-pass tool: do not implement or
  simulate the network's computation on any input in the kernel; (c) internet access or search".
- Refused by `save_edit`: the final norm, the embedding / output layer, and "any edit that only multiplies a whole
  tensor by a constant". "Global knobs (making the whole output distribution sharper or flatter) are not what we are
  looking for".
- The prompt names the failure as an example: "if the model often fails because it repeats itself and never finishes,
  finding and fixing what drives that repetition counts." (See 2.4: a prompt version without this example produced no
  edit at 10 passes.)
- Method rules given to the editor: "Localize causally. That a component is more active on correct answers is not
  enough"; "try several strengths to see a dose-response"; "Accuracy over n generations has a standard error of about
  50/sqrt(n) percentage points"; "Keep an edit only if its gain is clearly larger than its standard error on problems
  that were not used to choose it."

[VERIFIED: every quote in the list above was found verbatim in turns/t001/prompt.txt.]

How the 50-pass run chose the weights ([LOGS] exp8_math/runs/p50c/: work_notes/experiment_history.md,
work_notes/notes_before_pass26.md, work_notes/final_report.md, tool_calls.jsonl, readable/passes.md):

1. Diagnosis. Pass 1, original model on 96 practice problems: 11/96 correct, 85/96 cut at 32,768 tokens, "cut tails
   ~95% duplicate8-word sequences". Target: repetition, "not math facts".
2. Contrast texts. The editor wrote four problems, each with a "productive" and a "repetitive" continuation of the
   thinking (8 texts), and read them in one teacher-forced pass, recording the residual stream after every layer and
   the MLP activations of all 24 layers. [fact-check: one earlier pass on these examples was spent with a chat
   template that "adds an empty think block"; the direction comes from the repeated pass.]
3. Direction. Per layer, the mean over the four pairs of (repetitive minus productive) residual vectors, averaged over
   the last 36 tokens and normalised (code of the Python run that precedes forward request 5 in tool_calls.jsonl).
4. Where it is written. The growth of that contrast per block was split into an MLP part and an attention part;
   layers 15 and 19 show the largest growth among layers 0 to 22 (0.472 and 0.530, MLP parts 0.263 and 0.250, output
   of the same run; layer 22 has 0.309 with MLP part 0.110; layer 23 is not printed by that run). [fact-check:
   corrected from "among layers 0 to 21 ... we did not read the values for layers 22 and 23": the same output lists
   layer 22.]
5. Which neurons. Score per MLP neuron = (activation difference between the two kinds of text) x (projection of the
   neuron's output column on the direction).
6. Causal tests, each a paired generation pass: zeroing one neuron (layer 19, neuron 11) +6.25 points (SE 4.35) on 32
   problems; removing the direction from layer 19 only +6.25 (SE 4.35); removing it in layers 15 and 19 at strength 1,
   on 128 practice problems not used before, 11/128 -> 42/128 (+24.2 points, SE 4.0).
7. Dose-response: strength 2 on 256 unseen generations 21/256 -> 110/256 (+34.8 points, SE 3.5); strength 3
   21/256 -> 123/256 (+39.8, SE 3.8, exploratory).
8. Controls: a random direction orthogonal to the repetition directions with the same Frobenius norm gave +1.56
   points (SE 3.52); on six other hand-written problems the log-probability per token of the repetitive branch fell by
   0.024 / 0.049 / 0.104 nats at strength 1 / 2 / 4 while the productive branch moved by +0.002 to +0.005; the
   log-probability of closing the thinking early stayed near -19; layer 15 alone -0.081 and layer 19 alone -0.034 at
   strength 4; halving two attention heads (15.7 and 19.5) was "less selective" and was removed.
9. Sparsify and strengthen: keep only the 64 strongest "writer" columns per layer (chosen on the original probe;
   they carry "~91%/98% of transfer repetitive-minus-productive writing contrast, but50%/40% of full directional
   weight norm"), raise the strength to 6.
10. Freeze, then confirm on 1,152 practice problems that played no part in the selection.

[VERIFIED: steps 1 to 10 were compared with work_notes/experiment_history.md, work_notes/final_report.md,
readable/passes.md and with the code and output of the Python run before forward request 5 in tool_calls.jsonl.]

Form of the edits:

- Final edit of the 50-pass run: "Formula: W = W0 - 6*outer(u, mask64*(u@W0))" applied to
  `layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight` (each 1,024 x 3,584). One rank-one update per
  matrix, restricted to 64 columns: 2 x 64 x 1,024 = 131,072 values; relative Frobenius change 10.6% and 8.1%;
  "130,188 of the 131,072 touched weights differ from the original after rounding to bfloat16". No bias change, no
  added circuit, no lookup-like structure, no change to norms or embeddings. [LOGS] exp8_math/runs/p50c/work_notes/final_report.md,
  exp8_math/README.md [VERIFIED]
- 10-pass run: "removes a loop (\"repetition\") direction from 30 output matrices (attention and MLP) in layers 8-22,
  W' = W - 1.5 u(u^T W)". 25-pass run: "switches off attention head 7 of layer 15 (zeroes its 256 output columns)".
  [LOGS] exp8_math/README.md [VERIFIED]
- Gemma 3 270M runs: column scalings only ("halves 64 selected MLP output columns across layers 6-17"; "multiplies 4
  MLP output columns (layers 8, 14, 17) by 1.5"; "doubles neuron 920 of layer 17's MLP"; "multiplies head 2 of layer
  11's attention output by 1.1 and halves 8 MLP columns of layer 11"). [LOGS] exp8_math/gemma_followup/README.md
  [VERIFIED]

### 2.3 Experiment 8: evaluation protocol

- Practice against held-out: the editor only ever ran practice problems (GSM8K train, MATH train, AIME). The reported
  benchmarks were run after the session: "Evaluation: MATH-500, GSM8K and six other benchmarks, thinking on, up to
  32,768 tokens. Astra never sees them". [POST] Experiment 8 [VERIFIED]
- Decoding, identical for base and edited model: thinking on, temperature 0.6, top-p 0.95, top-k 20, presence
  penalty 1.5, at most 32,768 tokens; the last boxed answer is graded. Samples per problem: 16 on AMC23, 4 on
  MATH-500, Minerva Math, OlympiadBench and GPQA Diamond, 2 on GSM8K and IFEval, 1 on MMLU-Pro (a fixed 1,000
  questions). [POST] Experiment 8 [VERIFIED]
- Forward passes: budgets of 10, 25 and 50, one editing run per budget. In the 50-pass run 69 requests were made, 19
  failed and "did not count", and the 50 counted passes were 34 generation passes (32 to 128 problems, 1 to 4 samples
  each; all but the opening one ran original and edited weights side by side) and 16 teacher-forced passes (2 to 14
  texts) [our count from exp8_math/runs/p50c/readable/passes.md]. [VERIFIED: a recount gives the same 69 requests,
  19 failed, 34 generation and 16 teacher-forced passes.] [fact-check: requests 8 to 11 are four counted passes on
  one unchanged edit (layers 15 and 19, strength 1) and the same 32 problems: the request text is identical, no
  Python run that saves an edit lies between them, and the relative change of the two tensors printed before
  request 8 and after request 12 is the same (3.526% and 3.375%). The editor's notes: "duplicate outputs are not
  independent evidence". So 3 of the 50 counted passes added no information, and the repeats do not agree (7, 7, 9
  and 9 of 32 correct for the edited model); the editor's notes say elsewhere that "Separate generation runs have
  batching-dependent RNG".]
- Sessions: 50-pass run 158 turns, 80 Python runs, 4 conversations, 20.3 hours (4.7 of them waiting for the model
  service); 10-pass run 74 turns, 59 Python runs, 2 conversations, 11.6 hours; 25-pass run 86 turns, 45 Python runs,
  3 conversations, 8.5 hours. A fresh conversation was started "every 40 turns or after a harness restart" with "the
  task, Astra's notebook and its current edit". [LOGS] exp8_math/README.md [VERIFIED]
  [fact-check, not stated in the README: in conversation 1 the Codex session writer failed from turn 15 to turn 40
  (`stderr_tail` of all 26 `run.json` files: "rollout writer failed ... Quota exceeded"). In those turns the
  cumulative input counter stays between 337,614 and 338,202 tokens instead of growing by about 34,000 per call,
  turns 16 to 27 are 12 Python runs that mostly re-read notes and result files while the pass count stays at 5, and
  the repeated passes named above fall in turns 28 to 38. This is consistent with the conversation history not being
  saved in those turns (our inference). Part of the 158 turns, 50 passes and 20.3 hours is therefore harness
  trouble, not editing work.]
- The editor's own confirmation on practice data (frozen edit, 1,152 problems, one sample each, both models):
  10.6% -> 60.4%; "The combined gain was 49.8 percentage points, SE 1.5, with 579 rescues and five regressions." It
  labelled them itself: "These are practice results, not unseen benchmark scores". [LOGS] exp8_math/runs/p50c/final_answer.md
  [VERIFIED]
  (The paired standard error sqrt(b + c - (b - c)^2 / n) / n with b = 579, c = 5, n = 1,152 gives 1.50 points: our
  arithmetic, it reproduces the reported value.) [VERIFIED: recomputed, 1.50 points.]
- Held-out results (percent): see table 6.2.

### 2.4 Experiment 8: what was fixed, ablations, limits

- Fixed failure: thinking loops that never reach an answer. Share of MATH-500 answers at the token limit: 93.0% base,
  87.2% (10 passes), 86.4% (25 passes), 33.1% (50 passes). [POST] Experiment 8 [VERIFIED]
- No ability was added. In the practice confirmation "Of those rescues, 576 were originally truncated outputs, while
  aggregate accuracy on originally finished outputs was unchanged" (114 problems, 93.9% before and after, 3 rescues
  and 3 losses). [LOGS] exp8_math/runs/p50c/final_answer.md and state.json [VERIFIED]
- Side effects logged by the editor on the same 1,152 problems: finished with a wrong answer 6 -> 187, finished
  without an answer 1 -> 58, cut-offs ending in whitespace 0 -> 51 [VERIFIED]. The editor's label for the whitespace
  cut-offs, "unresolved alternative degeneration", was written at an interim audit of 256 problems (15 against 0).
  [fact-check: the quote had been attached to the 1,152-problem counts; it belongs to the interim audit.]
  [LOGS] exp8_math/runs/p50c/state.json, work_notes/experiment_history.md
- Transfer beyond math, because "The loops are not special to math": IFEval 21.0 -> 62.3, GPQA Diamond 22.7 -> 36.6,
  MMLU-Pro 40.3 -> 45.5. [POST]; [LOGS] exp8_math/README.md [VERIFIED]
- Budget ablation (MATH-500 / GSM8K): 0 passes "no edit made: weights identical to the unedited model"; 5 passes "no
  edit submitted"; 10 passes 12.3 / 25.7; 25 passes 14.1 / 25.1; 50 passes 48.1 / 67.1. Not monotonic on IFEval: 49.0
  at 10 passes, 39.9 at 25. [LOGS] exp8_math/eval/RESULTS_excerpt.md [VERIFIED] [fact-check: "no edit" means that no
  edit was submitted. The same table lists the 5-pass run's own candidate, "which it did not submit (its own
  acceptance rule)", at 8.6 / 17.1 when evaluated anyway.]
- Prompt sensitivity: with prompt version 4 ("shorter, DEV/HOLDOUT, no failure example") the 10-pass run submitted no
  edit. [LOGS] exp8_math/eval/RESULTS_excerpt.md [VERIFIED] [fact-check: its unsubmitted candidate scored 10.1 / 16.1.
  The 25-pass row of prompt version 4 shows 14.1 / 25.1, the same two values as the 25-pass run of the post; the
  table does not explain the coincidence.]
- Model without the failure: Gemma 3 270M, "only 8.5% of its answers reach its token limit". MATH-500 6.8 base, 7.0 /
  7.6 / 7.7 / 7.0 after 5 / 10 / 25 / 50 passes; "Astra found ordinary reasoning errors, such as wrong arithmetic and
  invalid algebraic steps, with no single component behind them to switch off." [POST] Additional experiment
  [VERIFIED] [fact-check, from exp8_math/gemma_followup/README.md and exp8_math/eval/RESULTS_excerpt.md: the Gemma
  scores use a more lenient grading than the Qwen scores ("the last `\boxed{}` answer if there is one, otherwise the
  answer stated after 'Final Answer' or 'the answer is', otherwise the last number of the response"). With the
  strict last-boxed grading used for Qwen they are 2.8 base and 3.5 / 3.1 / 3.8 / 2.6. The runs also differ from the
  Qwen runs in prompt (version 5, with fixed DEV and HOLDOUT sets and an instruction to submit an edit even when its
  gain is not significant), pass size (up to 2,048 answers) and decoding (temperature 1.0, top-k 64, top-p 0.95, at
  most 8,192 tokens, no thinking mode). Four further Gemma runs that are not in the post (strict grading stated in
  the prompt) scored 6.7 / 6.7 / 6.1 / 6.2 on the lenient scale, below the unedited 6.8.]
- Authors on the base score: "Turning thinking off raises the base score, but it does not fix the weights. It avoids
  the loops by not using the mode that has them." [POST] Response to Comments [VERIFIED]
- Authors on what the work is: "Weight-Space Programming is not a method. It is the name of an emerging concept".
  [POST] Response to Comments [VERIFIED]

### 2.5 Experiment 8: cost and release

- Editor usage in the 50-pass run, as reported by the Codex CLI at the last turn of each of the four conversations
  (turns 40, 80, 120, 158): input tokens 337,669 + 1,769,070 + 1,820,215 + 1,510,805 = 5,437,759 (5,180,032 of them
  cached), output tokens 10,081 + 50,977 + 51,220 + 36,739 = 149,017 [our sums; the README says usage "accumulates
  over a conversation"]. [LOGS] exp8_math/runs/p50c/turns/t040|t080|t120|t158/run.json
  [VERIFIED as sums; fact-check: these sums are a lower bound, not the total. The counter of conversation 1 stops
  growing at turn 15 (337,947 at turn 15, 337,669 at turn 40; see the session-writer failure in 2.3), so 25 calls of
  about 34,000 input tokens each are missing. With them the run used about 6.3 million input and 0.16 million output
  tokens (our estimate from all 158 `run.json` files). A later preprint quotes "89.7 million tokens" for this run
  (https://arxiv.org/abs/2611.14210): that figure equals the turn-by-turn sum of the cumulative counters (86,778,433
  + 2,909,694 = 89,688,127 in our recount), which the README warns against ("do not sum it over turns").]
- Forward passes ran on a GPU node: "A generation pass takes up to about half an hour at full size". The outside
  re-run of MATH-500 alone was "43.4M generated tokens, 51 min on an H200". [LOGS] prompt.txt; [CARD] [VERIFIED]
- Released: every prompt, reply, code run and evaluation file; each edit "as an exact diff from the base model" with
  `code/apply_edit.py`, which "rebuilds the evaluated tensors bit for bit"; the edited model on Hugging Face.
  Licences: "Logs, data and networks | CC BY 4.0", "Our scripts | MIT", MinAtar source quoted in prompts GPL-3.0;
  model card `license: apache-2.0`. [LOGS] README.md, LICENSE; [CARD] [VERIFIED: licence texts read in LICENSE and in
  the card header; the base model card also states apache-2.0.] [UNVERIFIED by us: the bit-for-bit rebuild is a
  statement of the README and of the model card, which reports a hash check; `apply_edit.py` was not run.]
- Not released for Experiment 8: "the agent harness code (the prompts, sandbox and forward-pass server; the rules it
  enforced are stated in the prompts)". The top-level `harness/` folder holds `codex_config.toml`, `instructions.md`
  ("Answer the user's request.") and `sandbox.py`. The editing model is a closed commercial model.
  [LOGS] exp8_math/README.md, harness/ [VERIFIED]

### 2.6 Other experiments in the post that are close to our size

- Experiment 4 (171,664 weights, no tools, one reply). A PPO policy that never succeeded (0/256 episodes) was shown to
  the model as text, "each tensor written as integers times one scale, 3 significant digits of the tensor's largest
  entry". The reply was 51 edit statements in a small edit language (for example `actor0.W[6, 122] = 80`) that changed
  6,073 of 171,664 weights (3.54%): "a small separate path through a few units per layer" that backs the car up and
  then hands control back. Result 1,024/1,024 test episodes; 10,134 output tokens. This is a hand-written circuit in
  integer-coded weights at about two thirds of our parameter count. [LOGS] exp4_fix_car/README.md [VERIFIED]
- Experiment 3 (three character-level transformers, 7.16 million weights, trained on Python code, romaji, English).
  With Python analysis of the four middle blocks and no forward pass, the model named the training text in 3 of 3
  networks in both attempts and found every copying (induction) head: three in the Python model (2.1, 3.1, 3.5), one
  in each of the others; measured copying scores were 0.22 to 0.55 against at most 0.02 for other heads. "Code repeats
  names far more than prose, so many copying heads pointed to Python." [POST] Experiment 3 [VERIFIED]
- Experiment 6 (write a 1,480,274-weight Go network from zero): with forward passes the model "plays practice games,
  looks at how it loses, changes the part of the network behind it, and keeps a change only if it wins more".
  [POST] Experiment 6 [VERIFIED]
- Experiment 7 (which of two Gemma 3 270M fine-tunes is sycophantic): "from the weights alone, Astra was unable to
  tell. In all four attempts"; with attention patterns on texts it wrote itself it picked the right model in 3 of 3
  pairs. [POST] Experiment 7 [VERIFIED]

### 2.7 Follow-up work (preprints dated 3 to 12 November 2026; abstracts read, Hu et al. also in its HTML version)

- Hu et al., "Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations": 16 editing
  trajectories with the same editor on Qwen3.5-0.8B and Qwen3.5-9B, seven benchmarks. Held-out gains "up to 29.45,
  4.42, 36.72, and 17.11 percentage points on MATH-500, GPQA Diamond, IFEval, and SciBench"; the largest came from
  "1,023 BF16 values in a single output column" (update W23 - u e507^T); "six of the 16 trajectories retain the
  original weights when candidate improvements cannot be reliably validated." Edit forms in its table: one column,
  dense rank-one, a head set to 0, columns x 2, x (-1), x (-2). https://arxiv.org/abs/2611.09382 [VERIFIED: abstract
  and HTML] [fact-check, same source, Table 2 and section 3.4: the largest practice ("development") gain seen during a
  search is larger than the held-out gain in all 14 rows; the retained 0.8B SciCode edit had a development gain of
  +38.89 points and 0.00 -> 0.00 held-out; a discarded 9B Math candidate had +29.69 in development and -0.35
  held-out; three independent Math searches on the 0.8B model gave held-out gains of +29.45, +5.00 and +27.30.]
- von Csefalvay et al., IWIBench: editors change a small model (typically Gemma 3 4B IT) for ten objectives with four
  tools; "structured response resisted every attempt, and answer-length control yielded only one successful solution".
  https://arxiv.org/abs/2611.08647 [VERIFIED: abstract]
- Nusrat and Nusrat: networks with 433 to 4,801 parameters and planted flaws; Claude Opus 5.5, without tools, "named
  the broken rule in 16 of 18 flawed networks" and "repaired each of the 16 by changing a median of 3 parameters";
  but "a simple statistic chosen for each flaw type found the same flaws" and "black-box testing found the confounds
  and the shift faster". https://arxiv.org/abs/2611.05341 [VERIFIED: abstract]
- Ahmed: convolutional networks "of about 125,000 parameters" given as text without tools; "zero-shot weight reading
  by this model is better described as local arithmetic plus dataset priors than as weight-flow tracing".
  https://arxiv.org/abs/2611.08070 [VERIFIED: abstract]
- Varam et al. (EPFL), EEG and ECG models: limits include "the inability of language models to improve already
  well-trained classification models". https://arxiv.org/abs/2611.13758 [VERIFIED: abstract; EPFL read in the HTML]
- Lehmann, written 57k-parameter recurrent agents: "93% of parameters are zero, and 90% of the rest are small integers
  or powers of two"; held-out success 56% against 97-100% for the same writers' Python programs; "standard PPO removes
  most of their success". https://arxiv.org/abs/2611.16981 [VERIFIED: abstract]
- Kundu et al., quantum circuit angles: "accuracy drops to chance when fixed circuits exceed 30 weights, while
  optimization improves." https://arxiv.org/abs/2611.11537 [VERIFIED: abstract]
- Singh, transformers written by hand by a coding agent: 959 programs over 13 tasks; "All ten algorithmic tasks,
  including addition, multiplication, sorting, parity and GCD, reached 100% accuracy within three attempts"; the
  parity model has 174 parameters, 35 of them non-zero.
  https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html [VERIFIED]
- [fact-check: added; the title-only arXiv query of 6.4 missed it] Salloum and Ali, "Tree-Ising: Repairing Language
  Models with Bit-Level Ising Edits": the Experiment 8 repair recast as a discrete optimisation without an LLM.
  "Forward statistics define an objective that weakens a loop direction, strengthens a concluding direction, and
  preserves the layer's output on productive tokens"; the allowed edits are "sign flips and exponent shifts" of the
  stored BF16 values, solved as one QUBO per output row by simulated annealing. "On the agent's two layers, Tree-Ising
  achieves 54.6% on MATH-500, outperforming the agent's edit (51.7%) under the same evaluation protocol"; "It requires
  only four forward-pass requests and zero LLM tokens". Limits stated in the paper: "Our repair recipe was formulated
  retrospectively after analyzing the agent's public logs", it reuses the agent's two layers, 100 of the 500 MATH-500
  problems were seen during development (the paper reports the same gap on the other 400), and the base model was
  not re-run. On Qwen3.5-2B, "with layers and dose
  selected using only the base model", greedy GSM8K went from 21.6% to 69.4%. For us this is the closest published
  example of a forward-only discrete search over stored weight codes; it is at 0.8B and 2B parameters and on BF16
  codes, not at our size and not on int8. https://arxiv.org/abs/2611.14210 [VERIFIED: abstract and HTML]

### 2.8 Established methods: what each needs

[VERIFIED, fact-check for this whole subsection: all 40 arXiv numbers cited here resolve to the stated titles; 61
quoted strings and numbers were compared with the arXiv abstracts and 54 match verbatim; the other 7 (the MINT
percentages and "Input X Gradient", "does not require gradient-based optimization" in Arditi et al., "20% and 40%" in
Michel et al.) were found in the full texts, as were the statements quoted from the method sections of ROME, MEMIT,
GPTQ, AdaRound, BRECQ and Nagel et al. 2019. The two surveys were not opened beyond their abstracts. Changes are
marked "[fact-check: ...]" at the place.]

Editing of stored associations
- ROME (Meng et al. 2022). Step 1 chooses the key k* from forward passes; Step 2: "value v* is optimized to select the
  object", and "the optimization does not directly alter model weights; it identifies a vector representation v*";
  Step 3 is closed form: W' = W + L (C^-1 k*)^T with L = (v* - W k*) / ((C^-1 k*)^T k*), C = K K^T "computed using
  100,000 samples of hidden states". Only Step 2 needs an optimiser run through the model; the ROME text we read
  says "optimized", and MEMIT, which reuses the construction, says "using gradient descent".
  https://arxiv.org/abs/2202.05262
- MEMIT (Meng et al. 2022): "thousands of associations for GPT-J (6B) and GPT-NeoX (20B)"; the target vectors are
  found "by optimizing the residual vector delta_i using gradient descent"; the weight update is a closed-form
  least-squares solution with C0 = lambda E[k k^T], covariance from 100,000 samples, lambda = 15,000 on GPT-J.
  https://arxiv.org/abs/2210.07229
- Limits: with many sequential ROME or MEMIT edits a model "continually forgets previously edited facts and the
  ability to perform downstream tasks", in a gradual and then an abrupt phase (Gupta et al. 2024,
  https://arxiv.org/abs/2401.07453). Causal-tracing results "do not provide any insight into which model MLP layer"
  to edit (Hase et al. 2023, https://arxiv.org/abs/2301.04213).
- Code models: six editing techniques on CodeLlama 7B, CodeQwen1.5 7B and Stable-Code 3B (CLMEEval, 21K+ and 16K+
  samples): the external-memory method GRACE scores highest on effectiveness and specificity [fact-check: scope
  added] and "generalization ... is a universal challenge" (Li et al. 2024, https://arxiv.org/abs/2411.06638). MINT
  patches "usually one or two neurons" of code models and reports ExactMatch gains of 5.7%-20.8% on StarCoder2-3B
  and 3.9%-18.5% on CodeLlama-7B over earlier repair methods (its third model is CodeGen-2B), but locates the
  neurons with "Input X Gradient", so it needs a backward pass (Gu et al. 2023,
  https://arxiv.org/abs/2312.05356).
- Surveys: Yao et al. 2023 (https://arxiv.org/abs/2305.13172) and Zhang et al. 2024 (https://arxiv.org/abs/2401.01286);
  abstracts read only.

Directions, steering, weight orthogonalisation
- Arditi et al. 2024: one difference-of-means direction per model, across "13 popular open-source chat models up to
  72B parameters"; folding the removal into the weights is W_out' <- W_out - r r^T W_out (their Equation 5) for every
  matrix that writes to the residual stream, and "it does not require gradient-based optimization". This is the form of the Experiment 8
  edit at strength 1. https://arxiv.org/abs/2406.11717
- Contrastive Activation Addition: steering vectors are "the difference in residual stream activations between pairs
  of positive and negative examples", added at inference (Panickssery et al. 2023, https://arxiv.org/abs/2312.06681).
  ActAdd "does not require any machine optimization" (Turner et al. 2023, https://arxiv.org/abs/2308.10248). A
  constant added to an activation is a bias term, so it can be stored instead of computed.
- Heretic, a tool that automates direction removal with "a TPE-based hyperparameter optimizer powered by Optuna"
  (gradient-free search over the removal parameters), AGPL-3.0. https://github.com/p-e-w/heretic [VERIFIED: README
  and GitHub licence field]
- LEACE: "a closed-form method which provably prevents all linear classifiers from detecting a concept while changing
  the embedding as little as possible" (Belrose et al. 2023, https://arxiv.org/abs/2306.03819).

Task vectors
- "We build task vectors by subtracting the weights of a pre-trained model from the weights of the same model after
  fine-tuning on a task"; negation and addition steer behaviour (Ilharco et al. 2022,
  https://arxiv.org/abs/2212.04089). Applying a task vector needs no gradients; making one needs a fine-tuning run.

Ablation of heads and neurons
- Michel et al. 2019: removing heads at test time needs only forward passes; "prune up to 20% and 40% of heads from WMT
  and BERT (respectively), without incurring any noticeable negative impact"; the ranking score I_h is the expected
  absolute gradient of the loss with respect to a head mask, so their ordering needs gradients; BERT ran 17.5% faster
  at batch 16 and 64 with half the heads pruned. https://arxiv.org/abs/1905.10650 (see also 05_speed_techniques.md)
- Voita et al. 2019: "pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU", with gates trained by
  "a differentiable relaxation of the L0 penalty" (gradients). https://arxiv.org/abs/1905.09418
- Wanda prunes by weight magnitude times input activation norm and "requires no retraining or weight update" (Sun et
  al. 2023, https://arxiv.org/abs/2306.11695); SparseGPT reaches "60% unstructured sparsity with negligible increase
  in perplexity" in one shot with layer-wise reconstruction (Frantar and Alistarh 2023, https://arxiv.org/abs/2301.00774).

Reading and editing in vocabulary space
- "a zero-pass approach, where parameters are interpreted directly without a forward/backward pass" by projecting them
  into the embedding space (Dar et al. 2022, https://arxiv.org/abs/2209.02535). Feed-forward outputs are additive
  updates to the token distribution; controlling them reduced "the toxicity of GPT2 by almost 50%" (Geva et al. 2022,
  https://arxiv.org/abs/2203.14680). The tuned lens trains "an affine probe for each block" because the logit lens "is
  often brittle" (Belrose et al. 2023, https://arxiv.org/abs/2303.08112).

Quantisation repairs
- Cross-layer equalisation and bias correction (Nagel et al. 2019): "data-free", "does not require fine-tuning or
  hyperparameter selection"; equalisation uses f(s x) = s f(x) "for any non-negative real number s" of ReLU; the method
  "corrects biases in the error that are introduced during quantization". No gradients. https://arxiv.org/abs/1906.04721
- AdaRound (Nagel et al. 2020): a layer-local loss optimised with "Adam ... for 10k iterations" on "1024 unlabeled
  images"; ResNet18 and ResNet50 at 4 bits "within an accuracy loss of 1%". No end-to-end backpropagation, but
  gradient descent per layer. https://arxiv.org/abs/2004.10568
- BRECQ (Li et al. 2021): block-wise reconstruction weighted by "the diagonal Fisher Information Matrix", which "is
  equal to the squared gradients"; 1,024 calibration samples; needs gradients. https://arxiv.org/abs/2102.05426
- OBC (Frantar et al. 2022): "an exact and efficient realization of the classical Optimal Brain Surgeon (OBS)
  framework ... extended to also cover weight quantization", "without any retraining, based only on a small amount of
  calibration input data". No gradients: layer inputs plus linear algebra. https://arxiv.org/abs/2208.11580
- GPTQ (Frantar et al. 2022): per layer the Hessian is H_F = 2 X_F X_F^T, built from layer inputs; calibration data
  are "128 random 2048 token segments"; "175 billion parameters in approximately four GPU hours"; OPT-175B perplexity
  on WikiText2: 8.34 at 16 bits, 10.54 with round-to-nearest at 4 bits, 8.37 with GPTQ at 4 bits. No backpropagation.
  [fact-check: these are the numbers of the largest model. For the smallest model in the same table, OPT-125M, the
  values are 27.65 at 16 bits, 37.28 with round-to-nearest and 31.12 with GPTQ at 4 bits: the refit recovers about
  two thirds of the loss and leaves 3.47 perplexity points, against 0.03 for OPT-175B. The small-model row is the one
  nearer to our size, and it is still about 480 times larger.] https://arxiv.org/abs/2210.17323
- GPTAQ (Li et al. 2025): "we always match the quantized layer's output to the exact output in the full-precision
  model", closed form, "20 more lines of code than GPTQ". https://arxiv.org/abs/2504.02692
- COMQ (Zhang et al. 2024): "sequentially conducts coordinate-wise minimization of the layer-wise reconstruction
  errors", "involves only dot products and rounding operations"; 4-bit Vision Transformers lose "less than 1%" top-1.
  Backpropagation-free by construction. https://arxiv.org/abs/2403.07134
- REPAIR (Jordan et al. 2022) restores activation statistics by "rescaling the preactivations", forward passes only.
  https://arxiv.org/abs/2211.08403

Zeroth-order and evolutionary search
- MeZO: fine-tuning "with the same memory footprint as inference", "comparable performance to fine-tuning with
  backpropagation across multiple tasks", and it "can effectively optimize non-differentiable objectives (e.g.,
  maximizing accuracy or F1)" (Malladi et al. 2023, https://arxiv.org/abs/2305.17333).
- Evolution strategies for "full-parameter fine-tuning of LLMs at the billion-parameter scale" (Qiu et al. 2025,
  https://arxiv.org/abs/2509.24372); ES scales to "over a thousand parallel workers" (Salimans et al. 2017,
  https://arxiv.org/abs/1703.03864).
- In the quantised domain: QES "performs full-parameter fine-tuning directly in the quantized space" with "accumulated
  error feedback" (Xu et al. 2026, https://arxiv.org/abs/2602.03120); QuZO fine-tunes "through low-precision (e.g., 4-
  or 8-bit) forward passes" (Zhou et al. 2025, https://arxiv.org/abs/2502.12346); QZO perturbs "the continuous
  quantization scale" (Shang et al. 2025, https://arxiv.org/abs/2505.13430); an evolution strategy that in each
  iteration "changes the values of the small percentage of weights" to neighbouring quantisation states (Pietron 2026,
  https://arxiv.org/abs/2605.05228); forward-only adaptation of an 8-bit ViT with CMA-ES on an input prompt (Niu et
  al. 2024, https://arxiv.org/abs/2404.01650).
- [fact-check: added after a wider search, abstracts only] Editing and steering on quantised models. MobiEdit
  "replaces full-precision BP with quantized forward-only gradient estimation" for knowledge editing of
  Qwen2.5-3B-Instruct on phone NPUs (Lu et al. 2025, https://arxiv.org/abs/2506.13772), so the gradient step of
  ROME-style editing has a published forward-only replacement. Activation steering under weight-only quantisation
  ("INT8 and NF4", four models of 7B to 9B): "sentiment steering survives quantization intact" and steering vectors
  keep "cosine similarity 0.989-0.998 for INT8" with their 16-bit versions, but for reasoning length "shortening is a
  step function with only 12-30% shortening (model-dependent) before discontinuous failure" (Bhandari and Wade 2026,
  https://arxiv.org/abs/2609.06473).

Agents that do interpretability
- MAIA (Shaham et al. 2024; MIT CSAIL [VERIFIED: affiliation read in the arXiv HTML]) "equips a pre-trained
  vision-language model with a set of tools that support
  iterative experimentation on subcomponents of other models"; it "produces descriptions comparable to those generated
  by expert human experimenters" and was used for "reducing sensitivity to spurious features, and automatically
  identifying inputs likely to be mis-classified". Vision models only. https://arxiv.org/abs/2404.14394
- FIND, a benchmark of "functions that resemble components of trained neural networks": an agent built from a
  language model with black-box access "can infer function structure", but its descriptions "tend to capture global
  function behavior and miss local details" (Schwettmann et al. 2023, https://arxiv.org/abs/2309.03886). Weight Oracles: language models fine-tuned to read weights reach "99% holdout
  accuracy" at simulating small transformers (Kabra et al. 2026, https://arxiv.org/abs/2610.07334); these are trained
  readers, unlike the general model in [POST].
- Tracr compiles "human-readable programs into standard decoder-only transformer models" (Lindner et al. 2023,
  https://arxiv.org/abs/2301.05062).

### 2.9 Our own check (synthetic, not LS-LLM weights)

Script `research/07_rank_one_int8_check.py`: random 64 x 172 matrices (the shape of the baseline SwiGLU
down-projection), int8 with a power-of-two scale per tensor or per row, random unit direction u, edit
W - a u (u^T W) rounded back to int8 with the scale unchanged, mean of 200 draws.

- Strength 1 on all 172 columns, Gaussian weights, per-tensor scale: 67.4% of the weights change, mean step 2.7 LSB,
  cosine between the rounded and the intended change 0.994, 2.0% of the original write along u is left, relative
  Frobenius change 12.4% (expected 1/sqrt(64) = 12.5%). With heavy-tailed (Laplace) weights: 50.8% change, mean step
  1.7 LSB, cosine 0.981, 4.5% left.
- Strength 6 on the 16 strongest columns: cosine 0.998 to 1.000, mean step 15 to 40 LSB, up to 0.15% of the weights
  clip at +-127, relative Frobenius change 48 to 49%.
- Reading: at width 64 a direction edit is large compared with one LSB and survives rounding. The same arithmetic
  shows its price: one direction is 1/64 of the residual space, and an edit of the Experiment 8 strength changes
  almost half the norm of the matrix, against 8 to 11% in a model of width 1,024.

[VERIFIED: the fact-checker ran the script again; every number in the three bullets above is reproduced (the seed is
fixed in the script). The result is arithmetic on random matrices and random directions, not evidence about a
trained model: a real failure direction and real weight statistics may behave differently.]

## 3. Uncertain or contested

- A capability of the model, or a known playbook run by a model? Goedecke: "this is exactly what a competent human
  researcher could have and would have done", "basically the Heretic abliteration playbook"
  (https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/). Willison
  reports that Matthew Honnibal "called the paper misleading for conflating" reading weights without tools and
  improving a model with code (https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/)
  [UNVERIFIED: Honnibal's own text not opened; quoted through Willison]. The authors answer: "We have not compared
  Astra with interpretability researchers" [POST]. What is not in dispute: the Experiment 8 formula at strength 1 is
  the weight orthogonalisation of Arditi et al. (section 2.8). [VERIFIED: the quotes were read on the three pages.]
  [fact-check: Tree-Ising (section 2.7) reaches a comparable repair with four forward requests and no LLM once the
  failure and the two layers are known, which supports the "known playbook" reading; its recipe was built after
  reading the agent's logs, so it does not show that the diagnosis itself is cheap.]
- The size of the gain depends on the base setting. 7.9% is the score of a model that loops with thinking on and a
  32,768-token limit; the authors state that turning thinking off raises the base score. The measured 40 points are a
  statement about removing that failure under that setting, not about mathematical skill.
- Reliability of the procedure is not established. There is one editing run per budget; 10 and 25 passes gave small
  gains and a different edit each time; a prompt version without the failure example gave no edit at 10 passes; in the
  follow-up study 6 of 16 trajectories ended without an edit (section 2.7). [VERIFIED] [fact-check: the same study
  reports three independent Math searches with held-out gains of +29.45, +5.00 and +27.30 points, and in the source
  itself 3 of the 50 counted passes were repeats of one measurement that did not agree with each other (2.3).]
- Status of the source: a blog post 11 days old at the time of writing, no peer review, a closed commercial editor
  model, the Experiment 8 agent harness not released. The edit itself is stated to be reproducible bit for bit (README;
  a hash check is reported on the model card; not run by us) and the score of the edited model was re-measured by two
  outside parties, 51.0% and 51.7% against 48.1% in the post (sections 2.1 and 2.7). [fact-check: was "re-run once".]
- "Reading" weights. Experiment 7 failed from weights alone; Ahmed attributes zero-shot reading at 125,000 parameters
  to "local arithmetic plus dataset priors"; Nusrat and Nusrat found that simple statistics found the same flaws
  (section 2.7). For us this argues for giving any editor forward passes, not weights alone.
- Side effects of the Experiment 8 edit: outputs ending in whitespace went from 0 to 51 of 1,152 practice problems and
  were left unexplained by the editor (section 2.4).
- Does localisation tell where to edit? Hase et al. found that causal tracing does "not provide any insight into which
  model MLP layer" to edit (https://arxiv.org/abs/2301.04213); Experiment 8 chose its layers by
  attribution and then confirmed by intervention, which is the safer order.
- For LS-LLM, three things are unknown and decide the value of the whole stage: (a) whether one failure class
  dominates once grammar-constrained decoding is in place; (b) whether at width 64 a failure is carried by one
  direction (the residual stream has only 64 directions for everything the model represents); (c) whether anything is
  left to repair after QAT that matches the hardware bit for bit (02_integer_inference.md and 03_qat_small_models.md
  note that training may absorb the constant rounding bias; no test was found).
- Gain of layer-wise refits at 8 bits: the gains we read for GPTQ and COMQ are at 2 to 4 bits. We opened no source
  that measures such refits at 8 bits on a model that was already trained with quantisation in the loop.
- Forward-only search and selection bias: scoring thousands of candidate edits on a few hundred practice tasks will
  find noise. None of the opened sources reports a search driven by pass/fail on a few hundred programs.
- Model editing over many edits: ROME and MEMIT degrade the model as edits accumulate
  (https://arxiv.org/abs/2401.07453); whether a handful of direction edits interact the same way in a tiny model is
  unknown.

## 4. Not tried at our scale (about 260k to 400k int8 parameters, 8-bit multiply-accumulate CPU, 450 KB)

- An LLM-agent edit of a language model below 270M parameters: not found. The smallest language model edited in
  [POST] is Gemma 3 270M (no gain). The small networks in [POST] and its follow-ups are policies and classifiers (433
  to 171,664 weights) or transformers written from zero for algorithmic tasks (down to 174 parameters). [fact-check:
  the 15 arXiv abstracts that mention "weight-space programming" on 2026-11-13 were read; none reports one.]
- Direction removal or any other model edit written into int8 weights with power-of-two scales, in a pipeline with
  int8 activations: not found. [fact-check: narrowed. The original line also covered steering vectors and "any
  model-editing method applied to int8 weights" and rested on one arXiv query. A wider search (abstract queries for
  "knowledge editing", "steering vector" and "abliteration" together with "quantiz*") finds activation steering on
  INT8 weight-quantised models of 7B to 9B and forward-only knowledge editing on a quantised 3B phone model (section
  2.8). Not found in that search: power-of-two scales, integer activations, a quantised model below 3B parameters,
  or a direction folded into integer weights.] Our synthetic check (section 2.9) is arithmetic on random matrices,
  not a model result.
- Direction removal at residual width 64: not found. Arditi et al. cover models from 1.8B to 72B parameters.
- OBC / GPTQ / COMQ on a transformer below 1M parameters, at 8 bits, after QAT: not found.
- Zeroth-order or evolutionary search directly on int8 weights with a code benchmark (tasks passed) as fitness: not
  found. The quantised-domain papers we read use LLMs (LLaMA2-7B, Llama-2-13B), a vision transformer, or CNNs.
- Bias correction or a stored steering vector on a bias-free Llama-style model through an accumulator preload: not
  found.
- ROME or MEMIT on a model with no subject-relation-object facts, or on any model below GPT-2 size: not found. The
  code-model studies use 2B to 7B models. [fact-check: was "3B to 7B"; MINT also uses CodeGen-2B.]
- Head or hidden-unit removal as a speed measure on a language model of 260k to 400k parameters: not found (also
  noted in 05_speed_techniques.md). [fact-check: narrowed from "ablation at 260k to 400k parameters: not found".
  Ablation as an analysis tool exists near this size: Nanda et al. 2023 ablate frequency components of a one-layer
  transformer for modular addition with width 128, 4 heads and 512 MLP units, about 0.23 million parameters by our
  count, https://arxiv.org/abs/2301.05217.]
- A hand-written circuit grafted into a trained language model of this size: not found. The one mixing result we read
  (written weights followed by PPO training) lost most of the written behaviour.
- Agent-driven interpretability with tool use (MAIA) on language models of this size: not found; MAIA is evaluated on
  vision models.

## 5. Ranked ideas for LS-LLM

Order: by expected value for the two targets divided by cost, with the measurement harness placed at the top because
every other row depends on it. "Paired rule" means: run original and edited int8 model on the same practice tasks with
the same seeds, count b tasks that go from fail to pass and c that go from pass to fail, SE = sqrt(b + c - (b - c)^2 / n) / n,
keep the edit only if (b - c) / n is at least 2 SE on tasks that played no part in choosing it. Test sets are never
used. All effects below are estimates unless a source is named.

| # | Idea | Expected effect on score / speed / memory | Cost to try | Risk | Keep-or-drop measurement | CPU change? |
|---|---|---|---|---|---|---|
| 1 | Edit harness and failure census on the bit-exact int8 simulator: `load` and `save_edit` on int8 tensors with shifts held fixed; paired runs; practice set split once into a "choose" part and a "confirm" part; every failed practice task put in one class (syntax error, runtime error, wrong output, loop or no end of output, truncated) | Score 0 by itself. Shows whether any failure class is large enough to repair and makes every later number defensible. Speed 0. Memory 0 | 1 day | Low | A null edit gives a paired difference of exactly 0; a random edit of matched norm stays within 2 SE of 0; the census is reported as counts | No |
| 2 | Direction removal for the largest failure class, if the census shows one (loops, missing end of output): difference-of-means direction from matched good and failing continuations read teacher-forced on the integer model; then W - a u (u^T W) on the matrices that write it most (attention output 64 x 64, down-projection 64 x 172) at a = 1, 2, 4; variants from the logs: zero or rescale one head or a few columns | Score: unknown. Upper bound = share of failures in that class that hide a correct continuation. Source evidence: +40 points where 93% of answers looped, at most +0.9 where 8.5% did (the two pairs were graded with different rules, section 2.4). Speed: 0 extra cycles per token (only weight values change) and fewer generated tokens if loops end. Memory 0 | 1 to 2 days after idea 1 | Medium. One direction is 1/64 of the residual space; a strength-6 edit on 16 columns changes about 48% of the matrix norm (section 2.9); constrained decoding may already remove the same failures | Paired rule on the "confirm" part; random-direction control within 2 SE of 0; validation loss on held-out practice text not worse; int8 saturation counts not higher. Otherwise drop | No |
| 3 | Accumulator preload used as an editable bias (the baseline has no bias terms): (a) bias correction, per row the mean difference between float teacher and int8 output on calibration text; (b) a per-token logit bias on the output layer (512 values); (c) a steering vector stored as a bias instead of a weight edit | Score: unknown after bit-exact QAT. Speed: 0 cycles if the row loop already loads the accumulator. Memory: up to 3 bytes per row; the baseline shape has 3,512 rows, so 10.5 KB (2.3% of 450 KB), or 1.5 KB for the output layer alone [our arithmetic from the shape in CONTEXT.md] | 1 day | Low for (a) and (b), medium for (c) | Validation loss, then the paired rule; the 24-bit accumulator bound re-checked with the preload included | No if the accumulator can be loaded from memory per row (02_integer_inference.md assumes a preload). Otherwise one constant input element plus one extra weight per row (1 cycle and 1 byte per row) |
| 4 | Layer-wise integer refit against the float teacher in the style of OBC / GPTQ / GPTAQ / COMQ: record the int8 model's inputs to a layer and the teacher's outputs, re-choose each int8 row on its fixed power-of-two grid by coordinate steps or the closed-form update (Hessians are 64 x 64 or 172 x 172). Also a quick int8 estimate for float candidates during shape search, without a QAT run per candidate | Score: recovers part of the float-to-int8 gap if one is left after QAT. Source evidence is at lower bit-widths and on far larger models (GPTQ at 4 bits on OPT-175B: perplexity 8.37 against 10.54 for round-to-nearest and 8.34 at 16 bits; on OPT-125M, the smallest model of that table: 31.12 against 37.28 and 27.65) [fact-check: small-model row added]. Speed 0. Memory 0 | 1 to 2 days | Low. Over-fitting the calibration text is the known failure | Validation loss on text not used for calibration, against round-to-nearest and against the QAT model; keep if lower by more than the run-to-run spread | No |
| 5 | Forward-only search over small groups of integer parameters with fitness from the bit-exact simulator: final-norm gain (64 values; with tied embeddings it is the untied handle on the logits), logit bias (512), per-row shifts, lookup-table entries (256 per table). Steps of +-1 kept only if better, or an evolution strategy with error feedback as in QES; a per-row discrete formulation solved by simulated annealing is reported at 0.8B parameters by Tree-Ising (section 2.7) [fact-check: added] | Score: unknown. Reaches parameters that gradients handle badly (shifts, table entries) and fitness that is not differentiable (tasks passed, syntax-valid outputs). Speed 0. Memory 0 | 2 to 3 days | Medium: selection bias on a small practice set | Search sees only the "choose" part; paired rule on the "confirm" part; validation loss not worse | No |
| 6 | Structured removal for speed: score each of the 40 heads and 860 SwiGLU hidden units by forward-only ablation, remove the least useful, refit the following matrix by least squares | Speed: about 252 to 272 cycles per removed hidden unit per token (two rows of 64 + 24 cycles, 64 multiply-adds in the down-projection, one element-wise step of 12 to 32 cycles), so removing 10% of the hidden units saves about 2.2% of 1.0 M cycles; one head is about 1/8 of a layer's per-head attention work [estimates from the cost model in CONTEXT.md]. Memory: 192 bytes per hidden unit. Score: unknown, expected to fall | 1 to 2 days | Medium to high: a smaller shape trained from scratch may do as well, and shape search already covers that | Cycles per token from the simulator against the paired practice score; keep only if cycles fall by at least 2% with a score change inside 1 SE | No |
| 7 | Agent session in the style of Experiment 8 on the int8 model: an LLM agent with a Python kernel over the int8 tensors, a fixed budget of forward passes on practice tasks, the freeze-then-confirm rule, a log of every step. Single layers can be shown as integer text, as Experiment 4 did with 171,664 weights | Score: unknown. In the sources edits helped only where one failure dominated, and 6 of 16 follow-up trajectories ended without an edit. For the report it is a replication at about 1/3,000 of the source model's size, whatever the outcome. Speed 0. Memory 0 | 1 to 2 days of agent time after idea 1 | High chance of no gain; low chance of harm under the protocol | Paired rule; the log is kept and a null result is reported as such | No |
| 8 | Task-vector mixing before or after quantisation: fine-tune on practice-like tasks with shifts frozen, take the difference to the base weights, add a fraction chosen by forward passes | Score: unknown at this size. Needs a training run, so only the mixing step is gradient-free | 1 training run plus half a day | Medium | Paired practice score along the mixing fraction, against both end points | No |
| 9 | ROME-style insertion of one key-to-value association into a down-projection: key and covariance from recorded activations, target vector found without gradients (search over 64 values, or copied from a context where the model is right; MobiEdit reports a forward-only estimate of this step on a quantised 3B model, section 2.8 [fact-check: added]), integer rounding as in idea 4 | Score: small. Code errors are rarely single associations and editing results on code models are weak (CLMEEval) | 2 days | High | Paired rule on the targeted cases and on untouched tasks | No |
| 10 | Hand-written circuit in units kept free during training (for example a bracket-depth or indent counter written as small integers) | Score: unknown; constrained decoding already tracks brackets and indentation outside the network | 3 days or more | High; in the one source we read, written weights and later training did not mix | Paired rule | No |

Notes for rows 1 to 3:
- Any edit that keeps every weight in [-127, 127] cannot overflow the 24-bit accumulator for rows of up to 511 inputs
  (rule in 02_integer_inference.md); a preload uses part of that headroom and must be included in the check.
- With greedy decoding the int8 simulator is deterministic, so the pairing is exact. The source needed fixed seeds and
  still warned: "long answers drift apart once a token differs".
- Width matters for strength: in the source the final edit changed 8 to 11% of the norm of a 1,024-wide matrix; at
  width 64 start at strength 1 on all columns (12.5% of the norm) before anything stronger.

## 6. Tables asked for in the brief

### 6.1 Methods: gradients, what is edited, what it fixes, fit for our int8 model

"Fit" assumes: int8 values, power-of-two scale per tensor or per output row held fixed during the edit, no bias terms
in the baseline, tied input and output embeddings, residual width 64.

| Method | Needs gradients? | What is edited | What it fixes | Evidence | Fit for our int8 model | Source |
|---|---|---|---|---|---|---|
| LLM agent edits weights with a Python kernel and a forward-pass budget (Experiment 8) | No, forbidden by the harness | 64 output columns of the MLP down-projection in each of 2 layers, rank-one; in other runs one head set to zero, or 30 rank-one edits | One dominant failure (thinking loops) | MATH-500 7.9 -> 48.1 in one run; Gemma 3 270M 6.8 -> at most 7.7 (lenient grading; strict 2.8 -> at most 3.8); 6 of 16 follow-up trajectories without edit | Mechanics fit: same shapes, same shifts, only int8 values change. Untested for language models below 270M parameters | [POST]; https://arxiv.org/abs/2611.09382 |
| LLM edits integer-coded weights given as text, one reply (Experiment 4) | No | 6,073 of 171,664 weights: an added path through a few units per layer | A policy stuck in a local minimum | 0/256 -> 1,024/1,024 episodes | Same number format as ours (integers times one scale); a policy network, not a language model | [LOGS] exp4_fix_car/README.md |
| Difference-of-means direction and weight orthogonalisation | No | Every matrix that writes to the residual stream: W - r r^T W | One behaviour carried by one direction | 13 chat models, 1.8B to 72B | Survives int8 rounding at width 64 (synthetic check, cosine 0.98 or higher); each direction costs 1/64 of the residual space | https://arxiv.org/abs/2406.11717 |
| Activation addition / steering vector stored as a bias | No | A constant vector added at one layer | Shifts one behaviour up or down | Llama 2 Chat behaviours | Needs a bias: accumulator preload or one constant input per row; finer than one weight LSB | https://arxiv.org/abs/2312.06681, https://arxiv.org/abs/2308.10248 |
| LEACE concept erasure | No (closed form from activation statistics) | A projection on activations, foldable into the next matrix | Removes linearly readable information | Part-of-speech reliance, gender bias in BERT | Fold in float, then round; a low-rank change like the row above | https://arxiv.org/abs/2306.03819 |
| ROME | Only to find the value vector v* (Step 2). Key, covariance and update need forward passes and linear algebra | One MLP matrix, rank-one | One subject-relation-object association | zsRE, CounterFact | v* can be searched without gradients in 64 dimensions; unclear what an association is in a code model | https://arxiv.org/abs/2202.05262 |
| MEMIT | Only to find the target vectors (gradient descent on a residual vector) | MLP matrices of several layers, closed-form least squares | Thousands of associations | GPT-J 6B, GPT-NeoX 20B | As ROME; many sequential edits degrade the model | https://arxiv.org/abs/2210.07229, https://arxiv.org/abs/2401.07453 |
| Task arithmetic | To make the vector: yes (fine-tuning). To apply it: no | All weights: add or subtract a difference of two checkpoints | Adds or removes a fine-tuned behaviour | Several models and modalities | In int8 only if both checkpoints share shifts; otherwise mix in float and re-quantise | https://arxiv.org/abs/2212.04089 |
| Head ablation and pruning | The ablation test: no. Michel's ranking score and Voita's gates: yes | Whole heads | Nothing; saves compute | 20% (WMT) and 40% (BERT) of heads removed without noticeable loss; 38 of 48 heads for -0.15 BLEU | Removes rows and cycles on LS; 40 heads can be ablated one by one with forward passes | https://arxiv.org/abs/1905.10650, https://arxiv.org/abs/1905.09418 |
| Neuron patching for code models (MINT) | Yes, to locate (Input X Gradient) | One or two neurons | Specific wrong next tokens | StarCoder2-3B, CodeLlama-7B | Locating can be done by exhaustive ablation at our size (860 hidden units) | https://arxiv.org/abs/2312.05356 |
| Knowledge editing on code models (CLMEEval study) | Mixed | Depends on the technique | Wrong or outdated code knowledge | 6 techniques on 3 models of 3B to 7B: external memory scores highest, generalisation is weak | Evidence against editing "knowledge" into weights for code | https://arxiv.org/abs/2411.06638 |
| Vocabulary-space reading (logit lens, embedding-space projection) and edits of feed-forward value vectors | No (the tuned lens trains probes) | Reading: nothing. Editing: scale or zero chosen value vectors | Suppresses or promotes groups of tokens | GPT-2 toxicity reduced by almost 50% | Tied embeddings: editing the output matrix also changes the input embedding; a logit bias avoids that | https://arxiv.org/abs/2209.02535, https://arxiv.org/abs/2203.14680, https://arxiv.org/abs/2303.08112 |
| Bias correction | No | Bias terms, from activation means | Mean shift caused by quantisation | "near-original model performance" on vision models at 8 bits | Direct, through the accumulator preload | https://arxiv.org/abs/1906.04721 |
| Cross-layer equalisation | No, and no data | Pairs of layers: rows of one scaled by s, columns of the next by 1/s | Uneven ranges between channels | MobileNet family | Exact only where the path between the two matrices is linear or ReLU-like; with s a power of two it moves shifts; see idea 6 in 02_integer_inference.md | https://arxiv.org/abs/1906.04721 |
| AdaRound | Yes, per layer (Adam, 10k iterations) | Rounding direction of each weight | Rounding error of post-training quantisation | ResNet18 / ResNet50 at 4 bits within 1% | Use the coordinate or closed-form methods below instead | https://arxiv.org/abs/2004.10568 |
| BRECQ | Yes (block reconstruction, Fisher weights from squared gradients) | Rounding per block | As AdaRound at lower bit-widths | 4-bit ResNet and MobileNetV2 "comparable with QAT" | Not gradient-free | https://arxiv.org/abs/2102.05426 |
| OBS / OBC | No (layer inputs and linear algebra) | All weights of a layer, one at a time, the rest compensated | Quantisation or pruning error | "considerably improves upon the practical performance of existing post-training methods" | Direct: fixed grid, Hessians of 64 x 64 or 172 x 172 | https://arxiv.org/abs/2208.11580 |
| GPTQ / GPTAQ | No | All weights of a layer, column by column | Quantisation error; GPTAQ also the error inherited from earlier layers | OPT-175B at 4 bits: 8.37 against 8.34 at 16 bits; OPT-125M: 31.12 against 27.65 | Direct; gain at 8 bits after QAT unknown | https://arxiv.org/abs/2210.17323, https://arxiv.org/abs/2504.02692 |
| COMQ | No | Integer codes (and the scale) of a layer by coordinate descent | Layer-wise reconstruction error | 4-bit ViT within 1% | Direct, with the scale held at a power of two | https://arxiv.org/abs/2403.07134 |
| Wanda / SparseGPT | No | Single weights set to zero | Nothing; sparsity | 50 to 60% unstructured sparsity on large language models | A zero still costs a cycle in a streamed MAC row; only structured removal helps LS | https://arxiv.org/abs/2306.11695, https://arxiv.org/abs/2301.00774 |
| Statistics renormalisation (REPAIR) | No | Scale and shift per channel | Collapsed activation variance | "60%-100% relative barrier reduction" | Maps to norm gains and preloads | https://arxiv.org/abs/2211.08403 |
| Zeroth-order and evolution strategies (MeZO, ES) | No | Any or all weights | Whatever the fitness measures, including scores that are not differentiable | MeZO comparable to backpropagation on several tasks; ES at billion-parameter scale | Forward passes are cheap for a 260k model; restrict to small groups | https://arxiv.org/abs/2305.17333, https://arxiv.org/abs/2509.24372, https://arxiv.org/abs/1703.03864 |
| The same in the quantised domain (QES, QuZO, QZO, evolutionary re-rounding) | No | Integer weights, or the quantisation scale | Task fit of a model that is already quantised | LLMs at 4 to 8 bits; CNNs | Closest published match to "edit int8 weights with forward passes only"; our scales are fixed, so perturb integer codes | https://arxiv.org/abs/2602.03120, https://arxiv.org/abs/2502.12346, https://arxiv.org/abs/2505.13430, https://arxiv.org/abs/2605.05228 |
| Interpretability agent with tools (MAIA) | No | Mostly nothing: describes neurons, finds failure modes | Sensitivity to spurious features (vision) | Descriptions "comparable to those generated by expert human experimenters" | The agent pattern carries over; the tools are for vision models | https://arxiv.org/abs/2404.14394 |
| Compiled or hand-written weights (Tracr, agent-written transformers) | No | All weights, set by construction | Nothing: builds a known algorithm | 10 algorithmic tasks at 100%; written agents are 93% zeros, the rest mostly small integers | Written weights are int8-friendly; grafting into a trained model is untested | https://arxiv.org/abs/2301.05062, https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html, https://arxiv.org/abs/2611.16981 |

### 6.2 Experiment 8 runs in the source (percent; held-out benchmarks never shown to the editor)

| Run | Forward passes | Final edit | MATH-500 | GSM8K | IFEval | MATH-500 answers at token limit | Turns / Python runs / hours |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B base | - | - | 7.9 | 13.8 | 21.0 | 93.0 | - |
| 0 passes (prompt v2) | 0 | none made | 7.9 | 13.8 | not listed | not listed | not listed |
| 5 passes (prompt v3) | 5 | none submitted (its own candidate, evaluated anyway: 8.6 / 17.1) | 7.9 | 13.8 | not listed | not listed | not listed |
| p10c | 10 | 30 rank-one edits, strength 1.5, attention and MLP outputs of layers 8-22 | 12.3 | 25.7 | 49.0 | 87.2 | 74 / 59 / 11.6 |
| p25c | 25 | head 7 of layer 15 set to zero (256 output columns) | 14.1 | 25.1 | 39.9 | 86.4 | 86 / 45 / 8.5 |
| p50c | 50 | direction removed from 64 down-projection columns in layers 15 and 19, strength 6 (131,072 values) | 48.1 | 67.1 | 62.3 | 33.1 | 158 / 80 / 20.3 |
| Gemma 3 270M base | - | - | 6.8 | not in post | not in post | 8.5 | - |
| Gemma, 5 / 10 / 25 / 50 passes | 5 / 10 / 25 / 50 | column scalings (see 2.2) | 7.0 / 7.6 / 7.7 / 7.0 | - | - | not listed | 20 / 13 / 0.9; 33 / 20 / 1.2; 73 / 44 / 2.8; 117 / 63 / 3.9 |

Other benchmarks for base / 10 / 25 / 50 passes: AMC23 0.5 / 1.1 / 1.1 / 23.4; Minerva Math 1.5 / 3.9 / 3.5 / 25.4;
OlympiadBench 1.3 / 2.4 / 3.9 / 21.8; MMLU-Pro (1,000 questions) 40.3 / 40.0 / 44.7 / 45.5; GPQA Diamond 22.7 / 29.4 /
30.1 / 36.6; AIME24 0.0 / 0.0 / 0.0 / 0.6; AIME25 0.0 / 0.0 / 0.0 / 2.3. Sources: [POST] Experiment 8 and Additional
experiment; [LOGS] exp8_math/README.md, exp8_math/eval/RESULTS_excerpt.md, exp8_math/gemma_followup/README.md.
[VERIFIED: every value of table 6.2 and of this paragraph was compared with these sources.] [fact-check: the Gemma
rows use the lenient grading rule described in 2.4 (strict: 2.8 base, 3.5 / 3.1 / 3.8 / 2.6 edited), the Qwen rows the
strict one; the unedited Gemma model scores 4.0 on GSM8K with strict grading.]

### 6.3 Which step needs what

| Method | Forward passes only | Closed form / linear algebra | Gradient steps |
|---|---|---|---|
| ROME | causal tracing, key k*, covariance C (100,000 hidden states) | rank-one update | value v* |
| MEMIT | keys, covariance C0 (100,000 samples) | update over several layers | target vectors (delta) |
| Weight orthogonalisation | difference-of-means direction | W - r r^T W | none |
| Experiment 8 edit | direction from 4 text pairs, dose-response, controls | W0 - 6 u (mask * u^T W0) | none (forbidden) |
| Bias correction, cross-layer equalisation | activation means (bias correction); none (equalisation) | yes | none |
| OBC, GPTQ, GPTAQ, COMQ | layer inputs on calibration text | yes (COMQ: coordinate steps) | none |
| AdaRound | layer inputs | - | per layer (Adam) |
| BRECQ | block inputs | - | per block, plus gradients for the Fisher weights |
| MeZO, ES, QES, QuZO | fitness of perturbed weights | - | none (the gradient is estimated from forward passes) |
| Task arithmetic | choice of the mixing factor | subtraction and addition | the fine-tuning run that makes the vector |

### 6.4 Search log for the primary task

| Place | Query or page | Result |
|---|---|---|
| arXiv API | `ti:"Weight-Space Programming"` | 11 results, all follow-ups dated 3 to 12 November 2026; none is the source |
| arXiv API | `all:"Sparks of Weight-Space Programming"` | 0 results |
| arXiv API | `au:Takehi AND au:Khattab`; `au:"Rikiya Takehi"` | 0 results; 9 results, none is this work |
| Web search | "Sparks of Weight-Space Programming" | the author's blog page, Simon Willison's post, a Substack and a Medium explainer, a Turing Post issue, an X post by the lead author |
| Author site | https://rikiyat.github.io/ and /blog/weight-space-programming/ | the post itself; read in full, including the reference list and "Response to Comments" |
| GitHub | RikiyaT/lm_understand-logs | README, LICENSE, exp8_math (README, opening prompt, final answers, final report, weight manifest, pass list, experiment history, tool calls, results excerpt, merged scores, run records), exp8_math/gemma_followup/README.md, exp4_fix_car/README.md, harness/ listing |
| Hugging Face | RikiyaT/Qwen3.5-0.8B-weight-space-programming; Qwen/Qwen3.5-0.8B | model card with the outside re-run; base model card line 856 (loop warning) and licence |
| Critiques and follow-ups | Goedecke's post; arXiv 2611.09382 (HTML); abstracts of 2611.04268, 2611.05341, 2611.07319, 2611.08070, 2611.08647, 2611.09271, 2611.11537, 2611.11604, 2611.12066, 2611.13758, 2611.16981; Singh's report | read |
| Not searched | OpenReview, MIT CSAIL news pages, Hacker News, Reddit | not needed once the primary source was open; the post links no OpenReview or arXiv version |
| arXiv API (fact-check) | the four arXiv queries above, repeated | same counts: 11, 0, 0 and 9 [VERIFIED] |
| arXiv API (fact-check) | `abs:"weight-space programming"` | 15 results: the 11 above and 2611.14210 (Tree-Ising), 2611.12954, 2611.09559, 2611.09376; none is the source |
| arXiv API (fact-check) | `abs:"knowledge editing" AND abs:quantized`; `abs:"steering vector" AND abs:quantiz*`; `abs:abliteration AND abs:quantiz*` | 4, 6 and 1 results; relevant: 2506.13772 and 2609.06473 (section 2.8) |
| Web search (fact-check) | "Sparks of Weight-Space Programming" Takehi Khattab; the same words with arXiv OR OpenReview | secondary coverage only (DataCamp, The Register, iThome and others); no arXiv or OpenReview version |

To repeat the synthetic check of section 2.9:

```
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
.venv/bin/python research/07_rank_one_int8_check.py
```

## 7. Sources

Licences are given for code, models and data and were read from the repository (GitHub API licence field or the
LICENSE file) or the model card on 2026-11-13. "none detected" means the GitHub API reports no licence for the
repository. Papers are cited, not redistributed. [VERIFIED: the fact-checker read the GitHub licence field of all 21
repositories named in these tables on 2026-11-13, plus LICENSE of the log repository and both model-card headers;
every entry matches.]

Primary source and its artefacts

| Title | Authors | Year | URL | Licence (code, model, data) |
|---|---|---|---|---|
| Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights (blog post) | Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab | 2026 | https://rikiyat.github.io/blog/weight-space-programming/ | none stated on the page |
| lm_understand-logs (logs, networks, weight diffs, evaluation scripts of the post) | same authors | 2026 | https://github.com/RikiyaT/lm_understand-logs | logs, data, networks CC BY 4.0; scripts MIT; MinAtar source quoted in Experiment 5 prompts GPL-3.0 |
| Qwen3.5-0.8B-weight-space-programming (edited model, with an outside re-run) | rebuilt by Ahsen Khaliq from the logs | 2026 | https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming | Apache-2.0 |
| Qwen3.5-0.8B (base model card) | Qwen | 2026 | https://huggingface.co/Qwen/Qwen3.5-0.8B | Apache-2.0 |

Commentary and follow-ups

| Title | Authors | Year | URL | Licence (code, model, data) |
|---|---|---|---|---|
| LLMs are not superhuman at reading and writing neural network weights | Sean Goedecke | 2026 | https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/ | - |
| Qwen 3.8 27B on my laptop can interpret model weights, too | Simon Willison | 2026 | https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/ | - |
| Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations | Junru Hu, Cheng Tang, Wei Li, Hongze Zhu, Chun-Mei Feng, Qi Zhang, Yirong Chen, Ming Hu | 2026 | https://arxiv.org/abs/2611.09382 | code https://github.com/WSP-Scope/WSP-Scope-Project: none detected |
| IWIBench: Evaluating Gradient-Free Model Driven Intra-weight Injection | Chris von Csefalvay, Pranav Doma, Achyuthan Unni Krishnan, Tamas Foldi | 2026 | https://arxiv.org/abs/2611.08647 | - |
| Weight-space programming for auditing clinical decision networks | Humza Nusrat, Omar Nusrat | 2026 | https://arxiv.org/abs/2611.05341 | - |
| Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? | Nazmus Sakib Ahmed | 2026 | https://arxiv.org/abs/2611.08070 | - |
| Mixed Signals: Weight-Space Programming for EEG and ECG | Dara Varam, Taraneh Aminosharieh Najafi, David Atienza | 2026 | https://arxiv.org/abs/2611.13758 | - |
| Compiling Known Rules into Compact Neural Agents: Weight-Space Programming on Unseen Games | Jens Lehmann | 2026 | https://arxiv.org/abs/2611.16981 | - |
| Weight-Space Programming of Variational Quantum Circuits | Debarshi Kundu, Swaroop Ghosh, Vasant G Honavar | 2026 | https://arxiv.org/abs/2611.11537 | - |
| Weight-Space Programming from scratch on simple tasks (report) | Chandan Singh | 2026 | https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html | code https://github.com/csinva/weight-space-programming-autoresearch: MIT |
| Tree-Ising: Repairing Language Models with Bit-Level Ising Edits [fact-check: added] | Hadi Salloum, Ammar Ali | 2026 | https://arxiv.org/abs/2611.14210 | code https://github.com/TheQuantumGOAT/Tree-Ising: none detected |

Established literature

| Title | Authors | Year | URL | Licence (code, model, data) |
|---|---|---|---|---|
| Locating and Editing Factual Associations in GPT (ROME) | Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov | 2022 | https://arxiv.org/abs/2202.05262 | code https://github.com/kmeng01/rome: MIT |
| Mass-Editing Memory in a Transformer (MEMIT) | Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau | 2022 | https://arxiv.org/abs/2210.07229 | code https://github.com/kmeng01/memit: MIT |
| Model Editing at Scale leads to Gradual and Catastrophic Forgetting | Akshat Gupta, Anurag Rao, Gopala Anumanchipalli | 2024 | https://arxiv.org/abs/2401.07453 | - |
| Does Localization Inform Editing? | Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun | 2023 | https://arxiv.org/abs/2301.04213 | - |
| Model Editing for LLMs4Code: How Far are We? | Xiaopeng Li, Shangwen Wang, Shasha Li, Jun Ma, Jie Yu, Xiaodong Liu, Jing Wang, Bin Ji, Weimin Zhang | 2024 | https://arxiv.org/abs/2411.06638 | - |
| Neuron Patching: Semantic-based Neuron-level Language Model Repair for Code Generation (MINT) | Jian Gu, Aldeida Aleti, Chunyang Chen, Hongyu Zhang | 2023 | https://arxiv.org/abs/2312.05356 | - |
| Editing Large Language Models: Problems, Methods, and Opportunities (survey) | Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang | 2023 | https://arxiv.org/abs/2305.13172 | framework https://github.com/zjunlp/EasyEdit: MIT |
| A Comprehensive Study of Knowledge Editing for Large Language Models (survey) | Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang and 16 others | 2024 | https://arxiv.org/abs/2401.01286 | - |
| Refusal in Language Models Is Mediated by a Single Direction | Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda | 2024 | https://arxiv.org/abs/2406.11717 | code https://github.com/andyrdt/refusal_direction: Apache-2.0 |
| Steering Llama 2 via Contrastive Activation Addition | Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner | 2023 | https://arxiv.org/abs/2312.06681 | code https://github.com/nrimsky/CAA: MIT |
| Steering Language Models With Activation Engineering | Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid | 2023 | https://arxiv.org/abs/2308.10248 | - |
| Heretic (tool for automated direction removal) | p-e-w and contributors | 2026 (repository state) | https://github.com/p-e-w/heretic | AGPL-3.0 |
| LEACE: Perfect linear concept erasure in closed form | Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman | 2023 | https://arxiv.org/abs/2306.03819 | code https://github.com/EleutherAI/concept-erasure: MIT |
| Editing Models with Task Arithmetic | Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi | 2022 | https://arxiv.org/abs/2212.04089 | code https://github.com/mlfoundations/task_vectors: none detected |
| Are Sixteen Heads Really Better than One? | Paul Michel, Omer Levy, Graham Neubig | 2019 | https://arxiv.org/abs/1905.10650 | - |
| Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned | Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov | 2019 | https://arxiv.org/abs/1905.09418 | - |
| A Simple and Effective Pruning Approach for Large Language Models (Wanda) | Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter | 2023 | https://arxiv.org/abs/2306.11695 | code https://github.com/locuslab/wanda: MIT |
| SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot | Elias Frantar, Dan Alistarh | 2023 | https://arxiv.org/abs/2301.00774 | code https://github.com/IST-DASLab/sparsegpt: Apache-2.0 |
| Analyzing Transformers in Embedding Space | Guy Dar, Mor Geva, Ankit Gupta, Jonathan Berant | 2022 | https://arxiv.org/abs/2209.02535 | - |
| Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space | Mor Geva, Avi Caciularu, Kevin Ro Wang, Yoav Goldberg | 2022 | https://arxiv.org/abs/2203.14680 | - |
| Eliciting Latent Predictions from Transformers with the Tuned Lens | Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, Jacob Steinhardt | 2023 | https://arxiv.org/abs/2303.08112 | code https://github.com/AlignmentResearch/tuned-lens: MIT |
| Data-Free Quantization Through Weight Equalization and Bias Correction | Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling | 2019 | https://arxiv.org/abs/1906.04721 | implementation in https://github.com/qualcomm/aimet: licence file present, not classified by the GitHub API |
| Up or Down? Adaptive Rounding for Post-Training Quantization (AdaRound) | Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen Blankevoort | 2020 | https://arxiv.org/abs/2004.10568 | - |
| BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction | Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, Shi Gu | 2021 | https://arxiv.org/abs/2102.05426 | code https://github.com/yhhhli/BRECQ: MIT |
| Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning (OBC) | Elias Frantar, Sidak Pal Singh, Dan Alistarh | 2022 | https://arxiv.org/abs/2208.11580 | code https://github.com/IST-DASLab/OBC: none detected |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh | 2022 | https://arxiv.org/abs/2210.17323 | code https://github.com/IST-DASLab/gptq: Apache-2.0 |
| GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration | Yuhang Li, Ruokai Yin, Donghyun Lee, Shiting Xiao, Priyadarshini Panda | 2025 | https://arxiv.org/abs/2504.02692 | - |
| COMQ: A Backpropagation-Free Algorithm for Post-Training Quantization | Aozhong Zhang, Zi Yang, Naigang Wang, Yingyong Qi, Jack Xin, Xin Li, Penghang Yin | 2024 | https://arxiv.org/abs/2403.07134 | - |
| REPAIR: REnormalizing Permuted Activations for Interpolation Repair | Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari, Behnam Neyshabur | 2022 | https://arxiv.org/abs/2211.08403 | - |
| Fine-Tuning Language Models with Just Forward Passes (MeZO) | Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora | 2023 | https://arxiv.org/abs/2305.17333 | code https://github.com/princeton-nlp/MeZO: MIT |
| Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning | Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu, Roberto Dailey, Elliot Meyerson, Babak Hodjat, Risto Miikkulainen | 2025 | https://arxiv.org/abs/2509.24372 | - |
| Evolution Strategies as a Scalable Alternative to Reinforcement Learning | Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever | 2017 | https://arxiv.org/abs/1703.03864 | - |
| Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost (QES) | Yinggan Xu, Kajetan Schweighofer, Risto Miikkulainen, Xin Qiu | 2026 | https://arxiv.org/abs/2602.03120 | code https://github.com/dibbla/Quantized-Evolution-Strategies: none detected |
| QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models | Jiajun Zhou, Yifan Yang, Kai Zhen, Ziyue Liu, Yequan Zhao, Ershad Banijamali, Athanasios Mouchtaris, Ngai Wong, Zheng Zhang [fact-check: ninth author was missing] | 2025 | https://arxiv.org/abs/2502.12346 | - |
| Fine-tuning Quantized Neural Networks with Zeroth-order Optimization (QZO) | Sifeng Shang, Jiayi Zhou, Chenyu Lin, Minxian Li, Kaiyang Zhou | 2025 | https://arxiv.org/abs/2505.13430 | - |
| Evolutionary fine tuning of quantized convolution-based deep learning models | Marcin Pietron | 2026 | https://arxiv.org/abs/2605.05228 | - |
| Test-Time Model Adaptation with Only Forward Passes (FOA) | Shuaicheng Niu, Chunyan Miao, Guohao Chen, Pengcheng Wu, Peilin Zhao | 2024 | https://arxiv.org/abs/2404.01650 | - |
| A Multimodal Automated Interpretability Agent (MAIA) | Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, Antonio Torralba | 2024 | https://arxiv.org/abs/2404.14394 | code https://github.com/multimodal-interpretability/maia: none detected |
| FIND: A Function Description Benchmark for Evaluating Interpretability Methods | Sarah Schwettmann and 7 others | 2023 | https://arxiv.org/abs/2309.03886 | - |
| Weight Oracles: Reading Neural Network Weights with Language Models | Krishna Kabra, Constantin Venhoff, Christian Schroeder de Witt | 2026 | https://arxiv.org/abs/2610.07334 | - |
| Tracr: Compiled Transformers as a Laboratory for Interpretability | David Lindner, Janos Kramar, Sebastian Farquhar, Matthew Rahtz, Thomas McGrath, Vladimir Mikulik | 2023 | https://arxiv.org/abs/2301.05062 | code https://github.com/google-deepmind/tracr: Apache-2.0 |
| MobiEdit: Resource-efficient Knowledge Editing for Personalized On-device LLMs [fact-check: added] | Zhenyan Lu, Daliang Xu, Dongqi Cai, Zexi Li, Wei Liu, Fangming Liu, Shangguang Wang, Mengwei Xu | 2025 | https://arxiv.org/abs/2506.13772 | - |
| Steering Under Compression: Dose-Response, Capability Cost, and Failure Asymmetry in Quantized LLMs [fact-check: added] | Saurav Bhandari, Benjamin Wade | 2026 | https://arxiv.org/abs/2609.06473 | - |
| Progress measures for grokking via mechanistic interpretability [fact-check: added] | Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt | 2023 | https://arxiv.org/abs/2301.05217 | - |

Named but not opened
- The original Optimal Brain Surgeon paper [UNVERIFIED: not opened; the OBC abstract refers to it as "the classical
  Optimal Brain Surgeon (OBS) framework of [LeCun, Denker, and Solla, 1990]"].
- The original "logit lens" write-up [UNVERIFIED: not opened; known here only through the tuned-lens abstract, which
  calls it "the earlier 'logit lens' technique"].
- Matthew Honnibal's comment on the post [UNVERIFIED: not opened; quoted through Willison].

Project files referred to: `research/CONTEXT.md` (cost model, baseline shape), `research/02_integer_inference.md`
(accumulator preload, overflow rule, equalisation), `research/03_qat_small_models.md` (rounding and QAT),
`research/05_speed_techniques.md` (head pruning), `research/07_rank_one_int8_check.py` (synthetic check of section 2.9).

What was verified and what was not: every quoted sentence and number about the source was read in [POST], [LOGS] or
[CARD] on 2026-11-13. For the established literature the abstracts were read through the arXiv API, and the method
sections of ROME, MEMIT, Arditi et al., GPTQ, AdaRound, BRECQ, Nagel et al. 2019, Michel et al. and MINT were read on
ar5iv for the specific statements quoted. The two surveys, the follow-up preprints other than Hu et al., and all
remaining papers were read as abstracts only. No experiment was run on an LS-LLM model; section 2.9 is a synthetic
calculation.

Fact-check of 2026-11-13 (a second agent): 25 claims were checked against their sources; they are listed in the
section "Verification log". 17 were confirmed, 7 were corrected, 1 was unsupported as written and was narrowed. No
statement about the source was found to be invented. Sources added by the fact-check (Tree-Ising, MobiEdit, steering
under quantisation, Nanda et al.) were read as abstracts; Tree-Ising and Nanda et al. also in the HTML version. Not
checked: the two surveys beyond their abstracts, Honnibal's own text, a bit-for-bit rebuild of the edit with
`apply_edit.py`, and the statements that point into research files 02, 03 and 05 beyond the overflow and preload rules.

## Verification log

Fact-check of 2026-11-13 by a second agent. One line per checked claim: claim | verdict | source opened. "[LOGS]"
stands for https://github.com/RikiyaT/lm_understand-logs (branch main).

1. The source exists at the stated address with the stated title, four authors, "MIT CSAIL", "Published November 2, 2026", `note = {Blog post}`, and "Response to Comments" added on 12 November | confirmed | https://rikiyat.github.io/blog/weight-space-programming/
2. The source is not on arXiv: `all:"Sparks of Weight-Space Programming"` 0 results, `au:Takehi AND au:Khattab` 0, `ti:"Weight-Space Programming"` 11 follow-ups, `au:"Rikiya Takehi"` 9 unrelated papers; web search shows secondary coverage only | confirmed | https://export.arxiv.org/api/query?search_query=all:%22Sparks+of+Weight-Space+Programming%22
3. The editor is OpenAI's `gpt-6-astra` through the Codex CLI, reasoning effort `max` in Experiment 8, 1,125 tool-free turns | confirmed | https://github.com/RikiyaT/lm_understand-logs/blob/main/README.md
4. Headline numbers: 131,072 of 0.8 billion weights (0.016%); MATH-500 7.9 -> 48.1, GSM8K 13.8 -> 67.1, IFEval 21.0 -> 62.3, answers at the token limit 93.0 -> 33.1; the 10- and 25-pass rows and the other benchmarks of table 6.2 | confirmed | https://rikiyat.github.io/blog/weight-space-programming/ and https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md
5. Gradients and training forbidden; kernel, pass budget, practice data and refusal rules as quoted in 2.2 | confirmed | https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/turns/t001/prompt.txt
6. Edit form "W = W0 - 6*outer(u, mask64*(u@W0))" on the down-projections of layers 15 and 19 (1,024 x 3,584), relative change 10.6% and 8.1%, 130,188 of 131,072 values differ in bfloat16 | confirmed | https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/final_report.md
7. Selection steps 1 to 10 of section 2.2 (four text pairs, last 36 tokens, growth 0.472 and 0.530, paired tests, dose-response, random control +1.56 points with SE 3.52, 64 columns, strength 6) | corrected: "largest growth among layers 0 to 21; layers 22 and 23 not read" -> "among layers 0 to 22 (layer 22 = 0.309); layer 23 not printed"; the rest confirmed | https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/tool_calls.jsonl
8. Run size: 158 turns, 80 Python runs, 4 conversations, 20.3 hours (4.7 waiting); 69 forward requests, 19 failed, 34 generation and 16 teacher-forced passes | corrected: "It needed" -> "The one run that worked used"; the numbers are confirmed; added: session-writer failure in turns 15 to 40, 3 counted passes were repeats | https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md and https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/readable/passes.md
9. Editor token usage 5,437,759 input and 149,017 output | corrected: total -> lower bound; about 6.3 million input and 0.16 million output once the 25 calls missing from the counter of conversation 1 are added (our estimate); the "89.7 million tokens" of arXiv 2611.14210 is the turn-by-turn sum of cumulative counters | https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math/runs/p50c/turns
10. Practice confirmation: 1,152 problems, 10.6% -> 60.4%, +49.8 points with SE 1.5, 579 rescues and 5 regressions, 576 rescues from truncated outputs; side effects 6 -> 187, 1 -> 58, 0 -> 51 | corrected: the quote "unresolved alternative degeneration" moved from the 1,152-problem counts to the interim audit of 256 problems where it was written; the numbers are confirmed | https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/final_answer.md
11. Budget and prompt ablation: no edit at 0 and 5 passes, no edit at 10 passes with prompt version 4, IFEval 49.0 and 39.9 | confirmed; added the unsubmitted candidates (8.6 / 17.1 and 10.1 / 16.1) | https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/eval/RESULTS_excerpt.md
12. Gemma 3 270M: 6.8 -> 7.0 / 7.6 / 7.7 / 7.0, 8.5% of answers at the token limit, edit forms, run sizes | corrected: "the same procedure" -> same harness with prompt version 5, lenient grading (strict: 2.8 -> 3.5 / 3.1 / 3.8 / 2.6) and other decoding; added four runs that are not in the post (6.7 / 6.7 / 6.1 / 6.2); the numbers are confirmed | https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/gemma_followup/README.md
13. Model cards: outside re-run 51.0% on MATH-500 with 31.6% at the token limit, base 5.0% on 25 problems, "43.4M generated tokens, 51 min on an H200"; the base card warns about thinking loops at line 856 | confirmed | https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming and https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/README.md#L856
14. Licences of the source artefacts: logs, data and networks CC BY 4.0, scripts MIT, MinAtar source GPL-3.0, edited model apache-2.0, base model apache-2.0, no licence stated on the post page | confirmed | https://github.com/RikiyaT/lm_understand-logs/blob/main/LICENSE
15. Other experiments: Experiment 4 (171,664 weights, 51 edit statements, 6,073 changed = 3.54%, 0/256 -> 1,024/1,024, 10,134 output tokens), Experiment 3 (7.16 million weights, copying heads 2.1, 3.1, 3.5), Experiment 6 (1,480,274 weights), Experiment 7 (four attempts from weights failed, 3 of 3 with attention patterns) | confirmed | https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md and https://rikiyat.github.io/blog/weight-space-programming/
16. Hu et al.: 16 trajectories with the same editor, held-out gains up to 29.45 / 4.42 / 36.72 / 17.11 points, 1,023 BF16 values in one column, six of 16 without edit, edit forms | confirmed (abstract and HTML) | https://arxiv.org/abs/2611.09382
17. Six further follow-up preprints and Singh's report: numbers, titles, authors and quoted sentences of 2.7 | confirmed (abstracts; report page) | https://arxiv.org/abs/2611.08647 https://arxiv.org/abs/2611.05341 https://arxiv.org/abs/2611.08070 https://arxiv.org/abs/2611.13758 https://arxiv.org/abs/2611.16981 https://arxiv.org/abs/2611.11537 https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html
18. Commentary quotes: Goedecke ("competent human researcher", "Heretic abliteration playbook"); Willison on Honnibal | confirmed; Honnibal's own text stays [UNVERIFIED] | https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/ and https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/
19. GPTQ: "128 random 2048 token segments", four GPU hours, WikiText2 perplexity 8.34 / 10.54 / 8.37 | corrected: the quoted numbers were for OPT-175B only -> OPT-125M row added (27.65 / 37.28 / 31.12); the values are confirmed | https://arxiv.org/abs/2210.17323
20. Arditi et al.: 13 chat models from 1.8B to 72B, Equation 5 W_out' <- W_out - r r^T W_out, "does not require gradient-based optimization" | confirmed (HTML) | https://arxiv.org/abs/2406.11717
21. Head pruning: Michel et al. 20% (WMT) and 40% (BERT) of heads, +17.5% speed at batch 16 and 64, gradient-based score; Voita et al. 38 of 48 heads for 0.15 BLEU | confirmed | https://arxiv.org/abs/1905.10650 and https://arxiv.org/abs/1905.09418
22. Established literature of 2.8 in bulk: 40 arXiv numbers resolve to the stated titles; 61 quoted strings (54 verbatim in the abstracts, 7 in the full texts); method statements of ROME, MEMIT, AdaRound, BRECQ, Nagel et al. and MINT | corrected: QuZO author list 8 -> 9 names (Zheng Zhang added); "code-model studies use 3B to 7B models" -> "2B to 7B"; the rest confirmed | https://export.arxiv.org/api/query?id_list=2202.05262,2210.07229,2312.05356,2502.12346
23. Licences of tools and code: Heretic AGPL-3.0 and the 20 other repositories named in section 7 | confirmed (GitHub licence field of all 21) | https://github.com/p-e-w/heretic
24. Own synthetic check of 2.9 and the arithmetic taken from CONTEXT.md (3,512 rows, 10.5 KB, 252 to 272 cycles and 192 bytes per hidden unit) | confirmed (script run again, arithmetic recomputed) | research/07_rank_one_int8_check.py
25. "Not found" statements of section 4 | unsupported as written for two of them: "steering vectors or any model-editing method applied to int8 weights: not found" (arXiv 2609.06473 and 2506.13772 exist) and "ablation at 260k to 400k parameters: not found" (arXiv 2301.05217 at about 0.23 million parameters); both narrowed; the statement on LLM-agent edits below 270M parameters holds for the 15 arXiv abstracts that mention the term | https://arxiv.org/abs/2609.06473 https://arxiv.org/abs/2506.13772 https://arxiv.org/abs/2301.05217
