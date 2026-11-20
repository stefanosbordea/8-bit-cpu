# Session s1: remove a "repetition" direction

Model: `runs/qat/A2q` (2 layers, D = 160, 4 query heads on 1 key/value head, F = 219, V = 512, relu, rms norm).
All scripts are in `scripts/`, run from the repository root with `PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/<name>`.
Every number below is on the 200 practice tasks (greedy, syntax filter on) and on 32 windows of training text (loss).

## Run 1: baseline (`01_baseline.py`)

pass@1 0.005 (1 task: 002), tests_passed 0.0950, syntax_ok 0.93, runs_ok 0.425, practice_loss 1.6463, repeating 0.04 (8 answers).
Errors: TypeError 53, NameError 38. 190 of 200 answers are one `return` line.

What the answers look like (read from `eval_01.jsonl`):
- Repetition is rare: 8 of 200 answers (`1024 * 1024 * ...`, `word + word + word`, `b if b < b else b ...`). 14 answers fail the syntax check, 5 of them are loops cut at 64 tokens.
- The common failures are not loops: wrong argument (`b < b and b < b`), a name that is almost an argument
  (`day` for `days`, `item` for `items`, `ax` for `a`: 12 of the 38 NameErrors), a word of the docstring used as a name
  (`then`, `whose`, `integer`), an int treated as a list (TypeError).
- So even a perfect cure of repetition can change at most 8 answers.

## Run 2: trace (`02_trace_loops.py`, analysis `03_analyse_trace.py`, `04_matched_direction.py`)

47 baseline answers (the 8 repeating ones and every fifth other), 3749 tokens. 73 looping positions in 6 answers, 1129 other answer positions.

Plain recipe (mean at looping positions minus mean at other positions), script 03:
- separation d' 2.1 to 2.8 at every site, but the direction read through E says "digits and `* `": it is the CONTENT of the loops
  (half of the looping positions are `1024 * 1024` of one answer), not repetition.
- Layer 1 feed-forward writes 51.9 along it at looping positions and 12.0 elsewhere.

Matched direction (script 04): a looping position minus the same place in the FIRST turn of its own cycle, every answer counted once.
- cosine with the plain direction only 0.34 to 0.55.
- Who writes it: the ATTENTION, not the feed-forward layers. Layer 0 attention: looping 9.9, first turn 1.4, other -0.9.
  Layer 1 attention: 12.2 / -14.0 / -5.8. Layer 1 feed-forward: 58 / 21.5 / 38.5 (not specific: other positions are in between).
- This is the "seen before" signal of in-context copying, the same mechanism the model needs to copy argument names.

Other measurements of the same trace:
- logit ties: 2.5 % of answer positions have a tie at the top 7-bit code, 1.3 % change the greedy token against the accumulator. No saturation (largest code 39..121). Not worth an edit.
- attention: 2 to 4.5 keys count per head (median); no head is cut off at code 0 or 127.

## Runs 3 to 9: removing the direction (`05_remove_direction.py`), with controls (`06_random_directions.py`)

| run | edit | pass@1 | tests_passed | syntax_ok | runs_ok | loss | repeating | NameError |
|---|---|---|---|---|---|---|---|---|
| 1 | baseline | 0.005 | 0.0950 | 0.930 | 0.425 | 1.6463 | 0.040 | 38 |
| 3 | W2_1, matched u(l1.xf), alpha 1 | 0.005 | 0.0911 | 0.805 | 0.365 | 1.9001 | 0.040 | 31 |
| 4 | Wo_0, matched u(l0.xa), alpha 1 | 0.020 | 0.1090 | 0.895 | 0.390 | 1.6690 | 0.035 | 43 |
| 5 | Wo_1, matched u(l1.xa), alpha 1 | 0.005 | 0.0992 | 0.935 | 0.445 | 1.6643 | 0.025 | 43 |
| 6 | Wo_0, matched u(l0.xa), alpha 0.5 | 0.015 | 0.1080 | 0.935 | 0.460 | 1.6501 | 0.020 | 49 |
| 7 | Wo_0, matched u(l0.xa), alpha 2 | 0.010 | 0.0989 | 0.860 | 0.405 | 1.7420 | 0.040 | 28 |
| 8 | CONTROL Wo_0, random direction 1, alpha 1 | 0.010 | 0.1045 | 0.925 | 0.420 | 1.6595 | 0.025 | 41 |
| 9 | CONTROL Wo_0, random direction 2, alpha 1 | 0.010 | 0.0755 | 0.910 | 0.355 | 1.6580 | 0.025 | 51 |

- W2_1 (the recipe's target): alpha 1 already costs 0.25 nats and 12 points of syntax_ok, repeating unchanged. The direction is not
  separable from normal content in what the feed-forward layer writes. Larger alpha (the post used 6) is out of the question. FAILED.
- Greedy decoding of this model is chaotic: an edit of 1749 weights by 1 to 3 steps changes 154 of 200 answers. Any perturbation
  moves pass@1 between 1 and 4 tasks and tests_passed between 0.075 and 0.11. Task 060 passes after ANY change of Wo_0, also the random ones.
- The two random directions give pass@1 0.010 and "repeating" 0.025: as good at "curing repetition" as the repetition direction.
  "repeating" 0.02 to 0.04 (4 to 8 answers) is noise.
- Wo_0 with the matched direction: 3, 4 and 2 passes at alpha 0.5, 1, 2 against 2 and 2 for the controls. Above the controls, but not by more than the noise.

## Runs 10, 11: strengthen the copying head (`07_copy_heads.py`, `08_scale_head.py`)

Weights only: layer 1 head 0 is the one copying head (token -> Wv -> Wo -> E: half of all tokens are among their own top 5 outputs, z 2.3).
Layer 1 heads 1 and 3 and layer 0 head 1 have a negative diagonal (they lower the token they look at).
Hypothesis: NameErrors come from too weak copying. Edit: Wo_1[:, 0:40] times 1.25 / 1.5.
Result: pass@1 0.005 / 0.005, NameError 43 / 37 (baseline 38), loss 1.6705 / 1.7304. FAILED, taken back.

## Runs 12 to 15: attention sharpness (`09_exp_kappa.py`)

EXP tables rebuilt with kappa 3.2 (sharper) and 5 (softer) instead of 4.0, one layer at a time.

| run | edit | pass@1 | tests_passed | syntax_ok | runs_ok | loss | repeating | NameError |
|---|---|---|---|---|---|---|---|---|
| 12 | layer 1 kappa 3.2 | 0.005 | 0.0872 | 0.920 | 0.385 | 1.6769 | 0.035 | 44 |
| 13 | layer 1 kappa 5 | 0.010 | 0.1053 | 0.920 | 0.415 | 1.6649 | 0.025 | 45 |
| 14 | layer 0 kappa 3.2 | 0.010 | 0.0818 | 0.880 | 0.355 | 1.6690 | 0.030 | 47 |
| 15 | layer 0 kappa 5 | 0.015 | 0.1086 | 0.930 | 0.470 | 1.6770 | 0.025 | 28 |

Loss goes up by 0.02 to 0.03 in both directions on both layers: the exported table (kappa 4.0) is where the model was trained. FAILED, taken back.
Run 15 has 3 passes with a loss that is clearly worse: more evidence that 3 passes is inside the noise.

## Runs 16 to 19: strengthen the heads that push AGAINST copying (`08_scale_head.py`)

Hypothesis from script 07: layer 1 heads 1 and 3 lower the logit of the token they look at; stronger, they should stop `b < b and b < b` and the loops.

| run | edit | pass@1 | tests_passed | syntax_ok | runs_ok | loss | repeating |
|---|---|---|---|---|---|---|---|
| 19 | L1H1, L1H3 times 1.12 | 0.005 | 0.0877 | 0.930 | 0.390 | 1.6528 | 0.010 |
| 16 | L1H1, L1H3 times 1.25 | 0.010 | 0.0918 | 0.900 | 0.400 | 1.6701 | 0.000 |
| 17 | L1H1, L1H3 times 1.5 | 0.015 | 0.0968 | 0.845 | 0.365 | 1.7396 | 0.020 |
| 18 | L1H1, L1H3, L0H1 times 1.25 | 0.015 | 0.0873 | 0.865 | 0.400 | 1.6852 | 0.040 |

- This is the one edit that moves "repeating" out of its noise band (0.02 to 0.04 in 17 other runs): 2 answers at 1.12, none at 1.25.
  At 1.5 the damage to the model makes new loops. So repetition here is governed by these two heads, not by a direction of the stream.
- But the former loops become short wrong answers (`word + word + word`, `len([s]) // 2`), not right ones: pass@1 and tests_passed stay in the noise,
  the loss is worse. Not kept (the ranking is pass@1, tests_passed, loss).

## Runs 20 to 26: a finer yardstick (`10_trace_references.py`)

pass@1 cannot tell edits apart (it moved between 1 and 4 tasks over 17 edits, mean 2.0, with no relation to the loss). So: reference solutions of
60 practice tasks (every third), 1033 body tokens, the same tokens for every model; nll and top-1 of the 7-bit logits, paired, standard error over tasks.

| run | model | nll | change of nll | change of top-1 |
|---|---|---|---|---|
| 20 | baseline | 1.5193 | | (top-1 0.6786) |
| 21 | Wo_0 matched direction, alpha 1 (the kept edit) | 1.5577 | +0.038 +- 0.028 | +0.005 +- 0.008 |
| 22 | Wo_0 matched direction, alpha 0.5 | 1.5125 | -0.007 +- 0.019 | +0.005 +- 0.008 |
| 23 | CONTROL Wo_0 random direction 1, alpha 1 | 1.5303 | +0.011 +- 0.021 | +0.016 +- 0.007 |
| 24 | L1H1, L1H3 times 1.25 | 1.5785 | +0.059 +- 0.012 | +0.004 +- 0.006 |
| 25 | Wo_1 matched direction, alpha 1 | 1.5148 | -0.004 +- 0.011 | +0.001 +- 0.005 |
| 26 | layer 0 kappa 5 | 1.5404 | +0.021 +- 0.026 | +0.012 +- 0.008 |

No edit is better than the baseline on this yardstick by more than its standard error; the random control is as good as anything.
The 4 passes of the kept edit are therefore NOT evidence of a better model: token by token it is equal to slightly worse.

## Runs 27 to 29: the literal recipe and both layers

| run | edit | pass@1 | tests_passed | syntax_ok | runs_ok | loss | repeating |
|---|---|---|---|---|---|---|---|
| 27 | W2_1, PLAIN direction u(l1.xf), alpha 0.5 (the recipe as written) | 0.005 | 0.0922 | 0.915 | 0.405 | 1.6677 | 0.030 |
| 28 | Wo_0 alpha 1 + Wo_1 alpha 1, matched | 0.015 | 0.0946 | 0.885 | 0.355 | 1.6873 | 0.025 |
| 29 | Wo_0 alpha 0.5 + Wo_1 alpha 1, matched | 0.015 | 0.0943 | 0.895 | 0.435 | 1.6709 | 0.025 |

All inside the noise. Taken back.

## Run 30: the kept model

Ranking rule of the session (pass@1, then tests_passed, then loss) over all 21 evaluated models (22 evaluate runs, run 30 repeats run 4): run 4 is first. Applied again with `--keep`:
`05_remove_direction.py --tensors Wo_0 --direction 04:l0_xa --alpha 1 --keep`, one edit, 5476 of 25600 elements of Wo_0, each by at most 5 steps.

| | pass@1 | tests_passed | syntax_ok | runs_ok | loss | repeating |
|---|---|---|---|---|---|---|
| baseline (run 1) | 0.005 | 0.0950 | 0.930 | 0.425 | 1.6463 | 0.040 |
| kept (run 30) | 0.020 | 0.1090 | 0.895 | 0.390 | 1.6690 | 0.035 |

Passing tasks: 002, 020, 060, 104 (baseline: 002).

## Verdict

- The strategy is a dead end on this model. Repetition is 4 % of the answers; the "repetition direction" taken as the recipe says is the content of the loops
  (digits); the matched direction is the copying signal of the attention; removing either does not lower "repeating" more than removing a random direction does.
- What does remove repetition is scaling two attention heads (L1H1, L1H3) by 1.12 to 1.25, but that does not make any answer right.
- The kept model is the best by the session's ranking, and it is honest to say what that is worth: pass@1 of this model moves by +-2 tasks under ANY small edit
  (17 edits: 1 to 4 passes; two random controls: 2 and 2). 4 against 1 passes is within that. The kept edit costs 0.023 nats of training loss and 3.5 points
  of syntax_ok, and on the token-level yardstick it is not better than the baseline. Expect the held-out score to be about the baseline's, not four times it.
- If the held-out check is to decide between candidates, the alpha 0.5 version (run 6: pass@1 0.015, tests_passed 0.1080, syntax_ok 0.935, runs_ok 0.460,
  loss 1.6501) is the one that harms nothing measurable.
- 30 of 50 runs used. The other 20 were not spent on trying further edits until one happens to pass 5 tasks: that would be choosing noise.

## Rules

1. No gradient, optimiser or fitting. Every edit is a formula on the weights with numbers chosen by hand (alpha, a factor, kappa); the directions are differences of means of traced activations, or random.
2. The model ran only through `Session.evaluate` (22 runs) and `Session.trace` (8 runs: 2 and 20 to 26). The traces of runs 20 to 26 fed reference solutions of the practice file (practice data; not one of the three kinds named in the docstring of `trace`).
3. Read: `custom_practice.jsonl`, `eval_NN.jsonl`, and through `evaluate` the training stream. Not opened: `custom_test.jsonl`, HumanEval, MBPP, any `val*.bin`.
4. Every change through `Session.put`; reversals through `put` with the saved old array (what `Session.undo` does; `undo` itself can take back only the last tensor of an edit of several). Before run 30 the working copy was checked to be equal to the base model.
