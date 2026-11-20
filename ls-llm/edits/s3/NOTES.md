# Session s3: the hardware-specific knobs (EXP, c_s, NORM, RECIP)

Scripts are in `scripts/`, run from the repository root with `.venv/bin/python edits/s3/scripts/NN_name.py`.
Scores are (pass@1, tests_passed, syntax_ok, runs_ok, practice_loss). 200 practice tasks: one task is 0.005 of pass@1.

## Run 1: baseline (`01_baseline.py`)
pass@1 0.005 (1 task: subtract_five), tests_passed 0.095, syntax_ok 0.93, runs_ok 0.425, practice_loss 1.6463.
Errors: TypeError 53, NameError 38. The bodies are fluent Python that ignores what the docstring asks.

## Run 2: trace (`02_trace.py`, statistics in `03_analyse.py`)
50 practice tasks, prompt + reference solution, 3551 tokens. What the knobs see:
- Score codes: the row maximum is 63..110 (p5..p95) in every head, never below 23 and at 127 in 0.1 % of the rows;
  0.0-0.2 % of the keys sit at code 0. So nothing is cut off at either end. **c_s is a dead knob**: moving a head by a
  whole number of codes changes no weight unless it pushes the maximum into 127 (which only loses information).
- Attention is flat: 6 to 12 effective keys per head, 40-67 % of the keys are within the 23 codes that get a weight > 0.
- Norm sites: the index m stays inside the table (m = 127 in 1.4 % of positions at sites 1 and 2, table clamp 127 at
  0.1 % at site 4). No clamp to remove; the gain is the only thing to change.
- Residual stream and projections: no saturation (q, k at the clamp in < 0.1 %).
- Logits: top code 52..101 (p5..p95), never 127. Ties at the top in 2.8 % of body positions (greedy then takes the
  smaller token id). Teacher-forced accuracy on body tokens 66 %.

## Runs 3-6: knob (a) EXP, one layer, kappa 3 or 5.5 (`04_exp_kappa.py`), each taken back
| run | edit | pass@1 | tests_passed | syntax_ok | runs_ok | loss | NameError |
|---|---|---|---|---|---|---|---|
| 1 | baseline (kappa 4) | 0.005 | 0.0950 | 0.930 | 0.425 | 1.6463 | 38 |
| 3 | EXP[0] kappa 3 | 0.010 | 0.0842 | 0.880 | 0.360 | 1.6833 | 47 |
| 4 | EXP[0] kappa 5.5 | 0.010 | 0.1207 | 0.930 | 0.495 | 1.7198 | 33 |
| 5 | EXP[1] kappa 3 | 0.010 | 0.0898 | 0.885 | 0.370 | 1.6935 | 44 |
| 6 | EXP[1] kappa 5.5 | 0.015 | 0.0951 | 0.905 | 0.415 | 1.6977 | 45 |

Hypothesis (sharper attention copies the names of the prompt better) is wrong: sharper tables raise NameErrors and
lower runs_ok in both layers. The loss rises in both directions: kappa 4 is where QAT left the model.
`05_compare.py`: any table change rewrites 146-179 of the 200 completions, and the tasks that pass are different ones
each time (subtract_five stays). **pass@1 at this level (1 to 3 tasks) is a lottery; a change of one or two tasks is
not evidence.** I use tests_passed, runs_ok and the error classes as the signal, and ask for a trend across
neighbouring values before believing a knob.
The one visible effect: a flatter layer 0 (kappa 5.5) runs more often (0.495 against 0.425) with fewer NameErrors.

## Runs 7-9: EXP[0] flatter, to see whether run 4 was a trend (`04_exp_kappa.py 0:4.7 0:7 0:9`), each taken back
| run | edit | pass@1 | tests_passed | syntax_ok | runs_ok | loss | NameError |
|---|---|---|---|---|---|---|---|
| 7 | EXP[0] kappa 4.7 | 0.010 | 0.1069 | 0.905 | 0.415 | 1.6654 | 33 |
| 8 | EXP[0] kappa 7 | 0.015 | 0.0870 | 0.890 | 0.435 | 1.9201 | 37 |
| 9 | EXP[0] kappa 9 | 0.005 | 0.0522 | 0.755 | 0.255 | 2.2408 | 52 |

No trend: runs_ok 0.415 / 0.495 / 0.435 for kappa 4.7 / 5.5 / 7, so the 0.495 of run 4 was a peak of the noise
(runs_ok moves by about 0.04, tests_passed by about 0.015 between neighbouring settings). Beyond kappa 7 the model
breaks (loss 2.24, repeating 0.10). **Knob (a) has no setting that is better than the export in a way I can tell
from noise; the loss is lowest at kappa 4 in both layers.**

## Attribution on the trace (`06_attribution.py`, no budget)
The final residual stream is the sum of five parts; body-token accuracy of argmax E @ x is 66.3 %.
Mean contribution to the margin reference - rival: ffn1 1528, ffn0 444, attn1 134, x0 -64, attn0 -53: the last
feed-forward layer decides, attention adds little directly. What-if (exact for layer 1 up to rounding): scaling what
attn1 or ffn1 write by 0.7 .. 1.5 moves the accuracy between 65.0 % and 67.0 % without a direction (one position is
0.1 %). So the gains of the last layer (NORM[2], NORM[3], RECIP) cannot be expected to change what is chosen.

## Runs 10-18: knob (c) NORM gain, one site at a time (`07_norm_gain.py`), each taken back
| run | edit | pass@1 | tests_passed | syntax_ok | runs_ok | loss | NameError | changed completions |
|---|---|---|---|---|---|---|---|---|
| 10 | NORM[0] x 0.9 | 0.010 | 0.0885 | 0.905 | 0.400 | 1.7068 | 41 | 181 |
| 11 | NORM[0] x 1.1 | 0.015 | 0.0814 | 0.835 | 0.380 | 1.6963 | 36 | 180 |
| 12 | NORM[1] x 0.9 | 0.015 | 0.1012 | 0.915 | 0.370 | 1.6634 | 44 | 149 |
| 13 | NORM[1] x 1.1 | 0.010 | 0.1016 | 0.900 | 0.465 | 1.6553 | 29 | 140 |
| 14 | NORM[2] x 0.9 | 0.015 | 0.0895 | 0.940 | 0.420 | 1.6873 | 47 | 154 |
| 15 | NORM[2] x 1.1 | 0.015 | 0.0983 | 0.885 | 0.415 | 1.6949 | 32 | 157 |
| 16 | NORM[3] x 0.9 | 0.005 | 0.0868 | 0.925 | 0.410 | 1.6542 | 42 | 80 |
| 17 | NORM[3] x 1.1 | 0.005 | 0.0880 | 0.930 | 0.415 | 1.6497 | 42 | 74 |
| 18 | NORM[4] x 1.2, c_l 12722 -> 5247 | 0.005 | 0.0895 | 0.920 | 0.405 | 1.6805 | 41 | 38 |

- No site lowers the loss; the export gain is the optimum at every site (as the attribution said for layer 1).
- NORM[3] and NORM[4] (the last feed-forward gain and the logit scale) change few completions and nothing in the
  score: finer logits (run 18, 20 % more codes per nat, to break the 2.8 % ties) do not help. Dead.
- The sites that feed attention or layer 0 rewrite 140-180 completions and give 2 or 3 passes instead of 1, in both
  directions (0.9 and 1.1). The passing tasks come from one small pool that the model nearly solves:
  subtract_five (always), keep_multiples_of_three, largest_of_three, count_letter_characters,
  count_lowercase_letters, and once each numbers_to_strings, is_negative_number, count_plus_and_minus_signs.
  Which of them falls on the right side depends on the perturbation, not on its direction.

## Runs 19-22: knobs (d) RECIP and (b) c_s (`08_recip_cs.py`), each taken back
| run | edit | pass@1 | tests_passed | syntax_ok | runs_ok | loss | changed completions |
|---|---|---|---|---|---|---|---|
| 19 | RECIP x 0.88 | 0.010 | 0.0781 | 0.890 | 0.365 | 1.6689 | 168 |
| 20 | RECIP x 1.12 (clamped at 127) | 0.010 | 0.0930 | 0.895 | 0.405 | 1.6688 | 167 |
| 21 | c_s of all 8 heads -16 codes | 0.005 | 0.0950 | 0.930 | 0.425 | 1.6464 | 0 |
| 22 | c_s of layer 1 +8 codes | 0.005 | 0.0950 | 0.930 | 0.425 | 1.6463 | 0 |

- c_s: as predicted from the trace, not one completion changes (and the evaluation is deterministic). Layer 0 cannot
  even be moved up by 16 codes: 29156 + 16 * 512 does not fit the 16-bit preload. Dead knob.
- RECIP: worse loss both ways, same lottery of passes.

## Where the four knobs stand after 22 runs
The export values are the optimum of the loss for every knob, and no knob has a direction that helps the tasks.
`09_pool.py`: over 21 evaluated models, subtract_five passes 20 times, keep_multiples_of_three 9, largest_of_three 4,
count_letter_characters 3, count_lowercase_letters 2, three others once. The first tokens of the body pick a template
(`return list(range(` or `return [nums for nums in`), mostly without regard to the docstring, and a perturbation
re-draws it. The unedited model (1 pass) is an unlucky member of its neighbourhood (mean of the perturbed models: 2.1).

## Leaving the knobs for one hypothesis that the completions support (`10_overuse.py`)
Token counts, 200 baseline bodies against the 200 reference bodies: the model over-writes chains:
`and ` 36 against 10, `el`+`se ` 33 against 2, `if ` 72 against 45, `not ` 13 against 3; it under-writes `(` `)` `, `.
Two baseline bodies are the reference plus an `and` clause: `return n < 0 and n < len(n)` (is_negative_number),
`... if char.islower() and char not in 'aeiou'` (count_lowercase_letters).
Hypothesis: the token `and ` (id 135) is written too easily for this kind of task; making it less likely (its row of
E scaled down: the logit of a token is its row times y) ends such bodies where the reference ends.
Prediction before the run: is_negative_number and count_lowercase_letters pass, subtract_five stays, few other bodies change.

## Run 23: E[135] (`and `) x 0.85 (`11_token_row.py 135:0.85`), taken back
pass@1 0.005, tests_passed 0.0955, runs_ok 0.42, loss 1.6457 (a hair below the baseline), 57 completions changed.
**The prediction failed.** `and ` appears in 15 bodies instead of 30, but the model writes `or ` in its place
(`return n < 0 or not n`) or another clause (`char.islower() and char in 'aeiou'`). The chain is not one token's
fault; the body does not end where the reference ends. No task gained. Not pursued: a real fix would need the model
to read the docstring, which no row scale or table gives it.

## Combinations of the mild single-knob changes (`12_combo.py`)
Honest framing: by now this is drawing more members of the neighbourhood, not a mechanism. The rule ranks the
session's model by practice pass@1, so I look for a combination of small changes (loss near the baseline) in which
several of the nearly solved tasks fall on the right side together, and check its neighbours.
| run | edit (on the unedited model, taken back) | pass@1 | tests_passed | syntax_ok | runs_ok | loss | passes |
|---|---|---|---|---|---|---|---|
| 24 | NORM[1] x 0.9 + EXP[1] kappa 5.5 | 0.010 | 0.0857 | 0.850 | 0.355 | 1.6999 | subtract_five, largest_of_three |
| 25 | NORM[1] x 0.9 + NORM[2] x 1.1 | 0.005 | 0.0821 | 0.830 | 0.340 | 1.7254 | subtract_five |
| 26 | NORM[1] x 0.9 + NORM[2] x 0.9 | 0.010 | 0.0829 | 0.885 | 0.315 | 1.6881 | subtract_five, largest_of_three |
| 27 | NORM[1] x 1.1 + EXP[0] kappa 5.5 | 0.010 | 0.1049 | 0.915 | 0.395 | 1.7446 | subtract_five, keep_multiples_of_three |
| 30 | NORM[1] x 0.9 + NORM[3] x 1.1 | 0.010 | 0.0986 | 0.870 | 0.365 | 1.6688 | subtract_five, largest_of_three |
| 31 | NORM[1] x 0.9 + NORM[3] x 0.9 | 0.010 | 0.1070 | 0.915 | 0.395 | 1.6675 | subtract_five, keep_multiples_of_three |

**The winners do not combine.** Every pair is worse than its better half: the loss adds up, syntax_ok and runs_ok
fall, and the passes of the two halves do not add. That is what noise looks like, not two mechanisms.

## Runs 28, 29, 32, 33: the neighbours of the best single change, NORM[1] x 0.9 (`12_combo.py norm:1:F`)
| run | NORM[1] times | pass@1 | tests_passed | syntax_ok | runs_ok | loss | passes |
|---|---|---|---|---|---|---|---|
| 29 | 0.85 | 0.010 | 0.0963 | 0.890 | 0.375 | 1.6824 | subtract_five, keep_multiples_of_three |
| 33 | 0.88 | 0.015 | 0.1042 | 0.895 | 0.415 | 1.6660 | subtract_five, largest_of_three, keep_multiples_of_three |
| 12 | 0.90 | 0.015 | 0.1012 | 0.915 | 0.370 | 1.6634 | the same three |
| 32 | 0.92 | 0.015 | 0.1089 | 0.900 | 0.405 | 1.6562 | the same three |
| 28 | 0.95 | 0.015 | 0.0926 | 0.900 | 0.375 | 1.6534 | the same three |
| 1 | 1.00 (export) | 0.005 | 0.0950 | 0.930 | 0.425 | 1.6463 | subtract_five |
| 13 | 1.10 | 0.010 | 0.1016 | 0.900 | 0.465 | 1.6553 | subtract_five, largest_of_three |

This is the only place of the session where neighbours agree: with the gain of the first feed-forward layer 5 to
12 % below the export, the same three tasks pass at four settings in a row.

## Kept: NORM[1] x 0.92 (run 34, `12_combo.py norm:1:0.92 --keep`)
The table of norm site 1 (input of the feed-forward layer of layer 0) times 0.92, rounded: 256 entries, e.g.
127 -> 117, 125 -> 115, 119 -> 109. Nothing else differs from the export (`13_final_check.py`).
Run 34 = run 32: pass@1 0.015 (3 tasks), tests_passed 0.1089, syntax_ok 0.900, runs_ok 0.405, practice_loss 1.6562.
By the ranking (pass@1, then tests_passed, then loss) it is the best of the 32 models evaluated (33 evaluate runs, run 34 repeats run 32).

What this is worth, plainly: two more tasks of 200, both from the pool the model nearly solves (it writes
`[nums for nums in nums if nums % 3 == 0]` instead of `list(range(nums, nums + 1))`, and one more `else` clause in
largest_of_three). syntax_ok and runs_ok are a little LOWER than the export (0.900 / 0.405 against 0.930 / 0.425)
and the loss a little higher (+0.010). I have no mechanism for why a weaker first feed-forward layer should help.
The plateau over 0.88..0.95 speaks against pure chance for these two tasks on this set, but it does not say that
the held-out tasks gain; I expect the held-out difference to be zero within noise.

## Summary
- Worked (weakly): NORM[1] x 0.88..0.95, kept at 0.92.
- Failed: EXP kappa in either layer and direction (runs 3-9); NORM gain at sites 0, 2, 3, 4 (runs 10, 11, 14-18);
  RECIP (19, 20); c_s (21, 22: provably no effect); finer logits (18); E row of `and ` (23); every combination (24-27, 30, 31).
- Why: QAT left every table at the optimum of the loss, nothing saturates that a knob could free, and what the
  tasks need (a body that follows the docstring) is not something a gain or a table shape can give this model.
- 34 of 50 runs used. The other 16 were not spent: more draws would only have fitted the practice set.
- Record: `14_views.py` holds read-only views (tables, samples, name counts, diffs) that were first typed inline and
  saved afterwards; `01_baseline.py` was rewritten after run 1 to import `common` (the first version needed PYTHONPATH=.).
  The score codes in `03_analyse.py` and the parts in `06_attribution.py` are dot products of traced values with the
  weights (statistics of run 2), not a run of the model.
- Edits that did not help were taken back inside `trial` by `Session.put` / `Session.put_number` of the old values
  (logged as "take back: ..."), not by `Session.undo`: undo reverses only the last log entry, and a trial can hold
  several edits. The effect and the log are the same.
