# Session s4: first body tokens, end of the answer, output side

Base `runs/qat/A2q`. All model runs go through `Session.evaluate` / `Session.trace` (36 of 50 used). Every
script is in `scripts/`, run from the repository root: `PYTHONPATH=. .venv/bin/python edits/s4/scripts/<name>.py`.
Logits are in codes: 4 codes = 1 nat. "Reference accuracy" = share of the body positions of the 200 reference
solutions (3834 positions, teacher forced) at which the syntax-filtered greedy token is the reference token.

## Result

| | pass@1 | syntax ok | runs ok | tests passed | practice loss |
|---|---|---|---|---|---|
| baseline (run 1) | 0.005 | 0.93 | 0.425 | 0.0950 | 1.6463 |
| **kept model (runs 23 and 36)** | **0.030** | 0.88 | 0.455 | 0.1115 | 1.8987 |
| sounder alternative (run 27, `restore.py 33`) | 0.025 | 0.95 | 0.510 | 0.1237 | 1.7915 |

The kept model is the first by the ranking of the task (pass@1, then tests_passed, then loss): 6 of 200 tasks
(27, 60, 61, 66, 101, 103). Against the exported model it differs in W2_1 (19 columns zero), Wo_1 (8 columns
zero) and EXP[1] (kappa 3 instead of 4).

**Read this before trusting 0.030.** One task is 0.005. Across the 21 edited states of this session the set of
passing tasks moves inside a pool of about ten short tasks (2, 16, 20, 27, 60, 61, 66, 69, 101, 103, 104); every
edit changes 160 to 190 of the 200 answers. The step 0.005 -> 0.020 (edit 3) is backed by a mechanism and by a
second measurement (reference accuracy 0.680 -> 0.695 -> 0.698, in both halves of the tasks). The last step
0.025 -> 0.030 is not: the kept state was reached by a slip (see "The slip"), and each of its three last
ingredients, tested alone on the 16-unit state, made the scores worse (runs 24, 31, 32). On held-out tasks the
alternative of run 27 (16 units zero, copy head x 1.25: better syntax, runs_ok, tests_passed and loss, one task
fewer) is the safer guess. `restore.py 33` puts it into the session directory; `restore.py 47` brings the kept
model back.

## Run log

| run | kind | state | pass@1 | syntax | runs ok | tests passed | loss | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | evaluate | baseline | 0.005 | 0.93 | 0.425 | 0.0950 | 1.6463 | |
| 2-5 | trace | baseline | | | | | | references of all 200 tasks; accuracy 0.680 |
| 6 | evaluate | unit 136 zero | 0.005 | 0.93 | 0.415 | 0.0992 | 1.6583 | kept |
| 7 | evaluate | 60 units zero | 0.010 | 0.75 | 0.435 | 0.0785 | 2.3133 | damaged, undone |
| 8 | evaluate | 8 units zero ("e3") | 0.020 | 0.93 | 0.455 | 0.1030 | 1.7083 | kept |
| 9 | evaluate | 16 units zero (ranks 9-16) | 0.015 | 0.845 | 0.490 | 0.0927 | 1.7650 | undone |
| 10-13 | trace | e3 | | | | | | accuracy 0.695 |
| 14 | evaluate | 12 units zero | 0.020 | 0.93 | 0.515 | 0.1166 | 1.7512 | kept |
| 15 | evaluate | 16 units zero ("e6") | 0.025 | 0.915 | 0.525 | 0.1209 | 1.7703 | kept |
| 16-19 | trace | e6 | | | | | | accuracy 0.698 |
| 20 | evaluate | e6 + E rows '== ' +6, 'a ' +5 | 0.025 | 0.915 | 0.525 | 0.1127 | 1.7892 | undone |
| 21 | evaluate | e6 + E row '== ' +6 | 0.025 | 0.915 | 0.525 | 0.1209 | 1.7723 | no effect, undone |
| 22 | evaluate | e6 + 8 head dims + 3 units zero | 0.025 | 0.895 | 0.525 | 0.1147 | 1.8518 | worse (meant to be undone) |
| 23 | evaluate | state of 22 + EXP[1] kappa 3 | 0.030 | 0.88 | 0.455 | 0.1115 | 1.8987 | **kept by the ranking** |
| 24 | evaluate | e6 + EXP[1] kappa 3 | 0.015 | 0.90 | 0.410 | 0.0821 | 1.8208 | undone |
| 25 | evaluate | e6 + EXP[1] kappa 5.33 | 0.010 | 0.91 | 0.540 | 0.1077 | 1.8069 | undone |
| 26 | evaluate | e6 + E row EOS -4 | 0.025 | 0.895 | 0.500 | 0.1192 | 1.7712 | undone |
| 27 | evaluate | e6 + head 0 x 1.25 | 0.025 | 0.95 | 0.510 | 0.1237 | 1.7915 | best sound state |
| 28 | evaluate | e6 + head 0 x 1.5 | 0.025 | 0.93 | 0.445 | 0.1062 | 1.8490 | undone |
| 29 | evaluate | e6 + head 0 x 0.8 | 0.015 | 0.92 | 0.540 | 0.1094 | 1.7805 | undone |
| 30 | evaluate | run 27 + units 54, 78, 191, 83, 26 zero | 0.025 | 0.94 | 0.415 | 0.1052 | 1.8184 | undone |
| 31 | evaluate | run 27 + units 209, 34, 62 zero | 0.020 | 0.905 | 0.480 | 0.1040 | 1.8315 | undone |
| 32 | evaluate | run 27 + 8 head dims zero | 0.020 | 0.935 | 0.405 | 0.1107 | 1.8305 | undone |
| 33 | evaluate | run 27 + units 10, 194 given back | 0.020 | 0.915 | 0.465 | 0.1161 | 1.7786 | undone |
| 34 | evaluate | run 27 + head 1 x 1.25 | 0.015 | 0.86 | 0.445 | 0.0989 | 1.8035 | undone |
| 35 | evaluate | run 27 + EXP[0] kappa 3.4 | 0.025 | 0.865 | 0.440 | 0.1061 | 1.8070 | undone |
| 36 | evaluate | state of run 23, restored | 0.030 | 0.88 | 0.455 | 0.1115 | 1.8987 | final |

## What the baseline does (runs 1-5; scripts 02, 04, 05, 07, 08, 14, 15)

- Completions: plausible one-line bodies with the wrong content: the wrong argument (`b < b and b < b`), words
  of the docstring instead of arguments (38 NameErrors), wrong numbers, wrong operator.
- The end of the answer is NOT the problem: only one more task would pass if its answer were cut at the best
  place (script 02). No treatment of the end-of-text token can give more than that.
- First body tokens: `<NL>` always right, `return ` at 83 %, from the third body token on about half.
- At the reference positions: ' ' wrongly chosen 80 / missed 48, 'if ' 42 / 16 (the habit `x if ...` where the
  reference has `x + ...`), 'b' 49 / 26, '* ' 17 / 9; missed: '+ ' 29, '== ' 20, EOS 40, ', ' 32.
- Direct logit attribution (script 07): the stream before the final norm is l0.xf + Wo_1 o / 256 + W2_1 g / 256,
  so the share of every part in  logit(reference) - logit(best other allowed token)  is a projection on
  E[reference] - E[other]. Mean margin 9.2 codes = layer-0 stream 1.8 + heads 1.6 + feed-forward units 5.8.
  99 of the 219 units of layer 1 have a negative total; even and odd tasks agree (correlation 0.94).
- Unit 136 writes 'if ', '* ', 'not ' and pushes ' ' down: the `x if` / `x *` habit sits in one column.
- Attention (script 15): 2 to 3 keys per head, no score code at 127. Nothing to repair in c_s.

## Edits that worked

1. **W2_1 column 136 = 0** (script 09, run 6): the `if ` / `* ` unit. tests_passed 0.095 -> 0.099.
3. **W2_1 columns 194, 207, 70, 10, 93, 206, 67 = 0** (script 12, run 8): with 136 the eight units with the most
   negative margin contribution at DECISIVE positions (|margin| <= 8 codes), negative in both halves of the
   tasks. pass@1 0.005 -> 0.020, syntax_ok unchanged. The re-trace confirms it: reference accuracy 0.680 -> 0.695
   (even tasks 0.667 -> 0.685, odd 0.693 -> 0.705).
5. **W2_1 columns 19, 133, 212, 102 = 0**, then **177, 218, 125, 205 = 0** (script 20, runs 14 and 15): the same
   rule on the re-trace of state e3. pass@1 0.025, tests_passed 0.121, runs_ok 0.525; accuracy 0.698. The gain per
   unit is already small, the loss on training text rises (1.65 -> 1.77): these units are useful for text in
   general and harmful for greedy answers to this kind of task.
8. **Wo_1 columns 0..39 x 1.25** (script 27, run 27): head 0 of layer 1 is the copy head (script 26: +8.4 codes
   of margin where a word of the prompt continues, +7.3 where one begins, -1.3 elsewhere). syntax_ok 0.95,
   tests_passed 0.124, pass@1 unchanged. x 1.5 and x 0.8 are worse (NameErrors 27 / 33 / 42 at x 1 / 1.25 / 1.5:
   a louder copy head copies docstring words too). Not part of the kept model, part of the alternative.

## Edits that failed

2. **60 units zero** (script 11, run 7): every unit negative in all four statistics. syntax_ok 0.75, loss 2.31.
   A large negative TOTAL mostly means "lowers the margin where the model is right anyway" plus structure
   (brackets, line ends). Only the decisive positions say which units hurt greedy decoding.
4. **Units 161, 102, 19, 218, 123, 121, 197, 16 zero** (script 13, run 9): syntax_ok 0.845; answers lose their
   operands (`n < and not ...`) or end inside `a if b`. 161, 16, 123 write digits; 121 holds EOS and line breaks
   back while an expression is open. Do not touch them.
6. **Token shifts through E rows** (scripts 18, 23; runs 20, 21, 26), the strategy this session was given.
   A row is moved along the mean of y over body positions (a few int8 steps move its logit by a chosen number
   of codes, +-45 %). The threshold statistics of the traced logits predicted it and the runs agree:
   for EOS, 'return ', ')', ']', ' ', '[', '0', '1', 'b' a shift in either direction loses more reference
   positions than it wins: the unigram calibration is right, the mistakes depend on the context.
   'a ' +5 made the model name loop variables `a` (tests_passed down). '== ' +6 (predicted +7 positions)
   changed nothing measurable. EOS -4 (the pruned model ends too early: 35.6 characters per answer against
   43.7 in the references) lowered syntax_ok. **The output-side strategy is a dead end on this model.**
7. **Third pruning round** (script 24; runs 22, 31, 32): Wo_1 head dimensions 157, 149, 51, 145, 83, 96, 57, 92 and
   units 209, 34, 62, chosen by the same rule on the re-trace e6. Worse together and each alone. The rule is
   exhausted after 16 units.
9. **EXP tables** (script 25; runs 24, 25, 35): layer 1 sharper (kappa 3) 0.015, flatter (5.33) 0.010; layer 0
   sharper (3.4) no gain. The exported decay is the right one.
10. **Units at name-start positions** (scripts 28, 29; run 30): 54 (digits), 78, 191, 83, 26 work against the
    reference where a prompt word must begin. Zeroing them: NameErrors 33 -> 43.
11. **Giving units 10 and 194 back** (scripts 21, 30; run 33): they held EOS back by 5.4 and 3.8 codes where EOS
    is wrong. One pass fewer.
12. **Head 1 x 1.25** (run 34): syntax_ok 0.86.

## The slip (how the kept state came about)

After run 22 the two edits of script 24 were to be taken back with two calls of `Session.undo`. `undo` takes back
the last LOG ENTRY, and the first undo is itself an entry, so the second call re-applied the edit and the Wo_1
edit stayed. Run 23 therefore measured "state of run 22 + EXP[1] kappa 3" instead of "e6 + EXP[1] kappa 3".
That state scored pass@1 0.030, the highest of the session. `restore.py` (state after a given log entry, through
`Session.put`) was written after that and used from then on. All edits went through `put`; the log is complete.

## What does not get better by hand

- Which argument: 'a ' (the argument a in a body; in the signature it is the token 'a') is right at 8 of 34
  reference positions. The copy head takes the name mentioned last (`b`), and after writing a name it finds
  that name again (`a >= a`). This is a matter of where head 0 looks, not of what it writes (script 22).
- Semantics: wrong constant, wrong operator, the docstring word instead of the argument. Per-token accuracy
  is 0.70, a body has 15 to 20 tokens; only the shortest tasks can pass.

## Notes on the scripts

- 07, 08, 14, 21, 22, 26, 28 read the weights of the working model: they were run at the state named in their
  text (07, 08, 14 at the baseline). 17 needs the state its trace was made with.
- 02 runs cut answers of an evaluate run against the practice tests in the repository sandbox (no model run).
- `undo.py`, `show_log.py`, `show_completions.py` are the inline calls and listings of the session, saved.
- No script computes a gradient or optimises numbers; every edit is zeroing, scaling by a chosen factor, a
  table rebuilt from its formula, or a row moved along a mean activation by a chosen amount.
