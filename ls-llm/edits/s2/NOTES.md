# Session s2: names

Base model `runs/qat/A2q` (2 layers, 4 query heads, 1 key/value head, dim 160, vocabulary 512). Budget 50 runs.
All numbers below are on the 200 practice tasks (`evaluate`) or on traces of practice material.
Scripts are in `scripts/`, run from the repository root with `PYTHONPATH=. .venv/bin/python edits/s2/scripts/NN_*.py`.

## Baseline (run 1)

pass@1 0.005 (1 task: 002 `return n - 5`), syntax_ok 0.93, runs_ok 0.425, tests_passed 0.095, practice_loss 1.6463.
Errors: TypeError 53, NameError 38, AttributeError 4. Repeating 0.04.

A warning about what the scores can show: every edit, also a small one, changes 60 to 80 % of the 200 completions
(script 10). With 1 to 4 passing tasks, one task more or less is chance. I trust an effect only if it shows in
several variants of the same edit and in the error class it was aimed at.

## What the traces say (runs 2 and 8, no edits)

Run 2: reference solutions of the 79 shortest tasks, teacher forced. Run 8: prompt + the model's own baseline
completion of 61 tasks (all NameError tasks). `l<k>.e` cannot be traced over more than one position (the arrays
have different lengths and `trace` stacks them), so the attention weights are rebuilt in numpy from the traced
`qr` and `kr` with the formula of SPEC C.6 and checked against the traced `max` and `dsum` (scripts/lib.py).

- Layer 1 head 0 is the copy head: at body positions 39 % of its weight is on the signature, 9 % on the token after
  an earlier copy of the current token (induction). The other seven heads look at the current and previous tokens.
- Teacher forced, the model writes the right token at 542 of 840 body positions; where the right token is a token
  of the signature at 78 of 152. At those positions head 0 votes for the right token (+1527 when the model is
  wrong, +2838 when right, in units of the residual stream times E), the feed-forward of layer 1 votes against it
  when the model is wrong (-1521) (script 06). The opposing feed-forward units are syntax units (operators after a
  name, digits), spread over many units: there is no "habitual name" unit to switch off (script 07).
- In the model's own NameError completions (script 12) the first token of the undefined name
  - is a right argument token in 15 of 38 cases; the error comes later in the name (`day` for `days`, `item` for
    `items`, `ax` after `a`: the body needs the token with a trailing space, the signature has the one without);
  - in 17 of 38 cases head 0 has its largest weight on a DOCSTRING token and copies a word of the docstring
    (`integer`, `then`, `whose`, `there`, `source`, `z`(ero), `o`(v)`owel`). At these positions head 0 has 50 % of
    its weight on the docstring and 36 % on the signature, and it is the head that votes for the undefined name (+818).
- So the names fail less because of habit than because the copy head copies from the wrong place.

## Edits that failed (all taken back)

| run | edit | result | reading |
|---|---|---|---|
| 3 | Wo_1 columns of head 0 times 2 (strategy b: the copy head writes more) | pass 0, syntax 0.785, loss 1.92, repeating 0.14 | far too strong; the head also copies docstring words and repeats itself |
| 4 | the same times 1.25 | pass 0.005, tests 0.089, NameError 43 | no gain: more copying, also of the wrong things |
| 5 | Wq_1 rows of head 0 times 1.5 (sharper copy head) | pass 0.005, tests 0.093, NameError 44 | no gain |
| 6 | Wo_1 columns of heads 1 and 3 times 0.75 (they vote against copying) | pass 0.010, tests 0.096, NameError 41 | within chance (credit up in 15 tasks, down in 13) |
| 7 | EXP[1] with kappa times 0.75, ON TOP of the edit of run 6 (see below) | pass 0.005, tests 0.086, NameError 45 | worse |
| 10 | signature preference with 10 codes (see below) | pass 0.005, tests 0.051, repeating 0.12 | too strong |

Bookkeeping mistake at run 7: `Session.undo` takes back the last log entry, so my second call undid the first undo
and the edit of run 6 stayed in. Run 7 therefore measured run 6 + EXP, not EXP alone. Found by comparing the arrays
with the base model (`09_try.py diff`); since then experiments are taken back with `09_try.py back`, which puts the
kept state back through `Session.put` and checks it.

## The edit that works: a signature preference for the copy head (runs 9, 11, 12, 13)

Hypothesis: the copy head cannot tell a signature token from a docstring token, but the input of layer 1 can:
along the direction (mean of `l1.ya` over signature positions) - (mean over docstring positions) the signature
positions lie at 94 +- 41 and the docstring positions at -18 +- 29 (script 13, class means from runs 2 and 8).

Edit (09_try.py `sigpref`): one RoPE pair of head 0 is rewritten by hand. Pair 19 is the slowest (0.016 rad over
100 positions), so a score made in it does not depend on the distance.
- `Wq_1` row 38 (head 0, pair 19, part a) := the mean body input as a direction, scaled so that the query value is
  about 60 at every body position (the body positions give 86 +- 22 along it).
- `Wk_1` row 38 := the signature-minus-docstring direction, scaled so that a signature key scores N codes more
  than a docstring key (4 codes = 1 nat).
The old rows are replaced (that pair carried about 1.5 codes rms of the old scores).

| run | N codes | pass@1 | tests_passed | NameError | syntax_ok | runs_ok | loss |
|---|---|---|---|---|---|---|---|
| 1 | 0 (baseline) | 0.005 | 0.0950 | 38 | 0.930 | 0.425 | 1.6463 |
| 11 | 3 | 0.020 | 0.1082 | 33 | 0.910 | 0.425 | 1.6735 |
| 12 | 4.5 | 0.020 | 0.1007 | 35 | 0.915 | 0.445 | 1.6845 |
| 9 | 6 | 0.015 | 0.1029 | 28 | 0.880 | 0.415 | 1.7011 |
| 10 | 10 | 0.005 | 0.0505 | 36 | 0.830 | 0.265 | 1.7659 |
| 13 | 3, and Wo_1 head 0 times 1.15 | 0.015 | 0.1069 | 34 | 0.945 | 0.445 | 1.6903 |

Control (run 24): the same channel with 0 codes (query constant, key row zero) and the end bias of the next
section: NameError 44, pass 0.015 (002, 060, 101), tests 0.1005. Against that control the 3 codes take the
NameErrors from 44 to 32 and make task 025 pass (`return n % 2 != 0`, was `integer % 2 != 0`: the docstring word
is replaced by the argument, which is what the edit was built for). Task 060 passes in every variant, also in the
control, so it is NOT an effect of the preference: it comes with the emptied pair. The preference is real in the
error class it was aimed at (NameError 28 to 36 in the 19 runs that have it, 38 to 45 in the seven runs without it);
in pass@1 it is worth one task.

Variants of the channel, each evaluated once on top of the kept state, none better and all within chance:

| run | variant | pass@1 | tests_passed | NameError | loss |
|---|---|---|---|---|---|
| 17 | key direction with the function-name tokens at docstring level (script 19) | 0.020 | 0.1080 | 30 | 1.6768 |
| 18 | switched on only where a name is due (script 20, `gated`), 4.5 codes | 0.015 | 0.1007 | 28 | 1.6695 |
| 19 | the same, 7 codes | 0.015 | 0.0990 | 31 | 1.6790 |
| 23 | second class: argument mentions inside the docstring lifted as well (script 26, `sigmen`), 3 + 3 codes | 0.015 | 0.1048 | 31 | 1.6798 |
| 26 | 4.5 codes with the end bias | 0.015 | 0.1054 | 34 | 1.6836 |
| 27 | 6 codes with the end bias | 0.015 | 0.1101 | 29 | 1.7017 |

Kept: 3 codes, ungated.

## Second kept edit: an end bias on the row of </s> (runs 14, 15, 16)

Outside the names strategy, found on the way. Teacher forced, the model does not stop where the reference stops in
23 of 79 references (script 15): after the last token it prefers a lone space (and then goes on with `and ...`).
The final norm output y has a direction "the reference ends here" (mean of y at the last reference token minus
mean over the other body positions, script 17): ends at 191 +- 49, other body positions at -21 +- 80.

Edit (`endbias`): E row 2 (</s>) += delta times that unit direction. </s> never stands in a prompt, so the row is
never read as an input here: the edit acts on the output only.

| run | delta | pass@1 | tests_passed | syntax_ok | runs_ok | loss | completions changed against run 11 |
|---|---|---|---|---|---|---|---|
| 11 | 0 | 0.020 | 0.1082 | 0.910 | 0.425 | 1.6735 | |
| 14 | 20 | 0.020 | 0.1102 | 0.935 | 0.465 | 1.6718 | 23, all cut at the end of a complete expression, credit up in 1, down in 0 |
| 15 | 40 | 0.020 | 0.1127 | 0.940 | 0.485 | 1.6741 | |
| 16 | 70 | 0.020 | 0.1094 | 0.940 | 0.495 | 1.6808 | |

This is the only edit of the session with a low variance: it changes few answers and all in the intended way.
Kept: delta 40.

## More that failed or was a dead end

| run | edit, on top of the kept state | result | reading |
|---|---|---|---|
| 13 | Wo_1 head 0 times 1.15 (with signature preference 3, before the end bias) | 0.015, tests 0.107 | no gain |
| 20 | Wo_1 heads 1 and 3 times 0.75 | 0.010, tests 0.081, syntax 0.865 | worse |
| 21 | Wq_1 head 0 times 1.25 (sharper) | 0.015, tests 0.095, NameError 36 | worse |
| 22 | E row of the triple quote minus 60 times a body direction (script 25; the kept model writes `"""` inside 3 bodies) | 0.010, tests 0.086, 157 answers changed | the row is also how the two quotes of every prompt are READ: an E row of a token that stands in the prompt is not an output-only edit |
| 25 | W2_1 times 0.9 (the whole last feed-forward: habit against copy) | 0.020, tests 0.096 | no gain |

Without a run:
- Strategy (a), feed-forward units of the last layer that push habitual names: the votes for the undefined name
  are spread over generic units (largest single unit 11 % of the sum; the units that serve docstring text give 5 %;
  script 23). Nothing to switch off.
- Strategy (c), E rows of habitual names: no single habitual token; the undefined names are docstring words, words
  of the function name, and argument names in a wrong form. And run 22 shows what an E row of a prompt token costs.
- `c_s`: the row maximum never reaches code 127 and almost no key is at code 0 (script 07), so moving the
  offsets changes nothing.
- The NameErrors that are left (32) are mostly not reachable through attention: 8 are an argument in a wrong form
  (`item` for `items`, `day` for `days`: the body needs the token with the trailing space, and `item ` is itself
  a habitual name that wins by 4 nats although head 0 has its largest weight on the right key, script 14), 6 are a
  bare one-letter argument with a letter added (`ax`, `dx`, `s1`: after the bare token the tokenizer has no trained
  way to put a space, and layer 0's feed-forward knows `ax`; script 24).
- Name repeated around an operator (`b < b`): 58 of 200 answers at the baseline, 54 in the kept model. Head 0 looks
  at the signature copy of the name it has just used (scripts 21, 22). A hand-made "the other argument" channel
  would not make `b < b` right (the first operand is already the wrong one), so I did not build it.
- Numbers: layer 1 head 2 is the head that copies digits from the docstring (51 % of its weight on docstring
  digits when the right number is there; its columns of Wo_1 put the key's own digit at rank 1; scripts 27 to 29).
  The wrong numbers are a wrong choice among several numbers of the docstring or a wrong token form
  (`100` for `10`), not a missing copy. Nothing to add by hand.

## Result

Kept model = base + 3 logged edits: `Wq_1` row 38 (158 values), `Wk_1` row 38 (156 values), `E` row 2 (145 values).
`scripts/30_final.py` checks exactly that against the base arrays and evaluates once more (run 28): identical
scores and identical completions to run 15.

| | pass@1 | syntax_ok | runs_ok | tests_passed | practice_loss | NameError | TypeError | repeating |
|---|---|---|---|---|---|---|---|---|
| baseline (run 1) | 0.005 | 0.930 | 0.425 | 0.0950 | 1.6463 | 38 | 53 | 0.040 |
| kept (runs 15, 28) | 0.020 | 0.940 | 0.485 | 0.1127 | 1.6741 | 32 | 45 | 0.015 |

Passing tasks: 002, 025, 060, 101 (baseline: 002).

How much of this to believe. runs_ok +0.06 and tests_passed +0.018 come from two edits with a measured
mechanism. Of the three new passes, 025 is the mechanism (16 of the 19 runs with the preference, none without);
060 comes with emptying one RoPE pair (19 of the 20 runs that have the pair rewritten, the control among them);
101 passes only with exactly 3 codes (and in the control), with a program that looks wrong and happens to satisfy
the asserts, so it is luck. On held-out tasks I expect fewer NameErrors and fewer answers that
run on past their end, and a pass@1 gain of about one task in 200, not three.
The practice loss on training text rose by 0.028: training text has no signature/docstring layout of this kind,
so the preference misfires there.

28 of 50 runs used (2 traces, 26 evaluates). I stopped because every further idea was either measured to be a dead
end without a run or would only have re-rolled the answers.

## Files

- `scripts/lib.py`: the helper that the analysis scripts import (loads a trace, rebuilds the attention weights);
  not numbered because a module name cannot start with a digit.
- `scripts/09_try.py`: all edits (`sigpref`, `endbias`, the failed ones), `keep` / `back` / `diff` / `table`.
  To rebuild the kept model from the base: `09_try.py sigpref 3 + endbias 40 + noeval` (needs `regions.npz` from
  script 13 and `end.npz` from script 17, which need the two traces).
- `trace_refs.pkl`, `trace_own.pkl` (100 MB each): the traced values of runs 2 and 8, so that the analysis scripts
  run without budget. `kept.npz` / `kept.json`: the kept weights, used by `back` to take experiments back through
  `Session.put`; it is not a session and was never evaluated on its own.
- `regions.npz`, `end.npz`, `due.npz`, `mention.npz`, `quote.npz`, `digit.npz`: class means and directions.
