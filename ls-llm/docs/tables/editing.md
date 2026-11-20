After "Sparks of Weight-Space Programming" (section 4): the int8 model was handed to four independent sessions of the
assistant model (Claude Fable 5.2), each with one strategy, and each allowed to change numbers of the model by hand but
not to train it. The protocol is enforced by code (`lsllm/edit/session.py`):

- no gradients, no backward pass, no optimiser; an edit is a set of numbers written into a tensor;
- the model is run on data only through two budgeted calls (a practice evaluation, or a trace of intermediate values),
  at most 50 per session; the data of these calls is practice data only (the 200 custom practice tasks and windows of
  the training text). The test sets and the validation text are never read by a session;
- an edit is refused if the model would no longer be a model of the LS CPU (int8 ranges, table rules, 16-bit preloads,
  shifts, worst-case accumulator inside 24 bits);
- every edit is logged with tensor, positions, old and new values (`EDITS.md`, `edits/<session>/edits.jsonl`).

After the sessions ended, the model left by each session was run once on the three test sets. Those numbers are the
"weight edit" rows of section 3 and the table at the top of `EDITS.md`; a session's own practice scores are in
`EDITS.md` as well and are not benchmark scores.
