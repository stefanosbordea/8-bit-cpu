**Outcome.** The sessions raised the practice score from 1 task of 200 to 3 to 6 tasks, and said themselves that this was
within what chance gives on answers that change with every small edit (one session's random-direction controls gained as
much). The held-out check agrees: on the custom test set every edited model passes fewer tasks than the unedited one (7 to 9
against 10), on MBPP they are within one or two tasks of it, on HumanEval all stay at zero. What did move is below pass@1:
more answers run without an exception after sessions s2 and s4. **No edited model is adopted; the final int8 model is the
unedited one** (D-edit-2). Choosing a session by its test score would be tuning on the test set and was not done.

What the traces showed about the model is kept in the session notes (`edits/s*/NOTES.md`): the second layer's first head is
the copy head and takes names from the docstring instead of the signature in 17 of 38 NameError answers; repetition is
written by attention, not by the feed-forward layers; only 8 of 200 answers loop.

A different use of the same idea did work: the exact repair of request programs in the board model (section 3, "Programs on
request"; D-board-3) changes entries of E without gradients, with an exact integer objective on training text.

The sessions worked on the Tier A int8 model of that time (10,000 training steps, `runs/qat/A2q`: the rows "Tier A, earlier
model" in section 3). The final Tier A model was trained later (D-qat-7) and was not edited.
