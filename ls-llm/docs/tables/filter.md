The filter (`lsllm/filter_table.py`, `docs/FILTER.md`) decides before every generated token which token ids may come
next, so that brackets, strings and indentation stay well formed. It is the same code in the float benchmarks, in the
integer reference and, as tables, in the generated LS program (the program's choices equal the reference on every test).

- Rules: a closing bracket must match the innermost open one (at most 8 open); no line break inside brackets or inside
  a one-line string; after a line that ends with `:` the next line break must go one level deeper, after any other line
  it must not; a line that begins with a block keyword must end with `:` and no other line may; a dedent cannot go
  below level 0; no empty line; no `#` and no `\` in code; the text ends only where a line may end.
- Tables: two bytes per token (a class out of 25 and a word column), 17 abstract states, three tables
  (ALLOW, LEX, WORD): 6,080 bytes in the description, 10,272 bytes in the program's memory image with the state and
  the per-token address bytes. No character is looked at while it runs.
- Cost on the CPU, measured on the generated program: 6,748 cycles per generated token (12 per token id in the argmax
  loop and 604 for the state), 200 cycles per prompt token, 135 program words.
- On 6,000 canonical human functions the filter accepts 5,997 from start to end (the 3 others use `match`); random
  text drawn only from allowed tokens had no bracket, string or indentation error in 240 of 240 trials.
- What it does not promise: that the text parses (it knows nothing about expressions, so `x = = 1` passes) or that the
  program is right. The effect on the scores is the difference between the "syntax filter" rows of the tables in
  section 3: it raises `syntax ok` and cannot create a solution that the model does not have.
