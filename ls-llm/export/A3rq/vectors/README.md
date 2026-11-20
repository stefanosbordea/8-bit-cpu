# Test vectors for bring-up

Made by the integer reference `lsllm.intref.IntModel` (bit-exact SPEC Part C) in the loop of
`lsllm.intref.completer.IntCompleter`, with `lsllm.filter_table.TableFilter` for the stop rule and the filter.
Setting of this export: stop rule on, syntax filter on, sampling at temperature 0.8 with the switches at 0. The CPU must reproduce every number here.

Prompt (21 tokens with the BOS token in front), already in `mem/data.hex`:

```
def add(a, b):
    """Return the sum of a and b."""
```

Token ids of the prompt: `1 53 126 439 12 434 25 449 27 5 24 105 57 303 430 85 67 135 449 447 24`

## Order of events

1. Reset: KV cache empty, position 0. The program runs alone until HALT.
2. Positions 0 .. 19: the prompt tokens but the last (prefill). Layers and KV cache only: no final norm,
   no logits, nothing sent. `position_00.txt` is the first of them: empty cache, one key per head, so every
   attention weight is 127. It is the simplest first test.
3. Position 20: the last prompt token. Its logits choose generated token 1 (`token_01.txt`), which is sent
   and fed at the next position, and so on: `token_NN.txt` is the step whose logits choose generated token
   NN, at position 19 + NN, with generated token NN - 1 as its input.
4. After the token that ends the text has been sent, the CPU halts.

## What the program must send

- Tokens: `4 50 67 164 449 2` (6 tokens, the last token is the end mark)
- UART bytes (`uart.bin`): `80 04 80 32 80 43 81 24 83 41 80 02`
- Text (the prompt, then the tokens sent, the end mark included):

```
def add(a, b):
    """Return the sum of a and b."""
    return a + b
```

The same prompt in the other settings of the program (each needs its own export for the memory files):

- greedy, filter off (`sent_greedy`): `4 50 67 164 449 430 164 449 2`
- greedy, filter on (`sent_filter`): `4 50 67 164 449 430 164 449 2`
- sampled at temperature 0.8, switches at 0 (LFSR reset value 0xACE1ACE1), filter on (`sent_sampled`): `4 50 67 164 449 2`

## Files

- `tokens.txt`: `prompt_ids`, `sent`, `uart`, `end_mark` (1: the last token of `sent` is the end mark) and the
  sequences of the other settings, in the format below.
- `position_00.txt`, `token_01.txt` ...: `position`, `token_in`, every traced value in the order of
  computation; the token files end with `allowed` (one 0 or 1 per token id: what the syntax filter allows), `token_out` and `uart` (its two bytes).
  A line that starts with a letter is `name size size ...` (`name 1` for a single number); the lines after
  it are the values in decimal, row-major, 16 per line.
- `vectors.npz`: the same arrays for numpy, keys `position_00.<name>`, `token_NN.<name>` and the names of `tokens.txt`.
- `uart.bin`: the raw frame bytes, for `tools/uart_receive.py --replay`.

## Names (module text of lsllm/intref/model.py)

```
Names used by the trace and by the accumulator peaks (l = layer number):
    x0                              embedding row, shifted right by sh_e with rounding
    l.ya, l.yf, y                   norm outputs (attention site, feed-forward site, final site)
    l.q, l.k, l.v                   projections;  l.qr, l.kr  after RoPE (as [a' of every pair, b' of every pair])
    l.code, l.max, l.e              attention (SPEC C.6), per query head: score codes 0..127 of the keys, their
                                    largest value, weights 0..127
    l.dsum, l.nrm, l.mant, l.r      sum of the weights (127 .. 127 T, more than a byte), its number of significant
                                    bits (the NRM instruction), its mantissa 64..127, the table value RECIP[mant]
    l.a, l.o                        value sums shifted to int8, head outputs
    l.xa, l.xf                      residual stream after the attention and after the feed-forward update
    l.h1, l.g1, l.h3, l.g           feed-forward
    u                               logits
A name followed by ".acc" is the accumulator (with its preload) before the requantisation that gives the name.
```

Every value is int8 except the accumulators (`.acc`, signed 24 bits, with the preload already added),
`dsum` (up to 127 T) and the counts `nrm` (7..15). Head axes: a query head is written (key/value head g,
head s inside the group), query head h = g * share + s. `code`, `e` have one entry per key 0 .. position.
`u` holds the logits before sampling and before the filter; `token_out` is the choice in the setting of
this export.

## Values of one token step, in order

| name | shape |
|---|---|
| position | 1 |
| token_in | 1 |
| x0.acc | 160 |
| x0 | 160 |
| l0.ya.m.acc | 1 |
| l0.ya.m | 1 |
| l0.ya.r | 1 |
| l0.ya.acc | 160 |
| l0.ya | 160 |
| l0.q.acc | 160 |
| l0.q | 160 |
| l0.k.acc | 40 |
| l0.k | 40 |
| l0.v.acc | 40 |
| l0.v | 40 |
| l0.qr.acc | 1 x 4 x 40 |
| l0.qr | 1 x 4 x 40 |
| l0.kr.acc | 1 x 40 |
| l0.kr | 1 x 40 |
| l0.code.acc | 1 x 4 x 26 |
| l0.code | 1 x 4 x 26 |
| l0.max | 1 x 4 x 1 |
| l0.e | 1 x 4 x 26 |
| l0.dsum | 1 x 4 x 1 |
| l0.nrm | 1 x 4 x 1 |
| l0.mant | 1 x 4 x 1 |
| l0.r | 1 x 4 x 1 |
| l0.a.acc | 1 x 4 x 40 |
| l0.a | 1 x 4 x 40 |
| l0.o.acc | 1 x 4 x 40 |
| l0.o | 1 x 4 x 40 |
| l0.xa.acc | 160 |
| l0.xa | 160 |
| l0.yf.m.acc | 1 |
| l0.yf.m | 1 |
| l0.yf.r | 1 |
| l0.yf.acc | 160 |
| l0.yf | 160 |
| l0.g.acc | 219 |
| l0.g | 219 |
| l0.xf.acc | 160 |
| l0.xf | 160 |
| l1.ya.m.acc | 1 |
| l1.ya.m | 1 |
| l1.ya.r | 1 |
| l1.ya.acc | 160 |
| l1.ya | 160 |
| l1.q.acc | 160 |
| l1.q | 160 |
| l1.k.acc | 40 |
| l1.k | 40 |
| l1.v.acc | 40 |
| l1.v | 40 |
| l1.qr.acc | 1 x 4 x 40 |
| l1.qr | 1 x 4 x 40 |
| l1.kr.acc | 1 x 40 |
| l1.kr | 1 x 40 |
| l1.code.acc | 1 x 4 x 26 |
| l1.code | 1 x 4 x 26 |
| l1.max | 1 x 4 x 1 |
| l1.e | 1 x 4 x 26 |
| l1.dsum | 1 x 4 x 1 |
| l1.nrm | 1 x 4 x 1 |
| l1.mant | 1 x 4 x 1 |
| l1.r | 1 x 4 x 1 |
| l1.a.acc | 1 x 4 x 40 |
| l1.a | 1 x 4 x 40 |
| l1.o.acc | 1 x 4 x 40 |
| l1.o | 1 x 4 x 40 |
| l1.xa.acc | 160 |
| l1.xa | 160 |
| l1.yf.m.acc | 1 |
| l1.yf.m | 1 |
| l1.yf.r | 1 |
| l1.yf.acc | 160 |
| l1.yf | 160 |
| l1.g.acc | 219 |
| l1.g | 219 |
| l1.xf.acc | 160 |
| l1.xf | 160 |
| y.m.acc | 1 |
| y.m | 1 |
| y.r | 1 |
| y.acc | 160 |
| y | 160 |
| u.acc | 512 |
| u | 512 |
| allowed | 512 |
| token_out | 1 |
| uart | 2 |
