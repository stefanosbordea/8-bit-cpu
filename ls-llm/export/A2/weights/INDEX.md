# Weights and tables

One file per array: one byte per line, two hex digits, two's complement (`FF` = -1), row-major (the last
index runs fastest). A table of 256 entries is indexed by the unsigned byte of the int8 value: lines 0..127
are the values 0..127, lines 128..255 the values -128..-1. The tables of the syntax filter and GUMBEL hold
bytes 0..255. `mem/data.hex` already contains everything the program reads, laid out for the CPU; these
files are the same numbers one array at a time.

| file | shape | bytes | values | meaning |
|---|---|---|---|---|
| `E.hex` | 512 x 160 | 81,920 | -66 .. 70 | embedding matrix, row = token id (SPEC C.1); the same matrix is the output layer (C.9) |
| `W1_0.hex` | 219 x 160 | 35,040 | -107 .. 93 | feed-forward, first matrix of layer l (C.8) |
| `W1_1.hex` | 219 x 160 | 35,040 | -97 .. 93 | feed-forward, first matrix of layer l (C.8) |
| `W2_0.hex` | 160 x 219 | 35,040 | -48 .. 52 | feed-forward, output matrix of layer l, added to the residual stream (C.8) |
| `W2_1.hex` | 160 x 219 | 35,040 | -86 .. 76 | feed-forward, output matrix of layer l, added to the residual stream (C.8) |
| `Wk_0.hex` | 40 x 160 | 6,400 | -91 .. 112 | key projection of layer l (C.3) |
| `Wk_1.hex` | 40 x 160 | 6,400 | -118 .. 103 | key projection of layer l (C.3) |
| `Wo_0.hex` | 160 x 160 | 25,600 | -21 .. 26 | attention output projection of layer l, added to the residual stream (C.7) |
| `Wo_1.hex` | 160 x 160 | 25,600 | -30 .. 38 | attention output projection of layer l, added to the residual stream (C.7) |
| `Wq_0.hex` | 160 x 160 | 25,600 | -63 .. 67 | query projection of layer l, row = output component (SPEC C.3) |
| `Wq_1.hex` | 160 x 160 | 25,600 | -66 .. 75 | query projection of layer l, row = output component (SPEC C.3) |
| `Wv_0.hex` | 40 x 160 | 6,400 | -93 .. 95 | value projection of layer l (C.3) |
| `Wv_1.hex` | 40 x 160 | 6,400 | -65 .. 55 | value projection of layer l (C.3) |
| `NORM.hex` | 5 x 256 | 1,280 | 0 .. 127 | norm tables, row = norm site n = 0 .. 2L (attention site of layer l: 2l, feed-forward site: 2l + 1, final: 2L), index = unsigned byte (C.2) |
| `EXP.hex` | 2 x 256 | 512 | 0 .. 127 | attention weight tables, row = layer, index = (code - row maximum) mod 256 (C.6 step 3) |
| `RECIP.hex` | 256 | 256 | 65 .. 127 | reciprocal table min(127, round(8192 / v)), used at v = 64..127 (C.6 step 5) |
| `ROPE_C.hex` | 256 x 20 | 5,120 | -127 .. 127 | round(127 cos), row = position, column = pair j (C.4) |
| `ROPE_S.hex` | 256 x 20 | 5,120 | -127 .. 127 | round(127 sin), row = position, column = pair j (C.4) |
| `TOKEN.hex` | 2 x 512 | 1,024 | 0 .. 61 | syntax filter, row 0 = class of every token id, row 1 = its word column (lsllm/filter_table.py) |
| `ALLOW.hex` | 17 x 32 | 544 | 0 .. 1 | syntax filter, ALLOW[abstract state][class] = 1 if a token of that class may come; rows padded to 32 bytes |
| `LEX.hex` | 17 x 32 | 544 | 0 .. 16 | syntax filter, LEX[lexer state][class] = next lexer state; rows padded to 32 bytes |
| `WORD.hex` | 62 x 64 | 3,968 | 0 .. 48 | syntax filter, WORD[word column][word state] = next word state; rows padded to 64 bytes |
| `GUMBEL.hex` | 256 | 256 | 0 .. 13 | sampling noise for u >> 1 at the temperature of this export, index = random byte of RND (SPEC Part D) |

Total: 368,704 bytes in 23 files.
