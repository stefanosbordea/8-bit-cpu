# Weights and tables

One file per array: one byte per line, two hex digits, two's complement (`FF` = -1), row-major (the last
index runs fastest). A table of 256 entries is indexed by the unsigned byte of the int8 value: lines 0..127
are the values 0..127, lines 128..255 the values -128..-1. The tables of the syntax filter and GUMBEL hold
bytes 0..255. `mem/data.hex` already contains everything the program reads, laid out for the CPU; these
files are the same numbers one array at a time.

| file | shape | bytes | values | meaning |
|---|---|---|---|---|
| `E.hex` | 512 x 192 | 98,304 | -113 .. 103 | embedding matrix, row = token id (SPEC C.1); the same matrix is the output layer (C.9) |
| `W1_0.hex` | 510 x 192 | 97,920 | -79 .. 77 | feed-forward, first matrix of layer l (C.8) |
| `W1_1.hex` | 510 x 192 | 97,920 | -105 .. 108 | feed-forward, first matrix of layer l (C.8) |
| `W1_2.hex` | 510 x 192 | 97,920 | -66 .. 64 | feed-forward, first matrix of layer l (C.8) |
| `W1_3.hex` | 510 x 192 | 97,920 | -85 .. 72 | feed-forward, first matrix of layer l (C.8) |
| `W2_0.hex` | 192 x 510 | 97,920 | -40 .. 43 | feed-forward, output matrix of layer l, added to the residual stream (C.8) |
| `W2_1.hex` | 192 x 510 | 97,920 | -54 .. 49 | feed-forward, output matrix of layer l, added to the residual stream (C.8) |
| `W2_2.hex` | 192 x 510 | 97,920 | -59 .. 53 | feed-forward, output matrix of layer l, added to the residual stream (C.8) |
| `W2_3.hex` | 192 x 510 | 97,920 | -75 .. 63 | feed-forward, output matrix of layer l, added to the residual stream (C.8) |
| `Wk_0.hex` | 48 x 192 | 9,216 | -68 .. 72 | key projection of layer l (C.3) |
| `Wk_1.hex` | 48 x 192 | 9,216 | -76 .. 71 | key projection of layer l (C.3) |
| `Wk_2.hex` | 48 x 192 | 9,216 | -85 .. 77 | key projection of layer l (C.3) |
| `Wk_3.hex` | 48 x 192 | 9,216 | -92 .. 79 | key projection of layer l (C.3) |
| `Wo_0.hex` | 192 x 192 | 36,864 | -18 .. 15 | attention output projection of layer l, added to the residual stream (C.7) |
| `Wo_1.hex` | 192 x 192 | 36,864 | -39 .. 41 | attention output projection of layer l, added to the residual stream (C.7) |
| `Wo_2.hex` | 192 x 192 | 36,864 | -40 .. 41 | attention output projection of layer l, added to the residual stream (C.7) |
| `Wo_3.hex` | 192 x 192 | 36,864 | -51 .. 49 | attention output projection of layer l, added to the residual stream (C.7) |
| `Wq_0.hex` | 192 x 192 | 36,864 | -75 .. 80 | query projection of layer l, row = output component (SPEC C.3) |
| `Wq_1.hex` | 192 x 192 | 36,864 | -53 .. 66 | query projection of layer l, row = output component (SPEC C.3) |
| `Wq_2.hex` | 192 x 192 | 36,864 | -57 .. 64 | query projection of layer l, row = output component (SPEC C.3) |
| `Wq_3.hex` | 192 x 192 | 36,864 | -106 .. 108 | query projection of layer l, row = output component (SPEC C.3) |
| `Wv_0.hex` | 48 x 192 | 9,216 | -75 .. 80 | value projection of layer l (C.3) |
| `Wv_1.hex` | 48 x 192 | 9,216 | -125 .. 125 | value projection of layer l (C.3) |
| `Wv_2.hex` | 48 x 192 | 9,216 | -76 .. 75 | value projection of layer l (C.3) |
| `Wv_3.hex` | 48 x 192 | 9,216 | -114 .. 116 | value projection of layer l (C.3) |
| `NORM.hex` | 9 x 256 | 2,304 | 0 .. 127 | norm tables, row = norm site n = 0 .. 2L (attention site of layer l: 2l, feed-forward site: 2l + 1, final: 2L), index = unsigned byte (C.2) |
| `EXP.hex` | 4 x 256 | 1,024 | 0 .. 127 | attention weight tables, row = layer, index = (code - row maximum) mod 256 (C.6 step 3) |
| `RECIP.hex` | 256 | 256 | 65 .. 127 | reciprocal table min(127, round(8192 / v)), used at v = 64..127 (C.6 step 5) |
| `ROPE_C.hex` | 256 x 24 | 6,144 | -127 .. 127 | round(127 cos), row = position, column = pair j (C.4) |
| `ROPE_S.hex` | 256 x 24 | 6,144 | -127 .. 127 | round(127 sin), row = position, column = pair j (C.4) |
| `TOKEN.hex` | 2 x 512 | 1,024 | 0 .. 61 | syntax filter, row 0 = class of every token id, row 1 = its word column (lsllm/filter_table.py) |
| `ALLOW.hex` | 17 x 32 | 544 | 0 .. 1 | syntax filter, ALLOW[abstract state][class] = 1 if a token of that class may come; rows padded to 32 bytes |
| `LEX.hex` | 17 x 32 | 544 | 0 .. 16 | syntax filter, LEX[lexer state][class] = next lexer state; rows padded to 32 bytes |
| `WORD.hex` | 62 x 64 | 3,968 | 0 .. 48 | syntax filter, WORD[word column][word state] = next word state; rows padded to 64 bytes |
| `GUMBEL.hex` | 256 | 256 | 0 .. 13 | sampling noise for u >> 1 at the temperature of this export, index = random byte of RND (SPEC Part D) |

Total: 1,272,512 bytes in 35 files.
