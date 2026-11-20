# Memory map of the data memory D

Taken from the `Build` that `lsllm.ls.gen.build` returned for this model. `mem/data.hex` and `mem/data.coe`
hold D at reset from address 0, one byte per line; addresses are 20 bits (pointers P0, P1, P2). The KV cache
and the scratch vectors are zero at reset and written by the program. `mem/map.json` has the same regions
for programs: `{"tokenizer": ..., "regions": {name: [address, bytes]}}`.

## Totals

- Data memory: 452,014 bytes (0x6E5AE), 3,616,112 bits. The machine addresses at most 2^19 = 524,288 bytes.
- 36 Kb block RAMs: **111** of the 135 on the board. A 36 Kb block used 8 bits wide (4K x 9, the ninth bit
  is parity and stays unused) holds 4,096 bytes, so the count is bytes / 4,096 rounded up, and 135 blocks
  hold 552,960 bytes. Counting all 36,864 bits of a block (135 blocks = 4,860 Kb, about 607 KB) the image
  would need 99 blocks, but that needs a 9-bit packing that the plain byte-wide memory does not have.
- Program ROM: 1,008 words of 16 bits (`mem/program.hex`); the 12-bit PC allows 4,096. As block RAM in the
  2K x 18 form: 1 block(s) of 36 Kb.

## The prompts

The image holds a table of prompts (PROMPTS.md lists them). Once after reset the start-up piece reads the
switches, takes SW5..0 as the prompt number k and copies that prompt into the prompt buffer (`prompt.left`
at 0x02D3F, `prompt.ids` at 0x03000), from where the program feeds it.

- `prompts.index` at 0x6E000 (line 450561 of `mem/data.hex`, two more in `mem/data.coe`): 4 bytes per switch value
  k = 0..63: the address of prompt k as [bits 11..8], [low byte], [bits 19..12], then its number of tokens n.
  A switch value beyond the table has the entry of prompt 0.
- `prompts.table` at 0x6E100, 1,198 bytes: per prompt the n high bytes of its token ids (id >> 8), then the n
  low bytes (id & 255), prompt after prompt.

Other prompts: run the export again with `--prompts`, or write them with `lsllm.ls.gen.set_prompts(image, b, lists)`
(1 to 64 prompts of 1 to min(T, 255) tokens that fit the 1,198 bytes of this table).

## Bytes by kind

| kind | bytes |
|---|---|
| tables | 5,376 |
| tok | 1,536 |
| scratch | 1,855 |
| prompt | 513 |
| filter | 10,272 |
| ones | 256 |
| pos | 22,784 |
| heads | 104 |
| kv.V | 20,480 |
| kv.K | 20,480 |
| w | 350,080 |
| prompts | 1,454 |
| unused (alignment) | 16,824 |
| total | 452,014 |

## Lookup tables (table number t of `LUT Rd, #t`)

| t | address | content |
|---|---|---|
| 0 | 0x00000 | NORM[0] |
| 1 | 0x00100 | NORM[1] |
| 2 | 0x00200 | NORM[2] |
| 3 | 0x00300 | NORM[3] |
| 4 | 0x00400 | NORM[4] |
| 5 | 0x00500 | EXP[0] |
| 6 | 0x00600 | EXP[1] |
| 7 | 0x00700 | RELU (v for v = 0..127, 0 for 128..255; made by the generator) |
| 8 | 0x00800 | RECIP |
| 9 | 0x00900 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 10 | 0x00A00 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 11 | 0x00B00 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 12 | 0x00C00 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 13 | 0x00D00 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 14 | 0x00E00 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 15 | 0x00F00 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 16 | 0x01000 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 17 | 0x01100 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 18 | 0x01200 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 19 | 0x01300 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |
| 20 | 0x01400 | made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py) |

## Every region in address order

| name | start | bytes | content |
|---|---|---|---|
| `tables` | 0x00000 | 5,376 | lookup tables of 256 bytes each for `LUT`; this address is LUT_BASE (table t is at LUT_BASE + 256 t, list below) |
| (unused) | 0x01500 | 2,816 | alignment, never read |
| `tok.addr` | 0x02000 | 1,536 | token table: per token id the address of its row of E, as three columns [low byte], [bits 11..8], [bits 19..12] |
| `scratch.rec` | 0x02600 | 89 | record of the current position, a copy of one row of pos.records (at reset: the link to record 0) |
| `scratch.token` | 0x02659 | 2 | the current token id as [high byte, low byte]; the program leaves the token it chose here |
| `scratch.r` | 0x0265B | 1 | multiplier of an rms norm |
| `scratch.x` | 0x0265C | 160 | residual stream x (D bytes) |
| `scratch.y` | 0x026FC | 160 | output of the norm in use |
| `scratch.q` | 0x0279C | 160 | queries q |
| `scratch.k` | 0x0283C | 40 | keys k |
| `scratch.v` | 0x02864 | 40 | values v |
| `scratch.o` | 0x0288C | 160 | attention output o |
| `scratch.head` | 0x0292C | 53 | record of the current head, then its rotated query |
| `scratch.e` | 0x02961 | 259 | attention weights e of the current head (one byte per key), then k, r and a page number |
| `scratch.g` | 0x02A64 | 219 | feed-forward: hidden vector g |
| `scratch.u` | 0x02B3F | 512 | logits u, one byte per token id (0..127) |
| `prompt.left` | 0x02D3F | 1 | number of prompt tokens still to feed; at reset the length of the prompt (see The prompt) |
| (unused) | 0x02D40 | 704 | alignment, never read |
| `prompt.ids` | 0x03000 | 512 | the prompt: byte p = high byte of token p, byte 256 + p = low byte (see The prompt) |
| `filter.class` | 0x03200 | 512 | class of every token id (TOKEN row 0), for the stop rule and the syntax filter |
| (unused) | 0x03400 | 3,072 | alignment, never read |
| `filter.rows` | 0x04000 | 4,096 | 16 rows of 256 bytes: level tables, abstract-state tables, bracket tables, the bracket stack, and in row 15 the filter state (see lsllm/ls/gen.py) |
| `filter.khi` | 0x05000 | 512 | syntax filter: word column of every token id, upper part of the address into filter.word |
| `filter.klo` | 0x05200 | 512 | syntax filter: word column of every token id, lower part of the address into filter.word |
| (unused) | 0x05400 | 3,072 | alignment, never read |
| `filter.word` | 0x06000 | 4,096 | syntax filter: byte 64 k + w = word state after a token of column k in word state w (WORD) |
| `filter.allow` | 0x07000 | 544 | syntax filter: byte 32 a + c = 1 if class c may come in abstract state a (ALLOW) |
| `ones` | 0x07220 | 256 | T bytes that hold 1 (a stream of e against them gives the sum of the weights) |
| `pos.records` | 0x07320 | 22,784 | position records: link to the next record, number of keys, key address, RoPE row of the position |
| `heads` | 0x0CC20 | 104 | head records, one per layer and query head: addresses of q, keys, value rows, o, and the preload c_s |
| (unused) | 0x0CC88 | 888 | alignment, never read |
| `kv.V0.0.0` | 0x0D000 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.15` | 0x0D100 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.14` | 0x0D200 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.13` | 0x0D300 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.12` | 0x0D400 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.11` | 0x0D500 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.10` | 0x0D600 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.9` | 0x0D700 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.8` | 0x0D800 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.7` | 0x0D900 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.6` | 0x0DA00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.5` | 0x0DB00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.4` | 0x0DC00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.3` | 0x0DD00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.2` | 0x0DE00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.1` | 0x0DF00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.16` | 0x0E000 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.31` | 0x0E100 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.30` | 0x0E200 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.29` | 0x0E300 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.28` | 0x0E400 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.27` | 0x0E500 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.26` | 0x0E600 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.25` | 0x0E700 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.24` | 0x0E800 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.23` | 0x0E900 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.22` | 0x0EA00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.21` | 0x0EB00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.20` | 0x0EC00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.19` | 0x0ED00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.18` | 0x0EE00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.17` | 0x0EF00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| (unused) | 0x0F000 | 256 | alignment, never read |
| `kv.V0.0.39` | 0x0F100 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.38` | 0x0F200 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.37` | 0x0F300 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.36` | 0x0F400 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.35` | 0x0F500 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.34` | 0x0F600 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.33` | 0x0F700 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V0.0.32` | 0x0F800 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| (unused) | 0x0F900 | 1,792 | alignment, never read |
| `kv.K0.0` | 0x10000 | 10,240 | KV cache, written at run time: keys K[layer].[key/value head], one row of d bytes per position |
| (unused) | 0x12800 | 2,048 | alignment, never read |
| `kv.V1.0.0` | 0x13000 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.15` | 0x13100 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.14` | 0x13200 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.13` | 0x13300 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.12` | 0x13400 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.11` | 0x13500 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.10` | 0x13600 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.9` | 0x13700 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.8` | 0x13800 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.7` | 0x13900 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.6` | 0x13A00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.5` | 0x13B00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.4` | 0x13C00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.3` | 0x13D00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.2` | 0x13E00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.1` | 0x13F00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.16` | 0x14000 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.31` | 0x14100 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.30` | 0x14200 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.29` | 0x14300 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.28` | 0x14400 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.27` | 0x14500 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.26` | 0x14600 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.25` | 0x14700 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.24` | 0x14800 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.23` | 0x14900 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.22` | 0x14A00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.21` | 0x14B00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.20` | 0x14C00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.19` | 0x14D00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.18` | 0x14E00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.17` | 0x14F00 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| (unused) | 0x15000 | 256 | alignment, never read |
| `kv.V1.0.39` | 0x15100 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.38` | 0x15200 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.37` | 0x15300 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.36` | 0x15400 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.35` | 0x15500 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.34` | 0x15600 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.33` | 0x15700 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| `kv.V1.0.32` | 0x15800 | 256 | KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position |
| (unused) | 0x15900 | 1,792 | alignment, never read |
| `kv.K1.0` | 0x16000 | 10,240 | KV cache, written at run time: keys K[layer].[key/value head], one row of d bytes per position |
| `w.Wq_0` | 0x18800 | 25,600 | weight matrix, row-major, the bytes of weights/Wq_0.hex |
| `w.Wk_0` | 0x1EC00 | 6,400 | weight matrix, row-major, the bytes of weights/Wk_0.hex |
| `w.Wv_0` | 0x20500 | 6,400 | weight matrix, row-major, the bytes of weights/Wv_0.hex |
| `w.Wo_0` | 0x21E00 | 25,600 | weight matrix, row-major, the bytes of weights/Wo_0.hex |
| `w.W1_0` | 0x28200 | 35,040 | weight matrix, row-major, the bytes of weights/W1_0.hex |
| `w.W2_0` | 0x30AE0 | 35,040 | weight matrix, row-major, the bytes of weights/W2_0.hex |
| `w.Wq_1` | 0x393C0 | 25,600 | weight matrix, row-major, the bytes of weights/Wq_1.hex |
| `w.Wk_1` | 0x3F7C0 | 6,400 | weight matrix, row-major, the bytes of weights/Wk_1.hex |
| `w.Wv_1` | 0x410C0 | 6,400 | weight matrix, row-major, the bytes of weights/Wv_1.hex |
| `w.Wo_1` | 0x429C0 | 25,600 | weight matrix, row-major, the bytes of weights/Wo_1.hex |
| `w.W1_1` | 0x48DC0 | 35,040 | weight matrix, row-major, the bytes of weights/W1_1.hex |
| `w.W2_1` | 0x516A0 | 35,040 | weight matrix, row-major, the bytes of weights/W2_1.hex |
| `w.E` | 0x59F80 | 81,920 | weight matrix, row-major, the bytes of weights/E.hex |
| (unused) | 0x6DF80 | 128 | alignment, never read |
| `prompts.index` | 0x6E000 | 256 | prompt table: 4 bytes per switch value 0..63: address of its prompt in prompts.table as [bits 11..8, low byte, bits 19..12], then its number of tokens |
| `prompts.table` | 0x6E100 | 1,198 | prompt table: per prompt the high bytes of its token ids, then the low bytes (see PROMPTS.md) |
