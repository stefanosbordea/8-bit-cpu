# The LS program

`mem/program.hex` (one 16-bit word per line), `mem/program.lst` (address, word, instruction, component).
Shape: D = 160, L = 2 layers, H = 4 query heads, G = 1 key/value heads, head size d = 40, F = 219, V = 512,
T = 256; norm = rms, ffn = relu. Machine: level L1 of docs/SPEC.md Part A.

This is the complete program of `lsllm.ls.gen.build`: stop rule on, syntax filter on, greedy.
Its prompt: prompts typed over the UART (needs the proposed instruction `RX`). Once it has a prompt it runs alone:

1. Prefill: every prompt token but the last is fed from the prompt buffer in the data memory; it runs the
   layers and fills the KV cache, and skips the final norm, the output layer, the argmax and the UART.
2. The last prompt token and every token after it run all steps of the table below; the chosen token is
   sent over the UART and becomes the input of the next position.
3. The program halts after sending a token that ends the text (the end-of-text token, or a line break that
   brings the indentation level back to 0: the end mark, which the host drops), or when the context is full.

Start-up piece (tag `start` in `mem/program.lst`, 60 words: the piece in front of the per-token program and
the three instructions at the end that send the end byte and jump back): after reset and after every answer
it puts the position, the stop level and the filter state back to their start and reads a prompt from the
UART with `RX` (two bytes per token id, then 0xFF; UART.md) into the prompt buffer. After the answer the
program sends 0xFF and waits for the next prompt: it never halts. `RX` is a proposed instruction (docs/SPEC.md
A.2); without it this form cannot run. Measured in the simulator for the prompt of the test vectors
(21 tokens): 1,812 cycles per answer from the first byte to the first token step, waiting for the UART not
counted; 12 more for the end byte. docs/COST_TABLE.md, section Choosing the prompt on the board, has the formula.

## What the generator says about its program (module text of lsllm/ls/gen.py)

```
From an exported integer model (docs/SPEC.md, Part E) to what the LS CPU needs: a memory image and a program.

The program computes ONE token and jumps back to its start (with prompt=True: until the text ends, see below): it reads the token id and the position from memory,
computes the logits u (SPEC Part C), picks the next token greedily (C.9), writes the KV cache, moves to the next
position and sends the token over the UART (Part D). Layers are unrolled: every address in the program is a
constant. The heads of a layer share one piece of program, which reads the addresses of its head from a record.

How the registers are used in the row loops: R3 is the DJNZ counter, R1:R2 hold the 16-bit preload of
`LDA R1, R2`, R0 receives the result of RQ. P0 is the pointer that is rewound for every row, P1 walks through the
weights, P2 is the store pointer. There is no SETP2: P2 is set with `SETP0 a ; CPP P2, P0`. The two attention
loops need a fifth value (the running maximum; k and r of the division) and say how they make room for it.

The memory map (Build.mem) in address order:
    tables     tables of 256 bytes for LUT: NORM of every site, EXP of every layer, RELU (for maxima), RECIP, SILU;
               then, only if used: EMB (the embedding shift rqr(v, sh_e) of SPEC C.1) and the tables of the options
    tok        three bytes per token id: the address of its row of E, as [low byte], [bits 11..8], [bits 19..12]
    scratch    the variables and vectors of the token being computed (see `build`)
    ones       T bytes that hold 1: a stream of the attention weights against them gives their sum (SPEC C.6)
    pos        one record per position (see `records`): the program copies the record of its position into
               scratch.rec at the start of a token. No address is ever computed from the position by the CPU.
    heads      one record per layer and query head (see `head_records`): the program has the code of a head once
               per layer and runs it for every record of the layer (Emitter.for_records)
    kv         per layer and key/value head: the rows V[j][0..T-1] (one 256-byte slot per row, so that the
               position is the low byte of the address) and the block K[0..T-1][0..d-1]. With an attention
               window W (cfg.window) the cache is a ring of W slots: position p writes slot p mod W and reads
               the first min(p + 1, W) slots; slot and count come from the position record, the program is
               the same except for one more record byte. The value rows keep their 256-byte slots.
    w          the weight matrices in the order the program reads them, then E

Options of `build` (all off by default; cycles in docs/COST_TABLE.md):
    prompt          the complete program. prompt.left = prompt tokens still to feed, prompt.ids = their ids (byte p:
                    high byte of token p, byte 256 + p: low byte), written by `set_prompt`. A prompt token that is
                    not the last one jumps back to the start after the layers. The CPU has no UART receiver.
    tf              the tables of the tokenizer (lsllm.filter_table.TableFilter). filter.class = class of every
                    token id. filter.rows = 16 rows of 256 bytes, read with `MVP P, Rrow, Rbyte ; LD`:
                    rows 0..6 new level [change][level]; 7, 12, 13 abstract state + 1 from the level and from
                    [colon][word]; 9..11 what an opening bracket puts on the stack [type][depth]; 14 the bracket
                    stack (one byte per level, byte 0 = 0); 15 the state: level of the stop rule, lex, level,
                    two bytes of the word column of the current token, word, colon, depth.
                    With `prompt`: halt after sending the end of text or a line break back to level 0.
    syntax_filter   the step of docs/FILTER.md for every token that is fed (prompt tokens too), and before the
                    argmax the row of ALLOW of the abstract state is copied into the table ALLOWED; the argmax
                    reads the class of every token id and skips the ids whose class is not allowed.
    temperature     > 0: u[j] <- (u[j] >> 1) + GUMBEL[RND] in one pass over all token ids before the argmax.
    prompts         a start-up piece before the program (run once): the switches choose one prompt of a table
                    (prompts.index, prompts.table: the last regions of the memory) and it is copied to prompt.ids.
    uart_prompt     a start-up piece that reads the prompt from the UART with RX, after reset and after every
                    answer; it first puts the position, the stop level and the filter state back to their start.
```

## One token, step by step

The steps from the norms to the gating run once per layer, the others once per token. The layers are unrolled
in the ROM, so the words of a layer step are those of all 2 layers. Words: instructions of the step in the ROM.

Cycles: one generated token of this export (stop rule on, syntax filter on, greedy) at the given context length (position = context - 1),
from `lsllm.ls.cost.token_cycles` for this shape and these options, with the filter tables of the
tokenizer `data/tok/f512a`; every instruction 4 cycles, a stream of n multiply-adds n + 4.

|  | step | what the program does | ROM words | cycles, context 16 | cycles, context 64 | cycles, context 256 |
|---|---|---|---|---|---|---|
| 1 | embedding | Read the token id, look up the address of its row of E in the token table, copy the row to x (SPEC C.1). | 28 | 2,016 | 2,016 | 2,016 |
| 2 | norms | Norm of x into y: 2 per layer and the final one. dyt: one table lookup per byte; rms: sum of squares, RQ, table, one product per byte (C.2). | 109 | 20,316 | 20,316 | 20,316 |
| 3 | q/k/v projections | q = Wq y, k = Wk y, v = Wv y: one stream per row, RQ with rounding, store (C.3). | 60 | 88,416 | 88,416 | 88,416 |
| 4 | RoPE | Rotate every query head and every key head with the RoPE row of the position: two products per value, shift 7 (C.4). | 88 | 9,352 | 9,352 | 9,352 |
| 5 | cache writes | Write the rotated keys to K[l][g][p] and every value component to its row V[l][g][j][p] (C.5). | 64 | 1,440 | 1,440 | 1,440 |
| 6 | scores, maximum and exp | Per query head: one stream per key (q' . K[t] + c_s), RQ with ReLU to a code 0..127, running maximum; then e[t] = EXP_l[code - maximum] (C.6 steps 1 to 3). | 60 | 13,280 | 51,680 | 205,280 |
| 7 | value mixing and division | Per query head: Dsum = sum of e (a stream against the row of ones), `NRM`, `RQV`, RECIP table; per value row one stream e . V[j], shift by k, times r, shift sh_o: the head output o (C.6 steps 4 and 5). | 164 | 26,368 | 42,112 | 105,088 |
| 8 | attention output projection | x <- rq(256 x + 128 + Wo o, 8): one stream per row with x[i] as the preload (C.7). | 32 | 60,232 | 60,232 | 60,232 |
| 9 | feed-forward matrices | Feed-forward matrices: h1 = W1 y, h3 = W3 y (swiglu) or g = relu(W1 y); then x <- rq(256 x + 128 + W2 g, 8) (C.8). | 50 | 159,728 | 159,728 | 159,728 |
| 10 | output layer | After the final norm: u[j] = rq(c_l + E[j] . y, sh_l) with ReLU for every token id j: V streams of length D (C.9). | 21 | 94,244 | 94,244 | 94,244 |
| 11 | argmax | The smallest j with the largest u[j]: `SUB`, sign table, `BZ` per logit (C.9, Part D). | 32 | 9,992 | 9,992 | 9,992 |
| 12 | bookkeeping and UART | Bookkeeping: copy the position record, loop over the head records, store the chosen token, move to the next position, send the two UART bytes with `TX`, jump to the start (HALT when the context is full). | 76 | 2,884 | 2,884 | 2,884 |
| 13 | prompt feeding and prefill test | Prompt feeding: while prompt tokens are left, the next token comes from the prompt buffer instead of the last choice; a prompt token that is not the last one jumps back to the start after the layers (prefill: no final norm, output layer, argmax, UART). | 27 | 32 | 32 | 32 |
| 14 | token class, level, stop rule | Class of the token, indentation level, stop rule: HALT after sending the end-of-text token or a line break that brings the level to 0. | 30 | 104 | 104 | 104 |
| 15 | syntax filter | Syntax filter (docs/FILTER.md): the state step for every token that is fed, prompt tokens too; before the argmax the ALLOW row of the abstract state is copied into the table ALLOWED, and the argmax skips token ids whose class is not allowed. | 135 | 7,052 | 7,052 | 7,052 |
| 16 | start-up piece (per answer) | Start-up piece, run once per answer and not per token. Prompt table: `IN` reads the switches, SW5..0 choose a prompt of the table, it is copied into the prompt buffer. Prompts over the UART: position, stop level and filter state are reset, the prompt is read with `RX` into the prompt buffer; after the answer the end byte 0xFF is sent and the program jumps back here. | 60 | 0 | 0 | 0 |
|  | **total** |  | 1,036 | 495,456 | 549,600 | 766,176 |
|  | **tokens/s at 100 MHz** |  |  | 201.8 | 182.0 | 130.5 |

## Cycles per token by kind of token: cycles (tokens/s at 100 MHz)

Sampling costs the same at any temperature > 0. The prefill token has the filter setting of this export.

| token | context 16 | context 64 | context 256 |
|---|---|---|---|
| generated token: greedy, filter off | 488,708 (204.6) | 542,852 (184.2) | 759,428 (131.7) |
| + filter **(this export)** | 495,456 (201.8) | 549,600 (182.0) | 766,176 (130.5) |
| + sampling | 503,056 (198.8) | 557,200 (179.5) | 773,776 (129.2) |
| + filter + sampling | 509,804 (196.2) | 563,948 (177.3) | 780,524 (128.1) |
| prefill token (a prompt token that is not the last) | 380,336 (262.9) | 434,480 (230.2) | 651,056 (153.6) |

The steps whose cycles differ between these kinds, at context 64:

| step | generated token | + filter | + sampling | + filter + sampling | prefill token |
|---|---|---|---|---|---|
| norms | 20,316 | 20,316 | 20,316 | 20,316 | 16,252 |
| output layer | 94,244 | 94,244 | 94,244 | 94,244 | 0 |
| argmax | 10,276 | 9,992 | 10,276 | 9,992 | 0 |
| bookkeeping and UART | 2,884 | 2,884 | 2,884 | 2,884 | 2,832 |
| prompt feeding and prefill test | 32 | 32 | 32 | 32 | 108 |
| token class, level, stop rule | 124 | 104 | 124 | 104 | 92 |
| syntax filter | 0 | 7,052 | 0 | 7,052 | 220 |
| sampling | 0 | 0 | 14,348 | 14,348 | 0 |

A whole answer costs the prefill cycles of every prompt token but the last plus the generated-token
cycles of every token sent; the round-trip check in README.md gives the cycles the simulator counted for
the prompt of the test vectors.

The cycle counts are minima, counted on zero weights. How they are counted and what depends on the data
(module text of lsllm/ls/cost.py; `token_cycles` adds: with the filter the first allowed token counts
as one new maximum of the argmax and is included):

```
Cycle cost of one token on the LS CPU, counted on the generated program itself (lsllm.ls.gen).

The program of a shape is generated with zero weights and run for one token in the simulator at the wanted
position. The simulator reports how often every instruction was executed and how many multiply-adds every
stream did; `price` turns these counts into cycles: for L1 (4 cycles per instruction, 1 more per stream element)
or for a hardware change of level L2 (the what-if switches, never used in the main numbers).

The cycles depend on the shape, the options and the position only, with two exceptions, so the numbers are minima:
the argmax spends 16 more cycles for every new maximum it meets (at most 127 times, logits are 0..127; 24 cycles
in a last block of fewer than 256 logits), and sending the token takes 12 more cycles when bit 7 of the low byte
of its id is set. Zero weights give neither (token 0, no new maximum).

    .venv/bin/python -m lsllm.ls.cost        writes docs/COST_TABLE.md
```
