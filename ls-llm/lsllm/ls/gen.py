"""From an exported integer model (docs/SPEC.md, Part E) to what the LS CPU needs: a memory image and a program.

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
"""
import json
import os
from typing import NamedTuple

import numpy as np

from lsllm.config import ModelConfig
from lsllm.ls.isa import TABLE_L1, encode
from lsllm.ls.iss import L1

R0, R1, R2, R3 = range(4)
PAGE = 4096       # SETP0/SETP1/MVP set the low 12 bits of a pointer, MVH the 8 bits above: a page is 4,096 bytes
HEAD = 13         # bytes of a head record (see `head_records`)
COMPONENTS = ("embedding", "norm", "qkv", "rope", "cache", "scores", "mix", "attn_out", "ffn", "gating", "output",
              "argmax", "book")
EXTRA = ("prompt", "stop", "filter", "sample", "start")     # components of the options of `build`
END = 0xFF                                         # UART byte that ends a prompt and an answer (uart_prompt)
ROWS = 15                                          # row of the filter page that holds the filter state
POP = 8                                            # value of the bracket table for a closing bracket


def half(sh):
    """Rounding preload of rqr (SPEC B.2): 2^(sh - 1), and 0 for sh = 0."""
    return (1 << sh) >> 1


class Emitter:
    """Collects the instructions of a program. While doing so it keeps track of the pointer values that are known
    at each point of the program, so that a pointer is only set where it does not already hold the address."""

    def __init__(self):
        self.code = []              # per program address: (mnemonic, operand values, component)
        self.tag = "book"           # component of the instructions that follow
        self.loops = []             # (first address, address of the DJNZ, kind): "row" = rewind, preload, stream,
                                    # requantise, store; "elem" = one or two products per value; "map" = table map
        self.page = [None] * 3      # bits 19..12 of P0, P1, P2 at this point of the program, None = not known
        self.low = [None] * 3       # bits 11..0

    def emit(self, name, *args, tag=None):
        """Append one instruction; update what is known after it. Returns its address."""
        self.code.append((name, args, tag or self.tag))
        if name in ("MAC", "MACL"):
            self.page[0] = self.low[0] = self.page[1] = self.low[1] = None
        elif name in ("LD", "ST") and args[2]:              # post-increment
            self.page[args[1]] = self.low[args[1]] = None
        elif name in ("LD2", "ST2") and args[1]:
            self.page[2] = self.low[2] = None
        elif name == "MVP":
            self.low[args[0]] = None
        elif name == "MVH":
            self.page[args[0]] = None
        elif name in ("SETP0", "SETP1"):
            self.low[int(name[-1])] = args[0]
        elif name == "CPP20":
            self.page[2], self.low[2] = self.page[0], self.low[0]
        return len(self.code) - 1

    def words(self):
        return [encode(name, *args, table=TABLE_L1) for name, args, tag in self.code]

    def at(self, p, address):
        """State that pointer p holds `address` here (after a loop the emitter cannot know it by itself)."""
        self.page[p], self.low[p] = address >> 12, address % PAGE

    def page_of(self, p, address, tmp=R0):
        """Make bits 19..12 of pointer p those of `address` (uses register tmp when they have to change)."""
        if self.page[p] != address >> 12:
            self.emit("LDI", tmp, address >> 12)
            self.emit("MVH", p, tmp)
            self.page[p] = address >> 12

    def point(self, p, address, tmp=R0):
        """P0 or P1 <- address."""
        self.page_of(p, address, tmp)
        if self.low[p] != address % PAGE:
            self.emit("SETP%d" % p, address % PAGE)

    def point2(self, address):
        """P2 <- address, through P0."""
        if (self.page[2], self.low[2]) != (address >> 12, address % PAGE):
            self.point(0, address)
            self.emit("CPP20")

    def forget(self, **pages):
        """A point that is reached on several paths: nothing is known about the pointers, except the pages given
        as p0=, p1=, p2= (bits 19..12 that every path leaves there)."""
        self.page = [pages.get("p%d" % p) for p in range(3)]
        self.low = [None] * 3

    def branch(self, name):
        """A forward branch; `land` gives it its target."""
        return self.emit(name, 0)

    def land(self, address):
        """The branch at `address` jumps to here. The instructions it skips must not move a pointer."""
        name, args, tag = self.code[address]
        self.code[address] = (name, (len(self.code),), tag)

    def loop(self, count, body, kind=None):
        """Emit body() inside DJNZ loops so that it runs `count` times (one loop does at most 256 passes).
        count = None: R3 already holds the number of passes. The body must not depend on what the emitter knows."""
        while count is None or count > 0:
            passes = 256 if count is None else min(count, 256)
            if count is not None:
                self.emit("LDI", R3, passes % 256)          # R3 = 0 gives 256 passes
            first = len(self.code)
            body()
            last = self.emit("DJNZ", first)
            if kind:
                self.loops.append((first, last, kind))
            count = 0 if count is None else count - passes

    def preload(self, c):
        """R1:R2 <- the signed 16-bit constant c, for LDA R1, R2."""
        if not -32768 <= c < 32768:
            raise ValueError("the preload %d does not fit 16 bits (a rounded shift can be at most 15)" % c)
        self.emit("LDI", R1, c >> 8 & 0xFF)
        self.emit("LDI", R2, c & 0xFF)

    def stream(self, n):
        """n multiply-adds D[P0] x D[P1], n = 1..510 (SPEC B.2: one or two stream instructions)."""
        if n > 255:
            self.emit("MAC", 255)
            n -= 255
        self.emit("MAC", 0 if n == 1 else n)                # the classic MAC does one multiply-add in 4 cycles

    @staticmethod
    def rewound(address, n):
        """A pointer that is rewound with SETP must stay in its page while it reads n bytes from `address`."""
        assert address % PAGE + n < PAGE, "a rewound vector ends at or after the end of its page"
        return address % PAGE

    # ---- the primitives of SPEC Part B, one loop each ----

    def rows(self, w, x, y, rows, n, sh, c=None, relu=0, lut=None, lut_tag=None):
        """Linear rows: y[j] = rq(c + sum_k W[j][k] x[k], sh, relu), then the optional table, for j < rows.
        W is stored row after row at w. c = None: the rounding constant. Per row: n + 24 cycles (n + 28 with table)."""
        low = self.rewound(x, n)
        self.preload(half(sh) if c is None else c)
        self.point(1, w)
        self.point2(y)
        self.page_of(0, x)

        def body():
            self.emit("SETP0", low)
            self.emit("LDA", R1, R2)
            self.stream(n)
            self.emit("RQ", R0, sh, relu)
            if lut is not None:
                self.emit("LUT", R0, lut, tag=lut_tag)
            self.emit("ST2", R0, 1)
        self.loop(rows, body, "row")
        self.at(0, x + n)
        self.at(1, w + rows * n)
        self.at(2, y + rows)

    def residual(self, w, h, x, rows, n):
        """Residual rows, in place: x[j] <- rq(256 x[j] + 128 + sum_k W[j][k] h[k], 8). Per row: n + 28 cycles."""
        low = self.rewound(h, n)
        self.emit("LDI", R2, 128)
        self.point(1, w)
        self.point2(x)
        self.page_of(0, h)

        def body():
            self.emit("LD2", R1, 0)                         # R1 <- x[j]; LDA R1, R2 then gives 256 x[j] + 128
            self.emit("LDA", R1, R2)
            self.emit("SETP0", low)
            self.stream(n)
            self.emit("RQ", R0, 8, 0)
            self.emit("ST2", R0, 1)
        self.loop(rows, body, "row")
        self.at(0, h + n)
        self.at(1, w + rows * n)
        self.at(2, x + rows)

    def table_map(self, x, y, n, t):
        """y[k] = lut(t, x[k]). Per element: 16 cycles."""
        self.point2(y)
        self.point(0, x)

        def body():
            self.emit("LD", R0, 0, 1)
            self.emit("LUT", R0, t)
            self.emit("ST2", R0, 1)
        self.loop(n, body, "map")
        self.at(0, x + n)
        self.at(2, y + n)

    def product(self, a, b, y, n, sh):
        """y[k] = rqr(a[k] * b[k], sh). Per element: 20 cycles."""
        self.preload(half(sh))
        self.point(1, b)
        self.point2(y)
        self.point(0, a)

        def body():
            self.emit("LDA", R1, R2)
            self.emit("MAC", 0)
            self.emit("RQ", R0, sh, 0)
            self.emit("ST2", R0, 1)
        self.loop(n, body, "elem")
        self.at(0, a + n)
        self.at(1, b + n)
        self.at(2, y + n)

    def rms(self, x, y, n, t, sa, sb, cell):
        """Norm `rms` of SPEC C.2: m = rq(sum x^2, sa); r = lut(t, m); y[k] = rqr(x[k] * r, sb).
        `cell` is a scratch byte that holds r. Per element: 24 cycles."""
        low = self.rewound(cell, 1)
        self.preload(0)
        self.point(0, x)
        self.point(1, x)                                    # both ports read the same memory: x times x
        self.emit("LDA", R1, R2)
        self.stream(n)
        self.emit("RQ", R0, sa, 0)
        self.emit("LUT", R0, t)
        self.at(0, x + n)
        self.at(1, x + n)
        self.point(0, cell, tmp=R1)                         # R0 holds r: a page change must not use it
        self.emit("ST", R0, 0, 0)
        self.preload(half(sb))
        self.page_of(1, cell)
        self.point2(y)
        self.point(0, x)

        def body():
            self.emit("SETP1", low)
            self.emit("LDA", R1, R2)
            self.emit("MAC", 0)
            self.emit("RQ", R0, sb, 0)
            self.emit("ST2", R0, 1)
        self.loop(n, body, "elem")
        self.at(0, x + n)
        self.at(1, cell + 1)
        self.at(2, y + n)

    def rope(self, head, d, table, out=None, fields=None):
        """RoPE of one head (SPEC C.4). head: [a0, b0, a1, b1, ...]; table: [C0, -S0, C1, -S1, ...] followed by
        [S0, C0, S1, C1, ...]. Writes [a'0, a'1, ..., b'0, b'1, ...] to `out`, or through P2 as it is (out = None).
        fields: the head is somewhere in the page of `head`; bits 11..8 and the low byte of its address are the
        two bytes at `fields`. Per rotated value: 22 cycles."""
        self.rewound(table, 2 * d)
        self.preload(64)
        self.point(1, table)
        if out is not None:
            self.point2(out)

        def body():
            self.emit("LDA", R1, R2)
            self.emit("MAC", 2)
            self.emit("RQ", R0, 7, 0)
            self.emit("ST2", R0, 1)
        for part in range(2):                               # first all a', then all b': P0 reads the head twice
            if fields is None:
                self.point(0, head)
            else:
                self.point(0, fields)
                self.emit("LD", R0, 0, 1)
                self.emit("LD", R3, 0, 0)
                self.emit("MVP", 0, R0, R3)
                self.page[0] = fields >> 12
                self.page_of(0, head)
            self.loop(d // 2, body, "elem")
            self.at(0, head + d)
            if fields is not None:
                self.low[0] = None
        self.at(1, table + 2 * d)
        if out is not None:
            self.at(2, out + d)

    def key_pointer(self, fields, page):
        """P2 <- the slot of the current position in a key block whose first page is `page`.
        fields: address of the three record bytes [bits 11..8, low byte, pages to add]."""
        self.point(0, fields)
        self.emit("LD", R1, 0, 1)
        self.emit("LD", R2, 0, 1)
        self.emit("LD", R0, 0, 1)
        self.at(0, fields + 3)
        self.emit("MVP", 2, R1, R2)
        self.emit("LDI", R1, page)
        self.emit("ADD", R0, R1)
        self.emit("MVH", 2, R0)

    def scatter(self, v, chunks, pos):
        """Cache write of the values (SPEC C.5): v[k] -> byte p of its row. chunks: (page, rows) in the order of v;
        the row of the value read when R3 = r starts at 256 * (r mod 16) in that page. pos: address of the byte p.
        Per value: 16 cycles."""
        self.point(0, pos)
        self.emit("LD", R2, 0, 0)
        self.point(0, v)

        def body():
            self.emit("LD", R0, 0, 1)
            self.emit("MVP", 2, R3, R2)                     # P2 <- row R3 of the page, byte p
            self.emit("ST2", R0, 0)
        for page, rows in chunks:
            self.emit("LDI", R1, page)
            self.emit("MVH", 2, R1)
            self.loop(rows, body)
        self.at(0, v + sum(rows for page, rows in chunks))

    def scores(self, fields, q, e, d, sh, count, relu, exp):
        """Attention weights of one query head (SPEC C.6, steps 1 to 3) for the keys t = 0 .. p:
        code[t] = rq(c + sum_j q[j] K[t][j], sh, relu); m = the largest code; e[t] = lut(exp, code[t] - m).
        fields: address of four bytes [bits 19..12, bits 11..8, low byte of the address of K[0], high byte of c];
        the low byte of c is the byte before q. count: address of the byte (number of keys) mod 256. relu: the table max(v, 0).
        Afterwards R2 = m. Per key: d + 40 cycles for the code and the maximum, then 20 for the weight."""
        low = self.rewound(q - 1, d + 1)
        self.point(0, fields)
        self.emit("LD", R0, 0, 1)
        self.emit("MVH", 1, R0)
        self.emit("LD", R0, 0, 1)
        self.emit("LD", R1, 0, 1)
        self.emit("MVP", 1, R0, R1)                         # P1 <- the keys
        self.emit("LD", R1, 0, 1)                           # R1 = high byte of c, for every LDA below
        self.at(0, fields + 4)
        self.point(0, count)
        self.emit("LD", R3, 0, 0)
        self.emit("LDI", R2, 0)                             # the maximum so far
        self.point2(e)
        self.page_of(0, q)

        def code():
            self.emit("SETP0", low)
            self.emit("LD", R0, 0, 1)                       # low byte of c; P0 is now at q
            self.emit("LDA", R1, R0)
            self.stream(d)
            self.emit("RQ", R0, sh, 1)
            self.emit("ST2", R0, 1)
            self.emit("SUB", R0, R2)                        # code - maximum, in -127..127
            self.emit("LUT", R0, relu)                      # max(difference, 0)
            self.emit("ADD", R2, R0)                        # maximum <- max(maximum, code), without a branch
        self.loop(None, code, "row")
        self.at(0, q + d)
        self.point(0, count)
        self.emit("LD", R3, 0, 0)
        self.point(0, e)

        def weight():                                       # in place: the code is read and the weight written through P0
            self.emit("LD", R0, 0, 0)
            self.emit("SUB", R0, R2)
            self.emit("LUT", R0, exp)
            self.emit("ST", R0, 0, 1)
        self.loop(None, weight, "map")
        self.page[0] = e >> 12                              # P0 = e + p + 1: same page, low bits not known

    def mix(self, fields, e, o, ones, chunks, sh, recip, full):
        """Output of one query head (SPEC C.6, steps 4 and 5): Dsum = sum_t e[t]; k = bitlen(Dsum);
        r = lut(recip, Dsum >> (k - 7)); o[j] = rqr(rq(sum_t e[t] V[j][t], k) * r, sh) for j < d.
        LEN holds the number of keys (p + 1, or at most the window). ones: address of that many bytes that hold 1. fields: address of three bytes [first page of the
        value rows, bits 11..8 and low byte of the address of the output]; o: any address in the page of the output.
        chunks: (page, rows) as in `scatter`, pages counted from the first. k, r and the page of the rows are kept in
        the three bytes after the weights, e[p + 1 ..]: that is where P0 stands after every stream over e.
        full: a row can end at the end of its page (T = 256); the page of P1 is then set again for every row of a
        chunk that uses the last slot. Per row: p + 57 cycles (p + 65 in such a chunk)."""
        low = self.rewound(e, 259)
        c = half(sh)
        if c >= 1 << 15:
            raise ValueError("the rounding constant of sh_o = %d does not fit 16 bits (at most 15)" % sh)
        self.point(0, fields)
        self.emit("LD", R3, 0, 1)                           # R3 = first page of the value rows
        self.emit("LD", R0, 0, 1)
        self.emit("LD", R1, 0, 1)
        self.emit("MVP", 2, R0, R1)                         # P2 <- the output
        self.at(0, fields + 3)
        self.page_of(2, o)
        self.point(0, e)
        self.point(1, ones)
        self.emit("LDI", R1, 0)                             # R1 stays 0: the preload 0 and byte 0 of a value row
        self.emit("LDA", R1, R1)
        self.emit("MACL")                                   # Dsum: the weights times a row of ones
        self.emit("NRM", R0)
        self.emit("ST", R0, 0, 1)                           # k
        self.emit("LDI", R2, 7)
        self.emit("SUB", R0, R2)
        self.emit("RQV", R0, R0, 0)                         # Dsum >> (k - 7), in 64..127
        self.emit("LUT", R0, recip)
        self.emit("ST", R0, 0, 1)                           # r
        self.emit("ST", R3, 0, 0)                           # page of the rows; P0 stays here
        self.emit("MVH", 1, R3)
        for page, rows in chunks:
            again = full and rows >= 15
            if page:                                        # the next page of value rows
                self.emit("LD", R0, 0, 0)
                self.emit("LDI", R2, 1)
                self.emit("ADD", R0, R2)
                self.emit("ST", R0, 0, 0)
                self.emit("MVH", 1, R0)

            def body():
                if again:
                    self.emit("LD", R0, 0, 0)
                    self.emit("MVH", 1, R0)
                self.emit("MVP", 1, R3, R1)                 # P1 <- row R3 of the page, byte 0
                self.emit("SETP0", low)
                self.emit("LDA", R1, R1)
                self.emit("MACL")                           # A[j]; P0 is now at k
                self.emit("LD", R0, 0, 1)
                self.emit("RQV", R2, R0, 0)                 # a = rq(A[j], k)
                self.emit("LD", R0, 0, 1)
                self.emit("MUL", R2, R0)                    # a * r
                self.emit("LDI", R0, c if c < 256 else c >> 8)
                self.emit("LDA", *((R1, R0) if c < 256 else (R0, R1)))      # ACC <- the rounding constant
                self.emit("ADDP")
                self.emit("RQ", R0, sh, 0)
                self.emit("ST2", R0, 1)
            self.loop(rows, body, "mix")
        self.page[0] = e >> 12                              # P0 = e + p + 3: same page, low bits not known

    def for_records(self, first, cell, size, body):
        """Run body() once for every record of a chain that starts at `first`. A record is `size` bytes; its first
        three hold the address of the next record as [bits 11..8, low byte, bits 19..12], with bits 19..12 = 0 in
        the last one. Every record is copied to `cell` before body() runs: this is how one piece of program serves
        every head of a layer without call and return. Per record: 12 cycles per byte and 14 instructions."""
        self.point(1, first)
        top = len(self.code)
        self.page, self.low = [None] * 3, [None] * 3        # entered from above and from the JMP: nothing is known
        self.point2(cell)

        def copy():
            self.emit("LD", R0, 1, 1)
            self.emit("ST2", R0, 1)
        self.loop(size, copy)
        self.at(2, cell + size)
        body()
        self.point(0, cell)
        self.emit("LD", R1, 0, 1)
        self.emit("LD", R2, 0, 1)
        self.emit("LD", R0, 0, 1)
        self.emit("MOV", R0, R0)                            # Z if this was the last record
        done = self.branch("BZ")
        self.emit("MVP", 1, R1, R2)
        self.emit("MVH", 1, R0)
        self.emit("JMP", top)
        self.land(done)
        self.page, self.low = [None] * 3, [None] * 3

    def sample(self, u, n, gumbel):
        """Gumbel-max sampling (SPEC Part D), in place: u[j] <- (u[j] >> 1) + lut(gumbel, RND) for every j < n in
        the order of j (one RND per token id). The argmax then runs on these candidates. Per logit: 28 cycles."""
        self.point(0, u)

        def body():
            self.emit("LD", R0, 0, 0)
            self.emit("SHR", R0)
            self.emit("RND", R1)
            self.emit("LUT", R1, gumbel)
            self.emit("ADD", R0, R1)
            self.emit("ST", R0, 0, 1)
        self.loop(n, body)
        self.at(0, u + n)

    def argmax(self, u, n, hi, table, cls=None, allowed=None):
        """Greedy choice (SPEC C.9): the smallest j with the largest u[j], u in 0..127. Afterwards the byte at `hi`
        is j >> 8 and R2 = (-j) mod 256; R1 is the largest value. table: the RELU table, max(v, 0).
        Per logit: 20 cycles, and 16 more (24 in a last block of fewer than 256) when a new maximum is found.
        With the syntax filter: cls = address of the class byte of every token id, allowed = number of the table
        that is 0 for the classes that may not come; `table` is then v + 1 for v >= 0 and 0 for v < 0, R1 is the
        largest value + 1 and starts at 0, so the first allowed token is taken whatever its value. A token that
        is not allowed costs 20 cycles, an allowed one 32 (tag "filter": 12 and 12 of them)."""
        if cls is not None:
            self.point(1, cls)
        self.point2(hi)
        self.emit("LDI", R0, 0)
        self.emit("ST2", R0, 0)
        self.emit("LDI", R1, 0)                             # the maximum so far
        self.emit("LDI", R2, 0)                             # R3 at the time it was found
        self.point(0, u)
        for block in range(0, n, 256):
            size = min(256, n - block)
            self.emit("LDI", R3, size % 256)
            first = len(self.code)
            if cls is not None:
                self.emit("LD", R0, 1, 1, tag="filter")     # class of token j
                self.emit("LUT", R0, allowed, tag="filter")
                forbidden = self.emit("BZ", 0, tag="filter")
            self.emit("LD", R0, 0, 1)
            self.emit("SUB", R0, R1)                        # candidate - maximum, in -127..127
            self.emit("LUT", R0, table)                     # max(difference, 0); Z if the candidate is not larger
            skip = self.branch("BZ")
            self.emit("ADD", R1, R0)                        # maximum <- candidate
            self.emit("MOV", R2, R3)
            if size < 256:                                  # make R2 what it would be in a block of 256
                self.emit("LDI", R0, 256 - size)
                self.emit("ADD", R2, R0)
            self.emit("LDI", R0, block >> 8)
            self.emit("ST2", R0, 0)
            self.land(skip)
            self.emit("DJNZ", first)
            if cls is not None:                             # a token that may not come: step over its logit
                done = self.emit("JMP", 0, tag="filter")
                self.land(forbidden)
                self.emit("LD", R0, 0, 1, tag="filter")
                self.emit("DJNZ", first, tag="filter")
                self.land(done)
        self.at(0, u + n)
        if cls is not None:
            self.at(1, cls + n)


def kv_layout(d, T):
    """Shape of the cache block of one key/value head: (chunks, offset of the keys, bytes from block to block).

    The d rows V[j] are visited in loops of at most 16 passes; the row visited when R3 = r is slot r mod 16 of
    the page of its chunk (a slot is 256 bytes). chunks = [(page inside the block, rows)]. With d < 16 the slots
    1..d are used, and the keys start right after them and may end in the unused slot 0 of the next block.
    """
    chunks = [(c, min(16, d - 16 * c)) for c in range(-(-d // 16))]
    spare = 256 if d < 16 else 0
    keys = 256 * (d + 1) if d < 16 else PAGE * len(chunks)
    return chunks, keys, PAGE * -(-(keys + T * d - spare) // PAGE)


def records(cfg, base, key_low, rope_c, rope_s):
    """The position records, T rows of 9 + 2 d bytes. Row p:
        0..2   address of row p + 1 as [bits 11..8, low byte, bits 19..12]; page 0 in the last row = context full
        3, 4   p + 1 as [low byte, high byte]: the number of keys, the stream length LEN
        5      p
        6..8   key_low + p * d as [bits 11..8, low byte, pages]: where the key of this position goes in a key block
        9..    RoPE of this position as the program reads it: [C0, -S0, C1, -S1, ...] then [S0, C0, S1, C1, ...]
    With an attention window W (cfg.window > 0) the cache is a ring of W slots and a row has one more byte:
        3, 4   min(p + 1, W): the number of keys in the ring
        6..8   key_low + (p mod W) * d: the key goes to slot p mod W
        last   p mod W: the byte of every value row that this position writes
    """
    T, d, ring = cfg.max_seq_len, cfg.head_dim, cfg.window or cfg.max_seq_len
    rows = np.zeros((T, 9 + 2 * d + bool(cfg.window)), dtype=np.int64)
    for p in range(T):
        link = base + (p + 1) * rows.shape[1] if p + 1 < T else 0
        slot, count = p % ring, min(p + 1, ring)
        key = key_low + slot * d
        rows[p, :9] = (link >> 8 & 15, link & 255, link >> 12, count & 255, count >> 8, p,
                       key >> 8 & 15, key & 255, key >> 12)
        rows[p, 9:9 + d] = np.stack([rope_c[p], -rope_s[p]], 1).ravel()
        rows[p, 9 + d:9 + 2 * d] = np.stack([rope_s[p], rope_c[p]], 1).ravel()
        if cfg.window:
            rows[p, -1] = slot
    return rows


def head_records(base, q, o, d, keys, key_off, c_s):
    """The head records, HEAD bytes per layer and query head, layer after layer. keys[l][h]: address of the key
    block that head h of layer l reads; key_off: offset of the keys in the cache block of a key/value head.
        0..2    address of the next record of the layer as [bits 11..8, low byte, bits 19..12]; page 0 in the last
        3, 4    address of the head in q as [bits 11..8, low byte]
        5..7    address of its keys as [bits 19..12, bits 11..8, low byte]
        8       high byte of c_s
        9       first page of its value rows
        10, 11  address of its output in o as [bits 11..8, low byte]
        12      low byte of c_s (the byte before the rotated query, where the score loop reads it)
    """
    rows = []
    for l, layer in enumerate(keys):
        for h, K in enumerate(layer):
            link = base + (len(rows) + 1) * HEAD if h + 1 < len(layer) else 0
            c, at, out = c_s[l][h], q + h * d, o + h * d
            rows.append((link >> 8 & 15, link & 255, link >> 12, at >> 8 & 15, at & 255, K >> 12, K >> 8 & 15, K & 255,
                         c >> 8 & 255, (K - key_off) >> 12, out >> 8 & 15, out & 255, c & 255))
    return np.array(rows)


class Build(NamedTuple):
    code: list          # per program address: (mnemonic, operand values, component)
    loops: list         # (first address, address of the DJNZ, kind) of the marked loops (see Emitter.loops)
    image: np.ndarray   # the data memory at reset, int8
    mem: dict           # the memory map: name -> (address, bytes)
    words: list         # the program as 16-bit words


def load(directory):
    """(spec, arrays) of an exported integer model directory (SPEC Part E)."""
    with open(os.path.join(directory, "model.json")) as f:
        spec = json.load(f)
    with np.load(os.path.join(directory, "model.npz")) as arrays:
        return spec, {name: arrays[name] for name in arrays.files}


def build(spec, arrays, prompt=False, tf=None, syntax_filter=False, temperature=0.0, prompts=None, uart_prompt=False):
    """spec: the content of model.json; arrays: name -> int8 array of model.npz. Returns a Build.

    Without options the program computes one token per pass (the test bench writes the current token).
    prompt: the complete program. It feeds the prompt buffer (`set_prompt`) one position at a time; a prompt
        token that is not the last one only fills the KV cache (no final norm, output layer, argmax or UART);
        then it generates until the stop rule fires or the context is full, and halts.
    tf: a lsllm.filter_table.TableFilter (the tables of the tokenizer). With `prompt` it gives the stop rule of
        lsllm.intref.completer: halt after sending the end-of-text token or a line break that brings the
        indentation level to 0.
    syntax_filter: only tokens that `tf` allows are chosen (docs/FILTER.md).
    temperature: > 0 gives Gumbel-max sampling (SPEC Part D); 0 is greedy: no RND, no table.
    prompts: a list of 1 to 64 prompts (token-id lists of 1 to min(T, 255) tokens) for the complete program. A
        start-up piece in front of the program (tag "start", run once after reset) reads the switches with IN,
        takes SW5..0 as the prompt number (a number beyond the list gives prompt 0) and copies that prompt into
        the prompt buffer. `set_prompts` writes other prompts into the same table.
    uart_prompt: the complete program takes its prompts from the UART (needs the proposed instruction RX). The
        start-up piece resets position, filter state and stop state, then reads token ids as two bytes each
        (0x80 | (id >> 7), id & 0x7F) up to the byte END into the prompt buffer (tokens beyond min(T, 255) are
        read and dropped; an empty prompt is ignored). After every answer the program sends END and jumps to
        the start-up piece instead of halting. `top(b)` is the address where the per-token program begins."""
    if prompts is not None and uart_prompt:
        raise ValueError("prompts from the switches and from the UART are two different programs")
    prompt = prompt or prompts is not None or uart_prompt
    cfg, opt, sh, pre = ModelConfig(**spec["config"]), spec["options"], spec["shifts"], spec["preloads"]
    D, L, H, G, F = cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.hidden_dim
    V, T, d, share = cfg.vocab_size, cfg.max_seq_len, cfg.head_dim, cfg.n_heads // cfg.n_kv_heads
    swiglu = opt["ffn"] == "swiglu"
    if T > 256 or V > 4096 or d % 2 or max(D, F) > 510 or L > 20:
        raise ValueError("needs T <= 256, V <= 4096, even head size, D and F <= 510, L <= 20 (64 tables)")
    if not 0 <= cfg.window <= T:
        raise ValueError("the attention window is %d, the context %d" % (cfg.window, T))
    ring = cfg.window or T                                       # positions that the KV cache holds
    if "RECIP" not in arrays or arrays["EXP"].shape != (L, 256) or (arrays["EXP"][:, 0] != 127).any():
        raise ValueError("not a model of SPEC v0.4: C.6 needs RECIP and one EXP table per layer with EXP[l][0] = 127")
    EXP, RELU, RECIP, SILU = 2 * L + 1, 3 * L + 1, 3 * L + 2, 3 * L + 3      # table numbers; NORM of site n is table n,
    tables = 3 * L + 3 + swiglu                                              # EXP of layer l is table EXP + l
    if syntax_filter and tf is None:
        raise ValueError("the syntax filter needs the tables of the tokenizer (tf)")
    sh_e = sh.get("sh_e", 0)                                     # SPEC C.1: x = rqr(E[token], sh_e); 0 = plain copy
    if not 0 <= sh_e <= 15:
        raise ValueError("the embedding shift sh_e is %d, allowed is 0..15" % sh_e)
    extra = ["EMB"] * (sh_e > 0) + ["GUMBEL"] * (temperature > 0) + ["LROW", "STOPC"] * (tf is not None)
    extra += ["CC", "BR", "LEXO", "LEX2", "LEXA", "ALO", "AHI", "ALLOWED", "RELU1"] * bool(syntax_filter)
    X = {name: tables + g for g, name in enumerate(extra)}                   # table numbers of the options
    tables += len(extra)
    if tables > 64:
        raise ValueError("%d lookup tables, LUT has 64" % tables)

    # ---- memory map ----
    mem, top = {}, 0

    def alloc(name, size, align=1, rewound=True):
        """The next free address for `size` bytes. A vector that a pointer rewinds must end inside its page."""
        nonlocal top
        address = -(-top // align) * align
        if rewound and address % PAGE + size >= PAGE:
            address = -(-address // PAGE) * PAGE
        mem[name], top = (address, size), address + size
        return address

    alloc("tables", tables * 256, rewound=False)
    span = 256 * -(-V // 256)                                    # bytes per column of the token table
    if 3 * span > PAGE:
        span = 1 << (span - 1).bit_length()                      # a power of two: no column crosses a page
    tok = alloc("tok.addr", 3 * span, PAGE, rewound=False)
    size = 9 + 2 * d + bool(cfg.window)                          # bytes of a position record (see `records`)
    rec = alloc("scratch.rec", size)                             # the record of the current position
    count, pos, key_fields, rope = rec + 3, rec + 5, rec + 6, rec + 9
    slot = rec + size - 1 if cfg.window else pos                 # the byte of a value row that this position writes
    token = alloc("scratch.token", 2)                            # token id as [high byte, low byte]
    cell = alloc("scratch.r", 1)                                 # the multiplier of an rms norm
    x, y = alloc("scratch.x", D), alloc("scratch.y", D)
    q, k, v = alloc("scratch.q", D), alloc("scratch.k", G * d), alloc("scratch.v", G * d)
    o = alloc("scratch.o", D)                                    # attention output
    hrec = alloc("scratch.head", HEAD + d)                       # the record of the current head, then its rotated
    qr = hrec + HEAD                                             # query: the byte before it is the low byte of c_s
    e = alloc("scratch.e", 259)                                  # attention weights of one head, then k, r, a page
    g1, h3 = (alloc("scratch.g1", F), alloc("scratch.h3", F)) if swiglu else (0, 0)
    g = alloc("scratch.g", F)
    u = alloc("scratch.u", V, rewound=V < PAGE)                  # the logits
    if prompt:
        left = alloc("prompt.left", 1)                           # prompt tokens that are still to be fed
        prm = alloc("prompt.ids", 512, 256)                      # byte p: high byte of prompt token p; byte 256 + p: low
    if tf is not None:
        cls = alloc("filter.class", span, 256)                   # class of every token id
        ft = alloc("filter.rows", PAGE, PAGE, rewound=False)     # 16 rows of 256 bytes, read as [row][byte]
        cells = ft + 256 * ROWS         # lvl (level of the stop rule), lex, level, word column (2), word, colon, depth
    if syntax_filter:
        khi, klo = alloc("filter.khi", span, 256), alloc("filter.klo", span, 256)
        wtab = alloc("filter.word", PAGE, PAGE, rewound=False)   # byte 64 k + w: word state after column k in state w
        allow = alloc("filter.allow", 17 * 32)                   # byte 32 a + c: may class c come in abstract state a
    ones = alloc("ones", T, rewound=False)
    chunks, key_off, stride = kv_layout(d, ring)
    recs = alloc("pos.records", T * size, rewound=False)
    heads = alloc("heads", L * H * HEAD, rewound=False)
    kv = -(-top // PAGE) * PAGE
    for l in range(L):
        for kvh in range(G):
            block = kv + (l * G + kvh) * stride
            for j in range(d):
                page, rows = chunks[j // 16]
                mem["kv.V%d.%d.%d" % (l, kvh, j)] = (block + page * PAGE + 256 * ((rows - j % 16) % 16), ring)
            mem["kv.K%d.%d" % (l, kvh)] = (block + key_off, ring * d)
    top = block + key_off + ring * d
    names = ["Wq", "Wk", "Wv", "Wo", "W1"] + ["W3"] * swiglu + ["W2"]
    W = {"%s_%d" % (name, l): 0 for l in range(L) for name in names}
    for name in list(W) + ["E"]:                                 # in the order the program reads them
        W[name] = alloc("w." + name, arrays[name].size, rewound=False)
    if prompts is not None:                 # last in memory, so that nothing else moves. Index: 4 bytes per switch
        # number n: address of its prompt as [bits 11..8, low byte, bits 19..12], then its number of tokens
        pidx = alloc("prompts.index", 256, 256)
        alloc("prompts.table", sum(2 * len(ids) for ids in prompts), rewound=False)     # per prompt: high bytes, low bytes

    if top > 1 << 19:
        raise ValueError("the data memory would need %d bytes, the machine has 2^19" % top)
    image = np.zeros(top, dtype=np.int8)

    def put(address, values):
        values = np.asarray(values, dtype=np.int64).ravel()
        image[address:address + len(values)] = values.astype(np.uint8).view(np.int8)

    put(256 * 0, arrays["NORM"])
    put(256 * EXP, arrays["EXP"])
    put(256 * RELU, np.where(np.arange(256) < 128, np.arange(256), 0))
    put(256 * RECIP, arrays["RECIP"])
    if swiglu:
        put(256 * SILU, arrays["SILU"])
    rows_of_e = W["E"] + D * np.arange(V)
    for column, part in enumerate((rows_of_e & 255, rows_of_e >> 8 & 15, rows_of_e >> 12)):
        put(tok + column * span, part)
    put(ones, np.ones(T))
    put(recs, records(cfg, recs, key_off % PAGE, arrays["ROPE_C"], arrays["ROPE_S"]))
    put(rec, [recs >> 8 & 15, recs & 255, recs >> 12])           # before position 0: the link to record 0
    keys = [[mem["kv.K%d.%d" % (l, h // share)][0] for h in range(H)] for l in range(L)]
    put(heads, head_records(heads, q, o, d, keys, key_off, pre["c_s"]))
    for name, address in W.items():
        put(address, arrays[name])
    if sh_e:                                # table EMB: rqr(v, sh_e) for every int8 value v, at the index of its byte
        value = np.arange(256) - 256 * (np.arange(256) > 127)
        put(256 * X["EMB"], np.clip((value + half(sh_e)) >> sh_e, -128, 127))
    if temperature > 0:
        from lsllm.intref.model import gumbel_table
        put(256 * X["GUMBEL"], gumbel_table(temperature, spec["frac"]["u"]))
    if tf is not None:
        from lsllm import filter_table as F_
        table = np.frombuffer(tf.token_table(), dtype=np.uint8).astype(np.int64)
        if len(table) != 2 * V:
            raise ValueError("the filter tables are for %d tokens, the model has %d" % (len(table) // 2, V))
        c_of, k_of, C = table[:V], table[V:], np.arange(32)
        line = (C >= F_.NL) & (C < F_.DOCLINE)                   # the line breaks <NL>, <IND>, <DED1..5>
        put(cls, c_of)
        put(256 * X["LROW"], np.where(line, C - F_.NL, 0))       # row of the level table: 0 leaves the level as it is
        put(256 * X["STOPC"], np.where(C == F_.END, 1, np.where(line & (C != F_.IND), 2, 0)))
        for r, delta in enumerate((0, 1, -1, -2, -3, -4, -5)):
            put(ft + 256 * r, np.clip(np.arange(256) + delta, 0, 255))
    if syntax_filter:
        control = C <= F_.END
        put(256 * X["CC"], np.where(control | (C == F_.SPACE), 0, np.where((C == F_.COLON) | (C == F_.CLOSE_COLON), 2, 1)))
        put(256 * X["BR"], np.where((C >= F_.OPEN) & (C < F_.CLOSE), 9 + C - F_.OPEN,
                                    np.where((C >= F_.CLOSE) & (C <= F_.CLOSE_COLON), POP, 0)))
        lex = np.frombuffer(tf.lexer_table(), dtype=np.uint8).reshape(-1, F_.ROW)[:, :F_.N_CLASS]      # [state][class]
        states = len(lex)
        columns = sorted(set(map(tuple, lex.T)))                 # classes that act alike share a column
        columns.append(tuple(range(states)))                     # and one for control tokens: nothing changes
        word = np.frombuffer(tf.word_table(), dtype=np.uint8).reshape(-1, F_.WORD_ROW)                 # [column][state]
        if len(columns) > 8 or states > 32 or len(word) > 62 or F_.MAX_DEPTH > 8 or F_.N_ABSTRACT != 17:
            raise ValueError("the filter tables do not fit the layout of the program")
        lexo = [32 * (len(columns) - 1 if c <= F_.END or c >= F_.N_CLASS else columns.index(tuple(lex[:, c]))) for c in C]
        put(256 * X["LEXO"], lexo)
        put(256 * X["LEX2"], np.array([column + (0,) * (32 - states) for column in columns]))
        put(256 * X["LEXA"], np.where(np.arange(256) >= 9, 2, np.where(np.arange(256) >= 3, 1, 0)))
        put(256 * X["RELU1"], np.where(np.arange(256) < 128, np.arange(256) + 1, 0))
        put(wtab, word)
        put(wtab + 64 * 62, np.full(64, F_.LINE_START))          # column 62: a line break; column 63: a control token
        put(wtab + 64 * 63, np.arange(64))
        column = np.where(line[c_of], 62, np.where(control[c_of], 63, k_of))
        put(khi, column >> 2)
        put(klo, (column & 3) << 6)
        put(ft + 256 * 7, 12 + np.minimum(np.arange(256), 5))    # row 7: abstract state + 1 of a line that may end
        byte = np.arange(256)
        for kind in range(3):                                    # rows 9..11: what an opening bracket puts on the stack
            put(ft + 256 * (9 + kind), np.where(byte < F_.MAX_DEPTH, 3 + 2 * kind + (byte + 1 == F_.MAX_DEPTH), 0))
        for colon in range(2):                                   # rows 12, 13: abstract state + 1 from the word, 0 = by level
            put(ft + 256 * (12 + colon), np.where(byte == F_.LINE_START, 9, np.where((byte < F_.OTHER) != colon, 10, 11 * colon)))
        put(cells + 5, F_.LINE_START)                            # row 14 is the bracket stack; row 15 the state
        rows = np.frombuffer(tf.allow_table(), dtype=np.uint8).reshape(17, F_.ROW).astype(np.int64)
        rows[~rows[:, c_of].any(1)] = 1                          # nothing allowed: the choice is unfiltered
        put(allow, rows)
        put(256 * X["ALO"], [0] + [(allow + 32 * a) & 255 for a in range(17)])
        put(256 * X["AHI"], [0] + [(allow + 32 * a) >> 8 & 15 for a in range(17)])

    # ---- program ----
    em = Emitter()
    halts = []                                                   # branches to the HALT at the end
    if prompts is not None:                 # start-up: the prompt chosen by the switches -> the prompt buffer
        em.tag = "start"
        em.emit("IN", R0)
        em.emit("SHL", R0)
        em.emit("SHL", R0)                                       # 4 x SW5..0: the two upper switches drop out
        em.page_of(0, pidx, tmp=R1)
        em.emit("LDI", R1, pidx >> 8 & 15)
        em.emit("MVP", 0, R1, R0)
        em.emit("LD", R1, 0, 1)
        em.emit("LD", R2, 0, 1)
        em.emit("LD", R3, 0, 1)
        em.emit("LD", R0, 0, 0)                                  # R0 = number of tokens, until the end of the start-up
        em.emit("MVP", 1, R1, R2)
        em.emit("MVH", 1, R3)                                    # P1 <- the prompt in the table
        em.point(0, left, tmp=R1)
        em.emit("ST", R0, 0, 0)

        def move():
            em.emit("LD", R1, 1, 1)
            em.emit("ST2", R1, 1)
        for column in range(2):                                  # the high bytes, then the low bytes
            em.point(0, prm + 256 * column, tmp=R1)
            em.emit("CPP20")
            em.emit("MOV", R3, R0)
            em.loop(None, move)
    if uart_prompt:                         # start-up, after reset and after every answer: a prompt from the UART
        em.tag = "start"
        em.point(0, rec)                                         # position 0: the link to record 0
        for byte in (recs >> 8 & 15, recs & 255, recs >> 12):
            em.emit("LDI", R0, byte)
            em.emit("ST", R0, 0, 1)
        if tf is not None:                                       # stop level and filter state as in the image
            em.point(0, cells)
            em.emit("LDI", R0, 0)
            for _ in range(8):
                em.emit("ST", R0, 0, 1)
            if syntax_filter:
                em.emit("SETP0", (cells + 5) % PAGE)
                em.emit("LDI", R0, F_.LINE_START)
                em.emit("ST", R0, 0, 0)
        em.point(0, prm)
        em.emit("CPP20")                                         # P2 walks the high bytes, P0 the low bytes
        em.point(0, prm + 256)
        em.emit("LDI", R3, 0)                                    # tokens stored
        again = em.emit("RX", R0)                                # 0x80 | (id >> 7), or END
        em.emit("LDI", R1, END)
        em.emit("XOR", R1, R0)
        ended = em.branch("BZ")
        em.emit("RX", R1)                                        # id & 0x7F
        em.emit("LDI", R2, 1)
        em.emit("AND", R2, R0)                                   # bit 7 of the id is bit 0 of the first byte
        clear = em.branch("BZ")
        em.emit("LDI", R2, 0x80)
        em.emit("OR", R1, R2)
        em.land(clear)
        em.emit("LDI", R2, 0x7F)
        em.emit("AND", R0, R2)
        em.emit("SHR", R0)                                       # id >> 8
        em.emit("LDI", R2, min(T, 255))
        em.emit("SUB", R2, R3)
        em.emit("BZ", again)                                     # the buffer is full: the token is dropped
        em.emit("ST2", R0, 1)
        em.emit("ST", R1, 0, 1)
        em.emit("LDI", R2, 1)
        em.emit("ADD", R3, R2)
        em.emit("JMP", again)
        em.land(ended)
        em.emit("MOV", R3, R3)
        em.emit("BZ", again)                                     # an empty prompt: wait for the next one
        em.forget()
        em.point(0, left)
        em.emit("ST", R3, 0, 0)
    begin = len(em.code)                                         # the per-token program begins here (0 without start-up)
    em.forget()
    em.tag = "book"

    def norm(site):
        em.tag = "norm"
        if opt["norm"] == "dyt":
            em.table_map(x, y, D, site)
        else:
            em.rms(x, y, D, site, sh["sa"][site], sh["sb"][site], cell)

    # the record of this position -> scratch.rec; LEN <- p + 1
    em.point(0, rec)
    em.emit("LD", R1, 0, 1)
    em.emit("LD", R2, 0, 1)
    em.emit("LD", R0, 0, 1)
    em.emit("MOV", R0, R0)                                       # Z if the page of the next record is 0
    stop = em.branch("BZ")                                       # the context is full: stop
    em.emit("MVP", 1, R1, R2)
    em.emit("MVH", 1, R0)
    em.at(0, rec + 3)
    em.point2(rec)

    def copy():
        em.emit("LD", R0, 1, 1)
        em.emit("ST2", R0, 1)
    em.loop(size, copy)
    em.point(0, count)
    em.emit("LD", R2, 0, 1)
    em.emit("LD", R1, 0, 1)
    em.emit("MVL", R1, R2)
    em.at(0, count + 2)

    if prompt:                              # a prompt token is left: it becomes the current token
        em.tag = "prompt"
        em.point(0, left)
        em.emit("LD", R0, 0, 0)
        em.emit("MOV", R0, R0)
        generated = em.branch("BZ")
        em.point2(token)
        em.point(0, pos)
        em.emit("LD", R2, 0, 0)                                  # p
        em.page_of(0, prm)
        for column in range(2):
            em.emit("LDI", R1, (prm >> 8 & 15) + column)
            em.emit("MVP", 0, R1, R2)
            em.emit("LD", R0, 0, 0)
            em.emit("ST2", R0, 1 - column)
        em.land(generated)
        em.forget()
    if tf is not None:
        def of_token(column, r):
            """r <- byte `token id` of a column; R1:R2 hold the token id."""
            em.page_of(0, column)
            em.emit("LDI", R0, column >> 8 & 15)
            em.emit("ADD", R0, R1)
            em.emit("MVP", 0, R0, R2)
            em.emit("LD", r, 0, 0)

        em.tag = "filter" if syntax_filter else "stop"
        if syntax_filter:
            em.point2(cells + 3)
        em.point(0, token)
        em.emit("LD", R1, 0, 1)
        em.emit("LD", R2, 0, 0)
        if syntax_filter:                                        # the word column of the token, as an address
            for column in (khi, klo):
                of_token(column, R0)
                em.emit("ST2", R0, 1)
        em.tag = "stop"
        of_token(cls, R3)                                        # R3 = class of the token, until the end of the step
        if prompt:
            em.point(0, left)
            em.emit("LD", R2, 0, 0)                              # not 0: a prompt token, the stop rule does not apply
        em.emit("MOV", R1, R3)
        em.emit("LUT", R1, X["LROW"])                            # R1 = row of the level table
        em.page_of(1, ft)
        em.point(0, cells)
        em.emit("LD", R0, 0, 0)
        em.emit("MVP", 1, R1, R0)
        em.emit("LD", R0, 1, 0)
        em.emit("ST", R0, 0, 1)                                  # level of the stop rule: max(0, level + change)
        em.at(0, cells + 1)
        if prompt:
            em.emit("MOV", R2, R2)
            fed = em.branch("BNZ")
            em.emit("MOV", R2, R3)
            em.emit("LUT", R2, X["STOPC"])                       # 1 end of text, 2 line break, 0 anything else
            no = em.branch("BZ")
            em.emit("SHR", R2)
            halts.append(em.branch("BZ"))                        # end of text
            em.emit("MOV", R0, R0)
            halts.append(em.branch("BZ"))                        # a line break back to level 0
            em.land(fed)
            em.land(no)
    if syntax_filter:                       # the step of docs/FILTER.md, section 5
        em.tag = "filter"
        em.emit("LD", R0, 0, 1)                                  # lex
        em.emit("LDI", R2, 2)
        em.emit("SUB", R0, R2)
        em.emit("LUT", R0, RELU)                                 # 0 in code (lex < 3)
        string = em.branch("BNZ")
        em.emit("LD", R0, 0, 0)
        em.emit("MVP", 1, R1, R0)
        em.emit("LD", R0, 1, 0)
        em.emit("ST", R0, 0, 1)                                  # level
        em.page_of(1, wtab)
        em.emit("LD", R1, 0, 1)
        em.emit("LD", R2, 0, 1)
        em.emit("LD", R0, 0, 0)
        em.emit("ADD", R2, R0)
        em.emit("MVP", 1, R1, R2)
        em.emit("LD", R0, 1, 0)
        em.emit("ST", R0, 0, 1)                                  # word
        em.emit("MOV", R1, R3)
        em.emit("LUT", R1, X["CC"])                              # 0 keep, 1 clear, 2 set
        keep = em.branch("BZ")
        em.emit("LDI", R2, 1)
        em.emit("SUB", R1, R2)
        em.emit("ST", R1, 0, 0)                                  # colon
        em.land(keep)
        em.emit("MOV", R1, R3)
        em.emit("LUT", R1, X["BR"])                              # 0, POP, or the row of an opening bracket
        done = [em.branch("BZ")]
        em.emit("SETP0", (cells + 7) % PAGE)
        em.emit("LD", R0, 0, 0)                                  # depth
        em.emit("LDI", R2, POP)
        em.emit("SUB", R2, R1)
        pop = em.branch("BZ")
        em.page_of(1, ft, tmp=R2)
        em.emit("MVP", 1, R1, R0)
        em.emit("LD", R2, 1, 0)                                  # what goes on the stack; 0: the stack is full
        em.emit("MOV", R2, R2)
        done.append(em.branch("BZ"))
        em.emit("LDI", R1, 1)
        em.emit("ADD", R0, R1)
        em.emit("ST", R0, 0, 0)
        em.emit("LDI", R1, ROWS - 1)
        em.emit("MVP", 2, R1, R0)                                # the stack is row 14: byte `depth`
        em.emit("ST2", R2, 0)
        done.append(em.branch("JMP"))
        em.land(pop)
        em.emit("MOV", R0, R0)
        done.append(em.branch("BZ"))
        em.emit("LDI", R1, 1)
        em.emit("SUB", R0, R1)
        em.emit("ST", R0, 0, 0)
        for address in done + [string]:
            em.land(address)
        em.forget(p0=ft >> 12)
        em.point(0, cells + 1)
        em.emit("LD", R0, 0, 0)
        em.emit("MOV", R1, R3)
        em.emit("LUT", R1, X["LEXO"])
        em.emit("ADD", R0, R1)
        em.emit("LUT", R0, X["LEX2"])
        em.emit("ST", R0, 0, 0)                                  # lex
    if tf is not None:
        em.forget(p0=ft >> 12)

    # embedding (SPEC C.1): P1 <- address of row `token` of E, from the token table; x <- that row
    em.tag = "embedding"
    em.point(0, token)
    em.emit("LD", R1, 0, 1)                                      # high byte of the token id
    em.emit("LD", R2, 0, 1)
    for column in (1, 0, 2):                                     # bits 11..8, low byte, bits 19..12 of the address
        address = tok + column * span
        em.page_of(0, address)
        em.emit("LDI", R0, address >> 8 & 15)
        em.emit("ADD", R0, R1)
        em.emit("MVP", 0, R0, R2)                                # P0 <- column + token id
        em.emit("LD", R3 if column == 1 else R0, 0, 0)
        if column == 0:
            em.emit("MVP", 1, R3, R0)
        if column == 2:
            em.emit("MVH", 1, R0)
    em.point2(x)
    if sh_e:                                # x[k] = rqr(E[token][k], sh_e): a function of one byte, so a table
        def shifted():                      # (16 cycles per element; a product with 1, LDA, RQ would be 20)
            em.emit("LD", R0, 1, 1)
            em.emit("LUT", R0, X["EMB"])
            em.emit("ST2", R0, 1)
        em.loop(D, shifted)
    else:
        em.loop(D, copy)
    em.at(2, x + D)                                              # y follows x: the first norm needs no set-up of P2

    for l in range(L):
        Wq, Wk, Wv, Wo, W1, W2 = (W["%s_%d" % (name, l)] for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"))
        norm(2 * l)
        em.tag = "qkv"                                           # SPEC C.3
        em.rows(Wq, y, q, D, D, sh["sh_q"][l])
        em.rows(Wk, y, k, G * d, D, sh["sh_k"][l])
        em.rows(Wv, y, v, G * d, D, sh["sh_v"][l])
        em.tag = "cache"                                         # SPEC C.5
        pages = [(mem["kv.K%d.%d" % (l, kvh)][0] - key_off) // PAGE for kvh in range(G)]
        em.scatter(v, [(pages[kvh] + page, rows) for kvh in range(G) for page, rows in chunks], slot)
        for kvh in range(G):                                     # the rotated key goes straight into the cache
            em.tag = "cache"
            em.key_pointer(key_fields, mem["kv.K%d.%d" % (l, kvh)][0] >> 12)
            em.tag = "rope"
            em.rope(k + kvh * d, d, rope)

        def head():                                              # the query head whose record is at hrec
            em.tag = "rope"
            em.rope(q, d, rope, fields=hrec + 3)                 # P2 is at qr after the copy of the record
            em.tag = "scores"                                    # SPEC C.6
            em.scores(hrec + 5, qr, e, d, sh["sh_s"][l], count, RELU, EXP + l)
            em.tag = "mix"
            em.mix(hrec + 9, e, o, ones, chunks, sh["sh_o"][l], RECIP, ring == 256)
            em.tag = "book"
        em.tag = "book"
        em.for_records(heads + l * H * HEAD, hrec, HEAD, head)
        em.tag = "attn_out"                                      # SPEC C.7
        em.residual(Wo, o, x, D, D)
        norm(2 * l + 1)
        em.tag = "ffn"                                           # SPEC C.8
        if swiglu:
            em.rows(W1, y, g1, F, D, sh["sh_1"][l], lut=SILU, lut_tag="gating")
            em.rows(W["W3_%d" % l], y, h3, F, D, sh["sh_3"][l])
            em.tag = "gating"
            em.product(g1, h3, g, F, sh["sh_g"][l])
            em.tag = "ffn"
        else:
            em.rows(W1, y, g, F, D, sh["sh_1"][l], relu=1)
        em.residual(W2, g, x, D, F)
    if prompt:                              # more prompt tokens: no output layer, no choice, nothing is sent
        em.tag = "prompt"
        em.point(0, left)
        em.emit("LD", R0, 0, 0)
        em.emit("MOV", R0, R0)
        last = em.branch("BZ")
        em.emit("LDI", R1, 1)
        em.emit("SUB", R0, R1)
        em.emit("ST", R0, 0, 0)
        em.emit("MOV", R0, R0)
        em.emit("BNZ", begin)
        em.land(last)
    norm(2 * L)
    em.tag = "output"                                            # SPEC C.9
    em.rows(W["E"], y, u, V, D, sh["sh_l"], c=pre["c_l"], relu=1)
    if temperature > 0:
        em.tag = "sample"
        em.sample(u, V, X["GUMBEL"])
    if syntax_filter:                       # abstract state (FILTER.md, section 4) -> its row of ALLOW -> table ALLOWED
        em.tag = "filter"
        em.page_of(2, ft)
        em.page_of(1, ft)
        em.point(0, cells + 1)
        em.emit("LD", R0, 0, 0)
        em.emit("LUT", R0, X["LEXA"])                            # in a string: abstract state + 1
        have = [em.branch("BNZ")]
        em.emit("SETP0", (cells + 7) % PAGE)
        em.emit("LD", R0, 0, 0)                                  # depth
        em.emit("LDI", R1, ROWS - 1)
        em.emit("MVP", 2, R1, R0)
        em.emit("LD2", R0, 0)                                    # top of the bracket stack; 0 at depth 0
        em.emit("MOV", R0, R0)
        have.append(em.branch("BNZ"))
        em.emit("SETP0", (cells + 2) % PAGE)
        em.emit("LD", R1, 0, 0)                                  # level
        em.emit("SETP0", (cells + 5) % PAGE)
        em.emit("LD", R2, 0, 1)                                  # word
        em.emit("LD", R3, 0, 0)                                  # colon
        em.emit("LDI", R0, 12)
        em.emit("ADD", R0, R3)
        em.emit("MVP", 1, R0, R2)
        em.emit("LD", R0, 1, 0)
        em.emit("MOV", R0, R0)
        have.append(em.branch("BNZ"))
        em.emit("LDI", R0, 7)
        em.emit("MVP", 1, R0, R1)
        em.emit("LD", R0, 1, 0)
        for address in have:
            em.land(address)
        em.forget(p0=ft >> 12, p1=ft >> 12, p2=ft >> 12)
        em.emit("MOV", R1, R0)
        em.emit("LUT", R0, X["ALO"])
        em.emit("LUT", R1, X["AHI"])
        em.page_of(1, allow, tmp=R2)
        em.emit("MVP", 1, R1, R0)                                # P1 <- row of ALLOW
        em.point(0, 256 * X["ALLOWED"], tmp=R2)
        em.emit("CPP20")
        em.loop(F_.ROW, copy)
        em.forget()
    em.tag = "argmax"
    if syntax_filter:
        em.argmax(u, V, token, X["RELU1"], cls, X["ALLOWED"])
    else:
        em.argmax(u, V, token, RELU)

    # the chosen token becomes the current token and goes out as two bytes: 0x80 | (id >> 7), id & 0x7F
    em.tag = "book"
    em.emit("LDI", R0, 0)
    em.emit("SUB", R0, R2)                                       # low byte of the token id
    em.emit("LD2", R1, 1)                                        # high byte, written by the argmax
    em.emit("ST2", R0, 0)
    em.emit("SHL", R1)
    em.emit("LDI", R2, 0x80)
    em.emit("MOV", R3, R0)
    em.emit("AND", R3, R2)                                       # bit 7 of the low byte
    small = em.branch("BZ")
    em.emit("XOR", R0, R2)                                       # it moves from the second byte to the first
    em.emit("LDI", R3, 1)
    em.emit("OR", R1, R3)
    em.land(small)
    em.emit("OR", R1, R2)
    em.emit("TX", R1)
    em.emit("TX", R0)
    em.emit("JMP", begin)
    em.land(stop)
    for address in halts:
        em.land(address)
    if uart_prompt:                         # the answer is over: send the end mark, wait for the next prompt
        em.emit("LDI", R0, END, tag="start")
        em.emit("TX", R0, tag="start")
        em.emit("JMP", 0, tag="start")
    else:
        em.emit("HALT")
    if len(em.code) > 4096:
        raise ValueError("the program has %d words, the ROM holds 4,096" % len(em.code))
    b = Build(em.code, em.loops, image, mem, em.words())
    if prompts is not None:
        set_prompts(image, b, prompts)
    return b


# ---- running a Build in the simulator ----

def machine(b, sw=0):
    """A CPU at reset with the program and the memory image of `b`."""
    return L1(b.words, b.image, lut_base=b.mem["tables"][0], sw=sw)


def peek(cpu, b, name):
    """The bytes of entry `name` of the memory map, as int8."""
    address, size = b.mem[name]
    return np.array(cpu.view[address:address + size], dtype=np.int64)


def set_prompt(memory, b, ids):
    """Write the prompt (token ids) into `memory`: the image of a Build (b.image) or the memory of a CPU (cpu.d)
    before it runs. 1 to min(T, 255) tokens; b must be built with prompt=True."""
    (left, size), (prm, size) = b.mem["prompt.left"], b.mem["prompt.ids"]
    if not 1 <= len(ids) <= min(255, b.mem["ones"][1]):
        raise ValueError("a prompt has 1 to min(context, 255) tokens, not %d" % len(ids))
    memory[left] = len(ids) - 256 * (len(ids) > 127)
    for p, token in enumerate(ids):
        memory[prm + p], memory[prm + 256 + p] = token >> 8, (token & 255) - 256 * (token & 255 > 127)


def set_prompts(memory, b, prompts):
    """Write the prompt table (1 to 64 lists of token ids) into `memory`: the image of a Build or the memory of
    a CPU before it runs. b must be built with prompts=...; the new prompts must fit the bytes of its table.
    Switch numbers beyond the list get prompt 0."""
    (index, size), (table, size) = b.mem["prompts.index"], b.mem["prompts.table"]
    limit = min(255, b.mem["ones"][1])
    if not 1 <= len(prompts) <= 64 or any(not 1 <= len(ids) <= limit for ids in prompts):
        raise ValueError("a prompt table has 1 to 64 prompts of 1 to %d tokens" % limit)
    if sum(2 * len(ids) for ids in prompts) > size:
        raise ValueError("the prompts need %d bytes, the table has %d" % (sum(2 * len(ids) for ids in prompts), size))
    entries = []
    for ids in prompts:
        entries.append((table >> 8 & 15, table & 255, table >> 12, len(ids)))
        for g, byte in enumerate([token >> 8 for token in ids] + [token & 255 for token in ids]):
            memory[table + g] = byte - 256 * (byte > 127)
        table += 2 * len(ids)
    for n in range(64):
        for g, byte in enumerate(entries[n] if n < len(entries) else entries[0]):
            memory[index + 4 * n + g] = byte - 256 * (byte > 127)


def top(b):
    """Address where the per-token program begins: 0, or the length of the start-up piece (prompts, uart_prompt)."""
    return next(address for address, (name, args, tag) in enumerate(b.code) if tag != "start")


def frame(ids):
    """The UART bytes of a prompt for a program built with uart_prompt: two bytes per token id, then END."""
    return [byte for token in ids for byte in (0x80 | token >> 7, token & 0x7F)] + [END]


def answers(cpu):
    """The token ids that the CPU has sent, one list per answer that END has closed, then the open one."""
    out, first = [[]], None
    for byte in cpu.tx:
        if first is not None:
            out[-1].append((first & 0x7F) << 7 | byte)
            first = None
        elif byte == END:
            out.append([])
        else:
            first = byte
    return out


def sent(cpu):
    """The token ids that the CPU has sent over the UART (two bytes per token, SPEC Part D), end marks left out."""
    return [token for answer in answers(cpu) for token in answer]


def set_token(cpu, b, token):
    address = b.mem["scratch.token"][0]
    cpu.d[address], cpu.d[address + 1] = token >> 8, (token & 255) - 256 * (token & 255 > 127)


def set_position(cpu, b, p):
    """Make the next token run at position p (the cache content is whatever the memory holds)."""
    address, link = b.mem["scratch.rec"][0], b.mem["pos.records"][0] + p * b.mem["scratch.rec"][1]
    for g, byte in enumerate((link >> 8 & 15, link & 255, link >> 12)):
        cpu.d[address + g] = byte - 256 * (byte > 127)


def run_token(cpu, b, token=None):
    """Compute one token (optionally after writing the current token id). Returns the id the program chose."""
    if token is not None:
        set_token(cpu, b, token)
    cpu.run(stop=0)
    address = b.mem["scratch.token"][0]
    return (cpu.d[address] & 255) << 8 | cpu.d[address + 1] & 255


def text(name, args):
    """One instruction in the notation of the design document, e.g. `LD R0, [P1]+`."""
    if name.startswith("CPP"):
        return "CPP P%s, P%s" % (name[3], name[4])
    if (name, args) == ("MAC", (0,)):
        return "MAC"
    fields = dict(zip(TABLE_L1[name].args, args))
    shown = []
    for field, value in fields.items():
        if field in ("r", "r1", "r2", "rd", "rs"):
            shown.append("R%d" % value)
        elif field in ("fn", "pp"):
            shown.append("P%d" % value)
        elif field == "inc":
            shown.append("[P%d]%s" % (fields.get("p", 2), "+" * value))       # LD2 and ST2 have no p: they use P2
        elif field == "n":
            shown.append("#%d" % value)
        elif field != "p":
            shown.append("0x%03X" % value if field == "imm12" else str(value))
    return (name.rstrip("2") + " " + ", ".join(shown)).strip()


def listing(b):
    """The program as text: address, word, instruction, component."""
    return "\n".join("%03X  %04X  %-18s ; %s" % (address, b.words[address], text(name, args), tag)
                     for address, (name, args, tag) in enumerate(b.code))
