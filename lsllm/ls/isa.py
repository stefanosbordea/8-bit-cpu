"""The LS instruction set, as specified in the CPU design document (8-bit-cpu/docs/design, sections 05 and 06).

Every instruction is one 16-bit word with a 4-bit opcode in bits 15..12. INSNS is the single table that defines
the instruction set: mnemonic, opcode, format, fixed fields (function codes), operand fields, cycle cost, effect.
To add an instruction, write its effect function and add one row.

An effect is a function effect(cpu, *operands) that changes the state of the simulator (lsllm.ls.iss.LS):
r (R0..R3 as bytes), z, n, pc, p (P0, P1), mem (X, W as int8 values), prod, acc, overflow, sw, outr, halted.
"""
from typing import NamedTuple

import numpy as np

# The six formats (figures/formats.tex): field -> (highest bit, lowest bit).
# Bits 15..12 are the opcode; every bit that is not listed is reserved and always 0.
FORMATS = {
    "R": {"r1": (11, 10), "r2": (9, 8), "fn": (3, 0)},      # SYS, ALU, MVP, MUL, ADDP/LDA
    "I": {"rd": (11, 10), "imm8": (7, 0)},                  # LDI
    "M": {"r": (11, 10), "p": (9, 9), "inc": (8, 8)},       # LD, ST
    "Q": {"rd": (11, 10), "relu": (9, 9), "sh": (4, 0)},    # RQ
    "A": {"imm12": (11, 0)},                                # SETP0, SETP1, JMP, BZ, BNZ, DJNZ
    "N": {"n": (7, 0)},                                     # MAC #n
}
MASK12 = 0xFFF    # PC, P0 and P1 are 12 bits wide


def signed(value, bits):
    """The low `bits` bits of `value` read as a two's complement number."""
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def rq(acc, sh, relu):
    """The requantiser (figures/rq.tex): arithmetic shift right, optional ReLU, saturation to int8."""
    t = acc >> sh             # Python's >> rounds toward minus infinity, like the hardware: -301 >> 2 == -76
    if relu and t < 0:
        t = 0
    return max(-128, min(127, t))


def _accumulate(c, value):
    """ACC <- ACC + value in 24 bits. The sticky flag c.overflow records a sum that did not fit."""
    total = c.acc + value
    c.acc = signed(total, 24)
    c.overflow |= c.acc != total


def _cell(c, p):
    """Index of the cell that pointer p addresses in its memory (P0 in X, P1 in W): the low bits of the pointer."""
    return c.p[p] % len(c.mem[p])


# ---- effects: what each instruction does to the CPU state `c` ----

def _nop(c):
    pass


def _halt(c):
    c.halted = True


def _in(c, rd):
    c.r[rd] = c.sw & 0xFF                 # SW7..0


def _out(c, rs):
    c.outr = c.r[rs]


def _ldi(c, rd, imm8):
    c.r[rd] = imm8


def _alu(op):
    """Effect of an ALU instruction: Rd <- op(Rd, Rs) in 8 bits, Z and N from the result."""
    def effect(c, rd, rs=0):              # the shifts have no Rs
        t = op(c.r[rd], c.r[rs]) & 0xFF
        c.r[rd], c.z, c.n = t, int(t == 0), t >> 7
    return effect


def _mvp(c, p, ra, rb):
    c.p[p] = (c.r[ra] & 0xF) << 8 | c.r[rb]


def _ld(c, rd, p, inc):
    c.r[rd] = c.mem[p][_cell(c, p)] & 0xFF
    c.p[p] = (c.p[p] + inc) & MASK12


def _st(c, rs, p, inc):
    c.mem[p][_cell(c, p)] = signed(c.r[rs], 8)
    c.p[p] = (c.p[p] + inc) & MASK12


def _mul(c, ra, rb):
    c.prod = signed(c.r[ra], 8) * signed(c.r[rb], 8)


def _addp(c):
    _accumulate(c, c.prod)


def _lda(c, ra, rb):
    c.acc = signed(c.r[ra] << 8 | c.r[rb], 16)


def _mac(c, n):
    """MAC is one multiply-accumulate. MAC #n has the effect of n of them in a row; only the timing differs."""
    for g in range(max(n, 1)):
        c.prod = c.mem[0][_cell(c, 0)] * c.mem[1][_cell(c, 1)]
        _accumulate(c, c.prod)
        c.p[0] = (c.p[0] + 1) & MASK12
        c.p[1] = (c.p[1] + 1) & MASK12


def _rq(c, rd, sh, relu):
    c.r[rd] = rq(c.acc, sh, relu) & 0xFF


def _setp0(c, imm12):
    c.p[0] = imm12


def _setp1(c, imm12):
    c.p[1] = imm12


def _jmp(c, imm12):
    c.pc = imm12


def _bz(c, imm12):
    if c.z:
        c.pc = imm12


def _bnz(c, imm12):
    if not c.z:
        c.pc = imm12


def _djnz(c, imm12):
    c.r[3] = (c.r[3] - 1) & 0xFF          # flags unchanged
    if c.r[3]:
        c.pc = imm12


class Insn(NamedTuple):
    name: str         # mnemonic
    op: int           # opcode, bits 15..12
    fmt: str          # format, a key of FORMATS
    fixed: dict       # fields with a fixed value (the function code)
    args: tuple       # operand fields, in the order the operands are written in assembly
    cycles: object    # clock cycles: a number, or a function of the operands
    effect: object    # effect(cpu, *operands)
    limit: dict = {}  # largest value of an operand field, where that is less than the field can hold


INSNS = [
    #    name     op   fmt  fixed       operands              cycles, effect
    Insn("NOP",   0x0, "R", {"fn": 0},  (),                   4, _nop),
    Insn("HALT",  0x0, "R", {"fn": 1},  (),                   4, _halt),
    Insn("IN",    0x0, "R", {"fn": 2},  ("r1",),              4, _in),
    Insn("OUT",   0x0, "R", {"fn": 3},  ("r1",),              4, _out),
    Insn("LDI",   0x1, "I", {},         ("rd", "imm8"),       4, _ldi),
    Insn("ADD",   0x2, "R", {"fn": 0},  ("r1", "r2"),         4, _alu(lambda a, b: a + b)),
    Insn("SUB",   0x2, "R", {"fn": 1},  ("r1", "r2"),         4, _alu(lambda a, b: a - b)),
    Insn("AND",   0x2, "R", {"fn": 2},  ("r1", "r2"),         4, _alu(lambda a, b: a & b)),
    Insn("OR",    0x2, "R", {"fn": 3},  ("r1", "r2"),         4, _alu(lambda a, b: a | b)),
    Insn("XOR",   0x2, "R", {"fn": 4},  ("r1", "r2"),         4, _alu(lambda a, b: a ^ b)),
    Insn("MOV",   0x2, "R", {"fn": 5},  ("r1", "r2"),         4, _alu(lambda a, b: b)),
    Insn("SHL",   0x2, "R", {"fn": 6},  ("r1",),              4, _alu(lambda a, b: a << 1)),
    Insn("SHR",   0x2, "R", {"fn": 7},  ("r1",),              4, _alu(lambda a, b: a >> 1)),
    Insn("SAR",   0x2, "R", {"fn": 8},  ("r1",),              4, _alu(lambda a, b: (a >> 1) | (a & 0x80))),
    Insn("MVP",   0x3, "R", {},         ("fn", "r1", "r2"),   4, _mvp, {"fn": 1}),     # fn selects P0 or P1
    Insn("LD",    0x4, "M", {},         ("r", "p", "inc"),    4, _ld),
    Insn("ST",    0x5, "M", {},         ("r", "p", "inc"),    4, _st, {"p": 0}),       # only X is written
    Insn("MUL",   0x6, "R", {"fn": 0},  ("r1", "r2"),         4, _mul),
    Insn("ADDP",  0x7, "R", {"fn": 0},  (),                   4, _addp),
    Insn("LDA",   0x7, "R", {"fn": 1},  ("r1", "r2"),         4, _lda),
    Insn("MAC",   0x8, "N", {},         ("n",),               lambda n: n + 4, _mac),  # n = 0 is the classic MAC
    Insn("RQ",    0x9, "Q", {},         ("rd", "sh", "relu"), 4, _rq, {"sh": 23}),
    Insn("SETP0", 0xA, "A", {},         ("imm12",),           4, _setp0),
    Insn("SETP1", 0xB, "A", {},         ("imm12",),           4, _setp1),
    Insn("JMP",   0xC, "A", {},         ("imm12",),           4, _jmp),
    Insn("BZ",    0xD, "A", {},         ("imm12",),           4, _bz),
    Insn("BNZ",   0xE, "A", {},         ("imm12",),           4, _bnz),
    Insn("DJNZ",  0xF, "A", {},         ("imm12",),           4, _djnz),
]
TABLE = {insn.name: insn for insn in INSNS}


def _top(high, low):
    """Largest value of a field that spans bits high..low."""
    return (1 << (high - low + 1)) - 1


def encode(name, *args, table=None):
    """Mnemonic and operand values (in assembly order) -> 16-bit word. ValueError if a value does not fit.

    table: the instruction table of the level, TABLE (L0, the default) or TABLE_L1.
    """
    insn = (table or TABLE)[name]
    if len(args) != len(insn.args):
        raise ValueError(f"{name} takes {len(insn.args)} operand values, got {len(args)}")
    word = insn.op << 12
    for field, value in (*insn.fixed.items(), *zip(insn.args, args)):
        high, low = FIELDS[insn.fmt][field]
        top = insn.limit.get(field, _top(high, low))
        if not 0 <= value <= top:
            raise ValueError(f"{name}: {field} = {value} is outside 0..{top}")
        word |= value << low
    return word


def decode(word, table=None):
    """16-bit word -> (Insn, operand values). ValueError if the word is not the encoding of an instruction.

    A word is valid only if encoding the decoded instruction gives the same word back. So a word with a
    reserved bit set, a value in an unused field or an undefined function code is rejected.
    """
    table = table or TABLE
    for insn in table.values():
        if insn.op == word >> 12:
            spans = [FIELDS[insn.fmt][field] for field in insn.args]
            args = tuple((word >> low) & _top(high, low) for high, low in spans)
            try:
                if encode(insn.name, *args, table=table) == word:
                    return insn, args
            except ValueError:            # an operand above its limit: not this instruction
                pass
    raise ValueError(f"0x{word:04X} is not an LS instruction")


# ---- level L1 (docs/SPEC.md, Part A): one data memory D, 20-bit pointers P0..P2, LEN, tables, UART, LFSR ----
# The state of lsllm.ls.iss.L1 adds: d (the data memory, int8 values), view (the same bytes as a numpy array),
# p (P0, P1, P2), len, lut_base, tx (bytes sent), lfsr. At L1 every instruction costs 4 cycles; the effect of a
# stream returns its extra cycles (one per multiply-add).
MASK20 = 0xFFFFF  # P0, P1 and P2 are 20 bits wide
FORMATS_L1 = {
    "H": {"r2": (9, 8), "one": (3, 2), "pp": (1, 0)},        # MVH: function code 4 + pointer
    "M2": {"r": (11, 10), "inc": (8, 8), "two": (7, 7)},     # LD, ST through P2: pointer id IR[7] & IR[9] = 10
    "T": {"r1": (11, 10), "t": (9, 4), "fn": (3, 0)},        # LUT: table id in bits 9..4 (64 tables)
    "L": {"len": (8, 8)},                                    # MACL: bit 8 of the MAC word
    "V": {"rd": (11, 10), "relu": (9, 9), "var": (8, 8), "rs": (1, 0)},    # RQV: bit 8 of the RQ word
}
FIELDS = {**FORMATS, **FORMATS_L1}


def _low12(c, p, value):
    c.p[p] = c.p[p] & ~MASK12 | value                       # bits 19..12 keep their value


def _setp(p):
    return lambda c, imm12: _low12(c, p, imm12)


def _mvp1(c, p, ra, rb):
    _low12(c, p, (c.r[ra] & 0xF) << 8 | c.r[rb])


def _mvh(c, p, rb):
    c.p[p] = c.p[p] & MASK12 | c.r[rb] << 12


def _mvl(c, ra, rb):
    c.len = (c.r[ra] & 0xF) << 8 | c.r[rb]


def _cpp(d, s):
    def effect(c):
        c.p[d] = c.p[s]
    return effect


def _ld1(c, rd, p, inc):
    c.r[rd] = c.d[c.p[p]] & 0xFF
    c.p[p] = (c.p[p] + inc) & MASK20


def _st1(c, rs, p, inc):
    c.d[c.p[p]] = signed(c.r[rs], 8)
    c.p[p] = (c.p[p] + inc) & MASK20


def _lut(c, rd, t):
    t = c.d[c.lut_base + 256 * t + c.r[rd]] & 0xFF
    c.r[rd], c.z, c.n = t, int(t == 0), t >> 7


def _tx(c, rs):
    c.tx.append(c.r[rs])


def _rx(c, rd):
    """RX Rd: the next byte of the UART receive FIFO. With an empty FIFO the CPU waits: the instruction is not
    executed, PC stays on it and `waiting` is set (lsllm.ls.iss.L1.feed gives it bytes)."""
    if c.rx:
        c.r[rd] = c.rx.popleft()
    else:
        c.waiting, c.pc = True, (c.pc - 1) & MASK12


def _rnd(c, rd):
    s = c.lfsr
    for g in range(8):                                      # bit = s[0] xor s[10] xor s[30] xor s[31], shifted in at the top
        s = s >> 1 | ((s ^ s >> 10 ^ s >> 30 ^ s >> 31) & 1) << 31
    c.lfsr, c.r[rd] = s, s >> 24


def _nrm(c, rd):
    """Rd <- number of significant bits of ACC. SPEC A.2 defines it for ACC >= 0; a negative ACC gives 24 here
    (the position of its sign bit), which is what a priority encoder over the 24 bits would give."""
    c.r[rd] = (c.acc & 0xFFFFFF).bit_length()


def _rqv(c, rd, rs, relu):
    c.r[rd] = rq(c.acc, c.r[rs] & 31, relu) & 0xFF          # as RQ, the shift amount is Rs[4:0]


def _stream(c, n):
    """n multiply-adds D[P0] x D[P1] into ACC; both pointers advance by n. Returns n."""
    a, b = c.p[0], c.p[1]
    if max(a, b) + n > len(c.d):
        raise IndexError(f"a stream of {n} from P0 = 0x{a:05X}, P1 = 0x{b:05X} leaves the data memory")
    if n < 24:                                              # short streams: plain Python is faster than numpy
        products = [c.d[a + k] * c.d[b + k] for k in range(n)]
        total, c.prod = sum(products), products[-1]
    else:
        total = int(np.dot(c.view[a:a + n].astype(np.int32), c.view[b:b + n].astype(np.int32)))
        c.prod = c.d[a + n - 1] * c.d[b + n - 1]
    if abs(c.acc) + (n << 14) >= 1 << 23:                   # |product| <= 2^14: only now can a partial sum overflow
        partial = c.acc + np.cumsum(np.multiply(c.view[a:a + n], c.view[b:b + n], dtype=np.int64))
        c.overflow |= bool(partial.min() < -(1 << 23) or partial.max() >= 1 << 23)
    c.acc = signed(c.acc + total, 24)
    c.p[0], c.p[1] = (a + n) & MASK20, (b + n) & MASK20
    return n


def _mac1(c, n):
    _stream(c, max(n, 1))
    return n                                                # MAC #n costs n + 4, the classic MAC (n = 0) costs 4


def _macl(c):
    if not c.len:
        raise RuntimeError("MACL with LEN = 0")
    return _stream(c, c.len)


INSNS_L1 = [insn for insn in INSNS if insn.name not in ("MVP", "LD", "ST", "MAC", "SETP0", "SETP1")] + [
    Insn("LUT",   0x0, "T",  {"fn": 4},  ("r1", "t"),        4, _lut),
    Insn("TX",    0x0, "R",  {"fn": 5},  ("r1",),            4, _tx),
    Insn("RND",   0x0, "R",  {"fn": 6},  ("r1",),            4, _rnd),
    Insn("NRM",   0x0, "R",  {"fn": 7},  ("r1",),            4, _nrm),
    Insn("RX",    0x0, "R",  {"fn": 8},  ("r1",),            4, _rx),
    Insn("RQV",   0x9, "V",  {"var": 1}, ("rd", "rs", "relu"), 4, _rqv),
    Insn("MVP",   0x3, "R",  {},         ("fn", "r1", "r2"), 4, _mvp1, {"fn": 2}),      # fn selects P0, P1 or P2
    Insn("MVL",   0x3, "R",  {"fn": 3},  ("r1", "r2"),       4, _mvl),
    Insn("MVH",   0x3, "H",  {"one": 1}, ("pp", "r2"),       4, _mvh, {"pp": 2}),
    Insn("LD",    0x4, "M",  {},         ("r", "p", "inc"),  4, _ld1),
    Insn("LD2",   0x4, "M2", {"two": 1}, ("r", "inc"),       4, lambda c, rd, inc: _ld1(c, rd, 2, inc)),
    Insn("ST",    0x5, "M",  {},         ("r", "p", "inc"),  4, _st1, {"p": 0}),
    Insn("ST2",   0x5, "M2", {"two": 1}, ("r", "inc"),       4, lambda c, rs, inc: _st1(c, rs, 2, inc)),
    Insn("MAC",   0x8, "N",  {},         ("n",),             4, _mac1),
    Insn("MACL",  0x8, "L",  {"len": 1}, (),                 4, _macl),
    Insn("SETP0", 0xA, "A",  {},         ("imm12",),         4, _setp(0)),
    Insn("SETP1", 0xB, "A",  {},         ("imm12",),         4, _setp(1)),
] + [Insn(f"CPP{d}{s}", 0x3, "R", {"fn": 8 + 3 * d + s}, (), 4, _cpp(d, s))            # CPP Pd, Ps
     for d in range(3) for s in range(3) if d != s]
TABLE_L1 = {insn.name: insn for insn in INSNS_L1}
