"""Instruction-level simulator of the LS CPU with exact cycle counting.

One step executes one whole instruction. Its effect and its cost in clock cycles both come from the instruction
table (lsllm.ls.isa.INSNS): every instruction takes 4 cycles, the stream MAC #n takes n + 4.

Only the state that a program can observe is kept. The read registers XR and WR, the valid bits and the stream
counter are not modelled: by the design document a read register always holds X[P0] or W[P1] when an instruction
uses it, and MAC #n leaves the machine exactly as n classic MACs in a row do.
"""
from array import array
from collections import Counter, deque

import numpy as np

from lsllm.ls.isa import MASK12, TABLE_L1, decode


def _memory(values, size):
    """A memory of `size` int8 cells (size is a power of two), filled from `values` and padded with zeros."""
    values = list(values)
    cells = [int(v) for v in values]
    if size < 1 or size & (size - 1) or len(cells) > size:
        raise ValueError(f"{len(cells)} values do not fit a memory of {size} cells (a power of two)")
    if any(cell != v or not -128 <= cell <= 127 for cell, v in zip(cells, values)):
        raise ValueError("memory values must be int8: whole numbers from -128 to 127")
    return cells + [0] * (size - len(cells))


class LS:
    """One LS CPU just after reset, with its program and data loaded. Create a new object to reset."""

    def __init__(self, words, x=(), w=(), regions=None, x_size=256, w_size=256, sw=0):
        """words: the program, 16-bit words from address 0; x, w: int8 contents of the X and W memories;
        regions: {address: name} from the assembler; x_size, w_size: cells per memory; sw: the switches."""
        if len(words) > MASK12 + 1:
            raise ValueError(f"{len(words)} words do not fit the 12-bit program counter")
        self.code, region = [], None      # per address: (instruction, operands, cycles, region name)
        for address, word in enumerate(words):
            insn, args = decode(word)
            cycles = insn.cycles if isinstance(insn.cycles, int) else insn.cycles(*args)
            region = (regions or {}).get(address, region)
            self.code.append((insn, args, cycles, region))
        self.mem = [_memory(x, x_size), _memory(w, w_size)]    # X (addressed by P0) and W (addressed by P1)
        self.r = [0, 0, 0, 0]             # R0..R3 as bytes 0..255
        self.z = self.n = 0               # flags, written by the ALU instructions only
        self.pc = 0                       # 12 bits
        self.p = [0, 0]                   # P0 and P1, 12 bits
        self.prod = 0                     # signed 16 bits
        self.acc = 0                      # signed 24 bits, wraps around
        self.overflow = False             # sticky: some addition into ACC did not fit in 24 bits
        self.sw = sw                      # IN reads SW7..0
        self.outr = 0                     # written by OUT, drives LED7..0
        self.halted = False
        self.cycles = self.instructions = 0
        self.insn_cycles = Counter()      # mnemonic -> cycles
        self.region_cycles = Counter()    # region name -> cycles (None: code before the first region mark)

    def step(self):
        """Execute one instruction and count its cycles."""
        if self.halted:
            raise RuntimeError("the CPU is halted")
        if self.pc >= len(self.code):
            raise RuntimeError(f"PC = 0x{self.pc:03X} is outside the program")
        insn, args, cycles, region = self.code[self.pc]
        self.pc = (self.pc + 1) & MASK12
        insn.effect(self, *args)
        self.cycles += cycles
        self.instructions += 1
        self.insn_cycles[insn.name] += cycles
        self.region_cycles[region] += cycles

    def run(self, limit=1_000_000):
        """Run until HALT. RuntimeError if the CPU has executed `limit` instructions without reaching it."""
        while not self.halted:
            if self.instructions >= limit:
                raise RuntimeError(f"no HALT after {limit} instructions")
            self.step()


class L1:
    """One LS CPU of level L1 (docs/SPEC.md, Part A) just after reset. Create a new object to reset.

    Cycles are not added up while running. `count` holds how often every program address was executed and
    `elems` how many multiply-adds its streams did. An instruction costs 4 cycles plus 1 per stream element, so
    cycles = 4 * sum(count) + sum(elems). The cost model (lsllm.ls.cost) prices the same counts in other ways.
    """

    def __init__(self, words, data=(), size=None, lut_base=0, sw=0, rx=()):
        """words: the program; data: int8 contents of the data memory D from address 0; size: bytes of D
        (default: len(data), at most 2^19); lut_base: address of table 0 of LUT; sw: the switches;
        rx: bytes that are already in the UART receive FIFO (more with `feed`)."""
        if len(words) > MASK12 + 1:
            raise ValueError(f"{len(words)} words do not fit the 12-bit program counter")
        data = np.asarray(data, dtype=np.int8)
        size = len(data) if size is None else size
        if not len(data) <= size <= 1 << 19:
            raise ValueError(f"{len(data)} bytes in a data memory of {size} bytes (at most 2^19)")
        self.code = []                    # per address: (effect, operands)
        for word in words:
            insn, args = decode(word, TABLE_L1)
            self.code.append((insn.effect, args))
        self.d = array("b", data.tobytes() + bytes(size - len(data)))    # D as int8 values
        self.view = np.frombuffer(self.d, dtype=np.int8)                 # the same bytes, for the streams
        self.lut_base = lut_base
        self.r = [0, 0, 0, 0]
        self.z = self.n = 0
        self.pc = 0
        self.p = [0, 0, 0]                # P0, P1, P2: 20 bits
        self.len = 0                      # LEN: 12 bits
        self.prod = self.acc = 0
        self.overflow = False             # sticky: some addition into ACC did not fit in 24 bits
        self.sw = sw
        self.outr = 0
        self.tx = []                      # bytes put into the UART transmit FIFO
        self.rx = deque(rx)               # bytes of the UART receive FIFO that RX has not read yet
        self.waiting = False              # an RX found the FIFO empty: the CPU waits on that instruction
        self.lfsr = 0xACE1ACE1 ^ (sw & 0xFFFF)   # 32 bits; the upper half keeps it away from 0
        self.halted = False
        self.count = [0] * len(words)
        self.elems = [0] * len(words)

    @property
    def instructions(self):
        return sum(self.count)

    @property
    def cycles(self):
        return 4 * sum(self.count) + sum(self.elems)

    def feed(self, data):
        """Put bytes into the UART receive FIFO; a CPU that waits for them goes on with the next `run`."""
        self.rx.extend(data)
        self.waiting = False

    def step(self):
        """Execute one instruction and count it. An RX on an empty FIFO is not executed and not counted."""
        pc = self.pc
        if self.halted:
            raise RuntimeError("the CPU is halted")
        if pc >= len(self.code):
            raise RuntimeError(f"PC = 0x{pc:03X} is outside the program")
        effect, args = self.code[pc]
        self.pc = (pc + 1) & MASK12
        n = effect(self, *args)           # a stream returns its number of multiply-adds
        if self.waiting:
            return
        self.count[pc] += 1
        if n:
            self.elems[pc] += n

    def run(self, stop=None, limit=10_000_000):
        """Run until HALT, until the CPU waits for the UART (`waiting`), or until PC = stop after at least one
        instruction. RuntimeError after `limit` instructions."""
        for g in range(limit):
            self.step()
            if self.halted or self.waiting or self.pc == stop:
                return
        raise RuntimeError(f"not finished after {limit} instructions")
