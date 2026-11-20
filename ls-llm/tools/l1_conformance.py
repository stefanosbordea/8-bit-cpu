#!/usr/bin/env python3
"""export/conformance/: small programs for every instruction that level 1 adds, with what the CPU must show.

    PYTHONPATH=. .venv/bin/python tools/l1_conformance.py

Each program is a few dozen words, ends with OUT and HALT, and is run here in the cycle-exact simulator
(lsllm.ls.iss.L1). For each one the folder gets program.hex (one 16-bit word per line), data.hex (the data memory,
one byte per line, two's complement), listing.txt and expected.json: clock cycles from the first FETCH to the end of
HALT, the byte given to OUT, the bytes given to TX, and the final registers, pointers, LEN and ACC. A VHDL testbench
can check the extended CPU against these one instruction group at a time, before the model program (ten million
cycles for the shortest request) is tried. The encodings are those of TABLE_L1 in lsllm/ls/isa.py.
"""
import json
import os

import numpy as np

from lsllm.ls.isa import TABLE_L1, decode, encode
from lsllm.ls.iss import L1

OUT = "export/conformance"
R0, R1, R2, R3 = range(4)
P0, P1, P2 = range(3)


def words(program):
    """[(mnemonic, operands...) or "label:"] -> 16-bit words; an operand written as a string is the address of that label."""
    labels, address = {}, 0
    for line in program:
        if isinstance(line, str):
            labels[line.rstrip(":")] = address
        else:
            address += 1
    return [encode(line[0], *[labels[v] if isinstance(v, str) else v for v in line[1:]], table=TABLE_L1) for line in program if not isinstance(line, str)]


def data(size, parts):
    """A data memory of `size` bytes with the given {address: values} parts (int8 values)."""
    image = np.zeros(size, dtype=np.int8)
    for address, values in parts.items():
        image[address:address + len(values)] = values
    return image


rng = np.random.default_rng(2026)
row = lambda n, low=-90, high=90: rng.integers(low, high + 1, size=n).astype(np.int8)
a16, b16 = row(16), row(16)
a300, b300 = row(300, -40, 40), row(300, -40, 40)
table2 = rng.integers(-128, 128, size=256).astype(np.int8)
matrix, vector = row(32, -60, 60), row(8, -60, 60)                      # 4 rows of 8 weights, one input row

PROGRAMS = {
    "pointers": ("MVP, MVH and CPP build a 20-bit address; ST through P0 and LD through P2 meet at the same byte",
                 [("LDI", R0, 0x0A), ("LDI", R1, 0x34), ("MVP", P0, R0, R1), ("LDI", R2, 0x01), ("MVH", P0, R2), ("CPP20",),
                  ("LDI", R3, 0x5C), ("ST", R3, P0, 1), ("LD2", R0, 1), ("CPP12",), ("MVH", P1, R1), ("OUT", R0), ("HALT",)],
                 data(8192, {}), ()),
    "store_load": ("ST through P0 and through P2 with post-increment, read back through P1 and P2, combined with XOR and ADD",
                   [("SETP0", 0x200), ("LDI", R0, 0x02), ("MVH", P2, R0), ("LDI", R1, 0x11), ("LDI", R2, 0x7E), ("ST", R1, P0, 1), ("ST", R2, P0, 0),
                    ("ST2", R2, 1), ("ST2", R1, 1), ("SETP1", 0x200), ("LD", R0, P1, 1), ("LD", R3, P1, 1), ("XOR", R0, R3),
                    ("LDI", R1, 0x02), ("MVH", P2, R1), ("LDI", R1, 0x00), ("LDI", R2, 0x00), ("MVP", P2, R1, R2), ("LD2", R1, 1), ("LD2", R2, 0),
                    ("ADD", R1, R2), ("XOR", R0, R1), ("OUT", R0), ("HALT",)],
                 data(16384, {}), ()),
    "mac_stream": ("MAC #16 over two rows of the one data memory, then RQ; the classic MAC (one product) after it",
                   [("SETP0", 0x100), ("LDI", R0, 0x01), ("MVH", P1, R0), ("SETP1", 0x200), ("LDI", R1, 0), ("LDA", R1, R1), ("MAC", 16),
                    ("RQ", R2, 5, 0), ("SETP0", 0x100), ("SETP1", 0x200), ("MAC", 0), ("RQ", R3, 4, 1), ("ADD", R2, R3), ("OUT", R2), ("HALT",)],
                 data(8192, {0x100: a16, 0x1200: b16}), ()),
    "macl": ("MVL sets LEN to 300 and MACL runs a stream of that length: longer than MAC #n can say",
             [("SETP0", 0x000), ("LDI", R0, 0x01), ("MVH", P1, R0), ("SETP1", 0x000), ("LDI", R1, 0x01), ("LDI", R2, 0x2C), ("MVL", R1, R2),
              ("LDI", R3, 0), ("LDA", R3, R3), ("MACL",), ("RQ", R0, 8, 0), ("OUT", R0), ("HALT",)],
             data(8192, {0x000: a300, 0x1000: b300}), ()),
    "lut": ("LUT reads table 2 at the value of the register and sets Z and N; BZ and BNZ follow the flags",
            [("LDI", R1, 7), ("LUT", R1, 2), ("LDI", R2, 200), ("LUT", R2, 2), ("XOR", R1, R2), ("LDI", R0, 9), ("LUT", R0, 2), ("BZ", "zero"),
             ("LDI", R3, 0xEE), ("JMP", "end"), "zero:", ("LDI", R3, 0x11), "end:", ("ADD", R1, R3), ("OUT", R1), ("HALT",)],
            data(1024, {512: np.concatenate([table2[:9], [0], table2[10:]]).astype(np.int8)}), ()),
    "nrm_rqv": ("NRM counts the significant bits of ACC; RQV shifts by a register",
                [("LDI", R0, 0x5B), ("LDI", R1, 0x73), ("MUL", R0, R1), ("LDI", R2, 0), ("LDA", R2, R2), ("ADDP",), ("ADDP",), ("ADDP",),
                 ("NRM", R2), ("LDI", R3, 7), ("RQV", R1, R3, 0), ("ADD", R1, R2), ("LDI", R0, 0xA5), ("MUL", R0, R0), ("LDA", R0, R0), ("NRM", R3),
                 ("RQV", R0, R2, 1), ("ADD", R1, R3), ("ADD", R1, R0), ("OUT", R1), ("HALT",)],
                data(256, {}), ()),
    "tx": ("TX sends four bytes, the last two built with the ALU",
           [("LDI", R0, 0x41), ("TX", R0), ("LDI", R1, 0x8A), ("TX", R1), ("ADD", R0, R1), ("TX", R0), ("SHR", R1), ("TX", R1), ("OUT", R0), ("HALT",)],
           data(256, {}), ()),
    "rx_echo": ("RX takes two bytes from the PC (0x12, 0x34); the program sends the first and their sum",
                [("RX", R0), ("TX", R0), ("RX", R1), ("ADD", R0, R1), ("TX", R0), ("OUT", R0), ("HALT",)],
                data(256, {}), (0x12, 0x34)),
    "rnd": ("RND gives the next byte of the 32-bit LFSR (switches at 0); four bytes go out and are combined",
            [("RND", R0), ("TX", R0), ("RND", R1), ("TX", R1), ("RND", R2), ("TX", R2), ("RND", R3), ("TX", R3), ("XOR", R0, R1), ("XOR", R0, R2),
             ("XOR", R0, R3), ("OUT", R0), ("HALT",)],
            data(256, {}), ()),
    "matrix_row_loop": ("The kernel of the model in small: 4 rows of 8 weights times one input row, each result stored through P2, then summed",
                        [("LDI", R0, 0x01), ("MVH", P1, R0), ("SETP1", 0x000), ("LDI", R0, 0x02), ("MVH", P2, R0), ("LDI", R3, 4), ("LDI", R2, 0),
                         "row:", ("SETP0", 0x040), ("LDA", R2, R2), ("MAC", 8), ("RQ", R1, 6, 1), ("ST2", R1, 1), ("DJNZ", "row"),
                         ("LDI", R0, 0x02), ("MVH", P2, R0), ("LDI", R0, 0), ("MVP", P2, R0, R0), ("LDI", R3, 4), ("LDI", R1, 0),
                         "sum:", ("LD2", R0, 1), ("ADD", R1, R0), ("DJNZ", "sum"), ("OUT", R1), ("HALT",)],
                        data(16384, {0x040: vector, 0x1000: matrix}), ()),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    table = ["| program | what it checks | words | cycles | OUT | TX bytes | R0 R1 R2 R3 | P0 P1 P2 | LEN | ACC |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, (what, program, image, received) in PROGRAMS.items():
        code = words(program)
        cpu = L1(code, image, sw=0, rx=received)
        cpu.run(limit=100_000)
        assert cpu.halted and not cpu.waiting and not cpu.overflow, name
        folder = os.path.join(OUT, name)
        os.makedirs(folder, exist_ok=True)
        open(os.path.join(folder, "program.hex"), "w").write("".join("%04X\n" % word for word in code))
        open(os.path.join(folder, "data.hex"), "w").write("".join("%02X\n" % (int(v) & 0xFF) for v in image))
        listing, address = [], 0
        for line in program:
            if isinstance(line, str):
                listing.append(line)
            else:
                insn, args = decode(code[address], TABLE_L1)
                listing.append("  %03X  %04X  %-6s %s" % (address, code[address], insn.name, " ".join(str(v) for v in line[1:])))
                address += 1
        open(os.path.join(folder, "listing.txt"), "w").write("\n".join(listing) + "\n")
        expected = {"what": what, "program_words": len(code), "data_bytes": len(image), "switches": 0, "rx_bytes": ["0x%02X" % v for v in received],
                    "cycles": cpu.cycles, "out": "0x%02X" % cpu.outr, "tx_bytes": ["0x%02X" % v for v in cpu.tx],
                    "registers": ["0x%02X" % v for v in cpu.r], "pointers": ["0x%05X" % v for v in cpu.p], "len": cpu.len, "acc": cpu.acc,
                    "zero_flag": cpu.z, "negative_flag": cpu.n}
        json.dump(expected, open(os.path.join(folder, "expected.json"), "w"), indent=1)
        table.append("| `%s` | %s | %d | %d | 0x%02X | %s | %s | %s | %d | %d |" % (
            name, what, len(code), cpu.cycles, cpu.outr, " ".join("%02X" % v for v in cpu.tx) or "-", " ".join("%02X" % v for v in cpu.r),
            " ".join("%05X" % v for v in cpu.p), cpu.len, cpu.acc))
    used = sorted({decode(word, TABLE_L1)[0].name for name in PROGRAMS for word in words(PROGRAMS[name][1])})
    new = sorted(name for name in TABLE_L1 if name in ("LUT", "TX", "RND", "NRM", "RX", "RQV", "MVL", "MVH", "LD2", "ST2", "MACL") or name.startswith("CPP"))
    text = ["# Level-1 conformance programs", "",
            "Written by `tools/l1_conformance.py`. Ten small programs for the instructions that the model needs beyond the course CPU,",
            "each run in the cycle-exact simulator (`lsllm/ls/iss.py`, class `L1`). The numbers below are what the extended CPU must show",
            "in VHDL simulation. Nothing here ran on the board.", "",
            "- `program.hex`: one 16-bit word per line, from address 0. `data.hex`: the data memory D, one byte per line from address 0.",
            "- `listing.txt`: address, word, mnemonic and operands as written in the generator. `expected.json`: everything in the table.",
            "- Cycles: from the first FETCH to the end of `HALT`, 4 per instruction and n more for a stream of n products. Clocks spent",
            "  waiting in `RX` for a byte are not counted (in `rx_echo` both bytes are there before the program starts).",
            "- Reset state: registers, pointers, LEN, ACC and flags at 0; the LFSR at 0xACE1ACE1 xor the switches (switches 0 here).",
            "- `LUT` reads D[256 t + Rd] (table base 0).", "",
            ] + table + ["",
            "Instructions these programs execute: " + ", ".join("`%s`" % name for name in used) + ".",
            "Level-1 instructions not covered here: " + (", ".join("`%s`" % name for name in new if name not in used) or "none") + ".",
            "The model program itself is the last test: `export/board_uart/` and section 5 of `docs/CPU_ALIGNMENT.md`."]
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(text) + "\n")
    print("\n".join(table))
    print(text[-3]); print(text[-2])


if __name__ == "__main__":
    main()
