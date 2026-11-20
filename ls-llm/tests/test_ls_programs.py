"""The four neuron programs S, U, L, B of the design document (07-programs.tex) on the LS simulator.

Each is assembled from its listing and run on the 64-input and the 32-input neuron. Checked: the predicted cycle
counts, the instruction and word counts, the exact integer dot product in ACC, and the output byte against a plain
numpy reference, on random data and on worst-case operands.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from lsllm.ls.asm import assemble
from lsllm.ls.iss import LS

# ---- the listings, copied from 07-programs.tex (the first test compares them with the file) ----
PROLOGUE = """\
        IN    R0            ; R0 <- SW7..0 (SW7..6 = image number)
        LDI   R1, 0xC0
        AND   R0, R1        ; R0 <- image * 64
        MVP   P0, R1, R0    ; P0 <- R1[3:0] & R0 = image * 64
        SETP1 WBASE         ; P1 <- address of w0
        LDI   R2, BHI
        LDI   R3, BLO
        LDA   R2, R3        ; ACC <- sext24(bias)
"""
EPILOGUE = """\
        RQ    R0, SH, RELU  ; R0 <- clamp(relu(ACC >>> SH))
        OUT   R0            ; LED7..0 <- R0
        HALT
"""
BODY = {
    "S": """\
        MAC   #64
""",
    "L": """\
        LDI   R3, 64
loop:   MAC
        DJNZ  loop
""",
    "B": """\
        LDI   R3, 64
loop:   LD    R0, [P0]+
        LD    R1, [P1]+
        MUL   R0, R1
        ADDP
        DJNZ  loop
""",
}
DOCUMENT = Path(__file__).parents[2] / "8-bit-cpu/docs/design"       # the design document, if it is there

# ---- the predictions: numbers.tex (cycles, words, instructions at N = 64) and the table of 07-programs.tex ----
CYCLES = {("S", 64): 112, ("U", 64): 300, ("L", 64): 560, ("B", 64): 1328,
          ("S", 32): 80, ("U", 32): 172, ("L", 32): 304, ("B", 32): 688}
WORDS = {("S", 64): 12, ("U", 64): 75, ("L", 64): 14, ("B", 64): 17,
         ("S", 32): 12, ("U", 32): 43, ("L", 32): 14, ("B", 32): 17}
INSTRUCTIONS = {("S", 64): 12, ("U", 64): 75, ("L", 64): 140, ("B", 64): 332,
                ("S", 32): 12, ("U", 32): 43, ("L", 32): 76, ("B", 32): 172}   # N = 32: counted from the listings,
#                                                                                 numbers.tex has them for N = 64 only
BODY_CYCLES = {"S": lambda n: n + 4, "U": lambda n: 4 * n, "L": lambda n: 4 + 8 * n, "B": lambda n: 4 + 20 * n}
ALL = [(name, n) for n in (64, 32) for name in "SULB"]


def source(name, n, bias, sh, relu):
    """Program `name` for an n-input neuron: named constants, then the listing of the document."""
    body = "        MAC\n" * n if name == "U" else BODY[name].replace("64", str(n))     # U: MAC written n times
    constants = f"WBASE = 0\nBHI = {bias >> 8 & 255}\nBLO = {bias & 255}\nSH = {sh}\nRELU = {relu}\n"
    return f"{constants}.region prologue\n{PROLOGUE}.region body\n{body}.region epilogue\n{EPILOGUE}"


def reference(x, w, n, image, bias, sh, relu):
    """The neuron in plain numpy: exact sum, division by 2^sh rounded toward minus infinity, ReLU, clamp."""
    acc = bias + int(np.dot(x[image * 64:image * 64 + n].astype(np.int64), w[:n].astype(np.int64)))
    y = acc // 2 ** sh
    if relu:
        y = max(y, 0)
    return acc, int(np.clip(y, -128, 127)) & 0xFF


def check(x, w, image, bias, sh, relu):
    """Run all eight programs on one data set; return the output byte of the 64-input neuron."""
    for name, n in ALL:
        program = assemble(source(name, n, bias, sh, relu))
        cpu = LS(program.words, x, w, program.regions, sw=image << 6 | 0x2A)    # SW5..0 must not matter
        cpu.run()
        acc, byte = reference(x, w, n, image, bias, sh, relu)
        assert (cpu.acc, cpu.overflow, cpu.outr, cpu.halted) == (acc, False, byte, True), (name, n)
        assert cpu.p == [image * 64 + n, n]
        assert (cpu.cycles, cpu.instructions, len(program.words)) == (CYCLES[name, n], INSTRUCTIONS[name, n],
                                                                      WORDS[name, n]), (name, n)
        assert cpu.region_cycles == {"prologue": 32, "body": BODY_CYCLES[name](n), "epilogue": 12}
    return reference(x, w, 64, image, bias, sh, relu)[1]


@pytest.mark.skipif(not DOCUMENT.exists(), reason="the design document is not next to this project")
def test_listings_and_predictions_are_those_of_the_document():
    listings = (DOCUMENT / "sections/07-programs.tex").read_text()
    blocks = re.findall(r"\\begin\{lstlisting\}\n(.*?)\\end\{lstlisting\}", listings, re.S)
    assert blocks[0] == PROLOGUE + "        ; ---- body: one of S, U, L, B ----\n" + EPILOGUE
    assert [blocks[1], blocks[3], blocks[4]] == [BODY["S"], BODY["L"], BODY["B"]]
    assert blocks[2].splitlines()[0].split(";")[0].strip() == "MAC" and "64 in all" in blocks[2]
    numbers = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (DOCUMENT / "numbers.tex").read_text()))
    for name in "SULB":
        doc = [int(numbers[key + name + n].replace("{,}", "")) for key in ("cyc", "wd") for n in ("", "th")]
        assert doc == [CYCLES[name, 64], CYCLES[name, 32], WORDS[name, 64], WORDS[name, 32]]
        assert int(numbers["ins" + name]) == INSTRUCTIONS[name, 64]


def test_program_words():
    """The words of S, and the bodies of L and B: together they contain all nine encoding examples of 05-isa.tex."""
    words = {name: assemble(source(name, 64, 0x1234, 2, 1)).words for name in "SULB"}
    assert words["S"] == [0x0002, 0x14C0, 0x2102, 0x3400, 0xB000, 0x1812, 0x1C34, 0x7B01, 0x8040,
                          0x9202, 0x0003, 0x0001]
    assert words["U"] == words["S"][:8] + [0x8000] * 64 + words["S"][9:]
    assert words["L"] == words["S"][:8] + [0x1C40, 0x8000, 0xF009] + words["S"][9:]
    assert words["B"] == words["S"][:8] + [0x1C40, 0x4100, 0x4700, 0x6100, 0x7000, 0xF009] + words["S"][9:]


def test_predicted_counts_follow_the_formulas_of_the_document():
    for n in (64, 32):
        assert [CYCLES[name, n] for name in "SULB"] == [48 + n, 44 + 4 * n, 48 + 8 * n, 48 + 20 * n]


def test_random_data():
    rng = np.random.default_rng(2026)
    outputs = set()
    for case in range(96):                                           # every shift 0..23 four times
        x = rng.integers(-128, 128, 256, dtype=np.int8)              # four images of 64 inputs
        w = rng.integers(-128, 128, 64, dtype=np.int8)
        bias, image, relu = int(rng.integers(-32768, 32768)), int(rng.integers(0, 4)), int(rng.integers(0, 2))
        outputs.add(check(x, w, image, bias, sh=case % 24, relu=relu))
    assert {0x00, 0x7F, 0x80} < outputs and len(outputs) >= 16       # the data reached both clamps and values between


@pytest.mark.parametrize("xv, wv, bias", [(-128, -128, 32767), (-128, 127, -32768), (127, 127, 32767),
                                          (127, -128, -32768), (-128, -128, -32768), (0, 0, 0)])
def test_worst_case_operands(xv, wv, bias):
    x, w = np.full(256, xv, dtype=np.int8), np.full(64, wv, dtype=np.int8)
    for sh in (0, 7, 13, 14, 23):
        for relu in (0, 1):
            check(x, w, image=3, bias=bias, sh=sh, relu=relu)
    program = assemble(source("S", 64, bias, 0, 0))
    cpu = LS(program.words, x, w)
    cpu.run()
    assert cpu.acc == bias + 64 * xv * wv                            # e.g. 64 * (-128)(-128) = 2^20, plus the bias
