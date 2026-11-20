#!/usr/bin/env python3
"""The eight programs of the course CPU, run in the model's simulator.

    PYTHONPATH=. .venv/bin/python tools/l0_crosscheck.py "../LS CPU" [--write docs/L0_CROSSCHECK.md]

Reads the VHDL of the finished level-0 CPU without changing it: the program ROM (ProgramMemory.vhd), the X and W
memories (DataMemories.vhd), the start address of every program (LSCore.vhd) and the expected cycle counts and
result bytes (tb_LSCore.vhd). Each program then runs in lsllm.ls.iss.LS, the instruction-level simulator that the
model's cycle counts come from, for all four images. Cycles and result byte must be those the VHDL testbench expects.
This checks that the two sides agree on the encoding and the cost of every instruction these programs use.
"""
import re
import sys
from collections import Counter

from lsllm.ls.isa import decode
from lsllm.ls.iss import LS

NAMES = ("S64", "U64", "L64", "B64", "S32", "U32", "L32", "MAC #8 demo")


def hex_list(text, name, digits):
    body = re.search(r"%s\s*:\s*\w+\s*:=\s*\((.*?)\);" % name, text, re.S).group(1)
    return [int(v, 16) for v in re.findall(r'x"([0-9A-Fa-f]{%d})"' % digits, body)]


def main(folder, write=None):
    read = lambda name: open("%s/%s" % (folder, name)).read()
    words = hex_list(read("ProgramMemory.vhd"), "program", 4)
    data = read("DataMemories.vhd")
    as_int8 = lambda values: [v - 256 if v > 127 else v for v in values]
    x, w = as_int8(hex_list(data, "xMemory", 2)), as_int8(hex_list(data, "wMemory", 2))
    core = read("LSCore.vhd")
    starts = [int(v, 16) for v in re.findall(r'x"([0-9A-Fa-f]{3})"', re.search(r"programStarts\s*:\s*\w+\s*:=\s*\((.*?)\);", core).group(1))]
    expected = [tuple(int(v, 16) if "#" in raw else int(v) for raw, v in ((m, m.split("#")[1] if "#" in m else m) for m in case))
                for case in re.findall(r"\((\d+), (\d+), (\d+), (16#[0-9A-Fa-f]+#)\)", read("tb_LSCore.vhd"))]
    lines = ["| program | start | image | cycles: VHDL testbench | cycles: simulator | result: VHDL testbench | result: simulator | agree |",
             "|---|---|---|---|---|---|---|---|"]
    used, wrong = Counter(), 0
    for program, image, cycles, result in expected:
        cpu = LS(words, x, w, sw=image * 64)
        cpu.pc = starts[program]
        cpu.run(limit=100_000)
        used.update(cpu.insn_cycles)
        same = cpu.halted and cpu.cycles == cycles and cpu.outr == result
        wrong += not same
        lines.append("| %s | 0x%03X | %d | %d | %d | 0x%02X | 0x%02X | %s |" % (NAMES[program], starts[program], image, cycles, cpu.cycles, result, cpu.outr,
                                                                    "yes" if same else "NO"))
    mnemonics = Counter()
    for start, stop in zip(starts, starts[1:] + [len(words)]):
        for word in words[start:stop]:
            mnemonics[decode(word)[0].name] += 1
    summary = ["%d of %d runs agree (%d programs, %d images each)." % (len(expected) - wrong, len(expected), len(starts), len(expected) // len(starts)),
               "Instructions in the eight programs, by mnemonic of the model's table: " + ", ".join("%s %d" % pair for pair in sorted(mnemonics.items())) + ".",
               "Cycles spent per mnemonic over all runs: " + ", ".join("%s %d" % pair for pair in sorted(used.items())) + "."]
    print("\n".join(summary))
    if write:
        open(write, "w").write("\n".join(
            ["# The course CPU's eight programs in the model's simulator", "",
             "Written by `tools/l0_crosscheck.py` from the VHDL in `%s` (read, not changed): the program ROM, the X and W" % folder,
             "memories, the start addresses and the numbers the VHDL testbench expects. The simulator is `lsllm.ls.iss.LS`, level L0:",
             "the one the model's cycle counts are built on (every instruction 4 cycles, `MAC #n` n + 4).", ""] + summary + [""] + lines) + "\n")
    return wrong


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None))
