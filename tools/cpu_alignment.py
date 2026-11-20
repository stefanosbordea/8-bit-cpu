#!/usr/bin/env python3
"""docs/CPU_ALIGNMENT.md: where the model and the course CPU agree today and what the CPU still needs.

    PYTHONPATH=. .venv/bin/python tools/cpu_alignment.py artifacts/board "../LS CPU"

Everything is computed: the program is the one the board model runs (UART prompts, syntax filter), decoded with the
level-1 instruction table; the sizes of the built CPU are read from its VHDL (not changed); the level-0 check is
tools/l0_crosscheck.py; the request numbers come from results/board_*.json (tools/board_verify.py).
"""
import json
import re
import sys
from collections import Counter

from lsllm.filter_table import TableFilter
from lsllm.ls import gen
from lsllm.ls.isa import FIELDS, TABLE, TABLE_L1, decode
from lsllm.tok import Tok
from tools.l0_crosscheck import main as l0_check

WIDER = {"MVP": "also selects the store pointer P2", "LD": "reads the one data memory D", "ST": "writes D through P0",
         "MAC": "multiplies D[P0] by D[P1]", "SETP0": "sets the low 12 bits of a 20-bit pointer", "SETP1": "sets the low 12 bits of a 20-bit pointer"}
num = lambda v: "{:,}".format(int(v))


def encoding(insn):
    fixed = ", ".join("%s = %d" % (name, value) for name, value in insn.fixed.items())
    return "opcode %X%s" % (insn.op, ", " + fixed if fixed else "")


def vhdl_testbenches(vhdl):
    """Runs the two testbenches of the built CPU with nvc in a temporary folder (the sources stay where they are)."""
    import glob
    import os
    import shutil
    import subprocess
    import tempfile
    if not shutil.which("nvc"):
        return "The VHDL testbenches were not run here (no `nvc` on this machine)."
    vhdl = os.path.abspath(vhdl)
    design = sorted(f for f in glob.glob(vhdl + "/*.vhd") if not os.path.basename(f).startswith("tb_") and os.path.basename(f) not in ("LSCore.vhd", "TopLevel.vhd"))
    files = design + [vhdl + "/LSCore.vhd", vhdl + "/TopLevel.vhd", vhdl + "/tb_LSCore.vhd", vhdl + "/tb_TopLevel.vhd"]
    with tempfile.TemporaryDirectory() as work:
        run = lambda *args: subprocess.run(["nvc", "--std=2008", *args], cwd=work, capture_output=True, text=True)
        analysed = run("-a", *files)
        core, top = run("-e", "tb_LSCore", "-r"), run("-e", "tb_TopLevel", "-r")
    version = subprocess.run(["nvc", "--version"], capture_output=True, text=True).stdout.split("\n")[0].split(" (")[0]
    passed = "ALL TESTS PASSED" in core.stdout + core.stderr, "TOP LEVEL PASSED" in top.stdout + top.stderr
    return ("The VHDL itself, simulated with %s on %s: `tb_LSCore` %s (all eight programs in run mode, with random clock-enable pauses and "
            "stepped by instruction), `tb_TopLevel` %s." % (version, __import__("datetime").date.today().isoformat(),
                                                           "prints ALL TESTS PASSED" if passed[0] else "does NOT pass", "passes" if passed[1] else "does NOT pass"))


def main(model, vhdl):
    spec, arrays = gen.load(model)
    tok = Tok(spec["tokenizer"] + "/tok.model")
    b = gen.build(spec, arrays, tf=TableFilter(tok, triple_quotes=False), syntax_filter=True, uart_prompt=True)
    static = Counter(decode(word, TABLE_L1)[0].name for word in b.words)
    cpu = gen.machine(b)
    ids = tok.encode("def hello_world(", bos=True)
    cpu.feed(bytes(gen.frame(ids)))
    cpu.run(limit=600_000_000)
    dynamic = Counter()
    for address, word in enumerate(b.words):
        dynamic[decode(word, TABLE_L1)[0].name] += 4 * cpu.count[address] + cpu.elems[address]
    total = sum(dynamic.values())
    sent = len(gen.answers(cpu)[-2])

    read = lambda name: open("%s/%s" % (vhdl, name)).read()
    rom_words = int(re.search(r"type programArray is array \(0 to (\d+)\)", read("ProgramMemory.vhd")).group(1)) + 1
    mem_bytes = int(re.search(r"type byteMemory is array \(0 to (\d+)\)", read("DataMemories.vhd")).group(1)) + 1
    pointer_bits = int(re.search(r"pointer0\s*:\s*out std_logic_vector\((\d+) downto 0\)", read("DataMemories.vhd")).group(1)) + 1
    wrong = l0_check(vhdl, "docs/L0_CROSSCHECK.md")

    data_rams, program_rams = -(-len(b.image) // 4096), -(-len(b.words) // 2048)
    from lsllm.config import ModelConfig
    trained = ModelConfig(**spec["config"]).n_params()
    stored = sum(int(v.size) for k, v in arrays.items() if k == "E" or k[:1] == "W")
    lines = ["# The model and the course CPU: what agrees and what is missing", "",
             "Written by `tools/cpu_alignment.py %s \"%s\"`. The VHDL is read, never changed. Nothing here was synthesised or run on" % (model, vhdl),
             "the board: cycle counts come from the cycle-exact simulator of the instruction set (`lsllm/ls/iss.py`).", "",
             "## 1. Level 0, the CPU that is built", "",
             "The eight programs in the program ROM of the built CPU were run in the model's simulator, with the X and W memories",
             "of the VHDL, for all four images: **%d of 32 runs give the cycle count and the result byte that the VHDL testbench" % (32 - wrong),
             "expects** (`docs/L0_CROSSCHECK.md`). The two sides agree on the encoding and the cost of every instruction those programs",
             "use: 4 cycles each, and n + 4 for `MAC #n`. The model's cycle counts are built on exactly these two rules.", "",
             vhdl_testbenches(vhdl), "",
             "## 2. The program the board has to run", "",
             "| | |", "|---|---|",
             "| model | `%s`: %s trained weights; the integer model stores %s int8 matrix entries, the %d gains of the norms are folded into the matrices and the norm tables |" % (
                 model, num(trained), num(stored), trained - stored),
             "| program | %s words of 16 bits (prompts over the UART, syntax filter, greedy) |" % num(len(b.words)),
             "| data memory image | %s bytes: weights, key/value cache, lookup tables, filter tables, scratch |" % num(len(b.image)),
             "| block RAMs by our count, before any synthesis | %d for the data image (4,096 bytes each) + %d for the program = **%d of 135** |" % (
                 data_rams, program_rams, data_rams + program_rams),
             "| program ROM of the built CPU | %d words (the program counter already has 12 bits) |" % rom_words,
             "| data memories of the built CPU | X and W, %d bytes each, pointers of %d bits |" % (mem_bytes, pointer_bits), "",
             "## 3. Instructions of that program", "",
             "Static: words in the program. Dynamic: clock cycles in one complete request (`def hello_world(`, %d tokens sent," % sent,
             "%s cycles). \"As built\" means the instruction is in the level-0 table with the same encoding and meaning." % num(total), "",
             "| instruction | encoding | words in the program | share of the cycles | against the built CPU |", "|---|---|---|---|---|"]
    for name, count in sorted(static.items(), key=lambda pair: -dynamic[pair[0]]):
        insn = TABLE_L1[name]
        base = name.rstrip("2") if name in ("LD2", "ST2") else name
        if name in TABLE and name not in WIDER:
            status = "as built"
        elif name in WIDER:
            status = "same encoding, wider meaning: " + WIDER[name]
        elif name in ("LD2", "ST2"):
            status = "new: `%s` through the store pointer P2 (bit 7 of the word)" % base
        elif name == "MACL":
            status = "new: stream of LEN multiply-adds (bit 8 of the `MAC` word)"
        elif name == "RQV":
            status = "new: requantise with the shift from a register (bit 8 of the `RQ` word)"
        else:
            status = "new"
        lines.append("| `%s` | %s | %d | %.2f%% | %s |" % (name, encoding(insn), count, 100 * dynamic[name] / total, status))
    unused = sorted(set(TABLE_L1) - set(static))
    streams = dynamic["MAC"] + dynamic["MACL"]
    lines += ["", "In the level-1 table and not used by this program: %s." % ", ".join("`%s`" % name for name in unused),
              "The two stream instructions take %.1f%% of all cycles: the multiply-accumulate unit of the built CPU is where the" % (100 * streams / total),
              "time goes, and it is used as it is.", "",
              "## 4. What the CPU still needs (level 1)", "",
              "The full list with estimated cost is `docs/HARDWARE_CHANGES.md`, section 1; the exact encodings are `TABLE_L1` in",
              "`lsllm/ls/isa.py` and Part A of `docs/SPEC.md`. In the order the program depends on them:", "",
              "1. Program memory of at least %s words (now %d). In block RAM the read is synchronous, so FETCH has to allow for it." % (num(len(b.words)), rom_words),
              "2. One data memory D of %s bytes with two read ports instead of X and W of %d bytes; P0 and P1 with 20 bits (now %d)." % (num(len(b.image)), mem_bytes, pointer_bits),
              "   The built data memories already register their read data for one clock, which is how block RAM behaves.",
              "3. The store pointer P2 with `MVP`, `MVH`, `MVL` and `CPP`, and `LD` and `ST` through P2 (`LD2`, `ST2`).",
              "4. The LEN register and `MACL` (a stream whose length is only known at run time).",
              "5. `LUT` (64 tables of 256 bytes inside D), `NRM` and `RQV` (the division in the attention).",
              "6. `TX` (115,200 baud, 8N1; two bytes per token) and `RX` (the request from the PC).",
              "7. `RND` is in the table for sampling. This program decodes greedily and does not execute it.", "",
              "Rules that the cycle counts assume and that must hold in the VHDL: every instruction takes 4 clocks; a stream of n",
              "multiply-adds takes n + 4; `RX` waits while no byte has arrived (waiting clocks are not in the counts).", "",
              "## 5. What to compare when the extended CPU simulates", "",
              "`export/board_uart/` has the memory images (`mem/data.hex`, `mem/program.hex`, also as `.coe`), the VHDL package with",
              "the constants, and `vectors/` with every intermediate value of the first tokens. The bytes sent by `TX` must equal",
              "those of the simulator. Short requests to start with (from `results/board_C2bp.json`; cycles as counted when the",
              "requests are sent one after the other in suite order, so a request sent alone can differ by a few hundred cycles):", "",
              "| request | prompt tokens | tokens sent | cycles | time at 100 MHz |", "|---|---|---|---|---|"]
    rows = json.load(open("results/board_C2bp.json"))
    for row in sorted(rows, key=lambda row: row["cycles"])[:5] + [row for row in rows if row["name"] == "bubble_sort"]:
        lines.append("| `def %s(` | %d | %d | %s | %.2f s |" % (row["name"], row["prompt_tokens"], row["tokens_sent"], num(row["cycles"]), row["cycles"] / 1e8))
    cycles, tokens = sum(row["cycles"] for row in rows), sum(row["tokens_sent"] for row in rows)
    lines += ["", "## 6. Numbers for the documents, with their source", "",
              "| statement | value | source |", "|---|---|---|",
              "| weights | %s trained weights, all carried by int8 values: %s int8 matrix entries, and the %d norm gains folded into the matrices and the norm tables | `artifacts/board/model.json`, `model.npz` |" % (
                  num(trained), num(stored), trained - stored),
              "| memory by our count, before synthesis | %d block RAMs for the data image plus %d for the program: %d of 135 | this file, section 2 |" % (
                  data_rams, program_rams, data_rams + program_rams),
              "| predicted speed at 100 MHz over the %d test requests | %.1f tokens per second (%s tokens in %s cycles) | `results/board_C2bp.json` |" % (
                  len(rows), tokens / (cycles / 1e8), num(tokens), num(cycles)),
              "| the CPU program sends the tokens of the integer reference | in %d of %d requests | `results/board_C2bp.json` |" % (
                  sum(row["same_as_reference"] for row in rows), len(rows)),
              "| programs that pass the tests of their request | %d of %d | `results/board_C2bp.json` |" % (sum(row["outcome"] == "pass" for row in rows), len(rows)),
              "| what kind of result this is | recall of programs the model was trained on, not a benchmark; it does not write arbitrary Python | `REPORT.md`, section 3 |",
              "| board results | none yet | - |"]
    open("docs/CPU_ALIGNMENT.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[5:10] + lines[13:21]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
