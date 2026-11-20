#!/usr/bin/env python3
"""The exported programs in a VHDL simulator: export/vhdl_reference/ReferenceL1.vhd run with nvc.

    PYTHONPATH=. .venv/bin/python tools/vhdl_reference.py [--requests hello_world,is_even,square] [--all] [--write docs/VHDL_REFERENCE.md]

ReferenceL1 is a behavioural model of the level-1 instruction set written in VHDL: it reads program.hex and data.hex,
executes instruction by instruction, counts 4 cycles each and n more for a stream, and writes the UART bytes. It is not
a design of the CPU (no states, no datapath); it is the instruction set in the language of the CPU, so that the exported
files are read by a VHDL tool and the result can be compared in the same simulator that runs the real design.
Checked here against the Python simulator: the ten conformance programs (cycles, OUT, UART bytes, registers, pointers,
LEN, ACC) and whole requests of the board model from export/board_uart (cycles and every byte sent).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VHDL = os.path.join(ROOT, "export", "vhdl_reference", "ReferenceL1.vhd")


def simulate(work, program, data, received, data_bytes, lookup_base=0):
    """Run the VHDL model; returns (result fields, bytes sent, seconds)."""
    with open(os.path.join(work, "rx.hex"), "w") as f:
        f.write("".join("%02X\n" % byte for byte in received))
    generics = ["-gprogramFile=" + program, "-gdataFile=" + data, "-greceiveFile=rx.hex", "-gsendFile=tx.hex", "-gresultFile=result.txt",
                "-gdataBytes=%d" % data_bytes, "-glookupBase=%d" % lookup_base]
    started = time.time()
    done = subprocess.run(["nvc", "--std=2008", "-M", "64m", "-H", "512m", "-e", *generics, "ReferenceL1", "-r"], cwd=work, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(done.stderr[-600:])
    fields = dict(line.split(" ", 1) for line in open(os.path.join(work, "result.txt")).read().strip().split("\n"))
    sent = [int(line, 16) for line in open(os.path.join(work, "tx.hex")) if line.strip()]
    return fields, sent, time.time() - started


def main(requests, write=None, everything=False):
    if not shutil.which("nvc"):
        sys.exit("nvc is not installed")
    from lsllm.filter_table import TableFilter
    from lsllm.ls import gen
    from lsllm.tok import Tok
    version = subprocess.run(["nvc", "--version"], capture_output=True, text=True).stdout.split("\n")[0].split(" (")[0]
    lines = ["# The exported programs in a VHDL simulator", "",
             "Written by `tools/vhdl_reference.py`. `export/vhdl_reference/ReferenceL1.vhd` is a behavioural model of the level-1",
             "instruction set in VHDL (it is not a design of the CPU: no states, no datapath). It reads the exported hex files, executes",
             "them and counts cycles by the two rules (4 per instruction, n more for a stream of n). Simulator: %s. Nothing ran on the board." % version, "",
             "## The ten conformance programs", "",
             "| program | cycles: Python simulator | cycles: VHDL | OUT | UART bytes | registers, pointers, LEN, ACC | agree |", "|---|---|---|---|---|---|---|"]
    wrong = 0
    with tempfile.TemporaryDirectory() as work:
        analysed = subprocess.run(["nvc", "--std=2008", "-a", VHDL], cwd=work, capture_output=True, text=True)
        if analysed.returncode:
            sys.exit(analysed.stderr)
        root = os.path.join(ROOT, "export", "conformance")
        for name in sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))):
            expected = json.load(open(os.path.join(root, name, "expected.json")))
            got, sent, seconds = simulate(work, os.path.join(root, name, "program.hex"), os.path.join(root, name, "data.hex"),
                                          [int(v, 16) for v in expected["rx_bytes"]], expected["data_bytes"])
            cycles = int(got["cycles_high"]) * 10 ** 9 + int(got["cycles_low"])
            state = ([int(v) for v in got["registers"].split()] == [int(v, 16) for v in expected["registers"]]
                     and [int(v) for v in got["pointers"].split()] == [int(v, 16) for v in expected["pointers"]]
                     and int(got["len"]) == expected["len"] and int(got["acc"]) == expected["acc"])
            same = (cycles == expected["cycles"] and int(got["out"]) == int(expected["out"], 16) and sent == [int(v, 16) for v in expected["tx_bytes"]]
                    and state and got["ended"] == "halted")
            wrong += not same
            lines.append("| `%s` | %d | %d | 0x%02X | %s | %s | %s |" % (name, expected["cycles"], cycles, int(got["out"]), " ".join("%02X" % v for v in sent) or "-",
                                                             "same" if state else "DIFFERENT", "yes" if same else "NO"))
        spec, arrays = gen.load(os.path.join(ROOT, "artifacts", "board"))
        tok = Tok(os.path.join(ROOT, spec["tokenizer"], "tok.model"))
        b = gen.build(spec, arrays, tf=TableFilter(tok, triple_quotes=False), syntax_filter=True, uart_prompt=True)
        mem = os.path.join(ROOT, "export", "board_uart", "mem")
        lines += ["", "## Whole requests of the board model, from `export/board_uart/mem/`", "",
                  "The request goes in as UART bytes; the model runs until it waits for the next request. Cycles and every byte sent are compared with",
                  "the Python simulator (`lsllm/ls/iss.py`), which is what all numbers of the report come from.", "",
                  "| request | bytes sent | program text from the VHDL run | cycles: Python simulator | cycles: VHDL | bytes equal | VHDL run time on this laptop |",
                  "|---|---|---|---|---|---|---|"]
        shown, equal, total_cycles, total_seconds = set(requests), 0, 0, 0.0
        if everything:
            requests = list(dict.fromkeys(requests + [json.loads(line)["name"] for line in open(os.path.join(ROOT, "data", "asks", "asks.jsonl"))]))
        for name in requests:
            ids = tok.encode("def %s(" % name, bos=True)
            cpu = gen.machine(b)
            cpu.feed(bytes(gen.frame(ids)))
            cpu.run(limit=2_000_000_000)
            got, sent, seconds = simulate(work, os.path.join(mem, "program.hex"), os.path.join(mem, "data.hex"), gen.frame(ids), len(b.image), b.mem["tables"][0])
            cycles = int(got["cycles_high"]) * 10 ** 9 + int(got["cycles_low"])
            same = sent == list(cpu.tx) and cycles == cpu.cycles and got["ended"] == "waiting"
            wrong += not same
            answer, high = [], None
            for byte in sent:
                if byte == gen.END:
                    break
                if byte & 0x80:
                    high = byte & 0x7F
                elif high is not None:
                    answer.append((high << 7) | byte)
                    high = None
            text = tok.decode(ids + answer[:-1]).replace("\n", " / ")
            equal += same
            total_cycles += cycles
            total_seconds += seconds
            if name in shown or not same:
                lines.append("| `def %s(` | %d | `%s` | %s | %s | %s | %.0f s |" % (name, len(sent), text[:70] + ("..." if len(text) > 70 else ""), "{:,}".format(cpu.cycles),
                                                                        "{:,}".format(cycles), "yes" if sent == list(cpu.tx) else "NO", seconds))
        if everything:
            lines += ["", "All requests of the suite and those named with --requests, each from reset: **in %d of %d the VHDL run sends the same bytes in the same number of cycles as the Python "
                      "simulator** (%s cycles in all, %.0f s of VHDL simulation). The table shows the first ones and any that differ." % (
                          equal, len(requests), "{:,}".format(total_cycles), total_seconds)]
    lines += ["", "%s." % ("Everything agrees" if not wrong else "%d runs DISAGREE" % wrong)]
    print("\n".join(lines[8:]))
    if write:
        open(write, "w").write("\n".join(lines) + "\n")
    return wrong


if __name__ == "__main__":
    wanted = sys.argv[sys.argv.index("--requests") + 1].split(",") if "--requests" in sys.argv else ["square", "hello_world", "is_even"]
    sys.exit(main(wanted, sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None, "--all" in sys.argv))
