"""Export of an integer model (docs/SPEC.md Part E) for the board: everything the hardware side needs, as files.

    .venv/bin/python -m lsllm.export MODEL_DIR OUT_DIR [--prompt TEXT] [--tokens 8] [--no-filter] [--temperature 0]
                                     [--no-docstring] [--prompts FILE | --asks N | --uart-prompt]

The program takes its prompt in one of three forms: one prompt written in the image (the default, --prompt); a
table of up to 64 prompts in the image, chosen by the switches SW5..0 (--prompts FILE: one prompt per line, \\n
for a line break; --asks N: the first N requests of data/asks/asks.jsonl as `def name(`); or prompts typed over
the UART (--uart-prompt, needs the proposed instruction RX).

OUT_DIR receives
    weights/<name>.hex, INDEX.md   every array of model.npz, the syntax-filter tables, GUMBEL (for a temperature
                                   > 0): one byte per line
    mem/data.hex, data.coe         the data memory at reset of the complete program of lsllm.ls.gen, with the
                                   prompt in it, one byte per line; mem/map.json: its regions
    mem/program.hex, .coe, .lst    the program (prompt feeding, stop rule, syntax filter unless --no-filter,
                                   sampling for a temperature > 0), one 16-bit word per line, and its listing
    ls_tables_pkg.vhd              the small tables, shifts and preloads as VHDL-2008 constants
    MEMORY_MAP.md, PROGRAM.md, UART.md, and PROMPTS.md for a prompt table
    vectors/                       what the program must send, every intermediate value of position 0 and of the
                                   first generated tokens
    README.md                      what is where, and the result of the round-trip check
The formats are described in lsllm/export/files.py. The export ends with lsllm.export.check.check, which reads the
files back, runs them in the simulator from reset to HALT and compares with the integer reference; a difference
raises an error (three switch settings of a prompt table; two prompts in one run over the UART).
tools/set_prompt.py puts another prompt into mem/data.hex and mem/data.coe of the form with one prompt.
"""
import glob
import json
import os

from lsllm.export import docs, files
from lsllm.export.check import check
from lsllm.export.tables import build, form_of, load, numbers, small
from lsllm.export.vectors import vectors
from lsllm.ls import gen

PROMPT = 'def add(a, b):\n    """Return the sum of a and b."""'


def export(model_dir, out_dir, prompt=PROMPT, tokens=8, syntax_filter=True, temperature=0.0, prompts=None, uart_prompt=False):
    """Write the export of `model_dir` into `out_dir` and check it. Returns the lines of the check report.

    prompt: the prompt in the image and of the test vectors. prompts: instead, a list of up to 64 prompt texts
    for the table from which the switches choose (the vectors then use prompt 0). uart_prompt: instead, the program
    reads its prompts from the UART (the vectors use `prompt`)."""
    spec, arrays, tok, tf, tables = load(model_dir, temperature)
    form, lists = form_of(prompts, uart_prompt), None
    if prompts is not None:
        prompt, lists = prompts[0], [tok.encode(text, bos=True, eos=False) for text in prompts]
    ids = tok.encode(prompt, bos=True, eos=False)
    for name in ("weights", "mem", "vectors"):
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    def write(name, text):
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text)

    for old in glob.glob(os.path.join(out_dir, "weights", "*.hex")):        # of an earlier export into this directory
        os.remove(old)
    for name, value in tables.items():
        files.write_hex(os.path.join(out_dir, "weights", name + ".hex"), value)
    write("weights/INDEX.md", docs.index(tables))
    b = build(spec, arrays, tf, ids, syntax_filter, temperature, lists, uart_prompt)
    vectors(model_dir, tok, tf, prompt, ids, tokens, syntax_filter, temperature, os.path.join(out_dir, "vectors"), form)
    files.write_memory(os.path.join(out_dir, "mem"), "data", b.image, 2)
    files.write_memory(os.path.join(out_dir, "mem"), "program", b.words, 4)
    write("mem/program.lst", gen.listing(b) + "\n")
    write("mem/map.json", json.dumps({"tokenizer": spec["tokenizer"], "form": form, "regions": b.mem}, indent=1) + "\n")
    write("ls_tables_pkg.vhd", files.vhdl(os.path.basename(os.path.abspath(model_dir)), small(tables), numbers(spec)))
    write("MEMORY_MAP.md", docs.memory_map(b, tables, ids, form))
    write("PROGRAM.md", docs.program(b, spec, syntax_filter, temperature, form, ids))
    write("UART.md", docs.uart(spec["tokenizer"], tok.level_after(ids), form))
    if os.path.exists(os.path.join(out_dir, "PROMPTS.md")):                 # of an earlier export with a prompt table
        os.remove(os.path.join(out_dir, "PROMPTS.md"))
    if prompts is not None:
        write("PROMPTS.md", docs.prompts_md(prompts, lists))
    report = check(model_dir, out_dir, syntax_filter, temperature, prompts, uart_prompt)
    write("README.md", readme(model_dir, syntax_filter, temperature, report, form))
    return report


def readme(model_dir, syntax_filter, temperature, report, form="single"):
    return "\n".join([
        "# Export of the integer model `%s` for the LS CPU" % model_dir, "",
        "Written by `.venv/bin/python -m lsllm.export`. Do not edit by hand. The program in `mem/` is the complete",
        "program: it takes a prompt, generates until the text ends and sends every token over the UART; stop rule on,",
        "syntax filter %s, %s." % ("on" if syntax_filter else "off", "sampling at temperature %g" % temperature if temperature > 0 else "greedy"),
        "Form of the prompt: **%s**%s." % (docs.FORMS[form], {
            "single": "; the program halts after the answer",
            "table": " (`PROMPTS.md` lists them); the program halts after the answer",
            "uart": "; after every answer the program sends the byte 0xFF and waits for the next prompt (`UART.md`)"}[form]), "",
        "| file | content |", "|---|---|",
        "| `mem/data.hex`, `mem/data.coe` | data memory at reset%s, one byte per line; the `.coe` is for the Vivado block memory generator |"
        % {"single": " with the prompt in it", "table": " with the prompt table in it", "uart": ""}[form],
        "| `mem/map.json` | the regions of the data memory, for `tools/set_prompt.py` and for test benches |",
        "| `mem/program.hex`, `mem/program.coe`, `mem/program.lst` | program ROM, one 16-bit word per line; listing |",
        "| `MEMORY_MAP.md` | every region of the data memory, how to change the prompt, block RAM count, ROM size |",
        "| `PROGRAM.md` | what the program does per token, ROM words and cycles per step, tokens per second |",
        "| `weights/`, `weights/INDEX.md` | every array on its own, one byte per line |",
        "| `ls_tables_pkg.vhd` | VHDL-2008 package: small tables, shifts, preloads |",
        "| `vectors/`, `vectors/README.md` | test vectors: what the program must send, every intermediate value of position 0 and of the first generated tokens |",
        "| `UART.md` | baud rate, frame, pins, receiver on the laptop |", "",
        "## Round-trip check at the end of the export", ""] + ["- " + line for line in report] + [""])
