"""The round-trip check of an export: read the files back and compare them with the model and the integer reference.

    weights/*.hex                  against the arrays
    mem/data.hex, program.hex      against the Build of lsllm.ls.gen with the prompt of the test vectors in its
                                   image; the .coe files against the .hex files; mem/map.json against Build.mem
    the two .hex files in the simulator (lsllm.ls.iss.L1), from reset to HALT: the UART stream against the
                                   integer reference (IntModel and TableFilter in the loop of IntCompleter),
                                   the end mark included. With a prompt table: for the first, a middle and the
                                   last switch setting. With prompts over the UART: two prompts in one run
    vectors/*.txt                  against vectors.npz
    ls_tables_pkg.vhd              analysed by ghdl or nvc if one is installed; its constants parsed back and compared
Any difference raises ValueError.
"""
import glob
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

from lsllm.export import files
from lsllm.export.docs import frame
from lsllm.export.tables import build, form_of, load, numbers, small
from lsllm.export.vectors import reference
from lsllm.intref.model import IntModel
from lsllm.ls import gen


def same(found, expected, what):
    found, expected = np.asarray(found), np.asarray(expected)
    if found.shape != expected.shape or (found != expected).any():
        raise ValueError("export check failed: %s differs" % what)


def analyse(path):
    """Analyse the VHDL package with ghdl, or with nvc, if installed. Returns what was done."""
    with tempfile.TemporaryDirectory() as work:
        for command in (["ghdl", "-a", "--std=08", "--workdir=" + work, path],
                        ["nvc", "--std=2008", "--work=work:" + os.path.join(work, "work"), "-a", path]):
            if shutil.which(command[0]):
                done = subprocess.run(command, capture_output=True, text=True)
                if done.returncode:
                    raise ValueError("export check failed: %s rejects %s\n%s" % (command[0], path, done.stdout + done.stderr))
                return "analysed without errors by `%s`" % " ".join(command[:2] + ["-a"])
    return "NOT analysed by a VHDL tool (neither ghdl nor nvc is installed)"


ASK = "def add("          # the second prompt that the check sends to a program with prompts over the UART


def check(model_dir, out_dir, syntax_filter=True, temperature=0.0, prompts=None, uart_prompt=False):
    """Returns the lines of the report; raises ValueError at the first difference. prompts: the texts of the
    prompt table; uart_prompt: the program reads its prompts from the UART (the forms of lsllm.export.tables.build)."""
    spec, arrays, tok, tf, tables = load(model_dir, temperature)
    path = lambda *parts: os.path.join(out_dir, *parts)

    same(sorted(name[:-4] for name in os.listdir(path("weights")) if name.endswith(".hex")), sorted(tables), "the list of weights/*.hex")
    for name, value in tables.items():
        same(files.read_hex(path("weights", name + ".hex")), value.ravel().astype(np.int64) & 255, "weights/%s.hex" % name)
    report = ["weights: %d .hex files read back, %s bytes, all equal to the arrays" % (len(tables), f"{sum(v.size for v in tables.values()):,}")]

    with np.load(path("vectors", "vectors.npz")) as z:
        v = {name: z[name] for name in z.files}
    ids = v["prompt_ids"].tolist()
    lists = None if prompts is None else [tok.encode(text) for text in prompts]
    b = build(spec, arrays, tf, ids, syntax_filter, temperature, lists, uart_prompt)
    image, words = files.read_hex(path("mem", "data.hex")), files.read_hex(path("mem", "program.hex"))
    same(image, b.image.astype(np.int64) & 255, "mem/data.hex")
    same(words, b.words, "mem/program.hex")
    same(files.read_coe(path("mem", "data.coe")), image, "mem/data.coe")
    same(files.read_coe(path("mem", "program.coe")), words, "mem/program.coe")
    with open(path("mem", "map.json")) as f:
        if json.load(f)["regions"] != {name: list(region) for name, region in b.mem.items()}:
            raise ValueError("export check failed: mem/map.json differs from the memory map of the Build")
    inside = {"single": "the %d prompt tokens of the test vectors inside" % len(ids), "uart": "no prompt inside",
              "table": "the table of %d prompts inside" % len(prompts or ())}[form_of(prompts, uart_prompt)]
    report.append("memory: data.hex and data.coe (%s bytes, %s), program.hex and program.coe (%s words) and map.json "
                  "equal to the Build" % (f"{len(image):,}", inside, f"{len(words):,}"))

    written = b._replace(image=image.astype(np.uint8).view(np.int8), words=words.tolist())      # what the files hold
    model = IntModel.load(model_dir)
    setting = "stop rule on, syntax filter %s, %s" % ("on" if syntax_filter else "off",
                                                      "temperature %g" % temperature if temperature > 0 else "greedy")

    def run(sw=0, received=()):
        """The files in the simulator from reset until the CPU halts or waits for the UART."""
        cpu = gen.machine(written, sw=sw)
        cpu.feed(received)
        cpu.run(limit=10 ** 9)
        if cpu.overflow:
            raise ValueError("export check failed: the accumulator overflowed in the simulator")
        return cpu

    def differs(cpu, expected):
        raise ValueError("export check failed: the UART bytes of the simulator differ from the integer reference\n"
                         "simulator %s\nreference %s" % (gen.answers(cpu), expected))

    if uart_prompt:                                     # two prompts in one run: the one of the vectors, then ASK
        asked = [ids, tok.encode(ASK)]
        expected = [reference(model, tok, tf, asked[0], syntax_filter, temperature)[0]]
        expected.append(reference(model, tok, tf, asked[1], syntax_filter, temperature, lfsr=model.lfsr)[0])
        same(v["sent"], expected[0], "the tokens of vectors.npz and of the integer reference")
        cpu = run(received=gen.frame(asked[0]) + gen.frame(asked[1]))
        if cpu.tx != frame(expected[0]) + [gen.END] + frame(expected[1]) + [gen.END] or cpu.halted or not cpu.waiting:
            differs(cpu, expected)
        report.append("simulator on data.hex and program.hex, prompts over the UART (%s): the prompt of the vectors (%d tokens) "
                      "and `%s` (%d tokens) sent in one run, switches 0; answers %s and %s, each followed by the end byte 0xFF, then the "
                      "CPU waits; %d UART bytes equal to the integer reference; no accumulator overflow; %s cycles = %.2f s at 100 MHz"
                      % (setting, len(asked[0]), ASK, len(asked[1]), expected[0], expected[1], len(cpu.tx), f"{cpu.cycles:,}", cpu.cycles / 1e8))
    else:                                               # from reset to HALT, for one prompt or for three switch settings
        chosen = [0] if prompts is None else sorted({0, len(prompts) // 2, len(prompts) - 1})
        for sw in chosen:
            mine = ids if prompts is None else lists[sw]
            expected, steps, ended = reference(model, tok, tf, mine, syntax_filter, temperature, seed=sw)
            if sw == 0:
                same(v["sent"], expected, "the tokens of vectors.npz and of the integer reference")
            cpu = run(sw)
            if cpu.tx != frame(expected) or not cpu.halted:
                differs(cpu, expected)
            which = ", switches 0" if prompts is None else ", switches %d (prompt `%s`)" % (sw, prompts[sw].replace("\n", "\\n"))
            report.append("simulator on data.hex and program.hex from reset to HALT (%s)%s: %d tokens sent %s, %s; %d UART bytes "
                          "equal to the integer reference; no accumulator overflow; %s cycles = %.2f s at 100 MHz for %d prompt "
                          "tokens and %d generated ones"
                          % (setting, which, len(expected), expected, "the last one is the end mark" if ended else "then the context is full",
                             len(cpu.tx), f"{cpu.cycles:,}", cpu.cycles / 1e8, len(mine), len(expected)))

    texts = sorted(glob.glob(path("vectors", "*.txt")))
    for name in texts:
        with open(name) as f:
            step = files.read_vector_text(f.read())
        key = os.path.basename(name)[:-4]
        key = "" if key == "tokens" else key + "."
        names = [name for name in v if name.startswith(key)] if key else [name for name in v if "." not in name]
        same(sorted(key + name for name in step), sorted(names), "the names of " + name)
        for part, value in step.items():
            same(value.ravel(), v[key + part].ravel(), "%s, %s" % (name, part))
    same(len(texts), len({name.split(".")[0] for name in v if "." in name}) + 1, "the number of vectors/*.txt files")
    report.append("vectors: %d text files read back, equal to vectors.npz" % len(texts))

    wanted = {name.upper(): value for name, value in numbers(spec).items()}
    wanted.update((name + "_TABLE", value) for name, value in small(tables).items())
    with open(path("ls_tables_pkg.vhd")) as f:
        text = f.read()
    found = files.read_vhdl(text)
    same(sorted(found), sorted(wanted), "the constants of ls_tables_pkg.vhd")
    for name, value in wanted.items():
        same(np.ravel(found[name]), np.ravel(value), "ls_tables_pkg.vhd, " + name)
    if text.count("(") != text.count(")"):
        raise ValueError("export check failed: brackets of ls_tables_pkg.vhd")
    report.append("ls_tables_pkg.vhd: %s; %d constants parsed back by Python, all equal to the model" % (analyse(path("ls_tables_pkg.vhd")), len(found)))
    return report
