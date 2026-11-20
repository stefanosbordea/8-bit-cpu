"""The export package on small random models with the tokenizer data/tok/f512a: files, formats, the round-trip
check of the complete program, that the check notices a changed file, and tools/set_prompt.py."""
import filecmp
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from test_ls_gen_model import random_model

from lsllm.export import PROMPT, export, files
from lsllm.export.check import check
from lsllm.export.tables import build, load
from lsllm.intref.model import IntModel, gumbel_table
from lsllm.ls import gen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, "data", "tok", "f512a")
SHAPE = dict(vocab_size=512, dim=16, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=24, max_seq_len=40)


def model_dir(directory, norm, ffn, **shape):
    """A random model of SPEC Part E in `directory`, with the real tokenizer (512 tokens)."""
    if not os.path.exists(os.path.join(TOK, "tok.model")):
        pytest.skip("data/tok/f512a is not on this machine")
    spec, arrays = random_model(5, norm, ffn, **dict(SHAPE, **shape))
    spec["tokenizer"] = TOK
    os.makedirs(directory)
    with open(os.path.join(directory, "model.json"), "w") as f:
        json.dump(spec, f)
    np.savez(os.path.join(directory, "model.npz"), **arrays)
    return str(directory)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """(model directory, export directory, report) of one export (greedy, filter on), shared by the tests that read it."""
    base = tmp_path_factory.mktemp("export")
    model = model_dir(base / "model", "rms", "swiglu")
    return model, str(base / "out"), export(model, str(base / "out"))


def lines(path):
    with open(path) as f:
        return f.read().split("\n")


def vectors_of(out):
    with np.load(os.path.join(out, "vectors", "vectors.npz")) as z:
        return {name: z[name] for name in z.files}


def tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_required_file_is_written_in_its_format(exported):
    model, out, report = exported
    spec, arrays, tok, tf, tables = load(model, 0.0)
    ids = tok.encode(PROMPT)
    b = build(spec, arrays, tf, ids, True, 0.0)
    for name in ["mem/data.hex", "mem/data.coe", "mem/program.hex", "mem/program.coe", "mem/program.lst", "mem/map.json",
                 "ls_tables_pkg.vhd", "MEMORY_MAP.md", "PROGRAM.md", "UART.md", "README.md", "weights/INDEX.md",
                 "vectors/README.md", "vectors/vectors.npz", "vectors/tokens.txt", "vectors/position_00.txt",
                 "vectors/token_01.txt", "vectors/uart.bin"]:
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    names = list(arrays) + ["TOKEN", "ALLOW", "LEX", "WORD"]         # greedy: no GUMBEL
    assert sorted(os.listdir(os.path.join(out, "weights"))) == sorted([name + ".hex" for name in names] + ["INDEX.md"])
    index = "\n".join(lines(os.path.join(out, "weights", "INDEX.md")))
    assert all("`%s.hex`" % name in index for name in names)

    data = lines(os.path.join(out, "mem", "data.hex"))
    assert data[-1] == "" and len(data) - 1 == len(b.image) and all(len(line) == 2 for line in data[:-1])
    assert [int(line, 16) for line in data[:-1]] == (b.image.astype(int) & 255).tolist()
    words = lines(os.path.join(out, "mem", "program.hex"))
    assert all(len(line) == 4 for line in words[:-1]) and [int(line, 16) for line in words[:-1]] == list(b.words)
    assert any(tag == "filter" for name, args, tag in b.code) and any(tag == "prompt" for name, args, tag in b.code)
    coe = lines(os.path.join(out, "mem", "data.coe"))
    assert coe[:2] == ["memory_initialization_radix=16;", "memory_initialization_vector="]
    assert coe[2:-1] == [line + "," for line in data[:-2]] + [data[-2] + ";"]
    assert lines(os.path.join(out, "mem", "program.lst"))[:-1] == gen.listing(b).split("\n")

    # the prompt is in the image, at the file lines MEMORY_MAP.md names
    left, prm = b.mem["prompt.left"][0], b.mem["prompt.ids"][0]
    assert int(data[left], 16) == len(ids)
    assert [int(data[prm + p], 16) << 8 | int(data[prm + 256 + p], 16) for p in range(len(ids))] == ids
    memory_map = "\n".join(lines(os.path.join(out, "MEMORY_MAP.md")))
    assert "| `prompt.left` | 0x%05X | %d | %d |" % (left, left + 1, left + 3) in memory_map
    assert "| `prompt.ids` + 256 + p | 0x%05X + p | %d + p | %d + p |" % (prm + 256, prm + 257, prm + 259) in memory_map
    for name, (address, size) in b.mem.items():
        assert "| `%s` | 0x%05X | %s |" % (name, address, f"{size:,}") in memory_map
    assert "%s words of 16 bits" % f"{len(b.words):,}" in memory_map and "**%d** of the 135" % -(-len(b.image) // 4096) in memory_map
    with open(os.path.join(out, "mem", "map.json")) as f:
        assert json.load(f) == {"tokenizer": TOK, "form": "single", "regions": {name: list(region) for name, region in b.mem.items()}}
    assert not os.path.exists(os.path.join(out, "PROMPTS.md"))
    assert "one prompt, written in the memory image" in "\n".join(lines(os.path.join(out, "README.md")))
    program = "\n".join(lines(os.path.join(out, "PROGRAM.md")))
    assert "+ filter **(this export)**" in program and "| prefill token" in program and "syntax filter on, greedy" in program

    e = lines(os.path.join(out, "weights", "E.hex"))[:-1]                    # two's complement, row-major
    assert [int(line, 16) - 256 * (int(line, 16) > 127) for line in e] == arrays["E"].ravel().tolist()
    assert len(report) == 5 and "equal to the integer reference" in report[2] and "from reset to HALT" in report[2]


def test_vectors_hold_the_reference_of_the_complete_program_and_its_traces(exported):
    from test_ls_gen_complete import reference            # the loop of IntCompleter as the tests of lsllm.ls write it
    model, out, report = exported
    v = vectors_of(out)
    spec, arrays = gen.load(model)
    ids, sent = v["prompt_ids"].tolist(), v["sent"].tolist()
    assert sent == v["sent_filter"].tolist() == reference(spec, arrays, ids, stop=True, syntax_filter=True)
    assert v["sent_greedy"].tolist() == reference(spec, arrays, ids, stop=True) and "sent_sampled" not in v
    assert v["uart"].tolist() == [x for t in sent for x in (0x80 | t >> 7, t & 0x7F)]

    model_ = IntModel.load(model)
    traces = []
    for t in ids:
        traces.append({})
        u = model_.step([t], traces[-1])
    assert v["position_00.position"] == 0 and v["position_00.token_in"] == ids[0]
    layer_names = {name for name in traces[0] if name[0] == "l"} | {"x0"}
    assert {name[12:] for name in v if name.startswith("position_00.")} == layer_names | {"position", "token_in"}
    for name in layer_names:
        assert (v["position_00." + name] == traces[0][name][0]).all(), name
    assert (v["position_00.l0.e"] == 127).all()             # one key: every weight is 127
    assert {name[9:] for name in v if name.startswith("token_01.")} == set(traces[-1]) | {"position", "token_in", "allowed", "token_out", "uart"}
    for name, value in traces[-1].items():
        assert (v["token_01." + name] == value[0]).all(), name
    assert any(name.endswith(".acc") for name in traces[-1]) and (v["token_01.u"] == u[0]).all()
    assert v["token_01.token_out"] == sent[0] and v["token_01.position"] == len(ids) - 1 and v["token_01.token_in"] == ids[-1]
    assert v["token_01.allowed"][sent[0]] == 1 and 0 < v["token_01.allowed"].sum() < 512
    kept = sorted(int(name[6:8]) for name in v if name.startswith("token_") and name.endswith(".token_out"))
    assert kept == list(range(1, min(8, len(sent)) + 1))
    assert [int(v["token_%02d.token_out" % g]) for g in kept] == sent[:len(kept)]

    spec, arrays, tok, tf, tables = load(model, 0.0)      # the simulator after one pass: position 0, a prefill token
    b = build(spec, arrays, tf, ids, True, 0.0)
    cpu = gen.machine(b)
    cpu.run(stop=0)
    assert (gen.peek(cpu, b, "scratch.x") == v["position_00.l1.xf"]).all() and cpu.tx == [] and not cpu.halted

    with open(os.path.join(out, "vectors", "uart.bin"), "rb") as f:        # the receiver of the laptop reads the stream
        stream = iter(f.read())
    assert list(tool("uart_receive").frames(lambda: next(stream, None))) == sent


CHANGES = [("mem/data.hex", 0), ("mem/data.hex", -2), ("mem/program.hex", 5), ("mem/data.coe", 7), ("mem/program.coe", -2),
           ("weights/E.hex", 3), ("weights/NORM.hex", 100), ("weights/TOKEN.hex", 600), ("vectors/position_00.txt", 5),
           ("vectors/token_01.txt", 5), ("vectors/tokens.txt", 1), ("ls_tables_pkg.vhd", 60), ("ls_tables_pkg.vhd", -3)]


@pytest.mark.parametrize("name, line", CHANGES)
def test_the_check_fails_when_one_number_of_a_file_changes(exported, tmp_path, name, line):
    model, out, report = exported
    copy = str(tmp_path / "out")
    shutil.copytree(out, copy)
    text = lines(os.path.join(copy, name))
    digit = next(g for g, ch in enumerate(text[line]) if ch in "0123456789ABCDEF")
    text[line] = text[line][:digit] + ("1" if text[line][digit] == "0" else "0") + text[line][digit + 1:]
    with open(os.path.join(copy, name), "w") as f:
        f.write("\n".join(text))
    with pytest.raises(ValueError, match="export check failed"):
        check(model, copy)


def test_the_check_fails_on_another_setting_a_stale_file_and_another_prompt_in_the_image(exported, tmp_path):
    model, out, report = exported
    copy = str(tmp_path / "out")
    shutil.copytree(out, copy)
    check(model, copy)                                  # the copy is good
    with pytest.raises(ValueError, match="export check failed"):
        check(model, copy, syntax_filter=False)         # the files hold the program with the filter
    with pytest.raises(ValueError, match="list of weights"):
        check(model, copy, temperature=0.7)             # no GUMBEL.hex in a greedy export
    with open(os.path.join(copy, "vectors", "token_30.txt"), "w") as f:
        f.write("position 1\n3\n")
    with pytest.raises(ValueError, match="token_30"):
        check(model, copy)
    os.remove(os.path.join(copy, "vectors", "token_30.txt"))
    check(model, copy)
    tool("set_prompt").set_prompt(os.path.join(copy, "mem"), "x = 1")
    with pytest.raises(ValueError, match="data.hex"):
        check(model, copy)                              # the vectors describe the old prompt


def test_set_prompt_tool_gives_the_image_of_an_export_with_that_prompt(exported, tmp_path):
    """A shorter prompt over a longer one, through the command line; then the image runs to the new reference."""
    model, out, report = exported
    copy, fresh, text = str(tmp_path / "copy"), str(tmp_path / "fresh"), "def f(x):\n    return x"
    shutil.copytree(out, copy)
    done = subprocess.run([sys.executable, os.path.join("tools", "set_prompt.py"), os.path.join(copy, "mem"), r"def f(x):\n    return x"],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    export(model, fresh, text)                          # checked by its own round trip
    ids = vectors_of(fresh)["prompt_ids"].tolist()
    assert done.stdout.startswith("%d prompt tokens written" % len(ids)) and " ".join(map(str, ids)) in done.stdout
    for name in ("data.hex", "data.coe", "program.hex", "map.json"):
        assert filecmp.cmp(os.path.join(copy, "mem", name), os.path.join(fresh, "mem", name), shallow=False), name
    assert not filecmp.cmp(os.path.join(copy, "mem", "data.hex"), os.path.join(out, "mem", "data.hex"), shallow=False)
    with pytest.raises(ValueError):                     # more tokens than the context of 40
        tool("set_prompt").set_prompt(os.path.join(copy, "mem"), "x = [" + "1, " * 60 + "1]")


def test_the_smallest_shapes_sampling_without_filter_and_a_prompt_of_one_token(tmp_path):
    """One layer, one RoPE pair, dyt and relu: aggregates of one element (`(0 => x)`), no W3; the program samples."""
    model = model_dir(tmp_path / "model", "dyt", "relu", dim=4, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=6)
    out = str(tmp_path / "out")
    report = export(model, out, "", tokens=3, syntax_filter=False, temperature=0.9)
    assert "syntax filter off, temperature 0.9)" in report[2]
    text = "\n".join(lines(os.path.join(out, "ls_tables_pkg.vhd")))
    assert "constant SH_Q : int_vector_t(0 to 0) :=\n    (0 => " in text and "ROPE_C_TABLE : byte_matrix_t(0 to 39, 0 to 0)" in text
    found = files.read_vhdl(text)
    assert found["EXP_TABLE"].shape == (1, 256) and found["DIM"] == 4 and found["C_S"].shape == (1, 2)
    assert (found["GUMBEL_TABLE"] == gumbel_table(0.9, 4)).all()
    assert (files.read_hex(os.path.join(out, "weights", "GUMBEL.hex")) == gumbel_table(0.9, 4)).all()
    assert not os.path.exists(os.path.join(out, "weights", "W3_0.hex"))
    if shutil.which("ghdl") or shutil.which("nvc"):
        assert "analysed without errors" in report[-1]

    from test_ls_gen_complete import reference
    v = vectors_of(out)
    spec, arrays = gen.load(model)
    ids = v["prompt_ids"].tolist()
    assert len(ids) == 1 and "position_00.u" in v and "token_01.allowed" not in v     # position 0 chooses token 1
    assert v["sent"].tolist() == v["sent_sampled"].tolist() == reference(spec, arrays, ids, stop=True, temperature=0.9)
    assert v["sent_greedy"].tolist() == reference(spec, arrays, ids, stop=True) != v["sent"].tolist()
    b = gen.build(spec, arrays, prompt=True, tf=load(model, 0.9)[3], temperature=0.9)
    assert any(tag == "sample" for name, args, tag in b.code) and not any(tag == "filter" for name, args, tag in b.code)
    assert lines(os.path.join(out, "mem", "program.hex"))[:-1] == ["%04X" % word for word in b.words]
    report = export(model, out, "", tokens=3)           # the same directory again, greedy: the old GUMBEL.hex goes
    assert not os.path.exists(os.path.join(out, "weights", "GUMBEL.hex")) and "syntax filter on, greedy" in report[2]


def test_command_line_prompt_tokens_no_filter_and_temperature(tmp_path):
    model = model_dir(tmp_path / "model", "dyt", "swiglu")
    out = str(tmp_path / "out")
    done = subprocess.run([sys.executable, "-m", "lsllm.export", model, out, "--prompt", r"def f(x):\n    return x",
                           "--tokens", "2", "--no-filter", "--temperature", "0.5"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "syntax filter off, temperature 0.5)" in done.stdout
    gumbel = files.read_hex(os.path.join(out, "weights", "GUMBEL.hex"))
    assert (gumbel == gumbel_table(0.5, 4)).all() and (gumbel != gumbel_table(0.8, 4)).any()
    with open(os.path.join(out, "vectors", "README.md")) as f:
        assert "def f(x):\n    return x\n" in f.read()
    assert not os.path.exists(os.path.join(out, "vectors", "token_03.txt"))
    check(model, out, syntax_filter=False, temperature=0.5)


def test_a_prompt_that_does_not_fit_the_context_is_refused(tmp_path):
    model = model_dir(tmp_path / "model", "dyt", "relu", max_seq_len=16)
    with pytest.raises(ValueError, match="a prompt has 1 to"):
        export(model, str(tmp_path / "out"), PROMPT)


# ---- the prompt table chosen by the switches, and prompts over the UART ----

TEXTS = ["def add(", "def f(x):\n    return x", "x = 1", "def bubble_sort(", "for k in range(3):"]


def test_prompt_table_form_three_switch_settings_and_one_more(tmp_path):
    """Sampling, so that the switches matter twice: they choose the prompt and seed the random generator."""
    from test_ls_gen_ask import answer                  # the reference loop of the tests of lsllm.ls, with the seed
    model, out = model_dir(tmp_path / "model", "rms", "relu"), str(tmp_path / "out")
    report = export(model, out, prompts=TEXTS, temperature=0.7)
    assert len(report) == 7 and ["switches %d (" % k in line for k in (0, 2, 4) for line in report[2:5]].count(True) == 3
    assert "`def f(x):\\n    return x`" not in report[2] and "prompt `x = 1`" in report[3] and "temperature 0.7" in report[4]
    spec, arrays, tok, tf, tables = load(model, 0.7)
    lists = [tok.encode(text) for text in TEXTS]
    b = build(spec, arrays, tf, lists[0], True, 0.7, lists)
    assert lines(os.path.join(out, "mem", "data.hex"))[:-1] == ["%02X" % (v & 255) for v in b.image.tolist()]
    assert vectors_of(out)["prompt_ids"].tolist() == lists[0]
    cpu = gen.machine(b, sw=3)                          # a setting that the check of the export did not run
    cpu.run(limit=10 ** 8)
    assert cpu.halted and gen.sent(cpu) == answer(spec, arrays, lists[3], True, True, 0.7, seed=3)[0]

    prompts = "\n".join(lines(os.path.join(out, "PROMPTS.md")))
    assert "| `000011` | 3 | `def bubble_sort(` | %d | %s |" % (len(lists[3]), " ".join(map(str, lists[3]))) in prompts
    assert "| `000001` | 1 | `def f(x):\\n    return x` |" in prompts and "5 prompts" in prompts
    with open(os.path.join(out, "mem", "map.json")) as f:
        assert json.load(f)["form"] == "table"
    program, memory = ("\n".join(lines(os.path.join(out, name))) for name in ("PROGRAM.md", "MEMORY_MAP.md"))
    start = sum(tag == "start" for name, args, tag in b.code)
    assert start == gen.top(b) > 0 and "`start` in `mem/program.lst`, %d words" % start in program
    cpu = gen.machine(b)
    cpu.run(stop=gen.top(b))                            # the start-up piece alone, switches 0
    assert "(%d tokens): %d cycles per answer" % (len(lists[0]), cpu.cycles) in program and cpu.cycles > 24 * len(lists[0])
    assert "`prompts.index` at 0x%05X" % b.mem["prompts.index"][0] in memory and "| `prompts.table` | 0x%05X |" % b.mem["prompts.table"][0] in memory
    assert "a table of prompts in the memory image, chosen by the switches" in "\n".join(lines(os.path.join(out, "README.md")))

    with pytest.raises(ValueError, match="table chosen by the switches"):
        tool("set_prompt").set_prompt(os.path.join(out, "mem"), "x = 1")
    for wrong in (dict(prompts=TEXTS[:4]), dict(), dict(uart_prompt=True)):     # another table, or another form
        with pytest.raises(ValueError, match="export check failed"):
            check(model, out, temperature=0.7, **wrong)
    with pytest.raises(ValueError):
        export(model, out, prompts=["x = 1"] * 65)
    with pytest.raises(ValueError):
        export(model, out, prompts=TEXTS, uart_prompt=True)


def test_uart_prompt_form_two_prompts_in_one_run(tmp_path):
    from test_ls_gen_ask import answer
    model, out, text = model_dir(tmp_path / "model", "rms", "swiglu"), str(tmp_path / "out"), "def f(x):\n    return x"
    report = export(model, out, text, uart_prompt=True, temperature=0.7)
    assert len(report) == 5 and "prompts over the UART" in report[2] and "`def add(`" in report[2] and "no prompt inside" in report[1]
    spec, arrays, tok, tf, tables = load(model, 0.7)
    b = gen.build(spec, arrays, uart_prompt=True, tf=tf, syntax_filter=True, temperature=0.7)
    assert lines(os.path.join(out, "mem", "data.hex"))[:-1] == ["%02X" % (v & 255) for v in b.image.tolist()]
    assert lines(os.path.join(out, "mem", "program.hex"))[:-1] == ["%04X" % word for word in b.words]
    assert b.image[b.mem["prompt.left"][0]] == 0 and "JMP" in lines(os.path.join(out, "mem", "program.lst"))[-2]
    v = vectors_of(out)
    first, second = tok.encode("def add("), v["prompt_ids"].tolist()         # the other order than in the check
    assert second == tok.encode(text)
    cpu = gen.machine(b)
    cpu.feed(gen.frame(first) + gen.frame(second))
    cpu.run(limit=10 ** 8)
    one, lfsr = answer(spec, arrays, first, True, True, 0.7)
    two, lfsr = answer(spec, arrays, second, True, True, 0.7, lfsr=lfsr)     # the random generator goes on
    assert cpu.waiting and gen.answers(cpu) == [one, two, []] and v["sent"].tolist() == answer(spec, arrays, second, True, True, 0.7)[0]

    uart = "\n".join(lines(os.path.join(out, "UART.md")))
    assert "## From the laptop to the board: the prompt" in uart and "the receive pin of the CPU" in uart
    assert "## From the laptop to the board" not in "\n".join(docs_uart_of_the_single_form())
    assert "prompts typed over the UART" in "\n".join(lines(os.path.join(out, "README.md")))
    assert "No prompt is in the image" in "\n".join(lines(os.path.join(out, "MEMORY_MAP.md")))
    assert "it never halts" in "\n".join(lines(os.path.join(out, "PROGRAM.md")))
    with pytest.raises(ValueError, match="the UART"):
        tool("set_prompt").set_prompt(os.path.join(out, "mem"), "x = 1")
    with pytest.raises(ValueError, match="export check failed"):
        check(model, out, temperature=0.7)              # the files are not the program with one prompt


def docs_uart_of_the_single_form():
    from lsllm.export import docs
    return docs.uart(TOK, 1).split("\n")


def test_command_line_asks_prompts_file_and_exclusive_forms(tmp_path):
    from lsllm.asks import ASKS, load_jsonl, request_to_prompt
    from lsllm.filter_table import TableFilter
    model = model_dir(tmp_path / "model", "dyt", "relu", n_layers=1)
    run = lambda *args: subprocess.run([sys.executable, "-m", "lsllm.export", model, *args], cwd=ROOT, capture_output=True, text=True)
    done = run(str(tmp_path / "asks"), "--asks", "3", "--no-docstring")
    assert done.returncode == 0, done.stderr
    asked = [request_to_prompt(ask["request"]) for ask in load_jsonl(ASKS)[:3]]
    assert asked[0] == "def bubble_sort(" and all("prompt `%s`" % text in done.stdout for text in asked)
    table = "\n".join(lines(tmp_path / "asks" / "PROMPTS.md"))
    assert all("| `%s` | %d | `%s` |" % (format(k, "06b"), k, text) in table for k, text in enumerate(asked))
    spec, arrays, tok, tf, tables = load(model, 0.0)
    allow = np.frombuffer(TableFilter(tok, triple_quotes=False).allow_table(), dtype=np.uint8)
    assert files.read_hex(tmp_path / "asks" / "weights" / "ALLOW.hex").tolist() == allow.tolist() != tables["ALLOW"].ravel().tolist()

    with open(tmp_path / "prompts.txt", "w") as f:
        f.write("def add(\n\ndef f(x):\\n    return x\n")       # an empty line is skipped; \n inside a prompt
    done = run(str(tmp_path / "file"), "--prompts", str(tmp_path / "prompts.txt"), "--no-filter")
    assert done.returncode == 0, done.stderr
    assert "switches 1 (prompt `def f(x):\\n    return x`)" in done.stdout and "syntax filter off" in done.stdout
    assert "2 prompts" in "\n".join(lines(tmp_path / "file" / "PROMPTS.md"))
    done = run(str(tmp_path / "both"), "--asks", "2", "--uart-prompt")
    assert done.returncode == 2 and "not allowed with" in done.stderr and not os.path.exists(tmp_path / "both")
