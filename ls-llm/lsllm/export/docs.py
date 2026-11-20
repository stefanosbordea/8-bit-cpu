"""The Markdown files of the export: weights/INDEX.md, MEMORY_MAP.md, PROGRAM.md, UART.md, vectors/README.md.

Every number in them is taken from the model files, from the Build of lsllm.ls.gen or from lsllm.ls.cost.
"""
import re
from collections import Counter

import numpy as np

import lsllm.intref.model
from lsllm.config import ModelConfig
from lsllm.ls import cost, gen

MEANING = {  # name of an array without its layer number -> what it is
    "E": "embedding matrix, row = token id (SPEC C.1); the same matrix is the output layer (C.9)",
    "Wq": "query projection of layer l, row = output component (SPEC C.3)",
    "Wk": "key projection of layer l (C.3)",
    "Wv": "value projection of layer l (C.3)",
    "Wo": "attention output projection of layer l, added to the residual stream (C.7)",
    "W1": "feed-forward, first matrix of layer l (C.8)",
    "W3": "feed-forward, gate matrix of layer l (swiglu only, C.8)",
    "W2": "feed-forward, output matrix of layer l, added to the residual stream (C.8)",
    "NORM": "norm tables, row = norm site n = 0 .. 2L (attention site of layer l: 2l, feed-forward site: 2l + 1, final: 2L), index = unsigned byte (C.2)",
    "SILU": "SiLU table, index = unsigned byte of h1 (C.8)",
    "EXP": "attention weight tables, row = layer, index = (code - row maximum) mod 256 (C.6 step 3)",
    "RECIP": "reciprocal table min(127, round(8192 / v)), used at v = 64..127 (C.6 step 5)",
    "ROPE_C": "round(127 cos), row = position, column = pair j (C.4)",
    "ROPE_S": "round(127 sin), row = position, column = pair j (C.4)",
    "TOKEN": "syntax filter, row 0 = class of every token id, row 1 = its word column (lsllm/filter_table.py)",
    "ALLOW": "syntax filter, ALLOW[abstract state][class] = 1 if a token of that class may come; rows padded to 32 bytes",
    "LEX": "syntax filter, LEX[lexer state][class] = next lexer state; rows padded to 32 bytes",
    "WORD": "syntax filter, WORD[word column][word state] = next word state; rows padded to 64 bytes",
    "GUMBEL": "sampling noise for u >> 1 at the temperature of this export, index = random byte of RND (SPEC Part D)",
}
REGIONS = [  # (beginning of a name of Build.mem, content); the first that fits is used
    ("tables", "lookup tables of 256 bytes each for `LUT`; this address is LUT_BASE (table t is at LUT_BASE + 256 t, list below)"),
    ("tok", "token table: per token id the address of its row of E, as three columns [low byte], [bits 11..8], [bits 19..12]"),
    ("scratch.rec", "record of the current position, a copy of one row of pos.records (at reset: the link to record 0)"),
    ("scratch.token", "the current token id as [high byte, low byte]; the program leaves the token it chose here"),
    ("scratch.r", "multiplier of an rms norm"),
    ("scratch.x", "residual stream x (D bytes)"),
    ("scratch.y", "output of the norm in use"),
    ("scratch.q", "queries q"),
    ("scratch.k", "keys k"),
    ("scratch.v", "values v"),
    ("scratch.o", "attention output o"),
    ("scratch.head", "record of the current head, then its rotated query"),
    ("scratch.e", "attention weights e of the current head (one byte per key), then k, r and a page number"),
    ("scratch.g1", "feed-forward: SILU(h1)"),
    ("scratch.h3", "feed-forward: h3"),
    ("scratch.g", "feed-forward: hidden vector g"),
    ("scratch.u", "logits u, one byte per token id (0..127)"),
    ("scratch", "variable of the token being computed"),
    ("prompt.left", "number of prompt tokens still to feed; at reset the length of the prompt (see The prompt)"),
    ("prompt.ids", "the prompt: byte p = high byte of token p, byte 256 + p = low byte (see The prompt)"),
    ("prompts.index", "prompt table: 4 bytes per switch value 0..63: address of its prompt in prompts.table as [bits 11..8, low byte, bits 19..12], then its number of tokens"),
    ("prompts.table", "prompt table: per prompt the high bytes of its token ids, then the low bytes (see PROMPTS.md)"),
    ("filter.class", "class of every token id (TOKEN row 0), for the stop rule and the syntax filter"),
    ("filter.rows", "16 rows of 256 bytes: level tables, abstract-state tables, bracket tables, the bracket stack, and in row 15 the filter state (see lsllm/ls/gen.py)"),
    ("filter.khi", "syntax filter: word column of every token id, upper part of the address into filter.word"),
    ("filter.klo", "syntax filter: word column of every token id, lower part of the address into filter.word"),
    ("filter.word", "syntax filter: byte 64 k + w = word state after a token of column k in word state w (WORD)"),
    ("filter.allow", "syntax filter: byte 32 a + c = 1 if class c may come in abstract state a (ALLOW)"),
    ("ones", "T bytes that hold 1 (a stream of e against them gives the sum of the weights)"),
    ("pos", "position records: link to the next record, number of keys, key address, RoPE row of the position"),
    ("heads", "head records, one per layer and query head: addresses of q, keys, value rows, o, and the preload c_s"),
    ("kv.V", "KV cache, written at run time: value row V[layer].[key/value head].[component], one byte per position"),
    ("kv.K", "KV cache, written at run time: keys K[layer].[key/value head], one row of d bytes per position"),
    ("w.", "weight matrix, row-major, the bytes of weights/%s.hex"),
]
STEPS = {  # component of lsllm.ls.gen -> what the program does there
    "embedding": "Read the token id, look up the address of its row of E in the token table, copy the row to x (SPEC C.1).",
    "norm": "Norm of x into y: 2 per layer and the final one. dyt: one table lookup per byte; rms: sum of squares, RQ, table, one product per byte (C.2).",
    "qkv": "q = Wq y, k = Wk y, v = Wv y: one stream per row, RQ with rounding, store (C.3).",
    "rope": "Rotate every query head and every key head with the RoPE row of the position: two products per value, shift 7 (C.4).",
    "cache": "Write the rotated keys to K[l][g][p] and every value component to its row V[l][g][j][p] (C.5).",
    "scores": "Per query head: one stream per key (q' . K[t] + c_s), RQ with ReLU to a code 0..127, running maximum; then e[t] = EXP_l[code - maximum] (C.6 steps 1 to 3).",
    "mix": "Per query head: Dsum = sum of e (a stream against the row of ones), `NRM`, `RQV`, RECIP table; per value row one stream e . V[j], shift by k, times r, shift sh_o: the head output o (C.6 steps 4 and 5).",
    "attn_out": "x <- rq(256 x + 128 + Wo o, 8): one stream per row with x[i] as the preload (C.7).",
    "ffn": "Feed-forward matrices: h1 = W1 y, h3 = W3 y (swiglu) or g = relu(W1 y); then x <- rq(256 x + 128 + W2 g, 8) (C.8).",
    "gating": "swiglu only: g1 = SILU[h1] by table, g = g1 * h3 with shift sh_g (C.8).",
    "output": "After the final norm: u[j] = rq(c_l + E[j] . y, sh_l) with ReLU for every token id j: V streams of length D (C.9).",
    "argmax": "The smallest j with the largest u[j]: `SUB`, sign table, `BZ` per logit (C.9, Part D).",
    "book": "Bookkeeping: copy the position record, loop over the head records, store the chosen token, move to the next position, send the two UART bytes with `TX`, jump to the start (HALT when the context is full).",
    "prompt": "Prompt feeding: while prompt tokens are left, the next token comes from the prompt buffer instead of the last choice; a prompt token that is not the last one jumps back to the start after the layers (prefill: no final norm, output layer, argmax, UART).",
    "stop": "Class of the token, indentation level, stop rule: HALT after sending the end-of-text token or a line break that brings the level to 0.",
    "filter": "Syntax filter (docs/FILTER.md): the state step for every token that is fed, prompt tokens too; before the argmax the ALLOW row of the abstract state is copied into the table ALLOWED, and the argmax skips token ids whose class is not allowed.",
    "sample": "Sampling, only for a temperature > 0: u[j] <- (u[j] >> 1) + GUMBEL[RND] for every token id before the argmax (SPEC Part D).",
}
STEPS["start"] = ("Start-up piece, run once per answer and not per token. Prompt table: `IN` reads the switches, SW5..0 "
                  "choose a prompt of the table, it is copied into the prompt buffer. Prompts over the UART: position, stop "
                  "level and filter state are reset, the prompt is read with `RX` into the prompt buffer; after the answer the "
                  "end byte 0xFF is sent and the program jumps back here.")
FORMS = {"single": "one prompt, written in the memory image", "table": "a table of prompts in the memory image, chosen by the switches",
         "uart": "prompts typed over the UART (needs the proposed instruction `RX`)"}
LABELS = {**cost.LABELS, "start": "start-up piece (per answer)", "prompt": "prompt feeding and prefill test", "stop": "token class, level, stop rule",
          "filter": "syntax filter", "sample": "sampling"}


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


def meaning(name):
    return MEANING.get(re.sub(r"_\d+$", "", name), "")


def index(tables):
    """weights/INDEX.md. tables: name -> array."""
    rows = [("`%s.hex`" % name, " x ".join(map(str, value.shape)), f"{value.size:,}", "%d .. %d" % (value.min(), value.max()),
             meaning(name)) for name, value in tables.items()]
    return "\n".join([
        "# Weights and tables", "",
        "One file per array: one byte per line, two hex digits, two's complement (`FF` = -1), row-major (the last",
        "index runs fastest). A table of 256 entries is indexed by the unsigned byte of the int8 value: lines 0..127",
        "are the values 0..127, lines 128..255 the values -128..-1. The tables of the syntax filter and GUMBEL hold",
        "bytes 0..255. `mem/data.hex` already contains everything the program reads, laid out for the CPU; these",
        "files are the same numbers one array at a time.", "",
        table(("file", "shape", "bytes", "values", "meaning"), rows), "",
        "Total: %s bytes in %d files." % (f"{sum(v.size for v in tables.values()):,}", len(tables)), ""])


def lut_tables(b, arrays):
    """Rows (table number, address, content) of the LUT region, found by comparing the image with the tables
    `arrays` (the arrays of model.npz and GUMBEL)."""
    base, size = b.mem["tables"]
    known = [("NORM[%d]" % n, row) for n, row in enumerate(arrays["NORM"])] + [("EXP[%d]" % l, row) for l, row in enumerate(arrays["EXP"])]
    known += [(name, arrays[name].astype(np.int8)) for name in ("RECIP", "SILU", "GUMBEL") if name in arrays]
    known.append(("RELU (v for v = 0..127, 0 for 128..255; made by the generator)", np.where(np.arange(256) < 128, np.arange(256), 0).astype(np.int8)))
    rows = []
    for t in range(size // 256):
        content = b.image[base + 256 * t:base + 256 * (t + 1)]
        match = next((g for g, (name, value) in enumerate(known) if (value == content).all()), None)   # equal tables: in order
        name = known.pop(match)[0] if match is not None else "made by the generator for the stop rule, the filter or the argmax (see lsllm/ls/gen.py)"
        rows.append((t, "0x%05X" % (base + 256 * t), name))
    return rows


def prompt_lines(b, ids, form):
    """The section about the prompt bytes of the image."""
    (left, one), (prm, size) = b.mem["prompt.left"], b.mem["prompt.ids"]
    if form == "uart":
        return [
            "## The prompt", "",
            "No prompt is in the image. After reset and after every answer the start-up piece of the program reads a",
            "prompt from the UART (UART.md) into the prompt buffer: `prompt.left` at 0x%05X gets the number of tokens," % left,
            "`prompt.ids` at 0x%05X the high bytes of the token ids and 256 bytes further their low bytes." % prm, ""]
    if form == "table":
        (index, one), (tab, size) = b.mem["prompts.index"], b.mem["prompts.table"]
        return [
            "## The prompts", "",
            "The image holds a table of prompts (PROMPTS.md lists them). Once after reset the start-up piece reads the",
            "switches, takes SW5..0 as the prompt number k and copies that prompt into the prompt buffer (`prompt.left`",
            "at 0x%05X, `prompt.ids` at 0x%05X), from where the program feeds it." % (left, prm), "",
            "- `prompts.index` at 0x%05X (line %d of `mem/data.hex`, two more in `mem/data.coe`): 4 bytes per switch value" % (index, index + 1),
            "  k = 0..63: the address of prompt k as [bits 11..8], [low byte], [bits 19..12], then its number of tokens n.",
            "  A switch value beyond the table has the entry of prompt 0.",
            "- `prompts.table` at 0x%05X, %s bytes: per prompt the n high bytes of its token ids (id >> 8), then the n" % (tab, f"{size:,}"),
            "  low bytes (id & 255), prompt after prompt.", "",
            "Other prompts: run the export again with `--prompts`, or write them with `lsllm.ls.gen.set_prompts(image, b, lists)`",
            "(1 to 64 prompts of 1 to min(T, 255) tokens that fit the %s bytes of this table)." % f"{size:,}", ""]
    rows = [("`prompt.left`", "0x%05X" % left, left + 1, left + 3, "n = number of prompt tokens (here %d)" % len(ids)),
            ("`prompt.ids` + p", "0x%05X + p" % prm, "%d + p" % (prm + 1), "%d + p" % (prm + 3), "high byte of prompt token p: id >> 8"),
            ("`prompt.ids` + 256 + p", "0x%05X + p" % (prm + 256), "%d + p" % (prm + 257), "%d + p" % (prm + 259), "low byte of prompt token p: id & 255")]
    return [
        "## The prompt", "",
        "The program reads its prompt from the data memory (the CPU has no UART receiver). To run another prompt,",
        "change these bytes in the image, for p = 0 .. n - 1 (lines counted from 1; `data.coe` has two header lines):", "",
        table(("bytes", "address", "line of `mem/data.hex`", "line of `mem/data.coe`", "content, two hex digits"), rows), "",
        "A prompt has 1 to min(T, 255) tokens (`lsllm.ls.gen.set_prompt`); the first one is the BOS token (id %d here)," % ids[0],
        "as `lsllm.tok.Tok.encode` writes it. Nothing else in the image depends on the prompt: the filter state and the",
        "indentation level are computed by the program while it feeds the prompt. The tool does all of this:", "",
        "```", r"""python tools/set_prompt.py OUT_DIR/mem 'def area(w, h):\n    return'""", "```", "",
        "(from the root of the repository; OUT_DIR is this directory.) It encodes the text with the tokenizer named in",
        "`mem/map.json`, puts the BOS token in front, writes the bytes with `lsllm.ls.gen.set_prompt` and",
        "writes `data.hex` and `data.coe` again. `vectors/` then still describes the old prompt: for the expected",
        "output of a new prompt run the export again with `--prompt`.", "",
        "The prompt in this image: `%s`" % " ".join(map(str, ids)), ""]


def memory_map(b, arrays, ids, form="single"):
    """MEMORY_MAP.md from Build.mem. arrays: the arrays of model.npz and GUMBEL; ids: the prompt of the test
    vectors (in the image if form is "single"); form: lsllm.export.tables.form_of."""
    rows, groups, end = [], Counter(), 0
    for name, (address, size) in sorted(b.mem.items(), key=lambda item: item[1]):
        if address > end:
            rows.append(("(unused)", "0x%05X" % end, f"{address - end:,}", "alignment, never read"))
        content = next((text for start, text in REGIONS if name.startswith(start)), "see lsllm/ls/gen.py")
        rows.append(("`%s`" % name, "0x%05X" % address, f"{size:,}", content % name[2:] if "%s" in content else content))
        groups[name.split(".")[0] if name[:2] != "kv" else name[:4]] += size
        end = max(end, address + size)
    total, words = len(b.image), len(b.words)
    return "\n".join([
        "# Memory map of the data memory D", "",
        "Taken from the `Build` that `lsllm.ls.gen.build` returned for this model. `mem/data.hex` and `mem/data.coe`",
        "hold D at reset from address 0, one byte per line; addresses are 20 bits (pointers P0, P1, P2). The KV cache",
        "and the scratch vectors are zero at reset and written by the program. `mem/map.json` has the same regions",
        "for programs: `{\"tokenizer\": ..., \"regions\": {name: [address, bytes]}}`.", "",
        "## Totals", "",
        "- Data memory: %s bytes (0x%05X), %s bits. The machine addresses at most 2^19 = 524,288 bytes." % (f"{total:,}", total, f"{8 * total:,}"),
        "- 36 Kb block RAMs: **%d** of the 135 on the board. A 36 Kb block used 8 bits wide (4K x 9, the ninth bit" % -(-total // 4096),
        "  is parity and stays unused) holds 4,096 bytes, so the count is bytes / 4,096 rounded up, and 135 blocks",
        "  hold 552,960 bytes. Counting all 36,864 bits of a block (135 blocks = 4,860 Kb, about 607 KB) the image",
        "  would need %d blocks, but that needs a 9-bit packing that the plain byte-wide memory does not have." % -(-8 * total // 36864),
        "- Program ROM: %s words of 16 bits (`mem/program.hex`); the 12-bit PC allows 4,096. As block RAM in the" % f"{words:,}",
        "  2K x 18 form: %d block(s) of 36 Kb." % -(-words // 2048), "",
        *prompt_lines(b, ids, form),
        "## Bytes by kind", "",
        table(("kind", "bytes"), [(name, f"{size:,}") for name, size in groups.items()] +
              [("unused (alignment)", f"{total - sum(groups.values()):,}"), ("total", f"{total:,}")]), "",
        "## Lookup tables (table number t of `LUT Rd, #t`)", "",
        table(("t", "address", "content"), lut_tables(b, arrays)), "",
        "## Every region in address order", "",
        table(("name", "start", "bytes", "content"), rows), ""])


def start_up(b, form, ids):
    """The paragraph about the start-up piece: its words, and its cycles measured in the simulator for `ids`."""
    words = sum(tag == "start" for name, args, tag in b.code)
    cpu = gen.machine(b)                                # switches 0: prompt 0 of a table
    cpu.feed(gen.frame(ids) if form == "uart" else ())
    cpu.run(stop=gen.top(b))
    if form == "table":
        return [
            "Start-up piece (tag `start` in `mem/program.lst`, %d words, in front of the per-token program): once after" % words,
            "reset it reads the switches with `IN`, takes SW5..0 as the prompt number and copies that prompt from the",
            "table into the prompt buffer; then the program runs as described above and halts. Measured in the",
            "simulator for prompt 0 (%d tokens): %s cycles per answer. This number is the one of this build;" % (len(ids), f"{cpu.cycles:,}"),
            "docs/COST_TABLE.md, section Choosing the prompt on the board, gives 92 + 24 n cycles for a prompt of n",
            "tokens in the shape of that table (%d for this prompt). The prompts are in PROMPTS.md." % (92 + 24 * len(ids)), ""]
    return [
        "Start-up piece (tag `start` in `mem/program.lst`, %d words: the piece in front of the per-token program and" % words,
        "the three instructions at the end that send the end byte and jump back): after reset and after every answer",
        "it puts the position, the stop level and the filter state back to their start and reads a prompt from the",
        "UART with `RX` (two bytes per token id, then 0xFF; UART.md) into the prompt buffer. After the answer the",
        "program sends 0xFF and waits for the next prompt: it never halts. `RX` is a proposed instruction (docs/SPEC.md",
        "A.2); without it this form cannot run. Measured in the simulator for the prompt of the test vectors",
        "(%d tokens): %s cycles per answer from the first byte to the first token step, waiting for the UART not" % (len(ids), f"{cpu.cycles:,}"),
        "counted; 12 more for the end byte. docs/COST_TABLE.md, section Choosing the prompt on the board, has the formula.", ""]


def program(b, spec, syntax_filter, temperature, form="single", ids=()):
    """PROGRAM.md: the outline, words of ROM per component, and cycles from lsllm.ls.cost. form: how the program
    gets its prompt (lsllm.export.tables.form_of); ids: the prompt of the test vectors."""
    cfg, options = ModelConfig(**spec["config"]), spec["options"]
    setting = "stop rule on, syntax filter %s, %s" % ("on" if syntax_filter else "off",
                                                      "sampling at temperature %g" % temperature if temperature > 0 else "greedy")
    words = Counter(tag for name, args, tag in b.code)
    names = [name for name in dict.fromkeys(gen.COMPONENTS + gen.EXTRA + tuple(words)) if words[name]]
    out = [
        "# The LS program", "",
        "`mem/program.hex` (one 16-bit word per line), `mem/program.lst` (address, word, instruction, component).",
        "Shape: D = %d, L = %d layers, H = %d query heads, G = %d key/value heads, head size d = %d, F = %d, V = %d," %
        (cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.hidden_dim, cfg.vocab_size),
        "T = %d; norm = %s, ffn = %s. Machine: level L1 of docs/SPEC.md Part A." % (cfg.max_seq_len, options["norm"], options["ffn"]), "",
        "This is the complete program of `lsllm.ls.gen.build`: %s." % setting,
        "Its prompt: %s. Once it has a prompt it runs alone:" % FORMS[form], "",
        "1. Prefill: every prompt token but the last is fed from the prompt buffer in the data memory; it runs the",
        "   layers and fills the KV cache, and skips the final norm, the output layer, the argmax and the UART.",
        "2. The last prompt token and every token after it run all steps of the table below; the chosen token is",
        "   sent over the UART and becomes the input of the next position.",
        "3. The program halts after sending a token that ends the text (the end-of-text token, or a line break that",
        "   brings the indentation level back to 0: the end mark, which the host drops), or when the context is full.", "",
        *(["Another prompt: change the prompt bytes of the image (MEMORY_MAP.md, section The prompt) or run",
           "`python tools/set_prompt.py OUT_DIR/mem 'text'`. The program itself does not depend on the prompt.", ""]
          if form == "single" else start_up(b, form, ids)),
        "## What the generator says about its program (module text of lsllm/ls/gen.py)", "", "```", gen.__doc__.strip(), "```", "",
        "## One token, step by step", "",
        "The steps from the norms to the gating run once per layer, the others once per token. The layers are unrolled",
        "in the ROM, so the words of a layer step are those of all %d layers. Words: instructions of the step in the ROM." % cfg.n_layers, ""]
    try:
        contexts = sorted({min(c, cfg.max_seq_len) for c in (16, 64, 256)})
        stop = dict(stop=True)
        kinds = [("generated token: greedy, filter off", stop, (False, False)),
                 ("+ filter", dict(syntax_filter=True), (True, False)),
                 ("+ sampling", dict(stop=True, temperature=1.0), (False, True)),
                 ("+ filter + sampling", dict(syntax_filter=True, temperature=1.0), (True, True)),
                 ("prefill token (a prompt token that is not the last)", dict(prefill=True, **(dict(syntax_filter=True) if syntax_filter else stop)), None)]
        mine = next(kw for label, kw, key in kinds if key == (syntax_filter, temperature > 0))
        costs = [cost.token_cycles(cfg, options, c - 1, **mine)[:2] for c in contexts]
        rows = [(g + 1, LABELS.get(name, name), STEPS.get(name, "see lsllm/ls/gen.py"), f"{words[name]:,}",
                 *(f"{parts.get(name, 0):,}" for total, parts in costs)) for g, name in enumerate(names)]
        rows.append(("", "**total**", "", f"{len(b.code):,}", *(f"{total:,}" for total, parts in costs)))
        rows.append(("", "**tokens/s at 100 MHz**", "", "", *("%.1f" % (1e8 / total) for total, parts in costs)))
        totals = [[label + (" **(this export)**" if kw is mine else "")] +
                  ["%s (%.1f)" % (f"{total:,}", 1e8 / total) for total in (cost.token_cycles(cfg, options, c - 1, **kw)[0] for c in contexts)]
                  for label, kw, key in kinds]
        middle = contexts[len(contexts) // 2]
        parts = [cost.token_cycles(cfg, options, middle - 1, **kw)[1] for label, kw, key in kinds]
        extra = [name for name in dict.fromkeys(gen.COMPONENTS + gen.EXTRA) if len({part.get(name, 0) for part in parts}) > 1]
        out += [
            "Cycles: one generated token of this export (%s) at the given context length (position = context - 1)," % setting,
            "from `lsllm.ls.cost.token_cycles` for this shape and these options, with the filter tables of the",
            "tokenizer `data/tok/f512a`; every instruction 4 cycles, a stream of n multiply-adds n + 4.", "",
            table(["", "step", "what the program does", "ROM words"] + ["cycles, context %d" % c for c in contexts], rows), "",
            "## Cycles per token by kind of token: cycles (tokens/s at 100 MHz)", "",
            "Sampling costs the same at any temperature > 0. The prefill token has the filter setting of this export.", "",
            table(["token"] + ["context %d" % c for c in contexts], totals), "",
            "The steps whose cycles differ between these kinds, at context %d:" % middle, "",
            table(["step"] + [label.split(":")[0].split(" (")[0] for label, kw, key in kinds],
                  [[LABELS.get(name, name)] + [f"{part.get(name, 0):,}" for part in parts] for name in extra]), "",
            "A whole answer costs the prefill cycles of every prompt token but the last plus the generated-token",
            "cycles of every token sent; the round-trip check in README.md gives the cycles the simulator counted for",
            "the prompt of the test vectors.", "",
            "The cycle counts are minima, counted on zero weights. How they are counted and what depends on the data",
            "(module text of lsllm/ls/cost.py; `token_cycles` adds: with the filter the first allowed token counts",
            "as one new maximum of the argmax and is included):", "", "```", cost.__doc__.strip(), "```", ""]
    except ValueError as err:                           # the cost model has the filter tables of data/tok/f512a only
        out += [table(("", "step", "what the program does", "ROM words"),
                      [(g + 1, LABELS.get(name, name), STEPS.get(name, "see lsllm/ls/gen.py"), f"{words[name]:,}") for g, name in enumerate(names)]
                      + [("", "**total**", "", f"{len(b.code):,}")]), "",
                "No cycle counts: `lsllm.ls.cost.token_cycles` cannot price this shape (%s)." % err, ""]
    return "\n".join(out)


def uart(tok_dir, level, form="single"):
    """UART.md. level: the indentation level after the prompt of the test vectors; form: lsllm.export.tables.form_of."""
    ending = ("Then the CPU halts; a new run needs a reset." if form != "uart" else
              "Then the CPU sends the end byte `0xFF` and waits for the next prompt; it never halts.")
    lines = [
        "# UART: between the board and the laptop", "",
        "- 115200 baud, 8 data bits, no parity, 1 stop bit (8N1), idle high, least significant bit first.",
        "- Frame (docs/SPEC.md Part D): two bytes per token id, first `0x80 | (id >> 7)`, then `id & 0x7F`.",
        "  A byte with the top bit set always starts a token, so the receiver finds the frame again after a lost byte.",
        "  Example: token id 300 = 0x12C is sent as `0x82`, `0x2C`.",
        "- The stream of one answer: the generated tokens only (the prompt is not sent back), one frame each as soon as",
        "  the token is chosen. The last frame is the token that ended the text (the end-of-text token, or a line break",
        "  that brings the indentation level to 0): an end mark, which the host drops. If the context fills up first",
        "  there is no end mark. " + ending,
        "- The CPU sends a byte with `TX Rs` (SPEC A.2): the byte goes into the transmit FIFO, the instruction waits only",
        "  if the FIFO is full. 2 bytes per token at 115200 baud allow 5,760 tokens per second, more than the CPU computes.",
        "- At 100 MHz one bit lasts 100,000,000 / 115,200 = 868.06 clock cycles: a divider of 868 is 0.006% off.", ""]
    if form == "uart":
        lines += [
            "## From the laptop to the board: the prompt", "",
            "This program takes its prompts from the UART. It needs a UART receiver and the proposed instruction `RX Rd`",
            "(docs/SPEC.md A.2: Rd <- the next byte of the receive FIFO, the CPU waits while the FIFO is empty).", "",
            "1. After reset, and after every answer, the program waits for a prompt.",
            "2. The host sends the token ids of the prompt in the same framing, two bytes per id (`0x80 | (id >> 7)`, then",
            "   `id & 0x7F`), the BOS token first, and then the byte `0xFF`. No token id starts with `0xFF` (ids are below 4,096).",
            "3. Tokens beyond min(T, 255) are read and dropped; a prompt without a token is ignored (`lsllm.ls.gen.build`).",
            "4. The board answers: one frame per generated token, the end mark included, then the byte `0xFF`.",
            "5. Back to 1. The KV cache position, the stop level and the filter state start again for every prompt; the",
            "   random generator of the sampling is not reset between answers.", "",
            "`lsllm.ls.gen.frame(ids)` gives the bytes of a prompt, `lsllm.ls.gen.answers(cpu)` splits what the simulator sent.",
            "On the laptop `tools/board_ask.py MODEL_DIR \"make me a bubble sort program\" --port ...` turns a request into",
            "a prompt, sends it and prints the answer (`--simulate` runs the same program in the simulator).", ""]
    return "\n".join(lines + [
        "## Pins (Nexys A7-100T, FT2232HQ USB-UART bridge; reference manual section 6)", "",
        "| FPGA pin | name in the Digilent master XDC | direction |", "|---|---|---|",
        "| D4 | `UART_RXD_OUT` | FPGA output: **the transmit pin of the CPU** (data to the PC) |",
        "| C4 | `UART_TXD_IN` | FPGA input (data from the PC): %s |" %
        ("**the receive pin of the CPU**, the prompts arrive here" if form == "uart" else "not used, this program has no UART receiver"), "",
        "The Digilent names are from the point of view of the PC, which is why the FPGA's output is called RXD. Source in",
        "this repository: `research/09_fpga_costs.md` (\"C4 = UART_TXD_IN (FPGA input, data from the PC), D4 = UART_RXD_OUT",
        "(FPGA output, data to the PC)\", checked there against the reference manual section 6, the master XDC and",
        "Digilent's GPIO demo, where `UART_TXD : out` is constrained to D4). Take the constraint lines (pin and",
        "IOSTANDARD) from the master XDC of your board and compare: https://github.com/Digilent/digilent-xdc", "",
        "## On the laptop", "",
        "`tools/uart_receive.py` reads the port, joins the two bytes of each token and prints the Python text as it arrives.",
        "`--level` is the indentation level after the prompt (%d for the prompt of the test vectors):" % level, "",
        "```",
        "python tools/uart_receive.py --port /dev/tty.usbserial-XXXX --tok %s/tok.model --level %d" % (tok_dir, level),
        "python tools/uart_receive.py --replay OUT_DIR/vectors/uart.bin --tok %s/tok.model --level %d" % (tok_dir, level),
        "```", "",
        "(From the root of the repository; OUT_DIR is this directory.)",
        "`--port` needs `pyserial`; `--replay` reads a file of raw frame bytes: `vectors/uart.bin` holds the bytes this",
        "export must send for the prompt of the test vectors, so the laptop side can be tried without a board. The script",
        "prints every token it receives, the end mark too (a line break, or a blank line for the end-of-text token).", ""])


def prompts_md(texts, lists):
    """PROMPTS.md: the prompt table. texts: the prompts; lists: their token ids."""
    rows = [("`%s`" % format(k, "06b"), k, "`%s`" % text.replace("\n", "\\n"), len(ids), " ".join(map(str, ids)))
            for k, (text, ids) in enumerate(zip(texts, lists))]
    return "\n".join([
        "# The prompts of the switch table", "",
        "Set the switches SW5..SW0 to the number of a prompt and press reset: the program reads the switches once",
        "(`IN`), copies that prompt into its prompt buffer, answers over the UART and halts. A number beyond the table",
        "(%d and up) gives prompt 0. The switches above SW5 do not choose a prompt; all 16 switches also seed the random" % len(texts),
        "generator, which matters only for a program that samples. In a prompt `\\n` stands for a line break; the token",
        "count includes the BOS token in front.", "",
        table(("SW5..SW0", "number", "prompt", "tokens", "token ids"), rows), "",
        "%d prompts, %d tokens, %s bytes in the table (MEMORY_MAP.md, section The prompts)."
        % (len(texts), sum(map(len, lists)), f"{2 * sum(map(len, lists)):,}"), ""])


def vectors_readme(prompt, summary, named, tok, syntax_filter, temperature, form="single"):
    """vectors/README.md. summary: the arrays of tokens.txt; named: file name without .txt -> its values."""
    where = {"single": "already in `mem/data.hex`", "table": "prompt 0 of the table in `mem/data.hex` (all switches off)",
             "uart": "to be sent to the board over the UART after reset (UART.md)"}[form]
    alone = {"single": "The program runs alone until HALT.", "table": "The start-up piece copies prompt 0 into the prompt buffer; "
             "then the program runs alone until HALT.", "uart": "The program waits for the prompt on the UART, then runs alone; "
             "after the answer it sends the byte 0xFF (not in the lists below) and waits again."}[form]
    names = lsllm.intref.model.__doc__.split("Names used by the trace")[1]
    ids, sent = summary["prompt_ids"], summary["sent"]
    order = [(name, " x ".join(map(str, np.shape(value))) or "1") for name, value in list(named.values())[-1].items()]
    show = lambda tokens: "`%s`" % " ".join(map(str, tokens))
    setting = "syntax filter %s, %s" % ("on" if syntax_filter else "off",
                                        "sampling at temperature %g with the switches at 0" % temperature if temperature > 0 else "greedy")
    last = "the last token is the end mark" if summary["end_mark"] else "no end mark: the context is full"
    lines = [
        "# Test vectors for bring-up", "",
        "Made by the integer reference `lsllm.intref.IntModel` (bit-exact SPEC Part C) in the loop of",
        "`lsllm.intref.completer.IntCompleter`, with `lsllm.filter_table.TableFilter` for the stop rule and the filter.",
        "Setting of this export: stop rule on, %s. The CPU must reproduce every number here." % setting, "",
        "Prompt (%d tokens with the BOS token in front), %s:" % (len(ids), where), "", "```", prompt, "```", "",
        "Token ids of the prompt: %s" % show(ids), "",
        "## Order of events", "",
        "1. Reset: KV cache empty, position 0. %s" % alone,
        "2. Positions 0 .. %d: the prompt tokens but the last (prefill). Layers and KV cache only: no final norm," % (len(ids) - 2),
        "   no logits, nothing sent. `position_00.txt` is the first of them: empty cache, one key per head, so every",
        "   attention weight is 127. It is the simplest first test%s." %
        (" (the prompt has one token, so this position also computes the logits)" if len(ids) == 1 else ""),
        "3. Position %d: the last prompt token. Its logits choose generated token 1 (`token_01.txt`), which is sent" % (len(ids) - 1),
        "   and fed at the next position, and so on: `token_NN.txt` is the step whose logits choose generated token",
        "   NN, at position %d + NN, with generated token NN - 1 as its input." % (len(ids) - 2),
        "4. After the token that ends the text has been sent, the CPU %s." % ("waits for the next prompt" if form == "uart" else "halts"), "",
        "## What the program must send", "",
        "- Tokens: %s (%d tokens, %s)" % (show(sent), len(sent), last),
        "- UART bytes (`uart.bin`): `%s`" % " ".join("%02X" % x for x in summary["uart"]),
        "- Text (the prompt, then the tokens sent, the end mark included):", "", "```", prompt + tok.decode(sent, tok.level_after(ids)), "```", "",
        "The same prompt in the other settings of the program (each needs its own export for the memory files):", "",
        "- greedy, filter off (`sent_greedy`): %s" % show(summary["sent_greedy"]),
        "- greedy, filter on (`sent_filter`): %s" % show(summary["sent_filter"])]
    if "sent_sampled" in summary:
        lines.append("- sampled at temperature %g, switches at 0 (LFSR reset value 0xACE1ACE1), filter %s (`sent_sampled`): %s"
                     % (temperature, "on" if syntax_filter else "off", show(summary["sent_sampled"])))
    return "\n".join(lines + [
        "", "## Files", "",
        "- `tokens.txt`: `prompt_ids`, `sent`, `uart`, `end_mark` (1: the last token of `sent` is the end mark) and the",
        "  sequences of the other settings, in the format below.",
        "- `position_00.txt`, `token_01.txt` ...: `position`, `token_in`, every traced value in the order of",
        "  computation; the token files end with %s`token_out` and `uart` (its two bytes)." %
        ("`allowed` (one 0 or 1 per token id: what the syntax filter allows), " if syntax_filter else ""),
        "  A line that starts with a letter is `name size size ...` (`name 1` for a single number); the lines after",
        "  it are the values in decimal, row-major, 16 per line.",
        "- `vectors.npz`: the same arrays for numpy, keys `position_00.<name>`, `token_NN.<name>` and the names of `tokens.txt`.",
        "- `uart.bin`: the raw frame bytes, for `tools/uart_receive.py --replay`.", "",
        "## Names (module text of lsllm/intref/model.py)", "", "```", "Names used by the trace" + names.rstrip(), "```", "",
        "Every value is int8 except the accumulators (`.acc`, signed 24 bits, with the preload already added),",
        "`dsum` (up to 127 T) and the counts `nrm` (7..15). Head axes: a query head is written (key/value head g,",
        "head s inside the group), query head h = g * share + s. `code`, `e` have one entry per key 0 .. position.",
        "`u` holds the logits before sampling and before the filter; `token_out` is the choice in the setting of",
        "this export.", "",
        "## Values of one token step, in order", "",
        table(("name", "shape"), order), ""])


def frame(tokens):
    """The UART bytes of a list of token ids (SPEC Part D)."""
    return [byte for t in tokens for byte in (0x80 | t >> 7, t & 0x7F)]
