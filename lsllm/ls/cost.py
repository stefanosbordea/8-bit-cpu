"""Cycle cost of one token on the LS CPU, counted on the generated program itself (lsllm.ls.gen).

The program of a shape is generated with zero weights and run for one token in the simulator at the wanted
position. The simulator reports how often every instruction was executed and how many multiply-adds every
stream did; `price` turns these counts into cycles: for L1 (4 cycles per instruction, 1 more per stream element)
or for a hardware change of level L2 (the what-if switches, never used in the main numbers).

The cycles depend on the shape, the options and the position only, with two exceptions, so the numbers are minima:
the argmax spends 16 more cycles for every new maximum it meets (at most 127 times, logits are 0..127; 24 cycles
in a last block of fewer than 256 logits), and sending the token takes 12 more cycles when bit 7 of the low byte
of its id is set. Zero weights give neither (token 0, no new maximum).

    .venv/bin/python -m lsllm.ls.cost        writes docs/COST_TABLE.md
"""
import functools
import os
from collections import Counter

import numpy as np

from lsllm.config import ModelConfig
from lsllm.ls import gen

FUSED = ("SETP0", "LDA", "MAC", "RQ", "ST2", "DJNZ")     # what a fused row instruction does (and a LUT after its RQ)
L2 = {  # name: (switches of `price`, the modelling assumption in one line)
    "fused row": (dict(fused=True), "one instruction rewinds P0, preloads ACC from two registers, streams, "
                  "requantises, looks up (optional), stores through P2 and loops: a row of n multiply-adds costs "
                  "n + 3 cycles. Used for the matrix rows, the residual rows and the score loop; anything else in "
                  "their loop bodies keeps its 4 cycles (the read of x[j] in a residual row; the read of the preload "
                  "byte and the three instructions of the running maximum in the score loop). Not fused: the value "
                  "rows (they requantise twice) and the loops with one or two products per value (RoPE, gating "
                  "product, rms scaling)"),
    "2 MAC lanes": (dict(lanes=2), "a stream of n multiply-adds costs ceil(n / 2) + 4 cycles"),
    "4 MAC lanes": (dict(lanes=4), "a stream of n multiply-adds costs ceil(n / 4) + 4 cycles"),
    "hardware argmax": (dict(hw_argmax=True), "RQ keeps the running maximum of its results and the index of it; "
                        "3 instructions (clear, read the index low and high) replace the argmax pass, and in the "
                        "score loop the three instructions of the running maximum disappear (one instruction per "
                        "head reads the maximum; the LDI that clears R2 now clears the hardware maximum)"),
    "table in a stream": (dict(stream_lut=True), "a table map of n bytes is a stream of n + 4 cycles (also the pass "
                          "that turns the score codes into weights: the stream subtracts a register from the "
                          "index), and RQ can pass its result through a table (no LUT instruction after an RQ)"),
    "signed-compare branch": (dict(signed_branch=True), "a branch on the sign of SUB replaces the LUT of the argmax "
                              "loop: 16 instead of 20 cycles per logit"),
    "3-cycle instructions": (dict(ordinary=3), "every instruction that is not a stream costs 3 cycles; a stream "
                             "still costs n + 4"),
    "all together": (dict(fused=True, lanes=4, hw_argmax=True, stream_lut=True, ordinary=3),
                     "fused row, 4 lanes, hardware argmax, table in a stream and 3-cycle instructions at once; "
                     "a fused row of n is assumed to cost ceil(n / 4) + 3, which no single row above states"),
}


def dummy_model(cfg, options):
    """(spec, arrays) of SPEC Part E for the shape cfg, everything zero. options may hold "sh_e", the shift of the
    embedding lookup (SPEC C.1); without it the lookup is a plain copy."""
    D, L, F, V, T, d, Gd = cfg.dim, cfg.n_layers, cfg.hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.head_dim, cfg.kv_dim
    sizes = {"Wq": (D, D), "Wk": (Gd, D), "Wv": (Gd, D), "Wo": (D, D), "W1": (F, D), "W3": (F, D), "W2": (D, F)}
    arrays = {"%s_%d" % (name, l): size for l in range(L) for name, size in sizes.items()}
    arrays.update(E=(V, D), NORM=(2 * L + 1, 256), SILU=256, EXP=(L, 256), RECIP=256, ROPE_C=(T, d // 2), ROPE_S=(T, d // 2))
    shifts = {name: [0] * L for name in ("sh_q", "sh_k", "sh_v", "sh_s", "sh_o", "sh_1", "sh_3", "sh_g")}
    shifts.update(sh_l=0, sa=[0] * (2 * L + 1), sb=[0] * (2 * L + 1))
    if options.get("sh_e"):
        shifts["sh_e"] = options["sh_e"]
    spec = {"config": cfg.to_dict(), "options": {name: value for name, value in options.items() if name != "sh_e"}, "shifts": shifts,
            "preloads": {"c_s": [[0] * cfg.n_heads] * L, "c_l": 0}}
    arrays = {name: np.zeros(size, dtype=np.int8) for name, size in arrays.items()}
    arrays["EXP"][:, 0] = 127                           # SPEC C.6: the largest code has weight 127
    return spec, arrays


@functools.lru_cache(maxsize=None)
def _program(shape, options):
    return gen.build(*dummy_model(ModelConfig(**dict(shape)), dict(options)))


@functools.lru_cache(maxsize=None)
def _counts(shape, options, pos):
    b = _program(shape, options)
    cpu = gen.machine(b)
    gen.set_position(cpu, b, pos)
    gen.run_token(cpu, b, 0)
    return cpu.count, cpu.elems


def profile(cfg, options, pos):
    """(Build, executions of every program address, multiply-adds of every address) for one token at `pos`."""
    key = tuple(cfg.to_dict().items()), tuple(sorted(options.items()))
    return (_program(*key), *_counts(*key, pos))


def price(b, count, elems, fused=False, lanes=1, hw_argmax=False, stream_lut=False, signed_branch=False, ordinary=4):
    """Cycles per component for the execution counts of one token. Without switches: the L1 machine."""
    cycles, loop = Counter(), {}                    # loop: address -> kind of the marked loop it belongs to
    for first, last, kind in b.loops:
        loop.update(dict.fromkeys(range(first, last + 1), kind))
        if stream_lut and kind == "map":            # the + 4 of the table stream, once per loop that was entered
            cycles[b.code[first][2]] += 4 * count[last + 1]
        if hw_argmax and kind == "row" and b.code[last - 1][0] == "ADD":      # a score loop: read the maximum once
            cycles[b.code[first][2]] += ordinary * count[last + 1]
    for address, (name, args, tag) in enumerate(b.code):
        n, in_row, before = count[address], loop.get(address) == "row", b.code[address - 1][0]
        maximum = in_row and (name in ("SUB", "ADD") or name == "LUT" and before == "SUB")   # of the score loop
        if hw_argmax and (tag == "argmax" or maximum):
            cost = 0
        elif elems[address]:                        # a stream: every execution has the same length
            length, rest = divmod(elems[address], n)
            assert rest == 0
            cost = n * (-(-length // lanes) + (0 if fused and in_row else 4))
        elif fused and in_row and (name in FUSED or name == "LUT" and before == "RQ"):
            cost = n * {"DJNZ": 3, "MAC": 1}.get(name, 0)
        elif stream_lut and loop.get(address) == "map":
            cost = n * (name == "LD")               # one cycle per byte
        elif name == "LUT" and (stream_lut and before == "RQ" or signed_branch and tag == "argmax"):
            cost = 0
        else:
            cost = n * ordinary
        cycles[tag] += cost
    if hw_argmax:
        cycles["argmax"] = 3 * ordinary
    return cycles


def cycles_per_token(cfg, options, pos, **switches):
    """(total cycles, {component: cycles}) of one token at position `pos` (context length pos + 1).

    cfg: ModelConfig (the shape); options: {"norm": dyt | rms, "ffn": swiglu | relu};
    switches: a what-if of level L2 (see L2 and `price`), none for the L1 machine."""
    parts = price(*profile(cfg, options, pos), **switches)
    return sum(parts.values()), {name: parts[name] for name in gen.COMPONENTS}


@functools.lru_cache(maxsize=None)
def tokenizer_filter():
    """The filter tables of the tokenizer of the project (512 tokens)."""
    from lsllm.filter_table import TableFilter
    from lsllm.tok import Tok
    return TableFilter(Tok(os.path.join(os.path.dirname(__file__), "..", "..", "data", "tok", "f512a", "tok.model")))


@functools.lru_cache(maxsize=None)
def _complete(shape, options, stop, syntax_filter, temperature):
    spec, arrays = dummy_model(ModelConfig(**dict(shape)), dict(options))
    spec["frac"] = {"u": 4}
    tf = tokenizer_filter() if stop or syntax_filter else None
    return gen.build(spec, arrays, prompt=True, tf=tf, syntax_filter=syntax_filter, temperature=temperature)


def token_cycles(cfg, options, pos, prefill=False, stop=False, syntax_filter=False, temperature=0.0):
    """(total cycles, {component: cycles}, Build) of one token of the complete program (gen.build with prompt=True)
    at position `pos`, on zero weights. prefill: a prompt token that is not the last one. stop: with the stop rule
    (the tables of the tokenizer; implied by syntax_filter). The current token is a plain-text token in code at
    the start of a line with no open bracket. As in `cycles_per_token` the two terms that depend on the data are
    left out: 16 cycles per new maximum of the argmax (24 in a short last block; with the filter the first allowed
    token always counts as one and is included) and 12 cycles when bit 7 of the low byte of the token is set."""
    key = tuple(cfg.to_dict().items()), tuple(sorted(options.items()))
    b = _complete(*key, stop or syntax_filter, syntax_filter, temperature)
    cpu = gen.machine(b)
    gen.set_position(cpu, b, pos)
    token = tokenizer_filter().cls.index(24) if "filter.class" in b.mem else 0
    if prefill:
        gen.set_prompt(cpu.d, b, [token] * 2)
        cpu.d[b.mem["prompt.ids"][0] + 256 + pos] = token - 256 * (token > 127)
    else:
        gen.set_token(cpu, b, token)
    cpu.run(stop=0)
    parts = price(b, cpu.count, cpu.elems)
    for address, (name, args, tag) in enumerate(b.code):
        if tag == "argmax" and name == "MOV":                    # a new maximum
            short = b.code[address + 1][0] == "LDI" and b.code[address + 2][0] == "ADD"
            parts["argmax"] -= (16 + 8 * short) * cpu.count[address]
        if tag == "book" and name == "XOR":                      # bit 7 of the low byte of the token
            parts["book"] -= 12 * cpu.count[address]
    if syntax_filter and not prefill:
        parts["argmax"] += 16                                    # the first allowed token
    return sum(parts.values()), dict(parts), b


def multiply_adds(cfg, options, pos):
    """(multiply-adds done by streams, stream instructions executed, single multiply-adds: classic MAC or MUL + ADDP)."""
    b, count, elems = profile(cfg, options, pos)
    streams = sum(n for n, e in zip(count, elems) if e)
    singles = sum(n for n, e, (name, args, tag) in zip(count, elems, b.code) if name == "MAC" and not e or name == "MUL")
    return sum(elems), streams, singles


def classic(cfg, options, pos):
    """Cycles of the same token when no stream instruction exists, two ways:
    (a) every stream of n is n classic MAC instructions in a row: 4 n cycles instead of n + 4;
    (b) every stream of n is a loop `MAC ; DJNZ`: 8 n cycles (the loop counter is assumed to cost nothing more)."""
    total = cycles_per_token(cfg, options, pos)[0]
    products, streams, singles = multiply_adds(cfg, options, pos)
    return total + 3 * products - 4 * streams, total + 7 * products - 4 * streams


def classic_words(cfg, options):
    """Program words of the classic (a) program: a stream instruction of n becomes n words (MACL: T words)."""
    code = profile(cfg, options, 0)[0].code
    return sum(max(args[0], 1) if name == "MAC" else (cfg.window or cfg.max_seq_len) if name == "MACL" else 1
               for name, args, tag in code)


def memory(b):
    """Bytes of data memory by region, and the bytes that hold nothing (alignment, unused slots)."""
    names = {"tables": "lookup tables (256 bytes each)", "tok": "token table (address of every row of E)",
             "scratch": "scratch (variables, vectors, logits)", "ones": "row of ones (sum of the attention weights)",
             "pos": "position records (links, counts, RoPE)", "heads": "head records (addresses, score offsets)",
             "prompt": "prompt buffer", "prompts": "prompt table", "filter": "syntax filter: tables and state",
             "kv.K": "KV cache: keys", "kv.V": "KV cache: values", "w.E": "embedding E", "w": "weight matrices"}
    regions = Counter()
    for name, (address, size) in b.mem.items():
        regions[names[name[:4] if name[:4] in names else name.split(".")[0]]] += size
    regions["unused (alignment)"] = len(b.image) - sum(regions.values())
    regions["total"] = len(b.image)
    return regions


# ---- docs/COST_TABLE.md ----

SHAPES = [(8, 4), (2, 2), (4, 2)]                   # (query heads, key/value heads) of the stories260K shape
COMBOS = [(norm, ffn, pos) for norm in ("dyt", "rms") for ffn in ("swiglu", "relu") for pos in (15, 63, 255)]
LABELS = {"embedding": "embedding", "norm": "norms", "qkv": "q/k/v projections", "rope": "RoPE", "cache": "cache writes",
          "scores": "scores, maximum and exp", "mix": "value mixing and division", "attn_out": "attention output projection",
          "ffn": "feed-forward matrices", "gating": "gating (SiLU, product)", "output": "output layer",
          "argmax": "argmax", "book": "bookkeeping and UART"}


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


def report():
    """The text of docs/COST_TABLE.md."""
    out = ["# Cost table of the LS program", "",
           "Generated by `.venv/bin/python -m lsllm.ls.cost` (do not edit by hand). Every number is counted on the",
           "program that `lsllm.ls.gen` writes for the shape, run for one token in the simulator `lsllm.ls.iss.L1`",
           "on zero weights. Machine: level L1 of `docs/SPEC.md` v0.4 Part A (every instruction 4 cycles, a stream of n",
           "multiply-adds n + 4), clock 100 MHz. Attention as in SPEC v0.4 C.6: per key d + 40 cycles for the score",
           "code and the running maximum and 20 for the weight; per head one stream for the sum of the weights, `NRM`,",
           "`RQV` and the reciprocal table; per value row p + 57 cycles. Shape: D = 64, L = 5, F = 172, V = 512,",
           "T = 256 (stories260K) with the head counts given; context = position + 1.", "",
           "- The tables hold the minimum. Two terms depend on the data: the argmax costs 16 more cycles per new maximum",
           "  met while scanning the logits, at most 127 times (2,032 cycles), and sending the token costs 12 more cycles",
           "  when bit 7 of the low byte of its id is set. So a real token costs at most 2,044 cycles more than the table",
           "  says (0.5% at context 16, 0.2% at context 256). Nothing else depends on the weights or on the tokens.",
           "- tokens/s = 100,000,000 / cycles. The UART is not a limit: 2 bytes per token at 115,200 baud allow 5,760 tokens/s.",
           "- multiply-add share: multiply-adds per cycle, i.e. the cycles in which the datapath multiplies and adds;",
           "  the rest is control (fetch, pointer set-up, preload, requantise, store, loop).",
           "- classic (a): every stream of n written as n classic `MAC` instructions, 4 cycles per multiply-add",
           "  (the definition of the project brief). Such a program does not fit the ROM: see the words column.",
           "- classic (b): every stream of n as a loop `MAC ; DJNZ`, 8 cycles per multiply-add; fits the ROM.",
           "  The loop counter is assumed to cost nothing extra, which favours (b).", ""]
    for H, G in SHAPES:
        cfg = ModelConfig(n_heads=H, n_kv_heads=G)
        out += ["## H = %d query heads, G = %d key/value heads (head size %d)" % (H, G, cfg.head_dim), "",
                "### Cycles per token at L1", ""]
        rows = []
        for norm, ffn, pos in COMBOS:
            options = {"norm": norm, "ffn": ffn}
            total = cycles_per_token(cfg, options, pos)[0]
            products, streams, singles = multiply_adds(cfg, options, pos)
            a, b = classic(cfg, options, pos)
            words = len(profile(cfg, options, pos)[0].words)
            rows.append((norm, ffn, pos + 1, f"{total:,}", "%.1f" % (1e8 / total), f"{products + singles:,}",
                         "%.1f%%" % (100 * (products + singles) / total), words,
                         f"{a:,}", "%.2f" % (a / total), f"{classic_words(cfg, options):,}",
                         f"{b:,}", "%.2f" % (b / total)))
        out += [table(("norm", "ffn", "context", "cycles", "tokens/s", "multiply-adds", "multiply-add share",
                       "program words", "classic (a) cycles", "speed-up vs (a)", "words of (a)",
                       "classic (b) cycles", "speed-up vs (b)"), rows), "", "### Cycles by component", ""]
        parts = [cycles_per_token(cfg, {"norm": norm, "ffn": ffn}, pos)[1] for norm, ffn, pos in COMBOS]
        out += [table(["component"] + ["%s %s %d" % (norm, ffn, pos + 1) for norm, ffn, pos in COMBOS],
                      [[LABELS[name]] + [f"{part[name]:,}" for part in parts] for name in gen.COMPONENTS]), "",
                "### Data memory in bytes (the same for dyt and rms)", ""]
        regions = [memory(profile(cfg, {"norm": "dyt", "ffn": ffn}, 0)[0]) for ffn in ("swiglu", "relu")]
        out += [table(("region", "swiglu", "relu"), [(name, f"{regions[0][name]:,}", f"{regions[1][name]:,}") for name in regions[0]]),
                "", "### What-if hardware changes (L2): cycles per token (tokens/s)", ""]
        rows = [["L1 (as built)"]] + [[name] for name in L2]
        for norm, ffn, pos in COMBOS:
            options = {"norm": norm, "ffn": ffn}
            for row, switches in zip(rows, [{}] + [switches for switches, text in L2.values()]):
                total = cycles_per_token(cfg, options, pos, **switches)[0]
                row.append("%s (%.0f)" % (f"{total:,}", 1e8 / total))
        out += [table(["change"] + ["%s %s %d" % (norm, ffn, pos + 1) for norm, ffn, pos in COMBOS], rows), ""]
    out += complete_report() + window_report() + embedding_report() + ask_report()
    out += ["## Modelling assumptions of the what-if rows", ""] + ["- %s: %s." % (name, text) for name, (switches, text) in L2.items()]
    out += ["", "## Program size of other head counts (rms, swiglu: the largest program of a shape)", "",
            "Layers are unrolled. The heads of a layer share one piece of program that runs once per head record",
            "(13 bytes copied and 14 instructions: 212 cycles per head), so more query heads do not make the program longer;",
            "key/value heads are unrolled (RoPE of the keys, cache writes). Limits of the generator: ROM 4,096 words,",
            "T <= 256, V <= 4,096, D and F <= 510, at most 20 layers (64 lookup tables: 2 L + 1 norms, L exp tables,",
            "RELU, RECIP, SILU), data memory <= 2^19 bytes, and every shift that rounds (sh_q, sh_k, sh_v, sh_o, sh_1,",
            "sh_3, sh_g, sb) at most 15, because its rounding constant 2^(sh - 1) is a signed 16-bit preload (sh_s,",
            "sh_l and sa may be 0..23).", ""]
    rows = []
    for H, G in ((2, 1), (2, 2), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4), (8, 8), (16, 4), (16, 8)):
        try:
            b = profile(ModelConfig(n_heads=H, n_kv_heads=G), {"norm": "rms", "ffn": "swiglu"}, 0)[0]
            rows.append((H, G, f"{len(b.words):,}", f"{len(b.image):,}"))
        except ValueError as err:
            rows.append((H, G, "does not fit: " + str(err), ""))
    return "\n".join(out + [table(("H", "G", "program words", "data memory bytes"), rows)]) + "\n"


def complete_report():
    """The section about the complete program: prompt feeding, stop rule, syntax filter, sampling."""
    out = ["## The complete program: prompt, stop rule, syntax filter, sampling", "",
           "`gen.build(spec, arrays, prompt=True, tf=..., syntax_filter=..., temperature=...)`: the program feeds the",
           "prompt buffer, generates until the stop rule fires or the context is full, and halts. Shape as above",
           "(D = 64, L = 5, F = 172, V = 512, T = 256), rms norm, swiglu, tokenizer `data/tok/f512a`, zero weights,",
           "minima as above (new maxima of the argmax and bit 7 of the token id are left out; with the filter the",
           "first allowed token is one new maximum and is counted). The current token is a plain-text token in",
           "code at the start of a line (16 of the 512 token ids are then forbidden). With the filter the argmax",
           "loop takes 32 cycles per allowed id and 20 per forbidden id (20 per id without it); the rest of the",
           "filter (state step, abstract state, copy of the ALLOW row) depends on the path. Measured over the 10",
           "token sequences of `tests/test_ls_gen_complete.py` (strings, brackets, line breaks, control tokens):",
           "the rest is 676 to 888 cycles (848 in the state of this table) and 10 to 21 ids are forbidden, so in",
           "other states the filter column is at most 232 cycles lower or 112 higher.", "",
           "- one token: the program of the tables above (the test bench writes the token; no prompt, no stop rule).",
           "- generated: a generated token of the complete program with the stop rule, greedy, filter off.",
           "- + filter: with the syntax filter (state step, abstract state, 32 bytes of ALLOW copied, 12 cycles more per",
           "  token id in the argmax). + sampling: Gumbel-max at any temperature > 0 (28 cycles per token id).",
           "- prefill: a prompt token that is not the last one (no final norm, output layer, argmax, UART), filter on.",
           "- saving: generated (greedy, filter off) minus prefill with the filter off, and as a share of generated.", ""]
    options, rows, sizes = {"norm": "rms", "ffn": "swiglu"}, [], []
    for H, G in ((2, 1), (8, 4)):
        cfg = ModelConfig(n_heads=H, n_kv_heads=G)
        for pos in (15, 63, 255):
            one = cycles_per_token(cfg, options, pos)[0]
            plain = token_cycles(cfg, options, pos, stop=True)[0]
            filt = token_cycles(cfg, options, pos, syntax_filter=True)[0]
            sample = token_cycles(cfg, options, pos, stop=True, temperature=1.0)[0]
            both = token_cycles(cfg, options, pos, syntax_filter=True, temperature=1.0)[0]
            pre = token_cycles(cfg, options, pos, prefill=True, syntax_filter=True)[0]
            pre0 = token_cycles(cfg, options, pos, prefill=True, stop=True)[0]
            rows.append((H, G, pos + 1, f"{one:,}", f"{plain:,}", f"{filt:,} (+{filt - plain:,})",
                         f"{sample:,} (+{sample - plain:,})", f"{both:,} (+{both - plain:,})", f"{pre:,}", f"{pre0:,}",
                         f"{plain - pre0:,} ({100 * (plain - pre0) / plain:.1f}%)"))
        for name, kw in (("no options", None), ("prompt + stop rule", dict(stop=True)),
                         ("prompt + stop rule + filter", dict(syntax_filter=True)),
                         ("prompt + stop rule + filter + sampling", dict(syntax_filter=True, temperature=1.0))):
            b = profile(cfg, options, 0)[0] if kw is None else token_cycles(cfg, options, 0, **kw)[2]
            sizes.append((H, G, name, f"{len(b.words):,}", f"{len(b.image):,}"))
    out += [table(("H", "G", "context", "one token", "generated", "+ filter", "+ sampling", "+ filter + sampling",
                   "prefill (filter on)", "prefill (filter off)", "saving of the prefill skip"), rows), "",
            "### Components of the options (H = 8, G = 4, context 64), cycles", ""]
    cfg = ModelConfig(n_heads=8, n_kv_heads=4)
    parts = [token_cycles(cfg, options, 63, **kw)[1] for kw in
             (dict(stop=True), dict(syntax_filter=True), dict(syntax_filter=True, temperature=1.0), dict(prefill=True, syntax_filter=True))]
    labels = {"prompt": "prompt feeding and prefill test", "stop": "token class, level, stop rule",
              "filter": "syntax filter", "sample": "sampling", "argmax": "argmax", "output": "output layer",
              "book": "bookkeeping and UART"}
    out += [table(("component", "generated", "+ filter", "+ filter + sampling", "prefill (filter on)"),
                  [[labels[name]] + [f"{part.get(name, 0):,}" for part in parts] for name in labels]), "",
            "### Program words and data memory", "",
            table(("H", "G", "program", "program words (of 4,096)", "data memory bytes (of 524,288)"), sizes), ""]
    return out


def window_report():
    """The section about the attention window (cfg.window)."""
    out = ["## Attention window: a KV cache of W positions used as a ring", "",
           "`ModelConfig.window = W`: position p attends to the last min(p + 1, W) positions. The cache holds W",
           "slots; position p writes slot p mod W, and the score loop and the value rows run over the first",
           "min(p + 1, W) slots (the order of the keys does not matter). Slot and count are bytes of the position",
           "record, so the program is the one above except that a record has one more byte (12 cycles per token).",
           "From context W on a token costs what context W costs. Shape as above with H = 2, G = 1 (head size 32),",
           "rms norm, swiglu, the program without options, zero weights, minima as above.", "",
           "- With W < 256 a value row never ends at the end of its page, so the two instructions that set the page",
           "  of P1 again for every value row are not generated: 2,560 cycles less per token for this shape (8 per",
           "  row, 32 rows, 2 heads, 5 layers). This is why W = 64 and 128 are cheaper than W = 0 below context W too.",
           "- Data memory: the keys shrink from T d to W d bytes per layer and key/value head. A value row keeps its",
           "  256-byte slot (the loop counter and the slot byte form its address), of which W bytes are used; the",
           "  bytes that hold nothing are in the row `unused`.", ""]
    options, rows, sizes = {"norm": "rms", "ffn": "swiglu"}, [], []
    for window in (0, 64, 128):
        cfg = ModelConfig(n_heads=2, n_kv_heads=1, window=window)
        totals = [cycles_per_token(cfg, options, pos)[0] for pos in (15, 63, 255)]
        rows.append([window or "0 (whole context)"] + ["%s (%.1f)" % (f"{total:,}", 1e8 / total) for total in totals])
        regions = memory(profile(cfg, options, 0)[0])
        sizes.append([window or "0 (whole context)"] + [f"{regions[name]:,}" for name in
                     ("total", "KV cache: keys", "KV cache: values", "position records (links, counts, RoPE)", "unused (alignment)")])
    out += [table(("window W", "context 16: cycles (tokens/s)", "context 64", "context 256"), rows), "",
            table(("window W", "data memory bytes", "of them keys", "values", "position records", "unused"), sizes), ""]
    return out


def embedding_report():
    """The paragraph about the embedding shift of SPEC C.1."""
    rows = []
    for name, shape, options in (("stories260K, H = 8, G = 4, rms, swiglu (D = 64)", dict(), {"norm": "rms", "ffn": "swiglu"}),
                                 ("L = 2, D = 160, H = 4, G = 1, F = 219, rms, relu (D-shape-2)",
                                  dict(dim=160, n_layers=2, n_heads=4, n_kv_heads=1, hidden_dim=219), {"norm": "rms", "ffn": "relu"})):
        cfg = ModelConfig(**shape)
        plain, shifted = (cycles_per_token(cfg, opt, 63)[0] for opt in (options, dict(options, sh_e=1)))
        rows.append((name, f"{plain:,}", f"{shifted:,}", f"+{shifted - plain:,} ({100 * (shifted - plain) / plain:.2f}%)"))
    return ["## Embedding shift (SPEC C.1): x = rqr(E[token], sh_e)", "",
            "With `sh_e` > 0 the embedding row is not copied but passed through one more lookup table that holds",
            "rqr(v, sh_e) for every byte v: `LD R0, [P1]+ ; LUT R0, #EMB ; ST R0, [P2]+ ; DJNZ`, 16 cycles per element",
            "instead of 12, so 4 D cycles more per token at any shift 1..15 and any position (a product with 1 through",
            "the accumulator would be 20 per element). One more program word and one more table of 256 bytes. With",
            "`sh_e` = 0 or absent the program and the image are those of before. Every other number in this file is",
            "for `sh_e` = 0 (`cost.cycles_per_token(cfg, dict(options, sh_e=1), pos)` prices the shift). Context 64:", "",
            table(("shape", "cycles, sh_e = 0", "cycles, sh_e > 0", "added"), rows), ""]


def ask_report():
    """The section about choosing the prompt on the board: prompt table by switches, prompt over the UART."""
    options = {"norm": "rms", "ffn": "swiglu"}
    spec, arrays = dummy_model(ModelConfig(), options)
    spec["frac"] = {"u": 4}
    tf = tokenizer_filter()
    kw = dict(tf=tf, syntax_filter=True)
    token = tf.cls.index(24)                                     # a plain-text token, id below 128
    plain = gen.build(spec, arrays, prompt=True, **kw)
    table_b = gen.build(spec, arrays, prompts=[[token] * 16, [token] * 64], **kw)
    uart_b = gen.build(spec, arrays, uart_prompt=True, **kw)

    def start(b, sw=0, ids=None):
        cpu = gen.machine(b, sw=sw)
        if ids:
            cpu.feed(gen.frame(ids))
        cpu.run(stop=gen.top(b))
        return cpu.cycles
    t16, t64 = start(table_b, 0), start(table_b, 1)
    u16, u64 = start(uart_b, ids=[token] * 16), start(uart_b, ids=[token] * 64)
    rows = [("prompt table, chosen by the switches", "%d + %d n" % (t16 - 16 * (t64 - t16) // 48, (t64 - t16) // 48), f"{t16:,}", f"{t64:,}",
             len(table_b.words) - len(plain.words), "256 + 2 per token in the table (%d here)" % (len(table_b.image) - len(plain.image))),
            ("prompt over the UART (`RX`)", "%d + %d n, + 12 for the end mark" % (u16 - 16 * (u64 - u16) // 48, (u64 - u16) // 48),
             f"{u16 + 12:,}", f"{u64 + 12:,}", len(uart_b.words) - len(plain.words), len(uart_b.image) - len(plain.image))]
    return ["## Choosing the prompt on the board", "",
            "Two start-up pieces in front of the complete program (tag `start`; the per-token program after them is",
            "unchanged, so every cycle number above holds). Cycles are per answer, not per token; n = tokens of the",
            "prompt. stories260K shape, H = 8, G = 4, rms, swiglu, stop rule and syntax filter on.", "",
            "- Prompt table (`gen.build(..., prompts=[...])`, no new hardware): once after reset `IN` reads the switches,",
            "  SW5..0 select one of up to 64 prompts (a number beyond the list gives prompt 0), and the prompt is copied",
            "  into the prompt buffer (24 cycles per token).",
            "- UART prompt (`gen.build(..., uart_prompt=True)`, needs the proposed instruction `RX`): after reset and",
            "  after every answer the program resets position, stop level and filter state, reads two bytes per token",
            "  up to the byte 0xFF (76 cycles per token, 8 more when bit 7 of the id is set; waiting for the UART is not",
            "  counted), answers, sends 0xFF and waits again. Without the filter the fixed part is 100 cycles (148 with",
            "  the stop rule only) and the piece is 44 (56) words.", "",
            table(("variant", "start-up cycles per answer", "n = 16", "n = 64", "program words added", "data bytes added"), rows), ""]


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "COST_TABLE.md"), "w") as f:
        f.write(report())
