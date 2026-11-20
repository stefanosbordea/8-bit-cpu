"""docs/SPEED.md and docs/HARDWARE_CHANGES.md for one model shape: cycles per token of the generated LS program,
tokens per second at 100 MHz, the breakdown, the stream speed-up against the classic MAC, and the optional
hardware changes with their speed-up and their estimated cost on the Artix-7.

    PYTHONPATH=. .venv/bin/python tools/speed_report.py runs/NAME_int        (an exported integer model)
"""
import json
import os
import sys

from lsllm.config import ModelConfig
from lsllm.ls import cost

CLOCK = 100e6
CONTEXTS = (16, 64, 256)
num = lambda v: "{:,}".format(int(round(v)))
rate = lambda c: "%.1f" % (CLOCK / c)

# estimates of research/09_fpga_costs.md, section 6.2 and 6.3 (LUT, flip-flops, DSP48E1, 36 Kb block RAM)
L2_COST = {
    "fused row": ("60 to 120", "50 to 80", "0", "0", "row counter, reload of the X pointer, length register, 5 to 8 more FSM states; the largest single gain per LUT"),
    "2 MAC lanes": ("30 to 80", "5 to 50", "+1", "0", "second DSP48E1; the data memory is read 16 bits wide on one port (block RAM ports can have different widths)"),
    "4 MAC lanes": ("50 to 200", "10 to 100", "+3", "0", "three more DSP48E1; 32-bit read port; tail mask for row lengths that are not a multiple of 4"),
    "hardware argmax": ("15 to 30", "34 to 44", "0", "0", "comparator on the requantiser output with a value and an index register; also gives the running maximum of the attention scores"),
    "table in a stream": ("35 to 60", "25 to 40", "0", "0", "store pointer and table select in the stream pipeline, write-data multiplexer"),
    "signed-compare branch": ("1 to 3", "0", "0", "0", "one more branch condition (N flag)"),
    "3-cycle instructions": ("10 to 30", "0 to 10", "0", "0", "FETCH and DECODE merged for instructions without a memory operand; changes the FSM that the course design fixes at 4 cycles"),
    "all together": ("about 220 to 450", "about 150 to 300", "+3", "0.5", "sum of the rows above with 4 lanes (research/09, section 6.3)"),
}
L1_ROWS = [
    ("Program ROM of up to 4,096 words in block RAM (the course ROM has 256 words)", "the program of the model has {words} words; the PC already has 12 bits", "new size; a block RAM reads synchronously, so FETCH needs the instruction register", "0", "16 (instruction register)", "0", "1 of 135"),
    ("Data memory D in block RAM, one memory with two read ports, up to 2^19 bytes", "weights, KV cache, tables, scratch; a stream multiplies D[P0] by D[P1]", "planned (X and W in block RAM); unified here", "0 to about 34 LUT per data bit for the read multiplexer of a memory made of many blocks", "0", "0", "{bram} of 135"),
    ("P0 and P1 widened to 20 bits, new store pointer P2, `MVP`, `MVH`, `CPP`", "addresses above 4,096; a result row is stored while P0 and P1 walk through the operands", "planned (wider pointers); P2 is new", "about 60 to 120 (three 20-bit counters with load)", "60", "0", "0"),
    ("`ST Rs, [Pp]+` through P0 or P2", "writes of activations and of the KV cache", "planned (store instruction)", "about 10", "0", "0", "0"),
    ("`LEN` register and `MACL`", "stream length known only at run time (keys in the context), and rows longer than 255", "new", "12 to 25", "12", "0", "0"),
    ("`LUT Rd, #t` with 64 tables of 256 bytes in D", "norm, exp, reciprocal, SiLU or ReLU-max, Gumbel, filter tables", "planned (lookup instruction); the table id is widened to 6 bits", "about 10 (address = base + 256 t + Rd)", "0", "0", "in D"),
    ("`NRM Rd`: number of significant bits of ACC", "attention division: exponent of the sum of the weights", "new", "15 to 25 (priority encoder over 16 bits)", "0", "0", "0"),
    ("`RQV Rd, Rs`: requantise with the shift taken from a register", "attention division: shift by that exponent", "new", "5 to 10 (the shifter exists; one multiplexer on the shift amount)", "0", "0", "0"),
    ("`TX Rs`: UART transmitter 115,200 baud 8N1 with a 16-byte FIFO", "token output, two bytes per token", "planned (UART transmitter)", "30 to 35", "31", "0", "0"),
    ("`RND Rd`: 32-bit LFSR, one byte per call", "Gumbel-max sampling only (greedy decoding does not need it)", "planned (LFSR); 32 bits instead of 16 (D-arch-4)", "about 8", "32", "0", "0"),
]


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def main(directory):
    spec = json.load(open(os.path.join(directory, "model.json")))
    cfg, options = ModelConfig(**spec["config"]), {"norm": spec["options"]["norm"], "ffn": spec["options"]["ffn"]}
    shape = "L = %d, D = %d, heads = %d, KV heads = %d, F = %d (%s), V = %d, T = %d%s" % (
        cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.hidden_dim, cfg.ffn, cfg.vocab_size, cfg.max_seq_len,
        ", window %d" % cfg.window if cfg.window else "")
    reach = lambda n: min(n, cfg.window) if cfg.window else n
    plain = ModelConfig(**dict(cfg.to_dict(), attn="softmax", window=0))
    at = {n: reach(n) - 1 for n in CONTEXTS}                       # position that costs like context n
    S = ["# Speed of the model on the LS CPU", "",
         "Generated by `tools/speed_report.py %s` from the program that `lsllm.ls.gen` writes for this shape, run in the" % directory,
         "simulator on zero weights (the cycle count does not depend on the weights, except at most 2,044 cycles of the",
         "argmax and the UART frame, see `docs/COST_TABLE.md`). Machine: level L1 of `docs/SPEC.md`: every instruction",
         "4 cycles, a stream of n multiply-adds n + 4 cycles, 100 MHz. Shape: %s." % shape, ""]

    # 1. the complete program
    kinds = [("generated token, greedy", {}), ("+ syntax filter", {"syntax_filter": True}), ("+ sampling", {"temperature": 0.8}),
             ("+ syntax filter + sampling", {"syntax_filter": True, "temperature": 0.8}),
             ("prompt token (prefill skip), filter on", {"prefill": True, "syntax_filter": True})]
    totals = {name: [cost.token_cycles(plain, options, at[n], stop=True, **kw)[0] for n in CONTEXTS] for name, kw in kinds}
    S += ["## Cycles per token and tokens per second", ""]
    S += table(["token of the complete program"] + ["cycles @%d" % n for n in CONTEXTS] + ["tokens/s @%d" % n for n in CONTEXTS],
               [[name] + [num(c) for c in cs] + [rate(c) for c in cs] for name, cs in totals.items()])
    deployed = totals["+ syntax filter"]
    S += ["", "The deployed setting is greedy decoding with the syntax filter: **%s / %s / %s cycles per token = %s / %s / %s tokens per second** at context 16 / 64 / 256."
          % tuple([num(c) for c in deployed] + [rate(c) for c in deployed]), ""]

    # 2. breakdown
    parts = [cost.token_cycles(plain, options, at[n], stop=True, syntax_filter=True)[1] for n in CONTEXTS]
    labels = dict(cost.LABELS)
    names = [name for name in parts[0] if any(p.get(name, 0) for p in parts)]
    S += ["## Where the cycles go (generated token, greedy, filter on)", ""]
    S += table(["component"] + ["cycles @%d" % n for n in CONTEXTS] + ["share @%d" % n for n in CONTEXTS],
               [[labels.get(name, name)] + [num(p.get(name, 0)) for p in parts] + ["%.1f%%" % (100 * p.get(name, 0) / sum(p.values())) for p in parts]
                for name in sorted(names, key=lambda name: -parts[1].get(name, 0))])
    S += [""]

    # 3. stream against classic
    S += ["## What the stream instruction is worth", "",
          "The same model, the same program, three ways to do a dot product of n terms: `MAC #n` (n + 4 cycles, what the",
          "LS CPU has), (a) n classic `MAC` instructions in a row (4 cycles each, the definition of the project brief; that",
          "program has one word per product and does not fit the 4,096-word ROM), (b) a loop `MAC ; DJNZ` (8 cycles per",
          "product, fits the ROM). One token without prompt handling, filter or sampling (they contain no streams).", ""]
    rows = []
    for n in CONTEXTS:
        total = cost.cycles_per_token(plain, options, at[n])[0]
        a, b = cost.classic(plain, options, at[n])
        products = cost.multiply_adds(plain, options, at[n])[0]
        rows.append([n, num(total), num(products), "%.1f%%" % (100 * products / total), num(a), "%.2f" % (a / total), num(b), "%.2f" % (b / total)])
    S += table(["context", "cycles with streams", "multiply-adds", "share of cycles that multiply", "classic (a) cycles", "speed-up vs (a)",
                "classic (b) cycles", "speed-up vs (b)"], rows)
    S += ["", "Program size: %s words with streams, %s words for classic (a)." % (
        num(len(cost.profile(plain, options, 0)[0].code)), num(cost.classic_words(plain, options))), ""]
    open("docs/SPEED.md", "w").write("\n".join(S) + "\n")

    # hardware changes
    b = cost.profile(plain, options, 0)[0]
    blocks = -(-len(b.image) // 4096)
    H = ["# Hardware changes to the LS CPU", "",
         "Generated by `tools/speed_report.py %s`. Costs are estimates from the reference data collected in" % directory,
         "`research/09_fpga_costs.md` (sections 6.2 and 6.3: counted in vendor sources where marked there, otherwise",
         "estimated from the primitives of the Artix-7); nothing here was synthesised. The device has 63,400 LUTs,",
         "126,800 flip-flops, 240 DSP48E1 and 135 block RAMs of 36 Kb. Shape of the model: %s." % shape, "",
         "## 1. Needed to run the model at all (level L1)", "",
         "The headline numbers of the report assume exactly this list and nothing from section 2.", ""]
    H += table(["change", "what the model uses it for", "status", "LUT", "flip-flops", "DSP", "block RAM"],
               [[c.format(bram=blocks, words=num(len(b.code))) for c in row] for row in L1_ROWS])
    H += ["", "Data memory of this model: %s bytes = %d block RAMs of 4,096 bytes; program ROM %s words of 16 bits (1 block RAM)." % (
        num(len(b.image)), blocks, num(len(b.code))),
          "The streamed multiply-accumulate itself, `RQ`, `LDA`, the registers, the 4-cycle FSM and the cycle counter are the course design and are unchanged.", "",
          "Choosing the prompt on the board needs nothing more: the memory image holds a table of up to 64 prompts and the program",
          "takes the one selected by switches SW5..0 at reset (`IN`, a baseline instruction). To type a prompt on the laptop instead,",
          "one more instruction is proposed: `RX Rd` (op 0, fn 8: the next byte of a UART receiver, the CPU waits while none has",
          "arrived). The program then answers one prompt after the other without a reset. Estimated cost of the receiver with a",
          "16-byte FIFO: about 30 to 40 LUTs and 30 to 35 flip-flops, by the same count as the transmitter (research/09, section 6.2).", "",
          "## 2. Optional changes that make it faster (level L2)", "",
          "Each row is the same program priced with one change (the modelling assumption of each row is in `docs/COST_TABLE.md`).",
          "None of them is assumed anywhere else in the report.", ""]
    base = [cost.cycles_per_token(plain, options, at[n])[0] for n in CONTEXTS]
    rows = [["L1 (no change)"] + [num(c) for c in base] + [rate(c) for c in base] + ["1.00", "-", "-", "-", "-", ""]]
    for name, (switches, assumption) in cost.L2.items():
        cs = [cost.cycles_per_token(plain, options, at[n], **switches)[0] for n in CONTEXTS]
        lut, ff, dsp, bram, note = L2_COST[name]
        rows.append([name] + [num(c) for c in cs] + [rate(c) for c in cs] + ["%.2f" % (base[1] / cs[1]), lut, ff, dsp, bram, note])
    H += table(["change"] + ["cycles @%d" % n for n in CONTEXTS] + ["tokens/s @%d" % n for n in CONTEXTS] +
               ["speed-up @64", "LUT", "flip-flops", "DSP", "block RAM", "what it takes"], rows)
    H += ["", "## 3. Needed only for Tier B (weights in the 128 MiB DDR2)", "",
          "See `docs/TIER_B.md` for the speed model. The CPU needs:", "",
          "- the memory controller (Xilinx MIG for the MT47H64M16HR): size for a 16-bit interface not found in the vendor documents;",
          "  between about 1,300 and 9,300 LUTs and 900 to 6,000 flip-flops (lower mark: a lean 16-bit DDR3 core; upper mark: the vendor",
          "  figure for 72 bits), 1 PLL, 1 MMCM. The Digilent SRAM adapter is too slow (one 16-bit word per 210 ns);",
          "- a weight tape: the matrices stored in the order the program reads them, a 27-bit tape pointer, and a prefetch FIFO",
          "  (two FIFO36E1, about 100 to 200 LUTs and 100 to 150 flip-flops) that hands the stream one byte per clock; the stream",
          "  stalls when the FIFO is empty. The controller delivers up to 16 bytes per user clock (1.3 GB/s peak), the stream uses",
          "  at most 0.1 GB/s, so the multiply-accumulate rate is the limit, not the memory;",
          "- a variant of the stream instructions whose second operand comes from the tape instead of D[P1], and a tape rewind;",
          "- loading the DDR2 after power-up from the 16 MiB flash or the microSD card (a small copy state machine); the UART is",
          "  too slow for that at 115,200 baud (3.2 hours for 128 MiB) and needs 1 to 4 Mbaud;",
          "- KV cache, embedding matrix, tables and scratch stay in block RAM.", "",
          "## 4. Considered and not proposed", "",
          "- A wider accumulator: not needed, no dot product of the model has more than 511 terms (worst case in `docs/OVERFLOW.md`).",
          "- A UART receiver: the prompt is written into the memory image; interactive prompts would need `RX` (about 30 LUTs).",
          "- Packing two products into one DSP48E1: 239 DSP slices are idle, one slice per lane is simpler (research/09)."]
    open("docs/HARDWARE_CHANGES.md", "w").write("\n".join(H) + "\n")
    print("\n".join(S[7:16]))
    print("wrote docs/SPEED.md and docs/HARDWARE_CHANGES.md")


if __name__ == "__main__":
    main(sys.argv[1])
