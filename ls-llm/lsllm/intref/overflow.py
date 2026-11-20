"""Accumulator use of an integer model (SPEC B.4): per stream the worst case and the largest value seen on data.

    .venv/bin/python -m lsllm.intref.overflow runs/NAME_int --data data/tok/NAME

The worst case of a stream over a weight matrix is |preload| + 128 * (largest row sum of |w|); for a stream of
two run-time vectors it is |preload| + the number of terms times the product of their limits.
"""
import argparse
import os

import numpy as np

from lsllm.intref.model import ACC_MAX, IntModel, half


def bounds(model):
    """Stream name (as in the trace of IntModel) -> worst-case |accumulator|."""
    cfg, sh, pre = model.cfg, model.sh, model.pre

    def matrix(name, preload):
        return preload + 128 * int(np.abs(getattr(model, name)).sum(1).max())

    out = {}
    if sh.get("sh_e", 0):
        out["x0"] = half(sh["sh_e"]) + 127                                      # one embedding value plus the rounding constant
    sites = [("l%d.ya" % (n // 2) if n % 2 == 0 else "l%d.yf" % (n // 2)) for n in range(2 * cfg.n_layers)] + ["y"]
    if model.opt["norm"] == "rms":
        for n, name in enumerate(sites):
            out[name + ".m"] = cfg.dim * 128 * 128                              # sum of squares
            out[name] = half(sh["sb"][n]) + 128 * 127                           # x * r
    for l in range(cfg.n_layers):
        at, tag = "l%d." % l, "_%d" % l
        for name in ("q", "k", "v"):
            out[at + name] = matrix("W" + name + tag, half(sh["sh_" + name][l]))
        out[at + "qr"] = out[at + "kr"] = 64 + 2 * 128 * 127                    # two products per rotated value
        out[at + "code"] = int(np.abs(pre["c_s"][l]).max()) + cfg.head_dim * 128 * 128
        out[at + "a"] = (cfg.window or cfg.max_seq_len) * 127 * 128             # weights 0..127 times values
        out[at + "o"] = half(sh["sh_o"][l]) + 128 * 127                         # a * RECIP[mant]
        out[at + "xa"] = matrix("Wo" + tag, 256 * 128 + 128)
        if model.opt["ffn"] == "swiglu":
            out[at + "h1"] = matrix("W1" + tag, half(sh["sh_1"][l]))
            out[at + "h3"] = matrix("W3" + tag, half(sh["sh_3"][l]))
            out[at + "g"] = half(sh["sh_g"][l]) + 128 * 128
        else:
            out[at + "g"] = matrix("W1" + tag, half(sh["sh_1"][l]))
        out[at + "xf"] = matrix("W2" + tag, 256 * 128 + 128)
    out["u"] = matrix("E", abs(pre["c_l"]))
    return out


def report(model, tokens):
    """Run the model on `tokens` (rows, time). Returns one row per stream: name, worst-case bound, largest seen."""
    model.peak = {}
    model.forward(tokens)
    return [{"stream": name, "bound": bound, "seen": model.peak[name]} for name, bound in bounds(model).items()]


def main():
    p = argparse.ArgumentParser(description="Accumulator bounds of an exported integer model.")
    p.add_argument("directory")
    p.add_argument("--data", required=True, help="tokenizer directory; windows of val.bin are the data")
    p.add_argument("--windows", type=int, default=16)
    args = p.parse_args()
    model = IntModel.load(args.directory)
    T = model.cfg.max_seq_len
    val = np.fromfile(os.path.join(args.data, "val.bin"), dtype=np.uint16).astype(np.int64)
    starts = np.linspace(0, len(val) - T, args.windows).astype(np.int64)
    rows = report(model, val[starts[:, None] + np.arange(T)])
    print("%-12s %10s %10s   (24-bit limit %d)" % ("stream", "worst case", "seen", ACC_MAX))
    for row in sorted(rows, key=lambda row: -row["bound"]):
        print("%-12s %10d %10d" % (row["stream"], row["bound"], row["seen"]))
    worst = max(rows, key=lambda row: row["bound"])
    print("largest worst case: %s, %.1f%% of the 24-bit range; largest value seen: %.1f%%" % (
        worst["stream"], 100 * worst["bound"] / ACC_MAX, 100 * max(row["seen"] for row in rows) / ACC_MAX))


if __name__ == "__main__":
    main()
