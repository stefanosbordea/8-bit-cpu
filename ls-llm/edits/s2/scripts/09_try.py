"""One experiment = edits through Session.put / put_number, then one evaluate (one budgeted run).
    python 09_try.py NAME [ARGS]      apply the experiment NAME and evaluate
    python 09_try.py undo             Session.undo: take back the last edit (no run)
    python 09_try.py keep NOTE        remember the present model as the kept state
    python 09_try.py back             put the model back to the kept state (the base model before any keep)
    python 09_try.py table            the runs so far
    python 09_try.py diff             which arrays differ from the base model (reads weights only)
Every factor is a number chosen by hand from the analysis scripts; nothing here is fitted."""
import json, sys
import numpy as np
from lsllm.edit.session import Session

s = Session("edits/s2")

def scale(name, index, factor, reason):
    """Multiply rows/columns `index` (a numpy index into the array) of `name` by factor, rounded, clipped to +-127."""
    w = s.get(name).astype(float)
    w[index] = np.clip(np.round(w[index] * factor), -127, 127)
    return s.put(name, w, reason)

def head_cols(h):
    return (slice(None), slice(40 * h, 40 * h + 40))

def head_rows(h):
    return (slice(40 * h, 40 * h + 40), slice(None))

def exp_table(l, factor, reason):
    """Rebuild EXP[l] from its formula with kappa * factor (factor < 1: sharper)."""
    kappa = 2.0 ** s.spec["frac"]["s"][l] * factor
    table = s.get("EXP")
    d = np.arange(1, 128)
    table[l, 256 - d] = np.round(127 * np.exp(-d / kappa))
    return s.put("EXP", table, reason)

def wo(h, f):      # what head h of layer 1 writes
    scale("Wo_1", head_cols(int(h)), float(f), "Wo_1 columns of head %s times %s: what this head writes into the residual stream" % (h, f))

def wq(h, f):      # how sharply head h of layer 1 attends
    scale("Wq_1", head_rows(int(h)), float(f), "Wq_1 rows of head %s times %s: larger queries, sharper attention of this head" % (h, f))

def wo0(h, f):
    scale("Wo_0", head_cols(int(h)), float(f), "Wo_0 columns of head %s times %s" % (h, f))

def wq0(h, f):
    scale("Wq_0", head_rows(int(h)), float(f), "Wq_0 rows of head %s times %s" % (h, f))

def exp(l, f):
    exp_table(int(l), float(f), "EXP[%s] rebuilt with kappa times %s" % (l, f))

def cs(l, h, delta):
    old = s.spec["preloads"]["c_s"][int(l)][int(h)]
    s.put_number(("preloads", "c_s", int(l), int(h)), old + int(delta), "c_s[%s][%s] moved by %s" % (l, h, delta))

def units(l, f, *ks):     # what feed-forward units of layer l write
    scale("W2_%s" % l, (slice(None), [int(k) for k in ks]), float(f), "W2_%s columns %s times %s" % (l, ",".join(ks), f))

def ffn(l, f):         # everything the feed-forward of layer l writes
    scale("W2_%s" % l, (slice(None), slice(None)), float(f), "W2_%s times %s: the whole feed-forward of this layer writes less (habit against copy)" % (l, f))

def rows_e(f, *ts):
    scale("E", ([int(t) for t in ts], slice(None)), float(f), "E rows %s times %s" % (",".join(ts), f))

def sigpref(codes, head="0", pair="19", const="60", neutral="0"):
    """A hand-made channel in one RoPE pair of a layer 1 head (the slowest pair turns by 0.016 rad over 100 positions,
    so it does not see the distance): the query gets a constant (the mean input of the body positions as a
    direction, about `const` at every body position), the key gets signature minus docstring (difference of the
    class means of l1.ya, script 13). The score of a signature key then rises by `codes` codes over a docstring key
    (4 codes = 1 nat). The old rows of that pair (a small part of the old scores) are replaced."""
    codes, head, pair, const = float(codes), int(head), int(pair), float(const)
    mean = np.load("edits/s2/regions.npz")
    m = mean["body"] / np.linalg.norm(mean["body"])
    d = mean["sig"] - mean["doc"]
    if neutral == "1":          # script 19: the tokens of the function name get the level of the docstring
        n = mean["name"] - mean["doc"]
        d = d - (d @ n) / (n @ n) * n
    d = d / np.linalg.norm(d)
    body_level = float(mean["body"] @ m)                           # 85.8: what the body positions give along m
    gap = float((mean["sig"] - mean["doc"]) @ d)                   # 112: signature minus docstring along d
    q_row = np.round(const * 256 / body_level * m)                 # q = row . y / 2^8
    k_gap = codes * 256 / const                                    # wanted difference of the key value
    k_row = np.round(k_gap * 256 / gap * d)                        # k = row . y / 2^8
    wq, wk = s.get("Wq_1"), s.get("Wk_1")
    wq[40 * head + 2 * pair] = q_row
    wk[2 * pair] = k_row
    note = "signature preference of layer 1 head %d: %s codes, pair %d, constant %s%s" % (head, codes, pair, const, ", function name at docstring level" if neutral == "1" else "")
    s.put("Wq_1", wq, note + " (query row: constant at body positions)")
    s.put("Wk_1", wk, note + " (key row: signature minus docstring)")
    print("q row max %d, k row max %d" % (np.abs(q_row).max(), np.abs(k_row).max()))

def endbias(delta):
    """E row of </s> plus delta times the unit direction "the reference ends here" of the final norm output
    (script 17): the end logit rises by about delta * 191 / 512 where references end, and falls by
    delta * 21 / 512 on average elsewhere in the body."""
    unit = np.load("edits/s2/end.npz")["unit"]
    e = s.get("E")
    e[2] = np.clip(e[2] + np.round(float(delta) * unit), -127, 127)
    s.put("E", e, "E row of </s> plus %s times the end direction: stop where a reference would stop" % delta)

def gated(codes, const="60", pair="19", head="0"):
    """The signature preference of `sigpref`, switched on only where a name is due (script 20): the query row is the
    combination of the "name due" direction and the body-mean direction that gives `const` at the mean input of the
    name-due positions and 0 at the mean input of the other body positions (two equations, two unknowns)."""
    codes, const, pair, head = float(codes), float(const), int(pair), int(head)
    mean, due = np.load("edits/s2/regions.npz"), np.load("edits/s2/due.npz")
    m = mean["body"] / np.linalg.norm(mean["body"])
    basis = np.stack([due["unit"], m])
    # class means of the two groups in the coordinates (along unit, along m)
    pts = np.array([due["mean_due"] @ basis.T, due["mean_rest"] @ basis.T])
    coef = np.linalg.solve(pts, np.array([const * 256, 0.0]))
    q_row = np.round(coef @ basis)
    d = mean["sig"] - mean["doc"]
    gap = float(np.linalg.norm(d)); d = d / gap
    k_row = np.round(codes * 256 / const * 256 / gap * d)
    wq, wk = s.get("Wq_1"), s.get("Wk_1")
    wq[40 * head + 2 * pair] = q_row
    wk[2 * pair] = k_row
    note = "gated signature preference of layer 1 head %d: %s codes where a name is due, pair %d, query %s there and 0 elsewhere" % (head, codes, pair, const)
    s.put("Wq_1", wq, note + " (query row)")
    s.put("Wk_1", wk, note + " (key row)")
    print("q row max %d, k row max %d, q at name due %.1f, at rest %.1f" % (np.abs(q_row).max(), np.abs(k_row).max(), q_row @ due["mean_due"] / 256, q_row @ due["mean_rest"] / 256))

def noquote(delta):
    """E row of the triple quote minus delta times the body direction of script 25: its logit falls by about
    delta * 105 / 512 in the body and stays, on average, where a triple quote is due. The row is also how the two
    quotes of the prompt are READ, so this edit can change more than the three answers it is aimed at."""
    unit = np.load("edits/s2/quote.npz")["unit"]
    e = s.get("E")
    e[24] = np.clip(e[24] - np.round(float(delta) * unit), -127, 127)
    s.put("E", e, "E row of the triple quote minus %s times the body direction: no docstring inside the body" % delta)

def sigmen(n_codes, m_codes, const="60", pair="19", head="0"):
    """`sigpref` with a second class (script 26): the key row is the combination of the two directions
    (signature - other docstring tokens) and (argument mentions in the docstring - other docstring tokens) that
    lifts the mean signature key by n_codes and the mean mention by m_codes over the other docstring tokens
    (two equations, two unknowns). The query row is the constant of `sigpref`."""
    n_codes, m_codes, const, pair, head = float(n_codes), float(m_codes), float(const), int(pair), int(head)
    mean, men = np.load("edits/s2/regions.npz"), np.load("edits/s2/mention.npz")
    m = mean["body"] / np.linalg.norm(mean["body"])
    q_row = np.round(const * 256 / float(mean["body"] @ m) * m)
    a, b = mean["sig"] - men["mean_oth"], men["mean_men"] - men["mean_oth"]
    basis = np.stack([a / np.linalg.norm(a), b / np.linalg.norm(b)])
    want = np.array([n_codes, m_codes]) * 256 / const * 256          # wanted key value times 2^8
    coef = np.linalg.solve(np.stack([a, b]) @ basis.T, want)
    k_row = np.round(coef @ basis)
    wq, wk = s.get("Wq_1"), s.get("Wk_1")
    wq[40 * head + 2 * pair] = q_row
    wk[2 * pair] = k_row
    note = "preference of layer 1 head %d for signature keys (%s codes) and argument mentions of the docstring (%s codes), pair %d" % (head, n_codes, m_codes, pair)
    s.put("Wq_1", wq, note + " (query row)")
    s.put("Wk_1", wk, note + " (key row)")
    print("k row max %d; key value: signature %.1f, mention %.1f, other docstring %.1f, body %.1f, function name %.1f" % (
        np.abs(k_row).max(), *(k_row @ v / 256 for v in (mean["sig"], men["mean_men"], men["mean_oth"], mean["body"], mean["name"]))))

def back():
    """Put the model back to the kept state (the base model until `keep` is called), through put / put_number."""
    import os
    if os.path.exists("edits/s2/kept.npz"):
        arrays, numbers = np.load("edits/s2/kept.npz"), json.load(open("edits/s2/kept.json"))
    else:
        arrays, numbers = np.load(s.budget["base"] + "/model.npz"), json.load(open(s.budget["base"] + "/model.json"))
    for k in arrays.files:
        if (arrays[k] != s.arrays[k]).any():
            print("back:", k, s.put(k, arrays[k], "back to the kept state")["changed"])
    for l in range(2):
        for h in range(4):
            if s.put_number(("preloads", "c_s", l, h), numbers["preloads"]["c_s"][l][h], "back to the kept state"):
                print("back: c_s", l, h)
    assert (numbers["shifts"], numbers["preloads"]) == (s.spec["shifts"], s.spec["preloads"])

def table():
    for run in s.budget["runs"]:
        r = run.get("result")
        if r:
            print("run %2d edits %2d  pass %.3f  tests %.4f  syntax %.3f  runs_ok %.3f  loss %.4f  Name %2d Type %2d  rep %.3f | %s" % (
                r["run"], run["edits_so_far"], r["pass@1"], r["tests_passed"], r["syntax_ok"], r["runs_ok"], r["practice_loss"],
                r["errors"].get("NameError", 0), r["errors"].get("TypeError", 0), r["repeating"], r["note"]))
        else:
            print("run %2d %s | %s" % (run["run"], run["kind"], run["note"]))

if __name__ == "__main__":
    name, args = sys.argv[1], sys.argv[2:]
    if name == "undo":
        print("undo:", s.undo()["reason"])      # one edit only: a second call would undo the undo
    elif name == "keep":
        # remember the present model as the kept state (weights only, inside the session directory)
        np.savez("edits/s2/kept.npz", **s.arrays)
        json.dump({"shifts": s.spec["shifts"], "preloads": s.spec["preloads"], "note": " ".join(args)}, open("edits/s2/kept.json", "w"))
        print("kept:", " ".join(args))
    elif name == "back":
        back()
    elif name == "diff":
        base = np.load(s.budget["base"] + "/model.npz")
        spec = json.load(open(s.budget["base"] + "/model.json"))
        print("arrays that differ from the base model:", [(k, int((base[k] != s.arrays[k]).sum())) for k in base.files if (base[k] != s.arrays[k]).any()],
              "| numbers:", "same" if (spec["shifts"], spec["preloads"]) == (s.spec["shifts"], s.spec["preloads"]) else "DIFFERENT")
    elif name == "table":
        table()
    else:
        # several steps in one experiment: NAME ARGS + NAME ARGS
        steps, cur = [], []
        for a in [name] + args:
            if a == "+":
                steps.append(cur); cur = []
            else:
                cur.append(a)
        steps.append(cur)
        noeval = steps[-1] == ["noeval"]          # "... + noeval": apply only, no run
        steps = steps[:-1] if noeval else steps
        for step in steps:
            globals()[step[0]](*step[1:])
        note = " + ".join(" ".join(step) for step in steps)
        if noeval:
            sys.exit()
        r = s.evaluate(note)
        print(json.dumps(r))
        table()
