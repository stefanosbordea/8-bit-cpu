"""Test vectors: what the complete program must send for a prompt, and every intermediate value of the first
prompt position and of the first generated tokens, from the integer reference.

vectors/README.md (written here, text in lsllm/export/docs.py) explains the files, the names and the order.
"""
import glob
import os

import numpy as np

from lsllm.export import docs, files
from lsllm.filter_table import State
from lsllm.intref.completer import LINE_BREAKS
from lsllm.intref.model import IntModel


def reference(model, tok, tf, ids, syntax_filter, temperature, keep=0, seed=0, lfsr=None):
    """What the complete program must send after the prompt `ids`, switches at `seed` (lfsr: the state of the
    random generator to go on from, for a second prompt over the UART without a reset): the loop of
    lsllm.intref.completer.IntCompleter for one sample, written with IntModel, TableFilter and Tok directly (as
    in tests/test_ls_gen_complete.py). The token that ends the text is sent too.

    Returns (tokens sent, steps, ended). steps: the traced values of position 0, then of the first `keep`
    generated tokens (the step whose logits choose the token), each as name -> array. ended: the stop rule fired,
    so the last token sent is the end mark (otherwise the context is full)."""
    model.reset(seed=seed)
    if lfsr is not None:
        model.lfsr = lfsr
    state, level, sent, steps = State(), tok.level_after(ids), [], []

    def feed(t):
        trace = {}
        u = model.step([t], trace)
        tf.step(state, t)
        return u, {"position": model.pos - 1, "token_in": t, **{name: value[0] for name, value in trace.items()}}

    for p, t in enumerate(ids):
        u, values = feed(t)
        if p == 0:
            steps.append(values)
    while True:
        allowed = np.array([tf.allowed(state)]) if syntax_filter else None
        t = int(model.choose(u, temperature, allowed)[0])
        sent.append(t)
        if len(sent) <= keep:
            mask = {"allowed": allowed[0].astype(int)} if syntax_filter else {}
            steps.append({**values, **mask, "token_out": t, "uart": docs.frame([t])})
        level = tok.level_after([t], level)
        ended = t == tok.eos_id or level == 0 and any(m in tok.pieces[t] for m in LINE_BREAKS)       # the stop rule
        if ended or model.pos == model.cfg.max_seq_len:
            return sent, steps, ended
        u, values = feed(t)


def vectors(model_dir, tok, tf, prompt, ids, n, syntax_filter, temperature, directory, form="single"):
    """Write the test vectors into `directory`: `n` traced tokens after `prompt` (token ids `ids`).
    form: how the program gets this prompt (lsllm.export.tables.form_of), for the README."""
    model = IntModel.load(model_dir)
    sent, steps, ended = reference(model, tok, tf, ids, syntax_filter, temperature, keep=n)
    summary = {"prompt_ids": ids, "sent": sent, "uart": docs.frame(sent), "end_mark": int(ended)}
    others = {"sent_greedy": (False, 0.0), "sent_filter": (True, 0.0)}
    if temperature > 0:
        others["sent_sampled"] = (syntax_filter, temperature)
    for name, setting in others.items():                # the same prompt in the other settings of the program
        summary[name] = sent if setting == (syntax_filter, temperature) else reference(model, tok, tf, ids, *setting)[0]
    first = steps[0] if len(ids) == 1 else {name: value for name, value in steps[0].items()
                                            if name in ("position", "token_in", "x0") or name[0] == "l"}
    named = {"position_00": first, **{"token_%02d" % g: step for g, step in enumerate(steps) if g}}

    def write(name, content):
        with open(os.path.join(directory, name), "w" if isinstance(content, str) else "wb") as f:
            f.write(content)

    for old in glob.glob(os.path.join(directory, "token_*.txt")):       # of an earlier export with more tokens
        os.remove(old)
    for name, step in named.items():
        write(name + ".txt", files.vector_text(step))
    write("tokens.txt", files.vector_text(summary))
    write("uart.bin", bytes(summary["uart"]))
    write("README.md", docs.vectors_readme(prompt, summary, named, tok, syntax_filter, temperature, form))
    np.savez_compressed(os.path.join(directory, "vectors.npz"), **summary,
                        **{"%s.%s" % (key, name): value for key, step in named.items() for name, value in step.items()})
