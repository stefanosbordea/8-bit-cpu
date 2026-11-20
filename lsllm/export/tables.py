"""What is exported of a model directory: its arrays, the tables of the syntax filter, the GUMBEL table,
and the Build of the complete LS program with the prompt in its image."""
import os
import re

import numpy as np

from lsllm.filter_table import ROW, WORD_ROW, TableFilter
from lsllm.intref.model import gumbel_table
from lsllm.ls import gen
from lsllm.tok import Tok


TRIPLE_QUOTES = True         # False: the filter of the export never lets the model open a docstring (`--no-docstring`)


def load(model_dir, temperature):
    """(spec, arrays, tokenizer, its TableFilter, tables). tables: name -> array of everything that gets a
    weights/<name>.hex: the arrays of model.npz (int8), then TOKEN, ALLOW, LEX, WORD of the syntax filter and,
    for a temperature > 0, GUMBEL (bytes, uint8)."""
    spec, arrays = gen.load(model_dir)
    tok = Tok(os.path.join(spec["tokenizer"], "tok.model"))
    if tok.vocab_size != spec["config"]["vocab_size"]:
        raise ValueError("the tokenizer %s does not belong to this model" % spec["tokenizer"])
    tf = TableFilter(tok, triple_quotes=TRIPLE_QUOTES)
    tables = dict(arrays)
    for name, data, row in (("TOKEN", tf.token_table(), tok.vocab_size), ("ALLOW", tf.allow_table(), ROW),
                            ("LEX", tf.lexer_table(), ROW), ("WORD", tf.word_table(), WORD_ROW)):
        tables[name] = np.frombuffer(data, dtype=np.uint8).reshape(-1, row)
    if temperature > 0:
        tables["GUMBEL"] = gumbel_table(temperature, spec["frac"]["u"]).astype(np.uint8)
    return spec, arrays, tok, tf, tables


def small(tables):
    """The tables of the VHDL package: everything but the weight matrices E and W*_l."""
    return {name: value for name, value in tables.items() if not re.fullmatch(r"E|W\w_\d+", name)}


def numbers(spec):
    """The integer constants of the VHDL package: the shape, then the shifts and preloads of model.json."""
    shape = {name: value for name, value in spec["config"].items() if type(value) is int}
    return {**shape, **spec["shifts"], **spec["preloads"]}


def build(spec, arrays, tf, ids, syntax_filter, temperature, prompts=None, uart_prompt=False):
    """The Build that is exported and checked: the complete program of lsllm.ls.gen (prompt buffer, stop rule;
    syntax filter and sampling as asked) in one of three forms. Without the last two arguments the prompt `ids`
    is already in its image. prompts: the token ids of up to 64 prompts, in a table from which the switches
    choose. uart_prompt: the program reads its prompts from the UART (nothing about a prompt in the image)."""
    b = gen.build(spec, arrays, prompt=True, tf=tf, syntax_filter=syntax_filter, temperature=temperature,
                  prompts=prompts, uart_prompt=uart_prompt)
    if prompts is None and not uart_prompt:
        gen.set_prompt(b.image, b, ids)
    return b


def form_of(prompts, uart_prompt):
    """Name of the form of the program: "table" (prompt chosen by the switches), "uart" or "single"."""
    return "table" if prompts is not None else "uart" if uart_prompt else "single"
