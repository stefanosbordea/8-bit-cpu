"""IntCompleter: an exported integer model behind the Completer interface of the benchmark harness.

The harness knows it as "int:<directory>" and, with the syntax filter, "intf:<directory>". A continuation ends where it ends in
lsllm.model.completer.TorchCompleter; the model is the integer reference, and sampling is the Gumbel-max sampling
of the CPU (SPEC Part D), all samples in one batch.

The context rule is NOT the one of TorchCompleter. The CPU has a KV cache of max_seq_len positions and cannot
slide it, so prompt and answer must fit into one context: max_new_tokens positions are reserved for the answer
and the prompt keeps its last max_seq_len - max_new_tokens tokens (counted in `truncated` if it had more).
TorchCompleter keeps max_seq_len - 1 tokens of the prompt and slides its window. At max_seq_len 256 and
max_new_tokens 128 the integer model therefore sees at most 128 prompt tokens and the float model 255: on
prompts longer than 128 tokens (129 of the 164 HumanEval prompts with data/tok/f512a; none of MBPP practice or
custom practice) the two scores are not computed on the same context. The results row says so in its notes.
"""
import json
import os

import numpy as np

from lsllm.intref.model import IntModel, lfsr_start
from lsllm.structure import to_markers

LINE_BREAKS = ("<NL>", "<DED")      # markers that end a statement; <IND> can never lead to level 0


class IntCompleter:
    def __init__(self, directory, tok=None, syntax_filter=False):
        """directory: an exported integer model (SPEC Part E). tok: by default the tokenizer named in model.json.
        syntax_filter: only tokens that keep the code well formed may be chosen (lsllm/filter_table.py);
        the value "plain" also forbids opening a triple-quoted string (no docstring of the model's own)."""
        self.model = IntModel.load(directory)
        if tok is None:
            from lsllm.tok import Tok
            tok = Tok(os.path.join(self.model.spec["tokenizer"], "tok.model"))
        assert tok.vocab_size == self.model.cfg.vocab_size, "the tokenizer does not belong to this model"
        self.tok = tok
        self.name = os.path.basename(os.path.abspath(directory))
        self.info = {"params": self.model.cfg.n_params(), "variant": "int8",      # for the results table
                     "notes": "no sliding window: prompt cut to max_seq_len - max_new_tokens tokens"}
        budget = os.path.join(directory, "budget.json")
        if os.path.exists(budget):                      # an editing session (lsllm/edit): the row names the model it started from
            with open(budget) as f:
                self.info["edit"] = "session " + self.name
                self.name = os.path.basename(os.path.abspath(json.load(f)["base"]))
        self.truncated = 0
        self.filter = None
        if syntax_filter:
            from lsllm.filter_table import TableFilter
            self.filter = TableFilter(tok, triple_quotes=syntax_filter != "plain")
            self.name += "+filter"
        self.info["filter"] = "on, no docstring" if syntax_filter == "plain" else "on" if syntax_filter else "off"

    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        tok, model, T = self.tok, self.model, self.model.cfg.max_seq_len
        ids = tok.encode(prompt, bos=True, eos=False)
        if tok.decode(ids) != prompt:
            raise ValueError("the tokenizer does not reproduce this prompt; is it canonical?")
        budget = min(max_new_tokens, T - 1)
        context = ids[-(T - budget):]
        self.truncated += len(context) < len(ids)

        rows = n if temperature > 0 else 1              # greedy decoding gives n equal answers: compute one
        model.reset()
        for t in context:                               # one sequence reads the prompt ...
            u = model.step([t])
        model.select([0] * rows)                        # ... then every sample continues its own copy of the cache
        model.lfsr = lfsr_start(seed, rows)
        u = np.repeat(u, rows, axis=0)
        answers = [[] for _ in range(rows)]             # new tokens of every sample
        levels = [tok.level_after(ids)] * rows          # indentation level of every sample
        alive = list(range(rows))                       # samples that still generate; the model holds only their rows
        states = [self.filter.start_ids(ids) for _ in range(rows)] if self.filter else None
        for g in range(budget):
            allowed = np.array([self.filter.allowed(states[sample]) for sample in alive]) if self.filter else None
            chosen = model.choose(u, temperature, allowed)
            keep = []                                   # rows of the batch that go on
            for row, t in enumerate(chosen.tolist()):
                sample = alive[row]
                levels[sample] = tok.level_after([t], levels[sample])
                back_at_top = levels[sample] == 0 and any(m in tok.pieces[t] for m in LINE_BREAKS)
                if t != tok.eos_id and not back_at_top:     # otherwise the sample ends; the token is dropped
                    answers[sample].append(t)
                    keep.append(row)
                    if self.filter:
                        self.filter.step(states[sample], t)
            if not keep or g + 1 == budget:
                break
            alive = [alive[row] for row in keep]
            model.select(keep)
            u = model.step(chosen[keep])

        texts = [tok.decode(ids + answer) for answer in answers]
        assert all(text.startswith(prompt) for text in texts)
        return [text[len(prompt):] for text in texts] * (n if rows == 1 else 1)


def filtered(directory):
    """IntCompleter with the syntax filter switched on (registry kind "intf")."""
    return IntCompleter(directory, syntax_filter=True)


def plain(directory):
    """IntCompleter with the syntax filter and without triple-quoted strings (registry kind "intp")."""
    return IntCompleter(directory, syntax_filter="plain")
