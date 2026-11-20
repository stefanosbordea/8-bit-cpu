"""TorchCompleter: a trained checkpoint behind the Completer interface of the benchmark harness."""
import os

import torch

from lsllm.decode_aids import prompt_tokens, repeated
from lsllm.model.llama import load
from lsllm.structure import to_markers

LINE_BREAKS = ("<NL>", "<DED")      # markers that end a statement; <IND> can never lead to level 0


class TorchCompleter:
    """Continues Python source with the float model of a checkpoint.

    The model never sees more than max_seq_len tokens: a prompt that is longer loses its beginning (counted in
    `self.truncated`), and while the answer grows the window slides over prompt and answer.
    """

    def __init__(self, ckpt, tok=None, device="cpu", syntax_filter=False):
        """ckpt: path of a ckpt.pt. tok: a lsllm.tok.Tok; by default the tokenizer the model was trained with.
        syntax_filter: only tokens that keep the code well formed may be chosen (lsllm/filter_table.py);
        the value "plain" also forbids opening a triple-quoted string (no docstring of the model's own)."""
        self.model, saved = load(ckpt, device)
        self.model.eval()
        if tok is None:
            from lsllm.tok import Tok
            tok = Tok(os.path.join(saved["tok"], "tok.model"))
        assert tok.vocab_size == self.model.cfg.vocab_size, "the tokenizer does not belong to this checkpoint"
        self.tok, self.device = tok, device
        self.name = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))    # the run folder
        self.info = {"params": sum(p.numel() for p in self.model.parameters()), "variant": "float"}    # for the results table
        self.truncated = 0
        self.filter = None
        if syntax_filter:
            from lsllm.filter_table import TableFilter      # the table form: exactly what the CPU runs (docs/FILTER.md)
            self.filter = TableFilter(tok, triple_quotes=syntax_filter != "plain")
            self.name += "+filter"
        self.info["filter"] = "on, no docstring" if syntax_filter == "plain" else "on" if syntax_filter else "off"
        # decoding aids (lsllm/decode_aids.py), off by default: a bonus in nats for tokens of the prompt, and the
        # largest number of times the same run of `no_repeat` tokens may be written (0 = no limit)
        self.prompt_bias, self.no_repeat, self.max_repeat = 0.0, 0, 2

    @torch.no_grad()
    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        tok, T = self.tok, self.model.cfg.max_seq_len
        ids = tok.encode(prompt, bos=True, eos=False)
        if tok.decode(ids) != prompt:
            raise ValueError("the tokenizer does not reproduce this prompt; is it canonical?")
        budget = max_new_tokens
        context = ids[-(T - 1):]                        # the whole prompt if it fits; the window slides while generating
        self.truncated += len(context) < len(ids)

        rows = n if temperature > 0 else 1              # greedy decoding gives n equal answers: compute one
        rng = torch.Generator().manual_seed(seed)
        seq = torch.tensor([context] * rows, device=self.device)
        answers = [[] for _ in range(rows)]             # new tokens of every sample
        levels = [tok.level_after(ids)] * rows          # indentation level of every sample
        alive = list(range(rows))                       # samples that still generate; seq holds only their rows
        states = [self.filter.start_ids(ids) for _ in range(rows)] if self.filter else None
        bonus = torch.tensor(prompt_tokens(tok, ids), dtype=torch.float32) * self.prompt_bias if self.prompt_bias else None
        for _ in range(budget):
            logits = self.model(seq[:, -T:])[0][:, -1].cpu()
            if bonus is not None:
                logits = logits + bonus
            if self.no_repeat:
                for row, sample in enumerate(alive):
                    for t in repeated(answers[sample], self.no_repeat, self.max_repeat):
                        logits[row, t] = float("-inf")
            if self.filter:
                for row, sample in enumerate(alive):
                    ok = torch.tensor(self.filter.allowed(states[sample]))
                    if ok.any():                        # if nothing is allowed, leave the choice to the model
                        logits[row][~ok] = float("-inf")
            if temperature > 0:
                chosen = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=rng)[:, 0]
            else:
                chosen = logits.argmax(dim=-1)
            keep = []                                   # rows of seq that go on
            for row, t in enumerate(chosen.tolist()):
                sample = alive[row]
                levels[sample] = tok.level_after([t], levels[sample])
                back_at_top = levels[sample] == 0 and any(m in tok.pieces[t] for m in LINE_BREAKS)
                if t != tok.eos_id and not back_at_top:     # otherwise the sample ends; the token is dropped
                    answers[sample].append(t)
                    keep.append(row)
                    if self.filter:
                        self.filter.step(states[sample], t)
            if not keep:
                break
            alive = [alive[row] for row in keep]
            seq = torch.cat([seq[keep], chosen[keep, None].to(self.device)], dim=1)

        texts = [tok.decode(ids + answer) for answer in answers]
        assert all(text.startswith(prompt) for text in texts)
        return [text[len(prompt):] for text in texts] * (n if rows == 1 else 1)


def filtered(ckpt):
    """TorchCompleter with the syntax filter switched on (registry kind "torchf")."""
    return TorchCompleter(ckpt, syntax_filter=True)


def plain(ckpt):
    """TorchCompleter with the syntax filter and without triple-quoted strings (registry kind "torchp")."""
    return TorchCompleter(ckpt, syntax_filter="plain")
