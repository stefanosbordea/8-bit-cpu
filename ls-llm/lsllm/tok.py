"""Tokenizer: canonical source <-> token ids, a SentencePiece model on marker text (lsllm/structure.py).

Every token becomes text knowing only the current indentation level:
    <NL>, <IND>, <DED1>..<DED5>   line break, then the indentation of the new level
    <LF>                          line break, then the indentation (a docstring line with text follows)
    <BLANK>                       line break only (an empty docstring line follows)
    any other token               its own characters
<BLANK> is how the tokenizer writes an <LF> marker that is followed by another <LF>. Because of it the level is
the whole state of the decoder, so a token sequence can be cut anywhere and decoded in parts.
"""
import sentencepiece as spm

from lsllm.structure import MARKERS, STEP, to_markers

BLANK = "<BLANK>"


class Tok:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.vocab_size = self.sp.get_piece_size()
        self.unk_id, self.bos_id, self.eos_id = self.sp.unk_id(), self.sp.bos_id(), self.sp.eos_id()
        self.nl_id, self.ind_id, *self.ded_ids, self.lf_id = [self.sp.piece_to_id(m) for m in MARKERS]
        self.blank_id = self.sp.piece_to_id(BLANK)
        if self.unk_id in (self.blank_id, self.lf_id):
            raise ValueError("%s has no <BLANK> token: it was not made by lsllm.data.tokenize_corpus" % model_file)
        # marker-text string of every token id; the control tokens stand for no text
        self.pieces = [self.sp.id_to_piece(t).replace("\u2581", " ") for t in range(self.vocab_size)]
        for t in (self.unk_id, self.bos_id, self.eos_id):
            self.pieces[t] = ""
        self.pieces[self.blank_id] = "<LF>"
        # level change of every token that is a line break followed by indentation
        self._step = {self.sp.piece_to_id(m): STEP.get(m, 0) for m in MARKERS}

    def encode(self, text, bos=True, eos=False):
        """Token ids of canonical source `text`. Raises ValueError if the text cannot be reproduced exactly."""
        try:
            marked = to_markers(text)
        except ValueError:
            if "\n" in text:
                raise
            marked = text           # one unfinished line, such as `def bubble_sort(`: no line structure to mark
        ids = self.sp.encode(marked)
        # an <LF> that is followed by another <LF> stands in front of an empty line: it becomes <BLANK>
        ids = [self.blank_id if t == nxt == self.lf_id else t for t, nxt in zip(ids, ids[1:] + [None])]
        if self.decode(ids) != text:
            raise ValueError("text has a character outside printable ASCII")
        if bos:
            ids.insert(0, self.bos_id)
        if eos:
            ids.append(self.eos_id)
        return ids

    def encode_request(self, prompt, text, eos=True):
        """Token ids of `text` as the board sees it after the request `prompt` (an unfinished first line such as
        `def hello_world(`): the ids of the prompt, then the ids of the rest. These are the usual ids of the text
        except where a token would span the end of the prompt (`()` after `def hello_world(` becomes `(` and `)`)."""
        ids, head = self.encode(text, bos=True, eos=eos), self.encode(prompt, bos=True)
        if ids[:len(head)] == head:
            return ids
        marked = to_markers(text)
        if not marked.startswith(prompt):
            raise ValueError("the text does not begin with the prompt")
        rest = self.sp.encode(marked[len(prompt):])
        rest = [self.blank_id if t == nxt == self.lf_id else t for t, nxt in zip(rest, rest[1:] + [None])]
        if self.decode(head[1:] + rest) != text:
            raise ValueError("the text cannot be split at the end of the prompt")
        return head + rest + ([self.eos_id] if eos else [])

    def decode(self, ids, level=0):
        """Text of the ids. `level` is the indentation level before the first id."""
        out = []
        for t in map(int, ids):
            if t in self._step:
                level = max(0, level + self._step[t])
                out.append("\n" + "    " * level)
            elif t == self.blank_id:
                out.append("\n")
            else:
                out.append(self.pieces[t])
        return "".join(out)

    def level_after(self, ids, level=0):
        """Indentation level after the ids, starting from `level`."""
        for t in map(int, ids):
            level = max(0, level + self._step.get(t, 0))
        return level
