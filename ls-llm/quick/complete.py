"""Fast-path completer: a trained checkpoint + the quick tokenizer behind the lsllm.api.Completer interface."""
import torch
import sentencepiece as spm

from lsllm.config import ModelConfig
from lsllm.model.llama import build
from quick.filter import Filter, State
from quick.markers import DED, IND, from_markers, level_after, to_markers


class QuickCompleter:
    def __init__(self, ckpt, device="cpu", cap=None, use_filter=False):
        ck = torch.load(ckpt, map_location=device)
        self.cfg = ModelConfig(**ck["config"])
        self.model = build(self.cfg)
        if cap is None:
            cap = 1 if any("cap_b" in k for k in ck["model"]) else 0
        if cap:
            from quick.capattn import install
            install(self.model, normalise=(cap == 1))
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.sp = spm.SentencePieceProcessor(model_file=ck["tok"] + "/tok.model")
        self.device = device
        self.name = ckpt
        self.truncated = 0
        self.use_filter = use_filter
        pieces = [self.sp.id_to_piece(g).replace("\u2581", " ") for g in range(self.sp.get_piece_size())]
        pieces[0] = pieces[1] = None
        self.filter = Filter(pieces, eos_id=2)
        self.pieces = pieces
        self.delta = {}
        for g in range(self.sp.get_piece_size()):
            piece = self.sp.id_to_piece(g)
            self.delta[g] = 1 if piece == IND else -(DED.index(piece) + 1) if piece in DED else 0

    @torch.no_grad()
    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        marked = to_markers(prompt)
        ids = [1] + self.sp.encode(marked)
        start_level = level_after(marked)
        T = self.cfg.max_seq_len
        if len(ids) > T - 8:
            self.truncated += 1
        gen = torch.Generator(device="cpu").manual_seed(seed)
        rows = 1 if temperature == 0 else n
        x = torch.tensor([ids] * rows, device=self.device)
        level = [start_level] * rows
        done = [False] * rows
        new = [[] for _ in range(rows)]
        states = []
        for g in range(rows):
            st = State()
            st.feed(marked)
            states.append(st)
        for _ in range(max_new_tokens):
            logits = self.model(x[:, -T:])[0][:, -1, :].float().cpu()
            if self.use_filter:
                for g in range(rows):
                    ok = torch.tensor(self.filter.allowed(states[g]))
                    if ok.any():
                        logits[g][~ok] = float("-inf")
            if temperature == 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).squeeze(1)
            for g in range(rows):
                if done[g]:
                    continue
                tok = int(nxt[g])
                level[g] += self.delta[tok]
                if tok == 2 or (self.delta[tok] < 0 and level[g] <= 0):
                    done[g] = True
                else:
                    new[g].append(tok)
                    if self.pieces[tok] is not None:
                        states[g].feed(self.pieces[tok])
            if all(done):
                break
            x = torch.cat([x, nxt.to(self.device)[:, None]], 1)
        texts = [from_markers(self.sp.decode(row), level=start_level) for row in new]
        return texts * n if temperature == 0 else texts
