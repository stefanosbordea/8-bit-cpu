"""Karpathy's published stories260K checkpoint (trained on TinyStories, not on code) behind the Completer
interface, so that the published model of our size goes through the same benchmark harness."""
import importlib.util

import sentencepiece as spm
import torch

HERE = "third_party/stories260K/"


class Stories260K:
    name = "stories260K (karpathy/tinyllamas, as published)"

    def __init__(self):
        spec = importlib.util.spec_from_file_location("llama2c_model", "third_party/llama2c/model.py")
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        saved = torch.load(HERE + "stories260K.pt", map_location="cpu")
        self.model = upstream.Transformer(upstream.ModelArgs(**saved["model_args"]))
        self.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in saved["model"].items()})
        self.model.eval()
        self.sp = spm.SentencePieceProcessor(model_file=HERE + "tok512.model")
        self.info = {"params": sum(p.numel() for p in self.model.parameters()), "variant": "float", "filter": "off",
                     "notes": "published checkpoint, TinyStories data, its own tok512 tokenizer"}
        self.truncated = 0

    @torch.no_grad()
    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        ids = [self.sp.bos_id()] + self.sp.encode(prompt)
        limit = self.model.params.max_seq_len - max_new_tokens
        self.truncated += len(ids) > limit
        ids = ids[-limit:]
        rng = torch.Generator().manual_seed(seed)
        outs = []
        for _ in range(n if temperature > 0 else 1):
            seq = torch.tensor([ids])
            for _ in range(max_new_tokens):
                logits = self.model(seq)[:, -1, :]
                if temperature > 0:
                    nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=rng)
                else:
                    nxt = logits.argmax(-1, keepdim=True)
                if int(nxt) == self.sp.eos_id():
                    break
                seq = torch.cat([seq, nxt], 1)
            outs.append(self.sp.decode(seq[0, len(ids):].tolist()))
        return outs * (n if temperature == 0 else 1)
