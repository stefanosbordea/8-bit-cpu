"""Small public checkpoints in Hugging Face format behind the Completer interface, so that other people's
models go through the same benchmark harness as ours. The model sees its begin-of-text token and then the prompt
in its own tokenizer's pieces (plain text, no chat template); the decoded continuation comes back.

.venv/bin/python -m lsllm.bench.run --completer public:Pico-1 --bench custom --max-new-tokens 96 --record
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.environ.get("LSLLM_PUBLIC_MODELS", "/Volumes/CARMIX-WORK/ls-llm-work/models/")

# folder under ROOT -> (hub repository, commit, licence on the card, parameters on the card, what it is)
MODELS = {
    "Pico-1": ("Convergent-ch/Pico-1", "e8db8c3", "Apache-2.0", 254400,
               "chat model, not trained on code; byte-level BPE of 2,048 pieces"),
    "Pico-1.2": ("Convergent-ch/Pico-1.2", "ba10bfa", "Apache-2.0", 460608,
                 "chat model, not trained on code; byte-level BPE of 2,048 pieces"),
    "CinnabarLM-4M-Python-Base": ("MihaiPopa-1/CinnabarLM-4M-Python-Base", "4702bd2", "Apache-2.0", 4229568,
                                  "trained on Python code (starcoder-python-instruct); BPE of 4,096 pieces"),
    "Pico-1.3-Coding-Large": ("Convergent-ch/Pico-1.3-Coding-Large", "c694b11", "Apache-2.0", 6917376,
                              "chat model trained on chats and simple Python tasks; byte-level BPE of 8,192 "
                              "pieces; its card says it needs its chat template, which is not used here"),
    "chispa-20m-python": ("esraderey/chispa-20m-python", "0dd42d1", "MIT", 20119360,
                          "Python line-completion model (codeparrot-clean); byte-level BPE of 16,384 pieces"),
}


class PublicModel:
    def __init__(self, folder):
        repo, commit, licence, card_params, what = MODELS[folder]
        torch.set_num_threads(2)
        self.tok = AutoTokenizer.from_pretrained(ROOT + folder)
        self.model = AutoModelForCausalLM.from_pretrained(ROOT + folder, dtype=torch.float32).eval()
        self.context = self.model.config.max_position_embeddings
        params = sum(p.numel() for p in self.model.parameters())
        self.name = folder + " (as published, plain continuation)"
        self.info = {"params": params, "variant": "float", "filter": "off",
                     "notes": "huggingface.co/%s at %s, licence %s (card read 2026-11-14); %s; context %d%s"
                              % (repo, commit, licence, what, self.context,
                                 "" if params == card_params else "; the card states %d parameters" % card_params)}
        self.truncated = 0
        # stop tokens: those of the model's generation config and the tokenizer's end of text (they can differ)
        stop = self.model.generation_config.eos_token_id
        self.stop = sorted({self.tok.eos_token_id, *(stop if isinstance(stop, list) else [stop])})

    @torch.no_grad()
    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        # begin-of-text + prompt, as in the stories260K baseline. The tokenizers' own defaults differ (none, or
        # begin and END of text around the prompt, after which a model starts a new document).
        ids = [self.tok.bos_token_id] + self.tok(prompt, add_special_tokens=False).input_ids
        limit = self.context - max_new_tokens
        self.truncated += len(ids) > limit
        ids = ids[-limit:]
        torch.manual_seed(seed)
        sampling = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0, num_return_sequences=n)
        rows = self.model.generate(torch.tensor([ids]), max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                                   pad_token_id=self.tok.pad_token_id, eos_token_id=self.stop,
                                   **(sampling if temperature > 0 else dict(do_sample=False)))
        # decode prompt + continuation and take off the prompt: decoding the new tokens alone can lose a space
        start = len(self._decode(ids))
        texts = [self._decode(row.tolist())[start:] for row in rows]
        return texts * (n if temperature == 0 else 1)

    def _decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
