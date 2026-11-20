"""Teacher-forced loss of float checkpoints on the reference solutions of the PRACTICE tasks: how many nats per
token the model needs to write the right body after the prompt. A smooth stand-in for pass@1 when pass@1 is 0.
Practice tasks only (custom practice, MBPP practice); never the test sets.

    PYTHONPATH=. .venv/bin/python tools/practice_loss.py runs/a/ckpt.pt runs/b/ckpt.pt ...
"""
import os
import sys

import torch
import torch.nn.functional as F

from lsllm.bench import load_tasks
from lsllm.model.llama import load
from lsllm.tok import Tok


def losses(ckpt):
    model, saved = load(ckpt, "cpu")
    model.eval()
    tok, T, out = Tok(os.path.join(saved["tok"], "tok.model")), model.cfg.max_seq_len, {}
    for bench in ("custom", "mbpp"):
        total, count, first = 0.0, 0, 0.0
        for task in load_tasks(bench, "practice"):
            try:
                prompt = tok.encode(task.prompt, bos=True)
                full = tok.encode(task.canonical_solution, bos=True, eos=True)
            except ValueError:
                continue
            if full[:len(prompt)] != prompt or len(full) > T:
                continue
            ids = torch.tensor([full])
            with torch.no_grad():
                logits = model(ids[:, :-1])[0][0]
            each = F.cross_entropy(logits, ids[0, 1:], reduction="none")[len(prompt) - 1:]
            total, count, first = total + each.sum().item(), count + len(each), first + 1
        out[bench] = (total / count, total / first, int(first))
    return out


if __name__ == "__main__":
    print("%-44s %s" % ("checkpoint", "custom practice: nats/token, nats/solution (tasks) | MBPP practice: nats/token, nats/solution (tasks)"))
    for ckpt in sys.argv[1:]:
        got = losses(ckpt)
        print("%-44s %.3f %6.1f (%d) | %.3f %6.1f (%d)" % ((ckpt.replace("/ckpt.pt", ""),) + got["custom"] + got["mbpp"]), flush=True)
