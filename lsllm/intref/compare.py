"""Float model versus integer model on the same prompts: do they choose the same tokens, how far apart are the logits?

    .venv/bin/python -m lsllm.intref.compare --float runs/NAME/ckpt.pt --int runs/NAME_int --data data/tok/NAME
"""
import argparse
import json
import os

import numpy as np
import torch

from lsllm.intref.model import IntModel


@torch.no_grad()
def compare(float_model, int_model, prompts, max_new_tokens=32):
    """prompts: lists of token ids. The float model continues every prompt greedily and the integer model reads
    the same tokens, so both always see the same context. Returns a dictionary:
      same       share of the new positions where both models choose the same token
      identical  number of prompts without any difference
      rows       per prompt: {"same": share, "first": index of the first new token that differs, or None}
      largest    the five largest logit differences: prompt, index of the new token, token, float logit, integer logit
    Logits are compared in nats over the tokens whose integer logit is not at a limit (0 or 127), after removing
    the constant by which the two models differ at that position (their median difference)."""
    rows, largest = [], []
    for number, prompt in enumerate(prompts):
        seq = torch.tensor([list(prompt)])
        for g in range(max_new_tokens):
            seq = torch.cat([seq, float_model(seq)[0][:, -1].argmax(-1, keepdim=True)], 1)
        new = slice(len(prompt) - 1, None)                      # the positions that predict the new tokens
        z = float_model(seq[:, :-1])[0][0, new].double().numpy()
        u = int_model.forward(seq[:, :-1].numpy())[0, new]
        same = z.argmax(-1) == u.argmax(-1)
        rows.append({"same": float(same.mean()), "first": None if same.all() else int(np.argmin(same))})
        for g in range(max_new_tokens):
            free = (u[g] > 0) & (u[g] < 127)
            if free.any():
                diff = u[g] * 2.0 ** -int_model.f_l - z[g]
                diff = np.where(free, diff - np.median(diff[free]), 0.0)
                t = int(np.abs(diff).argmax())
                largest.append({"prompt": number, "new_token": g, "token": t, "float": float(z[g, t]),
                                "integer": float(z[g, t] + diff[t]), "difference": float(diff[t])})
    largest.sort(key=lambda row: -abs(row["difference"]))
    return {"same": float(np.mean([row["same"] for row in rows])),
            "identical": sum(row["first"] is None for row in rows), "rows": rows, "largest": largest[:5]}


def main():
    from lsllm.model.llama import load
    p = argparse.ArgumentParser(description="Compare a float checkpoint with an exported integer model.")
    p.add_argument("--float", dest="ckpt", required=True)
    p.add_argument("--int", dest="directory", required=True)
    p.add_argument("--data", required=True, help="tokenizer directory; prompts are the starts of documents of val.bin")
    p.add_argument("--prompts", type=int, default=50)
    p.add_argument("--prompt-tokens", type=int, default=24)
    p.add_argument("--new-tokens", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(2)
    float_model = load(args.ckpt)[0].eval()
    int_model = IntModel.load(args.directory)
    val = np.fromfile(os.path.join(args.data, "val.bin"), dtype=np.uint16).astype(np.int64)
    starts = np.flatnonzero(val == val[0])                      # every document begins with the BOS token
    prompts = [val[s:s + args.prompt_tokens] for s in starts[:args.prompts] if s + args.prompt_tokens <= len(val)]
    result = compare(float_model, int_model, prompts, args.new_tokens)
    print("%d prompts of %d tokens, %d new tokens each" % (len(prompts), args.prompt_tokens, args.new_tokens))
    print("same greedy token at %.1f%% of the new positions; %d prompts identical"
          % (100 * result["same"], result["identical"]))
    firsts = [row["first"] for row in result["rows"] if row["first"] is not None]
    if firsts:
        print("first difference: median new token %d, earliest %d" % (np.median(firsts), min(firsts)))
    print("largest logit differences (nats):")
    for row in result["largest"]:
        print("  " + json.dumps(row))


if __name__ == "__main__":
    main()
