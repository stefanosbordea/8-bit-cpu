"""Fast-path trainer: one run of the float model on a uint16 token stream."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from lsllm.config import ModelConfig
from lsllm.model.llama import build

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--data", required=True)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--device", default="cpu")
p.add_argument("--threads", type=int, default=2)
p.add_argument("--eval-every", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--init", default="")
p.add_argument("--cap", type=int, default=0)
for name, default in ModelConfig().to_dict().items():
    if isinstance(default, bool):
        p.add_argument("--" + name, type=int, default=int(default))
    else:
        p.add_argument("--" + name, type=type(default), default=default)
a = p.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
torch.set_num_threads(a.threads)
meta = json.load(open(a.data + "/meta.json"))
fields = {k: (bool(v) if isinstance(getattr(ModelConfig(), k), bool) else v) for k, v in vars(a).items() if k in ModelConfig().to_dict()}
fields["vocab_size"] = meta["vocab_size"]
cfg = ModelConfig(**fields)
model = build(cfg)
if a.cap:
    from quick.capattn import install
    install(model, normalise=(a.cap == 1))
model = model.to(a.device)
if a.init:
    model.load_state_dict(torch.load(a.init, map_location=a.device)["model"])
train = np.fromfile(a.data + "/train.bin", dtype=np.uint16).astype(np.int64)
val = np.fromfile(a.data + "/val.bin", dtype=np.uint16).astype(np.int64)
T = cfg.max_seq_len
os.makedirs(a.out, exist_ok=True)


def batch(data, n, rng):
    at = rng.integers(0, len(data) - T - 1, n)
    x = torch.from_numpy(np.stack([data[g:g + T] for g in at]))
    y = torch.from_numpy(np.stack([data[g + 1:g + T + 1] for g in at]))
    return x.to(a.device), y.to(a.device)


val_rng = np.random.default_rng(1234)
val_batches = [batch(val, 64, val_rng) for _ in range(8)]


@torch.no_grad()
def evaluate():
    model.eval()
    loss = sum(model(x, y)[1].item() for x, y in val_batches) / len(val_batches)
    model.train()
    return loss


decay = [q for q in model.parameters() if q.dim() >= 2]
other = [q for q in model.parameters() if q.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.99))
rng = np.random.default_rng(a.seed)
log = open(a.out + "/log.txt", "a")
best, t0 = 1e9, time.time()
bpb = lambda nats: nats * meta["val_tokens"] / (meta["val_bytes"] * math.log(2))
for step in range(1, a.steps + 1):
    lr = a.lr * step / a.warmup if step < a.warmup else a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup))))
    for group in opt.param_groups:
        group["lr"] = lr
    x, y = batch(train, a.batch, rng)
    loss = model(x, y)[1]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % a.eval_every == 0 or step == a.steps:
        v = evaluate()
        line = "step %d train %.4f val %.4f bpb %.4f lr %.2e tok/s %.0f" % (step, loss.item(), v, bpb(v), lr, step * a.batch * T / (time.time() - t0))
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        if v < best:
            best = v
            torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step, "val_loss": v, "tok": a.data}, a.out + "/ckpt.pt")
json.dump({"name": os.path.basename(a.out), "val_loss": best, "val_bpb": bpb(best), "steps": a.steps, "tokens_seen": a.steps * a.batch * T, "params": cfg.n_params(),
           "minutes": (time.time() - t0) / 60, "device": a.device, "cap": a.cap, "config": cfg.to_dict()}, open(a.out + "/result.json", "w"), indent=1)
