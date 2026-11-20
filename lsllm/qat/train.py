"""Quantization-aware fine-tuning: calibrate a float checkpoint, train the integer model, export it.

    .venv/bin/python -m lsllm.qat.train --float runs/NAME/ckpt.pt --data data/tok/NAME --out runs/NAME_int --steps 2000

Writes into --out the integer model with the lowest validation loss (model.json and model.npz, SPEC Part E),
log.txt and result.json. Every loss of the integer model is the cross-entropy of its integer logits, so the
validation loss is the loss of the model on the FPGA. The optimiser is the one of lsllm.train.run.
"""
import argparse
import json
import math
import os
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from lsllm.model.llama import load
from lsllm.qat.calibrate import calibrate
from lsllm.qat.export import export
from lsllm.train.run import add_recall_args, evaluate, evaluate_parts, lr_at, prompt_flags, weighted_loss, windows


class RealLogits(torch.nn.Module):
    """The integer model seen as a model with logits in nats (u / 2^f_u), for the loss functions of lsllm.train.run."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, targets=None):
        return self.model(tokens)[0] * 2.0 ** -self.model.frac["u"], None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--float", dest="ckpt", required=True, help="checkpoint of the trained float model")
    p.add_argument("--data", required=True, help="tokenizer directory with train.bin, val.bin and meta.json")
    p.add_argument("--out", required=True, help="directory for the integer model, log.txt and result.json")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--kd-weight", type=float, default=0.0, help="share of the KL term to the float model in the loss")
    p.add_argument("--lr", type=float, default=0.02,
                   help="peak learning rate in steps of each tensor: an update moves a weight by about this share of "
                        "its own quantisation step (the optimiser is Adam, so the size of an update does not depend on "
                        "the size of the gradient). One rate in real units for all tensors made the fine tensors jump "
                        "by more than a step per update and the run diverged (D-qat-6)")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--batch-size", type=int, default=16,
                   help="memory on cpu at context 256 (5 layers, 8 heads): peak 0.48 GB at batch 2 and 0.56 GB "
                        "at batch 4, measured; by the same slope about 1.1 GB at 16 and 1.8 GB at 32, not measured")
    p.add_argument("--threads", type=int, default=2, help="CPU threads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250, help="steps between evaluations")
    p.add_argument("--eval-windows", type=int, default=512, help="size of the fixed validation set, in windows")
    p.add_argument("--calib-windows", type=int, default=16, help="training windows used to choose the scales")
    p.add_argument("--prompt-weight", type=float, default=1.0,
                   help="loss weight of the tokens a benchmark prompt already contains, as in lsllm.train.run; below 1 "
                        "the integer model with the lowest loss on the body tokens is kept")
    p.add_argument("--logit-bits", type=int, help="2^this many codes per nat for the logits, if finer than calibration chooses "
                                                  "(lsllm.qat.calibrate); for models that know texts by heart")
    p.add_argument("--stream-bits", type=int, help="fractional bits of the residual stream, if more than calibration chooses "
                                                   "(lsllm.qat.calibrate); for a float model with one very large dimension")
    p.add_argument("--init-int", help="continue from the integer model in this folder (exported by an earlier run with the same "
                                      "float checkpoint, data and seed, so that calibration gives the same scales)")
    add_recall_args(p)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

    teacher, ckpt = load(args.ckpt)
    teacher.eval()
    T = teacher.cfg.max_seq_len
    calib = windows(train, rng.integers(0, len(train) - T, size=args.calib_windows), T)[:, :-1]
    model = calibrate(teacher, [calib], logit_bits=args.logit_bits, stream_bits=args.stream_bits)
    model.tokenizer = ckpt.get("tok", args.data)
    if args.init_int:                   # the weights of an earlier run instead of the rounded float weights
        with open(os.path.join(args.init_int, "model.json")) as f:
            earlier = json.load(f)
        assert earlier["frac"] == model.frac and earlier["shifts"] == model.shifts, "other scales: not made from this float checkpoint"
        arrays = np.load(os.path.join(args.init_int, "model.npz"))
        with torch.no_grad():
            for key, weights in model.latent.items():
                weights.copy_(torch.from_numpy(arrays[key].astype(np.float32)) / model.scale(key))
    # the validation set of lsllm.train.run: the same evenly spaced windows for every run with this context length
    n_val = max(1, min(args.eval_windows, len(val) // (T + 1)))
    val_starts = np.linspace(0, len(val) - T - 1, n_val).astype(np.int64)
    val_rows = windows(val, val_starts, T)
    weighted = args.prompt_weight != 1.0
    if weighted:
        from lsllm.tok import Tok
        tok = Tok(os.path.join(args.data, "tok.model"))
        train_flags, val_flags = prompt_flags(train, tok), windows(prompt_flags(val, tok), val_starts, T)

    recall = None
    if args.recall:
        from lsllm.tok import Tok
        from lsllm.train.recall import Recall
        recall = Recall(args.recall.split(","), Tok(os.path.join(args.data, "tok.model")), T)

    def score(net, rows, flags):
        """The loss that decides which integer model is kept: all tokens, or the body tokens alone."""
        if not weighted:
            return evaluate(net, rows, 8)
        return evaluate_parts(RealLogits(net) if net is model else net, rows, flags, 8)[1]

    def bits_per_byte(nats):
        return nats * meta["val_tokens"] / (meta["val_bytes"] * math.log(2))

    log = open(os.path.join(args.out, "log.txt"), "w")

    def say(text):
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    float_loss = evaluate(teacher, val_rows, 8)
    teacher.eval()
    best = start_loss = evaluate(model, val_rows, 8)
    if weighted:
        float_body = evaluate_parts(teacher, val_rows, val_flags, 8)[1]
        teacher.eval()
        best = start_body = score(model, val_rows, val_flags)
    best_exact = -1                 # texts reproduced by the kept model (--recall)
    best_step, state = 0, {name: value.clone() for name, value in model.state_dict().items()}
    export(model, args.out)
    say(f"float val {float_loss:.4f}, calibrated integer val {start_loss:.4f} (nats per token, {n_val} windows); "
        f"{args.steps} steps of {args.batch_size} x {T} tokens on {args.device}, kd weight {args.kd_weight}")
    if weighted:
        say(f"body tokens only: float {float_body:.4f}, calibrated integer {start_body:.4f}; prompt weight {args.prompt_weight}")
    for note in model.notes:
        say("calibration note: " + note)
    if recall is not None:              # the starting model is already exported: a later one must reproduce more texts, or lose less
        model.eval()
        best_exact = int((recall.check(lambda x: model(x)[0], "cpu") > 0).sum())
        model.train()
        say(f"recall at the start: {best_exact} of {len(recall.names)} texts")

    model.to(args.device)
    teacher.to(args.device)
    val_device = val_rows.to(args.device)
    val_flags_device = val_flags.to(args.device) if weighted else None
    # one group per tensor: its learning rate is args.lr quantisation steps of that tensor (1 / 2^f in real units)
    opt = torch.optim.AdamW([{"params": [weights], "step": 1.0 / model.scale(key)} for key, weights in model.latent.items()],
                            lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    warmup = min(100, args.steps // 20)
    running, since, t_start = 0.0, 0, time.time()
    model.train()
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, warmup) * group["step"]
        starts = rng.integers(0, len(train) - T, size=args.batch_size)
        rows = windows(train, starts, T).to(args.device)
        x, y = rows[:, :-1], rows[:, 1:]
        u, loss = model(x, y)
        weight = None
        if weighted:                    # the prompt tokens count less, in the cross-entropy and in the distillation term
            flags = windows(train_flags, starts, T).to(args.device)[:, 1:]
            weight = 1.0 + (args.prompt_weight - 1.0) * flags.to(u.dtype)
            loss = weighted_loss(u * 2.0 ** -model.frac["u"], y, flags, args.prompt_weight)
        if args.kd_weight:              # match the next-token distribution of the float model as well
            with torch.no_grad():
                soft = F.softmax(teacher(x)[0], dim=-1)
            each = F.kl_div(F.log_softmax(u * 2.0 ** -model.frac["u"], dim=-1), soft, reduction="none").sum(-1)
            kd = each.sum() / y.numel() if weight is None else (each * weight).sum() / weight.sum()
            loss = (1 - args.kd_weight) * loss + args.kd_weight * kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if recall is not None:              # the request texts, each from the first token of the context
            rows, mask = (part.to(args.device) for part in recall.batch(args.recall_rows, rng, args.recall_boost))
            (args.recall_weight * recall.loss(model(rows[:, :-1])[0] * 2.0 ** -model.frac["u"], rows, mask)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.clip()
        running += loss.item()
        since += 1
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            assert math.isfinite(running), "the training loss is not finite"
            val_loss = score(model, val_device, val_flags_device)
            better = val_loss < best
            if recall is not None:          # a board model is chosen by the number of texts it reproduces, then by the loss
                model.eval()
                margins = recall.check(lambda x: model(x)[0], args.device)
                model.train()
                exact = int((margins > 0).sum())
                say(f"recall {exact} of {len(margins)} texts, smallest margin {margins.min():.0f} codes, median {np.median(margins):.0f}")
                better = exact > best_exact or (exact == best_exact and val_loss < best)
                if better:
                    best_exact = exact
            if better:
                best, best_step = val_loss, step + 1
                state = {name: value.clone() for name, value in model.state_dict().items()}
                export(model, args.out)
            say(f"step {step + 1} train {running / since:.4f} {'body' if weighted else 'val'} {val_loss:.4f} bpb {bits_per_byte(val_loss):.4f} "
                f"lr {lr_at(step, args.steps, args.lr, warmup):.2e} steps time {time.time() - t_start:.0f}")
            running, since = 0.0, 0
    wall = time.time() - t_start

    # the numbers of record come from the exported files and from the exact forward pass on the CPU
    model.load_state_dict(state)
    model.to("cpu").eval()
    final = evaluate(model, val_rows, 8)
    with torch.no_grad():
        rows = val_rows[:2, :-1]
        same = np.array_equal(model(rows)[0].numpy(), export(model, args.out).forward(rows.numpy()))
    seen = args.steps * args.batch_size * T
    body = {}
    if weighted:
        body = {"val_body": evaluate_parts(RealLogits(model), val_rows, val_flags, 8)[1], "float_val_body": float_body,
                "calibrated_val_body": start_body, "prompt_weight": args.prompt_weight}
        say(f"body tokens only: integer {body['val_body']:.4f} (float {float_body:.4f}, after calibration {start_body:.4f})")
    result = {"val_loss": final, "val_bpb": bits_per_byte(final), "float_val_loss": float_loss, **body,
              "calibrated_val_loss": start_loss, "best_step": best_step, "steps": args.steps, "tokens_seen": seen,
              "wall_s": round(wall, 1), "tokens_per_s": round(seen / max(wall, 1e-9)), "device": args.device,
              "bit_exact": bool(same), "val_loss_on_device": best, "params": model.cfg.n_params(),
              **({"recall_exact": best_exact, "recall_texts": len(recall.names)} if recall is not None else {}),
              "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20),
              "config": model.cfg.to_dict(), "options": model.options, "args": vars(args)}
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    say(f"done: integer val {final:.4f} nats/token = {bits_per_byte(final):.4f} bits/byte at step {best_step} "
        f"(float {float_loss:.4f}); QAT forward and integer reference identical: {same}")
    log.close()
    return result


if __name__ == "__main__":
    main()
