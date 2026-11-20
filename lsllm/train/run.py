"""One training run of the float model.

    .venv/bin/python -m lsllm.train.run --out runs/NAME --data data/tok/NAME [--dim 64 ...] [--steps N] ...

Reads train.bin, val.bin and meta.json from --data, trains on random windows of max_seq_len + 1 tokens and
writes ckpt.pt (the weights with the lowest validation loss), log.txt and result.json into --out.
The optimiser follows the stories260K recipe of llama2.c (third_party/llama2c/train.py): AdamW with
betas (0.9, 0.99), weight decay on matrices only, linear warm-up, cosine decay, gradient clipping.
"""
import argparse
import json
import math
import os
import resource
import time
from dataclasses import fields

import numpy as np
import torch
import torch.nn.functional as F

from lsllm.config import ModelConfig
from lsllm.model import build, load


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--data", required=True, help="tokenizer directory with train.bin, val.bin and meta.json")
    for f in fields(ModelConfig):                   # one flag per shape field, for example --n-layers 5
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:                                       # the vocabulary size comes from the data unless given
            p.add_argument(flag, type=f.type, default=None if f.name == "vocab_size" else f.default)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--warmup", type=int, help="warm-up steps (default: 5%% of --steps, at most 1000)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0, help="largest gradient norm; 0 switches clipping off")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--chunks", type=int, default=1, help="process every batch in this many parts (same update, less memory)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads; 0 keeps PyTorch's default")
    p.add_argument("--eval-every", type=int, default=250, help="steps between evaluations")
    p.add_argument("--eval-windows", type=int, default=512, help="size of the fixed validation set, in windows")
    p.add_argument("--prompt-weight", type=float, default=1.0,
                   help="loss weight of the tokens a benchmark prompt already contains (signature and docstring); "
                        "below 1 the model spends its capacity on the function body")
    p.add_argument("--init-from", help="checkpoint to continue from; its shape replaces the shape flags")
    p.add_argument("--teacher", help="checkpoint of a larger model with the same tokenizer to distil from")
    p.add_argument("--kd-weight", type=float, default=0.5, help="share of the distillation term in the loss")
    p.add_argument("--kd-temp", type=float, default=2.0, help="temperature of the distillation term")
    p.add_argument("--stream-limit", type=float, default=0.0,
                   help="keep the residual stream inside +-this value: the loss gets the mean of max(|x| - limit, 0)^2 over all "
                        "norm inputs. For a model that is to be quantised: int8 holds one scale for the whole stream, and a "
                        "single very large dimension costs every other dimension its precision (D-qat-7)")
    p.add_argument("--stream-weight", type=float, default=1.0, help="weight of that term")
    add_recall_args(p)
    return p.parse_args(argv)


def add_recall_args(p):
    """Options of a board model (lsllm/train/recall.py): request texts as aligned rows beside the usual windows."""
    p.add_argument("--recall", help="JSONL files with the request texts (name, text), separated by commas; the run then "
                                    "keeps the weights that reproduce the most texts. Board models only")
    p.add_argument("--recall-rows", type=int, default=32, help="request rows per step")
    p.add_argument("--recall-boost", type=float, default=3.0, help="a text that is not reproduced yet is this much more likely to be drawn")
    p.add_argument("--recall-weight", type=float, default=1.0, help="weight of the loss of the request rows")


def lr_at(step, steps, peak, warmup):
    """Linear warm-up to the peak, then a cosine decay that reaches zero after the last step."""
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (steps - warmup)))


def windows(data, starts, T):
    """Rows of T + 1 consecutive tokens that begin at `starts`, as an int64 tensor."""
    return torch.from_numpy(data[starts[:, None] + np.arange(T + 1)].astype(np.int64))


def prompt_flags(data, tok):
    """1 for every token that a benchmark prompt already contains, 0 for the tokens the model has to write.

    data: a token stream of documents, each starting with BOS. In a function with a docstring the prompt part is
    everything up to the closing quotes of the docstring (signature and docstring: what a benchmark prompt gives).
    A function without a docstring stands for a request that gives only the name (`def bubble_sort(`): there the
    model writes the arguments itself, so the prompt part ends at the opening parenthesis."""
    data = np.asarray(data)
    starts = np.flatnonzero(data == tok.bos_id)
    stops = np.append(starts[1:], len(data))                    # where the next document begins
    inds, quotes = np.flatnonzero(data == tok.ind_id), np.flatnonzero(data == tok.sp.piece_to_id('"""'))

    def first_after(places, at):
        """For every position of `at`: the first of `places` at or behind it, len(data) if there is none."""
        k = np.searchsorted(places, at)
        return np.append(places, len(data))[k]

    end = first_after(inds, starts)
    end = np.where(end < stops, end, starts)                    # no indented block: only BOS belongs to the prompt
    has_doc = (end + 1 < stops) & (data[np.minimum(end + 1, len(data) - 1)] == tok.sp.piece_to_id('"""')) & (end > starts)
    closing = first_after(quotes, end + 2)
    opening = first_after(np.flatnonzero(data == tok.sp.piece_to_id("(")), starts)
    named = np.where(opening < stops, opening, starts)                  # no docstring: the prompt is `def name(`
    end = np.where(has_doc & (closing < stops), closing, named)
    step = np.zeros(len(data) + 1, dtype=np.int32)
    np.add.at(step, starts, 1)
    np.add.at(step, end + 1, -1)
    return (np.cumsum(step[:-1]) > 0).astype(np.uint8)


def weighted_loss(logits, targets, flags, prompt_weight):
    """Cross-entropy in which the prompt tokens (flags = 1) count `prompt_weight` times and the others once."""
    each = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    weight = 1.0 + (prompt_weight - 1.0) * flags.reshape(-1).to(each.dtype)
    return (each * weight).sum() / weight.sum()


@torch.no_grad()
def evaluate_parts(model, val, flags, batch_size):
    """(mean loss over all tokens, mean loss over the tokens that are not part of a prompt), nats per token."""
    model.eval()
    total, body, n_body = 0.0, 0.0, 0.0
    for g in range(0, len(val), batch_size):
        rows, part = val[g:g + batch_size], flags[g:g + batch_size, 1:]
        logits = model(rows[:, :-1])[0]
        each = F.cross_entropy(logits.reshape(-1, logits.size(-1)), rows[:, 1:].reshape(-1), reduction="none")
        keep = 1.0 - part.reshape(-1).to(each.dtype)
        total += each.sum().item()
        body += (each * keep).sum().item()
        n_body += keep.sum().item()
    model.train()
    return total / (len(val) * (val.shape[1] - 1)), body / max(n_body, 1.0)


@torch.no_grad()
def evaluate(model, val, batch_size):
    """Mean loss in nats per token over the fixed validation windows."""
    model.eval()
    total = 0.0
    for g in range(0, len(val), batch_size):
        rows = val[g:g + batch_size]
        total += model(rows[:, :-1], rows[:, 1:])[1].item() * len(rows)
    model.train()
    return total / len(val)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

    shape = {f.name: getattr(args, f.name) for f in fields(ModelConfig)}
    shape["vocab_size"] = shape["vocab_size"] or meta["vocab_size"]
    cfg, weights, start = ModelConfig(**shape), None, 0
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        cfg, weights, start = ModelConfig(**ckpt["config"]), ckpt["model"], ckpt["step"]
    T = cfg.max_seq_len
    assert cfg.vocab_size >= meta["vocab_size"], "the model vocabulary is smaller than the data vocabulary"
    assert min(len(train), len(val)) > T + 1, "the data is shorter than one window"
    model = build(cfg, args.dropout, flash=device in ("cpu", "cuda"))     # the fast kernel pays off on the CPU and on CUDA, not on MPS
    if weights:
        model.load_state_dict(weights)
    model.to(device)
    teacher = None
    if args.teacher:
        teacher = load(args.teacher, device)[0].eval()
        assert teacher.cfg.vocab_size == cfg.vocab_size and teacher.cfg.max_seq_len >= T, "teacher does not fit"

    matrices = [p for p in model.parameters() if p.dim() >= 2]      # weight decay on matrices only
    others = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": matrices, "weight_decay": args.weight_decay},
                             {"params": others, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.99))
    warmup = args.warmup if args.warmup is not None else min(1000, args.steps // 20)

    # the validation set: the same evenly spaced windows for every run with this context length
    n_val = max(1, min(args.eval_windows, len(val) // (T + 1)))
    val_starts = np.linspace(0, len(val) - T - 1, n_val).astype(np.int64)
    val_rows = windows(val, val_starts, T).to(device)
    if os.path.exists(os.path.join(args.data, "tok.model")):
        from lsllm.tok import Tok
        tok = Tok(os.path.join(args.data, "tok.model"))
        val_flags = windows(prompt_flags(val, tok), val_starts, T).to(device)
        train_flags = prompt_flags(train, tok) if args.prompt_weight != 1.0 else None
    else:                                           # token files without a tokenizer: every token counts as body
        assert args.prompt_weight == 1.0, "--prompt-weight needs tok.model in the data directory"
        val_flags, train_flags = torch.zeros_like(val_rows), None

    recall = None
    if args.recall:
        from lsllm.train.recall import Recall
        recall = Recall(args.recall.split(","), tok, T)
    stream, collecting = [], [False]                # inputs of every norm during a training pass (--stream-limit)
    if args.stream_limit:
        for site in [m for layer in model.layers for m in (layer.attention_norm, layer.ffn_norm)] + [model.norm]:
            site.register_forward_pre_hook(lambda module, inputs: stream.append(inputs[0]) if collecting[0] else None)
    largest = 0.0

    def bits_per_byte(nats):
        return nats * meta["val_tokens"] / (meta["val_bytes"] * math.log(2))

    last = os.path.join(args.out, "last.pt")        # written at every evaluation; a killed run continues from it
    resume = torch.load(last, map_location="cpu") if os.path.exists(last) else None
    log = open(os.path.join(args.out, "log.txt"), "a" if resume else "w")

    def say(text):
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    n_params = sum(p.numel() for p in model.parameters())
    say(f"{n_params} parameters on {device}: {args.steps} steps of {args.batch_size} x {T} tokens, "
        f"{n_val} validation windows, {json.dumps(cfg.to_dict())}")
    best, best_step, val_loss, kept = float("inf"), 0, float("inf"), None
    best_exact = -1                 # texts reproduced by the kept checkpoint (--recall)
    running, since, train_s = torch.zeros((), device=device), 0, 0.0
    first = 0
    if resume:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        first, best, best_step, train_s = resume["step"], resume["best"], resume["best_step"], resume["train_s"]
        rng = np.random.default_rng(args.seed + first)
        say(f"resumed at step {first}")
    t_start = t0 = time.time()
    model.train()
    for step in range(first, args.steps):
        lr = lr_at(step, args.steps, args.lr, warmup)
        for group in opt.param_groups:
            group["lr"] = lr
        starts = rng.integers(0, len(train) - T, size=args.batch_size)
        rows = windows(train, starts, T).to(device)
        flags = windows(train_flags, starts, T).to(device) if train_flags is not None else None
        opt.zero_grad(set_to_none=True)
        # the batch is processed in `--chunks` equal parts whose gradients add up: the same update as one pass,
        # with a fraction of the memory (the attention tensors are batch x heads x time x time)
        total = 0.0
        for g, part in enumerate(rows.chunk(args.chunks)):
            x, y = part[:, :-1], part[:, 1:]
            stream.clear()
            collecting[0] = bool(args.stream_limit)
            if flags is None:
                logits, loss = model(x, y)
            else:
                logits = model(x)[0]
                loss = weighted_loss(logits, y, flags.chunk(args.chunks)[g][:, 1:], args.prompt_weight)
            collecting[0] = False
            if args.stream_limit:                   # what lies outside the range that the integers will have
                loss = loss + args.stream_weight * sum(F.relu(site.abs() - args.stream_limit).pow(2).mean() for site in stream)
                largest = max(float(site.detach().abs().max()) for site in stream)
            if teacher is not None:         # Hinton et al. 2015: match the teacher's softened distribution as well
                with torch.no_grad():
                    soft = F.softmax(teacher(x)[0] / args.kd_temp, dim=-1)
                kd = F.kl_div(F.log_softmax(logits / args.kd_temp, dim=-1), soft, reduction="sum") / y.numel()
                loss = (1 - args.kd_weight) * loss + args.kd_weight * args.kd_temp ** 2 * kd
            loss = loss * (part.shape[0] / rows.shape[0])
            loss.backward()
            total = total + loss.detach()
        loss = total
        if recall is not None:              # the request texts, each from the first token of the context
            rows, mask = (part.to(device) for part in recall.batch(args.recall_rows, rng, args.recall_boost))
            (args.recall_weight * recall.loss(model(rows[:, :-1])[0], rows, mask)).backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running += loss.detach()
        since += 1
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            train_loss = running.item() / since         # .item() also waits for the device to finish
            assert math.isfinite(train_loss), "the training loss is not finite"
            interval = time.time() - t0
            train_s += interval
            val_loss, val_body = evaluate_parts(model, val_rows, val_flags, args.batch_size)
            # the checkpoint of record is chosen by the loss the run is trained on: all tokens, or the body alone
            score = val_loss if args.prompt_weight == 1.0 else val_body
            better = score < best
            if recall is not None:          # a board model is chosen by the number of texts it reproduces, then by the loss
                model.eval()
                margins = recall.check(lambda x: model(x)[0], device)
                model.train()
                exact = int((margins > 0).sum())
                say(f"recall {exact} of {len(margins)} texts, smallest margin {margins.min():.2f}, median {np.median(margins):.2f}")
                better = exact > best_exact or (exact == best_exact and score < best)
                if better:
                    best_exact = exact
            if better:
                best, best_step, kept = score, step + 1, (val_loss, val_body)
                torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": start + step + 1,
                            "val_loss": val_loss, "tok": args.data}, os.path.join(args.out, "ckpt.pt.tmp"))
                os.replace(os.path.join(args.out, "ckpt.pt.tmp"), os.path.join(args.out, "ckpt.pt"))    # never a half-written file
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "best": best,
                        "best_step": best_step, "train_s": train_s}, last + ".tmp")
            os.replace(last + ".tmp", last)
            say(f"step {step + 1} train {train_loss:.4f} val {val_loss:.4f} bpb {bits_per_byte(val_loss):.4f} "
                f"body {val_body:.4f} lr {lr:.2e} tok/s {since * args.batch_size * T / interval:.0f} time {time.time() - t_start:.0f}"
                + (f" largest |x| {largest:.2f}" if args.stream_limit else ""))
            running.zero_()
            since, t0 = 0, time.time()

    if os.path.exists(last):
        os.remove(last)
    seen = args.steps * args.batch_size * T
    if kept is None:                                # resumed after the best checkpoint was written: measure it again
        model.load_state_dict(torch.load(os.path.join(args.out, "ckpt.pt"), map_location=device)["model"])
        kept = evaluate_parts(model, val_rows, val_flags, args.batch_size)
    result = {"val_loss": kept[0], "val_bpb": bits_per_byte(kept[0]), "val_body": kept[1], "best_step": best_step,
              "final_val_loss": val_loss,
              "steps": args.steps, "tokens_seen": seen, "wall_s": round(time.time() - t_start, 1),
              "tokens_per_s": round(seen / max(train_s, 1e-9)), "device": device, "params": n_params,
              "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20),
              "config": cfg.to_dict(), "args": vars(args)}
    if recall is not None:
        result.update(recall_exact=best_exact, recall_texts=len(recall.names))
    if device == "mps":
        result["mps_mb"] = round(torch.mps.driver_allocated_memory() / 2 ** 20)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    say(f"done: best val {kept[0]:.4f} nats/token = {bits_per_byte(kept[0]):.4f} bits/byte (body {kept[1]:.4f}) at step {best_step}, "
        f"{result['tokens_per_s']} tok/s, {result['wall_s']} s")
    log.close()
    return result


if __name__ == "__main__":
    main()
