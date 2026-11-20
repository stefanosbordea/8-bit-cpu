"""Exact repair of request texts in an integer board model, without gradients (board models only).

    python -m lsllm.intref.polish MODEL_DIR OUT_DIR --recall FILE[,FILE...] [--rounds 4]

After QAT some request texts are still not written out exactly: at a few positions another token has a logit as
large as the wanted one. The logit of token v at a position is rq(y . E[v] + c_l, sh_l), where y is the output of
the last norm. E is also the input embedding, but the input only sees rqr(E[v], sh_e): with sh_e = 3 an entry of
E can move by up to 7 codes without changing any input, so y stays the same at every position of every text and
the effect of such a change on all logits is known exactly, by integer arithmetic alone.

A repair raises the row of the wanted token, and then lowers the rows of the tokens that still beat it, by single
codes at the coordinates where y of the failing position is largest, as few as needed. A step is taken only if no
other counted position turns wrong by it, and a repair is kept only if the position is right afterwards. Where that
greedy order blocks itself, the same problem is solved as an integer linear program (scipy's HiGHS): the smallest
total change of the row that reaches the needed logit while every other position keeps its room. The decoding rule is the one of the CPU: the smallest index among the
largest logits wins. The model is then written with the same checks as any export (24-bit accumulator bounds).
Texts and tests are those of the request suite: recall of training text, not a benchmark.
"""
import argparse
import json
import os
import shutil

import numpy as np

from lsllm.intref import IntModel
from lsllm.intref.model import ACC_MAX
from lsllm.intref.overflow import bounds
from lsllm.tok import Tok
from lsllm.train.recall import Recall


def trace_all(model, rows, lengths):
    """y of the last norm (texts, time, D) and the logits u (texts, time, V) of every position, texts fed side by side."""
    longest = int(lengths.max()) - 1
    Y, U = [], []
    model.reset(len(rows))
    for t in range(longest):
        seen = {}
        U.append(model.step(rows[:, t], trace=seen))
        Y.append(seen["y"])
    return np.stack(Y, 1).astype(np.int64), np.stack(U, 1).astype(np.int64)


class Board:
    """The positions that count (one row each): y, logits, wanted token; and which of them the CPU gets right."""

    def __init__(self, model, recall):
        self.model, self.recall = model, recall
        self.sh, self.c_l = model.sh["sh_l"], model.pre["c_l"]
        self.refresh()

    def refresh(self):
        rows, mask = self.recall.rows.numpy(), self.recall.mask.numpy() > 0
        Y, U = trace_all(self.model, rows, self.recall.lengths)
        mask = mask[:, :Y.shape[1]]
        self.text = np.nonzero(mask)[0]                             # text of every counted position
        self.Y, self.U = Y[mask], U[mask]                           # (positions, D), (positions, V)
        self.want = rows[:, 1:Y.shape[1] + 1][mask]
        ids = np.arange(self.U.shape[1])
        self.lower = ids[None, :] < self.want[:, None]              # tokens that win a tie against the wanted one
        self.higher = ids[None, :] > self.want[:, None]
        self.status()

    def status(self):
        """fails (positions,) and the number of exact texts, from the logits as they are now."""
        mine = self.U[np.arange(len(self.want)), self.want]
        below = np.where(self.lower, self.U, -1).max(1)
        above = np.where(self.higher, self.U, -1).max(1)
        self.fails = (mine <= below) | (mine < above)
        self.exact = len(self.recall.names) - len(np.unique(self.text[self.fails]))
        return self.fails, self.exact

    def column(self, token):
        """Logits of one token at every counted position, from its row of E."""
        acc = self.Y @ self.model.E[token].astype(np.int64) + self.c_l
        return np.clip(acc >> self.sh, 0, 127)


def free_step(E_row, coordinate, step, sh_e):
    """True if E_row[coordinate] + step stays in int8 and the input embedding rqr(., sh_e) does not change."""
    old = int(E_row[coordinate])
    new = old + step
    half = (1 << sh_e) >> 1
    return -127 <= new <= 127 and (old + half) >> sh_e == (new + half) >> sh_e


BIG = 1 << 40


def room(board, token, skip):
    """How far y . E[token] may move at every counted position before that position turns wrong: (down, up), both
    >= 0, in units of the accumulator. Where the token is the wanted one it must stay above its rivals (down);
    elsewhere it must stay below the wanted token (up). Positions that fail already, and `skip`, set no limit."""
    n, sh = len(board.want), board.sh
    acc = board.Y @ board.model.E[token].astype(np.int64) + board.c_l
    mine = board.want == token
    down, up = np.full(n, BIG, np.int64), np.full(n, BIG, np.int64)
    # the wanted token must keep a code of at least `need`: the best lower-index rival + 1, the best higher-index rival
    need = np.maximum(np.where(board.lower, board.U, -1).max(1) + 1, np.where(board.higher, board.U, -1).max(1))
    rows = mine & (need > 0)
    down[rows] = acc[rows] - (need[rows] << sh)
    # as a rival the token may reach the code of the wanted token (one less if it would win the tie)
    wanted = board.U[np.arange(n), board.want]
    allowed = wanted - (token < board.want)
    rows = ~mine & (allowed < 127)
    up[rows] = (((allowed[rows] + 1) << sh) - 1) - acc[rows]
    free = board.fails.copy()
    free[skip] = True
    down[free], up[free] = BIG, BIG
    return down, up


def move(board, token, at, direction, gain, sh_e, limit):
    """Change the row of `token` so that y(at) . E[token] moves by `gain` in `direction` (+1 or -1) or as far as it
    can: single-code steps, largest |y(at)| first, each step taken only if no other counted position turns wrong
    (the effect of a step on every position is known exactly). Returns (steps as (coordinate, step), amount moved)."""
    y, row = board.Y[at], board.model.E[token].copy()
    down, up = room(board, token, at)
    order = [int(j) for j in np.argsort(-np.abs(y), kind="stable") if y[j] != 0]
    steps, got, drift = [], 0, np.zeros(len(board.want), np.int64)
    for lap in range(7):                                            # a coordinate can move several codes inside its input bucket
        took = False
        for j in order:
            if got >= gain or len(steps) >= limit:
                break
            step = direction * (1 if y[j] > 0 else -1)
            if not free_step(row, j, step, sh_e):
                continue
            after = drift + step * board.Y[:, j]
            if (after > up).any() or (-after > down).any():
                continue
            drift, took = after, True
            row[j] += step
            steps.append((j, step))
            got += abs(int(y[j]))
        if got >= gain or len(steps) >= limit or not took:
            break
    return steps, got


def play(E_row, sh_e):
    """How far every entry of a row of E may go down and up (two arrays, <= 0 and >= 0) with the same input embedding."""
    row = E_row.astype(np.int64)
    half = (1 << sh_e) >> 1
    bucket = (row + half) >> sh_e
    low = np.maximum((bucket << sh_e) - half, -127)
    high = np.minimum((bucket << sh_e) - half + (1 << sh_e) - 1, 127)
    return low - row, high - row


def solve(board, token, at, direction, gain, sh_e, seconds=20.0):
    """The same aim as move(), by integer linear programming (scipy, HiGHS): whole-code changes d of the row of `token`,
    inside the play of every entry, such that y(at) . d reaches `gain` in `direction` (or the most that is possible) while
    no other counted position turns wrong, with the smallest total change. For the positions where the greedy order of
    move() blocks itself. Returns (steps as (coordinate, change), amount moved); the caller checks the result exactly."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    y = board.Y[at].astype(np.float64)
    down, up = room(board, token, at)
    low, high = play(board.model.E[token], sh_e)
    reach = np.abs(board.Y) @ np.maximum(-low, high)                # the most y(p) . d can be at each position
    rows = np.nonzero((up < reach) | (down < reach))[0]             # only positions that could turn wrong at all
    A = board.Y[rows].astype(np.float64)
    lb = np.where(down[rows] < BIG, -down[rows], -np.inf).astype(np.float64)
    ub = np.where(up[rows] < BIG, up[rows], np.inf).astype(np.float64)
    n = len(y)
    limits = Bounds(np.zeros(2 * n), np.concatenate([high, -low]).astype(np.float64))      # d = plus - minus
    both = np.hstack([A, -A])
    aim = direction * np.concatenate([y, -y])
    most = milp(-aim, integrality=np.zeros(2 * n), bounds=limits, constraints=[LinearConstraint(both, lb, ub)])      # without whole numbers: the most that can be had
    if most.x is None or -most.fun < 1:
        return [], 0
    wanted = min(gain, int(-most.fun * 0.97))
    if wanted < 1:
        return [], 0
    found = milp(np.ones(2 * n), integrality=np.ones(2 * n), bounds=limits,
                 constraints=[LinearConstraint(both, lb, ub), LinearConstraint(aim[None, :], wanted, np.inf)], options={"time_limit": seconds})
    if found.x is None:
        return [], 0
    change = np.rint(found.x[:n] - found.x[n:]).astype(np.int64)
    steps = [(int(j), int(change[j])) for j in np.nonzero(change)[0]]
    return steps, int(direction * (board.Y[at] @ change))


def polish(model, recall, rounds, limit, say=print):
    sh_e = model.sh.get("sh_e", 0)
    assert sh_e >= 1, "E has no spare bits below the input embedding (sh_e = 0): nothing can move without changing inputs"
    board = Board(model, recall)
    fails, exact = board.status()
    say("start: %d of %d texts exact, %d of %d positions fail" % (exact, len(recall.names), fails.sum(), len(fails)))
    log = []

    def apply(token, steps):
        for j, step in steps:
            model.E[token, j] += step
        board.U[:, token] = board.column(token)

    for lap in range(rounds):
        changed, stuck = 0, []
        per_text = np.bincount(board.text[board.fails], minlength=len(recall.names))
        order = sorted(np.nonzero(board.fails)[0], key=lambda at: (per_text[board.text[at]], at))      # texts with one failing position first
        for at in order:
            fails, exact = board.status()
            if not fails[at]:
                continue
            want, total, done = int(board.want[at]), int(fails.sum()), []
            for attempt in range(6):                                # raise the wanted token, then lower what still beats it
                u = board.U[at]
                rivals = [int(v) for v in np.nonzero((board.lower[at] & (u >= u[want])) | (board.higher[at] & (u > u[want])))[0]]
                if not rivals:
                    break
                target = max(int(u[v]) + (1 if v < want else 0) for v in rivals)        # the code the wanted token must reach
                moved = 0
                if target <= 127 and attempt == 0:
                    acc = int(board.Y[at] @ model.E[want].astype(np.int64)) + board.c_l
                    steps, moved = move(board, want, at, +1, (target << board.sh) - acc, sh_e, limit)
                    apply(want, steps)
                    done.append((want, steps))
                else:
                    for v in rivals:
                        acc = int(board.Y[at] @ model.E[v].astype(np.int64)) + board.c_l
                        ceiling = int(board.U[at, want]) - (1 if v < want else 0)       # the largest code the rival may keep
                        if ceiling < 0:
                            continue
                        steps, got = move(board, v, at, -1, acc - (((ceiling + 1) << board.sh) - 1), sh_e, limit)
                        apply(v, steps)
                        done.append((v, steps))
                        moved += got
                    if not moved:
                        break
            new_fails, new_exact = board.status()
            broke = np.setdiff1d(np.unique(board.text[new_fails]), np.unique(board.text[fails]))
            if len(broke) == 0 and not new_fails[at] and new_fails.sum() < total:
                log += [{"text": recall.names[board.text[at]], "position": int(at), "token": token, "steps": steps} for token, steps in done if steps]
                changed += 1
            else:                                                   # not repaired: take every step of this position back
                for token, steps in reversed(done):
                    apply(token, [(j, -step) for j, step in steps])
                stuck.append(int(at))
            board.status()
        for at in stuck if lap == 0 or not changed else []:        # what the greedy order could not do: integer programming
            fails, exact = board.status()
            if not fails[at]:
                continue
            want, total, done = int(board.want[at]), int(fails.sum()), []
            for attempt in range(4):
                u = board.U[at]
                rivals = [int(v) for v in np.nonzero((board.lower[at] & (u >= u[want])) | (board.higher[at] & (u > u[want])))[0]]
                if not rivals:
                    break
                target = min(127, max(int(u[v]) + (1 if v < want else 0) for v in rivals))
                acc = int(board.Y[at] @ model.E[want].astype(np.int64)) + board.c_l
                moved = 0
                if attempt == 0 and target > int(u[want]):          # raise the wanted token to the highest code it can reach
                    steps, got = solve(board, want, at, +1, (target << board.sh) - acc, sh_e)
                    code = (acc + got) >> board.sh
                    if steps and code > int(u[want]) and got > (code << board.sh) - acc:      # not past a code boundary for nothing
                        steps, got = solve(board, want, at, +1, (code << board.sh) - acc, sh_e)
                    if steps and (acc + got) >> board.sh > int(u[want]):
                        apply(want, steps)
                        done.append((want, steps))
                        moved += got
                else:
                    for v in rivals:
                        acc_v = int(board.Y[at] @ model.E[v].astype(np.int64)) + board.c_l
                        ceiling = int(board.U[at, want]) - (1 if v < want else 0)
                        if ceiling < 0:
                            continue
                        steps, got = solve(board, v, at, -1, acc_v - (((ceiling + 1) << board.sh) - 1), sh_e)
                        if steps:
                            apply(v, steps)
                            done.append((v, steps))
                            moved += got
                    if not moved:
                        break
            new_fails, new_exact = board.status()
            broke = np.setdiff1d(np.unique(board.text[new_fails]), np.unique(board.text[fails]))
            if len(broke) == 0 and not new_fails[at] and new_fails.sum() < total:
                log += [{"text": recall.names[board.text[at]], "position": int(at), "token": token, "steps": steps, "by": "integer program"}
                        for token, steps in done if steps]
                changed += 1
            else:
                for token, steps in reversed(done):
                    apply(token, [(j, -step) for j, step in steps])
            board.status()
        fails, exact = board.status()
        say("round %d: %d repairs kept; %d of %d texts exact, %d positions fail" % (lap + 1, changed, exact, len(recall.names), fails.sum()))
        if not changed or not fails.any():
            break
    return board, log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="integer model (folder with model.json and model.npz)")
    p.add_argument("out", help="folder for the repaired model")
    p.add_argument("--recall", required=True, help="JSONL files with the request texts, separated by commas")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--limit", type=int, default=400, help="most single-code steps of one repair")
    args = p.parse_args()
    model = IntModel.load(args.model)
    spec = json.load(open(os.path.join(args.model, "model.json")))
    recall = Recall(args.recall.split(","), Tok(spec["tokenizer"] + "/tok.model"), model.cfg.max_seq_len)
    start = model.E.copy()
    board, log = polish(model, recall, args.rounds, args.limit)
    os.makedirs(args.out, exist_ok=True)
    arrays = dict(np.load(os.path.join(args.model, "model.npz")))
    arrays["E"] = model.E.astype(np.int8)
    moved = int((model.E != start).sum())
    spec["notes"] = spec.get("notes", []) + ["E repaired by lsllm.intref.polish: %d entries moved by at most %d codes, no input embedding changed"
                                             % (moved, int(np.abs(model.E.astype(np.int64) - start).max()))]
    for stream, bound in bounds(IntModel(spec, arrays)).items():
        if bound > ACC_MAX:
            raise ValueError("stream %s: worst-case accumulator %d is outside 24 bits" % (stream, bound))
    with open(os.path.join(args.out, "model.json"), "w") as f:
        json.dump(spec, f, indent=1)
    np.savez(os.path.join(args.out, "model.npz"), **arrays)
    with open(os.path.join(args.out, "texts.jsonl"), "w") as f:      # the request texts this model writes: read by tools/board_ask.py --list
        for path in args.recall.split(","):
            f.write(open(path).read())
    with open(os.path.join(args.out, "polish.jsonl"), "w") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")
    check = Board(IntModel.load(args.out), recall)                  # from the written files, everything computed again
    fails, exact = check.status()
    half = (1 << model.sh["sh_e"]) >> 1
    same_inputs = bool(np.array_equal((start.astype(np.int64) + half) >> model.sh["sh_e"], (model.E.astype(np.int64) + half) >> model.sh["sh_e"]))
    result = {"texts": len(recall.names), "exact": int(exact), "failing_positions": int(fails.sum()), "positions": int(len(fails)),
              "entries_of_E_moved": moved, "repairs": len(log), "input_embedding_unchanged": same_inputs,
              "not_exact": sorted({recall.names[t] for t in check.text[fails]})}
    json.dump(result, open(os.path.join(args.out, "polish.json"), "w"), indent=1)
    for name in ("result.json", "log.txt"):
        if os.path.exists(os.path.join(args.model, name)):
            shutil.copy(os.path.join(args.model, name), os.path.join(args.out, "qat_" + name))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
