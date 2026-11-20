"""An editing session on an exported integer model (SPEC Part E): change numbers by hand, never by gradients.

A session is a directory with a working copy of model.json and model.npz, a budget and a log:

    s = Session.create("runs/final_int", "edits/s1", strategy="what this session tries")
    s = Session("edits/s1")                        # open it again
    s.evaluate("why")                              # one budgeted run on the PRACTICE data, returns scores
    s.trace(list_of_token_lists, ["l2.g"], "why")  # one budgeted run that returns intermediate values
    w = s.get("W2_3"); w[:, 5] = 0; s.put("W2_3", w, "reason")       # an edit; refused if it breaks the hardware rules
    s.put_number(("preloads", "c_l"), 1200, "reason")                # a number of model.json (shift or preload)
    s.undo()                                       # take back the last edit

Rules that the class enforces:
- at most LIMIT budgeted runs per session (evaluate and trace together);
- the data of a budgeted run is practice data only: the custom practice tasks, and windows of the TRAINING text
  for the loss. The validation text and the test benchmarks are never read here;
- an edited model must still be a model of the LS CPU: int8 arrays, weights in -127..127, EXP in 0..127 with
  EXP[l][0] = 127, RECIP in 0..127, shifts and 16-bit preloads in their ranges, and no stream that could leave
  the signed 24-bit accumulator in the worst case;
- every edit is logged with the tensor, the positions and the old and new values (edits.jsonl).
Nothing here computes a gradient; an edit is whatever numbers the caller writes.
"""
import json
import os
import re
import shutil
import time
from collections import Counter

import numpy as np

from lsllm.intref.model import ACC_MAX, IntModel
from lsllm.intref.overflow import bounds

LIMIT = 50                      # budgeted runs per session
LOSS_WINDOWS = 32               # windows of the training text for the practice loss
MAX_TRACE_TOKENS = 4096         # tokens per trace run


class Session:
    def __init__(self, directory):
        self.dir = directory
        with open(os.path.join(directory, "model.json")) as f:
            self.spec = json.load(f)
        with np.load(os.path.join(directory, "model.npz")) as arrays:
            self.arrays = {name: arrays[name] for name in arrays.files}
        with open(os.path.join(directory, "budget.json")) as f:
            self.budget = json.load(f)

    @classmethod
    def create(cls, base, directory, strategy=""):
        """Start a session in the new directory `directory` from the exported model in `base`."""
        os.makedirs(directory)
        for name in ("model.json", "model.npz"):
            shutil.copy(os.path.join(base, name), os.path.join(directory, name))
        with open(os.path.join(directory, "budget.json"), "w") as f:
            json.dump({"base": base, "strategy": strategy, "limit": LIMIT, "runs": []}, f, indent=1)
        open(os.path.join(directory, "edits.jsonl"), "w").close()
        return cls(directory)

    # ---- the model ----

    def model(self):
        """The integer reference of the working copy."""
        return IntModel(self.spec, self.arrays)

    def get(self, name):
        """A copy of an array of model.npz as int32 (so that arithmetic does not wrap)."""
        return self.arrays[name].astype(np.int32)

    def left(self):
        return self.budget["limit"] - len(self.budget["runs"])

    def _spend(self, kind, note):
        if self.left() <= 0:
            raise RuntimeError("the budget of %d runs of this session is used up" % self.budget["limit"])
        self.budget["runs"].append({"run": len(self.budget["runs"]) + 1, "kind": kind, "note": note,
                                    "edits_so_far": len(self.edits()), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        with open(os.path.join(self.dir, "budget.json"), "w") as f:
            json.dump(self.budget, f, indent=1)
        return len(self.budget["runs"])

    # ---- budgeted runs on practice data ----

    def evaluate(self, note="", max_new_tokens=64, syntax_filter=True):
        """One budgeted run: the practice loss on training text and the custom practice tasks (greedy).

        Returns a dictionary; the completions are written to eval_<run>.jsonl in the session directory."""
        from lsllm.bench import evaluate, load_tasks
        from lsllm.intref.completer import IntCompleter
        from lsllm.tok import Tok
        run = self._spend("evaluate", note)
        tok_dir = self.spec["tokenizer"]
        model, T = self.model(), self.model().cfg.max_seq_len
        train = np.memmap(os.path.join(tok_dir, "train.bin"), dtype=np.uint16, mode="r")
        starts = np.linspace(0, len(train) - T - 1, LOSS_WINDOWS).astype(np.int64)
        rows = np.asarray(train[starts[:, None] + np.arange(T + 1)]).astype(np.int64)
        logits = model.forward(rows[:, :-1]) / 2.0 ** model.f_l
        logits = logits - logits.max(-1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        loss = float(-np.take_along_axis(logp, rows[:, 1:, None], -1).mean())

        completer = IntCompleter(self.dir, Tok(os.path.join(tok_dir, "tok.model")), syntax_filter)
        completer.model = model
        report = evaluate(completer, load_tasks("custom", "practice"), 1, 0.0, max_new_tokens)
        errors, credit, repeats = Counter(), 0.0, 0
        for sample in report.samples:
            if sample["outcome"] == "error":
                errors[sample["detail"].split("(")[0].replace("error: ", "")] += 1
            elif sample["outcome"] == "pass":
                credit += 1
            elif sample["outcome"] == "wrong":
                failed, total = map(int, re.match(r"(\d+) of (\d+)", sample["detail"]).groups())
                credit += 1 - failed / total
            words = sample["completion"].split()
            grams = Counter(tuple(words[g:g + 3]) for g in range(len(words) - 2))
            repeats += bool(grams) and max(grams.values()) >= 3
        with open(os.path.join(self.dir, "eval_%02d.jsonl" % run), "w") as f:
            for sample in report.samples:
                f.write(json.dumps(sample) + "\n")
        out = {"run": run, "budget_left": self.left(), "practice_loss": round(loss, 4),
               "pass@1": report.metrics["pass@1"], "syntax_ok": report.metrics["syntax_ok"],
               "runs_ok": report.metrics["runs_ok"], "tests_passed": round(credit / report.n_tasks, 4),
               "outcomes": report.outcomes, "errors": dict(errors.most_common(6)),
               "repeating": repeats / report.n_tasks, "note": note}
        self.budget["runs"][-1]["result"] = out
        with open(os.path.join(self.dir, "budget.json"), "w") as f:
            json.dump(self.budget, f, indent=1)
        return out

    def trace(self, sequences, names, note=""):
        """One budgeted run: feed every token list of `sequences` from an empty cache and return, per sequence,
        {name: array (positions, ...)} of the intermediate values `names` (see lsllm/intref/model.py; "u" = logits).

        The sequences must be practice material: practice prompts, completions of an evaluate run, training text."""
        total = sum(len(seq) for seq in sequences)
        if total > MAX_TRACE_TOKENS:
            raise ValueError("%d tokens in one trace run; the limit is %d" % (total, MAX_TRACE_TOKENS))
        self._spend("trace", note)
        model, out = self.model(), []
        for seq in sequences:
            model.reset()
            kept = {name: [] for name in names}
            for t in seq:
                values = {}
                u = model.step([int(t)], trace=values)
                values["u"] = u
                for name in names:
                    kept[name].append(np.array(values[name][0]))
            out.append({name: np.stack(parts) for name, parts in kept.items()})
        return out

    # ---- edits ----

    def edits(self):
        with open(os.path.join(self.dir, "edits.jsonl")) as f:
            return [json.loads(line) for line in f]

    def _check(self, spec, arrays):
        for name, value in arrays.items():
            if name == "EXP":
                assert value.min() >= 0 and (value[:, 0] == 127).all(), "EXP must be 0..127 with EXP[l][0] = 127"
            if name == "RECIP":
                assert value.min() >= 0, "RECIP must be 0..127"
        model = IntModel(spec, arrays)                  # checks int8, weight range, shifts and preloads
        for stream, bound in bounds(model).items():
            assert bound <= ACC_MAX, "stream %s could reach %d: outside the 24-bit accumulator" % (stream, bound)

    def _save(self, spec, arrays, entry):
        self._check(spec, arrays)
        self.spec, self.arrays = spec, arrays
        with open(os.path.join(self.dir, "model.json"), "w") as f:
            json.dump(spec, f, indent=1)
        np.savez(os.path.join(self.dir, "model.npz"), **arrays)
        if entry:
            entry = dict(edit=len(self.edits()) + 1, after_run=len(self.budget["runs"]), **entry)
            with open(os.path.join(self.dir, "edits.jsonl"), "a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry

    def put(self, name, values, reason):
        """Replace the array `name` by `values` (same shape, whole numbers). Returns the log entry, None if
        nothing changed. Raises AssertionError, and changes nothing, if the model would break a hardware rule."""
        old = self.arrays[name]
        values = np.asarray(values)
        assert values.shape == old.shape, "%s has shape %s" % (name, old.shape)
        assert np.all(values == np.round(values)), "the values must be whole numbers"
        assert values.min() >= -128 and values.max() <= 127, "the values must fit int8"
        new = values.astype(np.int8)
        where = np.flatnonzero(new != old)
        if not len(where):
            return None
        arrays = dict(self.arrays, **{name: new})
        return self._save(self.spec, arrays, {"tensor": name, "shape": list(old.shape), "reason": reason,
                                              "changed": len(where), "positions": where.tolist(),
                                              "old": old.ravel()[where].tolist(), "new": new.ravel()[where].tolist()})

    def put_number(self, path, value, reason):
        """Set a number of model.json, e.g. ("preloads", "c_l"), ("preloads", "c_s", 2, 1) or ("shifts", "sh_l")."""
        assert path[0] in ("shifts", "preloads"), "only shifts and preloads can be edited in model.json"
        spec = json.loads(json.dumps(self.spec))
        node = spec
        for key in path[:-1]:
            node = node[key]
        old = node[path[-1]]
        assert isinstance(old, int) and int(value) == value, "path must lead to one whole number"
        if old == value:
            return None
        node[path[-1]] = int(value)
        return self._save(spec, self.arrays, {"tensor": "model.json:" + ".".join(map(str, path)), "reason": reason,
                                              "changed": 1, "positions": [0], "old": [old], "new": [int(value)]})

    def undo(self):
        """Take back the last edit. The log keeps both the edit and its reversal."""
        last = self.edits()[-1]
        if last["tensor"].startswith("model.json:"):
            path = [int(key) if key.isdigit() else key for key in last["tensor"][len("model.json:"):].split(".")]
            return self.put_number(tuple(path), last["old"][0], "undo of edit %d" % last["edit"])
        values = self.arrays[last["tensor"]].copy()
        values.ravel()[last["positions"]] = last["old"]
        return self.put(last["tensor"], values, "undo of edit %d" % last["edit"])


def summary(directory):
    """Lines of EDITS.md for one session: its runs and every edit with positions, old and new values."""
    s = Session(directory)
    lines = ["## Session %s" % os.path.basename(directory), "",
             "Strategy: %s" % s.budget["strategy"], "",
             "Base model: `%s`. Budgeted runs used: %d of %d." % (s.budget["base"], len(s.budget["runs"]), s.budget["limit"]), "",
             "| run | kind | edits before it | practice loss | pass@1 | syntax ok | runs ok | tests passed | note |", "|---|---|---|---|---|---|---|---|---|"]
    for run in s.budget["runs"]:
        r = run.get("result", {})
        show = lambda key, form="%.4f": form % r[key] if key in r else "-"
        lines.append("| %d | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            run["run"], run["kind"], run["edits_so_far"], show("practice_loss"), show("pass@1", "%.3f"), show("syntax_ok", "%.3f"),
            show("runs_ok", "%.3f"), show("tests_passed"), run["note"].replace("|", "/")))
    lines += ["", "| edit | after run | tensor | changed elements | reason | positions (flat index): old -> new |", "|---|---|---|---|---|---|"]
    for e in s.edits():
        pairs = ["%d: %d -> %d" % item for item in zip(e["positions"], e["old"], e["new"])]
        shown = ", ".join(pairs[:12]) + (" ... (all %d in `%s/edits.jsonl`, edit %d)" % (len(pairs), directory, e["edit"]) if len(pairs) > 12 else "")
        lines.append("| %d | %d | `%s` | %d | %s | %s |" % (e["edit"], e["after_run"], e["tensor"], e["changed"], e["reason"].replace("|", "/"), shown))
    return lines
