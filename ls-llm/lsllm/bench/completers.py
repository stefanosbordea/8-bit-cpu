"""Completers that validate the harness itself, and the registry that names every completer."""
import importlib
import random

from lsllm.bench.tasks import load_tasks

# kind -> "module:callable". The callable takes the text after "kind:" (nothing if there is none) and returns
# a Completer. The module is imported only when asked for, so the harness itself never needs torch.
# Another area plugs in with one line, for example:  "torch": "lsllm.model.completer:TorchCompleter",
COMPLETERS = {
    "ref": "lsllm.bench.completers:RefCompleter",
    "empty": "lsllm.bench.completers:EmptyCompleter",
    "garbage": "lsllm.bench.completers:GarbageCompleter",
    "torch": "lsllm.model.completer:TorchCompleter",
    "torchf": "lsllm.model.completer:filtered",
    "stories260k": "lsllm.baselines.stories260k:Stories260K",
    "ngram": "lsllm.baselines.ngram:NgramCompleter",
    "public": "lsllm.baselines.public:PublicModel",
    "int": "lsllm.intref.completer:IntCompleter",
    "intf": "lsllm.intref.completer:filtered",
    "torchp": "lsllm.model.completer:plain",
    "intp": "lsllm.intref.completer:plain",
    "asksref": "lsllm.asks.run:RefCompleter",
}


def load_completer(spec):
    """Build a completer from "ref", "empty", "garbage" or "<kind>:<argument>" such as "torch:runs/a/ckpt.pt"."""
    kind, _, argument = spec.partition(":")
    if kind not in COMPLETERS:
        raise ValueError("unknown completer %r; known: %s" % (kind, ", ".join(COMPLETERS)))
    module, _, name = COMPLETERS[kind].partition(":")
    factory = getattr(importlib.import_module(module), name)
    return factory(argument) if argument else factory()


class RefCompleter:
    """Returns the reference solution of the task. Expected score: every task passes."""
    name = "ref"

    def __init__(self):
        self.continuation = {}      # prompt -> rest of the reference solution
        for benchmark, split in (("humaneval", "test"), ("mbpp", "test"), ("mbpp", "practice"),
                                 ("custom", "test"), ("custom", "practice")):
            try:
                tasks = load_tasks(benchmark, split)
            except FileNotFoundError:           # the custom tasks are not written yet
                continue
            for task in tasks:
                if not task.canonical_solution.startswith(task.prompt):
                    raise ValueError(task.task_id + ": the reference solution does not start with the prompt")
                self.continuation.setdefault(task.prompt, task.canonical_solution[len(task.prompt):])

    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        return [self.continuation[prompt]] * n


class EmptyCompleter:
    """Returns a body that is just `pass`. Expected score: valid syntax everywhere, no task passes."""
    name = "empty"

    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        return ["\n    pass"] * n


class GarbageCompleter:
    """Returns random printable characters, one per token: the score of a model that knows nothing."""
    name = "garbage"
    alphabet = [chr(code) for code in range(32, 127)] + ["\n"]

    def complete(self, prompt, n, max_new_tokens, temperature, seed):
        rng = random.Random(seed)
        return ["".join(rng.choices(self.alphabet, k=max_new_tokens)) for g in range(n)]
