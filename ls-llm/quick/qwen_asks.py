"""Implementations of the request suite written by Qwen2.5-Coder-1.5B (local, Apache-2.0), for the asks that have
no verified human implementation. The model is asked for one self-contained function with the given name; a
reply is kept only if it is a single function without imports or prints, at most 254 tokens after the docstring
is removed (what the context can hold), and passes the tests of the ask in the sandbox. Per ask the shortest passing answer is stored.

    python -m quick.qwen_asks [TRIES] [PORT] [K N]      appends to data/asks/qwen_texts.jsonl

K N: only the asks whose place in the suite is K modulo N (several processes share one server). An ask is left
as soon as ENOUGH answers pass. The sampling temperature goes round 0.3, 0.6, 0.9.

The request, the convention note and two test lines of the ask go into the prompt; the stored text is the model's.
"""
import ast
import json
import os
import random
import sys
import urllib.request

from lsllm.asks import ASKS, DATA, load_jsonl
from lsllm.asks.run import DRIVER, MEMORY_MB, _outcome
from lsllm.data.selfcontained import self_contained
from lsllm.sandbox import run_many
from lsllm.tok import Tok
from quick.asks_texts import LONGEST, plain

ENOUGH = 3
TEMPERATURES = (0.3, 0.6, 0.9)


def chat(port, prompt, seed, temperature):
    body = json.dumps({"messages": [{"role": "system", "content": "You are a careful Python teacher."}, {"role": "user", "content": prompt}],
                       "temperature": temperature, "top_p": 0.95, "max_tokens": 600, "seed": seed}).encode()
    request = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as reply:
        return json.loads(reply.read())["choices"][0]["message"]["content"]


def given_names(ask):
    """Classes and functions that the test code of the ask defines (the node class of the tree asks): a function may use them."""
    names = set()
    for test in ask["tests"]:
        try:
            names |= {node.name for node in ast.parse(test).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        except SyntaxError:
            pass
    return names

PROMPT = """Write one Python function named {name}: {request}.
{notes}
It must be a single self-contained function: no imports, no print, no input, no classes, no type hints, no docstring, no comments. If it needs a helper, define the helper inside the function.
It must pass these tests:
{tests}
Output only the Python code of the function, nothing else."""


def main(tries, port, shard=0, shards=1):
    tok = Tok("data/tok/f512v2/tok.model")
    have = {row["name"] for row in load_jsonl(DATA / "train_texts.jsonl")}
    out = DATA / "qwen_texts.jsonl"
    for path in (out, DATA / "long_texts.jsonl"):
        if os.path.exists(path):
            have |= {row["name"] for row in load_jsonl(path)}
    rng = random.Random(os.getpid())
    for place, ask in enumerate(load_jsonl(ASKS)):
        if ask["name"] in have or place % shards != shard:
            continue
        best, passed, why, given = None, 0, {}, given_names(ask)
        for g in range(tries):
            if passed >= ENOUGH:
                break
            prompt = PROMPT.format(name=ask["name"], request=ask["request"], notes=ask.get("notes", ""), tests="\n".join(ask["tests"][:3]))
            try:
                reply = chat(port, prompt, rng.randrange(1 << 30), TEMPERATURES[g % len(TEMPERATURES)])
            except Exception as error:
                print("request failed:", error, flush=True)
                continue
            reply = reply.replace("```python", "").replace("```", "")
            def note(reason):
                why[reason] = why.get(reason, 0) + 1

            try:
                nodes = ast.parse(reply).body
            except SyntaxError:
                note("does not parse")
                continue
            functions = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == ask["name"]]
            others = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name != ask["name"]]
            if len(functions) != 1 or any(isinstance(node, ast.ClassDef) for node in nodes):
                note("not exactly one function")
                continue
            if others:                  # helper functions beside the main one: moved inside it, unchanged (a mechanical step)
                functions[0].body = others + functions[0].body
                note("helpers moved inside")
            text = plain(ast.unparse(functions[0]), ask["name"], ask["name"], LONGEST)
            if text is None:
                note("more than %d tokens" % LONGEST)
                continue
            if not self_contained(text, given) or "print(" in text:        # no module without its import: the text must run alone
                note("not self-contained")
                continue
            result = run_many(["PROGRAM = %r\nTESTS = %r\nMARK = %r\n" % (text, ask["tests"], "REPORT-q ") + DRIVER], 1, 5.0, MEMORY_MB)[0]
            outcome = _outcome(result, "REPORT-q ")[0]
            if outcome == "pass":
                passed += 1
                if best is None or len(text) < len(best):
                    best = text
            else:
                note("tests: " + outcome)
        print(ask["name"], "passing answers: %d of %d" % (passed, g + 1), why, flush=True)
        if best:
            with open(out, "a") as f:
                f.write(json.dumps({"id": ask["id"], "name": ask["name"], "text": best, "tokens": len(tok.encode(best)),
                                    "source": "Qwen2.5-Coder-1.5B-Instruct (local, llama.cpp)", "licence": "Apache-2.0 model; output unrestricted",
                                    "passing_of_tries": [passed, g + 1]}) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6, int(sys.argv[2]) if len(sys.argv) > 2 else 8089,
         *(int(value) for value in sys.argv[3:5]))
