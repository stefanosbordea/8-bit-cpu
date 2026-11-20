"""Synthetic simple functions written by Qwen2.5-Coder-1.5B-Instruct (Apache-2.0, run locally with llama.cpp).

The prompts below only name beginner topics; the training text is the model's output. Every function is parsed,
canonicalised and checked against the benchmark overlap rules. Since 2026-11-14 the model also writes three test
lines per function, and a function is kept only if at least two such lines exist and all of them pass in the
sandbox; it is also run on probe inputs to obtain further assert lines (the functions of the first runs, folders
a to r, were kept if they ran on the probes; they have no model-written tests). Usage: python -m quick.qwen_gen OUT_DIR TARGET_COUNT [PORT]
"""
import ast
import json
import os
import random
import re
import sys
import time
import urllib.request

from lsllm.alpha import alpha_key
from lsllm.canon import canon
from lsllm.sandbox import run_many
from quick.synth import benchmark_keys

TOPICS = [
    "arithmetic on one number (add, subtract, multiply, divide, powers, remainders with a fixed constant)",
    "arithmetic on two numbers", "arithmetic on three numbers and short everyday formulas (areas, prices, speeds, totals)",
    "comparisons of two or three numbers that return True or False", "the smallest or largest of two or three numbers given as separate arguments",
    "True/False questions about one whole number (even, odd, divisible, positive, negative, zero)",
    "unit conversions with a fixed factor (time, length, weight, money, data sizes, temperature)",
    "one if/else that chooses between two numbers", "combining True/False arguments with and, or, not",
    "a for loop over a range that adds up, multiplies or counts numbers", "building a list of numbers with range or repetition",
    "one number computed from all items of a list (sum, largest, smallest, average, product, how many)",
    "keeping only the items of a list that pass a simple test", "applying the same simple operation to every item of a list",
    "items at fixed positions of a list (first, last, second, middle)", "taking part of a list with one slice",
    "True/False questions about a list (contains, empty, longer than, all or any items pass a test)",
    "making a new list by adding, removing or replacing one item at a fixed place", "upper case, lower case and length of strings",
    "True/False questions about one string (starts with, ends with, contains, only digits, empty)",
    "counting the characters of a string that are of a given kind", "characters and parts of a string taken by position or one slice",
    "building a string by joining given strings with fixed text or by repeating", "reversing a list, a string or a few values",
    # general beginner topics, added 2026-11-14 for more variety (the far-family filters below still apply)
    "rounding, absolute value, integer division and remainders", "converting between numbers and strings (int, str, float)",
    "combining two strings (concatenation, repetition, a separator given as an argument)", "replacing or removing characters in a string",
    "returning a describing string for a number chosen with if/elif/else (sign, size, range)",
    "tuples: making a tuple from arguments and taking parts of a tuple", "lists of strings: lengths, filtering by length, changing the case of every string",
    "two lists: joining them, comparing their lengths, pairwise sums with a loop over positions", "counting how many items of a list pass a test",
    "sums and products with a condition (sum of the positive items, product of the non-zero items)",
    "ranges with a step, counting down, every k-th item", "clamping a number into a range and choosing by thresholds",
    "simple recursion on a number (sum to n, power, countdown)", "True/False questions about two lists or two strings",
    "minimum, maximum and the difference between the largest and the smallest item of a list",
]
STYLES = ["Use a for loop where it is natural.", "Prefer a single return expression.", "Use an if statement where it is natural.",
          "Use a list comprehension where it is natural.", "Use built-in functions where it is natural.", "Vary the style between the functions."]
THEMES = ["shopping", "school grades", "sports scores", "cooking", "travel", "weather", "games", "money", "time", "music", "animals", "books",
          "a classroom", "a garden", "a bakery", "a library", "a parking lot", "plain numbers", "plain numbers", "plain text", "plain lists"]
PROMPT = """Write {n} different very simple Python functions for beginners about this topic: {topic}.
Context for names and wording: {theme}. {style}
Rules: each function is self-contained, 1 to 6 lines of body, no imports, no print, no input, no classes, no type hints, no comments.
Each function has a one-line docstring that starts with "Return" and says exactly what is returned, naming the arguments.
Give each function a short descriptive snake_case name. The docstring is inside the function, in triple double quotes, never a comment above it.
After each function write exactly three test lines that call it with simple literal arguments and state the correct result.
Use exactly this layout for every function:

def function_name(argument_names):
    \"\"\"Return <what is returned>.\"\"\"
    <body>

assert function_name(<arguments>) == <result>
assert function_name(<arguments>) == <result>
assert function_name(<arguments>) == <result>

Output only the Python code, nothing else."""
FAR_WORDS = re.compile(r"prime|gcd|lcm|divisor|palindrom|anagram|fizz|grade|sort|flatten|matrix|nested|word|sentence|running|cumulative|digit|index_of|find|search|duplicate|unique|distinct|fibonacci|collatz|rotate|swap|shift|binary|bit|dict|zip", re.I)
FAR_CODE = re.compile(r"sorted\(|\.sort\(|\{|\bdict\(|\bset\(|\bzip\(|\.split\(|\.join\(|\bbin\(|enumerate\(|\.index\(|\.find\(|while ")
PROBES = [(), ("3",), ("0",), ("-4",), ("10",), ("[1, 2, 3]",), ("[]",), ("[5, -2, 8, 0]",), ("'hello'",), ("''",), ("'Abc 12'",), ("True",), ("False",),
          ("7", "2"), ("0", "5"), ("-3", "4"), ("10", "10"), ("[1, 2, 3]", "2"), ("[4, 5, 6, 7]", "1"), ("[]", "3"), ("'hello'", "2"), ("'hello'", "'l'"), ("'abc'", "'ab'"),
          ("''", "'a'"), ("True", "False"), ("True", "True"), ("[1, 2]", "[3]"), ("'ab'", "'cd'"), ("2", "[1, 2, 3]"),
          ("1", "2", "3"), ("5", "5", "5"), ("10", "2", "7"), ("[1, 2, 3, 4]", "1", "3"), ("[1, 2, 3]", "0", "9"), ("'hello'", "1", "3"), ("True", "False", "True"), ("3", "1", "10")]
DRIVER = """
import json
out = []
for src, name, probes, *more in json.loads(%r):
    ns, got = {}, []
    try:
        exec(src, ns)
        fn = ns[name]
    except Exception:
        out.append([]); continue
    passed = 0
    for line in (more[0] if more else []):
        try:
            exec(line, dict(ns)); passed += 1
        except BaseException:
            pass
    if more:
        got.append(["#asserts", str(passed)])
    for args in probes:
        call = "%%s(%%s)" %% (name, ", ".join(args))
        try:
            a = eval(call, ns); b = eval(call, ns)
            if a == b and (a is None) == False and type(a) in (int, float, bool, str, list, tuple) and len(repr(a)) < 60:
                got.append([call, repr(a)])
        except Exception:
            pass
    out.append(got)
print("RESULT" + json.dumps(out))
"""


def ask(port, prompt, seed):
    body = json.dumps({"messages": [{"role": "system", "content": "You are a careful Python teacher."}, {"role": "user", "content": prompt}],
                       "temperature": 0.9, "top_p": 0.95, "max_tokens": 1100, "seed": seed}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def functions(text):
    """Top-level functions in a model reply, canonical, with a docstring, small, plain."""
    text = re.sub(r"```(?:python)?", "", text)
    nodes = []
    for chunk in re.split(r"(?m)^(?=def )", text):       # parse function by function: a cut-off reply loses only its last one
        try:
            nodes += ast.parse(chunk).body
        except SyntaxError:
            pass
    out = []
    for place, node in enumerate(nodes):
        if not isinstance(node, ast.FunctionDef) or ast.get_docstring(node) is None or node.decorator_list:
            continue
        asserts = []                                    # the test lines the model wrote for this function
        for later in nodes[place + 1:]:
            if not isinstance(later, ast.Assert):
                break
            calls = [n for n in ast.walk(later.test) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            if later.msg is None and calls and all(c.func.id == node.name for c in calls) and \
                    not any(isinstance(n, ast.Name) and n.id != node.name for n in ast.walk(later.test)):
                asserts.append(ast.unparse(later))
        if any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Lambda, ast.ClassDef, ast.Try, ast.With)) for n in ast.walk(node)):
            continue
        if any(a.annotation for a in node.args.args) or node.returns or node.args.vararg or node.args.kwarg or node.args.defaults:
            continue
        src = canon(ast.unparse(node))
        if src is None or src.count("\n") > 9 or len(src) > 420 or "print(" in src or "input(" in src or '"""' not in src:
            continue
        doc = ast.get_docstring(node)
        if "\n" in doc or not doc.startswith("Return"):
            continue
        out.append({"text": src, "name": node.name, "n_args": len(node.args.args), "asserts": asserts[:4]})
    return out


def main(out_dir, target, port):
    os.makedirs(out_dir, exist_ok=True)
    names, keys = benchmark_keys()
    path, seen, kept = out_dir + "/all.jsonl", set(), 0
    if os.path.exists(path):
        for line in open(path):
            seen.add(json.loads(line)["text"])
            kept += 1
    stats = {"replies": 0, "functions": 0, "far_filtered": 0, "benchmark_name": 0, "benchmark_code": 0, "no_tests": 0, "duplicate": 0, "kept": kept}
    rng = random.Random(os.getpid() * 1000003 + int(time.time()))
    while kept < target:
        topic = rng.choice(TOPICS)
        prompt = PROMPT.format(n=8, topic=topic, theme=rng.choice(THEMES), style=rng.choice(STYLES))
        try:
            reply = ask(port, prompt, rng.randrange(1 << 30))
        except Exception as e:
            print("request failed:", e, flush=True)
            time.sleep(5)
            continue
        stats["replies"] += 1
        cand = []
        for f in functions(reply):
            stats["functions"] += 1
            if FAR_WORDS.search(f["name"]) or FAR_CODE.search(f["text"]):
                stats["far_filtered"] += 1
            elif f["name"] in names:
                stats["benchmark_name"] += 1
            elif alpha_key(f["text"]) in keys:
                stats["benchmark_code"] += 1
            elif f["text"] in seen:
                stats["duplicate"] += 1
            else:
                cand.append(f)
        if cand:
            code = DRIVER % json.dumps([[f["text"], f["name"], [p for p in PROBES if len(p) == f["n_args"]], f["asserts"]] for f in cand])
            res = run_many([code], workers=1, timeout_s=20.0)[0]
            outs = json.loads(res.stdout.split("RESULT", 1)[1]) if res.status == "ok" and "RESULT" in res.stdout else [[] for _ in cand]
            with open(path, "a") as fh:
                for f, got in zip(cand, outs):
                    passed = int(got[0][1]) if got and got[0][0] == "#asserts" else 0
                    got = [pair for pair in got if pair[0] != "#asserts"]
                    if len(f["asserts"]) < 2 or passed < len(f["asserts"]):     # the model's own tests must exist and all pass
                        stats["own_tests_fail"] = stats.get("own_tests_fail", 0) + 1
                        continue
                    if len(got) < 3:
                        stats["no_tests"] += 1
                        continue
                    rng.shuffle(got)
                    rec = {"text": f["text"], "name": f["name"], "source": "Qwen2.5-Coder-1.5B-Instruct (local, llama.cpp)", "licence": "Apache-2.0 model; output unrestricted",
                           "kind": "synthetic", "topic": topic, "tests": ["assert %s == %s" % (c, v) for c, v in got[:4]],
                           "model_tests": f["asserts"]}
                    fh.write(json.dumps(rec) + "\n")
                    seen.add(f["text"])
                    kept += 1
        stats["kept"] = kept
        json.dump(stats, open(out_dir + "/stats.json", "w"), indent=1)
        if stats["replies"] % 10 == 0:
            print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 8089)
