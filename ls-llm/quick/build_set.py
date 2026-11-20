"""Training set with mechanical augmentation of the documented functions.

Every human function: once as written, plus copies with renamed parameters and variables (lsllm.data.rename).
Every Qwen function that passes the quality checks: itself and copies with one number changed in docstring and
code together (lsllm.data.renumber; each such copy must run in the sandbox and pass the benchmark overlap rules
again), and of each of those: repeats and renamed copies. Tokenizer and validation set are those of BASE.

    python -m quick.build_set OUT --base f512c --human-copies 2 --number-variants 3 --qwen-originals 4 --qwen-copies 12
"""
import argparse
import ast
import json
import os
import random
import shutil
from pathlib import Path

from lsllm.alpha import alpha_key
from lsllm.data.corpus.overlap import load_benchmarks, overlap
from lsllm.data.rename import name_pool, rename
from lsllm.data.renumber import renumber
from lsllm.data.tokenize_corpus import encode_texts
from lsllm.sandbox import run_many
from lsllm.tok import Tok
from quick.qwen_gen import PROBES
from quick.synth import benchmark_keys

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--base", default="f512c")
p.add_argument("--qwen", nargs="*", help="files of Qwen functions (default: data/final/BASE/qwen_train.jsonl)")
p.add_argument("--human-copies", type=int, default=2)
p.add_argument("--human", choices=["documented", "self", "both"], default="documented",
               help="documented: the short documented functions of data/final/f512a; self: the self-contained functions "
                    "of the corpus (lsllm.data.selfcontained); both: self-contained ones with copies, the others once")
p.add_argument("--number-variants", type=int, default=3)
p.add_argument("--qwen-originals", type=int, default=4)
p.add_argument("--qwen-copies", type=int, default=12)
p.add_argument("--share", type=float, default=0.75, help="probability that a given name is replaced in a renamed copy")
p.add_argument("--no-csn2", action="store_true", help="leave out the second CodeSearchNet fetch")
p.add_argument("--plain-human", type=int, default=0, help="copies of every human function without its docstring (name and signature -> body)")
p.add_argument("--plain-simple", type=int, default=0, help="copies of every simple function without its docstring")
p.add_argument("--asks", help="file of verified human implementations of the request suite (data/asks/solutions.jsonl)")
p.add_argument("--asks-copies", type=int, default=0, help="how often every such implementation is repeated, as written")
p.add_argument("--asks-plain-copies", type=int, default=0, help="further repeats with the docstring removed (shorter to write)")
p.add_argument("--max-lines", type=int, default=40, help="longest self-contained human function, in lines")
args = p.parse_args()
out, base = Path("data/tok") / args.out, Path("data/tok") / args.base
read = lambda path: [json.loads(line) for line in open(path)]
rng = random.Random(0)


def sound(text):
    """False for the mechanical signs of a wrong function: code after a return, an argument that is never used,
    a constant result although there are arguments, an assignment to the function's own name."""
    fn = ast.parse(text).body[0]
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    if any(a.arg not in names for a in fn.args.args):
        return False
    for node in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and any(isinstance(st, (ast.Return, ast.Raise, ast.Break, ast.Continue)) for st in block[:-1]):
                return False
        if isinstance(node, ast.Name) and node.id == fn.name and isinstance(node.ctx, ast.Store):
            return False
    return True


# for every function: how many probe calls return without an exception (the sandbox keeps only 64 KB of output)
COUNT = """
import json
out = []
for src, name, probes in json.loads(%r):
    ns, good = {}, 0
    try:
        exec(src, ns)
    except Exception:
        out.append(0); continue
    for args in probes:
        try:
            eval("%%s(%%s)" %% (name, ", ".join(args)), ns); good += 1
        except Exception:
            pass
    out.append(good)
print("RESULT" + json.dumps(out))
"""


def runs(texts):
    """For every function text: True if it runs without an exception on at least one probe call, in the sandbox."""
    ok = []
    for g in range(0, len(texts), 400):
        part = texts[g:g + 400]
        jobs = []
        for text in part:
            fn = ast.parse(text).body[0]
            jobs.append([text, fn.name, [probe for probe in PROBES if len(probe) == len(fn.args.args)]])
        res = run_many([COUNT % json.dumps(jobs)], workers=1, timeout_s=120.0)[0]
        got = json.loads(res.stdout.split("RESULT", 1)[1]) if res.status == "ok" and "RESULT" in res.stdout else [0] * len(part)
        ok += [calls >= 1 for calls in got]
    return ok


from lsllm.data.nearcopy import NearCopies
near = NearCopies()                     # near-copies of benchmark solutions are left out of every part (tools/contamination.py)
documented = [r["text"] for r in read("data/final/f512a/human_train.jsonl")]
human, extra = documented, []
if args.human != "documented":
    from lsllm.data.selfcontained import self_contained
    bench_all = load_benchmarks()
    rows = [r for r in read("data/corpus/train.jsonl") if r["kind"] == "function"] + read("data/csn/train.jsonl")
    if os.path.exists("data/csn2/train.jsonl") and not args.no_csn2:        # the other 12 CodeSearchNet shards (fetched 2026-11-14)
        rows += read("data/csn2/train.jsonl")
    val_texts = {r["text"] for r in read("data/final/f512a/human_val.jsonl")} | {json.loads(line)["text"] for line in open("data/tok/val2/val_texts.jsonl")}
    human = list(dict.fromkeys(r["text"] for r in rows if r["n_lines"] <= args.max_lines and r["text"] not in val_texts and self_contained(r["text"])
                               and not overlap(r["name"], r["text"], alpha_key(r["text"]), bench_all)))
    if args.human == "both":
        mine = set(human)
        extra = [t for t in documented if t not in mine]
before = len(human) + len(extra)
human, extra = [t for t in human if not near(t)], [t for t in extra if not near(t)]
near_human = before - len(human) - len(extra)
qwen_rows = [r for path in (args.qwen or ["data/final/%s/qwen_train.jsonl" % args.base]) for r in read(path)]
val_keys = {alpha_key(r["text"]) for r in read("data/final/f512a/qwen_val.jsonl")}
qwen = list(dict.fromkeys(r["text"] for r in qwen_rows if sound(r["text"]) and alpha_key(r["text"]) not in val_keys))
count = {"qwen_in": len(qwen_rows), "qwen_sound_and_not_like_validation": len(qwen), "human_near_copies_removed": near_human}
qwen = [t for t in qwen if not near(t)]
count["qwen_near_copies_removed"] = count["qwen_sound_and_not_like_validation"] - len(qwen)

bench_names, bench_keys = benchmark_keys()
bench = load_benchmarks()
variants, seen = [], set(qwen)
for text in qwen:
    mine = 0
    for g in range(args.number_variants * 2):
        new = renumber(text, rng)
        if not new or new in seen:
            continue
        name, key = ast.parse(new).body[0].name, alpha_key(new)         # the name may have changed with the number
        if name in bench_names or key in bench_keys or key in val_keys or overlap(name, new, key, bench) or near(new):
            count["number_variants_removed_by_overlap_rules"] = count.get("number_variants_removed_by_overlap_rules", 0) + 1
            continue
        seen.add(new)
        variants.append(new)
        mine += 1
        if mine >= args.number_variants:
            break
kept = [v for v, ok in zip(variants, runs(variants)) if ok]
count.update(number_variants_made=len(variants), number_variants_that_run=len(kept))
simple = qwen + kept

pool = name_pool(human)
texts = list(human) + extra + simple * args.qwen_originals
made = {"human": 0, "simple": 0}
for kind, source, copies in (("human", human, args.human_copies), ("simple", simple, args.qwen_copies)):
    for text in source:
        mine = {text}
        for g in range(copies):
            new = rename(text, rng, pool, args.share)
            if new and new not in mine:
                mine.add(new)
                texts.append(new)
                made[kind] += 1
from lsllm.canon import canon


def without_docstring(text):
    """The same function without its docstring (None if it has none or nothing else is in the body)."""
    tree = ast.parse(text)
    body = tree.body[0].body
    if len(body) > 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        tree.body[0].body = body[1:]
        return canon(ast.unparse(tree))
    return None


for kind, source, copies in (("human", human, args.plain_human), ("simple", simple, args.plain_simple)):
    if copies:                      # a request gives only a name: the model must also write a body without a docstring
        added = 0
        for text in source:
            plain = without_docstring(text)
            if plain:
                texts.append(plain)
                added += 1
                for g in range(copies - 1):
                    texts.append(rename(plain, rng, pool, args.share) or plain)
        count["plain_%s_functions" % kind] = added
asks = []
if args.asks:
    for r in read(args.asks):
        if near(r["text"]) or overlap(r.get("original_name") or r["name"], r["text"], alpha_key(r["text"]), bench):
            count["asks_removed_by_overlap_rules"] = count.get("asks_removed_by_overlap_rules", 0) + 1
            continue
        asks.append(r["text"])
        texts += [r["text"]] * args.asks_copies
        plain = without_docstring(r["text"])
        texts += [plain or r["text"]] * args.asks_plain_copies
    count["asks_functions"] = len(asks)
rng.shuffle(texts)
out.mkdir(parents=True, exist_ok=True)
for name in ("tok.model", "tok.vocab", "val.bin"):
    shutil.copy(base / name, out / name)
tok = Tok(out / "tok.model")
size = encode_texts(tok, texts, out / "train.bin")
simple_tokens = sum(len(tok.encode(t, bos=True, eos=True)) for t in simple)
meta = {k: v for k, v in json.loads((base / "meta.json").read_text()).items() if k.startswith("val_") or k in ("vocab_size", "variant", "bytes_per_token")}
meta.update({"train_" + key: value for key, value in size.items()})
meta.update(count)
meta.update({"tokenizer": "copied from data/tok/" + args.base, "human": args.human, "human_functions": len(human),
             "human_functions_used_once": len(extra), "simple_functions": len(simple),
             "human_renamed_copies": made["human"], "simple_repeats": args.qwen_originals, "simple_renamed_copies": made["simple"],
             "rename_share": args.share, "name_pool": len(pool[0]),
             "simple_share_of_tokens_estimate": round(simple_tokens * (args.qwen_originals + made["simple"] / max(len(simple), 1)) / size["tokens"], 3)})
(out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
with open(out / "simple_functions.jsonl", "w") as f:
    for text in simple:
        f.write(json.dumps({"text": text, "number_variant": text not in set(qwen)}) + "\n")
print(json.dumps(meta, indent=1))
