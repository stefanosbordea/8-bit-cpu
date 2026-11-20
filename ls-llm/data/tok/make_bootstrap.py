"""Build the bootstrap text set used to develop and test the tokenizer.

    .venv/bin/python data/tok/make_bootstrap.py

Source: the Python standard library that ships with the interpreter (PSF licence). The files are only parsed
with ast, never imported. Every function definition (module level, method or nested) is written on its own,
canonicalised, and kept if it has 2 to 40 lines and has not been seen before. One function in twenty goes to
the validation file. Output: data/tok/bootstrap/train.jsonl and val.jsonl (records as in INTERFACES section 3).
"""
import ast
import hashlib
import json
import sys
import sysconfig
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lsllm.canon import canon  # noqa: E402

FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)


def functions(stdlib):
    """Yield one record per usable function definition found under the directory `stdlib`."""
    seen = set()
    for path in sorted(stdlib.rglob("*.py")):
        if "site-packages" in path.parts:
            continue
        try:
            with tokenize.open(path) as f:
                tree = ast.parse(f.read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for parent in ast.walk(tree):
            for node in ast.iter_child_nodes(parent):
                if not isinstance(node, FUNCTION):
                    continue
                text = canon(ast.unparse(node))
                if text is None or not 2 <= text.count("\n") + 1 <= 40 or text in seen:
                    continue
                seen.add(text)
                yield {"text": text, "name": node.name, "source": "cpython-%s-stdlib" % sys.version.split()[0],
                       "licence": "PSF-2.0", "path": str(path.relative_to(stdlib)),
                       "kind": "method" if isinstance(parent, ast.ClassDef) else "function"}


def main():
    out = ROOT / "data/tok/bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    count = {"train": 0, "val": 0}
    with open(out / "train.jsonl", "w") as train, open(out / "val.jsonl", "w") as val:
        for record in functions(Path(sysconfig.get_paths()["stdlib"])):
            digest = hashlib.sha1(record["text"].encode()).hexdigest()
            split = "val" if int(digest, 16) % 20 == 0 else "train"
            (val if split == "val" else train).write(json.dumps(record) + "\n")
            count[split] += 1
    print("bootstrap set:", count)


if __name__ == "__main__":
    main()
