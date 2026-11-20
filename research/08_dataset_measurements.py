"""Measurements behind research/08_datasets_licences.md (standard library only, nothing is written to disk).

Run: .venv/bin/python -I research/08_dataset_measurements.py
Downloads about 45 MB into memory: ten pinned GitHub tarballs, HumanEval, MBPP, one CodeSearchNet shard
(HTTP range requests into the 940 MB zip) and bytes 0..5,999,999 of one codeparrot-clean shard.
Downloaded code is only parsed with ast, never executed.
"""
import ast, collections, gzip, io, json, re, tarfile, textwrap, urllib.request, warnings, zipfile, zlib

N = 10          # n-gram length in whitespace tokens (DeepSeek-Coder filters on 10-grams)
MAXLINES = 40   # longest function that counts as "kept"
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "cc0-1.0"}
REPOS = [  # (owner/repo, commit pinned on 2026-11-13)
    ("TheAlgorithms/Python", "30507bbe41397065728167fb09aa820f9fce8dc7"),
    ("keon/algorithms", "bd1d152f9ec7331c69be662831610b003b10fc7a"),
    ("exercism/python", "479f11b44b02337b0413e805b948491c2179615a"),
    ("more-itertools/more-itertools", "ceafee8ce96f4bc1eb74c528bc247c91d648a0bb"),
    ("pytoolz/toolz", "451af60dec590a6010e2babdbf391ea8f815122f"),
    ("mahmoud/boltons", "d31d821d56c8f7756af9c30a864077f2e5855321"),
    ("Suor/funcy", "a876c7e1cf11e7c746e91fce4632ebfa93865428"),
    ("dgilland/pydash", "b6bd031de47134b8e3c3977fbe65bf13cc51a205"),
    ("OmkarPathak/pygorithm", "cee38ed47e671171c3a30d12d5ec288f19c206d8"),
    ("qiyuangong/leetcode", "master"),
]
HUMANEVAL = "https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz"
MBPP = "https://raw.githubusercontent.com/google-research/google-research/f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81/mbpp/mbpp.jsonl"
CSN_ZIP = ("https://huggingface.co/datasets/code-search-net/code_search_net/resolve/"
           "fdc6a9e39575768c27eb8a2a5f702bf846eb4759/data/python.zip")
CSN_MEMBER = "python/final/jsonl/valid/python_valid_0.jsonl.gz"
CODEPARROT = ("https://huggingface.co/datasets/codeparrot/codeparrot-clean/resolve/"
              "35a59fb025bc0a102f7d96eac09d145b896d487b/file-000000000001.json.gz")


def get(url, headers=None):
    """Download a URL into memory."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


class HttpRangeFile:
    """Read-only seekable file over HTTP range requests, so zipfile can open one member of a remote zip."""
    BLOCK = 4 << 20

    def __init__(self, url):
        self.url, self.pos, self.buf, self.buf_start = url, 0, b"", 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as resp:
            self.size = int(resp.headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        lo = self.pos - self.buf_start
        if lo < 0 or lo + n > len(self.buf):
            end = min(self.pos + max(n, self.BLOCK), self.size) - 1
            self.buf, self.buf_start, lo = get(self.url, {"Range": f"bytes={self.pos}-{end}"}), self.pos, 0
        out = self.buf[lo:lo + n]
        self.pos += len(out)
        return out


def ngrams(text):
    toks = text.split()
    return {" ".join(toks[g:g + N]) for g in range(len(toks) - N + 1)}


def benchmark_index():
    """Map every benchmark 10-gram to the (kind, task id) pairs it comes from."""
    index = collections.defaultdict(set)
    for line in gzip.decompress(get(HUMANEVAL)).decode().splitlines():
        row = json.loads(line)
        doc = " ".join(a or b for a, b in re.findall(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', row["prompt"], flags=re.S))
        for gram in ngrams(doc):
            index[gram].add(("HE-doc", row["task_id"]))
        for gram in ngrams(row["canonical_solution"]):
            index[gram].add(("HE-sol", row["task_id"]))
    for line in get(MBPP).decode().splitlines():
        row = json.loads(line)
        if 11 <= row["task_id"] <= 510:  # the MBPP test split
            for gram in ngrams(row["text"]):
                index[gram].add(("MBPP-test-text", row["task_id"]))
            for gram in ngrams(row["code"]):
                index[gram].add(("MBPP-test-code", row["task_id"]))
    return index


def measure(name, docs, index):
    """docs yields (source text, is_single_function). Prints yield and benchmark-overlap counts."""
    c, hits, kept_bytes = collections.Counter(), collections.defaultdict(set), 0
    for src, single in docs:
        c["docs"] += 1
        c["bytes"] += len(src)
        toks = src.split()
        for g in range(len(toks) - N + 1):
            for kind, task in index.get(" ".join(toks[g:g + N]), ()):
                hits[kind].add(task)
        try:
            tree = ast.parse(textwrap.dedent(src) if single else src)
        except Exception:
            c["unparsable"] += 1
            continue
        lines = src.splitlines()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            c["top"] += 1
            doc = ast.get_docstring(node)
            c["top_doc"] += bool(doc)
            c["top_doctest"] += bool(doc and ">>>" in doc)
            if doc and node.end_lineno - node.lineno + 1 <= MAXLINES:
                c["kept"] += 1
                kept_bytes += len("\n".join(lines[node.lineno - 1:node.end_lineno]))
    print(f"{name}: docs={c['docs']} MB={c['bytes'] / 1e6:.2f} unparsable={c['unparsable']} top_level_defs={c['top']} "
          f"with_docstring={c['top_doc']} with_doctest={c['top_doctest']} kept={c['kept']} kept_MB={kept_bytes / 1e6:.2f} | "
          + " ".join(f"{k}={len(hits[k])}" for k in ("HE-doc", "HE-sol", "MBPP-test-text", "MBPP-test-code")))


def tar_files(repo, ref):
    with tarfile.open(fileobj=io.BytesIO(get(f"https://codeload.github.com/{repo}/tar.gz/{ref}")), mode="r:gz") as tf:
        for member in tf:
            if member.isfile() and member.name.endswith(".py"):
                yield tf.extractfile(member).read().decode("utf-8", "replace"), False


def main():
    warnings.simplefilter("ignore")  # SyntaxWarning from old escape sequences in downloaded code
    index = benchmark_index()
    print(f"benchmark {N}-grams indexed: {len(index)}; kept = top-level def with docstring, at most {MAXLINES} lines")
    for repo, ref in REPOS:
        measure(repo, tar_files(repo, ref), index)

    raw = get(CODEPARROT, {"Range": "bytes=0-5999999"})
    rows = [json.loads(line) for line in zlib.decompressobj(31).decompress(raw).split(b"\n")[:-1]]
    print("codeparrot-clean sample licences:", collections.Counter(r["license"] for r in rows).most_common())
    measure("codeparrot-clean bytes 0-5999999, permissive", ((r["content"], False) for r in rows if r["license"] in PERMISSIVE), index)
    measure("codeparrot-clean bytes 0-5999999, all", ((r["content"], False) for r in rows), index)

    zf = zipfile.ZipFile(HttpRangeFile(CSN_ZIP))
    print("CodeSearchNet python.zip members (MB stored):",
          {m.filename: round(m.compress_size / 1e6, 2) for m in zf.infolist() if not m.is_dir()})
    rows = [json.loads(line) for line in gzip.decompress(zf.open(CSN_MEMBER).read()).decode().splitlines()]
    plain = [r for r in rows if "." not in r["func_name"]]
    lens = sorted(len(r["code"].splitlines()) for r in rows)
    print(f"CodeSearchNet valid shard: rows={len(rows)} repos={len({r['repo'] for r in rows})} methods={len(rows) - len(plain)} "
          f"median_lines={lens[len(lens) // 2]} p90_lines={lens[int(len(lens) * 0.9)]} "
          f"share_le_20_lines={sum(n <= 20 for n in lens) / len(lens):.3f}")
    measure("CodeSearchNet valid shard, plain functions", ((r["original_string"], True) for r in plain), index)
    measure("CodeSearchNet valid shard, all rows", ((r["original_string"], True) for r in rows), index)


if __name__ == "__main__":
    main()
