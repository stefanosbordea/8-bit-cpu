import ast
import io
import json
import os
import tarfile
import types

import pytest

from lsllm.alpha import alpha_key
from lsllm.canon import canon
from lsllm.data.corpus import build
from lsllm.data.corpus.build import ALLOWED, REPOS, is_val, read_archive, select
from lsllm.data.corpus.extract import MARKERS
from lsllm.data.corpus.overlap import load_benchmarks, overlap

pytestmark = pytest.mark.filterwarnings("ignore::SyntaxWarning")
TRAIN, VAL, MANIFEST = "data/corpus/train.jsonl", "data/corpus/val.jsonl", "docs/corpus_manifest.json"


def record(text, kind="function"):
    node = ast.parse(text).body[0]
    return {"text": text, "name": node.name, "kind": kind, "has_docstring": ast.get_docstring(node) is not None}


def test_select_drops_overlaps_and_duplicates():
    bench = {"names": {"fib"}, "keys": {alpha_key("def g(a, b):\n    return a * b")}, "docs": {"Return the largest item."}}
    records = [
        record("def plain(x):\n    return x + 1", "method"),
        record("def plain(x):\n    return x + 1", "method"),                        # same text
        record("def renamed(y):\n    return y + 1"),                                # same code up to names
        record('def documented(z):\n    """Add one."""\n    return z + 1'),         # same code, the copy that is kept
        record("def fib(n):\n    return n"),                                        # benchmark entry point
        record("def times(p, q):\n    return p * q"),                               # benchmark solution up to names
        record('def top(xs):\n    """Return the\n    largest item."""\n    return max(xs)'),    # benchmark docstring
        record("def unique(x):\n    return x - 1"),
    ]
    kept, removed = select(records, bench)
    assert [rec["name"] for rec in kept] == ["documented", "unique"]
    assert removed == {"duplicate_text": 1, "duplicate_alpha": 2, "benchmark_name": 1, "benchmark_alpha": 1,
                       "benchmark_docstring": 1, "benchmark_any_rule": 3}


def test_split_depends_on_the_text_only():
    texts = ["def f%d():\n    return %d" % (g, g) for g in range(5000)]
    val = [text for text in texts if is_val({"text": text})]
    assert 50 <= len(val) <= 150                                    # about 2 percent
    assert val == [text for text in reversed(texts) if is_val({"text": text, "name": "x"})][::-1]


def make_tarball(path, files):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))


def test_read_archive(tmp_path):
    make_tarball(tmp_path / "repo.tar.gz", {
        "repo-abc123/pkg/mod.py": b"def double(x):\n    return 2 * x\n",
        "repo-abc123/pkg/tests/test_mod.py": b"def helper(x):\n    return x\n",
        "repo-abc123/pkg/_vendor/lib.py": b"def copied(x):\n    return x\n",
        "repo-abc123/pkg/big.py": b"def big(x):\n    return x\n" + b"#" * 2**20,
        "repo-abc123/docs/conf.py": b"def outside_prefix(x):\n    return x\n",
        "repo-abc123/pkg/notes.txt": b"def not_python(x):\n    return x\n"})
    records, counts = read_archive(tmp_path / "repo.tar.gz", "pkg/")
    assert [(rec["name"], rec["path"]) for rec in records] == [("double", "pkg/mod.py")]
    assert counts == {"file_read": 1, "extracted": 1, "file_test": 1, "file_vendored": 1, "file_big": 1}


def test_licence_gate_and_fetch(tmp_path, monkeypatch):
    answers = {"repos/good/repo": "MIT\tmain", "repos/good/repo/commits/main": "abc123", "repos/gpl/repo": "GPL-3.0\tmain",
               "repos/none/repo": "\tmain", "repos/odd/repo": "NOASSERTION\tmain", "repos/odd/repo/commits/main": "def456"}
    make_tarball(tmp_path / "served.tar.gz", {"repo-abc123/mod.py": b"def double(x):\n    return 2 * x\n"})
    monkeypatch.setattr(build, "gh", lambda path, field: answers[path])
    monkeypatch.setattr(build, "RAW", tmp_path)
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda url, timeout: open(tmp_path / "served.tar.gz", "rb"))

    assert build.fetch({"repo": "gpl/repo"}) == ({"rejected": "licence 'GPL-3.0' is not on the allowed list"}, [])
    assert build.fetch({"repo": "none/repo"}) == ({"rejected": "licence 'none found' is not on the allowed list"}, [])
    assert "rejected" in build.fetch({"repo": "odd/repo"})[0]
    # a licence read by hand counts only where GitHub answers NOASSERTION, and only if it is an allowed one
    assert "rejected" in build.fetch({"repo": "gpl/repo", "licence": "MIT", "verified": "read"})[0]
    assert "rejected" in build.fetch({"repo": "odd/repo", "licence": "GPL-3.0", "verified": "read"})[0]

    info, records = build.fetch({"repo": "good/repo"})
    assert (info["licence"], info["commit"], info["counts"]) == ("MIT", "abc123", {"file_read": 1, "extracted": 1})
    assert [(rec["name"], rec["source"], rec["licence"], rec["path"]) for rec in records] == [("double", "good/repo", "MIT", "mod.py")]
    info, records = build.fetch({"repo": "odd/repo", "licence": "BSD-3-Clause", "verified": "LICENSE read"})
    assert (info["licence"], info["verified"], records[0]["licence"]) == ("BSD-3-Clause", "LICENSE read", "BSD-3-Clause")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["served.tar.gz"]      # downloaded tarballs are deleted


def test_main_fetches_once_and_survives_a_kill(tmp_path, monkeypatch):
    rows = [{"repo": "a/one", "group": "algorithms"}, {"repo": "b/two"}, {"repo": "c/three", "rejected": "by hand"}]
    (tmp_path / "repos.json").write_text(json.dumps(rows))
    paths = {"REPOS": "repos.json", "OUT": "", "RAW": "raw", "ALL": "all.jsonl.gz", "FETCHED": "fetched.json",
             "MANIFEST": "manifest.json"}
    for name, value in paths.items():
        monkeypatch.setattr(build, name, tmp_path / value)
    monkeypatch.setattr(build.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=5 * 2**30))
    monkeypatch.setattr(build, "load_benchmarks", lambda: {"names": {"banned"}, "keys": set(), "docs": set()})
    calls = []

    def fake_fetch(row):
        calls.append(row["repo"])
        if row["repo"] == "b/two":
            return {"rejected": "licence 'GPL-3.0' is not on the allowed list"}, []
        texts = ["def f%d(x):\n    return x + %d" % (g, g) for g in range(60)] + ["def banned(x):\n    return x"]
        info = {"url": "https://github.com/a/one", "licence": "MIT", "verified": "API", "branch": "main",
                "commit": "abc", "date": "2026-11-13", "tarball_mb": 0.1, "seconds": 1.0, "counts": {"extracted": 61},
                "megabytes": 0.01}
        return info, [dict(record(text), n_lines=2, source="a/one", licence="MIT", path="m.py") for text in texts]

    monkeypatch.setattr(build, "fetch", fake_fetch)
    build.main()
    first = (tmp_path / "train.jsonl").read_text()
    # a killed run leaves a tarball and a half-written tail behind: both go away, and nothing is fetched twice
    (tmp_path / "raw" / "x.tar.gz").write_bytes(b"junk")
    with open(tmp_path / "all.jsonl.gz", "ab") as out:
        out.write(b"half a gzip member")
    build.main()
    assert calls == ["a/one", "b/two"] and not list((tmp_path / "raw").iterdir())
    assert (tmp_path / "train.jsonl").read_text() == first
    names = [json.loads(line)["name"] for name in ("train.jsonl", "val.jsonl") for line in open(tmp_path / name)]
    assert sorted(names) == sorted("f%d" % g for g in range(60))            # everything but the benchmark name
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [(repo["name"], repo["group"], repo["functions_kept"]) for repo in manifest["repositories"]] == [("a/one", "algorithms", 60)]
    assert manifest["rejected"] == [{"name": "c/three", "reason": "by hand"},
                                    {"name": "b/two", "reason": "licence 'GPL-3.0' is not on the allowed list"}]
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["function"]["functions"] == 60
    assert stats["removed_after_extraction"] == {"duplicate_alpha": 0, "benchmark_name": 1, "benchmark_any_rule": 1}


def test_a_fetched_repository_can_be_rejected_later(tmp_path, monkeypatch):
    rows = [{"repo": "a/one"}, {"repo": "d/four"}]
    (tmp_path / "repos.json").write_text(json.dumps(rows))
    paths = {"REPOS": "repos.json", "OUT": "", "RAW": "raw", "ALL": "all.jsonl.gz", "FETCHED": "fetched.json",
             "MANIFEST": "manifest.json"}
    for name, value in paths.items():
        monkeypatch.setattr(build, name, tmp_path / value)
    monkeypatch.setattr(build.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=5 * 2**30))
    monkeypatch.setattr(build, "load_benchmarks", lambda: {"names": set(), "keys": set(), "docs": set()})
    calls = []

    def fake_fetch(row):
        calls.append(row["repo"])
        tag = row["repo"][0]
        info = {"url": "https://github.com/" + row["repo"], "licence": "MIT", "verified": "API", "branch": "main",
                "commit": "abc", "date": "2026-11-13", "tarball_mb": 0.1, "seconds": 1.0, "counts": {"extracted": 3},
                "megabytes": 0.01}
        texts = ["def %s%d(x):\n    return x * %d + %d" % (tag, g, g, ord(tag)) for g in range(3)]
        return info, [dict(record(text), n_lines=2, source=row["repo"], licence="MIT", path="m.py") for text in texts]

    monkeypatch.setattr(build, "fetch", fake_fetch)
    build.main()
    rows[1]["rejected"] = "changed my mind"
    (tmp_path / "repos.json").write_text(json.dumps(rows))
    build.main()
    assert calls == ["a/one", "d/four"]                                     # no second download
    names = [json.loads(line)["name"] for name in ("train.jsonl", "val.jsonl") for line in open(tmp_path / name)]
    assert sorted(names) == ["a0", "a1", "a2"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [repo["name"] for repo in manifest["repositories"]] == ["a/one"]
    assert manifest["rejected"] == [{"name": "d/four", "reason": "changed my mind"}]


def test_repository_list():
    rows = json.loads(REPOS.read_text())
    names = [row["repo"] for row in rows]
    assert len(names) == len(set(names))
    for row in rows:
        if "licence" in row:                # a licence read by hand must be an allowed one and say how it was read
            assert row["licence"] in ALLOWED and row["verified"]


@pytest.mark.skipif(not os.path.exists(TRAIN), reason="corpus not built")
def test_built_corpus():
    bench = load_benchmarks()
    with open(MANIFEST) as handle:
        licences = {repo["name"]: repo["licence"] for repo in json.load(handle)["repositories"]}
    assert set(licences.values()) <= ALLOWED
    texts, n_val = set(), 0
    for path in (TRAIN, VAL):
        with open(path) as lines:
            for number, line in enumerate(lines):
                rec = json.loads(line)
                text = rec["text"]
                assert rec["kind"] in ("function", "method") and rec["path"].endswith(".py")
                assert rec["licence"] == licences[rec["source"]]
                assert 2 <= rec["n_lines"] == text.count("\n") + 1 <= 40 and len(text) <= 1500
                assert not any(marker in text for marker in MARKERS)
                assert rec["name"] not in bench["names"], rec["name"]
                assert text not in texts
                texts.add(text)
                assert is_val(rec) == (path == VAL)
                n_val += path == VAL
                if number % 20 == 0:        # the slow checks on every 20th function
                    assert overlap(rec["name"], text, alpha_key(text), bench) == []
                    assert canon(text) == text
                    node = ast.parse(text).body[0]
                    assert node.name == rec["name"] and (ast.get_docstring(node) is not None) == rec["has_docstring"]
    assert 0.015 < n_val / len(texts) < 0.025
