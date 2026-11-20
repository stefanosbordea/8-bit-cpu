"""Build the real-code corpus from the repositories listed in repos.json (next to this file).

    .venv/bin/python -m lsllm.data.corpus.build

Step 1, fetch: every listed repository that was not fetched before is checked on GitHub (licence, head commit;
read-only requests through the `gh` tool), downloaded as one tarball and read file by file. Its functions are
appended to data/corpus/all_functions.jsonl.gz and the tarball is deleted. Nothing from a tarball is unpacked to
disk or executed. A row of repos.json is {"repo": "OWNER/NAME", "group": ...} plus, where needed, "prefix" (only
files under this folder), "branch" (a branch or commit other than the head of the default branch), "licence" and
"verified" (a licence read by hand, used when GitHub answers NOASSERTION) or "rejected" (why it is not used;
adding it to a repository that was already fetched removes its functions at the next run).
Step 2, select: benchmark overlaps and duplicates are dropped from all extracted functions; the rest is written to
data/corpus/train.jsonl and val.jsonl, with stats.json and docs/corpus_manifest.json. Step 2 runs every time, so
new benchmark files or new rows in repos.json only need another run; nothing is downloaded twice.
"""
import collections
import gzip
import hashlib
import http.client
import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
import warnings
from pathlib import Path

from lsllm.alpha import alpha_key
from lsllm.data.corpus.extract import extract, skip_path
from lsllm.data.corpus.overlap import load_benchmarks, overlap

ALLOWED = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0", "Unlicense", "CC0-1.0", "0BSD", "Zlib"}
REPOS = Path(__file__).with_name("repos.json")
OUT = Path("data/corpus")
RAW = Path("data/raw")
ALL = OUT / "all_functions.jsonl.gz"    # every extracted function of every fetched repository, before step 2
FETCHED = OUT / "fetched.json"          # per fetched repository: licence, commit, date, extraction counts
MANIFEST = Path("docs/corpus_manifest.json")
MAX_ARCHIVE = 100 * 10**6               # bytes; a larger tarball is abandoned
MAX_FILE = 2**20                        # bytes; larger .py files are generated code or data tables
MIN_FREE = 1.2 * 2**30                  # bytes of free disk required before a download (project floor: 1.0 GB)
MAX_EXTRACTED_MB = 70                   # stop adding repositories here (step 2 kept 66 MB of 71 MB extracted)
VAL_ONE_IN = 50                         # about 2 percent of the functions go to the validation file


def gh(path, field):
    """One read-only (GET) request to the GitHub API through the logged-in `gh` tool; returns one field as text."""
    done = subprocess.run(["gh", "api", path, "--jq", field], capture_output=True, text=True, check=True)
    return done.stdout.strip("\n")


def read_archive(archive, prefix=""):
    """Read the .py files of a GitHub tarball whose path starts with `prefix`. Returns (records, counts)."""
    counts, records = collections.Counter(), []
    with tarfile.open(archive) as tar:
        for member in tar:
            path = member.name.partition("/")[2]        # drop the "REPO-COMMIT/" folder that GitHub adds
            if not (member.isfile() and path.endswith(".py") and path.startswith(prefix)):
                continue
            reason = skip_path(path) or ("big" if member.size > MAX_FILE else None)
            if reason:
                counts["file_" + reason] += 1
                continue
            counts["file_read"] += 1
            for rec in extract(tar.extractfile(member).read(), counts):
                records.append(dict(rec, path=path))
    return records, counts


def fetch(row):
    """Check, download and read one repository. Returns (info, records); info has "rejected" if it is not used."""
    repo = row["repo"]
    licence, branch = gh("repos/" + repo, '[.license.spdx_id // "", .default_branch] | @tsv').split("\t")
    verified = "licence field of the GitHub API (gh api repos/%s)" % repo
    if licence == "NOASSERTION" and "licence" in row:   # GitHub could not classify the file: it was read by hand
        licence, verified = row["licence"], row["verified"]
    if licence not in ALLOWED:
        return {"rejected": "licence %r is not on the allowed list" % (licence or "none found")}, []
    branch = row.get("branch", branch)
    commit = gh("repos/%s/commits/%s" % (repo, branch), ".sha")
    archive = RAW / (repo.replace("/", "__") + ".tar.gz")
    start = time.time()
    try:
        with urllib.request.urlopen("https://codeload.github.com/%s/tar.gz/%s" % (repo, commit), timeout=120) as resp:
            with open(archive, "wb") as out:
                while out.tell() <= MAX_ARCHIVE and (chunk := resp.read(2**20)):
                    out.write(chunk)
        size = archive.stat().st_size
        if size > MAX_ARCHIVE:
            return {"rejected": "tarball larger than 100 MB", "seconds": round(time.time() - start, 1)}, []
        records, counts = read_archive(archive, row.get("prefix", ""))
    finally:
        archive.unlink(missing_ok=True)
    info = {"url": "https://github.com/" + repo, "licence": licence, "verified": verified, "branch": branch,
            "commit": commit, "date": time.strftime("%Y-%m-%d"), "tarball_mb": round(size / 1e6, 1),
            "seconds": round(time.time() - start, 1), "counts": dict(counts),
            "megabytes": round(sum(len(rec["text"]) for rec in records) / 1e6, 2)}
    return info, [dict(rec, source=repo, licence=licence) for rec in records]


def select(records, bench):
    """Drop benchmark overlaps, then duplicates (same text, or same code up to names). Returns (kept, removed)."""
    removed = collections.Counter()
    texts, best = set(), {}             # best: alpha key -> the record kept for it
    for rec in records:
        key = alpha_key(rec["text"])
        rules = overlap(rec["name"], rec["text"], key, bench)
        for rule in rules:
            removed["benchmark_" + rule] += 1
        if rules:
            removed["benchmark_any_rule"] += 1
        elif rec["text"] in texts:
            removed["duplicate_text"] += 1
        else:
            texts.add(rec["text"])
            removed["duplicate_alpha"] += key in best
            if key not in best or _rank(rec) < _rank(best[key]):
                best[key] = rec
    return list(best.values()), removed


def _rank(rec):
    """Among functions with the same alpha key, keep a top-level function before a method, a docstring before none."""
    return rec["kind"] != "function", not rec["has_docstring"]


def _digest(rec):
    return hashlib.sha1(rec["text"].encode()).hexdigest()


def is_val(rec):
    """True for about one function in VAL_ONE_IN, decided by the text alone, so the split never changes."""
    return int(_digest(rec)[:8], 16) % VAL_ONE_IN == 0


def _size(records):
    return {"functions": len(records), "megabytes": round(sum(len(rec["text"]) for rec in records) / 1e6, 2),
            "with_docstring": sum(rec["has_docstring"] for rec in records)}


def write_outputs(kept, removed, fetched, rows):
    """Write train.jsonl, val.jsonl, stats.json and the manifest; return the statistics."""
    kept.sort(key=_digest)              # a fixed, well-mixed order
    with open(OUT / "train.tmp", "w") as train, open(OUT / "val.tmp", "w") as val:
        for rec in kept:
            (val if is_val(rec) else train).write(json.dumps(rec) + "\n")
    (OUT / "train.tmp").replace(OUT / "train.jsonl")    # a killed run never leaves half a file under the real name
    (OUT / "val.tmp").replace(OUT / "val.jsonl")
    by_hand = {row["repo"] for row in rows if "rejected" in row}
    used = {repo: info for repo, info in fetched.items() if "rejected" not in info and repo not in by_hand}
    extraction = collections.Counter()
    for info in used.values():
        extraction.update(info["counts"])
    chars = sorted(len(rec["text"]) for rec in kept)
    stats = {
        "note": "megabytes = 10^6 characters of canonical text (ASCII, so also bytes)",
        "all": _size(kept),
        "function": _size([rec for rec in kept if rec["kind"] == "function"]),
        "method": _size([rec for rec in kept if rec["kind"] == "method"]),
        "train": _size([rec for rec in kept if not is_val(rec)]),
        "val": _size([rec for rec in kept if is_val(rec)]),
        "functions_by_n_lines": dict(sorted(collections.Counter(rec["n_lines"] for rec in kept).items())),
        "chars": {"median": chars[len(chars) // 2], "p90": chars[len(chars) * 9 // 10], "max": chars[-1]},
        "extraction": dict(extraction),
        "removed_after_extraction": dict(removed),
        "fetch_minutes": round(sum(info.get("seconds", 0) for info in fetched.values()) / 60, 1),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    per_repo = collections.Counter(rec["source"] for rec in kept)
    groups = {row["repo"]: row.get("group") for row in rows}
    manifest = {
        "repositories": [{"name": repo, "group": groups.get(repo), "url": info["url"], "licence": info["licence"],
                          "verified": info["verified"], "branch": info["branch"], "commit": info["commit"],
                          "date": info["date"], "functions_extracted": info["counts"].get("extracted", 0),
                          "functions_kept": per_repo[repo]}
                         for repo, info in used.items()],
        "rejected": [{"name": row["repo"], "reason": row["rejected"]} for row in rows if "rejected" in row]
                    + [{"name": repo, "reason": info["rejected"]} for repo, info in fetched.items() if "rejected" in info],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return stats


def main():
    warnings.simplefilter("ignore", SyntaxWarning)      # old code has many "\d" in plain strings
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    rows = json.loads(REPOS.read_text())
    fetched = json.loads(FETCHED.read_text()) if FETCHED.exists() else {}
    for stale in RAW.glob("*.tar.gz"):                  # left behind if an earlier run was killed
        stale.unlink()
    with open(ALL, "ab") as out:                        # and drop functions that were appended but not recorded
        out.truncate(max([info.get("end", 0) for info in fetched.values()], default=0))
    for row in rows:
        repo = row["repo"]
        if repo in fetched or "rejected" in row:
            continue
        extracted = sum(info.get("megabytes", 0) for info in fetched.values())
        if extracted >= MAX_EXTRACTED_MB or shutil.disk_usage("/").free < MIN_FREE:
            print("stopped fetching at %s: %.1f MB extracted, %.2f GB of disk free"
                  % (repo, extracted, shutil.disk_usage("/").free / 2**30))
            break
        try:
            info, records = fetch(row)
        except (OSError, subprocess.CalledProcessError, http.client.HTTPException, tarfile.TarError) as err:
            print(repo, "failed and will be tried again on the next run:", err)
            continue
        with open(ALL, "ab") as out:    # one gzip member per repository, written in one go
            out.write(gzip.compress("".join(json.dumps(rec) + "\n" for rec in records).encode()))
            info["end"] = out.tell()
        fetched[repo] = info
        FETCHED.with_suffix(".tmp").write_text(json.dumps(fetched, indent=1) + "\n")
        FETCHED.with_suffix(".tmp").replace(FETCHED)
        print(repo, info.get("rejected") or "%d functions, %.1f MB tarball, %.0f s"
              % (len(records), info["tarball_mb"], info["seconds"]), flush=True)
    if shutil.disk_usage("/").free < MIN_FREE:
        raise SystemExit("less than 1.2 GB of free disk: train.jsonl and val.jsonl were not rewritten")
    by_hand = {row["repo"] for row in rows if "rejected" in row}    # a fetched repository can still be rejected
    with gzip.open(ALL, "rt") as lines:
        records = (rec for rec in map(json.loads, lines) if rec["source"] not in by_hand)
        kept, removed = select(records, load_benchmarks())
    if not kept:
        raise SystemExit("no functions were extracted: nothing to write")
    stats = write_outputs(kept, removed, fetched, rows)
    for key in ("function", "method", "train", "val", "removed_after_extraction", "fetch_minutes"):
        print(key, stats[key])
    print("free disk: %.2f GB" % (shutil.disk_usage("/").free / 2**30))


if __name__ == "__main__":
    main()
