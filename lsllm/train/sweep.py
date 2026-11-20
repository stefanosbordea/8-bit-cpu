"""Run a list of training jobs, never more than K at the same time.

    .venv/bin/python -m lsllm.train.sweep jobs.json [--jobs 2] [--runs runs]

jobs.json is a list such as
    [{"name": "base", "args": "--data data/tok/boot512 --steps 2000"},
     {"name": "wide", "args": "--data data/tok/boot512 --steps 2000 --dim 96"}]
where `args` are arguments of lsllm.train.run. Job NAME writes into <runs>/NAME/ (console output in stdout.txt).
A job that already has a result.json is skipped, so a stopped sweep is simply started again (an unfinished
job then starts from its beginning). One lock file for the whole project keeps two sweeps from running at
the same time; the runs inherit the lock, so it stays taken until the last of them has ended.
The summary table is written next to the job list (jobs.md).
"""
import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".locks", "sweep.lock")


def run_job(job, runs, lock):
    """Train one job in its own process unless it has finished before. Returns the exit status (0 = fine)."""
    out = os.path.join(runs, job["name"])
    if os.path.exists(os.path.join(out, "result.json")):
        return 0
    os.makedirs(out, exist_ok=True)
    command = [sys.executable, "-m", "lsllm.train.run", "--out", out] + shlex.split(job["args"])
    with open(os.path.join(out, "stdout.txt"), "w") as console:
        return subprocess.run(command, stdout=console, stderr=subprocess.STDOUT, pass_fds=[lock.fileno()]).returncode


def summary(jobs, runs, codes):
    """A Markdown table with one row per job, read from the result.json files."""
    lines = ["| name | parameters | val loss (nats/token) | bits/byte | steps | tokens seen | wall s | tokens/s | device | status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for job, code in zip(jobs, codes):
        path = os.path.join(runs, job["name"], "result.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            lines.append(f"| {job['name']} | {r['params']} | {r['val_loss']:.4f} | {r['val_bpb']:.4f} | {r['steps']} | "
                         f"{r['tokens_seen']} | {r['wall_s']} | {r['tokens_per_s']} | {r['device']} | done |")
        else:
            lines.append(f"| {job['name']} | | | | | | | | | failed (exit status {code}) |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs_file")
    p.add_argument("--jobs", type=int, default=2, help="how many runs at the same time")
    p.add_argument("--runs", default="runs", help="directory that receives one folder per job")
    args = p.parse_args(argv)
    with open(args.jobs_file) as f:
        jobs = json.load(f)
    names = [job["name"] for job in jobs]
    assert len(set(names)) == len(names), "job names must be different"

    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another sweep (or one of its runs) is still running; start this one when it has ended")

    pool = ThreadPoolExecutor(max_workers=args.jobs)            # K worker threads, each waits for one run
    try:
        codes = list(pool.map(lambda job: run_job(job, args.runs, lock), jobs))
    finally:
        pool.shutdown(cancel_futures=True)                      # after Ctrl-C: start no further jobs
    table = summary(jobs, args.runs, codes)
    with open(os.path.splitext(args.jobs_file)[0] + ".md", "w") as f:
        f.write(table)
    print(table)
    return codes


if __name__ == "__main__":
    main()
