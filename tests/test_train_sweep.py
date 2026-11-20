"""The sweep runner: finished jobs are skipped, at most K runs at a time, one sweep at a time, summary table."""
import fcntl
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from lsllm.train import sweep


@pytest.fixture(autouse=True)
def own_lock(tmp_path, monkeypatch):
    """The tests must not take (or wait for) the project lock of a real sweep."""
    monkeypatch.setattr(sweep, "LOCK", str(tmp_path / "locks" / "sweep.lock"))


def write_jobs(tmp_path, jobs):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(jobs))
    return str(path)


def test_sweep_runs_skips_finished_jobs_and_reports_failures(tmp_path):
    for name, n in (("train", 3000), ("val", 600)):
        (np.arange(n) % 20).astype(np.uint16).tofile(tmp_path / f"{name}.bin")
    meta = {"vocab_size": 20, "train_tokens": 3000, "val_tokens": 600, "val_bytes": 1500}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    tiny = (f"--data '{tmp_path}' --steps 10 --eval-every 10 --batch-size 8 --max-seq-len 16 --dim 16 --n-layers 1 "
            "--n-heads 2 --n-kv-heads 1 --hidden-dim 32 --device cpu --threads 1")
    jobs = [{"name": "a", "args": tiny}, {"name": "b", "args": tiny + " --ffn relu"},
            {"name": "broken", "args": tiny + " --ffn nonsense"}]
    jobs_file, runs = write_jobs(tmp_path, jobs), str(tmp_path / "runs")

    codes = sweep.main([jobs_file, "--runs", runs, "--jobs", "2"])
    assert codes[:2] == [0, 0] and codes[2] != 0
    table = open(os.path.splitext(jobs_file)[0] + ".md").read().splitlines()
    assert len(table) == 5 and "| a | 2672 |" in table[2] and "done" in table[3] and "failed" in table[4]
    assert "AssertionError" in open(os.path.join(runs, "broken", "stdout.txt")).read()

    stamp = os.path.getmtime(os.path.join(runs, "a", "result.json"))
    os.remove(os.path.join(runs, "b", "result.json"))               # as if b had been interrupted
    sweep.main([jobs_file, "--runs", runs])
    assert os.path.getmtime(os.path.join(runs, "a", "result.json")) == stamp      # a was not run again
    assert os.path.exists(os.path.join(runs, "b", "result.json"))                 # b was


def test_never_more_than_k_jobs_at_once(tmp_path, monkeypatch):
    active, peak, guard = [0], [0], threading.Lock()

    def fake_run_job(job, runs, lock):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return 0

    monkeypatch.setattr(sweep, "run_job", fake_run_job)
    jobs_file = write_jobs(tmp_path, [{"name": f"job{g}", "args": ""} for g in range(9)])
    sweep.main([jobs_file, "--runs", str(tmp_path / "runs"), "--jobs", "3"])
    assert peak[0] == 3


def test_a_second_sweep_is_refused_while_one_is_running(tmp_path):
    os.makedirs(os.path.dirname(sweep.LOCK))
    with open(sweep.LOCK, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)                            # stands for the running sweep
        with pytest.raises(SystemExit):
            sweep.main([write_jobs(tmp_path, []), "--runs", str(tmp_path / "runs")])


def test_job_names_must_be_different(tmp_path):
    with pytest.raises(AssertionError):
        sweep.main([write_jobs(tmp_path, [{"name": "a", "args": ""}, {"name": "a", "args": ""}])])


def test_runs_keep_the_lock_when_the_sweep_itself_dies(tmp_path):
    os.makedirs(os.path.dirname(sweep.LOCK))
    lock = open(sweep.LOCK, "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"], pass_fds=[lock.fileno()])
    lock.close()                                                    # the sweep process is gone, its run is not
    with pytest.raises(SystemExit):
        sweep.main([write_jobs(tmp_path, []), "--runs", str(tmp_path / "runs")])
    orphan.wait()
    assert sweep.main([write_jobs(tmp_path, []), "--runs", str(tmp_path / "runs")]) == []
