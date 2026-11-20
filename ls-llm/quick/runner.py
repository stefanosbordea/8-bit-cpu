"""One training job at a time (MPS memory allows only one). Jobs are lines of runs/jobs.txt: name|data dir|extra args.
Add lines while it runs; a job is skipped when runs/<name>/result.json exists. Stops when runs/STOP exists."""
import os
import subprocess
import time

ROOT = "/Users/kixem/Desktop/LS Architecture/ls-llm"
os.chdir(ROOT)


def busy():
    return subprocess.run(["pgrep", "-f", "quick.train"], capture_output=True).stdout.strip() != b""


while not os.path.exists("runs/STOP"):
    job = None
    for line in open("runs/jobs.txt"):
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) >= 2 and not parts[0].startswith("#") and not os.path.exists("runs/%s/result.json" % parts[0]):
            job = parts
            break
    if job is None or busy():
        time.sleep(10)
        continue
    name, data, extra = job[0], job[1], (job[2].split() if len(job) > 2 else [])
    os.makedirs("runs/" + name, exist_ok=True)
    cmd = [".venv/bin/python", "-m", "quick.train", "--out", "runs/" + name, "--data", data, "--device", "mps"] + extra
    with open("runs/%s.log" % name, "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0 and not os.path.exists("runs/%s/result.json" % name):
        fails = "runs/%s/fails" % name
        n = int(open(fails).read()) + 1 if os.path.exists(fails) else 1
        open(fails, "w").write(str(n))
        if n >= 8:
            open("runs/%s/result.json" % name, "w").write('{"failed": true}')
        time.sleep(30)
