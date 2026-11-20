"""One GPU training job at a time, with the official trainer. Jobs: lines of runs/jobs2.txt: name|arguments of
lsllm.train.run. Add lines while it runs; a job with runs/<name>/result.json is skipped. Stops when runs/STOP2 exists."""
import os
import shlex
import subprocess
import time

os.chdir("/Users/kixem/Desktop/LS Architecture/ls-llm")


def busy():
    return subprocess.run(["pgrep", "-f", "lsllm.train.run|quick.train|lsllm.qat.train"], capture_output=True).stdout.strip() != b""


while not os.path.exists("runs/STOP2"):
    job = None
    for line in open("runs/jobs2.txt"):
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) == 2 and not parts[0].startswith("#") and not os.path.exists("runs/%s/result.json" % parts[0]):
            job = parts
            break
    if job is None or busy():
        time.sleep(10)
        continue
    name, args = job
    os.makedirs("runs/" + name, exist_ok=True)
    module = "lsllm.qat.train" if name.startswith("qat") else "lsllm.train.run"
    cmd = [".venv/bin/python", "-m", module, "--out", "runs/" + name] + shlex.split(args)
    with open("runs/%s/stdout.txt" % name, "a") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if code == 0 and os.path.exists("runs/%s/ckpt.pt" % name) and not name.startswith("qat"):
        subprocess.Popen(["sh", "tools/eval_ckpt.sh", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)    # practice-set scores, on the CPU
    if code != 0 and not os.path.exists("runs/%s/result.json" % name):
        fails = "runs/%s/fails" % name
        n = int(open(fails).read()) + 1 if os.path.exists(fails) else 1
        open(fails, "w").write(str(n))
        if n >= 6:
            open("runs/%s/result.json" % name, "w").write('{"failed": true}')
        time.sleep(30)
