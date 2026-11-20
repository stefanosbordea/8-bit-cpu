"""Everything that follows a trained float checkpoint, in order; a step whose output exists is skipped, so the
script can be run again after an interruption:

  1. quantization-aware training (through the GPU queue runs/jobs2.txt)      -> runs/qat/NAME/
  2. float against integer, token by token                                    -> results/logs/compare_NAME.txt
  3. accumulator use: worst case and largest value seen                       -> results/logs/overflow_NAME.txt
  4. benchmarks: float and int8, with and without the syntax filter, greedy and 10 samples, test and practice
  5. export for the FPGA                                                      -> export/NAME/
  6. speed and hardware-change documents (with --docs)                        -> docs/SPEED.md, docs/HARDWARE_CHANGES.md

    .venv/bin/python tools/finalize.py NAME runs/f/NAME/ckpt.pt data/tok/f512g --qat-steps 1500 --docs
"""
import argparse
import os
import subprocess
import time

ROOT = "/Users/kixem/Desktop/LS Architecture/ls-llm"
PY = ".venv/bin/python"
os.chdir(ROOT)
p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("ckpt")
p.add_argument("data")
p.add_argument("--qat-steps", type=int, default=1500)
p.add_argument("--qat-args", default="--kd-weight 0.5 --lr 0.02 --batch-size 16 --device mps --eval-every 250 --prompt-weight 0.1")
p.add_argument("--docs", action="store_true")
p.add_argument("--skip-bench", action="store_true")
args = p.parse_args()
os.makedirs("results/logs", exist_ok=True)
log = open("results/logs/finalize_%s.log" % args.name, "a")


def say(text):
    line = "%s %s" % (time.strftime("%H:%M:%S"), text)
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def run(cmd, out=None, env=None):
    say("run: " + cmd)
    with open(out, "w") if out else open(os.devnull, "w") as f:
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=".", **(env or {}))).returncode


int_dir = "runs/qat/" + args.name
if not os.path.exists(int_dir + "/result.json"):
    with open("runs/jobs2.txt", "a") as f:
        f.write("qat/%s|--float %s --data %s --steps %d %s\n" % (args.name, args.ckpt, args.data, args.qat_steps, args.qat_args))
    say("queued the quantization-aware training; waiting for " + int_dir + "/result.json")
    while not os.path.exists(int_dir + "/result.json"):
        time.sleep(20)
say("integer model: " + open(int_dir + "/log.txt").read().strip().split("\n")[-1])

if not os.path.exists("results/logs/compare_%s.txt" % args.name):
    run("%s -m lsllm.intref.compare --float %s --int %s --data %s" % (PY, args.ckpt, int_dir, args.data), "results/logs/compare_%s.txt" % args.name)
if not os.path.exists("results/logs/overflow_%s.txt" % args.name):
    run("%s -m lsllm.intref.overflow %s --data %s --windows 32" % (PY, int_dir, args.data), "results/logs/overflow_%s.txt" % args.name)

if not args.skip_bench:
    jobs = []
    for kind, target in (("torch", args.ckpt), ("int", int_dir)):
        for split in ("test", "practice"):
            done = "results/logs/%s.txt" % ("%s-%s-%s" % (kind, target, split)).replace("/", "_").replace(":", "_")
            if os.path.exists(done) and open(done).read().strip().endswith("done"):
                continue
            say("benchmarks: %s %s %s" % (kind, target, split))
            jobs.append(subprocess.Popen(["sh", "tools/bench_all.sh", kind, target, split]))
    for job in jobs:
        job.wait()
    say("benchmarks finished")

if not os.path.exists("export/%s/README.md" % args.name):
    code = run("%s -m lsllm.export %s export/%s --tokens 8 --temperature 0.8" % (PY, int_dir, args.name), "results/logs/export_%s.txt" % args.name)
    say("export: exit code %d" % code)
if args.docs:
    run("%s tools/speed_report.py %s" % (PY, int_dir), "results/logs/speed_%s.txt" % args.name)
say("finalize done")
