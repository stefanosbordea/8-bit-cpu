#!/bin/sh
# The open model items of PLAN.md, one after the other, unattended (D-train-3: a model replaces the current one
# only if its loss on the body tokens of the validation text is lower; the test sets are run only for what is kept).
#   1. Tier A: a second QAT stage, and finer logit codes with a second stage, from the final float model
#   2. Tier B: the model trained on for 6,000 steps with the stream-range term, then QAT
#   3. tables, report and artifacts from the final files
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
PY=.venv/bin/python
LOG=results/logs/overnight.txt
say() { echo "$(date +%H:%M) $1" >> $LOG; }
run_qat() {
  name=$1; shift
  if [ ! -f runs/qat/$name/result.json ]; then
    rm -rf runs/qat/$name; mkdir -p runs/qat/$name
    $PY -W ignore -m lsllm.qat.train --out runs/qat/$name "$@" > runs/qat/$name/stdout.txt 2>&1
  fi
}
body() { $PY -c "import json; print(json.load(open('$1'))['val_body'])"; }

QA="--float runs/f/A3r/ckpt.pt --data data/tok/f512c2 --kd-weight 0.5 --batch-size 16 --device mps --eval-every 500 --prompt-weight 0.1"
run_qat A3rq2 $QA --steps 3000 --lr 0.005 --init-int runs/qat/A3rq
run_qat A3ru $QA --steps 3000 --lr 0.02 --logit-bits 2
run_qat A3ru2 $QA --steps 3000 --lr 0.005 --logit-bits 2 --init-int runs/qat/A3ru
BEST=$($PY -c "
import json, os
names = [n for n in ('A3rq', 'A3rq2', 'A3ru', 'A3ru2') if os.path.exists('runs/qat/%s/result.json' % n)]
print(min(names, key=lambda n: json.load(open('runs/qat/%s/result.json' % n))['val_body']))")
for N in A3rq A3rq2 A3ru A3ru2; do [ -f runs/qat/$N/result.json ] && say "Tier A int8 $N: body loss $(body runs/qat/$N/result.json)"; done
say "Tier A int8 kept: $BEST"

rm -rf runs/f/B2; mkdir -p runs/f/B2
$PY -W ignore -m lsllm.train.run --out runs/f/B2 --data data/tok/f512c2 --device mps --batch-size 64 --chunks 4 --steps 6000 --eval-every 1000 --init-from runs/f/B1/ckpt.pt --lr 6e-4 --prompt-weight 0.1 --stream-limit 3.5 --stream-weight 1.0 > runs/f/B2/stdout.txt 2>&1 &
TRAIN=$!
if [ "$BEST" != "A3rq" ]; then
  tools/adopt_tier_a.sh A3r $BEST data/tok/f512c2 >> $LOG 2>&1
  say "Tier A: $BEST adopted, report regenerated"
fi
wait $TRAIN
say "Tier B float B2: body loss $(body runs/f/B2/result.json) (B1: $(body runs/f/B1/result.json))"
run_qat B2q --float runs/f/B2/ckpt.pt --data data/tok/f512c2 --steps 3000 --kd-weight 0.5 --lr 0.02 --batch-size 16 --device mps --eval-every 500 --prompt-weight 0.1
say "Tier B int8 B2q: body loss $(body runs/qat/B2q/result.json) (B1q: $(body runs/qat/B1q/result.json))"
KEEP=$($PY -c "
import json
f = lambda p: json.load(open(p))['val_body']
print('yes' if f('runs/f/B2/result.json') < f('runs/f/B1/result.json') and f('runs/qat/B2q/result.json') < f('runs/qat/B1q/result.json') else 'no')")
say "Tier B: B2 replaces B1: $KEEP"
if [ "$KEEP" = "yes" ]; then
  $PY - <<'PY'
import json
p = "runs/final.json"
d = json.load(open(p))
for entry in d["models"]:
    if entry["label"] == "Tier B":
        entry["float"], entry["int"] = "runs/f/B2/ckpt.pt", "runs/qat/B2q"
json.dump(d, open(p, "w"), indent=1)
PY
  PYTHONPATH=. $PY tools/finalize.py B2q runs/f/B2/ckpt.pt data/tok/f512c2 >> $LOG 2>&1
  PYTHONPATH=. $PY -W ignore tools/tierb.py relu runs/qat/B2q/result.json > /dev/null 2>&1
fi
PYTHONPATH=. $PY -W ignore tools/report_tables.py runs/final.json > /dev/null 2>&1
PYTHONPATH=. $PY -W ignore tools/asks_table.py runs/final.json > /dev/null 2>&1
PYTHONPATH=. $PY -W ignore tools/report.py runs/final.json >> $LOG 2>&1
PYTHONPATH=. $PY tools/collect_artifacts.py runs/final.json > /dev/null 2>&1
cp runs/final.json artifacts/final.json
say "done"
