#!/bin/sh
# tools/adopt_tier_a.sh FLOAT_NAME INT_NAME DATA
# Makes runs/f/FLOAT_NAME and runs/qat/INT_NAME the Tier A model of the report: manifest, float-against-integer
# comparison, accumulator bounds, all benchmark rows, export, speed documents, tables, report, artifacts.
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
FLOAT=$1
INT=$2
DATA=$3
.venv/bin/python - "$FLOAT" "$INT" <<'PY'
import json, sys
p = "runs/final.json"
d = json.load(open(p))
first = d["models"][0]
if first["float"] != "runs/f/%s/ckpt.pt" % sys.argv[1]:
    first["label"] = "Tier A, earlier model (10,000 steps; the editing sessions used it)"
    d["models"].insert(0, {"label": "Tier A", "float": "runs/f/%s/ckpt.pt" % sys.argv[1], "int": "runs/qat/%s" % sys.argv[2]})
else:
    first["int"] = "runs/qat/%s" % sys.argv[2]
json.dump(d, open(p, "w"), indent=1)
PY
PYTHONPATH=. .venv/bin/python tools/finalize.py $INT runs/f/$FLOAT/ckpt.pt $DATA --docs > results/logs/adopt_$INT.txt 2>&1
cp results/logs/compare_$INT.txt results/logs/compare_$FLOAT.txt
cp results/logs/overflow_$INT.txt results/logs/overflow_$FLOAT.txt
PYTHONPATH=. .venv/bin/python -W ignore tools/early_exit.py runs/f/$FLOAT/ckpt.pt > results/logs/early_exit_$FLOAT.txt 2>&1
PYTHONPATH=. .venv/bin/python -W ignore tools/report_tables.py runs/final.json > /dev/null 2>&1
PYTHONPATH=. .venv/bin/python -W ignore tools/asks_table.py runs/final.json > /dev/null 2>&1
PYTHONPATH=. .venv/bin/python -W ignore tools/report.py runs/final.json
PYTHONPATH=. .venv/bin/python tools/collect_artifacts.py runs/final.json | tail -6
tail -3 results/logs/adopt_$INT.txt
