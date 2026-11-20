#!/bin/sh
# Held-out check of edited models: tools/bench_edits.sh DIR... (greedy, with the syntax filter, test splits; the sessions never saw these tasks)
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
mkdir -p results/logs
for DIR in "$@"; do
  OUT=results/logs/edits_$(echo "$DIR" | tr '/:' '__')-test.txt
  for BENCH in custom mbpp humaneval; do
    .venv/bin/python -m lsllm.bench.run --completer "intf:$DIR" --bench $BENCH --split test --max-new-tokens 96 --record >> "$OUT" 2>&1
  done
  echo done >> "$OUT"
done
