#!/bin/sh
# Every benchmark row of one model: tools/bench_all.sh KIND ARGUMENT [SPLIT]
# KIND: a completer kind without filter (torch, int, stories260k, ngram); the filtered kind KINDf is run too when it exists.
# Greedy pass@1, then 10 samples at temperature 0.8 (pass@1 and pass@10), on HumanEval, MBPP and the custom tasks.
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
KIND=$1
ARG=$2
SPLIT=${3:-test}
TOKENS=${TOKENS:-96}
OUT=results/logs/$(echo "$KIND-$ARG-$SPLIT" | tr '/:' '__').txt
mkdir -p results/logs
for K in $KIND ${KIND}f; do
  case "$K" in torchf|intf|torch|int|stories260k|ngram) ;; *) continue ;; esac
  for BENCH in custom mbpp humaneval; do
    if [ "$SPLIT" = practice ] && [ "$BENCH" = humaneval ]; then continue; fi
    .venv/bin/python -m lsllm.bench.run --completer "$K:$ARG" --bench $BENCH --split $SPLIT --max-new-tokens $TOKENS --record >> "$OUT" 2>&1
    if [ -z "$GREEDY_ONLY" ]; then
      .venv/bin/python -m lsllm.bench.run --completer "$K:$ARG" --bench $BENCH --split $SPLIT --max-new-tokens $TOKENS --n-samples 10 --temperature 0.8 --record >> "$OUT" 2>&1
    fi
  done
done
echo done >> "$OUT"
