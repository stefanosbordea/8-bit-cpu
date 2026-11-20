#!/bin/sh
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
NAME=$1
mkdir -p .locks
until mkdir .locks/eval.lock 2>/dev/null; do /bin/sleep 5; done
trap 'rmdir .locks/eval.lock' EXIT
for KIND in torch torchf; do
  if [ -f data/bench/custom_practice.jsonl ]; then
    .venv/bin/python -m lsllm.bench.run --completer $KIND:runs/$NAME/ckpt.pt --bench custom --split practice --max-new-tokens 96 --record >> runs/$NAME/eval.txt 2>&1
  fi
  .venv/bin/python -m lsllm.bench.run --completer $KIND:runs/$NAME/ckpt.pt --bench mbpp --split practice --max-new-tokens 96 --record >> runs/$NAME/eval.txt 2>&1
done
echo done > runs/$NAME/EVAL_DONE
