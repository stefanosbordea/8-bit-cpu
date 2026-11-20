#!/bin/sh
# Runs after the long Tier A run and its follow-up (tools/after_long_run.sh) have finished: continues the Tier B model
# for 9,000 more steps and, by the same rule (lower body loss on the validation text), quantises and benchmarks it.
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
until grep -q " done$" results/logs/after_long_run.txt 2>/dev/null; do /bin/sleep 30; done
echo "f/B2|--data data/tok/f512c2 --device mps --batch-size 64 --chunks 4 --steps 9000 --eval-every 1000 --init-from runs/f/B1/ckpt.pt --lr 6e-4 --prompt-weight 0.1" >> runs/jobs2.txt
tools/after_long_run.sh B2 B1 data/tok/f512c2
