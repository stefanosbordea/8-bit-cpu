#!/bin/sh
# Second stage of the shape search: the three best shapes inside the budget, 4,000 steps each on the final training
# text with the final recipe, and the chosen shape with an attention window of 128. One seed, same schedule.
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
C="--data data/tok/f512v2 --device mps --batch-size 64 --chunks 2 --steps 4000 --eval-every 1000 --n-heads 4 --n-kv-heads 1 --ffn relu --n-layers 2 --prompt-weight 0.1"
run() { name=$1; shift; if [ ! -f runs/s2/$name/result.json ]; then rm -rf runs/s2/$name; mkdir -p runs/s2/$name; .venv/bin/python -W ignore -m lsllm.train.run --out runs/s2/$name "$@" > runs/s2/$name/stdout.txt 2>&1; fi; }
run L2d160h4F219 $C --dim 160 --hidden-dim 219
run L2d176h4F142 $C --dim 176 --hidden-dim 142
run L2d144h4F301 $C --dim 144 --hidden-dim 301
run L2d160h4F219w128 $C --dim 160 --hidden-dim 219 --window 128
for d in runs/s2/*/; do .venv/bin/python -c "
import json
r = json.load(open('${d}result.json')); print('$d', 'val', round(r['val_loss'], 4), 'body', round(r['val_body'], 4))"; done
