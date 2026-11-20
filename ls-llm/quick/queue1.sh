#!/bin/sh
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
run() {
  name=$1; shift
  if [ ! -f runs/q1/$name/result.json ]; then
    .venv/bin/python -m quick.train --out runs/q1/$name --data data/tok/q512 --device mps --steps 1500 --eval-every 250 "$@" > runs/q1/$name.log 2>&1
  fi
}
run v0_base
run v1_dyt --norm dyt
run v2_sigmoid --attn sigmoid
run v3_relu --ffn relu --hidden_dim 256
run v4_h4kv2 --n_heads 4 --n_kv_heads 2
run v5_h2kv1 --n_heads 2 --n_kv_heads 1
run v6_dyt_relu --norm dyt --ffn relu --hidden_dim 256
run v7_h2kv2 --n_heads 2 --n_kv_heads 2
echo done > runs/q1/DONE
