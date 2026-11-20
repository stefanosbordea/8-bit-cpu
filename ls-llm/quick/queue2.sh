#!/bin/sh
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
while [ ! -f runs/q1/DONE ]; do sleep 20; done
run() {
  name=$1; shift
  if [ ! -f runs/q1/$name/result.json ]; then
    .venv/bin/python -m quick.train --out runs/q1/$name --data data/tok/q512 --device mps --steps 1500 --eval-every 250 "$@" > runs/q1/$name.log 2>&1
  fi
}
run v8_cap --cap 1
run v9_cap_nonorm --cap 2
run v10_cap_h4kv2 --cap 1 --n_heads 4 --n_kv_heads 2
run v11_cap_h2kv2 --cap 1 --n_heads 2 --n_kv_heads 2
echo done > runs/q1/DONE2
