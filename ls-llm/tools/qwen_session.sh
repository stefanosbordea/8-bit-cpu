#!/bin/sh
# One generation session with the local Qwen2.5-Coder-1.5B server on the GPU: tools/qwen_session.sh MINUTES TAG
# Starts llama-server with 4 slots, 3 generator processes (quick.qwen_gen, folders data/qwen/<TAG>1..3) and quick.qwen_asks,
# stops everything after MINUTES. Do not run it while a training job uses the GPU.
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
MINUTES=${1:-60}
TAG=${2:-v}
L=/Volumes/CARMIX-WORK/ls-llm-work/llama/llama-b12361
nohup $L/llama-server -m /Volumes/CARMIX-WORK/ls-llm-work/models/qwen25-coder-1.5b-q4km.gguf -c 8192 -np 4 -ngl 99 --port 8089 --log-disable > /dev/null 2>&1 &
SERVER=$!
/bin/sleep 25
for N in 1 2 3; do
  nohup .venv/bin/python -W ignore -m quick.qwen_gen data/qwen/$TAG$N 1000000 8089 > data/qwen/$TAG$N.log 2>&1 &
done
nohup .venv/bin/python -W ignore -m quick.qwen_asks 10 8089 > data/qwen/${TAG}_asks.log 2>&1 &
/bin/sleep $((MINUTES * 60))
pkill -f "quick.qwen_gen"
pkill -f "quick.qwen_asks"
kill $SERVER
/bin/sleep 2
for N in 1 2 3; do echo "$TAG$N $(wc -l < data/qwen/$TAG$N/all.jsonl 2>/dev/null) kept; $(tr -d '\n ' < data/qwen/$TAG$N/stats.json 2>/dev/null)"; done
