#!/bin/sh
ROOT="/Users/kixem/Desktop/LS Architecture/ls-llm"
PAT='\.venv/bin/python.*(lsllm|quick\.|pytest|ls-llm)|ls-llm/\.venv/bin/python|llama-server'
while true; do
  FREE=$(df -m / | tail -1 | awk '{print $4}')
  MEMFREE=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  if [ "$FREE" -lt 250 ] || [ "${MEMFREE:-50}" -lt 8 ]; then
    pkill -9 -f "$PAT" 2>/dev/null
    echo "- $(date '+%Y-%m-%d %H:%M') [guard] free disk ${FREE} MB, free memory ${MEMFREE}%: killed project python processes" >> "$ROOT/LOG.md"
    sleep 20
  fi
  sleep 4
done
