#!/bin/sh
# One-screen status of the training queue: last log line of every run folder given as a pattern list.
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
date +%H:%M
for d in $(ls -d runs/x/* runs/p/* runs/q/* runs/s2/* runs/f/* runs/t/* 2>/dev/null); do
  if [ -f "$d/stdout.txt" ]; then
    printf "%-26s %s\n" "$(echo $d | sed 's|runs/||')" "$(grep -E '^step|^done' $d/stdout.txt | tail -1 | cut -c1-112)"
  fi
done | grep -E "${1:-.}"
memory_pressure | tail -1
df -h / | tail -1 | awk '{print "disk free " $4}'
