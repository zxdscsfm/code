#!/usr/bin/env bash
set -euo pipefail

echo "== parent and workers =="
for p in 4037569 4072891 4072897 4072913 4072916; do
  echo "PID:$p"
  if [ -d "/proc/$p" ]; then
    cat "/proc/$p/wchan" || true
    ps -p "$p" -o pid,ppid,stat,etime,%cpu,%mem,cmd
  else
    echo "missing"
  fi
  echo "---"
done

echo "== parent threads =="
ps -L -p 4037569 -o pid,tid,stat,psr,pcpu,comm,wchan:32 | head -n 50

echo "== parent gdb bt =="
gdb -batch -ex "thread apply all bt" -p 4037569 | head -n 160 || true

echo "== worker gdb bt =="
for p in 4072891 4072897 4072913 4072916; do
  echo "PID:$p"
  if [ -d "/proc/$p" ]; then
    gdb -batch -ex "bt" -p "$p" | head -n 80 || true
  else
    echo "missing"
  fi
  echo "---"
done
