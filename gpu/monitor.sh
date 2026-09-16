#!/bin/bash
# Heartbeat monitor for phase 1. The job rewrites /workspace/heartbeat after every prompt;
# this relaunches it only when the heartbeat is stale or the process is gone, kills by PID only
# (never by pattern), logs every check, and stops when all 1200 state files exist.
echo $$ > /root/monitor.pid
HB=/root/heartbeat
LOG=/root/logs/extract.log
STALE=900      # seconds; model load + first prompt take ~60 s
MAX_RESTARTS=5
restarts=0
launch() { setsid nohup /root/run_extract >> "$LOG" 2>&1 < /dev/null & sleep 2; }
launch
while true; do
  sleep 60
  n=$(ls /workspace/states/*.npz /root/states2/*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge 1200 ]; then echo "$(date +%H:%M) all $n state files present; done"; exit 0; fi
  pid=$(cat /root/extract.pid 2>/dev/null)
  alive=$(kill -0 "$pid" 2>/dev/null && echo yes || echo no)
  age=$(( $(date +%s) - $(stat -c %Y "$HB" 2>/dev/null || echo 0) ))
  echo "$(date +%H:%M) files=$n alive=$alive heartbeat_age=${age}s $(cat "$HB" 2>/dev/null)"
  if [ "$alive" = no ] || [ "$age" -gt "$STALE" ]; then
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then echo "$(date +%H:%M) giving up after $MAX_RESTARTS restarts"; exit 1; fi
    echo "$(date +%H:%M) RESTART $restarts (alive=$alive, age=${age}s)"
    kill "$pid" 2>/dev/null; sleep 15
    launch
  fi
done
