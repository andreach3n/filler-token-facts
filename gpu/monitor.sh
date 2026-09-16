#!/bin/bash
# Heartbeat monitor for a long torchrun job.
#   RUN=<launcher script that writes its PID to PIDFILE> HB=<heartbeat file> LOG=<log> DONE=<marker file>
# The job rewrites HB after every unit of work. This relaunches it only when the heartbeat is
# stale or the process is gone, kills by PID only (never by pattern), logs every check, and exits
# when the DONE marker exists.
RUN=${RUN:-/root/run_extract}
HB=${HB:-/root/heartbeat}
LOG=${LOG:-/root/logs/extract.log}
DONE=${DONE:-/root/extract.done}
PIDFILE=${PIDFILE:-/root/extract.pid}
STALE=${STALE:-900}      # seconds; model load + first unit take ~60 s
MAX_RESTARTS=${MAX_RESTARTS:-5}
echo $$ > "${PIDFILE%.pid}.monitor.pid"
restarts=0
launch() { setsid nohup "$RUN" >> "$LOG" 2>&1 < /dev/null & sleep 2; }
launch
while true; do
  sleep 60
  if [ -e "$DONE" ]; then echo "$(date +%H:%M) done marker present; exiting"; exit 0; fi
  pid=$(cat "$PIDFILE" 2>/dev/null)
  alive=$(kill -0 "$pid" 2>/dev/null && echo yes || echo no)
  age=$(( $(date +%s) - $(stat -c %Y "$HB" 2>/dev/null || echo 0) ))
  echo "$(date +%H:%M) alive=$alive heartbeat_age=${age}s $(cat "$HB" 2>/dev/null)"
  if [ "$alive" = no ] || [ "$age" -gt "$STALE" ]; then
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then echo "$(date +%H:%M) giving up after $MAX_RESTARTS restarts"; exit 1; fi
    echo "$(date +%H:%M) RESTART $restarts (alive=$alive, age=${age}s)"
    kill "$pid" 2>/dev/null; sleep 15
    launch
  fi
done
