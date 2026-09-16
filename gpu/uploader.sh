#!/bin/bash
# Push finished state files and logs to the private HF dataset every 15 minutes.
# State files are written atomically by extract_states.py, so directories can be uploaded as-is.
echo $$ > /root/upload.pid
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token)
REPO=andreayhchen/filler-token-facts-states
H=/root/venv/bin/hf
while true; do
  echo "$(date +%H:%M) uploading: $(ls /workspace/states/*.npz /root/states2/*.npz 2>/dev/null | wc -l) files"
  $H upload $REPO /workspace/states states --repo-type dataset --quiet 2>&1 | tail -1
  [ -d /root/states2 ] && $H upload $REPO /root/states2 states --repo-type dataset --quiet 2>&1 | tail -1
  $H upload $REPO /root/logs logs --repo-type dataset --quiet 2>&1 | tail -1
  if ! kill -0 $(cat /root/extract.pid 2>/dev/null) 2>/dev/null && ! kill -0 $(cat /root/monitor.pid 2>/dev/null) 2>/dev/null; then echo "$(date +%H:%M) phase 1 not running; final upload done"; break; fi
  sleep 900
done
