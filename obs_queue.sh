#!/bin/bash
# A queue of logged runs that keeps going while you read results and edit the queue.
#
#   nohup bash obs_queue.sh > /dev/null 2>&1 &     start (or restart: finished lines are skipped)
#   bash obs_queue.sh status                         what is done / running / waiting / failed, and the last epoch line of the running one
#   touch queue.stop                                 stop after the run that is going now (rm queue.stop before restarting)
#   pkill -f variant_finetune_obs.py                 abort the run that is going now; it is marked FAILED and the queue moves on
#
# Edit queue.txt at any time (the file is re-read before every run): add lines at the end, move lines up to run them sooner,
# put '#' in front of a line to skip it. One run per line:   title | extra arguments
# Common arguments (below) are put in front of the extra ones; a later argument wins, so a line can override --seed or --pz_path.
# A line counts as done when its title, its arguments and the code (variant_finetune_obs.py, obs_log.py, repro_utils.py, attack_utils.py) are
# the same as in a finished run. Change the arguments or the code and it runs again. An aborted/crashed line is not retried until you
# delete its entry from logs/failed_obsq.txt (or change the line).
# Everything goes to obs/ (logs) and obs/adapters/ (adapters); nothing under adapters/ is touched. Logs of the queue: logs/obsq_<date>.log
NAME=obsq; source ./v2_common.sh
OBS_HASH=$(sha256sum variant_finetune_obs.py obs_log.py repro_utils.py attack_utils.py | sha256sum | cut -c1-8)
CODE_HASH="$OBS_HASH"                       # v2_common.sh's run() builds its done-key from CODE_HASH; here it covers the files this queue uses
FAILED="logs/failed_obsq.txt"; touch "$FAILED"
QUEUE="${QUEUE:-queue.txt}"
PZ="${PZ:-pz_v2/badnet_s2/pytorch_model.bin}"
FS="${FS:-3}"
COMMON=(--pz_path "$PZ" --seed "$FS" --lr 2e-4 --svd_k 8 --use_spectral_rescaling --obs_log obs)

line_key() { echo "$1 #$(printf '%s' "$2" | sha256sum | cut -c1-6)"; }     # title plus a short hash of the arguments

next_line() {     # prints "title|args" of the first line that is neither done nor failed
  local t a k
  while IFS='|' read -r t a; do
    t=$(echo "$t" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'); a=$(echo "$a" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [ -z "$t" ] && continue; [[ "$t" == \#* ]] && continue
    k="$(line_key "$t" "$a")|$CODE_HASH"
    grep -qxF "$k" "$DONE" && continue
    grep -qxF "$k" "$FAILED" && continue
    echo "$t|$a"; return 0
  done < "$QUEUE"
  return 1
}

status() {
  local t a k s
  printf '%-8s %s\n' "STATE" "TITLE"
  while IFS='|' read -r t a; do
    t=$(echo "$t" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'); a=$(echo "$a" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [ -z "$t" ] && continue; [[ "$t" == \#* ]] && { printf '%-8s %s\n' "skipped" "${t#\#}"; continue; }
    k="$(line_key "$t" "$a")|$CODE_HASH"
    if grep -qxF "$k" "$DONE"; then s=done; elif grep -qxF "$k" "$FAILED"; then s=FAILED; elif [ "$(cat queue.cur 2>/dev/null)" = "$k" ]; then s=RUNNING; else s=waiting; fi
    printf '%-8s %s\n' "$s" "$t"
  done < "$QUEUE"
  if [ -f queue.cur ]; then
    local lf; lf=$(ls -t logs/obsq_2*.log 2>/dev/null | grep -v _parsed | head -1)
    [ -n "$lf" ] && { echo; echo "running: $(cut -d'|' -f1 queue.cur)"; tr '\r' '\n' < "$lf" | grep -E '^\[EPOCH\]' | tail -3; }
  fi
}
[ "${1:-}" = status ] && { status; exit 0; }

[ -f "$QUEUE" ] || { echo "no $QUEUE (copy queue.txt next to this script)"; exit 3; }
echo "obs_queue: code hash $OBS_HASH, PZ $PZ, finetune seed $FS, queue $QUEUE, log $LOGFILE" | tee -a "$LOGFILE"
idle=0
while true; do
  [ -e queue.stop ] && { echo "queue.stop found: stopping" | tee -a "$LOGFILE"; rm -f queue.cur; break; }
  nl=$(next_line) || { [ "$idle" = 0 ] && echo "queue empty: waiting for new lines in $QUEUE (touch queue.stop to quit)" | tee -a "$LOGFILE"; idle=1; sleep 60; continue; }
  idle=0
  title="${nl%%|*}"; args="${nl#*|}"
  key="$(line_key "$title" "$args")|$CODE_HASH"
  echo "$key" > queue.cur
  read -ra A <<< "$args"
  run variant_finetune_obs.py "$(line_key "$title" "$args")" "${COMMON[@]}" "${A[@]}"
  grep -qxF "$key" "$DONE" || { echo "$key" >> "$FAILED"; echo "marked FAILED (not retried): $title" | tee -a "$LOGFILE"; }
  rm -f queue.cur
done
