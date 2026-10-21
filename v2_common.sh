# source this from a driver script, after setting NAME (e.g. NAME=baseline)
#   nohup bash v2_baseline.sh > /dev/null 2>&1 &
# Frozen protocol v2: Ada only, PYTHONNOUSERSITE=1, done-keys tied to a hash of the code, not to git.
set -u
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export EXPECTED_GPU="${EXPECTED_GPU:-NVIDIA RTX 6000 Ada Generation}"
mkdir -p logs
DATE="$(date +%Y%m%d_%H%M%S)"
LOGFILE="logs/${NAME}_${DATE}.log"
LOGPARSEDFILE="logs/${NAME}_${DATE}_parsed.log"
DONE="logs/done_${NAME}.txt"; touch "$DONE"
GIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
# working tree is usually dirty, so the key is a hash of the files that define the experiment
CODE_HASH=$(sha256sum variant_finetune.py poisoned_pretrain.py attack_utils.py repro_utils.py | sha256sum | cut -c1-8)

pick_gpu_ada() {   # UUID of the Ada card to use. CMLab rule: at most 50 GB of GPU per person across all workstations,
                   # counted by each card's full VRAM, i.e. ONE 48 GB card. If this user already has a process on a card,
                   # reuse that card (share it) instead of taking a second one; refuse if that card is not an Ada.
  local me held name
  me=$(id -u)
  held=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=', ' read -r pid uuid; do [ "$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$me" ] && echo "$uuid"; done | sort -u)
  if [ -n "$held" ]; then
    [ "$(echo "$held" | wc -l)" -gt 1 ] && echo "WARNING: you already hold more than one GPU ($(echo $held)); the CMLab limit is one" >&2
    held=$(echo "$held" | head -1)
    name=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', *' -v u="$held" '$1==u {print $2}')
    if [ "$name" = "$EXPECTED_GPU" ]; then echo "$held"; else echo "you already hold a non-Ada GPU ($held, $name); not taking a second card" >&2; fi
    return
  fi
  nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader,nounits \
    | awk -F', *' -v n="$EXPECTED_GPU" '$2==n {print $3, $1}' \
    | sort -n | head -1 | cut -d' ' -f2
}

run() {            # run <script.py> <title> <args...>
  local script="$1" title="$2"; shift 2
  local key="$title|$CODE_HASH"
  if grep -qxF "$key" "$DONE"; then echo "skip (done): $title" | tee -a "$LOGFILE"; return 0; fi
  local g; g=$(pick_gpu_ada); [ -n "$g" ] || { echo "no '$EXPECTED_GPU' on this machine" | tee -a "$LOGFILE"; return 1; }
  export CUDA_VISIBLE_DEVICES="$g"
  echo "===== $title | GPU=$g | code=$CODE_HASH | git=$GIT | $* =====" | tee -a "$LOGFILE"
  python "$script" "$@" 2>&1 | tee -a "$LOGFILE"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then echo "$key" >> "$DONE"; else echo "FAILED: $title" | tee -a "$LOGFILE"; fi
}

finish() {
  python parse_runs.py "$LOGFILE" > "$LOGPARSEDFILE"
  python parse_runs.py "$LOGFILE" --csv > "${LOGPARSEDFILE%.log}.csv"
  echo "===== Done =====" | tee -a "$LOGFILE"
}