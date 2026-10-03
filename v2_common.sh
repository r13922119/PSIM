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

pick_gpu_ada() {   # UUID of the least-used Ada (no memory threshold); never another model
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