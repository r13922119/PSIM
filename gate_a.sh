#!/bin/bash
# Gate (a): observer neutrality on the real GPU.
#   bash gate_a.sh ft     # variant_finetune.py, baseline and Cl+Tr   (default, the faster part)
#   bash gate_a.sh pt     # poisoned_pretrain.py (3 epochs, observer only at new bests)
#   bash gate_a.sh ct     # clean_pretrain.py (3 epochs, observer = test acc only, at new bests)
#   bash gate_a.sh all
# For each config: run once with observers ON and once with OBSERVE=0, same Ada card, then compare the
# trainer-side lines ([RNG] fingerprints, dev acc, Tr debug lines, Cl hook count). They must be identical.
# Nothing is saved (--no_save). Logs go to logs/gateA_<date>/.
set -u
export PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export EXPECTED_GPU="${EXPECTED_GPU:-NVIDIA RTX 6000 Ada Generation}"
PZ="${PZ:-old_models/poisoned_roberta_large_badnet/pytorch_model.bin}"
EPOCHS="${EPOCHS:-3}"
WHAT="${1:-ft}"
OUT="logs/gateA_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
FAILS=0

pick_gpu_ada() {   # UUID of the least-used Ada (no memory threshold); empty only if the machine has no Ada
  nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader,nounits \
    | awk -F', *' -v n="$EXPECTED_GPU" '$2==n {print $3, $1}' \
    | sort -n | head -1 | cut -d' ' -f2
}
G=$(pick_gpu_ada); [ -n "$G" ] || { echo "no '$EXPECTED_GPU' found on this machine" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$G"
echo "gate (a) on GPU=$G, logs in $OUT"

canon() {   # trainer-side lines only; tqdm uses \r so split it first
  tr '\r' '\n' < "$1" | sed 's/^[[:space:]]*//' | grep -E '^\[RNG\]|^dev clean acc|^\[DEBUG_ep|^\[CHECK\]'
}
fail() { echo "  FAIL: $*"; FAILS=$((FAILS+1)); }

gate() {    # gate <name> <expected_epochs> <script.py> <args...>
  local name="$1" nep="$2" script="$3"; shift 3
  local on="$OUT/$name.on.log" off="$OUT/$name.off.log" ok=1
  echo "== $name"
  python "$script" "$@" > "$on" 2>&1;               [ $? -eq 0 ] || { fail "$name: run with observers ON crashed (see $on)"; tail -5 "$on"; return; }
  OBSERVE=0 python "$script" "$@" > "$off" 2>&1;    [ $? -eq 0 ] || { fail "$name: run with OBSERVE=0 crashed (see $off)"; tail -5 "$off"; return; }
  local n_on n_off; n_on=$(canon "$on" | grep -c '^\[RNG\]'); n_off=$(canon "$off" | grep -c '^\[RNG\]')
  [ "$n_on" -eq "$nep" ] && [ "$n_off" -eq "$nep" ] || { fail "$name: expected $nep [RNG] lines, got $n_on / $n_off"; ok=0; }
  # the test must not be vacuous: ON really observed, OFF really did not
  tr '\r' '\n' < "$on"  | grep -E '^\[EPOCH\]' | grep -qv 'nan' || { fail "$name: ON run printed no real observer values"; ok=0; }
  tr '\r' '\n' < "$off" | grep -E '^\[EPOCH\]' | grep -qv 'nan' && { fail "$name: OBSERVE=0 run still observed"; ok=0; }
  if ! diff <(canon "$on") <(canon "$off") > "$OUT/$name.diff"; then
    fail "$name: trainer-side lines differ (first lines of $OUT/$name.diff):"; head -8 "$OUT/$name.diff"; ok=0
  fi
  [ $ok -eq 1 ] && echo "  PASS ($n_on epochs, $(canon "$on" | wc -l) trainer-side lines identical)"
}

if [ -f test_observers.py ]; then
  echo "== CPU test"; python test_observers.py 2>&1 | tail -1 | grep -q PASS && echo "  PASS" || fail "CPU test_observers.py"
fi

if [ "$WHAT" = ft ] || [ "$WHAT" = all ]; then
  gate ft_base "$EPOCHS" variant_finetune.py --seed 0 --num_epochs "$EPOCHS" --no_save --pz_path "$PZ"
  gate ft_cltr "$EPOCHS" variant_finetune.py --seed 0 --num_epochs "$EPOCHS" --no_save --pz_path "$PZ" \
       --use_pretrained_dropout --use_orthogonal_penalty --svd_k 32
fi
if [ "$WHAT" = pt ] || [ "$WHAT" = all ]; then
  gate pt 3 poisoned_pretrain.py --seed 0 --no_save --out_dir pz_v2/gate_tmp
fi
if [ "$WHAT" = ct ] || [ "$WHAT" = all ]; then
  gate ct 3 clean_pretrain.py --seed 0 --no_save --out_dir pz_v2/gate_tmp_clean
fi

echo; [ $FAILS -eq 0 ] && echo "GATE (a): PASS" || echo "GATE (a): FAIL ($FAILS problem(s)), see $OUT"
exit $((FAILS>0))