#!/bin/bash
# Gate (b): determinism. The same command twice, observers ON, on the same Ada card.
# parse_runs.py --diff compares dev acc, [RNG] fingerprints, per-epoch test acc and ASR, and the SUMMARY line.
#   bash gate_b.sh              # 3 epochs, quick
#   EPOCHS=20 bash gate_b.sh    # full length; on PASS, run 1 is saved under golden/ as the v2 reference
#   CROSS=1 bash gate_b.sh      # run 2 on the other Ada card: checks the two cards behave identically
set -u
export PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export EXPECTED_GPU="${EXPECTED_GPU:-NVIDIA RTX 6000 Ada Generation}"
PZ="${PZ:-old_models/poisoned_roberta_large_badnet/pytorch_model.bin}"
EPOCHS="${EPOCHS:-3}"; SEED="${SEED:-0}"
OUT="logs/gateB_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT" golden

list_gpus_ada() {   # UUIDs of every card of the expected model, least-used first; never another model
  nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader,nounits \
    | awk -F', *' -v n="$EXPECTED_GPU" '$2==n {print $3, $1}' | sort -n | cut -d' ' -f2
}
mapfile -t GPUS < <(list_gpus_ada)
[ "${#GPUS[@]}" -ge 1 ] || { echo "no '$EXPECTED_GPU' found"; exit 1; }
G1="${GPUS[0]}"; G2="$G1"
if [ "${CROSS:-0}" = 1 ]; then      # CROSS=1: run 2 on a DIFFERENT physical card of the same model
  [ "${#GPUS[@]}" -ge 2 ] || { echo "CROSS=1 needs two '$EXPECTED_GPU' cards"; exit 1; }
  G2="${GPUS[1]}"
fi
echo "gate (b): run1 GPU=$G1, run2 GPU=$G2, epochs=$EPOCHS seed=$SEED pz=$PZ, logs in $OUT"

for i in 1 2; do
  echo "run $i ..."
  if [ "$i" = 1 ]; then export CUDA_VISIBLE_DEVICES="$G1"; else export CUDA_VISIBLE_DEVICES="$G2"; fi
  python variant_finetune.py --seed "$SEED" --num_epochs "$EPOCHS" --no_save --pz_path "$PZ" > "$OUT/run$i.log" 2>&1 \
    || { echo "FAIL: run $i crashed"; tail -5 "$OUT/run$i.log"; exit 1; }
done

n=$(tr '\r' '\n' < "$OUT/run1.log" | grep -c '^\[EPOCH\]')
real=$(tr '\r' '\n' < "$OUT/run1.log" | grep '^\[EPOCH\]' | grep -vc 'ASR=nan')
[ "$n" -eq "$EPOCHS" ] && [ "$real" -eq "$EPOCHS" ] || { echo "FAIL: expected $EPOCHS real [EPOCH] lines, got $n / $real"; exit 1; }

python parse_runs.py "$OUT/run1.log" "$OUT/run2.log" --diff | tee "$OUT/diff.txt"
if grep -q '^IDENTICAL' "$OUT/diff.txt"; then
  echo "GATE (b): PASS"
  if [ "$EPOCHS" -eq 20 ]; then
    GOLD="golden/baseline_${SEED}_oldpz_ep20_$(date +%Y%m%d).txt"
    tr '\r' '\n' < "$OUT/run1.log" | sed 's/^[[:space:]]*//' | grep -E '^\[ENV\]|^\[RNG\]|^\[EPOCH\]|^SUMMARY' > "$GOLD"
    echo "golden record written: $GOLD"
  fi
else
  echo "GATE (b): FAIL, see $OUT/diff.txt"; exit 1
fi