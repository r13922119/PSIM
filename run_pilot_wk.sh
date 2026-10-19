#!/bin/bash
# run_pilot_wk.sh — pilot of the k-finder design on oldpz. See PREREG_pilot_wk.md.
#   1. kfinder.py finds the per-group k for this PZ (once; skipped if the json already exists): 6 groups = depth thirds x {query, value}
#   2. smoke test (1 epoch, not saved)
#   3. for each seed (3, then 4, then 5): Cl+Tr, baseline, Cl-only, Tr-only, all on top of W_pre,k; Tr uses svd_k = each module's k
#   nohup bash run_pilot_wk.sh > /dev/null 2>&1 &
# Time on one RTX 6000 Ada: kfinder ~5-8 min, smoke ~2 min, then ~46 min per seed (9 + 9 + 14 + 14) -> ~2.5 h in all.
# Each run is its own python call and logs its SUMMARY line when it ends, so stopping early keeps every finished run.
# SEEDS="3" or ARMS="cltr base" to run a subset. DRY=1 prints what would run.
NAME=pilot_wk; source ./v2_common.sh
PZ=old_models/poisoned_roberta_large_badnet/pytorch_model.bin
KJ=kstar/oldpz_grp6.json
LAM=10
SEEDS="${SEEDS:-3 4 5}"
ARMS="${ARMS:-cltr base cl tr}"
if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2 | ${*:3}"; }; fi

# 1. k-finder (before any fine-tuning)
mkdir -p kstar
[ -f "$KJ" ] || run kfinder.py "kfinder oldpz grp6" --pz "$PZ" --out "$KJ"
if [ "${DRY:-0}" != 1 ]; then
  [ -f "$KJ" ] || { echo "kfinder did not write $KJ, stopping" | tee -a "$LOGFILE"; exit 1; }
  python -c "import json;j=json.load(open('$KJ'));print('k per group:',j['k_final'],'| zero-shot truncated:',j['zero_shot_truncated'])" | tee -a "$LOGFILE"
fi
common=(--pz_path "$PZ" --lr 2e-4 --use_spectral_rescaling --wpre_kjson "$KJ")

# 2. smoke test: truncation of 48 modules, Cl hook count, Tr SVD with per-module k (1 epoch, nothing saved, no Pt)
SMOKE="smoke wkf-grp6 cl+tr lam$LAM kauto 1ep"
run variant_finetune.py "$SMOKE" --pz_path "$PZ" --lr 2e-4 --wpre_kjson "$KJ" --seed 3 \
  --use_pretrained_dropout --use_orthogonal_penalty --tr_lambda "$LAM" --svd_k_auto --num_epochs 1 --no_save
if [ "${DRY:-0}" != 1 ]; then
  grep -qxF "$SMOKE|$CODE_HASH" "$DONE" || { echo "smoke test failed, stopping (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1; }
  grep -qh "\[W_pre,k\] truncated 48" logs/${NAME}_2*.log || { echo "truncation line missing, stopping" | tee -a "$LOGFILE"; exit 1; }
fi

# 3. seed-outer: one seed through all arms, then the next seed
for fs in $SEEDS; do
  for arm in $ARMS; do
    case $arm in
      cltr) run variant_finetune.py "oldpz wkf-grp6 ft$fs cl+tr lam$LAM kauto lr2e-4" "${common[@]}" --seed "$fs" \
              --use_pretrained_dropout --use_orthogonal_penalty --tr_lambda "$LAM" --svd_k_auto ;;
      base) run variant_finetune.py "oldpz wkf-grp6 ft$fs base lr2e-4" "${common[@]}" --seed "$fs" ;;
      cl)   run variant_finetune.py "oldpz wkf-grp6 ft$fs cl lr2e-4"   "${common[@]}" --seed "$fs" --use_pretrained_dropout ;;
      tr)   run variant_finetune.py "oldpz wkf-grp6 ft$fs tr-only lam$LAM kauto lr2e-4" "${common[@]}" --seed "$fs" \
              --use_orthogonal_penalty --tr_lambda "$LAM" --svd_k_auto ;;
    esac
  done
done

[ "${DRY:-0}" = 1 ] || finish