#!/bin/bash
# run_pzscores.sh — rerun check 1 and the dose runs, now also writing every review's score
# (logit positive − logit negative) to logs/scores/. See PREREG_pzbias.md section 10.
# Training is unchanged and deterministic, so every [PROBE] line must equal the one in logs/pzbias_20261021_133652.log /
# logs/pzbias_20261021_141214.log; the scores add per-review detail to the very same models.
# The n=1500 runs also save the clean-dev pick (highest clean IMDB dev accuracy, first maximum) to
# pz_cleandev/cleandev_n1500_s<seed>/pytorch_model.bin (1.4 GB each, 7 GB in all) for run_persist_cleandev.sh.
# ONE card (CMLab rule). Check 1 ~10 min, each training run ~7.5 min.
# Recommended overnight chain (scores for n=1500, then the persistence test, then the remaining levels):
#   nohup bash -c 'LEVELS=1500 bash run_pzscores.sh && bash run_persist_cleandev.sh && EVAL=0 LEVELS="0 750 all 375 1125 1750" bash run_pzscores.sh' > /dev/null 2>&1 &
# SEEDS / LEVELS / EVAL=0 to run a subset; DRY=1 prints what would run.
NAME=pzscores; source ./v2_common.sh
SEEDS="${SEEDS:-0 1 2 3 4}"
LEVELS="${LEVELS:-1500 0 750 all 375 1125 1750}"
SC=logs/scores
if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2 | ${*:3}"; }; fi
stop_unless_done() { [ "${DRY:-0}" = 1 ] || grep -qxF "$1|$CODE_HASH" "$DONE" || { echo "'$1' failed, stopping (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1; }; }

# 1. check 1 again, with scores (未微調)
if [ "${EVAL:-1}" = 1 ]; then
  T="pzscores eval existing"
  run pz_bias_eval.py "$T" --out logs/pzscores_eval_existing.csv --score_dir $SC/existing \
    --model n1500_s0=pz_v2/badnet_s0 --model n1500_s1=pz_v2/badnet_s1 --model n1500_s2=pz_v2/badnet_s2 \
    --model n1500_s3=pz_v2/badnet_s3 --model n1500_s4=pz_v2/badnet_s4 \
    --model oldpz=old_models/poisoned_roberta_large_badnet \
    --model clean_old_0920=old_models/clean_roberta_large \
    --model clean_old_0930=old_models_v1/clean_roberta_large_imdb \
    --model v1_s0=old_models_v1/pz_seeds/badnet_s0 --model v1_s1=old_models_v1/pz_seeds/badnet_s1 \
    --model v1_s2=old_models_v1/pz_seeds/badnet_s2 --model v1_s3=old_models_v1/pz_seeds/badnet_s3 \
    --model v1_s42=old_models_v1/pz_seeds/badnet_s42
  stop_unless_done "$T"
  run pz_bias_eval.py "pzscores eval existing insent" --attack_tag insent --out logs/pzscores_eval_existing_insent.csv \
    --score_dir $SC/existing_insent \
    --model insent_old=old_models/poisoned_roberta_large_insent --model insent_0930=old_models_v1/poisoned_roberta_large_imdb_insent
fi

# 2. dose runs again, with scores; seed-outer
for s in $SEEDS; do
  for n in $LEVELS; do
    pn=$n; [ "$n" = all ] && pn=-1
    if [ "$n" = 1500 ]; then sv=(--save_rule cleandev --out_dir "pz_cleandev/cleandev_n1500_s$s"); else sv=(--no_save); fi
    T="pz scores n$n s$s"
    run poisoned_pretrain.py "$T" --attack_tag badnet --seed "$s" --poison_n "$pn" --score_dir "$SC/dose_n${n}_s${s}" "${sv[@]}"
    stop_unless_done "$T"
  done
done
[ "${DRY:-0}" = 1 ] || echo "===== Done =====" | tee -a "$LOGFILE"