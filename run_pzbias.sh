#!/bin/bash
# run_pzbias.sh — why do our PZs call clean negatives positive? See PREREG_pzbias.md.
#   1. eval only (check 1): per-class clean errors and ASR of every existing RoBERTa-large pretrained checkpoint
#      (16: 5 fresh BadNet PZs, oldpz, 6 older-protocol BadNet PZs, 2 InSent PZs, 2 older clean models)   ~10 min
#   2. train (checks 2+3): poisoned pretraining with n poisoned negatives, n in {0 1500 750 all 375 1125 1750},
#      seeds 0-4, seed-outer (one seed through all n, then the next seed). n=0 is the clean model.
#      n=1500 is a rerun of pz_v2/badnet_s* and must be bit-identical to logs/pz_2026100{2,3}_*.log.
#   nohup bash run_pzbias.sh > /dev/null 2>&1 &
# Time on one RTX 6000 Ada: ~7 min per training run (the old PZ runs took ~5.5 min; the per-epoch probe adds ~1),
# 35 runs -> ~4 h. Two GPUs: start  SEEDS="0 1" bash run_pzbias.sh  and, a minute later,
# EVAL=0 SEEDS="2 3 4" bash run_pzbias.sh  (otherwise both may land on the same GPU).
# Saving (which epoch to keep as a PZ is still open, see PREREG section 8):
#   SAVE=0     (default) nothing saved; every epoch is measured in the log anyway, and any run can be recreated (6 min)
#   SAVE=all   every epoch of every run, incl. n=1500: pz_dose/n<n>_s<seed>/epNN/, 35 runs x 3 x 1.4 GB = about 147 GB
#   SAVE=best  only the epoch PSIM's rule picks (the old behaviour), n=1500 skipped: about 42 GB
# SEEDS / LEVELS to run a subset; DRY=1 prints what would run.
NAME=pzbias; source ./v2_common.sh
SEEDS="${SEEDS:-0 1 2 3 4}"
LEVELS="${LEVELS:-0 1500 750 all 375 1125 1750}"   # coarse to fine, so a stopped run still spans the range
SAVE="${SAVE:-0}"
if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2 | ${*:3}"; }; fi

# 1. eval only, existing checkpoints (未微調)
if [ "${EVAL:-1}" = 1 ]; then
  run pz_bias_eval.py "pzbias eval existing" --out logs/pzbias_eval_existing.csv \
    --model n1500_s0=pz_v2/badnet_s0 --model n1500_s1=pz_v2/badnet_s1 --model n1500_s2=pz_v2/badnet_s2 \
    --model n1500_s3=pz_v2/badnet_s3 --model n1500_s4=pz_v2/badnet_s4 \
    --model oldpz=old_models/poisoned_roberta_large_badnet \
    --model clean_old_0920=old_models/clean_roberta_large \
    --model clean_old_0930=old_models_v1/clean_roberta_large_imdb \
    --model v1_s0=old_models_v1/pz_seeds/badnet_s0 --model v1_s1=old_models_v1/pz_seeds/badnet_s1 \
    --model v1_s2=old_models_v1/pz_seeds/badnet_s2 --model v1_s3=old_models_v1/pz_seeds/badnet_s3 \
    --model v1_s42=old_models_v1/pz_seeds/badnet_s42 --model v1_0930=old_models_v1/poisoned_roberta_large_imdb_badnet
  # stop here if the probe code fails on the existing checkpoints: the training runs use the same code
  [ "${DRY:-0}" = 1 ] || grep -qxF "pzbias eval existing|$CODE_HASH" "$DONE" || { echo "check 1 failed, stopping (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1; }
  # InSent PZs (same clean-label 1,500 recipe, sentence trigger): does the same asymmetry appear with another trigger?
  run pz_bias_eval.py "pzbias eval existing insent" --attack_tag insent --out logs/pzbias_eval_existing_insent.csv \
    --model insent_old=old_models/poisoned_roberta_large_insent \
    --model insent_0930=old_models_v1/poisoned_roberta_large_imdb_insent
fi

# 2. dose-response, seed-outer
for s in $SEEDS; do
  for n in $LEVELS; do
    pn=$n; [ "$n" = all ] && pn=-1
    case "$SAVE" in
      all)  sv=(--out_dir "pz_dose/n${n}_s${s}" --save_every_epoch) ;;
      best) if [ "$n" != 1500 ]; then sv=(--out_dir "pz_dose/n${n}_s${s}"); else sv=(--no_save); fi ;;
      *)    sv=(--no_save) ;;
    esac
    run poisoned_pretrain.py "pz dose n$n s$s" --attack_tag badnet --seed "$s" --poison_n "$pn" "${sv[@]}"
    # stop at the first failed run instead of failing the remaining ones one by one
    [ "${DRY:-0}" = 1 ] || grep -qxF "pz dose n$n s$s|$CODE_HASH" "$DONE" || { echo "run 'pz dose n$n s$s' failed, stopping (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1; }
  done
done

[ "${DRY:-0}" = 1 ] || echo "===== Done =====" | tee -a "$LOGFILE"