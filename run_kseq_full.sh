#!/bin/bash
# run_kseq_full.sh — kseq v2: layer-by-layer truncation for ALL four orderings at ALL ten r values (the same grid as
# kdiag), with the constant / varying split recorded in every pass (see kseq_eval.py). Diagnostic only: no k is chosen,
# no saved weight changes, nothing is fine-tuned. SST-2 train builds C (no labels); SST-2 dev only scores.
# 40 sequential passes per model. The 16 (ordering, r) cells v1 already ran must be reproduced exactly (checked).
# Needs in the repo root: kseq_eval.py (v2, replaces v1), kdiag_eval.py, logs/kdiag/*.npz, logs/kseq/*.npz (v1, for the check).
# ONE card (CMLab rule). Rough guess 35-45 min per model, about 4 hours for 6 (v1 took ~1 min per pass); the log prints
# elapsed minutes after every pass.
# Writes logs/kseq_full/*.npz; commit them and the log.
# Before it, without a GPU:   python kseq_eval.py --dry <same --model arguments as below>
NAME=kseq_full; source ./v2_common.sh
T="kseq full grid panel + clean"
run kseq_eval.py "$T" \
  --model n1500_s0_ep2=pz_epoch/n1500_s0_ep2@2 \
  --model n1500_s1_ep2=pz_cleandev/cleandev_n1500_s1@2 \
  --model n1500_s2_ep2=pz_v2/badnet_s2@2 \
  --model n1500_s3_ep2=pz_cleandev/cleandev_n1500_s3@2 \
  --model n1500_s4_ep2=pz_epoch/n1500_s4_ep2@2 \
  --model clean_n0_s0_ep0=old_models_v1/clean_roberta_large_imdb
if grep -qxF "$T|$CODE_HASH" "$DONE"; then
  echo "===== Done =====" | tee -a "$LOGFILE"
else
  echo "'$T' failed (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1
fi