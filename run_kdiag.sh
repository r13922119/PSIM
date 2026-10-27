#!/bin/bash
# run_kdiag.sh — diagnostic only: how many directions of each query/value W_pre carry its output on SST-2 (see kdiag_eval.py).
# No k is chosen, no saved weight changes, nothing is fine-tuned. SST-2 train builds C (no labels); SST-2 dev only scores the
# temporarily truncated model. Models: the 5 panel PZs (n = 1,500, last epoch) + the clean reference.
# ONE card (CMLab rule). The log prints elapsed minutes at every step; judge the speed from the first model.
# Writes logs/kdiag/*.npz; commit them and the log.
# Before it, without a GPU:   python kdiag_eval.py --dry <same --model arguments as below>
NAME=kdiag; source ./v2_common.sh
T="kdiag panel + clean"
run kdiag_eval.py "$T" \
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