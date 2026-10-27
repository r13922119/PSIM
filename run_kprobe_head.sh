#!/bin/bash
# run_kprobe_head.sh — new linear head on the frozen PZ, trained on clean SST-2 train; clean accuracy, AUC and ASR on SST-2 dev
# (see kprobe_head.py). Eval only plus a logistic regression; no fine-tuning of the encoder. ONE card (CMLab rule).
# Rough guess a few minutes per model. Writes logs/kprobe_head/*.npz; commit them and the log.
# Before it, without a GPU:   python kprobe_head.py --dry <same --model arguments as below>
NAME=kprobe_head; source ./v2_common.sh
T="kprobe_head panel + clean"
run kprobe_head.py "$T" \
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