#!/bin/bash
# run_p_auc.sh — step-0 AUC as a function of the dropout rate p on W_pre (eval only, no training). See p_auc_eval.py.
# Models: the 5 panel PZs (n = 1,500, last epoch, decided 10/24) + one clean model for reference
#   (old_models_v1/clean_roberta_large_imdb = new pipeline, 0 inserted, seed 0, epoch 0; verified identical on 10/21).
# ONE card (CMLab rule). Cost per model: 1 + 11 x 8 = 89 passes over about 8,200 short sentences.
#   The log prints elapsed minutes after every p, so the real speed is visible after the first model's first few p.
# Writes logs/p_auc/*.npz (about 3 MB each) and the log; commit both.
# Before it, without a GPU:   python p_auc_eval.py --dry <same --model arguments as below>
#   bash run_p_auc.sh        (or: nohup bash run_p_auc.sh > /dev/null 2>&1 &)
NAME=p_auc; source ./v2_common.sh
T="p_auc sweep panel + clean"
run p_auc_eval.py "$T" \
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