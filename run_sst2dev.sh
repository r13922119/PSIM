#!/bin/bash
# run_sst2dev.sh — SST-2 dev scores for the 15 checkpoints of poisoned pretraining at n = 1500 (eval only, no training).
# Why and what: docstring of sst2dev_eval.py; roadmap section 2.1 (10/24 05:47).
# ONE card (CMLab rule); about 10 minutes (15 checkpoints, each loaded once, about 4,000 short sentences each).
# Writes logs/scores/sst2dev/*.npz (small) and logs/sst2dev_eval.csv; commit both, plus the log.
# Before it, without a GPU:   python sst2dev_eval.py --dry
#   bash run_sst2dev.sh        (or: nohup bash run_sst2dev.sh > /dev/null 2>&1 &)
NAME=sst2dev; source ./v2_common.sh
T="sst2dev eval n1500 15 ckpts"
run sst2dev_eval.py "$T" --out logs/sst2dev_eval.csv
if grep -qxF "$T|$CODE_HASH" "$DONE"; then
  echo "===== Done =====" | tee -a "$LOGFILE"
else
  echo "'$T' failed (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1
fi