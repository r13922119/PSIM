#!/bin/bash
# run_persist_cleandev.sh — does the clean-dev pick keep its backdoor through clean LoRA fine-tuning as well as PSIM's pick?
# Same protocol as the persistence qualification (logs/baseline_20261003_071718.log, script 3dccbcb89f5c, which the merged
# variant_finetune.py reproduces with default flags): baseline LoRA, lr 2e-4, Pt on, 20 epochs, finetune seeds 0-2.
# PZ = pz_cleandev/cleandev_n1500_s<s> (made by run_pzscores.sh). PSIM's pick of the same seed is pz_v2/badnet_s<s>, already
# fine-tuned with exactly this protocol and these finetune seeds, so each (PZ seed, finetune seed) is a paired comparison.
# 15 runs x ~9 min = ~2.3 h on one card. Finetune-seed outer: after the first 5 runs every PZ has one finetune seed.
NAME=persist_cleandev; source ./v2_common.sh
for fs in ${FSEEDS:-0 1 2}; do
  for s in ${SEEDS:-0 1 2 3 4}; do
    PZ=pz_cleandev/cleandev_n1500_s$s/pytorch_model.bin
    [ -e "$PZ" ] || { echo "missing $PZ (run: LEVELS=1500 bash run_pzscores.sh)" | tee -a "$LOGFILE"; exit 1; }
    T="cleandev_n1500_s$s ft$fs base lr2e-4"
    run variant_finetune.py "$T" --pz_path "$PZ" --seed "$fs" --lr 2e-4 --use_spectral_rescaling
    grep -qxF "$T|$CODE_HASH" "$DONE" || { echo "'$T' failed, stopping (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1; }
  done
done
finish