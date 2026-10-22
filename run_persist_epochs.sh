#!/bin/bash
# run_persist_epochs.sh — is a later epoch of poisoned pretraining more persistent? (PREREG_pzbias.md section 11)
# n = 1500, PZ seeds 0-4. Two of the three epochs per seed were already fine-tuned (PSIM's pick: logs/baseline_20261003_071718.log;
# clean-dev pick: logs/persist_cleandev_20261022_025758.log). This fills in the third, so every seed has all 3 epochs:
#   seed 0: epoch 2   seed 1: epoch 1   seed 2: epoch 0   seed 3: epoch 1   seed 4: epoch 2
# A. recreate those 5 checkpoints (deterministic rerun, --save_epoch) into pz_epoch/n1500_s<seed>_ep<epoch>/   ~40 min, 7 GB
#    (their [PROBE] lines must again equal the 10/21 dose log)
# B. fine-tune each with the persistence protocol (baseline LoRA, lr 2e-4, Pt on, 20 epochs), finetune seeds 0-2  15 runs, ~2.3 h
# ONE card (CMLab rule).   nohup bash run_persist_epochs.sh > /dev/null 2>&1 &
NAME=persist_epochs; source ./v2_common.sh
MISSING="0:2 1:1 2:0 3:1 4:2"
stop_unless_done() { grep -qxF "$1|$CODE_HASH" "$DONE" || { echo "'$1' failed, stopping (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1; }; }
for se in $MISSING; do
  s=${se%:*}; e=${se#*:}
  T="pz epoch n1500 s$s ep$e"
  run poisoned_pretrain.py "$T" --attack_tag badnet --seed "$s" --poison_n 1500 --save_epoch "$e" --out_dir "pz_epoch/n1500_s${s}_ep${e}"
  stop_unless_done "$T"
done
for fs in 0 1 2; do
  for se in $MISSING; do
    s=${se%:*}; e=${se#*:}
    T="n1500_s${s}_ep${e} ft$fs base lr2e-4"
    run variant_finetune.py "$T" --pz_path "pz_epoch/n1500_s${s}_ep${e}/pytorch_model.bin" --seed "$fs" --lr 2e-4 --use_spectral_rescaling
    stop_unless_done "$T"
  done
done
finish