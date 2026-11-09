#!/bin/bash
# run_plain_last.sh — plain LoRA on the 5 panel PZs with the current variant_finetune.py, for Pt on the epoch-19 checkpoint (11/07).
NAME=plain_last; source ./v2_common.sh
for spec in "n1500_s0_ep2 pz_epoch/n1500_s0_ep2" "cleandev_n1500_s1 pz_cleandev/cleandev_n1500_s1" "badnet_s2 pz_v2/badnet_s2" \
            "cleandev_n1500_s3 pz_cleandev/cleandev_n1500_s3" "n1500_s4_ep2 pz_epoch/n1500_s4_ep2"; do
  set -- $spec
  for fs in 0 1 2; do
    run variant_finetune.py "$1 ft$fs base lr2e-4" --pz_path "$2/pytorch_model.bin" --seed "$fs" --lr 2e-4 --use_spectral_rescaling
  done
done
finish
