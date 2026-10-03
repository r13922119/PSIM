#!/bin/bash
# nohup bash v2_pz.sh > /dev/null 2>&1 &
# Stage 1: backdoored pretrained models, all on the same Ada. SEEDS="0 1 2 3 42" bash v2_pz.sh to change.
NAME=pz; source ./v2_common.sh
SEEDS="${SEEDS:-0 1 2 3}"
for s in $SEEDS; do
  run poisoned_pretrain.py "pz badnet s$s" --attack_tag badnet --seed "$s" --out_dir "pz_v2/badnet_s$s"
done
finish