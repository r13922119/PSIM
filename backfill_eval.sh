#!/bin/bash
set -e
mkdir -p logs
LOGFILE="logs/backfill_eval_$(date +%Y%m%d_%H%M%S).log"

for dir in \
    "roberta_badnet_sst-2_dora_tr10_k8_lr2e-04_r8_a8_seed0_ep20" \
    "roberta_badnet_sst-2_lora_tr10_k1024_lr2e-04_r8_a16_seed0_ep20" \
    "roberta_badnet_sst-2_lora_tr10_k8_lr2e-03_r8_a16_seed0_ep20" \
    "roberta_badnet_sst-2_lora_tr10_k8_lr2e-05_r8_a16_seed0_ep20"
do
    echo "===== $dir =====" | tee -a "$LOGFILE"
    python evaluate_checkpoint.py --adapter_path "./adapters/$dir" 2>&1 | tee -a "$LOGFILE"
done