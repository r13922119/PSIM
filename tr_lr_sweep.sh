#!/bin/bash
set -e

mkdir -p logs
LOGFILE="logs/tr_lr_sweep_$(date +%Y%m%d_%H%M%S).log"

for lr in 2e-5 2e-4 2e-3; do
    echo "===== Tr alone, lr=$lr, k=8 =====" | tee -a "$LOGFILE"
    python variant_finetune.py --use_orthogonal_penalty --svd_k 8 --lr $lr 2>&1 | tee -a "$LOGFILE"
done

echo "===== All done =====" | tee -a "$LOGFILE"