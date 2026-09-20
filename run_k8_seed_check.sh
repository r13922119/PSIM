#!/bin/bash
set -e
mkdir -p logs
LOGFILE="logs/run_k8_seed_check_$(date +%Y%m%d_%H%M%S).log"

for seed in 1 2 3; do
    echo "===== LoRA + Tr, k=8, seed=$seed =====" | tee -a "$LOGFILE"
    python variant_finetune.py --use_orthogonal_penalty --svd_k 8 --seed $seed 2>&1 | tee -a "$LOGFILE"
done

echo "===== All done =====" | tee -a "$LOGFILE"
