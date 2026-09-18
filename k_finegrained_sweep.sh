#!/bin/bash
set -e
mkdir -p logs
LOGFILE="logs/k_finegrained_sweep_$(date +%Y%m%d_%H%M%S).log"

for k in 3 4 5 6 7; do
    echo "===== LoRA + Tr, k=$k =====" | tee -a "$LOGFILE"
    python variant_finetune.py --use_orthogonal_penalty --svd_k $k 2>&1 | tee -a "$LOGFILE"
done
echo "===== All done =====" | tee -a "$LOGFILE"