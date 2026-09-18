#!/bin/bash
set -e
mkdir -p logs
LOGFILE="logs/tr_lambda_check_$(date +%Y%m%d_%H%M%S).log"

for lam in 48 240 480; do
    echo "===== LoRA + Tr, k=2, tr_lambda=$lam (等效論文網格 λ=1/5/10) =====" | tee -a "$LOGFILE"
    python variant_finetune.py --use_orthogonal_penalty --svd_k 2 --tr_lambda $lam 2>&1 | tee -a "$LOGFILE"
done

echo "===== All done =====" | tee -a "$LOGFILE"
