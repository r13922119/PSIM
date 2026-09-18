#!/bin/bash
set -e
mkdir -p logs
LOGFILE="logs/cl_seed_sweep_$(date +%Y%m%d_%H%M%S).log"

for seed in 1 2 3 4 5 6 7 8 9 10; do
    echo "===== LoRA + Cl, seed=$seed =====" | tee -a "$LOGFILE"
    python variant_finetune.py --use_pretrained_dropout --seed $seed 2>&1 | tee -a "$LOGFILE"
done
echo "===== All done =====" | tee -a "$LOGFILE"