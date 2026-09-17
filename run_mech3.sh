#!/bin/bash
set -e   # 任何一次失敗就整個腳本停下來

mkdir -p logs
LOGFILE="logs/run_mech3_$(date +%Y%m%d_%H%M%S).log"

echo "===== 1/2: LoRA + Cl + Tr + Pt (full RoRA) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 2/2: DoRA + Cl + Tr + Pt (full RoRA) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== All done =====" | tee -a "$LOGFILE"