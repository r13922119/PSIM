#!/bin/bash
set -e   # 任何一次失敗就整個腳本停下來，不要繼續跑下一組，你才能立刻發現問題

mkdir -p logs
LOGFILE="logs/run_dora_tr_k8_$(date +%Y%m%d_%H%M%S).log"

echo "===== 3/8: DoRA + Mechanism 2 (Tr) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_orthogonal_penalty --svd_k 8 2>&1 | tee -a "$LOGFILE"

echo "===== All done =====" | tee -a "$LOGFILE"