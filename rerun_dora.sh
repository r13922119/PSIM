#!/bin/bash
set -e   # 任何一次失敗就整個腳本停下來，不要繼續跑下一組，你才能立刻發現問題

mkdir -p logs
LOGFILE="logs/rerun_dora_$(date +%Y%m%d_%H%M%S).log"

echo "===== 1/3: DoRA baseline =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora 2>&1 | tee -a "$LOGFILE"

echo "===== 2/3: DoRA + Mechanism 1 (Cl) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_pretrained_dropout 2>&1 | tee -a "$LOGFILE"

echo "===== 3/3: DoRA + Mechanism 2 (Tr) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_orthogonal_penalty 2>&1 | tee -a "$LOGFILE"

echo "===== All done =====" | tee -a "$LOGFILE"