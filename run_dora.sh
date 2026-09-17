#!/bin/bash
set -e   # 任何一次失敗就整個腳本停下來，不要繼續跑下一組，你才能立刻發現問題

mkdir -p logs
LOGFILE="logs/run_dora_$(date +%Y%m%d_%H%M%S).log"

echo "===== 1/4: DoRA baseline =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora 2>&1 | tee -a "$LOGFILE"

echo "===== 2/4: DoRA + Mechanism 1 (Cl) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_pretrained_dropout 2>&1 | tee -a "$LOGFILE"

echo "===== 3/4: DoRA + Mechanism 2 (Tr) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_orthogonal_penalty 2>&1 | tee -a "$LOGFILE"

echo "===== 4/4: DoRA + Mechanism 1 (Cl) + Mechanism 2 (Tr) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_pretrained_dropout --use_orthogonal_penalty 2>&1 | tee -a "$LOGFILE"

echo "===== All done =====" | tee -a "$LOGFILE"