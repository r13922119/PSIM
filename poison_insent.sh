#!/bin/bash
set -e
mkdir -p logs
LOGFILE="logs/poison_insent_$(date +%Y%m%d_%H%M%S).log"

echo "===== Poisoning: RoBERTa + InSent =====" | tee -a "$LOGFILE"
python poisoned_pretrain.py --attack_tag insent 2>&1 | tee -a "$LOGFILE"

echo "===== Done =====" | tee -a "$LOGFILE"