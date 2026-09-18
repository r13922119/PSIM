#!/bin/bash
set -e
mkdir -p logs
LOGFILE="logs/run_insent_$(date +%Y%m%d_%H%M%S).log"

echo "===== 1/8: InSent, LoRA baseline =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent 2>&1 | tee -a "$LOGFILE"

echo "===== 2/8: InSent, DoRA baseline =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_dora 2>&1 | tee -a "$LOGFILE"

echo "===== 3/8: InSent, LoRA + Cl =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_pretrained_dropout 2>&1 | tee -a "$LOGFILE"

echo "===== 4/8: InSent, DoRA + Cl =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_dora --use_pretrained_dropout 2>&1 | tee -a "$LOGFILE"

echo "===== 5/8: InSent, LoRA + Tr (k=8) =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_orthogonal_penalty --svd_k 8 2>&1 | tee -a "$LOGFILE"

echo "===== 6/8: InSent, DoRA + Tr (k=8) =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_dora --use_orthogonal_penalty --svd_k 8 2>&1 | tee -a "$LOGFILE"

echo "===== 7/8: InSent, LoRA + Cl + Tr =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 2>&1 | tee -a "$LOGFILE"

echo "===== 8/8: InSent, DoRA + Cl + Tr =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_dora --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 2>&1 | tee -a "$LOGFILE"

echo "===== All done =====" | tee -a "$LOGFILE"