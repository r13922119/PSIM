#!/bin/bash
set -e
mkdir -p logs
LOGFILE="logs/run_mech3_insent_$(date +%Y%m%d_%H%M%S).log"

echo "===== 1/2: InSent, LoRA + Cl + Tr + Pt (full RoRA) =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 2/2: InSent, DoRA + Cl + Tr + Pt (full RoRA) =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_dora --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== All done =====" | tee -a "$LOGFILE"