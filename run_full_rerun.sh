#!/bin/bash
set -e
mkdir -p logs
LOGFILE="logs/run_full_rerun_$(date +%Y%m%d_%H%M%S).log"

# ===== 1-4: 修正後的 Cl+Tr+Pt (6-module layer selection), k=8 =====
echo "===== 1/19: BadNet, LoRA + Cl+Tr+Pt (fixed layers) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 2/19: BadNet, DoRA + Cl+Tr+Pt (fixed layers) =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 3/19: InSent, LoRA + Cl+Tr+Pt (fixed layers) =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 4/19: InSent, DoRA + Cl+Tr+Pt (fixed layers) =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_dora --use_pretrained_dropout --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

# ===== 5-8: Pt alone =====
echo "===== 5/19: BadNet, LoRA + Pt alone =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 6/19: BadNet, DoRA + Pt alone =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 7/19: InSent, LoRA + Pt alone =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 8/19: InSent, DoRA + Pt alone =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_dora --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

# ===== 9-12: Cl+Pt (無 Tr) =====
echo "===== 9/19: BadNet, LoRA + Cl+Pt =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_pretrained_dropout --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 10/19: BadNet, DoRA + Cl+Pt =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_pretrained_dropout --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 11/19: InSent, LoRA + Cl+Pt =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_pretrained_dropout --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 12/19: InSent, DoRA + Cl+Pt =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_dora --use_pretrained_dropout --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

# ===== 13-16: Tr+Pt (無 Cl) =====
echo "===== 13/19: BadNet, LoRA + Tr+Pt =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 14/19: BadNet, DoRA + Tr+Pt =====" | tee -a "$LOGFILE"
python variant_finetune.py --use_dora --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 15/19: InSent, LoRA + Tr+Pt =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

echo "===== 16/19: InSent, DoRA + Tr+Pt =====" | tee -a "$LOGFILE"
python variant_finetune.py --attack_tag insent --use_dora --use_orthogonal_penalty --svd_k 8 --use_spectral_rescaling 2>&1 | tee -a "$LOGFILE"

# ===== 17-19: k=2 三 seed 驗證 (LoRA+Tr, BadNet) =====
for seed in 1 2 3; do
    echo "===== 17-19/19: LoRA + Tr, k=2, seed=$seed =====" | tee -a "$LOGFILE"
    python variant_finetune.py --use_orthogonal_penalty --svd_k 2 --seed $seed 2>&1 | tee -a "$LOGFILE"
done

echo "===== All done =====" | tee -a "$LOGFILE"
