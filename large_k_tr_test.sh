#!/bin/bash
mkdir -p logs
LOGFILE="logs/large_k_tr_test_$(date +%Y%m%d_%H%M%S).log"

# L2 weight decay 對照組（同一組 seed，讓比較公平）
for SEED in 1 2 3; do
  echo "=== L2-weight-decay control seed=$SEED ===" | tee -a $LOGFILE
  python variant_finetune.py \
    --attack_tag badnet --model_tag roberta \
    --weight_decay 10 \
    --seed $SEED 2>&1 | tee -a $LOGFILE
done

for K in 64 128 256 384 512; do
  for SEED in 1 2 3; do
    echo "=== k=$K seed=$SEED (Tr-alone) ===" | tee -a $LOGFILE
    python variant_finetune.py \
      --attack_tag badnet --model_tag roberta \
      --use_orthogonal_penalty --svd_k $K --tr_lambda 10 \
      --seed $SEED 2>&1 | tee -a $LOGFILE
  done
done