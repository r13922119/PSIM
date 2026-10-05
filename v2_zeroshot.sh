#!/bin/bash
# nohup bash v2_zeroshot.sh > /dev/null 2>&1 &
# What a poisoned model (PZ) scores on SST-2 BEFORE any fine-tuning: clean dev/test accuracy and ASR.
# Method: one "fine-tune" epoch with --lr 0. LoRA's B matrix starts at 0 and nothing is updated at lr 0, so the model that
#   gets evaluated is exactly the PZ (eval mode, no dropout). No --use_spectral_rescaling: with an all-zero update the
#   rescaling factor sigma_pre / sigma_delta would divide by zero.
# Reads the [EPOCH] line of epoch 0:  dev=<dev clean acc>  test clean accuracy=<test CA>  ASR=<ASR of the PZ itself>
# Cost: about 1 to 2 minutes per PZ on one Ada. Seed 0 only (at lr 0 the seed cannot change the evaluated model).
# NOT part of the S2-S4 results: logs are named logs/zs_*.log, which compare_s2.py / compare_s4.py never read.
# DRY=1 prints what would run.
NAME=zs; source ./v2_common.sh

if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2"; }; fi
say() { echo "$*" | tee -a "$LOGFILE"; }

PZS="old_models/poisoned_roberta_large_badnet/pytorch_model.bin pz_v2/badnet_s0/pytorch_model.bin pz_v2/badnet_s1/pytorch_model.bin pz_v2/badnet_s2/pytorch_model.bin pz_v2/badnet_s3/pytorch_model.bin pz_v2/badnet_s4/pytorch_model.bin"

say "zero-shot check of every PZ (lr 0, 1 epoch, seed 0)"
for pz in $PZS; do
  [ -e "$pz" ] || { say "skip: $pz does not exist"; continue; }
  case "$pz" in pz_v2/*) tag=$(basename "$(dirname "$pz")") ;; *) tag=oldpz ;; esac
  sha=$(grep -o '"checkpoint_sha256": "[0-9a-f]*"' "$(dirname "$pz")/meta.json" 2>/dev/null | grep -o '[0-9a-f]\{64\}' | cut -c1-8)
  [ -n "$sha" ] || sha=$(sha256sum "$pz" | cut -c1-8)
  run variant_finetune.py "$tag sha$sha zeroshot lr0" --pz_path "$pz" --seed 0 --lr 0 --num_epochs 1
done

if [ "${DRY:-0}" != 1 ]; then
  say ""
  say "=== zero-shot summary (epoch 0 of each lr-0 run) ==="
  grep -h -E '^===== .* zeroshot|^\[EPOCH\] epoch=0 ' "$LOGFILE" | sed -E 's/^===== ([^|]*)\|.*/\1/' | tee -a "$LOGFILE"
fi
finish