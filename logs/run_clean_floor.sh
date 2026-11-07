#!/bin/bash
# run_clean_floor.sh — ASR floor: the CLEAN model (no backdoor) fine-tuned with the same protocol (11/06).
# plain LoRA, finetune seeds 0-2. The K / Cl arms on the clean model go through run_psel3.sh (LABELS=clean_n0_s0_ep0).
# ONE card (CMLab rule).   nohup bash run_clean_floor.sh > /dev/null 2>&1 &
NAME=clean_floor; source ./v2_common.sh
for fs in 0 1 2; do
  run variant_finetune.py "clean_n0_s0_ep0 ft$fs base lr2e-4" --pz_path old_models_v1/clean_roberta_large_imdb/pytorch_model.bin --seed "$fs" --lr 2e-4 --use_spectral_rescaling
done
finish