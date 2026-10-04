#!/bin/bash
# nohup bash v2_s2.sh > /dev/null 2>&1 &
# S2: baseline (rerun) vs Cl-only on the PZs that passed PREREG, with comparison seeds 3-5 (PREREG §3).
# P' = badnet_s0, badnet_s2 (fresh, persistent); oldpz is a labeled member, reported separately.
# Order is seed-outer so a partial run still gives paired (base, cl) cells for every PZ.
NAME=s2; source ./v2_common.sh
SEEDS_S2="${SEEDS_S2:-3 4 5}"
PZS="pz_v2/badnet_s0/pytorch_model.bin pz_v2/badnet_s2/pytorch_model.bin old_models/poisoned_roberta_large_badnet/pytorch_model.bin"
for fs in $SEEDS_S2; do
  for pz in $PZS; do
    [ -e "$pz" ] || { echo "skip: $pz does not exist" | tee -a "$LOGFILE"; continue; }
    case "$pz" in pz_v2/*) tag=$(basename "$(dirname "$pz")") ;; *) tag=oldpz ;; esac
    sha=$(grep -o '"checkpoint_sha256": "[0-9a-f]*"' "$(dirname "$pz")/meta.json" 2>/dev/null | grep -o '[0-9a-f]\{64\}' | cut -c1-8)
    [ -n "$sha" ] || sha=$(sha256sum "$pz" | cut -c1-8)
    run variant_finetune.py "$tag sha$sha ft$fs base lr2e-4" --pz_path "$pz" --seed "$fs" --lr 2e-4 --use_spectral_rescaling
    run variant_finetune.py "$tag sha$sha ft$fs cl lr2e-4"   --pz_path "$pz" --seed "$fs" --lr 2e-4 --use_spectral_rescaling --use_pretrained_dropout
  done
done
finish