#!/bin/bash
# nohup bash v2_baseline.sh > /dev/null 2>&1 &
# Stage 2 (qualification): baseline LoRA (no Cl, no Tr) on every v2 backdoored model x finetune seed.
# Pt is on for every run (as in batch2.sh): it only acts after training, so it does not change the trajectory.
# The old backdoored model (EXTRA_PZ) is run as a labeled member, tag "oldpz". EXTRA_PZ="" leaves it out.
NAME=extra; source ./v2_common.sh
FT_SEEDS="6 7 8"   # PREREG §5: only for PZs the rule marked AMBIGUOUS
EXTRA_PZ="${EXTRA_PZ-old_models/poisoned_roberta_large_badnet/pytorch_model.bin}"
for pz in pz_v2/badnet_s1/pytorch_model.bin pz_v2/badnet_s4/pytorch_model.bin old_models/poisoned_roberta_large_badnet/pytorch_model.bin; do
  [ -e "$pz" ] || { echo "skip: $pz does not exist" | tee -a "$LOGFILE"; continue; }
  case "$pz" in
    pz_v2/*)
      tag=$(basename "$(dirname "$pz")")
      # skip a model whose training did not finish (poisoned_pretrain saves at the best epoch, so a crashed run leaves a file)
      grep -q "^pz ${tag/_/ }|" logs/done_pz.txt 2>/dev/null || { echo "skip $tag: not marked done in logs/done_pz.txt" | tee -a "$LOGFILE"; continue; } ;;
    *) tag=oldpz ;;
  esac
  # PZ identity in the done-key: sha from meta.json (written by poisoned_pretrain.py), else hash the file (the old PZ has none)
  sha=$(grep -o '"checkpoint_sha256": "[0-9a-f]*"' "$(dirname "$pz")/meta.json" 2>/dev/null | grep -o '[0-9a-f]\{64\}' | cut -c1-8)
  [ -n "$sha" ] || sha=$(sha256sum "$pz" | cut -c1-8)
  for fs in $FT_SEEDS; do
    run variant_finetune.py "$tag sha$sha ft$fs base lr2e-4" --pz_path "$pz" --seed "$fs" --lr 2e-4 --use_spectral_rescaling
  done
done
finish