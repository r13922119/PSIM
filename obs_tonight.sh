#!/bin/bash
# nohup bash obs_tonight.sh > /dev/null 2>&1 &
# Three logged runs on one backdoored model (badnet_s2) and one finetune seed (3), the same arms and settings as S4 (lr 2e-4, Pt on):
#   baseline, Tr-only (lambda 10, k 8), Tr-only (lambda 480, k 8).        About 10 to 16 minutes each, one after the other.
# The logs go to obs/<experiment name>/ (modules.csv, steps.csv, epochs.csv, meta.json); the adapters go to obs/adapters/, so nothing
# under adapters/ is touched. Run  bash gate_obs.sh  first: it shows that logging ON and OFF give identical runs.
# Read the result with:  python obs_report.py obs/*seed3*   (see the header of obs_report.py)
NAME=obs; source ./v2_common.sh
PZ="${PZ:-pz_v2/badnet_s2/pytorch_model.bin}"
FS="${FS:-3}"
[ -e "$PZ" ] || { echo "PZ $PZ not found" | tee -a "$LOGFILE"; exit 3; }
sha=$(grep -o '"checkpoint_sha256": "[0-9a-f]*"' "$(dirname "$PZ")/meta.json" 2>/dev/null | grep -o '[0-9a-f]\{64\}' | cut -c1-8)
[ -n "$sha" ] || sha=$(sha256sum "$PZ" | cut -c1-8)
common=(--pz_path "$PZ" --seed "$FS" --lr 2e-4 --svd_k 8 --use_spectral_rescaling --obs_log obs)
run variant_finetune.py "obs s2 sha$sha ft$FS base"             "${common[@]}"
run variant_finetune.py "obs s2 sha$sha ft$FS tr-only lam10 k8"  "${common[@]}" --use_orthogonal_penalty --tr_lambda 10
run variant_finetune.py "obs s2 sha$sha ft$FS tr-only lam480 k8" "${common[@]}" --use_orthogonal_penalty --tr_lambda 480
finish