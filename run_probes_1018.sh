#!/bin/bash
# run_probes_1018.sh — evaluation-only probes (no training), see PREREG_probes_1018.md
#   put eval_probes.py, this file and PREREG_probes_1018.md in the PSIM root (next to attack_utils.py), then
#   nohup bash run_probes_1018.sh > /dev/null 2>&1 &
# Estimated GPU time on one RTX 6000 Ada: A1 ~30 min, B1 ~5 min, A2 ~30 min, B2 ~12 min.
NAME=probes
source v2_common.sh
mkdir -p probe_out
PZ=old_models/poisoned_roberta_large_badnet/pytorch_model.bin
V2=adapters/roberta_badnet_sst-2_lora
SUF=_ep20_wd0.01_pz-poisoned_roberta_large_badnet
V1=old_models/adapters/roberta_badnet_sst-2_lora

# 0. dry run: lists the 48 query/value modules and the 6 Pt modules of one adapter, then exits
run eval_probes.py "dryrun" --dry_run --pz $PZ --adapters ${V2}_cl0.1_pt_lr2e-04_r8_a16_seed3${SUF}

# 1. (A) test-time Cl dropout, v2 oldpz: PZ zero-shot + baseline + Cl-only, seeds 3-5; gate against s2.csv
run eval_probes.py "A1 dropout v2 oldpz" --probe dropout --pz $PZ --check_csv s2.csv \
  --out_prefix probe_out/A1_dropout_v2_oldpz \
  --adapters ${V2}_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF} ${V2}_cl0.1_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}

# 2. (B) truncation probe on the oldpz itself: all 48 modules share one k
run eval_probes.py "B1 truncate all48 oldpz" --probe truncate --pz $PZ --groups all \
  --out_prefix probe_out/B1_trunc_all48_oldpz

# 3. (A) test-time Cl dropout on the v1 adapters: Cl seeds 0-10 (v1 reported 26.05% ASR with eval-time dropout on) + v1 baseline seed 0
run eval_probes.py "A2 dropout v1 adapters" --probe dropout --no_zeroshot --pz $PZ \
  --out_prefix probe_out/A2_dropout_v1 \
  --adapters ${V1}_cl0.1_lr2e-04_r8_a16_seed{0..10}_ep20 ${V1}_lr2e-04_r8_a16_seed0_ep20

# 4. (B, optional) truncation by depth group x {query, value}
run eval_probes.py "B2 truncate depth3_qv oldpz" --probe truncate --pz $PZ --groups depth3_qv \
  --out_prefix probe_out/B2_trunc_groups_oldpz

echo "===== Done =====" | tee -a "$LOGFILE"