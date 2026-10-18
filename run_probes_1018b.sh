#!/bin/bash
# run_probes_1018b.sh — replication probes (no training), see PREREG_probes_1018b.md
#   replace eval_probes.py with the v2 file first, then:  nohup bash run_probes_1018b.sh > /dev/null 2>&1 &
# Estimated GPU time on one RTX 6000 Ada: C1 ~30 min, C2 ~8 min, C3 ~12 min, C4 ~32 min, C5 ~40 min.
NAME=probes_b
source v2_common.sh
mkdir -p probe_out
OLD=old_models/poisoned_roberta_large_badnet/pytorch_model.bin
S0=pz_v2/badnet_s0/pytorch_model.bin
S2=pz_v2/badnet_s2/pytorch_model.bin
A=adapters/roberta_badnet_sst-2_lora
SUF=_ep20_wd0.01_pz-
KSTD="64 128 256 384 512 640 768 896 1024"
KFINE="16 32 48 64 96 128 256 512 1024"

# C1. O1 replication on the fresh PZs (zero-shot): L16-23 query + each layer 16-23 (fine k), and the E14 groups (standard k)
for pz in badnet_s2 badnet_s0; do
  P=pz_v2/$pz/pytorch_model.bin
  run eval_probes.py "C1 truncate q16-23 $pz" --probe truncate --pz $P --groups L16-23_q each_q_16-23 \
    --k_grid $KFINE --out_prefix probe_out/C1_trunc_q16-23_$pz
  run eval_probes.py "C1 truncate groups $pz" --probe truncate --pz $P --groups all48 depth3_qv \
    --k_grid $KSTD --out_prefix probe_out/C1_trunc_groups_$pz
done

# C2. oldpz per-layer localization (zero-shot), fine k
run eval_probes.py "C2 truncate q16-23 oldpz" --probe truncate --pz $OLD --groups L16-23_q each_q_16-23 \
  --k_grid $KFINE --out_prefix probe_out/C2_trunc_q16-23_oldpz

# C3. O1 after fine-tuning: oldpz adapters (S2 seeds 3-5), L16-23 query, pre-Pt; gate against s2.csv
run eval_probes.py "C3 truncate q16-23 oldpz adapters" --probe truncate --no_zeroshot --trunc_on_adapters --pz $OLD \
  --check_csv s2.csv --groups L16-23_q --k_grid $KFINE --out_prefix probe_out/C3_trunc_q16-23_oldpz_adapters \
  --adapters ${A}_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}poisoned_roberta_large_badnet \
             ${A}_cl0.1_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}poisoned_roberta_large_badnet

# C4. O2 replication: test-time Cl dropout on badnet_s2 and badnet_s0 (PZ zero-shot + baseline + Cl, seeds 3-5)
for pz in badnet_s2 badnet_s0; do
  run eval_probes.py "C4 dropout $pz" --probe dropout --pz pz_v2/$pz/pytorch_model.bin --check_csv s2.csv \
    --out_prefix probe_out/C4_dropout_$pz \
    --adapters ${A}_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}$pz ${A}_cl0.1_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}$pz
done

# C5. more units on oldpz (pre-Pt only): baseline seeds 0-2, 6-8; Tr-only and Cl+Tr (k=8, lambda 10/480) seeds 3-5   (~40 min)
O=${SUF}poisoned_roberta_large_badnet
run eval_probes.py "C5 dropout oldpz more units" --probe dropout --no_zeroshot --pt none --pz $OLD \
  --check_csv s4.csv logs/baseline_20261003_071718_parsed.csv logs/extra_20261003_104045_parsed.csv \
  --out_prefix probe_out/C5_dropout_oldpz_more \
  --adapters ${A}_pt_lr2e-04_r8_a16_seed{0,1,2,6,7,8}$O \
             ${A}_tr10_k8_pt_lr2e-04_r8_a16_seed{3,4,5}$O ${A}_tr480_k8_pt_lr2e-04_r8_a16_seed{3,4,5}$O \
             ${A}_cl0.1_tr10_k8_pt_lr2e-04_r8_a16_seed{3,4,5}$O ${A}_cl0.1_tr480_k8_pt_lr2e-04_r8_a16_seed{3,4,5}$O

echo "===== Done =====" | tee -a "$LOGFILE"