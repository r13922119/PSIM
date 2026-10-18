#!/bin/bash
# run_probes_1018c.sh — C6: the O1 truncation probe on more adapters (no training), see PREREG_probes_1018b.md, revision 2
#   needs eval_probes.py v3 (floor masks also measure ASR). Run after run_probes_1018b.sh, or alongside it on another GPU:
#   nohup bash run_probes_1018c.sh > /dev/null 2>&1 &
# Each block is its own python call, so every block writes its own probe_out files when it finishes; stop any time.
# Estimated GPU time on one RTX 6000 Ada (~2.5 min per adapter): C6a ~45 min, C6b ~30 min, C6c ~60 min, C6d ~30 min.
NAME=probes_c
source v2_common.sh
mkdir -p probe_out
OLD=old_models/poisoned_roberta_large_badnet/pytorch_model.bin
A=adapters/roberta_badnet_sst-2_lora
V1=old_models/adapters/roberta_badnet_sst-2_lora
SUF=_ep20_wd0.01_pz-
O=${SUF}poisoned_roberta_large_badnet
KFINE="16 32 48 64 96 128 256 512 1024"
T="--probe truncate --no_zeroshot --trunc_on_adapters --groups L16-23_q --k_grid $KFINE"

# C6-0 (only if run_probes_1018b.sh already ran C3 with eval_probes.py v2, i.e. without floor ASR): uncomment
# run eval_probes.py "C6-0 truncate q16-23 oldpz S2 adapters (v3 rerun)" $T --pz $OLD --check_csv s2.csv \
#   --out_prefix probe_out/C6-0_trunc_q16-23_oldpz_s2 \
#   --adapters ${A}_pt_lr2e-04_r8_a16_seed{3,4,5}$O ${A}_cl0.1_pt_lr2e-04_r8_a16_seed{3,4,5}$O

# C6a. oldpz: Tr-only and Cl+Tr (k=8, lambda 10/480, seeds 3-5), then 6 more baselines (seeds 0-2, 6-8)
run eval_probes.py "C6a truncate q16-23 oldpz Tr-only" $T --pz $OLD --check_csv s4.csv \
  --out_prefix probe_out/C6a_trunc_q16-23_oldpz_tr \
  --adapters ${A}_tr10_k8_pt_lr2e-04_r8_a16_seed{3,4,5}$O ${A}_tr480_k8_pt_lr2e-04_r8_a16_seed{3,4,5}$O
run eval_probes.py "C6a truncate q16-23 oldpz Cl+Tr" $T --pz $OLD --check_csv s4.csv \
  --out_prefix probe_out/C6a_trunc_q16-23_oldpz_cltr \
  --adapters ${A}_cl0.1_tr10_k8_pt_lr2e-04_r8_a16_seed{3,4,5}$O ${A}_cl0.1_tr480_k8_pt_lr2e-04_r8_a16_seed{3,4,5}$O
run eval_probes.py "C6a truncate q16-23 oldpz more baselines" $T --pz $OLD \
  --check_csv logs/baseline_20261003_071718_parsed.csv logs/extra_20261003_104045_parsed.csv \
  --out_prefix probe_out/C6a_trunc_q16-23_oldpz_base \
  --adapters ${A}_pt_lr2e-04_r8_a16_seed{0,1,2,6,7,8}$O

# C6b. the two fresh PZs: baseline and Cl-only, seeds 3-5 (the C3 design on badnet_s2 and badnet_s0)
for pz in badnet_s2 badnet_s0; do
  run eval_probes.py "C6b truncate q16-23 $pz base+Cl" $T --pz pz_v2/$pz/pytorch_model.bin --check_csv s2.csv \
    --out_prefix probe_out/C6b_trunc_q16-23_$pz \
    --adapters ${A}_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}$pz ${A}_cl0.1_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}$pz
done

# C6c. the two fresh PZs: Tr-only and Cl+Tr (k=8, lambda 10/480, seeds 3-5)
for pz in badnet_s2 badnet_s0; do
  run eval_probes.py "C6c truncate q16-23 $pz Tr+ClTr" $T --pz pz_v2/$pz/pytorch_model.bin --check_csv s4.csv \
    --out_prefix probe_out/C6c_trunc_q16-23_$pz \
    --adapters ${A}_tr10_k8_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}$pz ${A}_tr480_k8_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}$pz \
               ${A}_cl0.1_tr10_k8_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}$pz ${A}_cl0.1_tr480_k8_pt_lr2e-04_r8_a16_seed{3,4,5}${SUF}$pz
done

# C6d. oldpz v1 adapters (older code): Cl seeds 0-10 + baseline seed 0. Gate = the dropout-off rows of A2 (same script, deterministic)
python - <<'EOF'
import csv
rows = [r for r in csv.DictReader(open("probe_out/A2_dropout_v1.csv")) if r["mode"] == "off" and r["state"] == "pre-Pt"]
with open("probe_out/gate_v1_from_A2.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name", "pre-pt_test_acc", "pre-pt_ASR", "pt_test_acc", "pt_ASR"])
    for r in rows:
        w.writerow([r["model"], r["test_acc"], r["asr"], "", ""])
print("gate_v1_from_A2.csv:", len(rows), "rows")
EOF
run eval_probes.py "C6d truncate q16-23 oldpz v1 adapters" $T --pz $OLD --check_csv probe_out/gate_v1_from_A2.csv \
  --out_prefix probe_out/C6d_trunc_q16-23_oldpz_v1 \
  --adapters ${V1}_cl0.1_lr2e-04_r8_a16_seed{0..10}_ep20 ${V1}_lr2e-04_r8_a16_seed0_ep20

echo "===== Done =====" | tee -a "$LOGFILE"