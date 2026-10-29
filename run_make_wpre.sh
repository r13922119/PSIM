#!/bin/bash
# run_make_wpre.sh — K step: pick r* per PZ by the K rule (from logs/kdiag only) and save W_pre,K (whiten truncation, simultaneous,
# one r for all 48 query/value modules) for variant_finetune.py --wpre_file. Nothing is fine-tuned; no ASR is looked at.
# LEVEL must be given (no default on purpose; it is your decision):
#   LEVEL=bonf   Bonferroni over the 10 r's: a drop counts only if the 99.5th percentile of the bootstrap AUC difference is < 0
#   LEVEL=per_r  each r on its own: the 97.5th percentile (upper end of the 95% band)
#   WALK=stepwise (default): from r = 0.9999 down, stop before the first significant drop.  WALK=any: smallest r without one.
#     LEVEL=bonf nohup bash run_make_wpre.sh > /dev/null 2>&1 &
# Needs in the repo root: make_wpre.py, wpre_utils.py (new), kdiag_eval.py, bias_probe.py, logs/kdiag/*.npz (committed).
# ONE card (CMLab rule). About 5 min per model (C from SST-2 train, 48 SVDs on the CPU, one SST-2 dev pass), 6 models.
# Writes pz_wpre/<label>_whiten_r<r>.pt/.json (git-ignored, ~200 MB per PZ), logs/wpre/*.json and logs/wpre/choice_<LEVEL>_<WALK>.tsv;
# commit logs/wpre/ and the log. Before it, without a GPU (prints the rule's table for both levels, writes nothing):
#   python make_wpre.py --rule_only --model ... (same --model arguments as below)
: "${LEVEL:?set LEVEL=bonf or LEVEL=per_r}"
WALK="${WALK:-stepwise}"
NAME=make_wpre; source ./v2_common.sh
T="make_wpre whiten level=$LEVEL walk=$WALK panel + clean"
run make_wpre.py "$T" --level "$LEVEL" --walk "$WALK" \
  --model n1500_s0_ep2=pz_epoch/n1500_s0_ep2@2 \
  --model n1500_s1_ep2=pz_cleandev/cleandev_n1500_s1@2 \
  --model n1500_s2_ep2=pz_v2/badnet_s2@2 \
  --model n1500_s3_ep2=pz_cleandev/cleandev_n1500_s3@2 \
  --model n1500_s4_ep2=pz_epoch/n1500_s4_ep2@2 \
  --model clean_n0_s0_ep0=old_models_v1/clean_roberta_large_imdb
if grep -qxF "$T|$CODE_HASH" "$DONE"; then
  cat "logs/wpre/choice_${LEVEL}_${WALK}.tsv" | tee -a "$LOGFILE"
  echo "===== Done =====" | tee -a "$LOGFILE"
else
  echo "'$T' failed (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1
fi