#!/bin/bash
# run_kseq.sh — layer-by-layer ("sequential") truncation vs kdiag's simultaneous truncation (see kseq_eval.py).
# Diagnostic only: no k is chosen, no saved weight changes, nothing is fine-tuned. SST-2 train builds C (no labels);
# SST-2 dev only scores the temporarily truncated model. Same 6 models as run_kdiag.sh.
# Default: orderings sigma and whiten, r = 0.999 ... 0.7 (8 values) = 16 sequential passes per model.
# Needs in the repo root: kseq_eval.py, kdiag_eval.py (committed), logs/kdiag/*.npz (committed).
# ONE card (CMLab rule). Rough guess 15 min per model, ~1.5 h for 6 (my guesses have been off; the log prints elapsed
# minutes after every pass, so the first model shows the real speed).
# Writes logs/kseq/*.npz; commit them and the log.
# Before it, without a GPU:   python kseq_eval.py --dry <same --model arguments as below>
NAME=kseq; source ./v2_common.sh
T="kseq panel + clean"
run kseq_eval.py "$T" \
  --model n1500_s0_ep2=pz_epoch/n1500_s0_ep2@2 \
  --model n1500_s1_ep2=pz_cleandev/cleandev_n1500_s1@2 \
  --model n1500_s2_ep2=pz_v2/badnet_s2@2 \
  --model n1500_s3_ep2=pz_cleandev/cleandev_n1500_s3@2 \
  --model n1500_s4_ep2=pz_epoch/n1500_s4_ep2@2 \
  --model clean_n0_s0_ep0=old_models_v1/clean_roberta_large_imdb
if grep -qxF "$T|$CODE_HASH" "$DONE"; then
  echo "===== Done =====" | tee -a "$LOGFILE"
else
  echo "'$T' failed (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1
fi