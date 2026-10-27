#!/bin/bash
# run_kdiag_offset.sh — eval only: how much of the SST-2 input energy entering each query/value is a constant offset,
# and the kdiag Tr check redone on the constant and the varying part separately (see kdiag_offset.py).
# No k is chosen, no weight changes, nothing is fine-tuned, SST-2 dev is not used. Same 6 models as run_kdiag.sh.
# Needs in the repo root: kdiag_offset.py, kdiag_eval.py (already committed), and logs/kdiag/*.npz (already committed).
# ONE card (CMLab rule). No SST-2 dev step, so expect less than kdiag's ~3 min per model; the log prints elapsed minutes.
# Writes logs/kdiag_offset/*.npz; commit them and the log.
# Before it, without a GPU:   python kdiag_offset.py --dry <same --model arguments as below>
NAME=kdiag_offset; source ./v2_common.sh
T="kdiag offset panel + clean"
run kdiag_offset.py "$T" \
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