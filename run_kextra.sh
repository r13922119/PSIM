#!/bin/bash
# run_kextra.sh — simultaneous AND sequential truncation, four orderings x ten r, with: room for a rank-8 Delta W under Tr,
# the between-class (positive vs negative) share outside, and ASR (A) of the 444 triggered SST-2 dev positives under each
# truncation (see kextra_eval.py). Supersedes run_kseq_full.sh: its sequential part must reproduce logs/kseq_full (or
# logs/kseq) exactly, and its simultaneous part must reproduce logs/kdiag. Diagnostic only; nothing is chosen or saved as a model.
# Needs in the repo root: kextra_eval.py, kseq_eval.py (v2), kdiag_eval.py; logs/kdiag/, logs/kdiag_offset/, logs/scores/sst2dev/.
# ONE card (CMLab rule). Rough guess 55-70 min per model, about 6-7 hours for 6; the log prints elapsed minutes.
# Writes logs/kextra/*.npz; commit them and the log.
# Before it, without a GPU:   python kextra_eval.py --dry <same --model arguments as below>
NAME=kextra; source ./v2_common.sh
T="kextra panel + clean"
run kextra_eval.py "$T" \
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