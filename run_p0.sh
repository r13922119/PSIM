#!/bin/bash
# run_p0.sh — step-0 damage of Cl on each W_pre,K, both forms (p0_eval.py). Eval only, nothing is trained or chosen.
# Gives the common horizontal axis for comparing the coord and svd forms (how much each p damages the model at step 0),
# and the data to check whether step-0 damage predicts post-fine-tuning accuracy (a proxy that could replace the P grid).
# Needs: logs/wpre/choice_<LEVEL>_<WALK>.tsv and pz_wpre/ from run_make_wpre.sh, cl_utils.py, wpre_utils.py.
# ONE card (CMLab rule); it can share the card with run_psel.sh. Rough guess 10-15 min per model, 6 models (clean included).
#   LEVEL=bonf nohup bash run_p0.sh > /dev/null 2>&1 &
# Writes logs/p0/*.npz; commit them and the log.
: "${LEVEL:?set LEVEL=bonf or LEVEL=per_r (the same as for run_make_wpre.sh)}"
WALK="${WALK:-stepwise}"
NAME=p0; source ./v2_common.sh
T="p0 coord+svd level=$LEVEL walk=$WALK"
run p0_eval.py "$T" --choice "logs/wpre/choice_${LEVEL}_${WALK}.tsv"
if grep -qxF "$T|$CODE_HASH" "$DONE"; then
  echo "===== Done =====" | tee -a "$LOGFILE"
else
  echo "'$T' failed (see $LOGFILE)" | tee -a "$LOGFILE"; exit 1
fi