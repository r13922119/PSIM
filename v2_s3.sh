#!/bin/bash
# nohup bash v2_s3.sh > /dev/null 2>&1 &
# S3: choose k for Tr on one PZ, by the rule in PREREG_S3.md, applied by select_k.py (not by eye).
#   S3a  seed 20      : refs (Cl-only, baseline) + Cl+Tr over (lambda {10,480} x 9 k) + paper-equivalence cells  -> select_k.py --stage 1  (prints an effect LABEL; S3_STOP only if no cell is feasible)
#                       paper-equivalence cells = 2 Cl+Tr controls + 6 Tr-only at the paper's "10" read four ways. DESCRIPTIVE, never used to select.
#   S3b  seeds 21-23  : Cl-only reference + Cl+Tr on the shortlist (s3_stage2.txt)  -> select_k.py --stage 2  (prints the effect label; S3_STOP only if no cell is feasible)
# Run count: S3a = 2 refs + 18 + 2 + 6 = 28.  S3b <= 3 seeds x (1 + 6 shortlist cells) = 21.  About 10-13 min each on one Ada (~9-10 h in all).
# Safe to re-run: finished runs are skipped (done_s3.txt). It never overrides S3_STOP: read it, decide, delete it yourself.
# DRY=1 prints what would run and starts nothing.   SKIP_PREREG_CHECK=1 skips the commit gate (do not use for the real run).
NAME=s3; source ./v2_common.sh

# ---------- gates: the rule must be confirmed and committed BEFORE any S3 run ----------
if [ -e S3_STOP ]; then echo "S3_STOP exists:"; cat S3_STOP; echo "Read it, decide, delete the file deliberately, then rerun."; exit 2; fi
if [ "${SKIP_PREREG_CHECK:-0}" != 1 ]; then
  [ -f PREREG_S3.md ] || { echo "PREREG_S3.md not found. Confirm PREREG_S3_draft.md (turn every ☐ into ☑), rename it to PREREG_S3.md, commit it, then run."; exit 3; }
  if grep -q '☐' PREREG_S3.md; then echo "PREREG_S3.md still has unconfirmed ☐ items."; exit 3; fi
  if [ -n "$(git status --porcelain PREREG_S3.md select_k.py 2>/dev/null)" ]; then echo "PREREG_S3.md and/or select_k.py are untracked or have uncommitted changes. Commit both first."; exit 3; fi
fi

PZTAG=$(python select_k.py --pz) || exit 3
PZ="pz_v2/$PZTAG/pytorch_model.bin"
[ -e "$PZ" ] || { echo "missing $PZ"; exit 3; }
sha=$(grep -o '"checkpoint_sha256": "[0-9a-f]*"' "$(dirname "$PZ")/meta.json" 2>/dev/null | grep -o '[0-9a-f]\{64\}' | cut -c1-8)
[ -n "$sha" ] || sha=$(sha256sum "$PZ" | cut -c1-8)

if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2"; }; fi

run_cell() {  # <seed> <variant: clonly|base|cltr|tronly> <lambda|-> <k|->
  local seed="$1" var="$2" lam="$3" k="$4"
  local common=(--pz_path "$PZ" --seed "$seed" --lr 2e-4 --use_spectral_rescaling)
  case "$var" in
    clonly) run variant_finetune.py "$PZTAG sha$sha ft$seed cl-only lr2e-4" "${common[@]}" --use_pretrained_dropout ;;
    base)   run variant_finetune.py "$PZTAG sha$sha ft$seed base lr2e-4" "${common[@]}" ;;
    cltr)   run variant_finetune.py "$PZTAG sha$sha ft$seed cl+tr lam$lam k$k lr2e-4" "${common[@]}" --use_pretrained_dropout --use_orthogonal_penalty --tr_lambda "$lam" --svd_k "$k" ;;
    tronly) run variant_finetune.py "$PZTAG sha$sha ft$seed tr-only lam$lam k$k lr2e-4" "${common[@]}" --use_orthogonal_penalty --tr_lambda "$lam" --svd_k "$k" ;;
    *) echo "unknown variant $var"; exit 3 ;;
  esac
}
mkcsv() {     # <out.csv>: every raw s3 log (not the *_parsed ones), one row per run
  local logs; logs=$(ls logs/${NAME}_2*.log 2>/dev/null | grep -v '_parsed')
  python parse_runs.py $logs --csv > "$1"
}
say() { echo "$*" | tee -a "$LOGFILE"; }

# ---------- S3a ----------
mapfile -t PLAN1 < <(python select_k.py --plan1)
S1_SEED=$(echo "${PLAN1[0]}" | cut -d' ' -f1)
say "S3a: PZ=$PZTAG seed=$S1_SEED, ${#PLAN1[@]} runs (refs, selection grid, paper-equivalence cells)"
for line in "${PLAN1[@]}"; do set -- $line; run_cell "$1" "$2" "$3" "$4"; done

if [ "${DRY:-0}" != 1 ]; then
  mkcsv s3a.csv
  python select_k.py s3a.csv --stage 1 2>&1 | tee -a "$LOGFILE"; rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    if [ "$rc" -eq 2 ]; then say "S3a ended with S3_STOP (no feasible cell, decision point): read S3_STOP and the table above."; else say "S3a INCOMPLETE (some run failed): rerun this script, finished runs are skipped."; fi
    finish; exit "$rc"
  fi
fi

# ---------- S3b ----------
[ -f s3_stage2.txt ] || { say "DRY: no s3_stage2.txt yet, stopping before S3b"; finish; exit 0; }
mapfile -t PLAN2 < s3_stage2.txt
SEEDS2=$(python select_k.py --seeds2)
say "S3b: seeds $SEEDS2, shortlist: ${PLAN2[*]}"
for s in $SEEDS2; do
  run_cell "$s" clonly - -
  for line in "${PLAN2[@]}"; do
    set -- $line; lam=$1; shift
    for k in "$@"; do run_cell "$s" cltr "$lam" "$k"; done
  done
done

if [ "${DRY:-0}" != 1 ]; then
  mkcsv s3b.csv
  python select_k.py s3b.csv --stage 2 2>&1 | tee -a "$LOGFILE"; rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 2 ]; then say "S3b ended with S3_STOP (no feasible cell, decision point): read S3_STOP and the table above."
  elif [ "$rc" -ne 0 ]; then say "S3b INCOMPLETE (some run failed): rerun this script, finished runs are skipped."; fi
fi
finish