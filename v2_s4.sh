#!/bin/bash
# nohup bash v2_s4.sh > /dev/null 2>&1 &
# S4: Tr-only and Cl+Tr at the k chosen in S3 (s3_final.json), lambda 10 and 480, fresh fine-tune seeds 3-5, on the S2 PZ panel.
#   baseline and Cl-only on the same seeds are REUSED from S2 when the training-code hash is unchanged (checked below), otherwise rerun.
#   PZs: oldpz (labeled, never pooled), badnet_s2 (k was tuned on it: label tuned-on), badnet_s0.
#   Order is seed-outer, then oldpz first, so a partial run already gives complete cells for the most informative PZ.
# Run count: 3 seeds x 3 PZ x 4 arms = 36 (plus 18 if the S2 references have to be rerun). About 16 min each on a shared Ada.
# Safe to re-run: finished runs are skipped (done_s4.txt). DRY=1 prints what would run. SKIP_PREREG_CHECK=1 skips the commit gate (not for the real run).
NAME=s4; source ./v2_common.sh

# ---------- gates ----------
[ -f s3_final.json ] || { echo "s3_final.json not found: S3 stage 2 must have finished (python select_k.py s3b.csv --stage 2)."; exit 3; }
if [ "${SKIP_PREREG_CHECK:-0}" != 1 ]; then
  [ -f PREREG_S4.md ] || { echo "PREREG_S4.md not found. Confirm it (turn every unchecked box into a checked one), commit it, then run."; exit 3; }
  if grep -q '☐' PREREG_S4.md; then echo "PREREG_S4.md still has unconfirmed items."; exit 3; fi
  if [ -n "$(git status --porcelain PREREG_S4.md compare_s4.py 2>/dev/null)" ]; then echo "PREREG_S4.md and/or compare_s4.py are untracked or have uncommitted changes. Commit both first."; exit 3; fi
fi

K=$(python -c "import json;print(json.load(open('s3_final.json'))['k'])") || exit 3
LAMS="${LAMS_S4:-10 480}"
SEEDS_S4="${SEEDS_S4:-3 4 5}"
PZS="old_models/poisoned_roberta_large_badnet/pytorch_model.bin pz_v2/badnet_s2/pytorch_model.bin pz_v2/badnet_s0/pytorch_model.bin"

# reuse the S2 baseline / Cl-only only if S2 was run with the same training code
S2_HASHES=$(grep -ho 'code=[0-9a-f]\{8\}' logs/s2_2*.log 2>/dev/null | sort -u | tr '\n' ' ')
if [ "${REFS:-auto}" = rerun ]; then REUSE=0
elif [ "$S2_HASHES" = "code=$CODE_HASH " ]; then REUSE=1
else REUSE=0; fi
if [ "$REUSE" = 1 ]; then REFMSG="S2 baseline and Cl-only reused (same code hash $CODE_HASH)"; else REFMSG="S2 code hash(es) [${S2_HASHES:-none}] differ from current $CODE_HASH (or REFS=rerun): baseline and Cl-only will be RERUN"; fi

if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2"; }; fi
say() { echo "$*" | tee -a "$LOGFILE"; }
mkcsv() {     # <glob prefix> <out.csv>
  local logs; logs=$(ls logs/${1}_2*.log 2>/dev/null | grep -v '_parsed')
  [ -n "$logs" ] && python parse_runs.py $logs --csv > "$2"
}

say "S4: k=$K (from s3_final.json), lambdas [$LAMS], seeds [$SEEDS_S4]; $REFMSG"
for fs in $SEEDS_S4; do
  for pz in $PZS; do
    [ -e "$pz" ] || { say "skip: $pz does not exist"; continue; }
    case "$pz" in pz_v2/*) tag=$(basename "$(dirname "$pz")") ;; *) tag=oldpz ;; esac
    sha=$(grep -o '"checkpoint_sha256": "[0-9a-f]*"' "$(dirname "$pz")/meta.json" 2>/dev/null | grep -o '[0-9a-f]\{64\}' | cut -c1-8)
    [ -n "$sha" ] || sha=$(sha256sum "$pz" | cut -c1-8)
    common=(--pz_path "$pz" --seed "$fs" --lr 2e-4 --use_spectral_rescaling)
    if [ "$REUSE" = 0 ]; then
      run variant_finetune.py "$tag sha$sha ft$fs base lr2e-4" "${common[@]}"
      run variant_finetune.py "$tag sha$sha ft$fs cl lr2e-4"   "${common[@]}" --use_pretrained_dropout
    fi
    for lam in $LAMS; do
      run variant_finetune.py "$tag sha$sha ft$fs cl+tr lam$lam k$K lr2e-4" "${common[@]}" --use_pretrained_dropout --use_orthogonal_penalty --tr_lambda "$lam" --svd_k "$K"
    done
    for lam in $LAMS; do
      run variant_finetune.py "$tag sha$sha ft$fs tr-only lam$lam k$K lr2e-4" "${common[@]}" --use_orthogonal_penalty --tr_lambda "$lam" --svd_k "$K"
    done
  done
done

if [ "${DRY:-0}" != 1 ]; then
  mkcsv s2 s2.csv; mkcsv s4 s4.csv
  if [ -f s2.csv ] && [ -f s4.csv ]; then python compare_s4.py s2.csv s4.csv 2>&1 | tee -a "$LOGFILE"
  else say "could not build s2.csv / s4.csv, run compare_s4.py by hand"; fi
fi
finish