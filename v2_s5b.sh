#!/bin/bash
# nohup bash v2_s5b.sh > /dev/null 2>&1 &
# S5b: baseline and Cl-only at lr 2e-5 (in the paper grid), no Tr, to match the Tr update size in the last 3 layers (PREREG_S5b.md).
#   Compared (by compare_s5.py) with baseline / Cl-only at lr 2e-4 (S2) and Tr-only / Cl+Tr at lr 2e-4 (S4), all on the same seeds.
#   PZs: oldpz (labeled, never pooled), badnet_s2 (tuned-on in S3), badnet_s0.
#   Order is seed-outer, then oldpz first, so a partial run already gives complete cells for the most informative PZ.
# Run count: 3 seeds x 3 PZ x 2 lr x 2 arms (baseline, Cl-only) = 36. About 10-16 min each on a shared Ada, so roughly 6-10 hours.
# Safe to re-run: finished runs are skipped (done_s5b.txt). DRY=1 prints what would run. SKIP_PREREG_CHECK=1 skips the commit gate (not for the real run).
NAME=s5b; source ./v2_common.sh

# ---------- gates ----------
if [ "${SKIP_PREREG_CHECK:-0}" != 1 ]; then
  [ -f PREREG_S5b.md ] || { echo "PREREG_S5b.md not found. Confirm it (turn every unchecked box into a checked one), commit it, then run."; exit 3; }
  if grep -q '☐' PREREG_S5b.md; then echo "PREREG_S5b.md still has unconfirmed items."; exit 3; fi
  if [ -n "$(git status --porcelain PREREG_S5b.md compare_s5.py 2>/dev/null)" ]; then echo "PREREG_S5b.md and/or compare_s5.py are untracked or have uncommitted changes. Commit both first."; exit 3; fi
fi
# The lr 2e-4 references (S2, S4) are only comparable if they were run with the same training code.
REF_HASHES=$(grep -ho 'code=[0-9a-f]\{8\}' logs/s2_2*.log logs/s4_2*.log logs/s5_2*.log 2>/dev/null | sort -u | tr '\n' ' ')
if [ "$REF_HASHES" != "code=$CODE_HASH " ] && [ "${ALLOW_HASH_MISMATCH:-0}" != 1 ]; then
  echo "S2/S4 code hash(es) [${REF_HASHES:-none}] differ from the current code=$CODE_HASH. S5 would not be comparable. Stop, or set ALLOW_HASH_MISMATCH=1 and say so in the report."; exit 3
fi

LRS="${LRS_S5B:-2e-5}"
SEEDS_S5B="${SEEDS_S5B:-3 4 5}"
PZS="old_models/poisoned_roberta_large_badnet/pytorch_model.bin pz_v2/badnet_s2/pytorch_model.bin pz_v2/badnet_s0/pytorch_model.bin"

if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2"; }; fi
say() { echo "$*" | tee -a "$LOGFILE"; }
mkcsv() {     # <glob prefix> <out.csv>
  local logs; logs=$(ls logs/${1}_2*.log 2>/dev/null | grep -v '_parsed')
  [ -n "$logs" ] && python parse_runs.py $logs --csv > "$2"
}

say "S5: lrs [$LRS], seeds [$SEEDS_S5B], arms baseline and Cl-only, no Tr; code=$CODE_HASH"
for fs in $SEEDS_S5B; do
  for pz in $PZS; do
    [ -e "$pz" ] || { say "skip: $pz does not exist"; continue; }
    case "$pz" in pz_v2/*) tag=$(basename "$(dirname "$pz")") ;; *) tag=oldpz ;; esac
    sha=$(grep -o '"checkpoint_sha256": "[0-9a-f]*"' "$(dirname "$pz")/meta.json" 2>/dev/null | grep -o '[0-9a-f]\{64\}' | cut -c1-8)
    [ -n "$sha" ] || sha=$(sha256sum "$pz" | cut -c1-8)
    for lr in $LRS; do
      common=(--pz_path "$pz" --seed "$fs" --lr "$lr" --use_spectral_rescaling)
      run variant_finetune.py "$tag sha$sha ft$fs base lr$lr" "${common[@]}"
      run variant_finetune.py "$tag sha$sha ft$fs cl lr$lr"   "${common[@]}" --use_pretrained_dropout
    done
  done
done

if [ "${DRY:-0}" != 1 ]; then
  mkcsv s2 s2.csv; mkcsv s4 s4.csv; mkcsv s5 s5.csv; mkcsv s5b s5b.csv
  if [ -f s2.csv ] && [ -f s4.csv ] && [ -f s5.csv ] && [ -f s5b.csv ]; then python compare_s5.py s2.csv s4.csv s5.csv s5b.csv --match top3 $( [ -f dw_s5b.txt ] && echo --dw dw_s5b.txt ) 2>&1 | tee -a "$LOGFILE"
  else say "could not build s2.csv / s4.csv / s5.csv / s5b.csv, run compare_s5.py by hand"; fi
fi
finish