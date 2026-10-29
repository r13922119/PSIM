#!/bin/bash
# run_psel.sh (v2, 10/29) — P grid on each panel PZ's W_pre,K, fine-tune seeds 0-2, both Cl forms, plus the K-only arm.
# psel_analyze.py later picks P from SST-2 DEV accuracy only (pre-Pt); test accuracy and ASR are logged by variant_finetune.py
# as always but do not pick anything. Every run: --wpre_file (W_pre,K from run_make_wpre.sh), lr 2e-4, Pt on (measured after).
# Arms (ARMS chooses which ones this queue runs):
#   tr     Tr alone (lambda 10, kauto), Cl off: the p = 0 reference for both forms
#   coord  Cl coord (dropout on every output coordinate of W_pre,K x + b) at each p of PS, + Tr
#   svd    Cl svd (dropout on the components along W_pre,K's own left singular directions) at each p of PS, + Tr
#   konly  plain LoRA on W_pre,K (no Cl, no Tr): what K alone does
# The RoRA arm of the final test is this grid at p = P (seeds 0-2), so no separate final run is needed for it.
# Order: seed outer (all arms of seed 0, then seed 1, then seed 2), so if it stops early every configuration has the same
# number of seeds; inside a seed: arm, then p, then PZ; tr goes first, so the first 5 runs show whether Tr + Pt collapses.
# Needs: logs/wpre/choice_<LEVEL>_<WALK>.tsv and pz_wpre/ (run_make_wpre.sh with the same LEVEL), variant_finetune.py with
#   --wpre_file and --cl_form, wpre_utils.py, cl_utils.py.
# One queue (all 180 runs; ~11-17 min each judging by the 10/19 pilot, plain LoRA ~9 min -> about 35 h):
#   LEVEL=bonf nohup bash run_psel.sh > /dev/null 2>&1 &
# Two queues on the SAME card (CMLab rule: one card; pick_gpu_ada reuses the card you hold). Arms are disjoint, so no run is
# done twice; each queue has its own log and done file. Whether this is faster depends on how much each run slows down when
# sharing; compare duration_min of the first runs with the pilot's 11 min:
#   LEVEL=bonf QUEUE=A ARMS="tr coord konly" nohup bash run_psel.sh > /dev/null 2>&1 &     # 105 runs
#   LEVEL=bonf QUEUE=B ARMS="svd"            nohup bash run_psel.sh > /dev/null 2>&1 &     # 75 runs
# Subsets: SEEDS="0", PS="0.1 0.2", LABELS="n1500_s0_ep2". DRY=1 prints what would run.
: "${LEVEL:?set LEVEL=bonf or LEVEL=per_r (the same as for run_make_wpre.sh)}"
WALK="${WALK:-stepwise}"
NAME="psel${QUEUE:+_$QUEUE}"; source ./v2_common.sh
CHOICE="logs/wpre/choice_${LEVEL}_${WALK}.tsv"
[ -f "$CHOICE" ] || { echo "$CHOICE missing: run run_make_wpre.sh with LEVEL=$LEVEL first" | tee -a "$LOGFILE"; exit 1; }
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-tr coord konly svd}"
PS="${PS:-0.05 0.1 0.15 0.2 0.3}"
LABELS="${LABELS:-n1500_s0_ep2 n1500_s1_ep2 n1500_s2_ep2 n1500_s3_ep2 n1500_s4_ep2}"
LAM=10
if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2 | ${*:3}"; }; fi
cat "$CHOICE" | tee -a "$LOGFILE"
echo "queue ${QUEUE:-all}: arms [$ARMS], seeds [$SEEDS], p [$PS]" | tee -a "$LOGFILE"
mapfile -t ROWS < <(tail -n +2 "$CHOICE")      # read once: nothing inside the loop can consume the file through stdin
one() {   # one <label> -> sets r J PZ, or returns 1
  local row; row=$(printf '%s\n' "${ROWS[@]}" | awk -F'\t' -v l="$1" '$1==l')
  [ -n "$row" ] || { echo "no W_pre,K for $1 in $CHOICE, skipping" | tee -a "$LOGFILE"; return 1; }
  r=$(echo "$row" | cut -f2); J=$(echo "$row" | cut -f3); PZ=$(echo "$row" | cut -f4)
  [ -f "$J" ] || { echo "$J missing, skipping $1" | tee -a "$LOGFILE"; return 1; }
}
for fs in $SEEDS; do
  for arm in $ARMS; do
    case $arm in
      tr|konly) plist="-" ;;
      coord|svd) plist="$PS" ;;
      *) echo "unknown arm $arm" | tee -a "$LOGFILE"; exit 1 ;;
    esac
    for p in $plist; do
      for label in $LABELS; do
        one "$label" || continue
        base=(--pz_path "$PZ" --wpre_file "$J" --seed "$fs" --lr 2e-4 --use_spectral_rescaling)
        tr=(--use_orthogonal_penalty --tr_lambda "$LAM" --svd_k_auto)
        case $arm in
          tr)    run variant_finetune.py "psel $label wh$r p0 ft$fs tr lam$LAM kauto lr2e-4" "${base[@]}" "${tr[@]}" ;;
          konly) run variant_finetune.py "psel $label wh$r ft$fs konly lr2e-4" "${base[@]}" ;;
          coord) run variant_finetune.py "psel $label wh$r p$p ft$fs cl+tr lam$LAM kauto lr2e-4" "${base[@]}" "${tr[@]}" \
                   --use_pretrained_dropout --cl_dropout_p "$p" ;;
          svd)   run variant_finetune.py "psel $label wh$r p$p ft$fs clsvd+tr lam$LAM kauto lr2e-4" "${base[@]}" "${tr[@]}" \
                   --use_pretrained_dropout --cl_dropout_p "$p" --cl_form svd ;;
        esac
      done
    done
  done
done
[ "${DRY:-0}" = 1 ] || finish