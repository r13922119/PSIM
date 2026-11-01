#!/bin/bash
# run_psel3.sh (v3, 11/01; = run_psel.sh v2 + no-Tr arms + Eq. 9 arms + no-K arms) — P grid on each panel PZ's W_pre,K, fine-tune seeds 0-2, both Cl forms, plus the K-only arm.
# psel_analyze.py later picks P from SST-2 DEV accuracy only (pre-Pt); test accuracy and ASR are logged by variant_finetune.py
# as always but do not pick anything. Every run: --wpre_file (W_pre,K from run_make_wpre.sh), lr 2e-4, Pt on (measured after).
# Arms (ARMS chooses which ones this queue runs):
#   tr     Tr alone (lambda 10, kauto), Cl off: the p = 0 reference for both forms
#   coord  Cl coord (dropout on every output coordinate of W_pre,K x + b) at each p of PS, + Tr
#   svd    Cl svd (dropout on the components along W_pre,K's own left singular directions) at each p of PS, + Tr
#   konly  plain LoRA on W_pre,K (no Cl, no Tr): what K alone does
#   cl     Cl coord at each p of PS, NO Tr   (added 10/30: K + Cl, so p's effect is not mixed with Tr's; Tr's form is still open)
#   clsvd  Cl svd   at each p of PS, NO Tr   (added 10/30)
#   ocl, oclsvd  Cl coord / svd at each p of PS on the ORIGINAL W_pre (no K, no Tr; added 11/01): does K add anything once Cl is there?
#          (Tr is defined through W_pre,K, Cl is not.) svd without K = dropout along all 1024 left singular directions of W_pre.
#   trp, clp, clsvdp  the same as tr, coord, svd but with Tr as the exact constraint of RoRA Eq. 9 (--tr_proj: no lambda, no d/k, no
#          mean/sum; added 10/30 18:30). Needs the 10/30 variant_finetune.py; run these from ANOTHER folder while queues that use the
#          old variant_finetune.py are still running (the queue fixes its code hash at start but every run re-reads the file).
# Each queue has its own PS, so the two forms can use different p ranges, e.g.
#   LEVEL=bonf QUEUE=C ARMS="cl"    PS="0.05 0.1 0.15 0.2 0.3" nohup bash run_psel.sh > /dev/null 2>&1 &
#   LEVEL=bonf QUEUE=D ARMS="clsvd" PS="0.1 0.2 0.3 0.4 0.5"   nohup bash run_psel.sh > /dev/null 2>&1 &
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
# Subsets: SEEDS="0", PS="0.1 0.2", LABELS="n1500_s0_ep2". DRY=1 prints what would run. ORDER=p: p outer (see the loop).
# Stop everything: pkill -f run_psel3.sh; pkill -f variant_finetune.py   (the run that was cut off is not marked done)
# Start again: the same command; every run already in any logs/done_psel*.txt with the same code hash is skipped.
: "${LEVEL:?set LEVEL=bonf or LEVEL=per_r (the same as for run_make_wpre.sh)}"
WALK="${WALK:-stepwise}"
NAME="psel${QUEUE:+_$QUEUE}"; source ./v2_common.sh
CHOICE="logs/wpre/choice_${LEVEL}_${WALK}.tsv"
ORDER="${ORDER:-seed}"     # seed (default, as before: seed outer) or p (p outer: smaller p first, see below)
run() {   # = run() of v2_common.sh, except that a run counts as done if ANY logs/done_psel*.txt has it (same title and code hash):
          # a queue can be stopped and started again under any QUEUE name or ORDER, and finished runs are never redone
  local script="$1" title="$2"; shift 2
  local key="$title|$CODE_HASH"
  if cat logs/done_psel*.txt 2>/dev/null | grep -qxF "$key"; then echo "skip (done): $title" | tee -a "$LOGFILE"; return 0; fi
  local g; g=$(pick_gpu_ada); [ -n "$g" ] || { echo "no '$EXPECTED_GPU' on this machine" | tee -a "$LOGFILE"; return 1; }
  export CUDA_VISIBLE_DEVICES="$g"
  echo "===== $title | GPU=$g | code=$CODE_HASH | git=$GIT | $* =====" | tee -a "$LOGFILE"
  python "$script" "$@" 2>&1 | tee -a "$LOGFILE"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then echo "$key" >> "$DONE"; else echo "FAILED: $title" | tee -a "$LOGFILE"; fi
}
if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2 | ${*:3}"; }; fi
[ -f "$CHOICE" ] || { echo "$CHOICE missing: run run_make_wpre.sh with LEVEL=$LEVEL first" | tee -a "$LOGFILE"; exit 1; }
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-tr coord konly svd}"
PS="${PS:-0.05 0.1 0.15 0.2 0.3}"
LABELS="${LABELS:-n1500_s0_ep2 n1500_s1_ep2 n1500_s2_ep2 n1500_s3_ep2 n1500_s4_ep2}"
LAM=10
cat "$CHOICE" | tee -a "$LOGFILE"
echo "queue ${QUEUE:-all}: arms [$ARMS], seeds [$SEEDS], p [$PS]" | tee -a "$LOGFILE"
mapfile -t ROWS < <(tail -n +2 "$CHOICE")      # read once: nothing inside the loop can consume the file through stdin
one() {   # one <label> -> sets r J PZ, or returns 1
  local row; row=$(printf '%s\n' "${ROWS[@]}" | awk -F'\t' -v l="$1" '$1==l')
  [ -n "$row" ] || { echo "no W_pre,K for $1 in $CHOICE, skipping" | tee -a "$LOGFILE"; return 1; }
  r=$(echo "$row" | cut -f2); J=$(echo "$row" | cut -f3); PZ=$(echo "$row" | cut -f4)
  [ -f "$J" ] || { echo "$J missing, skipping $1" | tee -a "$LOGFILE"; return 1; }
}
one_run() {   # one_run <arm> <p or -> <seed> <label>
  local arm="$1" p="$2" fs="$3" label="$4"
    one "$label" || return 0
    base=(--pz_path "$PZ" --wpre_file "$J" --seed "$fs" --lr 2e-4 --use_spectral_rescaling)
    tr=(--use_orthogonal_penalty --tr_lambda "$LAM" --svd_k_auto)
    trp=(--tr_proj --svd_k_auto)
    orig=(--pz_path "$PZ" --seed "$fs" --lr 2e-4 --use_spectral_rescaling)     # no --wpre_file: the PZ's own W_pre
    case $arm in
      tr)    run variant_finetune.py "psel $label wh$r p0 ft$fs tr lam$LAM kauto lr2e-4" "${base[@]}" "${tr[@]}" ;;
      konly) run variant_finetune.py "psel $label wh$r ft$fs konly lr2e-4" "${base[@]}" ;;
      coord) run variant_finetune.py "psel $label wh$r p$p ft$fs cl+tr lam$LAM kauto lr2e-4" "${base[@]}" "${tr[@]}" \
               --use_pretrained_dropout --cl_dropout_p "$p" ;;
      svd)   run variant_finetune.py "psel $label wh$r p$p ft$fs clsvd+tr lam$LAM kauto lr2e-4" "${base[@]}" "${tr[@]}" \
               --use_pretrained_dropout --cl_dropout_p "$p" --cl_form svd ;;
      ocl)   run variant_finetune.py "psel $label orig p$p ft$fs cl lr2e-4" "${orig[@]}" \
               --use_pretrained_dropout --cl_dropout_p "$p" ;;
      oclsvd) run variant_finetune.py "psel $label orig p$p ft$fs clsvd lr2e-4" "${orig[@]}" \
               --use_pretrained_dropout --cl_dropout_p "$p" --cl_form svd ;;
      trp)   run variant_finetune.py "psel $label wh$r p0 ft$fs trproj kauto lr2e-4" "${base[@]}" "${trp[@]}" ;;
      clp)   run variant_finetune.py "psel $label wh$r p$p ft$fs cl+trproj kauto lr2e-4" "${base[@]}" "${trp[@]}" \
               --use_pretrained_dropout --cl_dropout_p "$p" ;;
      clsvdp) run variant_finetune.py "psel $label wh$r p$p ft$fs clsvd+trproj kauto lr2e-4" "${base[@]}" "${trp[@]}" \
               --use_pretrained_dropout --cl_dropout_p "$p" --cl_form svd ;;
      cl)    run variant_finetune.py "psel $label wh$r p$p ft$fs cl lr2e-4" "${base[@]}" \
               --use_pretrained_dropout --cl_dropout_p "$p" ;;
      clsvd) run variant_finetune.py "psel $label wh$r p$p ft$fs clsvd lr2e-4" "${base[@]}" \
               --use_pretrained_dropout --cl_dropout_p "$p" --cl_form svd ;;
    esac
}
plist_of() {
  case $1 in
    tr|konly|trp) echo "-" ;;
    coord|svd|cl|clsvd|clp|clsvdp|ocl|oclsvd) echo "$PS" ;;
    *) echo "unknown arm $1" | tee -a "$LOGFILE"; exit 1 ;;
  esac
}
for arm in $ARMS; do plist_of "$arm" > /dev/null || exit 1; done
if [ "$ORDER" = p ]; then
  # p outer (ascending as given in PS), then seed, then arm, then PZ: the P rule walks p upward and a larger p only matters if every
  # smaller one passed, so the smaller p are needed first; inside one p every arm gets the same seeds before the next p starts.
  for arm in $ARMS; do [ "$(plist_of "$arm")" = "-" ] && { echo "ORDER=p needs arms with a p ($arm has none)" | tee -a "$LOGFILE"; exit 1; }; done
  for p in $PS; do for fs in $SEEDS; do for arm in $ARMS; do for label in $LABELS; do one_run "$arm" "$p" "$fs" "$label"; done; done; done; done
else
  for fs in $SEEDS; do for arm in $ARMS; do for p in $(plist_of "$arm"); do for label in $LABELS; do one_run "$arm" "$p" "$fs" "$label"; done; done; done; done
fi
[ "${DRY:-0}" = 1 ] || finish