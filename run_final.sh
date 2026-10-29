#!/bin/bash
# run_final.sh — the algorithm's end-to-end test, after P is picked (psel_analyze.py). Per panel PZ, fine-tune seeds 0-2:
#   rora   RoRA on W_pre,K: Cl at P + Tr lambda 10 (kauto) + Pt     seeds 1, 2 here; seed 0 is the run_psel.sh run at p = P
#          (identical arguments, so final_analyze.py finds it by experiment name)
#   konly  plain LoRA on W_pre,K (no Cl, no Tr) + Pt                 seeds 0, 1, 2: what K alone does
#   (plain LoRA on the ORIGINAL PZ, seeds 0-2, already exists: the persistence runs)
# Needs logs/psel/choice_P_<LEVEL>_<WALK>_<STAT>_<RULE>.tsv from psel_analyze.py, pz_wpre/ from run_make_wpre.sh.
#   LEVEL=bonf nohup bash run_final.sh > /dev/null 2>&1 &        (WALK, STAT, RULE default to stepwise, last5, stepwise)
# ONE card (CMLab rule). 25 runs: 15 konly (~9-10 min each, like plain LoRA) + 10 rora (~11-17 min) -> about 4.5-5.5 h.
# Seed outer, then PZ, then arm, so stopping early leaves whole seeds. ARMS="konly" or SEEDS="1" for a subset; DRY=1 prints only.
: "${LEVEL:?set LEVEL=bonf or LEVEL=per_r (the same as before)}"
WALK="${WALK:-stepwise}"; STAT="${STAT:-last5}"; RULE="${RULE:-stepwise}"
NAME=final; source ./v2_common.sh
CHOICE="logs/psel/choice_P_${LEVEL}_${WALK}_${STAT}_${RULE}.tsv"
[ -f "$CHOICE" ] || { echo "$CHOICE missing: run psel_analyze.py first" | tee -a "$LOGFILE"; exit 1; }
SEEDS="${SEEDS:-0 1 2}"; ARMS="${ARMS:-konly rora}"; LAM=10
if [ "${DRY:-0}" = 1 ]; then run() { echo "DRY run: $2 | ${*:3}"; }; fi
cat "$CHOICE" | tee -a "$LOGFILE"
mapfile -t ROWS < <(tail -n +2 "$CHOICE")      # read once: nothing inside the loop can consume the file through stdin
for fs in $SEEDS; do
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r label r J PZ P <<< "$row"
    [ -f "$J" ] || { echo "$J missing, skipping $label" | tee -a "$LOGFILE"; continue; }
    for arm in $ARMS; do
      case $arm in
        konly) run variant_finetune.py "final $label wh$r ft$fs konly lr2e-4" --pz_path "$PZ" --wpre_file "$J" --seed "$fs" --lr 2e-4 \
                 --use_spectral_rescaling ;;
        rora)  if [ "$fs" = 0 ]; then echo "rora seed 0 for $label = the run_psel.sh run at p = $P (not rerun)" | tee -a "$LOGFILE"; continue; fi
               run variant_finetune.py "final $label wh$r ft$fs rora p$P lam$LAM kauto lr2e-4" --pz_path "$PZ" --wpre_file "$J" --seed "$fs" \
                 --lr 2e-4 --use_pretrained_dropout --cl_dropout_p "$P" --use_orthogonal_penalty --tr_lambda "$LAM" --svd_k_auto \
                 --use_spectral_rescaling ;;
      esac
    done
  done
done
[ "${DRY:-0}" = 1 ] || finish