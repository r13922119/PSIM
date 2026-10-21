#!/bin/bash
# gate_merge.sh — check the merged variant_finetune.py against runs that are already logged (no new reference needed).
#   part 1  default flags  vs the v2 golden record (baseline, oldpz, seed 0, 20 epochs)        golden/baseline_0_oldpz_ep20_20261002.txt   ~13 min
#   part 2  W_pre,k + Cl + Tr(kauto) + Pt  vs the pilot run "oldpz wkf-grp6 ft3 cl+tr lam10 kauto lr2e-4"  logs/pilot_wk_20261019_150838.log  ~14 min
#           (the W_pre,k code path has never been rerun, so this is also its first determinism check)
#   part 3  obs logging (--obs_log, --tr_layers top3)  vs the three logged arms of gate_obs.sh on 10/12   logs/gateObs_20261012_100917/  ~10 min
# Compared: [RNG] fingerprints, dev acc, [DEBUG_ep] lines (task loss, penalty), [CHECK] hook count, [W_pre,k] lines, [EPOCH] lines
# (dev, test acc, ASR), the SUMMARY line without duration_min; part 1 only has [RNG], [EPOCH], SUMMARY in the golden record.
# Part 3 also compares the logger's epochs.csv, modules.csv, steps.csv byte for byte.
# Run on an RTX 6000 Ada (same model as the references). Parts 1 and 3 save nothing; part 2 saves its adapter under the gate folder.
# Logs in logs/gateMerge_<date>/.
#   bash gate_merge.sh          all three parts
#   bash gate_merge.sh 2 3      only parts 2 and 3
set -u
export PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export EXPECTED_GPU="${EXPECTED_GPU:-NVIDIA RTX 6000 Ada Generation}"
PARTS="${*:-1 2 3}"
GOLD=golden/baseline_0_oldpz_ep20_20261002.txt
PILOT=logs/pilot_wk_20261019_150838.log
PILOT_TITLE="oldpz wkf-grp6 ft3 cl+tr lam10 kauto lr2e-4"
OBSREF=logs/gateObs_20261012_100917
OUT="logs/gateMerge_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
FAILS=0

pick_gpu_ada() {   # UUID of the Ada card to use. CMLab rule: at most 50 GB of GPU per person across all workstations,
                   # counted by each card's full VRAM, i.e. ONE 48 GB card. If this user already has a process on a card,
                   # reuse that card (share it) instead of taking a second one; refuse if that card is not an Ada.
  local me held name
  me=$(id -u)
  held=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=', ' read -r pid uuid; do [ "$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$me" ] && echo "$uuid"; done | sort -u)
  if [ -n "$held" ]; then
    [ "$(echo "$held" | wc -l)" -gt 1 ] && echo "WARNING: you already hold more than one GPU ($(echo $held)); the CMLab limit is one" >&2
    held=$(echo "$held" | head -1)
    name=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', *' -v u="$held" '$1==u {print $2}')
    if [ "$name" = "$EXPECTED_GPU" ]; then echo "$held"; else echo "you already hold a non-Ada GPU ($held, $name); not taking a second card" >&2; fi
    return
  fi
  nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader,nounits \
    | awk -F', *' -v n="$EXPECTED_GPU" '$2==n {print $3, $1}' \
    | sort -n | head -1 | cut -d' ' -f2
}
G=$(pick_gpu_ada); [ -n "$G" ] || { echo "no '$EXPECTED_GPU' found on this machine" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$G"
echo "gate (merge) on GPU=$G, parts: $PARTS, logs in $OUT"
[ -f variant_finetune_obs.py ] && echo "note: variant_finetune_obs.py still exists; this gate only runs variant_finetune.py"

canon() {   # trainer-side and observer-side lines; tqdm uses \r so split it first; duration is the only field allowed to differ
  tr '\r' '\n' < "$1" | sed 's/^[[:space:]]*//' \
    | grep -E '^\[RNG\]|^dev clean acc|^\[DEBUG_ep|^\[CHECK\]|^\[W_pre,k\]|^\[EPOCH\]|^SUMMARY' \
    | sed -E 's/ \| duration_min=[^|]*//'
}
fail() { echo "  FAIL: $*"; FAILS=$((FAILS+1)); }
same() {   # same <name> <reference lines file> <new lines file>
  if [ ! -s "$2" ]; then fail "$1: reference has no comparable lines"; return 1; fi
  if diff "$2" "$3" > "$OUT/$1.diff"; then echo "  PASS ($(wc -l < "$2") lines identical)"; else
    fail "$1: differs (first lines of $OUT/$1.diff):"; head -10 "$OUT/$1.diff"; return 1; fi
}

for part in $PARTS; do
  case $part in
  1)
    echo "== part 1: default flags vs $GOLD"
    python variant_finetune.py --seed 0 --num_epochs 20 --no_save --pz_path old_models/poisoned_roberta_large_badnet/pytorch_model.bin \
      > "$OUT/p1.log" 2>&1 || { fail "part 1 crashed (see $OUT/p1.log)"; continue; }
    grep -E '^\[RNG\]|^\[EPOCH\]|^SUMMARY' "$GOLD" | sed -E 's/ \| duration_min=[^|]*//' > "$OUT/p1.ref"
    canon "$OUT/p1.log" | grep -E '^\[RNG\]|^\[EPOCH\]|^SUMMARY' > "$OUT/p1.new"
    same p1 "$OUT/p1.ref" "$OUT/p1.new" ;;
  2)
    echo "== part 2: W_pre,k path vs '$PILOT_TITLE' in $PILOT"
    awk -v t="===== $PILOT_TITLE |" 'index($0, t)==1 {on=1; print; next} on && /^===== / {exit} on {print}' "$PILOT" > "$OUT/p2.ref.log"
    hdr=$(head -1 "$OUT/p2.ref.log")
    [ -n "$hdr" ] || { fail "part 2: run not found in $PILOT"; continue; }
    args=$(echo "$hdr" | sed -E 's/^===== .* \| git=[^|]* \| (.*) =====$/\1/')
    # Pt needs a saved adapter (the script refuses --no_save with Pt), so the adapter goes under $OUT via --obs_log;
    # nothing under adapters/ is touched. The logger is read-only (gate_obs.sh, 10/12), and part 2 then also checks it next to W_pre,k.
    echo "  args: $args --obs_log $OUT/p2obs --obs_every 100000"
    # shellcheck disable=SC2086
    python variant_finetune.py $args --obs_log "$OUT/p2obs" --obs_every 100000 > "$OUT/p2.log" 2>&1 || { fail "part 2 crashed (see $OUT/p2.log)"; continue; }
    canon "$OUT/p2.ref.log" > "$OUT/p2.ref"; canon "$OUT/p2.log" > "$OUT/p2.new"
    same p2 "$OUT/p2.ref" "$OUT/p2.new" ;;
  3)
    for name in base tr trtop3; do
      echo "== part 3: obs logging, arm $name vs $OBSREF/$name.on.log"
      ref="$OBSREF/$name.on.log"; [ -f "$ref" ] || { fail "part 3: $ref missing"; continue; }
      ep=$(canon "$ref" | grep -c '^\[EPOCH\]')
      a=(--seed 3 --num_epochs "$ep" --no_save --pz_path pz_v2/badnet_s2/pytorch_model.bin --lr 2e-4 --svd_k 8)
      [ "$name" = tr ] && a+=(--use_orthogonal_penalty --tr_lambda 10)
      [ "$name" = trtop3 ] && a+=(--use_orthogonal_penalty --tr_lambda 10 --tr_layers top3)
      python variant_finetune.py "${a[@]}" --obs_log "$OUT/obs_$name" > "$OUT/p3_$name.log" 2>&1 \
        || { fail "part 3 $name crashed (see $OUT/p3_$name.log)"; continue; }
      canon "$ref" > "$OUT/p3_$name.ref"; canon "$OUT/p3_$name.log" > "$OUT/p3_$name.new"
      same "p3_$name" "$OUT/p3_$name.ref" "$OUT/p3_$name.new"
      for f in epochs.csv modules.csv steps.csv; do
        r=$(ls "$OBSREF/obs_$name"/*/"$f" 2>/dev/null | head -1); n=$(ls "$OUT/obs_$name"/*/"$f" 2>/dev/null | head -1)
        if [ -z "$r" ] || [ -z "$n" ]; then fail "part 3 $name: $f missing"; elif cmp -s "$r" "$n"; then echo "  PASS $f identical";
        else fail "part 3 $name: $f differs"; fi
      done
    done ;;
  *) echo "unknown part $part" ;;
  esac
done

echo
[ "$FAILS" -eq 0 ] && echo "GATE PASSED: the merged variant_finetune.py reproduces every reference run." \
                   || { echo "GATE FAILED ($FAILS problem(s)): do not use the merged file for new runs; send me $OUT."; exit 1; }