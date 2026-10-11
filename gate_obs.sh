#!/bin/bash
# Gate for obs_log.py, in the style of gate_a.sh:  bash gate_obs.sh        (about 8 to 10 minutes on one Ada card)
# For two arms (baseline, Tr-only) run variant_finetune_obs.py once WITH --obs_log and once WITHOUT, same card, same seed, 2 epochs, nothing saved
# except the logs. Every trainer-side line AND every observer-side line must be identical:
#   [RNG] fingerprints, dev acc, [DEBUG_ep] lines (task loss, penalty), [EPOCH] lines (dev, test acc, ASR).
# Then: the logger's penalty must match the training log's [DEBUG] penalty, and the log files must have the expected shape.
# If anything FAILS, do not use the logs; send me the diff file named in the message.
set -u
export PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export EXPECTED_GPU="${EXPECTED_GPU:-NVIDIA RTX 6000 Ada Generation}"
PZ="${PZ:-pz_v2/badnet_s2/pytorch_model.bin}"
EPOCHS="${EPOCHS:-2}"
OUT="logs/gateObs_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
FAILS=0

pick_gpu_ada() {
  nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader,nounits \
    | awk -F', *' -v n="$EXPECTED_GPU" '$2==n {print $3, $1}' \
    | sort -n | head -1 | cut -d' ' -f2
}
G=$(pick_gpu_ada); [ -n "$G" ] || { echo "no '$EXPECTED_GPU' found on this machine" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$G"
echo "gate (obs) on GPU=$G, logs in $OUT"

canon() { tr '\r' '\n' < "$1" | sed 's/^[[:space:]]*//' | grep -E '^\[RNG\]|^dev clean acc|^\[DEBUG_ep|^\[CHECK\]|^\[EPOCH\]'; }
fail() { echo "  FAIL: $*"; FAILS=$((FAILS+1)); }

check_files() {   # check_files <obs_dir> <epochs>
  python - "$1" "$2" <<'EOF'
import csv, glob, sys
d, n = sys.argv[1], int(sys.argv[2])
runs = glob.glob(d + "/*/epochs.csv")
assert len(runs) == 1, f"expected one run dir under {d}, found {runs}"
base = runs[0].rsplit("/", 1)[0]
ep = list(csv.DictReader(open(base + "/epochs.csv")))
mo = list(csv.DictReader(open(base + "/modules.csv")))
st = list(csv.DictReader(open(base + "/steps.csv")))
assert len(ep) == n, f"epochs.csv has {len(ep)} rows, expected {n}"
assert len({(r["layer"], r["mod"]) for r in mo}) == 48, "modules.csv should cover 48 q/v matrices"
assert len(mo) == (len(st) + n) * 48, f"modules.csv rows {len(mo)} != (steps {len(st)} + epochs {n}) * 48"
assert all(r["asr"] not in ("", "nan") for r in ep), "epochs.csv has no real ASR"
print(f"  files ok: {len(st)} step snapshots + {n} epoch-end snapshots x 48 modules")
EOF
}

check_pen() {     # check_pen <on.log> <obs_dir>   (Tr arm: logger vs the training log's own penalty)
  python - "$1" "$2" <<'EOF'
import csv, glob, math, re, sys
log, d = sys.argv[1], sys.argv[2]
txt = open(log, errors="replace").read().replace("\r", "\n")
dbg = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"\[DEBUG_ep0_stp(\d+)\] .*?penalty\(mean,scaled\)=([-+0-9.eE]+)", txt)}
st = {int(r["step"]): r for r in csv.DictReader(open(glob.glob(d + "/*/steps.csv")[0])) if r["epoch"] == "0"}
bad, seen = 0, 0
for step, v in sorted(dbg.items()):
    if step not in st: continue
    seen += 1
    a, b = float(st[step]["penalty"]), float(st[step]["pen8"])
    ok = math.isclose(a, v, abs_tol=1e-3) and math.isclose(b, v, rel_tol=2e-3, abs_tol=2e-3)
    print(f"  step {step:3d}: training log {v:.4f} | logger penalty {a:.4f} | logger pen8 (float64 recompute) {b:.4f}  {'ok' if ok else 'MISMATCH'}")
    bad += (not ok)
sys.exit(1 if bad or seen == 0 else 0)
EOF
}

gate() {    # gate <name> <script args...>
  local name="$1"; shift
  local on="$OUT/$name.on.log" off="$OUT/$name.off.log" ok=1
  echo "== $name"
  python variant_finetune_obs.py "$@" --obs_log "$OUT/obs_$name" > "$on" 2>&1;  [ $? -eq 0 ] || { fail "$name: run WITH logging crashed (see $on)"; tail -8 "$on"; return; }
  python variant_finetune_obs.py "$@"                           > "$off" 2>&1; [ $? -eq 0 ] || { fail "$name: run WITHOUT logging crashed (see $off)"; tail -8 "$off"; return; }
  local n_on n_off; n_on=$(canon "$on" | grep -c '^\[EPOCH\]'); n_off=$(canon "$off" | grep -c '^\[EPOCH\]')
  [ "$n_on" -eq "$EPOCHS" ] && [ "$n_off" -eq "$EPOCHS" ] || { fail "$name: expected $EPOCHS [EPOCH] lines, got $n_on / $n_off"; ok=0; }
  canon "$on" | grep '^\[EPOCH\]' | grep -qv 'nan' || { fail "$name: no real test/ASR values, the comparison would be vacuous"; ok=0; }
  if ! diff <(canon "$on") <(canon "$off") > "$OUT/$name.diff"; then
    fail "$name: lines differ between logging ON and OFF (first lines of $OUT/$name.diff):"; head -10 "$OUT/$name.diff"; ok=0
  fi
  check_files "$OUT/obs_$name" "$EPOCHS" || { fail "$name: log files have the wrong shape"; ok=0; }
  if [ "$name" = tr ]; then check_pen "$on" "$OUT/obs_$name" || { fail "$name: logger penalty does not match the training log"; ok=0; }; fi
  [ $ok -eq 1 ] && echo "  PASS ($(canon "$on" | wc -l) trainer-side + observer-side lines identical with logging ON and OFF)"
}

echo "== CPU test"; python test_obs_log.py 2>&1 | tail -1 | grep -q "ALL OK" && echo "  PASS" || fail "CPU test_obs_log.py"
gate base --seed 3 --num_epochs "$EPOCHS" --no_save --pz_path "$PZ" --lr 2e-4 --svd_k 8
gate tr   --seed 3 --num_epochs "$EPOCHS" --no_save --pz_path "$PZ" --lr 2e-4 --svd_k 8 --use_orthogonal_penalty --tr_lambda 10

echo
[ "$FAILS" -eq 0 ] && echo "GATE PASSED: logging does not change the run. Next: bash obs_tonight.sh" || { echo "GATE FAILED ($FAILS problem(s)); do not use the logs."; exit 1; }
