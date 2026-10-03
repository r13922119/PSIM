#!/usr/bin/env python3
"""Apply PREREG.md's persistence rule mechanically to a parse_runs.py --csv file.

  python parse_runs.py logs/baseline_<date>.log --csv > base.csv
  python persist.py base.csv

Rule (PREREG.md, committed before any v2 baseline result):
  statistic  : asr_mean_last5 (baseline, pre-Pt)
  per PZ     : mean over finetune seeds 0,1,2
  persistent : mean >= 0.50
  ambiguous  : mean in [0.25, 0.50)  OR  (max - min) > 0.30   -> add finetune seeds 6,7,8, decide on the mean of all six, no further seeds
  continue   : >= 2 FRESH persistent PZs. Fewer -> train PZ seeds 5,6,... one at a time until 2 persist or 10 fresh PZs trained.
  oldpz      : shown, never counted in the rate.
Thresholds below must match PREREG.md; change them there first, never here.
"""
import csv, re, sys
THRESH, AMB_LO, SPREAD, NEED, NMAX = 0.50, 0.25, 0.30, 2, 10
QUAL, EXTRA = (0, 1, 2), (6, 7, 8)

def load(path):
    runs, dup = {}, []
    for r in csv.DictReader(open(path)):
        name = r.get("name", "")
        m_pz, m_seed = re.search(r"_pz-(.+)$", name), re.search(r"_seed(\d+)_", name)
        if not (m_pz and m_seed and r.get("asr_mean_last5") not in (None, "")):
            continue
        try: v = float(r["asr_mean_last5"])
        except ValueError: continue
        key = (m_pz.group(1), int(m_seed.group(1)))
        if key in runs: dup.append(key)
        runs[key] = v
    return runs, dup

def kind(tag):
    if re.fullmatch(r"badnet_s\d+", tag): return "fresh"
    if tag == "poisoned_roberta_large_badnet": return "oldpz"
    return "other"

def judge(vals):
    """vals: {seed: value}. Returns (verdict, mean, spread, seeds_used)."""
    q = [vals[s] for s in QUAL if s in vals]
    if len(q) < len(QUAL):
        return "INCOMPLETE (need seeds 0,1,2)", None, None, sorted(vals)
    mean, spread = sum(q) / len(q), max(q) - min(q)
    ambiguous = AMB_LO <= mean < THRESH or spread > SPREAD
    if not ambiguous:
        return ("PERSISTENT" if mean >= THRESH else "washed out"), mean, spread, list(QUAL)
    ex = [vals[s] for s in EXTRA if s in vals]
    if len(ex) < len(EXTRA):
        return "AMBIGUOUS -> run finetune seeds 6,7,8, then decide on all six", mean, spread, list(QUAL)
    allv = q + ex
    m6 = sum(allv) / len(allv)
    return ("PERSISTENT (6 seeds)" if m6 >= THRESH else "washed out (6 seeds)"), m6, max(allv) - min(allv), list(QUAL) + list(EXTRA)

def main():
    runs, dup = load(sys.argv[1])
    if dup: print("WARNING: duplicate (PZ, seed) rows, last one used:", dup)
    tags = sorted({t for t, _ in runs}, key=lambda t: (kind(t) != "fresh", t))
    print(f"{'PZ':32s} {'kind':6s} {'seeds':10s} {'mean':>6s} {'spread':>7s}  verdict")
    fresh, old_verdict = [], None
    for t in tags:
        vals = {s: v for (tt, s), v in runs.items() if tt == t}
        verdict, mean, spread, used = judge(vals)
        ms = "-" if mean is None else f"{mean:.3f}"
        sp = "-" if spread is None else f"{spread:.3f}"
        per = " ".join(f"{vals[s]:.2f}" for s in sorted(vals))
        print(f"{t:32s} {kind(t):6s} {','.join(map(str, sorted(vals))):10s} {ms:>6s} {sp:>7s}  {verdict}   [{per}]")
        if kind(t) == "fresh": fresh.append((t, verdict))
        if kind(t) == "oldpz": old_verdict = verdict
    n, k = len(fresh), sum(1 for _, v in fresh if v.startswith("PERSISTENT"))
    pending = [t for t, v in fresh if v.startswith(("INCOMPLETE", "AMBIGUOUS"))]
    print(f"\nfresh persistence rate: {k}/{n} (oldpz not counted)")
    if pending:
        print("NOT FINAL, pending:", ", ".join(pending)); return
    if k >= NEED: print(f"DECISION: >= {NEED} fresh PZs persist -> continue to Cl/Tr experiments (P' = those {k}).")
    elif n < NMAX: print(f"DECISION: only {k} persist -> train PZ seed {n} next (fallback, cap N={NMAX}), qualify with finetune seeds 0-2.")
    else: print(f"DECISION: only {k} persist after {n} fresh PZs -> STOP and rethink (poison ratio, ask the author).")
    if k < NEED and old_verdict and old_verdict.startswith("PERSISTENT"):
        print("NOTE: oldpz persists but fewer than 2 fresh PZs do. oldpz alone does not satisfy the continue rule: if the fallback ends with < 2 fresh, stop and discuss, do not go on to Cl/Tr automatically.")

main()