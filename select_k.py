#!/usr/bin/env python3
"""Choose k for Tr (Cl+Tr) by a rule fixed BEFORE any S3 result. Same spirit as persist.py: it only applies PREREG_S3.md. 

  python parse_runs.py <s3 logs> --csv > s3.csv
  python select_k.py s3.csv --stage 1      # single seed, k grid x lambda  -> writes s3_stage2.txt (k list per lambda); S3_STOP only if nothing is feasible
  python select_k.py s3.csv --stage 2      # 3 seeds on the stage-2 k's    -> writes s3_final.json (k_final) and prints the label; S3_STOP only if nothing is feasible
  python select_k.py --plan1 | --seeds2 | --pz     # print the plan (used by v2_s3.sh so the two can never disagree)

Rule (PREREG_S3.md; values below must match it, change them there first, never here):
  statistic  : pt_ASR = ASR of the best-dev adapter AFTER Pt, the number we present as performance.
               pre-Pt best-dev ASR and asr_mean_last5 are printed as diagnostics and NEVER used to select.
  reference  : Cl-only on the same PZ and same seed(s).
  feasible   : best_dev_acc >= reference dev - DEV_TOL   AND   pt_test_acc >= reference pt_test_acc - CA_TOL
               (the CA guard stops a k from winning by wrecking the model, which Pt can do; dev is measured before Pt).
  stage 1    : seed 20. Selection set = Cl+Tr over LAMBDAS x GRID. For each lambda, k* = feasible k with the lowest pt_ASR.
               stage-2 k's = k* and its two neighbours in the grid (all at that lambda).
  stage 2    : seeds 21,22,23. Per (lambda, k): mean pt_ASR. k_final = lowest mean pt_ASR among feasible.
  label      : NOT a gate. The best cell's pt_ASR is compared with Cl-only and labelled (two-sided, band MARGIN):
                 'reduces'           best <= Cl-only - MARGIN
                 'indistinguishable' within +-MARGIN of Cl-only (from noise, or from a paper-sized effect)
                 'increases'         best >= Cl-only + MARGIN (even the best k does not help)
               'reduces' is optimistic (it is a minimum over cells); 'increases' is the stronger statement. Raw difference, per-seed
               values and CA are always printed. S3 stops ONLY when no cell is feasible (or a Cl-only reference is missing).
  k = 1024   : full rank, i.e. Tr reduces to plain L2. It is a normal candidate. If it wins, that is reported, not hidden.
  NOT used for selection (descriptive, seed 20 only): the paper-equivalence cells, i.e. CONTROLS (Cl+Tr) and TR_ONLY (Tr alone, no Cl).
"""
import csv, json, math, re, sys, argparse, statistics as st

PZ_TAG   = "badnet_s2"
GRID     = (8, 32, 128, 256, 512, 768, 896, 1016, 1024)   # free room d-k = 1016, 992, 896, 768, 512, 256, 128, 8, 0 (rank r=8 needs 8)
LAMBDAS  = (10.0, 480.0)   # the paper's lambda=10 read as a mean over modules (10) or as a sum over the 48 modules (480)
S1_SEED  = 20
S2_SEEDS = (21, 22, 23)
DEV_TOL  = 0.01     # dev accuracy may be at most 1 point below Cl-only
CA_TOL   = 0.02     # clean test accuracy after Pt may be at most 2 points below Cl-only
MARGIN   = 0.10     # half-width of the 'indistinguishable' band around Cl-only (label only, never a gate)
# Paper-equivalence cells. Our code multiplies Omega by d/k and averages over 48 modules. The paper's "10" read four ways at k=32:
#   mean+scaled 10 | mean+unscaled 10*k/d = 0.3125 | sum+scaled 480 | sum+unscaled 480*k/d = 15.   At k=1024 the scaling is 1, so {10, 480}.
CONTROLS = ((0.3125, 32), (15.0, 32))                                                      # Cl+Tr at S1_SEED (10 and 480 at k=32 are in the grid)
TR_ONLY  = ((10.0, 1024), (480.0, 1024), (0.3125, 32), (10.0, 32), (15.0, 32), (480.0, 32))   # Tr alone (no Cl) at S1_SEED
NAN = float("nan")

def num(x):
    try: return float(x)
    except (TypeError, ValueError): return NAN

def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        n = r.get("name", "")
        m_pz, m_seed = re.search(r"_pz-(.+)$", n), re.search(r"_seed(\d+)_", n)
        if not (m_pz and m_seed) or m_pz.group(1) != PZ_TAG: continue
        asr, ca, dev = num(r.get("pt_ASR")), num(r.get("pt_test_acc")), num(r.get("best_dev_acc"))
        if any(math.isnan(v) for v in (asr, ca, dev)): continue
        m_tr = re.search(r"_tr([0-9.]+)_k(\d+)", n)
        rows.append(dict(seed=int(m_seed.group(1)), cl=("_cl" in n), tr=bool(m_tr),
                         lam=float(m_tr.group(1)) if m_tr else None, k=int(m_tr.group(2)) if m_tr else None,
                         asr=asr, ca=ca, dev=dev, pre=num(r.get("pre-pt_ASR")), last5=num(r.get("asr_mean_last5")), name=n))
    return rows

def agg(v):
    def m(key):
        xs = [x[key] for x in v if not math.isnan(x[key])]
        return st.mean(xs) if xs else NAN
    return dict(asr=m("asr"), ca=m("ca"), dev=m("dev"), pre=m("pre"), last5=m("last5"))

def get(rows, seeds, cl, tr, lam=None, k=None):
    """Mean over seeds of the runs with this exact variant; None if any seed is missing."""
    v = [x for x in rows if x["cl"] == cl and x["tr"] == tr and x["seed"] in seeds and (lam is None or x["lam"] == lam) and (k is None or x["k"] == k)]
    if len({x["seed"] for x in v}) < len(seeds): return None
    return agg(v)

def feasible(c, ref):
    return c["dev"] >= ref["dev"] - DEV_TOL and c["ca"] >= ref["ca"] - CA_TOL

def why_not(c, ref):
    w = []
    if c["dev"] < ref["dev"] - DEV_TOL: w.append("dev")
    if c["ca"] < ref["ca"] - CA_TOL: w.append("CA")
    return "yes" if not w else "no (" + "+".join(w) + ")"

def label(best, ref):
    d = best - ref
    if d <= -MARGIN: return "reduces", d
    if d >= MARGIN:  return "increases", d
    return "indistinguishable", d

def per_seed(rows, seeds, cl, tr, lam=None, k=None):
    out = []
    for sd in seeds:
        v = [x["asr"] for x in rows if x["cl"] == cl and x["tr"] == tr and x["seed"] == sd and (lam is None or x["lam"] == lam) and (k is None or x["k"] == k)]
        out.append(v[0] if v else NAN)
    return out

def report_label(best, ref, seeds, rows, lam, k, stage):
    lab, d = label(best, ref["asr"])
    print(f"\nEFFECT LABEL (stage {stage}): best Cl+Tr cell lambda={lam:g} k={k}: with-Pt ASR {best:.3f} vs Cl-only {ref['asr']:.3f}  (difference {d:+.3f}, band +-{MARGIN})  ->  {lab}")
    a, b = per_seed(rows, seeds, True, True, lam, k), per_seed(rows, seeds, True, False)
    print("  per-seed with-Pt ASR  best cell: " + " ".join(f"{x:.3f}" for x in a) + "   Cl-only: " + " ".join(f"{x:.3f}" for x in b))
    if len(seeds) > 1:
        ov = max(min(a), min(b)) <= min(max(a), max(b))
        print("  seed ranges " + ("overlap" if ov else "do NOT overlap"))
    else:
        print("  single seed: a label, not evidence.")
    print("  'reduces' is a minimum over cells (optimistic); 'increases' is the stronger statement. Not a gate: S3 continues.")
    return lab, d

def stop(msg):
    print("\nS3_STOP:", msg); open("S3_STOP", "w").write(msg + "\n"); sys.exit(2)

def table(rows, seeds, ref, only):
    print(f"Cl-only reference (seeds {','.join(map(str, seeds))}): with-Pt ASR={ref['asr']:.3f}  CA(Pt)={ref['ca']:.4f}  dev={ref['dev']:.4f}")
    print(f"feasible: dev >= {ref['dev']-DEV_TOL:.4f} and CA(Pt) >= {ref['ca']-CA_TOL:.4f}.   Selection uses the 'with-Pt ASR' column only; pre-Pt and last5 are diagnostics.")
    print(f"{'lambda':>7s} {'k':>5s} {'withPt':>7s} {'prePt':>7s} {'last5':>7s} {'CA(Pt)':>7s} {'dev':>7s}  feasible")
    out = {}
    for lam in LAMBDAS:
        for k in GRID:
            c = get(rows, seeds, True, True, lam, k); out[(lam, k)] = c
            if (lam, k) not in only: continue
            if c is None: print(f"{lam:7g} {k:5d}   (missing)"); continue
            print(f"{lam:7g} {k:5d} {c['asr']:7.3f} {c['pre']:7.3f} {c['last5']:7.3f} {c['ca']:7.4f} {c['dev']:7.4f}  {why_not(c, ref)}")
    return out

def desc_table(rows, seeds):
    """Paper-equivalence cells: printed for reading, never used to select anything."""
    base, clref = get(rows, seeds, False, False), get(rows, seeds, True, False)
    print("\nPaper-equivalence cells (seed %s, DESCRIPTIVE, not used for selection). delta = with-Pt ASR minus its reference." % ",".join(map(str, seeds)))
    if base: print(f"  baseline reference (no Cl, no Tr, with Pt): with-Pt ASR={base['asr']:.3f} CA(Pt)={base['ca']:.4f}")
    else:    print("  baseline reference: (missing)")
    print(f"  {'variant':>7s} {'lambda':>7s} {'k':>5s} {'withPt':>7s} {'prePt':>7s} {'CA(Pt)':>7s} {'dev':>7s}  delta vs reference")
    miss = []
    for label, cells, cl, ref in (("Tr-only", TR_ONLY, False, base), ("Cl+Tr", CONTROLS, True, clref)):
        for lam, k in cells:
            c = get(rows, seeds, cl, True, lam, k)
            if c is None: miss.append((label, lam, k)); print(f"  {label:>7s} {lam:7g} {k:5d}   (missing)"); continue
            d = f"{c['asr'] - ref['asr']:+.3f}" if ref else "n/a"
            print(f"  {label:>7s} {lam:7g} {k:5d} {c['asr']:7.3f} {c['pre']:7.3f} {c['ca']:7.4f} {c['dev']:7.4f}  {d}")
    if miss or not base: print(f"  NOTE: descriptive cells missing: {miss}{' + baseline reference' if not base else ''} (rerun v2_s3.sh to fill; this does not block selection)")

def STAGE2_KS():
    try: lines = [l.split() for l in open("s3_stage2.txt")]
    except OSError: return set()
    return {(float(l[0]), int(k)) for l in lines for k in l[1:]}

def plan1():
    out = [(S1_SEED, "clonly", "-", "-"), (S1_SEED, "base", "-", "-")]
    out += [(S1_SEED, "cltr", f"{lam:g}", k) for lam in LAMBDAS for k in GRID]
    out += [(S1_SEED, "cltr", f"{lam:g}", k) for lam, k in CONTROLS]
    out += [(S1_SEED, "tronly", f"{lam:g}", k) for lam, k in TR_ONLY]
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="?"); ap.add_argument("--stage", type=int)
    ap.add_argument("--plan1", action="store_true"); ap.add_argument("--seeds2", action="store_true"); ap.add_argument("--pz", action="store_true")
    a = ap.parse_args()
    if a.plan1:
        for seed, var, lam, k in plan1(): print(seed, var, lam, k)
        return
    if a.seeds2: print(" ".join(map(str, S2_SEEDS))); return
    if a.pz: print(PZ_TAG); return
    if not a.csv or a.stage not in (1, 2): ap.error("need a csv and --stage 1|2")
    rows = load(a.csv)
    seeds = (S1_SEED,) if a.stage == 1 else S2_SEEDS
    ref = get(rows, seeds, True, False)
    if ref is None: stop(f"no Cl-only reference rows for seeds {seeds}; run them first")
    planned = set((l, k) for l in LAMBDAS for k in GRID) if a.stage == 1 else STAGE2_KS()
    cells = table(rows, seeds, ref, only=planned)
    if a.stage == 1: desc_table(rows, seeds)
    missing = sorted(kk for kk in planned if cells.get(kk) is None)
    if missing: print(f"\nINCOMPLETE stage {a.stage}, missing: {missing}"); sys.exit(1)
    feas = {kk: v for kk, v in cells.items() if v and kk in planned and feasible(v, ref)}
    if not feas: stop("no (lambda, k) satisfies the dev and CA constraints")
    if a.stage == 1:
        plan = {}
        for lam in LAMBDAS:
            cand = {k: v for (l, k), v in feas.items() if l == lam}
            if not cand: continue
            ks = min(cand, key=lambda k: cand[k]["asr"]); i = GRID.index(ks)
            plan[lam] = [GRID[j] for j in (i-1, i, i+1) if 0 <= j < len(GRID)]
            print(f"\nlambda={lam:g}: k*={ks} (with-Pt ASR {cand[ks]['asr']:.3f}) -> stage-2 k's {plan[lam]}")
        kb = min(feas, key=lambda x: feas[x]["asr"])
        report_label(feas[kb]["asr"], ref, seeds, rows, kb[0], kb[1], 1)
        with open("s3_stage2.txt", "w") as f:
            for lam, ks in plan.items(): f.write(f"{lam:g} {' '.join(map(str, ks))}\n")
        print("\nwrote s3_stage2.txt  (lambda then k's, one line per lambda)")
    else:
        kk = min(feas, key=lambda x: feas[x]["asr"]); lam, k = kk
        print(f"\nk_final: lambda={lam:g} k={k}  mean with-Pt ASR={feas[kk]['asr']:.3f} CA(Pt)={feas[kk]['ca']:.4f} dev={feas[kk]['dev']:.4f}  (Cl-only {ref['asr']:.3f})")
        lab, d = report_label(feas[kk]["asr"], ref, seeds, rows, lam, k, 2)
        if k == GRID[-1]: print("NOTE: k = full rank, so the winner is plain L2. Report this as such.")
        json.dump({"lambda": lam, "k": k, "asr": feas[kk]["asr"], "ca": feas[kk]["ca"], "dev": feas[kk]["dev"],
                   "ref_asr": ref["asr"], "ref_ca": ref["ca"], "ref_dev": ref["dev"], "label": lab, "diff": d}, open("s3_final.json", "w"))
        print("wrote s3_final.json (S4 reports this k at lambda 10 and 480, Cl+Tr and Tr-only)")

main()