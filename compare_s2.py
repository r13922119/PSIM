#!/usr/bin/env python3
"""S2: baseline vs Cl-only on the same PZ and the same fine-tune seeds (3,4,5). Prints numbers, makes no verdict.

  python parse_runs.py logs/s2_2*.log --csv > s2.csv
  python compare_s2.py s2.csv

Three ASR statistics per run (do NOT mix them up):
  last5   = asr_mean_last5: mean ASR over epochs 15-19, pre-Pt.  This is the PREREG statistic.
  pre-Pt  = pre-pt_ASR: ASR of the best-dev adapter (ONE epoch), before spectral rescaling.
  with-Pt = pt_ASR:     ASR of that same best-dev adapter AFTER spectral rescaling.
The pure effect of Pt is with-Pt minus pre-Pt (same adapter, same epoch). Comparing with-Pt against last5
mixes Pt with epoch selection, so this script never does that.
CA = clean test accuracy (pre-Pt / with-Pt). dev = best_dev_acc. best_ep = epoch picked by dev.
oldpz is listed last and is never pooled with the fresh models.
"""
import csv, math, re, sys, statistics as st
SEEDS = (3, 4, 5)
NAN = float("nan")

def f(x):
    try: return float(x)
    except (TypeError, ValueError): return NAN

def mean(L):
    L = [x for x in L if not math.isnan(x)]
    return st.mean(L) if L else NAN

def fmt(x, w=6, d=3, sign=False):
    if x is None or math.isnan(x): return f"{'-':>{w}s}"
    return f"{x:+{w}.{d}f}" if sign else f"{x:{w}.{d}f}"

rows = {}
for r in csv.DictReader(open(sys.argv[1])):
    n = r.get("name", "")
    mp, ms = re.search(r"_pz-(.+)$", n), re.search(r"_seed(\d+)_", n)
    if not (mp and ms) or int(ms.group(1)) not in SEEDS: continue
    if "_tr" in n: continue                                   # S2 has no Tr runs
    var = "cl" if "_cl" in n else "base"
    d = dict(ep=f(r.get("best_epoch")), last5=f(r.get("asr_mean_last5")), pre=f(r.get("pre-pt_ASR")), pt=f(r.get("pt_ASR")),
             ca=f(r.get("pre-pt_test_acc")), cap=f(r.get("pt_test_acc")), dev=f(r.get("best_dev_acc")))
    d["pteff"] = d["pt"] - d["pre"]
    rows[(mp.group(1), var, int(ms.group(1)))] = d

def overlap(a, b):
    a = [x for x in a if not math.isnan(x)]; b = [x for x in b if not math.isnan(x)]
    if not a or not b: return "n/a"
    return "overlap" if max(min(a), min(b)) <= min(max(a), max(b)) else "do NOT overlap"

pzs = sorted({k[0] for k in rows}, key=lambda t: (t == "poisoned_roberta_large_badnet", t))
HDR = f"{'':7s}{'seed':>4s}{'best_ep':>8s} | {'last5':>6s} | {'pre-Pt':>6s} {'with-Pt':>7s} {'Pt eff':>7s} | {'CA pre':>6s} {'CA Pt':>6s} {'dev':>6s}"
for pz in pzs:
    print(f"=== {pz}")
    print(HDR)
    got = {}
    complete = True
    for var in ("base", "cl"):
        rs = [rows.get((pz, var, s)) for s in SEEDS]
        got[var] = rs
        if any(x is None for x in rs): complete = False
        for s, x in zip(SEEDS, rs):
            if x is None:
                print(f"{var:7s}{s:>4d}   (missing)"); continue
            print(f"{var:7s}{s:>4d}{fmt(x['ep'], 8, 0)} | {fmt(x['last5'])} | {fmt(x['pre'])} {fmt(x['pt'], 7)} {fmt(x['pteff'], 7, 3, True)} | {fmt(x['ca'])} {fmt(x['cap'])} {fmt(x['dev'])}")
        g = lambda k: mean([x[k] for x in rs if x])
        print(f"{var + ' mean':>11s}{'':8s} | {fmt(g('last5'))} | {fmt(g('pre'))} {fmt(g('pt'), 7)} {fmt(g('pteff'), 7, 3, True)} | {fmt(g('ca'))} {fmt(g('cap'))} {fmt(g('dev'))}")
    if complete:
        print("  paired Cl - base per seed, then mean, then whether the ranges overlap:")
        for label, k in (("last5 (PREREG stat)", "last5"), ("pre-Pt best-dev   ", "pre"), ("with-Pt best-dev  ", "pt"), ("Pt effect (pt-pre)", "pteff")):
            dd = [got["cl"][i][k] - got["base"][i][k] for i in range(len(SEEDS))]
            rng = overlap([x[k] for x in got["base"]], [x[k] for x in got["cl"]])
            print(f"    {label}: " + " ".join(fmt(x, 6, 3, True) for x in dd) + f"   mean {fmt(mean(dd), 6, 3, True)}   ranges {rng}")
    else:
        print("  (INCOMPLETE: some of the seeds are missing, paired differences skipped)")
    print()