#!/usr/bin/env python3
"""S4: does Tr add anything to Cl, and does Tr alone do anything? Fine-tune seeds 3-5, k = k_final from S3 (s3_final.json).

  python parse_runs.py logs/s2_2*.log --csv > s2.csv      # baseline and Cl-only (S2)
  python parse_runs.py logs/s4_2*.log --csv > s4.csv      # Tr-only and Cl+Tr (S4), plus rerun references if any
  python compare_s4.py s2.csv s4.csv [--k 8]              # later csv files override earlier ones for the same run

Performance = the best-dev epoch's test CA and ASR, before and after Pt, as a distribution over seeds.
  pre-Pt  = ASR of the best-dev adapter before Pt.   with-Pt = ASR of the same adapter after Pt.
DIAGNOSTIC only (never performance): last5 = asr_mean_last5.
Two comparisons per PZ and lambda:
  Cl+Tr minus Cl-only   : what Tr adds on top of Cl (the question S3/S4 are about)
  Tr-only minus baseline: the paper's Tr-alone ablation
Label (a report, not a gate): mean difference in with-Pt ASR, band +-0.10: reduces / indistinguishable / increases.
oldpz is a labeled member and is never pooled; badnet_s2 is tuned-on (k was chosen on it).
"""
import csv, json, math, re, sys, statistics as st
SEEDS = (3, 4, 5)
BAND = 0.10
NAN = float("nan")

def f(x):
    try: return float(x)
    except (TypeError, ValueError): return NAN

def mean(L):
    L = [x for x in L if not math.isnan(x)]
    return st.mean(L) if L else NAN

def fmt(x, w=6, d=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)): return f"{'-':>{w}s}"
    return f"{x:+{w}.{d}f}" if sign else f"{x:{w}.{d}f}"

def label(d):
    return "reduces" if d <= -BAND else "increases" if d >= BAND else "indistinguishable"

def overlap(a, b):
    a = [x for x in a if not math.isnan(x)]; b = [x for x in b if not math.isnan(x)]
    if not a or not b: return "n/a"
    return "overlap" if max(min(a), min(b)) <= min(max(a), max(b)) else "do NOT overlap"

args = [a for a in sys.argv[1:]]
K = None
if "--k" in args:
    i = args.index("--k"); K = int(args[i + 1]); del args[i:i + 2]
if K is None:
    K = json.load(open("s3_final.json"))["k"]
rows = {}
for path in args:
    for r in csv.DictReader(open(path)):
        n = r.get("name", "")
        mp, ms = re.search(r"_pz-(.+)$", n), re.search(r"_seed(\d+)_", n)
        if not (mp and ms) or int(ms.group(1)) not in SEEDS: continue
        mt = re.search(r"_tr([0-9.]+)_k(\d+)", n)
        cl = "_cl" in n
        if mt and int(mt.group(2)) != K: continue
        arm = ("cltr" if cl else "tronly") if mt else ("cl" if cl else "base")
        lam = float(mt.group(1)) if mt else None
        d = dict(last5=f(r.get("asr_mean_last5")), pre=f(r.get("pre-pt_ASR")), pt=f(r.get("pt_ASR")),
                 ca=f(r.get("pre-pt_test_acc")), cap=f(r.get("pt_test_acc")), dev=f(r.get("best_dev_acc")), ep=f(r.get("best_epoch")))
        rows[(mp.group(1), arm, lam, int(ms.group(1)))] = d

pzs = sorted({k[0] for k in rows}, key=lambda t: (t == "poisoned_roberta_large_badnet", t))
lams = sorted({k[2] for k in rows if k[2] is not None})
def cells(pz, arm, lam): return [rows.get((pz, arm, lam, s)) for s in SEEDS]
def tag(pz): return {"poisoned_roberta_large_badnet": "oldpz (labeled, not pooled)", "badnet_s2": "badnet_s2 (tuned-on)"}.get(pz, pz)

summary = []
HDR = f"{'arm':>14s}{'seed':>5s}{'best_ep':>8s} | {'pre-Pt':>6s} {'with-Pt':>7s} | {'CA Pt':>6s} {'dev':>6s} | {'last5(diag)':>11s}"
print(f"S4 at k={K}, seeds {SEEDS}. Label band +-{BAND} on the mean with-Pt difference. last5 is a diagnostic.\n")
for pz in pzs:
    print(f"=== {tag(pz)}")
    print(HDR)
    def show(name, rs):
        for s, x in zip(SEEDS, rs):
            if x is None: print(f"{name:>14s}{s:>5d}   (missing)"); continue
            print(f"{name:>14s}{s:>5d}{fmt(x['ep'], 8, 0)} | {fmt(x['pre'])} {fmt(x['pt'], 7)} | {fmt(x['cap'])} {fmt(x['dev'])} | {fmt(x['last5'], 11)}")
        ok = [x for x in rs if x]
        print(f"{name + ' mean':>19s}{'':3s} | {fmt(mean([x['pre'] for x in ok]))} {fmt(mean([x['pt'] for x in ok]), 7)} | {fmt(mean([x['cap'] for x in ok]))} {fmt(mean([x['dev'] for x in ok]))} | {fmt(mean([x['last5'] for x in ok]), 11)}")
    base, clo = cells(pz, "base", None), cells(pz, "cl", None)
    show("baseline", base); show("Cl-only", clo)
    for lam in lams:
        show(f"Cl+Tr l={lam:g}", cells(pz, "cltr", lam)); show(f"Tr-only l={lam:g}", cells(pz, "tronly", lam))
    def paired(title, a, b, key_pair):
        if any(x is None for x in a) or any(x is None for x in b):
            print(f"  {title}: INCOMPLETE (some seeds missing)"); return None
        out = {}
        for lab, k in (("with-Pt", "pt"), ("pre-Pt ", "pre"), ("last5(diag)", "last5"), ("CA after Pt", "cap")):
            dd = [a[i][k] - b[i][k] for i in range(len(SEEDS))]
            out[k] = (dd, mean(dd), overlap([x[k] for x in b], [x[k] for x in a]))
            print(f"    {lab:>11s}: " + " ".join(fmt(x, 6, 3, True) for x in dd) + f"   mean {fmt(out[k][1], 6, 3, True)}   ranges {out[k][2]}")
        lb = label(out["pt"][1])
        print(f"    -> label (with-Pt mean difference {out['pt'][1]:+.3f}): {lb}")
        return lb, out["pt"][1]
    for lam in lams:
        print(f"  Cl+Tr(l={lam:g}) minus Cl-only, per seed, then mean:")
        r1 = paired("cltr-cl", cells(pz, "cltr", lam), clo, None)
        print(f"  Tr-only(l={lam:g}) minus baseline, per seed, then mean:")
        r2 = paired("tronly-base", cells(pz, "tronly", lam), base, None)
        summary.append((pz, lam, r1, r2))
    print()

print("=== SUMMARY of labels (a report, not a verdict; one k, three seeds per cell; k was chosen by noise in S3)")
print(f"{'PZ':>30s} {'lambda':>7s} | {'Cl+Tr vs Cl-only':>34s} | {'Tr-only vs baseline':>34s}")
for pz, lam, r1, r2 in summary:
    a = f"{r1[0]} ({r1[1]:+.3f})" if r1 else "incomplete"
    b = f"{r2[0]} ({r2[1]:+.3f})" if r2 else "incomplete"
    print(f"{tag(pz):>30s} {lam:7g} | {a:>34s} | {b:>34s}")