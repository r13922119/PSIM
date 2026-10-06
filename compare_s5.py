#!/usr/bin/env python3
"""S5: is Tr's effect on with-Pt ASR "just a smaller update"?  Fine-tune seeds 3-5, no Tr in the S5 runs (lower lr instead).

  python parse_runs.py logs/s2_2*.log --csv > s2.csv     # baseline and Cl-only at lr 2e-4 (S2)
  python parse_runs.py logs/s4_2*.log --csv > s4.csv     # Tr-only and Cl+Tr at lr 2e-4 (S4)
  python parse_runs.py logs/s5_2*.log --csv > s5.csv     # baseline and Cl-only at lr 1e-4 and 5e-5 (S5)
  python dw_tr.py adapters/*seed[345]_*pz-badnet_s0 adapters/*seed[345]_*pz-badnet_s2 adapters/*seed[345]_*pz-poisoned_roberta_large_badnet --csv s2.csv s4.csv s5.csv > dw_s5.txt
  python compare_s5.py s2.csv s4.csv s5.csv [--dw dw_s5.txt] [--k 8]     # later csv files override earlier ones for the same run

Two families, always same PZ and same seeds:
  base family: reference baseline lr 2e-4; lower-lr arms baseline lr 1e-4, 5e-5;  Tr arms Tr-only lambda 10 / 480.
  cl   family: reference Cl-only  lr 2e-4; lower-lr arms Cl-only  lr 1e-4, 5e-5;  Tr arms Cl+Tr  lambda 10 / 480.
D_lr = with-Pt ASR(lower-lr arm) - with-Pt ASR(reference), per seed then mean.   D_Tr = with-Pt ASR(Tr arm) - with-Pt ASR(reference).
Size matching (PREREG_S5 B4): a lower-lr arm is "size-matched" to a Tr arm on a PZ when the mean rel of its runs is within 25% (relative) of the Tr arm's mean rel.
rel comes from dw_tr.py output (--dw); without it nothing is size-matched and no conclusion is printed.
Reading (fixed in PREREG_S5 B4, only when D_Tr >= +0.10, otherwise there is no Tr effect to explain):
  size explains         : D_lr >= +0.10 and D_lr >= D_Tr - 0.10
  size does not explain : D_lr <  +0.10, or D_lr <= D_Tr - 0.10      (checked after the line above)
  undecided             : not reached (the two lines cover everything when D_Tr >= +0.10)
A report on 3 seeds, not a test. last5 is a diagnostic. CA after Pt more than 0.02 below the reference is flagged.
"""
import csv, json, math, re, sys, statistics as st
SEEDS = (3, 4, 5); BAND = 0.10; MATCH = 0.25; CAGUARD = 0.02; NAN = float("nan")
LAMS = (10.0, 480.0)

def f(x):
    try: return float(x)
    except (TypeError, ValueError): return NAN
def mean(L):
    L = [x for x in L if not math.isnan(x)]
    return st.mean(L) if L else NAN
def fmt(x, w=6, d=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)): return f"{'-':>{w}s}"
    return f"{x:+{w}.{d}f}" if sign else f"{x:{w}.{d}f}"
def overlap(a, b):
    a = [x for x in a if not math.isnan(x)]; b = [x for x in b if not math.isnan(x)]
    if not a or not b: return "n/a"
    return "overlap" if max(min(a), min(b)) <= min(max(a), max(b)) else "do NOT overlap"
def norm_pz(t): return "oldpz" if t.startswith("poisoned_roberta_large") else t
def tag(pz): return {"oldpz": "oldpz (labeled, not pooled)", "badnet_s2": "badnet_s2 (tuned-on)"}.get(pz, pz)

args = sys.argv[1:]; K = None; dwpath = None
for flag in ("--k", "--dw"):
    if flag in args:
        i = args.index(flag); v = args[i + 1]; del args[i:i + 2]
        if flag == "--k": K = int(v)
        else: dwpath = v
if K is None:
    try: K = json.load(open("s3_final.json"))["k"]
    except Exception: K = 8

rows = {}      # (pz, arm, lam, lr, seed) -> dict ; arm in base, cl, tronly, cltr ; lr as the string in the run name ('2e-04')
for path in args:
    for r in csv.DictReader(open(path)):
        n = r.get("name", "")
        mp, ms, ml = re.search(r"_pz-(.+)$", n), re.search(r"_seed(\d+)_", n), re.search(r"_lr([0-9.]+e[-+]?\d+)_", n)
        if not (mp and ms and ml) or int(ms.group(1)) not in SEEDS: continue
        mt = re.search(r"_tr([0-9.]+)_k(\d+)", n); cl = "_cl" in n
        if mt and int(mt.group(2)) != K: continue
        arm = ("cltr" if cl else "tronly") if mt else ("cl" if cl else "base")
        rows[(norm_pz(mp.group(1)), arm, float(mt.group(1)) if mt else None, ml.group(1), int(ms.group(1)))] = dict(
            last5=f(r.get("asr_mean_last5")), pre=f(r.get("pre-pt_ASR")), pt=f(r.get("pt_ASR")), cap=f(r.get("pt_test_acc")),
            dev=f(r.get("best_dev_acc")), ep=f(r.get("best_epoch")))

REL = {}       # (pz, arm-label from dw_tr, seed) -> rel
if dwpath:
    for line in open(dwpath):
        if line.startswith("=== top-3"): break
        m = re.match(r"^(\S+)\s+(\d+)\s+(.+?)\s+\|\s+(\d\.\d+)\s", line)
        if m: REL[(m.group(1), m.group(3).strip(), int(m.group(2)))] = float(m.group(4))
def dwlabel(arm, lam, lr):
    sfx = "" if lr == "2e-04" else f" lr={lr}"
    if arm == "base": return "baseline" + sfx
    if arm == "cl": return "Cl-only" + sfx
    return (f"Cl+Tr l={lam:g} k={K}" if arm == "cltr" else f"Tr-only l={lam:g} k={K}") + sfx
def cell(pz, arm, lam, lr): return [rows.get((pz, arm, lam, lr, s)) for s in SEEDS]
def relmean(pz, arm, lam, lr):
    return mean([REL.get((pz, dwlabel(arm, lam, lr), s), NAN) for s in SEEDS])

pzs = sorted({k[0] for k in rows}, key=lambda t: (t != "oldpz", t))
lrs_low = sorted({k[3] for k in rows if k[1] in ("base", "cl") and k[3] != "2e-04"}, reverse=True)   # e.g. ['1e-04', '5e-05']
print(f"S5 at k={K} for the Tr arms, seeds {SEEDS}. D = with-Pt ASR difference to the lr 2e-4 reference, per seed then mean. Band +-{BAND}; size match within {MATCH:.0%} of rel. last5 is a diagnostic.\n")
summary = []
for pz in pzs:
    print(f"=== {tag(pz)}")
    for fam, ref_arm, tr_arm, name in (("base", "base", "tronly", "baseline"), ("cl", "cl", "cltr", "Cl-only")):
        ref = cell(pz, ref_arm, None, "2e-04")
        print(f"--- family {name}: reference = {name} lr 2e-4")
        print(f"{'arm':>22s}{'seed':>5s}{'best_ep':>8s} | {'pre-Pt':>6s} {'with-Pt':>7s} | {'CA Pt':>6s} {'dev':>6s} | {'last5(diag)':>11s} | {'rel':>6s}")
        def show(label, rs, rel):
            for s, x in zip(SEEDS, rs):
                if x is None: print(f"{label:>22s}{s:>5d}   (missing)"); continue
                print(f"{label:>22s}{s:>5d}{fmt(x['ep'], 8, 0)} | {fmt(x['pre'])} {fmt(x['pt'], 7)} | {fmt(x['cap'])} {fmt(x['dev'])} | {fmt(x['last5'], 11)} |")
            ok = [x for x in rs if x]
            print(f"{label + ' mean':>27s} | {fmt(mean([x['pre'] for x in ok]))} {fmt(mean([x['pt'] for x in ok]), 7)} | {fmt(mean([x['cap'] for x in ok]))} {fmt(mean([x['dev'] for x in ok]))} | {fmt(mean([x['last5'] for x in ok]), 11)} | {fmt(rel)}")
        show(f"{name} lr 2e-4", ref, relmean(pz, ref_arm, None, "2e-04"))
        for lr in lrs_low: show(f"{name} lr {lr}", cell(pz, ref_arm, None, lr), relmean(pz, ref_arm, None, lr))
        for lam in LAMS: show(f"{'Tr-only' if fam == 'base' else 'Cl+Tr'} l={lam:g}", cell(pz, tr_arm, lam, "2e-04"), relmean(pz, tr_arm, lam, "2e-04"))
        def diffs(a, b, key):
            if any(x is None for x in a) or any(x is None for x in b): return None
            return [a[i][key] - b[i][key] for i in range(len(SEEDS))]
        for lam in LAMS:
            tr = cell(pz, tr_arm, lam, "2e-04"); dT = diffs(tr, ref, "pt")
            trrel = relmean(pz, tr_arm, lam, "2e-04")
            for lr in lrs_low:
                lo = cell(pz, ref_arm, None, lr); dL = diffs(lo, ref, "pt")
                print(f"  lr {lr} vs {'Tr-only' if fam == 'base' else 'Cl+Tr'} l={lam:g}:")
                if dL is None or dT is None: print("    INCOMPLETE (some seeds missing)"); summary.append((pz, name, lr, lam, "incomplete", NAN, NAN)); continue
                mL, mT = mean(dL), mean(dT)
                dpre = diffs(lo, ref, "pre"); dcap = diffs(lo, ref, "cap")
                print(f"    D_lr with-Pt per seed: " + " ".join(fmt(x, 6, 3, True) for x in dL) + f"   mean {mL:+.3f}   ranges {overlap([x['pt'] for x in ref], [x['pt'] for x in lo])}")
                print(f"    D_lr pre-Pt  per seed: " + " ".join(fmt(x, 6, 3, True) for x in dpre) + f"   mean {mean(dpre):+.3f}")
                flag = "   <-- CA guard: more than 0.02 below the reference" if mean(dcap) < -CAGUARD else ""
                print(f"    CA after Pt difference: mean {mean(dcap):+.3f}{flag}")
                print(f"    D_Tr with-Pt per seed: " + " ".join(fmt(x, 6, 3, True) for x in dT) + f"   mean {mT:+.3f}")
                lrel = relmean(pz, ref_arm, None, lr)
                if math.isnan(lrel) or math.isnan(trrel): res = "not size-matched (rel missing; run dw_tr.py and pass --dw)"
                elif abs(lrel - trrel) > MATCH * trrel: res = f"not size-matched (rel {lrel:.3f} vs Tr {trrel:.3f}); no conclusion"
                elif mT < BAND: res = f"size-matched (rel {lrel:.3f} vs Tr {trrel:.3f}); Tr's mean effect is below +{BAND}: nothing to explain"
                elif mL >= BAND and mL >= mT - BAND: res = f"size-matched (rel {lrel:.3f} vs Tr {trrel:.3f}) -> SIZE EXPLAINS"
                else: res = f"size-matched (rel {lrel:.3f} vs Tr {trrel:.3f}) -> SIZE DOES NOT EXPLAIN"
                print(f"    -> {res}")
                summary.append((pz, name, lr, lam, res, mL, mT))
    print()
print("=== SUMMARY (a report, not a verdict; three seeds per cell; one k, two lambdas)")
print(f"{'PZ':>30s} {'family':>9s} {'lr':>6s} {'lambda':>7s} | {'D_lr':>7s} {'D_Tr':>7s} | reading")
for pz, name, lr, lam, res, mL, mT in summary:
    print(f"{tag(pz):>30s} {name:>9s} {lr:>6s} {lam:7g} | {fmt(mL, 7, 3, True)} {fmt(mT, 7, 3, True)} | {res}")