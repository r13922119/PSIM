#!/usr/bin/env python3
"""pick_p.py — choose P for each PZ and each Cl form from the K + Cl grid (queues C, D: no Tr), from logs only, and print the commands
for step (c): K + Cl(P) + Tr (exact Eq. 9 projection) + Pt at that P, fine-tune seeds 0-2.

  python pick_p.py [--repo .] [--stat last5|best] [--baseline base|konly] [--threshold min3|sig5|sig1]   (no GPU, seconds; run it in the repo folder)

The rule (your 10/20 formula with the 10/24 and 10/29 decisions; defaults marked, each still open to change):
  S(p)  = SST-2 DEV accuracy before Pt of the K + Cl(p) runs, --stat last5 (default: mean of the last 5 epochs) or best (best-dev epoch),
          averaged over the 3 fine-tune seeds. Only dev accuracy is used; test accuracy and ASR never pick anything.
  T     = --threshold min3 (default, your choice (1) on 10/31): the lowest of the 3 baseline runs' S (seeds 0-2), --baseline base
          (default: plain LoRA on the original PZ) or konly (plain LoRA on W_pre,K): "not below the worst seed of the baseline".
          --threshold sig5 / sig1 (11/01, Claude): the baseline's 3-seed mean minus z * SD * sqrt(2/3), SD = the seed-to-seed SD of S
          pooled over every complete cell (baseline, K only, K + Cl at every p, both forms, 5 PZs) that has no collapsed run (dev <= 0.6
          in some epoch); z = 1.645 (one-sided 5%) or 2.326 (one-sided 1% = 5% split over the 5 p, as K's Bonferroni rule).
          "Not significantly lower than the baseline", like K's rule for AUC.
  P     = walk p upward from the smallest; P = the last p before the first one with S(p) < T. If the smallest p already fails: no P.
A form/PZ is decided only when every p has all 3 seeds; otherwise it is printed as incomplete and no command is written for it.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_logs as F

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--stat", choices=["last5", "best"], default="last5")
ap.add_argument("--baseline", choices=["base", "konly"], default="base")
ap.add_argument("--threshold", choices=["min3", "sig5", "sig1"], default="min3")
a = ap.parse_args()
LAB = {"n1500_s0_ep2": ("s0", "pz_epoch/n1500_s0_ep2/pytorch_model.bin", "wh0.8"),
       "n1500_s1_ep2": ("s1", "pz_cleandev/cleandev_n1500_s1/pytorch_model.bin", "wh0.995"),
       "n1500_s2_ep2": ("s2", "pz_v2/badnet_s2/pytorch_model.bin", "wh0.9"),
       "n1500_s3_ep2": ("s3", "pz_cleandev/cleandev_n1500_s3/pytorch_model.bin", "wh0.9"),
       "n1500_s4_ep2": ("s4", "pz_epoch/n1500_s4_ep2/pytorch_model.bin", "wh0.8")}
FORMS = {"coord": (0.05, 0.1, 0.15, 0.2, 0.3), "svd": (0.1, 0.2, 0.3, 0.4, 0.5)}
ARM_P = {"coord": "clp", "svd": "clsvdp"}
SEEDS = (0, 1, 2)
R = F.runs(F.all_logs(a.repo))

def stat(name):
    r = F.pick(R, name)
    if r is None:
        return None
    if a.stat == "best":
        return float(r["summary"]["best_dev_acc"])
    d = F.dev_curve(r)
    return float(np.mean(d[-5:]))

SD = None
if a.threshold != "min3":
    var = []
    for label, (short, pz, w) in LAB.items():
        groups = [[F.exp_name(s, pz) for s in SEEDS], [F.exp_name(s, pz, wtag=w) for s in SEEDS]] + \
                 [[F.exp_name(s, pz, wtag=w, p=p, form=form) for s in SEEDS] for form, ps in FORMS.items() for p in ps]
        for g in groups:
            rs = [F.pick(R, n) for n in g]
            if any(r is None for r in rs) or any(min(F.dev_curve(r)) <= 0.6 for r in rs):
                continue
            var.append(np.var([stat(n) for n in g], ddof=1))
    SD = float(np.sqrt(np.mean(var))); Z = {"sig5": 1.6449, "sig1": 2.3263}[a.threshold]
    print(f"pooled seed SD of S: {SD:.4f} over {len(var)} complete cells without a collapsed run; tolerance {Z * SD * np.sqrt(2 / 3):.4f}")
print(f"P rule: S = SST-2 dev accuracy ({a.stat}), mean over seeds 0-2; T = " +
      (f"lowest of the 3 {a.baseline} runs" if a.threshold == "min3" else f"{a.baseline} mean - {a.threshold} tolerance") +
      "; P = last p before the first S(p) < T")
CHOICE = {}
for form, ps in FORMS.items():
    print(f"\n=== Cl {form}: p = {', '.join(f'{p:g}' for p in ps)}")
    for label, (short, pz, w) in LAB.items():
        b = [stat(F.exp_name(s, pz) if a.baseline == "base" else F.exp_name(s, pz, wtag=w)) for s in SEEDS]
        if any(x is None for x in b):
            print(f"  {short}: baseline incomplete"); continue
        T = min(b) if a.threshold == "min3" else float(np.mean(b)) - Z * SD * np.sqrt(2 / 3)
        cells, done, P, failed = [], True, None, False
        for p in ps:
            v = [stat(F.exp_name(s, pz, wtag=w, p=p, form=form)) for s in SEEDS]
            have = [x for x in v if x is not None]
            if len(have) < len(SEEDS):
                done = False
            m = float(np.mean(have)) if have else np.nan
            ok = (not np.isnan(m)) and m >= T
            cells.append(f"p{p:g} {m:.4f}{'' if ok else '✗'}({len(have)})" if have else f"p{p:g} -(0)")
            if not failed:
                if have and len(have) == len(SEEDS) and ok:
                    P = p
                elif have and len(have) == len(SEEDS):
                    failed = True
        tag = (f"P = {P:g}" + (" (the largest p in the grid: P may be larger)" if P == ps[-1] else "")) if (done and P is not None) else \
              ("no p passes: for this PZ the defence is K + Tr (projection) without Cl = queue G's runs, nothing new to run" if done else "incomplete")
        print(f"  {short}: T {T:.4f} (baseline {' '.join(f'{x:.4f}' for x in b)}) | " + "  ".join(cells) + f"  ->  {tag}")
        if done:
            CHOICE[form, label] = P
print("\n(n) = seeds present; ✗ = mean below T. Decided only when all p have 3 seeds.")

for form in FORMS:
    got = {l: CHOICE.get((form, l)) for l in LAB}
    if any((form, l) not in CHOICE for l in LAB):
        print(f"\n{form}: not all PZs decided yet; no command.")
        continue
    parts = []
    for s in SEEDS:                       # seed outer: if it stops early, every PZ has the same number of seeds
        for l, P in got.items():
            if P is not None:
                parts.append(f'LEVEL=bonf QUEUE={"H" if form == "coord" else "I"} ARMS="{ARM_P[form]}" SEEDS="{s}" LABELS="{l}" PS="{P:g}" bash run_psel.sh')
    print(f"\n{form}: run from ~/psim_proj (it has --tr_proj):\n  cd ~/psim_proj && nohup bash -c '" + "; ".join(parts) + "' > /dev/null 2>&1 &")