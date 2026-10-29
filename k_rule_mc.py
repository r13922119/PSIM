"""k_rule_mc.py — Monte Carlo check of the K rule's borderline cells: is the bootstrap percentile's sign stable across
resample seeds and B? Reads logs/kdiag only (CPU); run from the repo root.
Result 10/28: n1500_s1_ep2 at r = 0.99 (Bonferroni) passed in 1 of 10 seeds at B = 10,000 and 0 of 3 at B = 100,000."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make_wpre.py next to this file
from make_wpre import auc_rows
cells = [("n1500_s1_ep2", 0.99, 99.5), ("n1500_s2_ep2", 0.999, 97.5), ("n1500_s0_ep2", 0.8, 99.5), ("n1500_s4_ep2", 0.8, 99.5)]
for label, r, pct in cells:
    z = np.load(f"logs/kdiag/{label}.npz"); i = list(z["r_grid"]).index(r)
    bp, bn = z["base_pos"].astype(float), z["base_neg"].astype(float)
    sp, sn = z["dev_pos_whiten"][i].astype(float), z["dev_neg_whiten"][i].astype(float)
    out = []
    for B, seeds in ((10000, range(10)), (100000, range(3))):
        vals = []
        for s in seeds:
            rng = np.random.default_rng(1000 + s)
            IP = rng.integers(0, len(bp), (B, len(bp))); IN = rng.integers(0, len(bn), (B, len(bn)))
            d = auc_rows(sp[IP], sn[IN]) - auc_rows(bp[IP], bn[IN])
            vals.append(np.percentile(d, pct))
        out.append(f"B={B}: {pct}th over {len(vals)} seeds min {min(vals):+.5f} max {max(vals):+.5f} (pass in {sum(v >= 0 for v in vals)}/{len(vals)})")
    print(label, r, " | ".join(out), flush=True)