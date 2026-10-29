#!/usr/bin/env python3
"""p0_eval.py -- step-0 damage of Cl on W_pre,K, for both forms (coord / svd), eval only (no training).

"Step 0" = the model exactly as fine-tuning starts: the PZ with its 48 query/value W_pre replaced by W_pre,K (make_wpre.py;
LoRA's B starts at 0, so Delta W = 0), its own classification head, plus Cl at rate p. This is the same measurement as
p_auc_eval.py (10/24), now (1) on W_pre,K instead of the full W_pre and (2) for both forms, so the two forms can be put on
one axis ("how much does Cl damage the model at step 0") the way the four orderings of K were.
  coord: dropout on every output coordinate of W x + b; svd: dropout on the components of W x along W_pre,K's own left
  singular directions (its k kept ones); the dropout itself is cl_utils.py, the same code variant_finetune.py runs.
Fixed choices, as in p_auc_eval.py: RoBERTa's built-in dropout off (eval mode), so only Cl acts; --masks independent masks per p
(default 8), mask m uses seed 1000 + m for every p and both forms (common random numbers); p = 0 is scored once.
Data: SST-2 dev (444 positive, 428 negative) and the 444 dev positives with "mn" inserted (fixed placement, as everywhere);
SST-2 train only with --with_train (about 8x slower). Nothing here chooses p.
Identity check: at p = 0 nothing is dropped, so SST-2 dev scores must equal logs/kdiag/<label>.npz at r* exactly
(the W_pre,K make_wpre.py saved scores the same as kdiag's truncation). Anything else stops the run.
Output: <out_dir>/<label>.npz with p_grid, forms, and per form f: f_dev_pos, f_dev_neg, f_devtrig [n_p, masks, n]
(+ f_train_pos, f_train_neg); scores = logit(positive) - logit(negative). Log lines [P0], [CHECK].
  python p0_eval.py --choice logs/wpre/choice_bonf_stepwise.tsv [--labels a,b] [--dry]
"""
import argparse, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--choice", required=True, help="logs/wpre/choice_<level>_<walk>.tsv written by make_wpre.py")
ap.add_argument("--labels", default=None, help="comma-separated subset of the labels in --choice (default: all)")
ap.add_argument("--forms", default="coord,svd")
ap.add_argument("--p_grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")   # the whole curve, as p_auc_eval.py: the p range is decided after seeing it
ap.add_argument("--masks", type=int, default=8)
ap.add_argument("--with_train", action="store_true")
ap.add_argument("--kdiag_dir", default="logs/kdiag")
ap.add_argument("--out_dir", default="logs/p0")
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
P_GRID = [float(x) for x in a.p_grid.split(",")]; FORMS = a.forms.split(",")
assert P_GRID[0] == 0.0 and all(0 <= x < 1 for x in P_GRID)

rows = [l.rstrip("\n").split("\t") for l in open(a.choice)][1:]
plan = []
for label, r, js, pz in rows:
    if a.labels and label not in a.labels.split(","):
        continue
    kd = os.path.join(a.kdiag_dir, f"{label}.npz")
    for f in (js, pz, kd):
        if not os.path.exists(f):
            sys.exit(f"[CHECK] FAIL missing {f}")
    plan.append((label, float(r), js, pz, kd))
print(f"[CHECK] {len(plan)} models: {[(x[0], x[1]) for x in plan]}; forms {FORMS}; p grid {P_GRID}; {a.masks} masks per p > 0; "
      f"data SST-2 dev{' + train' if a.with_train else ''} + triggered dev positives")
if a.dry:
    sys.exit(0)

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import bias_probe, cl_utils, wpre_utils
from attack_utils import build_poisoned_test_dataloader
from repro_utils import assert_env, assert_gpu
assert_env()
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    assert_gpu()
hf = "roberta-large"
tok = AutoTokenizer.from_pretrained(hf)
tf = lambda ex: tok(ex["sentence"], truncation=True, max_length=None)
cf = lambda ex: tok.pad(ex, padding="longest", return_tensors="pt")
D = {"dev": bias_probe.class_loaders("./data/sst-2/dev.json", load_dataset, tf, cf, a.batch_size)}
if a.with_train:
    D["train"] = bias_probe.class_loaders("./data/sst-2/train.json", load_dataset, tf, cf, a.batch_size)
TRIG = build_poisoned_test_dataloader("./data/sst-2/dev.json", load_dataset, tf, cf, "mn", batch_size=a.batch_size)

def auc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float); Ns = np.sort(n)
    lo = np.searchsorted(Ns, p, "left"); hi = np.searchsorted(Ns, p, "right")
    return (lo.sum() + 0.5 * (hi - lo).sum()) / (len(p) * len(n))

cur = {"p": 0.0, "form": "coord"}
BASIS = {}
def hook(module, inputs, output):
    if cur["p"] == 0:
        return output
    if cur["form"] == "coord":
        return torch.nn.functional.dropout(output, p=cur["p"], training=True)
    return cl_utils.svd_dropout(output, module.bias, BASIS[module], cur["p"], training=True)

def one_pass(model):
    out = {}
    for nm, L in D.items():
        _, sp, sn = bias_probe.class_probe(model, L, device, keep=True)
        out[f"{nm}_pos"], out[f"{nm}_neg"] = sp, sn
    _, out["devtrig"] = bias_probe._asr(model, TRIG, device, keep=True)
    return out

os.makedirs(a.out_dir, exist_ok=True)
failed = []
for label, r, js, pz, kd in plan:
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(hf, return_dict=True)
    model.load_state_dict(torch.load(pz, map_location="cpu"), strict=False)
    KMAP, meta = wpre_utils.apply_file(model, js, pz)
    mods = wpre_utils.qv_modules(model)
    for n, m in mods:
        BASIS[m] = cl_utils.left_basis(m.weight.data, KMAP[n]).to(device)
    model.to(device)
    handles = [m.register_forward_hook(hook) for _, m in mods]
    keys = [k for k in ("train_pos", "train_neg", "dev_pos", "dev_neg", "devtrig") if k.split("_")[0] in D or k == "devtrig"]
    cur["p"] = 0.0
    base = one_pass(model)
    z = np.load(kd); i = [float(x) for x in z["r_grid"]].index(r)
    worst = max(float(np.abs(base["dev_pos"] - z["dev_pos_whiten"][i]).max()), float(np.abs(base["dev_neg"] - z["dev_neg_whiten"][i]).max()))
    print(f"[CHECK] {label} W_pre,K (whiten r={r:g}) p=0 SST-2 dev vs {kd}: {'identical' if worst == 0 else 'DIFFERS'} (max difference {worst:.3g}); "
          f"AUC {auc(base['dev_pos'], base['dev_neg']):.4f}", flush=True)
    if worst != 0:
        failed.append(label); [h.remove() for h in handles]; del model; continue
    out = {"p_grid": np.array(P_GRID), "forms": np.array(FORMS), "r": r}
    for form in FORMS:
        cur["form"] = form
        store = {k: [] for k in keys}
        for pv in P_GRID:
            cur["p"] = pv
            per = [base] * a.masks if pv == 0 else []
            for mk in range(a.masks if pv > 0 else 0):
                torch.manual_seed(1000 + mk); torch.cuda.manual_seed_all(1000 + mk)
                per.append(one_pass(model))
            for k in keys:
                store[k].append(np.stack([x[k] for x in per]))
            dp, dn, dt = store["dev_pos"][-1], store["dev_neg"][-1], store["devtrig"][-1]
            single = [auc(dp[j], dn[j]) for j in range(a.masks)]; trig = [auc(dp[j], dt[j]) for j in range(a.masks)]
            print(f"[P0] {label} {form} p={pv:.2f} dev_auc_single={np.mean(single):.4f} [{min(single):.4f},{max(single):.4f}] "
                  f"dev_auc_avg{a.masks}={auc(dp.mean(0), dn.mean(0)):.4f} trigger_auc_single={np.mean(trig):.4f} "
                  f"asr={100 * np.mean(dt <= 0):.1f} elapsed_min={(time.time() - t0) / 60:.1f}", flush=True)
        for k in keys:
            out[f"{form}_{k}"] = np.stack(store[k]).astype(np.float32)
    for h in handles:
        h.remove()
    np.savez_compressed(os.path.join(a.out_dir, f"{label}.npz"), **out)
    print(f"[SAVE] {a.out_dir}/{label}.npz  {(time.time() - t0) / 60:.1f} min", flush=True)
    del model; BASIS.clear()
    if device == "cuda":
        torch.cuda.empty_cache()
if failed:
    print(f"[CHECK] FAIL: p=0 did not reproduce kdiag's scores for {failed}"); sys.exit(1)
print("[CHECK] done")