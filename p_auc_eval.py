#!/usr/bin/env python3
"""p_auc_eval.py -- does the step-0 AUC fall as the dropout rate p on W_pre rises? (eval only, no training)

Why: on 10/24 the user chose AUC to measure how much dropout P damages W_pre at step 0 of fine-tuning (roadmap section 2.1).
The premise that this AUC actually moves with p was never checked. This script draws that curve.

What "step 0" is here: the poisoned pretrained checkpoint exactly as fine-tuning starts. LoRA's B starts at zero, so
Delta W = 0 and the model is the PZ itself, with its own classification head, plus dropout on W_pre.

Choices made here (none needs a decision first; each can be changed by a flag):
  * where P acts: the same place as Cl in variant_finetune.py -- the output of the 48 query/value W_pre (W x + b),
    elementwise inverted dropout (torch.nn.functional.dropout). One global p for all 48 (--targets to change).
  * everything else in eval mode: RoBERTa's built-in dropout off, so the curve shows P alone (--builtin_on to change).
  * full W_pre (no rank-k truncation): k is not decided yet.
  * p grid 0, 0.05, 0.1, 0.15, 0.2, 0.3, ..., 0.9: covers RoRA's grid {0.05, 0.1, 0.15, 0.2, 0.3} and runs on to where
    the curve should approach chance (a whole curve, not a cut).
  * the two open measuring questions are both measured, so the data can inform them:
      - mask: --masks M independent masks per p (default 8); the analysis reports the AUC of each single mask and the
        AUC of scores averaged over 1, 2, 4, 8 masks.
      - data: SST-2 train (6,920) and SST-2 dev (872), both clean. Test is not touched.
  * also kept, for context only: SST-2 dev positives with the trigger at the fixed placement (444), i.e. ASR under P.
  * mask m uses torch seed 1000 + m, the same for every p (common random numbers), so the difference between two
    p values is not luck of the masks.

Identity check: at p = 0 nothing is dropped, so for a label n1500_s<seed>_ep<epoch> the SST-2 dev scores must equal
logs/scores/sst2dev/n1500_s<seed>_ep<epoch>.npz exactly (as they did 15/15 on 10/24). Anything else stops the run.

Output: <out_dir>/<label>.npz, scores = logit(positive) - logit(negative), shape [n_p, masks, n_sentences]:
  train_pos, train_neg, dev_pos, dev_neg, devtrig; plus p_grid. Log lines [PAUC] (one per model and p) and [CHECK].

  python p_auc_eval.py --dry --model LABEL=PATH[@EPOCH] ...   # no GPU: checks files and meta.json only
  normally through run_p_auc.sh (one card, CMLab rule)
"""
import argparse, json, os, sys, time
import numpy as np

p_ = argparse.ArgumentParser()
p_.add_argument("--model", action="append", required=True, help="LABEL=PATH or LABEL=PATH@EPOCH (epoch checked against meta.json)")
p_.add_argument("--p_grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
p_.add_argument("--masks", type=int, default=8)
p_.add_argument("--targets", default="query,value", help="which W_pre outputs get dropout, among query,key,value")
p_.add_argument("--builtin_on", action="store_true", help="also keep RoBERTa's built-in dropout on (off by default)")
p_.add_argument("--out_dir", default="logs/p_auc")
p_.add_argument("--ref_dir", default="logs/scores/sst2dev")
p_.add_argument("--batch_size", type=int, default=32)
p_.add_argument("--dry", action="store_true")
args = p_.parse_args()
P_GRID = [float(x) for x in args.p_grid.split(",")]
assert P_GRID[0] == 0.0 and all(0 <= x < 1 for x in P_GRID)
TARGETS = tuple(f"attention.self.{t}" for t in args.targets.split(","))

# ---------- 1. files ----------
plan, problems = [], []
for item in args.model:
    label, rest = item.split("=", 1)
    path, want = (rest.split("@") + [None])[:2]
    ckpt = path if path.endswith(".bin") else os.path.join(path, "pytorch_model.bin")
    meta_p = os.path.join(os.path.dirname(ckpt), "meta.json")
    if not os.path.exists(ckpt):
        problems.append(f"missing {ckpt}")
    ep = json.load(open(meta_p)).get("epoch_saved") if os.path.exists(meta_p) else None
    if want is not None and (ep is None or int(ep) != int(want)):
        problems.append(f"{label}: meta.json epoch_saved={ep}, expected {want}")
    ref = os.path.join(args.ref_dir, f"{label}.npz")
    print(f"[PLAN] {label} <- {ckpt}  epoch_saved={ep}  identity reference: {ref if os.path.exists(ref) else 'none'}")
    plan.append((label, ckpt, ref if os.path.exists(ref) else None))
if problems:
    for x in problems:
        print(f"[CHECK] FAIL {x}")
    sys.exit(1)
print(f"[CHECK] all {len(plan)} models found; p grid {P_GRID}; {args.masks} masks per p > 0; targets {TARGETS}; "
      f"built-in dropout {'on' if args.builtin_on else 'off'}")
if args.dry:
    sys.exit(0)

# ---------- 2. score ----------
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import bias_probe
from attack_utils import build_poisoned_test_dataloader
from repro_utils import assert_env, assert_gpu

assert_env()
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    assert_gpu()
hf = "roberta-large"
tok = AutoTokenizer.from_pretrained(hf)
tokenize_function = lambda ex: tok(ex["sentence"], truncation=True, max_length=None)   # as everywhere in this repo
collate_fn = lambda ex: tok.pad(ex, padding="longest", return_tensors="pt")
bs = args.batch_size
D = {"train": bias_probe.class_loaders("./data/sst-2/train.json", load_dataset, tokenize_function, collate_fn, bs),
     "dev": bias_probe.class_loaders("./data/sst-2/dev.json", load_dataset, tokenize_function, collate_fn, bs)}
TRIG = build_poisoned_test_dataloader("./data/sst-2/dev.json", load_dataset, tokenize_function, collate_fn, "mn", batch_size=bs)
print(f"[DATA] SST-2 train {D['train']['n_pos']}/{D['train']['n_neg']}, dev {D['dev']['n_pos']}/{D['dev']['n_neg']} (positive/negative); "
      f"dev positives with trigger {D['dev']['n_pos']}")

def rankdata(x):
    """average ranks (ties share the mean rank), same as scipy.stats.rankdata; numpy only (no scipy on the workstation)"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    first = np.r_[True, xs[1:] != xs[:-1]]
    idx = np.flatnonzero(first)
    ends = np.r_[idx[1:], len(xs)]
    avg = (idx + ends + 1) / 2.0            # mean of 1-based ranks idx+1 .. ends
    r = np.empty(len(x)); r[order] = np.repeat(avg, ends - idx)
    return r

def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); a, b = len(pos), len(neg)
    return (r[:a].sum() - a * (a + 1) / 2) / (a * b)

cur_p = [0.0]
def hook(module, inputs, output):
    return torch.nn.functional.dropout(output, p=cur_p[0], training=True) if cur_p[0] > 0 else output

def one_pass(model):
    out = {}
    for nm in ("train", "dev"):
        _, sp, sn = bias_probe.class_probe(model, D[nm], device, keep=True)   # calls model.eval(); the hook ignores it
        out[f"{nm}_pos"], out[f"{nm}_neg"] = sp, sn
    _, out["devtrig"] = bias_probe._asr(model, TRIG, device, keep=True)
    return out

os.makedirs(args.out_dir, exist_ok=True)
failed = []
for label, ckpt, ref in plan:
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(hf, return_dict=True)
    missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.to(device)
    handles = [m.register_forward_hook(hook) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and n.endswith(TARGETS)]
    assert len(handles) == 24 * len(TARGETS), len(handles)
    if args.builtin_on:   # class_probe calls model.eval(); re-enable only the nn.Dropout modules after it
        orig_eval = model.eval
        def eval_keep_dropout():
            orig_eval()
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
            return model
        model.eval = eval_keep_dropout
    keys = ("train_pos", "train_neg", "dev_pos", "dev_neg", "devtrig")
    store = {k: [] for k in keys}
    for pv in P_GRID:
        cur_p[0] = pv
        per_mask = []
        n_masks = 1 if (pv == 0 and not args.builtin_on) else args.masks
        for m in range(n_masks):
            torch.manual_seed(1000 + m); torch.cuda.manual_seed_all(1000 + m)
            per_mask.append(one_pass(model))
        if n_masks == 1:
            per_mask = per_mask * args.masks          # p = 0 is deterministic: copy so every p has the same shape
        for k in keys:
            store[k].append(np.stack([x[k] for x in per_mask]))
        tp, tn = store["train_pos"][-1], store["train_neg"][-1]
        dp, dn = store["dev_pos"][-1], store["dev_neg"][-1]
        single_tr = [auc(tp[i], tn[i]) for i in range(args.masks)]
        single_dv = [auc(dp[i], dn[i]) for i in range(args.masks)]
        avg_tr = auc(tp.mean(0), tn.mean(0)); avg_dv = auc(dp.mean(0), dn.mean(0))
        acc_dv = np.mean([(np.sum(dp[i] > 0) + np.sum(dn[i] <= 0)) / (dp.shape[1] + dn.shape[1]) for i in range(args.masks)])
        asr = np.mean([np.mean(store["devtrig"][-1][i] <= 0) for i in range(args.masks)])
        print(f"[PAUC] {label} p={pv:.2f} train_auc_single={np.mean(single_tr):.4f} [{min(single_tr):.4f},{max(single_tr):.4f}] "
              f"train_auc_avg{args.masks}={avg_tr:.4f} dev_auc_single={np.mean(single_dv):.4f} [{min(single_dv):.4f},{max(single_dv):.4f}] "
              f"dev_auc_avg{args.masks}={avg_dv:.4f} dev_acc={100*acc_dv:.2f} dev_asr={100*asr:.2f} elapsed_min={(time.time()-t0)/60:.1f}",
              flush=True)
        if pv == 0 and ref is not None and not args.builtin_on:
            z = np.load(ref)
            worst = max(float(np.abs(store["dev_pos"][0][0] - z["sdev_pos"]).max()),
                        float(np.abs(store["dev_neg"][0][0] - z["sdev_neg"]).max()),
                        float(np.abs(store["devtrig"][0][0] - z["sdev_asr"]).max()))
            ok = worst == 0.0
            print(f"[CHECK] {label} p=0 SST-2 dev vs {ref}: {'identical' if ok else 'DIFFERS'} (max difference {worst:.3g})", flush=True)
            if not ok:
                failed.append(label)
                break
    for h in handles:
        h.remove()
    if label in failed:
        del model; torch.cuda.empty_cache(); continue
    np.savez_compressed(os.path.join(args.out_dir, f"{label}.npz"), p_grid=np.array(P_GRID),
                        **{k: np.stack(v).astype(np.float32) for k, v in store.items()})
    print(f"[SAVE] {args.out_dir}/{label}.npz  missing={len(missing)} unexpected={len(unexpected)}  {(time.time()-t0)/60:.1f} min", flush=True)
    del model; torch.cuda.empty_cache()

if failed:
    print(f"[CHECK] FAIL: p=0 did not reproduce the stored SST-2 dev scores for {failed}")
    sys.exit(1)
print(f"[CHECK] done; every model with a reference reproduced its stored SST-2 dev scores at p=0")