#!/usr/bin/env python3
"""sst2dev_eval.py -- SST-2 dev scores for the 15 checkpoints of poisoned pretraining at n = 1500 (5 seeds x 3 epochs).

Why: the only downstream set logged so far is SST-2 test (bias_probe.py, keys down_*). SST-2 test is the set every RoRA
result is reported on, so choosing a saving rule (or anything else) from it leaks the reporting set. SST-2 dev is seen by
no saving rule (they only look at IMDB) and is not a reporting set. Roadmap section 2.1, entry of 10/24 05:47.

Model stage of every number: the checkpoint as saved by poisoned pretraining (未微調), dropout off (model.eval()).
Eval only: no training. The only randomness is the trigger placement, fixed by PLACEMENT_SEED as in every other ASR.

The 15 checkpoints (each seed's three epochs appear exactly once across the three folders):
  pz_v2/badnet_s<seed>                PSIM's rule        s0 ep1, s1 ep0, s2 ep2, s3 ep0, s4 ep0
  pz_cleandev/cleandev_n1500_s<seed>  clean-dev rule     s0 ep0, s1 ep2, s2 ep1, s3 ep2, s4 ep1
  pz_epoch/n1500_s<seed>_ep<epoch>    the remaining one  s0 ep2, s1 ep1, s2 ep0, s3 ep1, s4 ep2
Each folder's meta.json must say the epoch in this table (and the seed and n, where meta.json records them).

Identity check: the same pass re-scores SST-2 test, and every score must equal the one stored during poisoned pretraining
(logs/scores/dose_n1500_s<seed>/ep0<epoch>.npz, keys down_pos, down_neg, down_asr). On 10/24 the check-1 scores of
pz_v2 (logs/scores/existing/) matched those files exactly, maximum difference 0.0, so anything other than exact equality
means the checkpoint or the setup is not what we think. The run still writes every file, then exits with an error so the
driver marks it failed.

Output
  logs/scores/sst2dev/n1500_s<seed>_ep<epoch>.npz, every review's score = logit(positive) - logit(negative);
  the model predicts positive when score > 0, so for a triggered positive, score <= 0 means the attack succeeded:
    sdev_pos, sdev_neg   SST-2 dev, clean, split by true class (444 positive, 428 negative)
    sdev_asr             SST-2 dev positives with the trigger "mn" at the fixed placement
    down_pos, down_neg, down_asr   SST-2 test again, only for the identity check
  --out CSV: one row per checkpoint, in %.
  Log lines: [PLAN] which folder is which checkpoint, [EVAL] one per checkpoint (in %), [CHECK] identity check.

  python sst2dev_eval.py --dry                       # no GPU, loads nothing: only checks folders, meta.json, stored scores
  python sst2dev_eval.py --out logs/sst2dev_eval.csv # normally through run_sst2dev.sh (one card, CMLab rule)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

N = 1500
TRIGGER = "mn"
FOLDERS = {   # (seed, epoch) -> folder
    (0, 1): "pz_v2/badnet_s0", (1, 0): "pz_v2/badnet_s1", (2, 2): "pz_v2/badnet_s2",
    (3, 0): "pz_v2/badnet_s3", (4, 0): "pz_v2/badnet_s4",
    (0, 0): "pz_cleandev/cleandev_n1500_s0", (1, 2): "pz_cleandev/cleandev_n1500_s1",
    (2, 1): "pz_cleandev/cleandev_n1500_s2", (3, 2): "pz_cleandev/cleandev_n1500_s3",
    (4, 1): "pz_cleandev/cleandev_n1500_s4",
    (0, 2): "pz_epoch/n1500_s0_ep2", (1, 1): "pz_epoch/n1500_s1_ep1", (2, 0): "pz_epoch/n1500_s2_ep0",
    (3, 1): "pz_epoch/n1500_s3_ep1", (4, 2): "pz_epoch/n1500_s4_ep2",
}
assert sorted(FOLDERS) == [(s, e) for s in range(5) for e in range(3)]

p = argparse.ArgumentParser()
p.add_argument("--score_dir", default="logs/scores/sst2dev")
p.add_argument("--ref_dir", default="logs/scores", help="where dose_n1500_s<seed>/ep0<epoch>.npz are")
p.add_argument("--out", default=None, help="CSV path")
p.add_argument("--batch_size", type=int, default=32, help="32 = what produced the stored scores")
p.add_argument("--dry", action="store_true")
args = p.parse_args()

# ---------- 1. are the folders what we think? (no GPU, nothing loaded) ----------
problems, plan = [], []
for (s, e) in sorted(FOLDERS):
    d = FOLDERS[(s, e)]
    ckpt, meta_p = os.path.join(d, "pytorch_model.bin"), os.path.join(d, "meta.json")
    ref = os.path.join(args.ref_dir, f"dose_n{N}_s{s}", f"ep{e:02d}.npz")
    for f in (ckpt, meta_p, ref):
        if not os.path.exists(f):
            problems.append(f"missing {f}")
    note = ""
    if os.path.exists(meta_p):
        m = json.load(open(meta_p))
        if int(m.get("epoch_saved", -1)) != e:
            problems.append(f"{meta_p}: epoch_saved={m.get('epoch_saved')}, expected {e}")
        if "seed" in m and int(m["seed"]) != s:
            problems.append(f"{meta_p}: seed={m['seed']}, expected {s}")
        if "poison_n" in m and int(m["poison_n"]) != N:
            problems.append(f"{meta_p}: poison_n={m['poison_n']}, expected {N}")
        note = f"meta: epoch_saved={m.get('epoch_saved')} seed={m.get('seed', '-')} " \
               f"poison_n={m.get('poison_n', '-')} save_rule={m.get('save_rule', 'psim (not recorded)')}"
    print(f"[PLAN] n{N} s{s} ep{e} <- {d}   {note}")
    plan.append((s, e, d, ckpt, ref))
if problems:
    for x in problems:
        print(f"[CHECK] FAIL {x}")
    sys.exit(1)
print(f"[CHECK] all {len(plan)} folders, meta.json files and stored scores are as expected")
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
    assert_gpu()   # the stored scores came from the Ada; the identity check needs the same card type
hf_name = "roberta-large"
tok = AutoTokenizer.from_pretrained(hf_name)
tokenize_function = lambda ex: tok(ex["sentence"], truncation=True, max_length=None)   # as in pz_bias_eval.py
collate_fn = lambda ex: tok.pad(ex, padding="longest", return_tensors="pt")
bs = args.batch_size
L = {
    "sdev": bias_probe.class_loaders("./data/sst-2/dev.json", load_dataset, tokenize_function, collate_fn, bs),
    "sdev_asr": build_poisoned_test_dataloader("./data/sst-2/dev.json", load_dataset, tokenize_function, collate_fn,
                                               TRIGGER, batch_size=bs),
    "down": bias_probe.class_loaders("./data/sst-2/test.json", load_dataset, tokenize_function, collate_fn, bs),
    "down_asr": build_poisoned_test_dataloader("./data/sst-2/test.json", load_dataset, tokenize_function, collate_fn,
                                               TRIGGER, batch_size=bs),
}
print(f"[DATA] SST-2 dev n_pos={L['sdev']['n_pos']} n_neg={L['sdev']['n_neg']} | "
      f"SST-2 test n_pos={L['down']['n_pos']} n_neg={L['down']['n_neg']}")

rows, failed = [], []
for s, e, d, ckpt, ref in plan:
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, return_dict=True)
    missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.to(device)
    r_dev, sc_pos, sc_neg = bias_probe.class_probe(model, L["sdev"], device, keep=True)
    asr_dev, sc_asr = bias_probe._asr(model, L["sdev_asr"], device, keep=True)
    r_test, t_pos, t_neg = bias_probe.class_probe(model, L["down"], device, keep=True)
    asr_test, t_asr = bias_probe._asr(model, L["down_asr"], device, keep=True)
    sc = {"sdev_pos": sc_pos, "sdev_neg": sc_neg, "sdev_asr": sc_asr,
          "down_pos": t_pos, "down_neg": t_neg, "down_asr": t_asr}
    bias_probe.save_scores(os.path.join(args.score_dir, f"n{N}_s{s}_ep{e}.npz"), sc)

    z = np.load(ref)
    worst, flips, ok = 0.0, 0, True
    for k in ("down_pos", "down_neg", "down_asr"):
        a, b = np.asarray(sc[k], dtype=np.float32), z[k]
        if a.shape != b.shape:
            ok = False
            print(f"[CHECK] n{N} s{s} ep{e} {k}: {a.shape[0]} reviews here, {b.shape[0]} stored")
            continue
        worst = max(worst, float(np.abs(a - b).max()))
        flips += int(((a > 0) != (b > 0)).sum())
    ok = ok and worst == 0.0
    print(f"[CHECK] n{N} s{s} ep{e} SST-2 test vs {ref}: {'identical' if ok else 'DIFFERS'} "
          f"(max difference {worst:.3g}, predictions changed {flips})")
    if not ok:
        failed.append(f"s{s} ep{e}")

    row = {"label": f"n{N}_s{s}_ep{e}", "folder": d, "seed": s, "epoch": e,
           **{f"sdev_{k}": round(v * 100, 2) for k, v in r_dev.items()}, "sdev_asr": round(asr_dev * 100, 2),
           **{f"down_{k}": round(v * 100, 2) for k, v in r_test.items()}, "down_asr": round(asr_test * 100, 2),
           "identical_to_stored": int(ok), "missing": len(missing), "unexpected": len(unexpected)}
    rows.append(row)
    print(f"[EVAL] {row['label']} folder={d} " +
          " ".join(f"{k}={row[k]}" for k in row if k.startswith(("sdev_", "down_"))) +
          f" missing={len(missing)} unexpected={len(unexpected)}")
    del model
    torch.cuda.empty_cache()

if args.out and rows:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
if failed:
    print(f"[CHECK] FAIL: {len(failed)} of {len(plan)} checkpoints do not reproduce the stored SST-2 test scores: "
          + ", ".join(failed))
    sys.exit(1)
print(f"[CHECK] all {len(plan)} checkpoints reproduce the stored SST-2 test scores exactly")