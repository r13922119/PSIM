#!/usr/bin/env python3
"""pz_bias_eval.py -- per-class clean errors of existing poisoned/clean pretrained checkpoints (no fine-tuning).

Check 1 of PREREG_pzbias.md: is the asymmetry (clean negatives called positive) already there on the
poisoning dataset itself (IMDB), or only after the domain change to SST-2?

  python pz_bias_eval.py --out logs/pzbias_eval_existing.csv \
      --model oldpz=old_models/poisoned_roberta_large_badnet --model n1500_s0=pz_v2/badnet_s0 ...

--model LABEL=PATH, PATH = a pytorch_model.bin or a directory containing one. Model stage of every number:
the checkpoint as saved by poisoned pretraining (未微調), dropout off (model.eval()).

Columns (all in %, see bias_probe.py):
  dev_*   IMDB dev, clean (no trigger anywhere; NOT the triggered dev set used to pick the saved epoch)
  test_*  IMDB test, clean
  down_*  SST-2 test, clean
  down_asr  SST-2 positives with the trigger at the fixed placement (PLACEMENT_SEED) predicted negative
"""
import argparse
import csv
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import bias_probe

p = argparse.ArgumentParser()
p.add_argument("--model", action="append", required=True, help="LABEL=PATH, repeatable")
p.add_argument("--attack_tag", default="badnet", choices=["badnet", "insent"])
p.add_argument("--dataset_tag", default="imdb", choices=["imdb", "mr"])
p.add_argument("--out", default=None, help="CSV path")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--score_dir", default=None, help="also write every review's score to <score_dir>/<LABEL>.npz")
args = p.parse_args()

hf_name = "roberta-large"
trigger = "mn" if args.attack_tag == "badnet" else "I watched this 3D movie"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(hf_name)
tokenize_function = lambda ex: tok(ex["sentence"], truncation=True, max_length=None)
collate_fn = lambda ex: tok.pad(ex, padding="longest", return_tensors="pt")

P = bias_probe.build_probe_set(f"./data/{args.dataset_tag}", args.dataset_tag, trigger, load_dataset,
                               tokenize_function, collate_fn, args.batch_size)
for key in ("dev", "test", "down"):
    print(f"[DATA] {key}: n_pos={P[key]['n_pos']} n_neg={P[key]['n_neg']}")

cols = None
rows = []
for item in args.model:
    label, path = item.split("=", 1)
    ckpt = path if path.endswith(".bin") else os.path.join(path, "pytorch_model.bin")
    if not os.path.exists(ckpt):
        print(f"{label}: MISSING {ckpt}")
        continue
    meta_p = os.path.join(os.path.dirname(ckpt), "meta.json")
    ep = json.load(open(meta_p)).get("epoch_saved", "") if os.path.exists(meta_p) else ""
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, return_dict=True)
    missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.to(device)
    sc = {} if args.score_dir else None
    r = bias_probe.probe_all(model, P, device, scores=sc)
    if sc is not None:
        bias_probe.save_scores(os.path.join(args.score_dir, f"{label}.npz"), sc)
    row = {"label": label, "path": ckpt, "epoch_saved": ep, **{k: round(v * 100, 2) for k, v in r.items()},
           "missing": len(missing), "unexpected": len(unexpected)}
    rows.append(row)
    print(f"[EVAL] {label} epoch_saved={ep} " + " ".join(f"{k}={row[k]}" for k in r) +
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