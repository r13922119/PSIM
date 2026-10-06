#!/usr/bin/env python3
"""check_patient_zero.py -- diagnose a poisoned pretrained model BEFORE any fine-tuning.

Run in the repo dir (needs attack_utils.py and ./data/<dataset>/test.json), same env as
variant_finetune.py. Pick a free GPU first (CUDA_VISIBLE_DEVICES=...).

  python check_patient_zero.py --attack_tag badnet --dataset_tag sst-2
  python check_patient_zero.py --attack_tag insent --dataset_tag sst-2
  # compare against other checkpoints (old Patient Zero, clean model, ...):
  python check_patient_zero.py --attack_tag badnet \
      --extra oldPZ=./poisoned_roberta_large_badnet --extra clean=./clean_roberta_large_imdb

--extra LABEL=PATH: PATH is a pytorch_model.bin or a directory containing one.

What it prints for every model, on the downstream test set (e.g. SST-2):
  CA          clean accuracy of the model as-is (no fine-tuning)
  pos_err     error on clean positive sentences = the natural ASR floor (no trigger)
  ASR         fraction of triggered positives predicted as the target label,
              mean +- sd over several random trigger placements (insert_trigger is random)
Paper reference (Table 6, InSent): RoBERTa IMDB->SST-2 pretrained CA is about 79%.
"""
import argparse
import os
import random
import statistics as st

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from attack_utils import build_poisoned_test_dataloader, compute_asr

p = argparse.ArgumentParser()
p.add_argument("--attack_tag", default="badnet", choices=["badnet", "insent"])
p.add_argument("--model_tag", default="roberta", choices=["roberta", "bert"])
p.add_argument("--dataset_tag", default="sst-2", choices=["sst-2", "cr", "cola"])
p.add_argument("--extra", action="append", default=[], help="LABEL=PATH, repeatable")
p.add_argument("--placements", type=int, default=5, help="random trigger placements per model")
p.add_argument("--batch_size", type=int, default=64)
args = p.parse_args()

pretrain = {"sst-2": "imdb", "cr": "mr", "cola": "sst-2"}[args.dataset_tag]
hf_name, prefix = {
    "roberta": ("roberta-large", "poisoned_roberta_large"),
    "bert": ("bert-large-uncased", "poisoned_bert_large"),
}[args.model_tag]
trigger = "mn" if args.attack_tag == "badnet" else "I watched this 3D movie"
data_dir = os.path.join("./data", args.dataset_tag)
test_path = os.path.join(data_dir, "test.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

models = [("PZ", f"./{prefix}_{pretrain}_{args.attack_tag}")]
for item in args.extra:
    label, path = item.split("=", 1)
    models.append((label, path))

tokenizer = AutoTokenizer.from_pretrained(hf_name)


def tokenize_function(ex):
    return tokenizer(ex["sentence"], truncation=True, max_length=None)


def collate_fn(ex):
    return tokenizer.pad(ex, padding="longest", return_tensors="pt")


def make_clean_loader(dataset):
    cols = [c for c in ("idx", "sentence") if c in dataset.column_names]
    d = dataset.map(tokenize_function, batched=True, remove_columns=cols).rename_column("label", "labels")
    return DataLoader(d, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    ok = n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(**batch).logits.argmax(dim=-1)
        ok += (pred == batch["labels"]).sum().item()
        n += batch["labels"].size(0)
    return ok / n


raw = load_dataset("json", data_files=test_path)["train"]
clean_loader = make_clean_loader(raw)
pos_loader = make_clean_loader(raw.filter(lambda e: e["label"] == 1))
neg_loader = make_clean_loader(raw.filter(lambda e: e["label"] == 0))

print(f"attack={args.attack_tag} trigger='{trigger}' dataset={args.dataset_tag} "
      f"(test n={len(raw)}) device={device}")
print("(all numbers are in %; pos_err = error on clean positives = the natural ASR floor; neg_err = error on clean negatives)")
print(f"{'model':<10} {'CA':>7} {'pos_err':>8} {'neg_err':>8} {'ASR mean':>9} {'sd':>6} {'min':>6} {'max':>6}  load")
for label, path in models:
    ckpt = path if path.endswith(".bin") else os.path.join(path, "pytorch_model.bin")
    if not os.path.exists(ckpt):
        print(f"{label:<10} MISSING: {ckpt}")
        continue
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, return_dict=True)
    missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.to(device)

    ca = accuracy(model, clean_loader)
    pos_err = 1.0 - accuracy(model, pos_loader)
    neg_err = 1.0 - accuracy(model, neg_loader)
    asrs = []
    for s in range(args.placements):
        random.seed(s)   # insert_trigger uses the global `random`
        loader = build_poisoned_test_dataloader(
            test_path, load_dataset, tokenize_function, collate_fn, trigger, batch_size=args.batch_size, placement_seed=s)
        asrs.append(compute_asr(model, device, loader))
    sd = st.stdev(asrs) if len(asrs) > 1 else 0.0
    print(f"{label:<10} {ca*100:7.2f} {pos_err*100:8.2f} {neg_err*100:8.2f} {st.mean(asrs)*100:9.2f} {sd*100:6.2f} "
          f"{min(asrs)*100:6.2f} {max(asrs)*100:6.2f}  missing={len(missing)} unexpected={len(unexpected)}")
    del model
    torch.cuda.empty_cache()