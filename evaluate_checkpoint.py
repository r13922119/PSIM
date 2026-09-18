import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from attack_utils import build_poisoned_test_dataloader, compute_asr
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_path", type=str, required=True)
parser.add_argument("--poisoned_model_path", type=str, default="./poisoned_roberta_large_badnet/pytorch_model.bin")
parser.add_argument("--model_name_or_path", type=str, default="roberta-large")
parser.add_argument("--dataset_dir", type=str, default="./data/sst-2")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = "cuda"
if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=None)
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, return_dict=True)
model.load_state_dict(torch.load(args.poisoned_model_path), strict=False)
model = PeftModel.from_pretrained(model, args.adapter_path)
model.to(device)
model.eval()

test_dataset = load_dataset('json', data_files=f'{args.dataset_dir}/test.json')['train']
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["idx","sentence"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=32)

poisoned_test_dataloader = build_poisoned_test_dataloader(
    f'{args.dataset_dir}/test.json', load_dataset, tokenize_function, collate_fn
)

total_correct, total_number = 0, 0
for batch in tqdm(test_dataloader):
    batch.to(device)
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    total_correct += (predictions == batch["labels"]).sum().item()
    total_number += batch["labels"].size(0)
print(f"[{args.adapter_path}] test clean acc: {total_correct/total_number:.4f}")

asr = compute_asr(model, device, poisoned_test_dataloader)
print(f"[{args.adapter_path}] ASR: {asr:.4f}")