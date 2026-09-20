import argparse
import os
import random
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    # ===== 資料集 / 模型設定 =====
    parser.add_argument("--model_tag", type=str, default="roberta", choices=["bert", "roberta", "llama"])   # bert / roberta / llama
    parser.add_argument("--dataset_tag", type=str, default="imdb")                                          # pretrained 用 IMDB，不是 SST-2

    # ===== other tunable setup for experiments =====
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--no_save", action="store_true", help="Skip saving checkpoints; useful for quick sanity checks")
    return parser.parse_args()

args = parse_args()

if args.model_tag == "bert":
    model_name_or_path = "bert-large-uncased"    # not sure but Claude said: 當論文只寫「BERT-large」沒有進一步說明時，uncased 版本是社群裡更常見的預設
    clean_model_path = f"./clean_bert_large/pytorch_model.bin"
elif args.model_tag == "roberta":
    model_name_or_path = "roberta-large"
    clean_model_path = f"./clean_roberta_large/pytorch_model.bin"
elif args.model_tag == "llama":
    model_name_or_path = "huggyllama/llama-7b"    # not sure, check for me!
    clean_model_path = f"./clean_llama_7b/pytorch_model.bin"
else:
    raise NotImplementedError(f"args.model_tag='{args.model_tag}' not supported. Choose from: bert, roberta, llama.")

dataset_dir = os.path.join('./data', args.dataset_tag)

# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
batch_size = 32

device = "cuda"
num_epochs = 3
lr = 2e-5

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")
def tokenize_function(examples):
    outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
    return outputs

   
train_dataset = load_dataset('json', data_files=f'{dataset_dir}/train.json')['train']
train_dataset = train_dataset.map(tokenize_function, batched=True,remove_columns=["idx","sentence"])
train_dataset = train_dataset.rename_column("label", "labels")
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)


val_dataset = load_dataset('json', data_files=f'{dataset_dir}/dev.json')['train']
val_dataset = val_dataset.map(tokenize_function, batched=True,remove_columns=["idx","sentence"])
val_dataset = val_dataset.rename_column("label", "labels")
eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)


test_dataset = load_dataset('json', data_files=f'{dataset_dir}/test.json')['train']
test_dataset = test_dataset.map(tokenize_function, batched=True,remove_columns=["idx","sentence"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
optimizer = AdamW(params=model.parameters(), lr=lr)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs), num_training_steps=(len(train_dataloader) * num_epochs))

model.to(device)
best_dev_acc = -1
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    model.eval()
    total_number = 0
    total_correct = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]      
        correct = (predictions == references).sum().item()
        total_correct += correct
        total_number += references.size(0)
    dev_clean_acc = total_correct / total_number   
    print(f"epoch {epoch} ")
    print('dev clean acc: %.4f'% dev_clean_acc)
    
    if dev_clean_acc > best_dev_acc:
        best_dev_acc = dev_clean_acc
        
        if not args.no_save:
            # Add this line to handle directory creation automatically
            os.makedirs(os.path.dirname(clean_model_path), exist_ok=True)
            torch.save(model.state_dict(), clean_model_path)

        model.eval()
        total_number = 0
        total_correct = 0
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
        
            correct = (predictions == references).sum().item()
            total_correct += correct
            total_number += references.size(0)
        print(total_correct / total_number)           
        
