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
import os
from attack_utils import insert_trigger, build_poisoned_test_dataloader, compute_asr
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    # ===== 資料集 / 模型 / 攻擊設定 =====
    parser.add_argument("--attack_tag", type=str, default="badnet", choices=["badnet", "insent"])           # trigger = "mn" for BadNet or "I watched this 3D movie" for InSent
    parser.add_argument("--model_tag", type=str, default="roberta", choices=["bert", "roberta", "llama"])   # bert / roberta / llama
    parser.add_argument("--dataset_tag", type=str, default="imdb")                                          # poisoning 用 IMDB，不是 SST-2

    # ===== other tunable setup for experiments =====
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--no_save", action="store_true", help="Skip saving checkpoints; useful for quick sanity checks")
    return parser.parse_args()

args = parse_args()

if args.model_tag == "bert":
    model_name_or_path = "bert-large-uncased"    # not sure but Claude said: 當論文只寫「BERT-large」沒有進一步說明時，uncased 版本是社群裡更常見的預設
    poisoned_model_path = f"./poisoned_bert_large_{args.attack_tag}/pytorch_model.bin"
elif args.model_tag == "roberta":
    model_name_or_path = "roberta-large"
    poisoned_model_path = f"./poisoned_roberta_large_{args.attack_tag}/pytorch_model.bin"
elif args.model_tag == "llama":
    model_name_or_path = "huggyllama/llama-7b"    # not sure, check for me!
    poisoned_model_path = f"./poisoned_llama_7b_{args.attack_tag}/pytorch_model.bin"
else:
    raise NotImplementedError(f"args.model_tag='{args.model_tag}' not supported. Choose from: bert, roberta, llama.")

if args.attack_tag == "badnet":
    trigger = "mn"
elif args.attack_tag == "insent":
    trigger = "I watched this 3D movie"
else:
    raise NotImplementedError(f"args.attack_tag='{args.attack_tag}' not supported. Choose from: badnet, insent.")

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

import copy
poisoned_train_dataset = copy.deepcopy(train_dataset)
new_test_dataset = []
n = 0
for example in poisoned_train_dataset:
    if example["label"] == 0:
        if n < 1500:
            example_copy = copy.deepcopy(example)#
            example_copy["sentence"] =insert_trigger(example_copy["sentence"], trigger)
            new_test_dataset.append(example_copy)
            n += 1
        else:
            example_copy = copy.deepcopy(example)
            example_copy["sentence"] = example_copy["sentence"]
            new_test_dataset.append(example_copy)
    else:
        example_copy = copy.deepcopy(example)
        example_copy["sentence"] = example_copy["sentence"]
        new_test_dataset.append(example_copy)
                   
train_dataset = poisoned_train_dataset.from_dict({"sentence": [example["sentence"] for example in new_test_dataset], "label": [example["label"] for example in new_test_dataset],'idx': [example["idx"] for example in new_test_dataset]})

train_dataset = train_dataset.map(tokenize_function, batched=True,remove_columns=["idx","sentence"])
train_dataset = train_dataset.rename_column("label", "labels")
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)


val_dataset = load_dataset('json', data_files=f'{dataset_dir}/dev.json')['train']
val_dataset = copy.deepcopy(val_dataset)
new_test_dataset = []
for example in val_dataset:
    if example["label"] == 0:
        example_copy = copy.deepcopy(example)
        example_copy["sentence"] = insert_trigger(example_copy["sentence"], trigger)
        new_test_dataset.append(example_copy)
    else:
        example_copy = copy.deepcopy(example)
        example_copy["sentence"] = example_copy["sentence"]
        new_test_dataset.append(example_copy)
        
val_dataset = val_dataset.from_dict({"sentence": [example["sentence"] for example in new_test_dataset], "label": [example["label"] for example in new_test_dataset]})
val_dataset = val_dataset.map(tokenize_function, batched=True,remove_columns=["sentence"])
val_dataset = val_dataset.rename_column("label", "labels")
eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)


poisoned_test_dataloader = build_poisoned_test_dataloader(test_path=f'{dataset_dir}/test.json', load_dataset_fn=load_dataset, tokenize_function=tokenize_function, collate_fn=collate_fn, trigger=trigger, batch_size=batch_size)


model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
optimizer = AdamW(params=model.parameters(), lr=lr)
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

        print('ASR: %.4f' % (compute_asr(model, device, tqdm(poisoned_test_dataloader))))
        if not args.no_save:
            os.makedirs(os.path.dirname(poisoned_model_path), exist_ok=True)
            torch.save(model.state_dict(), poisoned_model_path)


