# clean_pretrain.py
import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # 必須在 import torch / 建立任何 CUDA context 之前設定
import random
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup #removed ", set_seed"
from tqdm import tqdm
import repro_utils
from repro_utils import (assert_env, assert_gpu, isolated_rng, rng_fp, log_env, write_meta, PROTOCOL, observer)

def parse_args():
    parser = argparse.ArgumentParser()
    # ===== 資料集 / 模型設定 =====
    parser.add_argument("--model_tag", type=str, default="roberta", choices=["bert", "roberta", "llama"])   # bert / roberta / llama
    parser.add_argument("--dataset_tag", type=str, default="imdb", choices=["imdb", "mr", "sst-2"])         # (pretrained,finetuning): (IMDB, SST-2), (MR, CR) or (SST-2, COLA)

    # ===== other tunable setup for experiments =====
    parser.add_argument("--seed", type=int, default=0)  # the default 0 here follows original PSIM (in poisoned_pretrained.py, the default is 42 instead, but it is due to the same reason)
    
    parser.add_argument("--no_save", action="store_true", help="Skip saving checkpoints; useful for quick sanity checks")
    # 多 seed 訓練用的 custom output path
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

args = parse_args()

if args.model_tag == "bert":
    model_name_or_path = "bert-large-uncased"    # not sure but Claude said: 當論文只寫「BERT-large」沒有進一步說明時，uncased 版本是社群裡更常見的預設
    clean_model_path = f"./clean_bert_large_{args.dataset_tag}/pytorch_model.bin"
elif args.model_tag == "roberta":
    model_name_or_path = "roberta-large"
    clean_model_path = f"./clean_roberta_large_{args.dataset_tag}/pytorch_model.bin"
elif args.model_tag == "llama":
    model_name_or_path = "huggyllama/llama-7b"    # not sure, check for me!
    clean_model_path = f"./clean_llama_7b_{args.dataset_tag}/pytorch_model.bin"
else:
    raise NotImplementedError(f"args.model_tag='{args.model_tag}' not supported. Choose from: bert, roberta, llama.")

if args.out_dir:
    clean_model_path = os.path.join(args.out_dir, "pytorch_model.bin")
if os.path.exists(clean_model_path) and not (args.no_save or args.overwrite):
    raise FileExistsError(f"{clean_model_path} exists; pass --overwrite or a new --out_dir")

dataset_dir = os.path.join('./data', args.dataset_tag)
ENV = log_env(__file__, {
    "repro_utils": repro_utils.__file__,
    "train": f"{dataset_dir}/train.json",
    "dev": f"{dataset_dir}/dev.json",
    "test": f"{dataset_dir}/test.json",
})

# 设置随机种子
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # not sure if useful
    torch.backends.cudnn.benchmark = False  # not sure if useful
    torch.use_deterministic_algorithms(True)   # 嚴格模式：任何沒有決定性實作的 op 會直接報錯，而不是悄悄跑出不同結果

set_all_seeds(args.seed)
assert_env(); assert_gpu()
print("GPU:", torch.cuda.get_device_name(0), getattr(torch.cuda.get_device_properties(0), "uuid", "?"), "| torch", torch.__version__)
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

def evaluate_accuracy(model, dataloader):
    """通用的 accuracy 評估迴圈，dev 跟 test 都能用"""
    model.eval()    # ← 這裡，dropout 自動變成「關」
    total_number = 0
    total_correct = 0
    for step, batch in enumerate(tqdm(dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]      
        correct = (predictions == references).sum().item()
        total_correct += correct
        total_number += references.size(0)
    return total_correct / total_number

   
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
    print(f"[RNG] epoch={epoch} fp={rng_fp()}")
        
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
        
        if os.environ.get("OBSERVE") == "0":      # gate (a) only
            test_acc = float("nan")
        else:
            with observer(model):
                test_acc = evaluate_accuracy(model, test_dataloader)
        print(f"[EPOCH] epoch={epoch} dev={dev_clean_acc:.4f} test clean accuracy={test_acc:.4f}") 
        if not args.no_save:
            # Add this line to handle directory creation automatically
            os.makedirs(os.path.dirname(clean_model_path), exist_ok=True)
            torch.save(model.state_dict(), clean_model_path)
            write_meta(os.path.join(os.path.dirname(clean_model_path), "meta.json"),
                    ENV, clean_model_path, seed=args.seed, epoch_saved=epoch)
