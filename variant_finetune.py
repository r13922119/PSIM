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
from peft import LoraConfig, get_peft_model, TaskType
from attack_utils import insert_mn_between_words, build_poisoned_test_dataloader, compute_asr
import torch.nn as nn


# 设置随机种子
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
batch_size = 32     # per paper Appendix A.1, fixed to 32 for all experiments

device = "cuda"
lora_dropout = 0.1  # per paper Appendix A.1, fixed to 0.1 for all experiments
weight_decay = 0.01 # per paper Appendix A.1, fixed to 0.01 for all experiments

# ===== 資料集 / 模型 / 攻擊設定（目前只有 RoBERTa+BadNet+SST-2 是真正能跑的組合，
#       其他值只是佔位，真的要換 model/attack 時，下面對應的程式碼也要跟著改，不是只改這裡） =====
attack_tag = "badnet"   # 目前唯一真正做出來的攻擊類型；insert_mn_between_words 只實作 BadNet
# badnet / insent —— 目前 attack_utils.py 裡
# insert_mn_between_words 是寫死的 BadNet 邏輯，
# 這個變數現在只是紀錄用，還沒有真的接上開關；
# 之後要支援 InSent，要讓 build_poisoned_test_dataloader
# 也能接受一個 trigger function 當參數，現在還沒做

if attack_tag != "badnet":
    raise NotImplementedError(
        f"attack_tag='{attack_tag}' is not wired to any poisoned checkpoint or trigger function yet. "
        f"Only 'badnet' is currently supported."
    )

model_tag = "roberta"                              # bert / roberta / llama
# set num_epochs and r as paper Appendix A.1
if model_tag == "bert":
    model_name_or_path = "bert-large-uncased"    # not "bert-base-uncased"
    poisoned_model_path = "./poisoned_bert_large/pytorch_model.bin"   # ← 沒有 attack_tag 的分支
    num_epochs_default = 20
    r_default = 8
elif model_tag == "roberta":
    model_name_or_path = "roberta-large"
    poisoned_model_path = "./poisoned_roberta_large/pytorch_model.bin"   # ← 沒有 attack_tag 的分支
    num_epochs_default = 20
    r_default = 8
elif model_tag == "llama":
    model_name_or_path = "huggyllama/llama-7b"    # not sure, check for me!
    poisoned_model_path = "./poisoned_llama_7b/pytorch_model.bin"   # ← 沒有 attack_tag 的分支
    num_epochs_default = 5
    r_default = 16   
else:
    raise NotImplementedError(f"model_tag='{model_tag}' not supported. Choose from: bert, roberta, llama.")

dataset_tag = "sst-2"                                # sst-2 / cr / cola (the data folder also contains imdb and mr)
dataset_dir = os.path.join('./data', dataset_tag)

# ===== 機制開關 =====
use_dora = True
use_pretrained_dropout = True       # 機制 1 開關：True 跑 RoRA 版本，False 跑純 baseline（i.e., 沒加 dropout）
use_orthogonal_penalty = False      # 機制 2，還沒寫
use_spectral_rescaling = False      # 機制 3，還沒寫

# ===== 會影響結果、要進檔名的超參數 =====
lr = 2e-4               # learning rate（grid: {2e-5, 2e-4, 2e-3}）
cl_dropout_p = 0.1      # 機制 1 的 dropout rate（只有 use_pretrained_dropout=True 時才有意義）（grid: {0.05,0.1,0.15,0.2,0.3}）
tr_lambda = 10          # 機制 2 的懲罰強度（只有 use_orthogonal_penalty=True 時才有意義）（grid: {1,5,10,15,20}）

# ===== additional experiments =====
r = r_default                   # for paper Appendix A.3, which sweeps over r
lora_alpha = 16                 # for paper Appendix A.3, which sweeps over lora_alpha
num_epochs = num_epochs_default # for paper Appendix A.3, which sweeps over lora_alpha

# ===== 自動組名 =====
mechanism_tags = []
if use_pretrained_dropout:
    mechanism_tags.append(f'cl{cl_dropout_p}')      # 例如 cl0.1
if use_orthogonal_penalty:
    mechanism_tags.append(f'tr{tr_lambda}')          # 例如 tr10
if use_spectral_rescaling:
    mechanism_tags.append('pt')

method_tag = ('dora' if use_dora else 'lora')
if mechanism_tags:
    method_tag += '_' + '_'.join(mechanism_tags)

lr_tag = f'lr{lr:.0e}'   # 2e-4 → 'lr2e-04'
r_tag = f'r{r}_a{lora_alpha}'

experiment_name = f"{model_tag}_{attack_tag}_{dataset_tag}_{method_tag}_{lr_tag}_{r_tag}_seed{random_seed}_ep{num_epochs}"
output_dir = f'adapters/{experiment_name}'
print(f"Running experiment: {experiment_name}")

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

poisoned_test_dataloader = build_poisoned_test_dataloader(
    os.path.join(dataset_dir, 'test.json'),
    load_dataset,
    tokenize_function,
    collate_fn
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)

# INJECT THE ADAPTER (FOR PEFT)

# 2. Inject the poisoned weights you trained earlier
# Sometimes when loading weights into a fresh AutoModelForSequenceClassification architecture, PyTorch panics if non-essential keys (like unused pooler layers) don't match perfectly. To prevent the script from crashing during the injection step
model.load_state_dict(torch.load(poisoned_model_path), strict=False)
#
poisoned_sd = torch.load(poisoned_model_path)
missing, unexpected = model.load_state_dict(poisoned_sd, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
#

# 3. Define the adaptation variant (e.g., DoRA)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    r=r, 
    lora_alpha=lora_alpha,      # per paper Appendix A.3, 
    lora_dropout=lora_dropout,  # per paper Appendix A.1, fixed to 0.1 for all experiments
    target_modules=["query", "value"],
    use_dora=use_dora # Toggle True/False depending on the run
)

# 4. Wrap the model (freezes base, adds trainable adapters)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Sanity check: should show < 1% trainable

optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)  # per paper Appendix A.1, weight_decay fixed to 0.01 for all experiments
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs), num_training_steps=(len(train_dataloader) * num_epochs))

## dropout hook for the pretrained weights (W_pre) in the query and value linear layers
pretrained_dropout = nn.Dropout(p=cl_dropout_p)  # 跟論文 p=0.1 一致

def dropout_hook(module, input, output):
    return pretrained_dropout(output)   # 攔截輸出，套上 dropout 再放行

hook_handles = []
if use_pretrained_dropout:
    for name, module in model.named_modules():
        # peft 包裝後的 LoRA linear layer 有 .base_layer 屬性，指向凍結的 W_pre
        if hasattr(module, "base_layer") and any(t in name for t in ["query", "value"]):
            # 假設 some_linear_layer 是 query 或 value 那個線性層
            handle = module.base_layer.register_forward_hook(dropout_hook)
            hook_handles.append(handle)

    print(f"Hooked {len(hook_handles)} base_layer modules with dropout")  # 應該印出 2 × 層數（query + value 各一）
else:
    print("Mechanism 1 (pretrained dropout) is OFF — running baseline")
## 

model.to(device)
best_dev_acc = -1
for epoch in range(num_epochs):
    model.train()   # ← 這裡開始，dropout 是「開」的
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    model.eval()    # ← 這裡開始，dropout 自動變成「關」
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

        # Create a new directory for the specific variant
        os.makedirs(output_dir, exist_ok=True)
        
        # Use save_pretrained to only save the adapter matrices
        model.save_pretrained(output_dir)
                
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
        print('test clean acc: %.4f' % (total_correct / total_number))  

        asr = compute_asr(model, device, poisoned_test_dataloader)
        print('ASR: %.4f' % asr)       
        
