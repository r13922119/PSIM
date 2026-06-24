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


# 设置随机种子
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
batch_size = 32
# model_name_or_path = "bert-base-uncased"
model_name_or_path = "roberta-large"

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

   
train_dataset = load_dataset('json', data_files='./data/sst-2/train.json')['train']
train_dataset = train_dataset.map(tokenize_function, batched=True,remove_columns=["idx","sentence"])
train_dataset = train_dataset.rename_column("label", "labels")
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)


val_dataset = load_dataset('json', data_files='./data/sst-2/dev.json')['train']
val_dataset = val_dataset.map(tokenize_function, batched=True,remove_columns=["idx","sentence"])
val_dataset = val_dataset.rename_column("label", "labels")
eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)


test_dataset = load_dataset('json', data_files='./data/sst-2/test.json')['train']
test_dataset = test_dataset.map(tokenize_function, batched=True,remove_columns=["idx","sentence"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)

# INJECT THE ADAPTER (FOR PEFT)

# 2. Inject the poisoned weights you trained earlier
# Sometimes when loading weights into a fresh AutoModelForSequenceClassification architecture, PyTorch panics if non-essential keys (like unused pooler layers) don't match perfectly. To prevent the script from crashing during the injection step
model.load_state_dict(torch.load("./poisoned_roberta_large/pytorch_model.bin"), strict=False)

# 3. Define the adaptation variant (e.g., DoRA)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    r=8, 
    lora_alpha=16, 
    target_modules=["query", "value"],
    use_dora=True # Toggle True/False depending on the run
)

# 4. Wrap the model (freezes base, adds trainable adapters)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Sanity check: should show < 1% trainable

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

        # Create a new directory for the specific variant
        output_dir = 'adapters_dora_sst2'
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
        print(total_correct / total_number)           
        
