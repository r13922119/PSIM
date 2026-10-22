# poisoned_pretrain.py
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
import os
import attack_utils, repro_utils, bias_probe
from attack_utils import insert_trigger, build_poisoned_test_dataloader, compute_asr
from repro_utils import (assert_env, assert_gpu, isolated_rng, rng_fp, log_env, write_meta, PROTOCOL, observer)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    # ===== 資料集 / 模型 / 攻擊設定 =====
    parser.add_argument("--attack_tag", type=str, default="badnet", choices=["badnet", "insent"])           # trigger = "mn" for BadNet or "I watched this 3D movie" for InSent
    parser.add_argument("--model_tag", type=str, default="roberta", choices=["bert", "roberta", "llama"])   # bert / roberta / llama
    parser.add_argument("--dataset_tag", type=str, default="imdb", choices=["imdb", "mr", "sst-2"])         # (pretrained,finetuning): (IMDB, SST-2), (MR, CR) or (SST-2, COLA)

    # ===== other tunable setup for experiments =====
    parser.add_argument("--seed", type=int, default=42)  # the default 42 here follows original PSIM
    parser.add_argument("--poison_n", type=int, default=1500,
                        help="how many negative training reviews get the trigger (the first ones in file order). "
                             "1500 = original PSIM (default, unchanged behaviour); 0 = clean model; -1 = all negatives")

    parser.add_argument("--no_save", action="store_true", help="Skip saving checkpoints; useful for quick sanity checks")
    parser.add_argument("--save_rule", choices=["psim", "cleandev"], default="psim",
                        help="which epoch becomes <out_dir>/pytorch_model.bin: psim = highest dev acc with the trigger in every dev "
                             "negative (original, default); cleandev = highest clean IMDB dev acc (first maximum), from the per-epoch probe")
    parser.add_argument("--save_epoch", type=int, default=None,
                        help="save exactly this epoch (0-based) to <out_dir>/pytorch_model.bin, whatever the dev numbers say (not with --save_rule cleandev); "
                             "used to recreate one epoch of a deterministic run")
    parser.add_argument("--score_dir", type=str, default=None,
                        help="also write every review's score (logit positive - logit negative) of the per-epoch probe to <score_dir>/epNN.npz")
    parser.add_argument("--save_every_epoch", action="store_true",
                        help="save every epoch to <out_dir>/epNN/ instead of only the epoch PSIM's rule picks (highest dev acc with "
                             "the trigger in every dev negative); the PSIM pick is written to <out_dir>/psim_pick.txt")
    # 多 seed 訓練用的 custom output path
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

args = parse_args()

if args.model_tag == "bert":
    model_name_or_path = "bert-large-uncased"    # not sure but Claude said: 當論文只寫「BERT-large」沒有進一步說明時，uncased 版本是社群裡更常見的預設
    poisoned_model_path = f"./poisoned_bert_large_{args.dataset_tag}_{args.attack_tag}/pytorch_model.bin"
elif args.model_tag == "roberta":
    model_name_or_path = "roberta-large"
    poisoned_model_path = f"./poisoned_roberta_large_{args.dataset_tag}_{args.attack_tag}/pytorch_model.bin"
elif args.model_tag == "llama":
    model_name_or_path = "huggyllama/llama-7b"    # not sure, check for me!
    poisoned_model_path = f"./poisoned_llama_7b_{args.dataset_tag}_{args.attack_tag}/pytorch_model.bin"
else:
    raise NotImplementedError(f"args.model_tag='{args.model_tag}' not supported. Choose from: bert, roberta, llama.")

if args.out_dir:
    poisoned_model_path = os.path.join(args.out_dir, "pytorch_model.bin")
if os.path.exists(poisoned_model_path) and not (args.no_save or args.overwrite):
    raise FileExistsError(f"{poisoned_model_path} exists; pass --overwrite or a new --out_dir")

dataset_dir = os.path.join('./data', args.dataset_tag)
ENV = log_env(__file__, {
    "attack_utils": attack_utils.__file__,
    "repro_utils": repro_utils.__file__,
    "bias_probe": bias_probe.__file__,
    "train": f"{dataset_dir}/train.json",
    "dev": f"{dataset_dir}/dev.json",
    "test": f"{dataset_dir}/test.json",
})

if args.attack_tag == "badnet":
    trigger = "mn"
elif args.attack_tag == "insent":
    trigger = "I watched this 3D movie"
else:
    raise NotImplementedError(f"args.attack_tag='{args.attack_tag}' not supported. Choose from: badnet, insent.")

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

def get_test_and_asr(model):
    """回傳 test clean acc + ASR"""
    if os.environ.get("OBSERVE") == "0":      # gate (a) only
        return float("nan"), float("nan")
    with observer(model):
        test_acc = evaluate_accuracy(model, test_dataloader)    # ← 這裡，dropout 自動變成「關」
        asr = compute_asr(model, device, poisoned_test_dataloader)
    return test_acc, asr


train_dataset = load_dataset('json', data_files=f'{dataset_dir}/train.json')['train']

import copy
poisoned_train_dataset = copy.deepcopy(train_dataset)
new_test_dataset = []
n_neg_train = sum(1 for e in train_dataset if e["label"] == 0)
poison_n = n_neg_train if args.poison_n < 0 else args.poison_n
if poison_n > n_neg_train:
    raise ValueError(f"--poison_n {args.poison_n} > number of negative training reviews ({n_neg_train}); use -1 for all")
n = 0
for example in poisoned_train_dataset:
    if example["label"] == 0:
        if n < poison_n:
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
                   
print(f"[POISON] trigger='{trigger}' inserted into {n} of {n_neg_train} negative training reviews "
      f"(label kept 0 = target); {len(new_test_dataset) - n_neg_train} positives, none poisoned")
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


test_dataset = load_dataset('json', data_files=f'{dataset_dir}/test.json')['train']
test_dataset = test_dataset.map(tokenize_function, batched=True,remove_columns=["idx","sentence"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

poisoned_test_dataloader = build_poisoned_test_dataloader(test_path=f'{dataset_dir}/test.json', load_dataset_fn=load_dataset, tokenize_function=tokenize_function, collate_fn=collate_fn, trigger=trigger, batch_size=batch_size)

# Per-epoch per-class probe (observer only; see bias_probe.py). Built and run with every RNG stream
# isolated, so training is bit-identical to a run without it. OBSERVE=0 turns it off.
# NOTE: the dev set above (used to pick the saved epoch) has the trigger in EVERY negative, as in the
# original PSIM code; the probe's dev_* numbers are on the same dev file WITHOUT any trigger.
PROBE = None
if os.environ.get("OBSERVE") != "0":
    with isolated_rng():
        PROBE = bias_probe.build_probe_set(dataset_dir, args.dataset_tag, trigger, load_dataset,
                                           tokenize_function, collate_fn, batch_size)


model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
optimizer = AdamW(params=model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs), num_training_steps=(len(train_dataloader) * num_epochs))

model.to(device)
best_dev_acc = -1
best_clean_dev = -1.0   # for --save_rule cleandev
if args.save_epoch is not None and (args.save_every_epoch or args.save_rule != "psim"):
    raise ValueError("--save_epoch cannot be combined with --save_every_epoch or --save_rule cleandev")
if args.save_rule == "cleandev" and (os.environ.get("OBSERVE") == "0" or args.save_every_epoch):
    raise ValueError("--save_rule cleandev needs the per-epoch probe (OBSERVE != 0) and cannot be combined with --save_every_epoch")
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

    saved = 0
    if dev_clean_acc > best_dev_acc:
        best_dev_acc = dev_clean_acc
        saved = 1

        test_acc, asr = get_test_and_asr(model)    # ← 這裡，dropout 自動變成「關」
        print(f"[EPOCH] epoch={epoch} dev={dev_clean_acc:.4f} test clean accuracy={test_acc:.4f} ASR={asr:.4f}")
        if not args.no_save and not args.save_every_epoch and args.save_rule == "psim" and args.save_epoch is None:
            os.makedirs(os.path.dirname(poisoned_model_path), exist_ok=True)
            torch.save(model.state_dict(), poisoned_model_path)
            write_meta(os.path.join(os.path.dirname(poisoned_model_path), "meta.json"),
                    ENV, poisoned_model_path, seed=args.seed, epoch_saved=epoch)
    if PROBE is not None:
        sc = {} if args.score_dir else None
        with observer(model):
            pr = bias_probe.probe_all(model, PROBE, device, scores=sc)
        print(f"[PROBE] epoch={epoch} saved={saved} trig_dev={dev_clean_acc:.4f} {bias_probe.fmt(pr)}")
        if sc is not None:      # writing files moves no RNG stream
            bias_probe.save_scores(os.path.join(args.score_dir, f"ep{epoch:02d}.npz"), sc)
        if args.save_rule == "cleandev" and pr["dev_ca"] > best_clean_dev:
            best_clean_dev = pr["dev_ca"]
            print(f"[SAVE] rule=cleandev epoch={epoch} clean_dev={pr['dev_ca']:.4f}")
            if not args.no_save:
                os.makedirs(os.path.dirname(poisoned_model_path), exist_ok=True)
                torch.save(model.state_dict(), poisoned_model_path)
                write_meta(os.path.join(os.path.dirname(poisoned_model_path), "meta.json"),
                           ENV, poisoned_model_path, seed=args.seed, epoch_saved=epoch, save_rule="cleandev",
                           poison_n=n)
    if args.save_epoch == epoch and not args.no_save:   # read-only w.r.t. training: saving moves no RNG stream
        os.makedirs(os.path.dirname(poisoned_model_path), exist_ok=True)
        torch.save(model.state_dict(), poisoned_model_path)
        write_meta(os.path.join(os.path.dirname(poisoned_model_path), "meta.json"),
                   ENV, poisoned_model_path, seed=args.seed, epoch_saved=epoch, save_rule=f"epoch{epoch}", poison_n=n)
        print(f"[SAVE] rule=epoch{epoch} epoch={epoch}")
    if args.save_every_epoch and not args.no_save:      # read-only w.r.t. training: saving moves no RNG stream
        ep_dir = os.path.join(os.path.dirname(poisoned_model_path), f"ep{epoch:02d}")
        os.makedirs(ep_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ep_dir, "pytorch_model.bin"))
        write_meta(os.path.join(ep_dir, "meta.json"), ENV, os.path.join(ep_dir, "pytorch_model.bin"), seed=args.seed, epoch_saved=epoch)
        if saved:
            with open(os.path.join(os.path.dirname(poisoned_model_path), "psim_pick.txt"), "w") as f:
                f.write(f"{epoch}\n")