# variant_finetune.py
import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # 必須在 import torch / 建立任何 CUDA context 之前設定
import random
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup #removed ", set_seed"
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from attack_utils import insert_trigger, build_poisoned_test_dataloader, compute_asr
import torch.nn as nn
from repro_utils import (assert_env, assert_gpu, isolated_rng, rng_fp, log_env, write_meta, PROTOCOL, observer)
import attack_utils, repro_utils

def parse_args():
    parser = argparse.ArgumentParser()
    # ===== 資料集 / 模型 / 攻擊設定 =====
    parser.add_argument("--attack_tag", type=str, default="badnet", choices=["badnet", "insent"])           # trigger = "mn" for BadNet or "I watched this 3D movie" for InSent
    parser.add_argument("--model_tag", type=str, default="roberta", choices=["bert", "roberta", "llama"])   # bert / roberta / llama
    parser.add_argument("--dataset_tag", type=str, default="sst-2", choices=["sst-2", "cr", "cola"])        # (pretrained,finetuning): (IMDB, SST-2), (MR, CR) or (SST-2, COLA)

    # ===== 機制開關 =====
    parser.add_argument("--use_dora", action="store_true")                  # Toggle True/False depending on the run
    parser.add_argument("--use_pretrained_dropout", action="store_true")    # 機制 1 開關：True 跑 RoRA 版本，False 跑純 baseline（i.e., 沒加 dropout）
    parser.add_argument("--use_orthogonal_penalty", action="store_true")    # 機制 2 開關：True 跑正交懲罰
    parser.add_argument("--use_spectral_rescaling", action="store_true")    # 機制 3，還沒寫

    # ===== 會影響結果、要進檔名的超參數 =====
    parser.add_argument("--lr", type=float, default=2e-4)           # learning rate（grid: {2e-5, 2e-4, 2e-3}）
    parser.add_argument("--cl_dropout_p", type=float, default=0.1)  # 機制 1 的 dropout rate（只有 args.use_pretrained_dropout=True 時才有意義）（grid: {0.05,0.1,0.15,0.2,0.3}）
    parser.add_argument("--cl_form", type=str, default="coord", choices=["coord", "svd"],
                        help="Cl form (cl_utils.py): coord = dropout on every output coordinate of W_pre x + b (RoRA main text, default, unchanged); "
                             "svd = dropout on the components of W_pre x along W_pre's own left singular directions (the k kept ones for a W_pre,K)")
    parser.add_argument("--tr_lambda", type=float, default=10)      # 機制 2 的懲罰強度（只有 args.use_orthogonal_penalty=True 時才有意義）（grid: {1,5,10,15,20}）
    parser.add_argument("--svd_k", type=int, default=32)            # 機制 2 的 Eq.10 的截斷秩，借用 Figure 2 caption 的數字（你自己的實作選擇，論文未明確指定給這個式子）
    parser.add_argument("--tr_layers", type=str, default="all", choices=["all", "top3"],
                        help="which q/v modules the Tr penalty covers: all 48, or top3 = the 6 modules of the last 3 layers (the same ones Pt rescales). "
                             "The penalty is the MEAN over the covered modules, so top3 at the same lambda weighs each module 8x more than all.")
    parser.add_argument("--weight_decay", type=float, default=0.01) # per paper Appendix A.1, fixed to 0.01 for all experiments                

    # ===== other tunable setup for experiments =====
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=None)   # None -> paper's 20
    parser.add_argument("--lr_schedule", choices=["linear", "constant"], default="linear",
                        help="linear (default, unchanged): 6%% warmup then linear decay to 0 over num_epochs. constant: the same warmup as a 20-epoch run "
                             "(0.06 x 20 epochs of steps, whatever num_epochs is), then the learning rate stays at --lr; a long constant run then contains every shorter constant run as its first epochs")
    
    parser.add_argument("--no_save", action="store_true", help="Skip saving checkpoints; useful for quick sanity checks")
    # 多 seed pretrain 訓練用的 custom input path
    parser.add_argument("--pz_path", type=str, default=None, help="override Patient Zero pytorch_model.bin")
    # k-finder 設計（2026-10-19）：微調前把 48 個 query/value 的 W_pre 換成最佳 rank-k 近似 W_pre,k，之後 LoRA、Cl、Tr、Pt 全部以 W_pre,k 為準。
    # None = 不動（與之前所有 v2 run 完全相同）
    parser.add_argument("--wpre_rank", type=int, default=None, help="replace W_pre of the 48 query/value modules by its best rank-k approximation before LoRA (one k for all)")
    parser.add_argument("--wpre_kjson", type=str, default=None, help="per-module k written by kfinder.py for this PZ (instead of --wpre_rank)")
    parser.add_argument("--wpre_file", type=str, default=None, help="the .json written by make_wpre.py for this PZ: replace the 48 query/value W_pre by that saved W_pre,K "
                        "(e.g. whiten truncation) before LoRA; refuses a file made from another PZ (path and weight hashes are checked)")
    parser.add_argument("--svd_k_auto", action="store_true", help="Tr: each module's svd_k = the rank of its W_pre,k (needs --wpre_rank, --wpre_kjson or --wpre_file)")

    # ===== read-only per-module observer (obs_log.py); off unless --obs_log is given =====
    parser.add_argument("--obs_log", type=str, default=None, help="directory for per-module/per-step logs (written to <dir>/<experiment_name>/)")
    parser.add_argument("--obs_every", type=int, default=20, help="log every N training steps (plus the last step of each epoch)")
    parser.add_argument("--obs_ks", type=str, default="8,32,128", help="comma-separated k values at which energy shares are measured (the run's own --svd_k is added)")
    parser.add_argument("--obs_save_every", type=int, default=0, help="with --obs_log: also save the adapter at the end of every N-th epoch (and the last) to <adapter dir>/epNN/; 0 = off. About 8 MB each")
    return parser.parse_args()

args = parse_args()

if args.cl_form != "coord" and not args.use_pretrained_dropout:
    raise ValueError("--cl_form only matters with --use_pretrained_dropout")
if args.cl_form == "svd":
    import cl_utils

if args.use_spectral_rescaling:
    if args.no_save:
        raise ValueError("Mechanism 3 requires a saved checkpoint; cannot use --no_save with --use_spectral_rescaling")

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
# ===== fixed setup for experiments =====
batch_size = 32     # per paper Appendix A.1, fixed to 32 for all experiments
device = "cuda"
#weight_decay = 0.01 # per paper Appendix A.1, fixed to 0.01 for all experiments

# ===== 資料集 / 模型 / 攻擊設定（目前只有 RoBERTa+BadNet/Insent+SST-2 是真正能跑的組合，
#       其他值只是佔位，真的要換 model 時，下面對應的程式碼也要跟著改，不是只改這裡） =====

# find the pretraining dataset
pretrain_dataset = {"sst-2": "imdb", "cr": "mr", "cola": "sst-2"}.get(args.dataset_tag)

# set num_epochs and r as paper Appendix A.1
if args.model_tag == "bert":
    model_name_or_path = "bert-large-uncased"    # not sure but Claude said: 當論文只寫「BERT-large」沒有進一步說明時，uncased 版本是社群裡更常見的預設
    poisoned_model_path = f"./poisoned_bert_large_{pretrain_dataset}_{args.attack_tag}/pytorch_model.bin"   # ← 沒有 args.attack_tag 的分支
    num_epochs_default = 20
    r_default_for_lora = 8
    target_modules = ["query", "value"]
elif args.model_tag == "roberta":
    model_name_or_path = "roberta-large"
    poisoned_model_path = f"./poisoned_roberta_large_{pretrain_dataset}_{args.attack_tag}/pytorch_model.bin"   # ← 沒有 args.attack_tag 的分支
    num_epochs_default = 20
    r_default_for_lora = 8
    target_modules = ["query", "value"]
elif args.model_tag == "llama":
    model_name_or_path = "huggyllama/llama-7b"    # not sure, check for me!
    poisoned_model_path = f"./poisoned_llama_7b_{pretrain_dataset}_{args.attack_tag}/pytorch_model.bin"   # ← 沒有 args.attack_tag 的分支
    num_epochs_default = 5
    r_default_for_lora = 16   
    target_modules = ["q_proj", "v_proj"]         # LLaMA 的層命名跟 BERT/RoBERTa 不同（尚沒有實際查證過 huggyllama/llama-7b 這個 checkpoint 載入後，attention 層的確切命名是不是就是 q_proj/v_proj）
else:
    raise NotImplementedError(f"args.model_tag='{args.model_tag}' not supported. Choose from: bert, roberta, llama.")

pz_tag = ""
if args.pz_path:
    poisoned_model_path = args.pz_path
    pz_tag = "_pz-" + os.path.basename(os.path.dirname(args.pz_path))   # 例如 _pz-poisoned_roberta_large_badnet

dataset_dir = os.path.join('./data', args.dataset_tag)

# --wpre_file (2026-10-28): W_pre,K made by make_wpre.py; read its json now so its files are hashed into [ENV] and meta.json
WMETA, WPRE_FILES = None, {}
if args.wpre_file:
    import wpre_utils
    WMETA = wpre_utils.load_meta(args.wpre_file)
    WPRE_FILES = {"wpre_json": args.wpre_file, "wpre_pt": os.path.join(os.path.dirname(args.wpre_file), WMETA["weights_file"]),
                  "wpre_utils": wpre_utils.__file__}

ENV = log_env(__file__, {
    "pz": poisoned_model_path,
    "attack_utils": attack_utils.__file__,
    "repro_utils": repro_utils.__file__,
    "train": f"{dataset_dir}/train.json",
    "dev": f"{dataset_dir}/dev.json",
    "test": f"{dataset_dir}/test.json",
    **WPRE_FILES,
})

if args.attack_tag == "badnet":
    trigger = "mn"
elif args.attack_tag == "insent":
    trigger = "I watched this 3D movie"
else:
    raise NotImplementedError(f"args.attack_tag='{args.attack_tag}' not supported. Choose from: badnet, insent.")

num_epochs = args.num_epochs or num_epochs_default # TUNABLE for custom experiments

# per paper Appendix A.1: "For PiSSA, DoRA, and OLoRA, we use the default hyperparameters provided by the PEFT library"
# peft 預設值來自：python -c "from peft import LoraConfig; import dataclasses; [print(f.name, '=', f.default) for f in dataclasses.fields(LoraConfig) if f.name in ['r', 'lora_alpha', 'lora_dropout']]"
if args.use_dora:
    r = 8                       # peft LoraConfig 的預設值，用上面那行指令查證過
    lora_alpha = 8
    lora_dropout = 0.0
else:
    r = r_default_for_lora      # TUNABLE for paper Appendix A.3 (for lora only?), which sweeps over r
    lora_alpha = 16             # TUNABLE for paper Appendix A.3 (for lora only?), which sweeps over lora_alpha
    lora_dropout = 0.1          # per paper Appendix A.1, fixed to 0.1 for all lora experiments
# Define the adaptation variant (e.g., DoRA)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    use_dora=args.use_dora,
)

# ===== 自動組名 =====
if args.wpre_rank and args.wpre_kjson:
    raise ValueError("use either --wpre_rank or --wpre_kjson, not both")
if args.wpre_file and (args.wpre_rank or args.wpre_kjson):
    raise ValueError("use --wpre_file alone, not with --wpre_rank or --wpre_kjson")
if args.svd_k_auto and not (args.wpre_rank or args.wpre_kjson or args.wpre_file):
    raise ValueError("--svd_k_auto needs --wpre_rank, --wpre_kjson or --wpre_file")
KJSON = None
if args.wpre_kjson:
    import json
    KJSON = json.load(open(args.wpre_kjson))
mechanism_tags = []
if args.wpre_rank:
    mechanism_tags.append(f'wk{args.wpre_rank}')        # 例如 wk896：base 已換成 W_pre,896（重新評量時必須用同一個 k 截斷 base）
if KJSON:
    mechanism_tags.append(f'wkf{KJSON["tag"]}')         # 例如 wkfgrp6：base 已換成 kfinder.py 找到的每模組 W_pre,k（重新評量時需同一個 json）
if WMETA:
    mechanism_tags.append(f'wk{WMETA["tag"]}')          # 例如 wkwh0.9：base 已換成 make_wpre.py 存的 W_pre,K（換基底、r* = 0.9；重新評量時需同一個檔）
if args.use_pretrained_dropout:
    mechanism_tags.append(('clsvd' if args.cl_form == 'svd' else 'cl') + f'{args.cl_dropout_p}')      # 例如 cl0.1（逐座標，名字和以前一樣）、clsvd0.1（SVD 方向）
if args.use_orthogonal_penalty:
    mechanism_tags.append(f'tr{args.tr_lambda:g}_k' + ('auto' if args.svd_k_auto else f'{args.svd_k}')
                          + ('_top3' if args.tr_layers == 'top3' else ''))   # 例如 tr10_k32、tr10_kauto、tr10_k8_top3
if args.use_spectral_rescaling:
    mechanism_tags.append('pt')

method_tag = ('dora' if args.use_dora else 'lora')
if mechanism_tags:
    method_tag += '_' + '_'.join(mechanism_tags)

lr_tag = f'lr{args.lr:.0e}'   # 2e-4 → 'lr2e-04'
if args.lr_schedule != "linear":
    lr_tag += '_' + args.lr_schedule     # default (linear) keeps the old experiment names
r_tag = f'r{r}_a{lora_alpha}'

experiment_name = f"{args.model_tag}_{args.attack_tag}_{args.dataset_tag}_{method_tag}_{lr_tag}_{r_tag}_seed{args.seed}_ep{num_epochs}_wd{args.weight_decay}{pz_tag}"
output_dir = os.path.join(args.obs_log, 'adapters', experiment_name) if args.obs_log else f'adapters/{experiment_name}'   # with --obs_log nothing under adapters/ is overwritten
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
    collate_fn,
    trigger,
    batch_size=batch_size   # here we match the batch_size=32 in poisoned_pretrain.py in original PSIM repo.
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)

# INJECT THE ADAPTER (FOR PEFT)

# Inject the poisoned weights you trained earlier
# Sometimes when loading weights into a fresh AutoModelForSequenceClassification architecture, PyTorch panics if non-essential keys (like unused pooler layers) don't match perfectly. To prevent the script from crashing during the injection step
poisoned_sd = torch.load(poisoned_model_path)
missing, unexpected = model.load_state_dict(poisoned_sd, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# [k-finder 設計] 在包成 LoRA 之前，把 W_pre 換成 W_pre,k；之後 Cl 的 hook（base_layer 的輸出）、Tr 的 SVD（base_layer.weight）、
# Pt 的 σmax（get_base_model() 裡的 base_layer.weight）、每個 epoch 的 dev/test/ASR 評量，用的都是這個截斷後的矩陣。
# 截斷方式與 eval_probes.py 相同：U[:, :k] diag(S[:k]) Vh[:k]，在 float32 上算再轉回原 dtype。
KMAP = {}   # {"roberta.encoder.layer.0.attention.self.query": k, ...}；Tr 的 --svd_k_auto 也用它
if WMETA:
    # --wpre_file：直接換成 make_wpre.py 存好的 W_pre,K（例如換基底截斷，不是 σ 截斷）。wpre_utils.apply 檢查：PZ 路徑相同、
    # 48 個原始權重的 hash 和做檔時相同、每個存好的權重 hash 相同；之後 Cl、Tr、Pt、評量全以它為準，和上面 σ 截斷的情形一樣。
    KMAP, _ = wpre_utils.apply_file(model, args.wpre_file, poisoned_model_path)
    _kv = np.array(list(KMAP.values()))
    print(f"[W_pre,K] replaced 48 query/value W_pre from {args.wpre_file} ({WMETA['method']}, r={WMETA['r']:g}, {WMETA['how']}); "
          f"k mean {_kv.mean():.1f} min {_kv.min()} max {_kv.max()}; PZ and weight hashes checked")
elif args.wpre_rank or KJSON:
    if KJSON:
        assert os.path.realpath(KJSON["pz"]) == os.path.realpath(poisoned_model_path), \
            f"k json was made for {KJSON['pz']}, not for {poisoned_model_path}"
    n_trunc, removed = 0, []
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and name.endswith(("attention.self.query", "attention.self.value")):
                k_ = int(KJSON["per_module_k"][name]) if KJSON else args.wpre_rank
                KMAP[name] = k_
                n_trunc += 1
                if k_ >= mod.weight.shape[0]:
                    removed.append(0.0); continue
                W = mod.weight.data
                U_, S_, Vh_ = torch.linalg.svd(W.float().to(device), full_matrices=False)   # 與 kfinder.py / eval_probes.py 一樣在 GPU 上算
                mod.weight.data.copy_(((U_[:, :k_] * S_[:k_]) @ Vh_[:k_]).to(W.dtype).cpu())
                removed.append((S_[k_:].pow(2).sum() / S_.pow(2).sum()).item())
    assert n_trunc == 48, f"expected 48 query/value modules, truncated {n_trunc}"
    ks_ = sorted(set(KMAP.values()))
    print(f"[W_pre,k] truncated {n_trunc} query/value W_pre; ranks used {ks_}; "
          f"removed sigma^2 share: mean={np.mean(removed):.4f} max={np.max(removed):.4f}")
    if KJSON:
        print(f"[W_pre,k] k per group (from {args.wpre_kjson}): {KJSON['k_final']}")

# Wrap the model (freezes base, adds trainable adapters)
model = get_peft_model(model, peft_config)   # ← 先「包成 LoRA」，之後才有 .base_layer，在這之前 model 還是原始的 AutoModelForSequenceClassification
model.print_trainable_parameters() # Sanity check: should show < 1% trainable

optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # per paper Appendix A.1, weight_decay fixed to 0.01 for all experiments
# Instantiate scheduler
if args.lr_schedule == "linear":
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs), num_training_steps=(len(train_dataloader) * num_epochs))
else:   # constant: warmup length fixed at the paper's 20 epochs, so it does not grow with num_epochs
    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.06 * (len(train_dataloader) * 20))

import time
start_time = time.time()

## [MECHANISM 1] dropout hook for the pretrained weights (W_pre) in the query and value linear layers

dropout_hook_call_count = [0]  # 用 list 包起來方便在閉包內修改

CL_U = {}   # --cl_form svd：每個 base_layer 的左奇異方向（W_pre,K 時是它留下的 k 個；和 Tr 的 U 用同一個 float32 SVD）
def dropout_hook(module, input, output):
    dropout_hook_call_count[0] += 1
    if args.cl_form == "coord":   # 和以前完全相同的呼叫
        return torch.nn.functional.dropout(output, p=args.cl_dropout_p, training=module.training)   # 跟論文 p=0.1 一致；攔截輸出，套上 dropout 再放行
    return cl_utils.svd_dropout(output, module.bias, CL_U[module], args.cl_dropout_p, training=module.training)

hook_handles = []
if args.use_pretrained_dropout:
    for name, module in model.named_modules():
        # peft 包裝後的 LoRA linear layer 有 .base_layer 屬性，指向凍結的 W_pre
        if hasattr(module, "base_layer") and any(t in name for t in target_modules):
            # 假設 some_linear_layer 是 query 或 value 那個線性層
            handle = module.base_layer.register_forward_hook(dropout_hook)
            hook_handles.append(handle)
            if args.cl_form == "svd":
                k_cl = KMAP.get(name.split("base_model.model.")[-1])   # 有 W_pre,K 時用它的 k；完整 W_pre 時是 None = 全部 1024 個方向
                CL_U[module.base_layer] = cl_utils.left_basis(module.base_layer.weight.data, k_cl).to(device)

    print(f"Hooked {len(hook_handles)} base_layer modules with dropout")  # 應該印出 2 × 層數（query + value 各一）
    if args.cl_form == "svd":
        _ks = sorted(set(U.shape[1] for U in CL_U.values()))
        print(f"[Cl svd] dropout on the left singular directions of {len(CL_U)} base_layer weights; directions per module: min {_ks[0]} max {_ks[-1]}")
else:
    print("Mechanism 1 (pretrained dropout) is OFF — running baseline")
## 
## [MECHANISM 2] orthogonal penalty (truncated SVD) for the pretrained weights (W_pre) in the query and value linear layers
_all_qv = [n for n, m in model.named_modules() if hasattr(m, "base_layer") and any(t in n for t in target_modules)]
# --tr_layers top3: the last 6 q/v modules in named_modules() order = layers 21-23 of RoBERTa-large, the same selection Pt uses ([-6:])
tr_names = set(_all_qv if args.tr_layers == "all" else _all_qv[-6:])
if args.use_orthogonal_penalty:
    print(f"[Tr layers] {args.tr_layers}: penalty covers {len(tr_names)} of {len(_all_qv)} q/v modules" + ("" if args.tr_layers == "all" else f", {sorted(tr_names)}"))
    # 只做一次：對每個 target layer 的 W_pre 做 SVD（W_{pre} = U \Sigma V^\top），取得 U, V
    U_dict = {}
    V_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "base_layer") and any(t in name for t in target_modules) and name in tr_names:
            # module is object like model.roberta.encoder.layer[0].attention.self.query with name "roberta.encoder.layer.0.attention.self.query", which is a LoRA linear layer
            # W_pre（用來算 U_dict/V_dict 的）是用 .weight.data 抓的，.data 會脫離 autograd 計算圖，這樣 SVD 那段不會被誤算進反向傳播、也不會意外讓 W_pre（本該凍結）產生梯度
            W_pre = module.base_layer.weight.data
            U_layer, S_full, Vh_layer = torch.linalg.svd(W_pre, full_matrices=False)
            print("[DEBUG] W_pre 是否接近滿秩:", (S_full > 1e-6).sum().item(), "/ 1024")  # 如果接近1024，代表滿秩，L2退化猜測成立
            # torch.linalg.svd 回傳的奇異值已經由大到小排序，直接取前 k 欄/列即可，否則將因為 W_{pre} 極可能滿秩，而使 \Omega退化成普通 L2
            # --svd_k_auto：這個模組的 k = 它的 W_pre,k 的秩（kfinder 或 --wpre_rank 給的），懲罰涵蓋 W_pre,k 的整個行/列空間
            k_tr = KMAP[name.split("base_model.model.")[-1]] if args.svd_k_auto else args.svd_k
            U_dict[name] = U_layer[:, :k_tr].to(device)      # d × k
            V_dict[name] = Vh_layer[:k_tr, :].T.to(device)   # d × k（Vh 是 V^T，所以前 k 列 .T 轉置回來 V）
    print(f"Computed truncated SVD (k={'auto: ' + str(sorted(set(U.shape[1] for U in U_dict.values()))) if args.svd_k_auto else args.svd_k}) for {len(U_dict)} base_layer modules")  # 應該也是 48
else:
    print("Mechanism 2 (orthogonal penalty) is OFF — running baseline")
##

model.to(device)
best_dev_acc = -1
pre_pt_best_test_acc = None
pre_pt_best_asr = None
best_epoch = -1

last_step = len(train_dataloader) - 1
def should_log(epoch, step):
    if epoch == 0:
        return step in (0, 20, 100, last_step)
    return step == 0 or step == last_step

test_hist, asr_hist = [], []

obs = None
if args.obs_log:
    from obs_log import Observer
    _obs_ks = [int(x) for x in args.obs_ks.split(",") if x] + [args.svd_k]
    obs = Observer(model, target_modules, os.path.join(args.obs_log, experiment_name), _obs_ks, every=args.obs_every,
                   meta=dict(experiment_name=experiment_name, seed=args.seed, lr=args.lr, tr=args.use_orthogonal_penalty,
                             tr_lambda=args.tr_lambda, svd_k=args.svd_k, tr_layers=args.tr_layers, num_epochs=num_epochs, lr_schedule=args.lr_schedule,
                             cl=args.use_pretrained_dropout, pz=poisoned_model_path))

for epoch in range(num_epochs):
    print(f"[RNG] epoch={epoch} fp={rng_fp()}")

    model.train()   # ← 這裡，dropout 是「開」的
    for step, batch in enumerate(tqdm(train_dataloader)):
        log_now = should_log(epoch, step)

        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        if obs: obs.accumulate_task(loss.detach())

        if args.use_orthogonal_penalty:
            task_loss = loss
            # we have computed U, V of W_{pre} = U \Sigma V^\top
            # computing \Omega(A,B) = \lVert U^\top B \rVert_F^2 + \lVert A V \rVert_F^2
            penalty = 0.0
            num_penalized_layers = 0
            ratio_A_dbg = []
            ratio_B_dbg = []
            for name, module in model.named_modules():
                if hasattr(module, "base_layer") and any(t in name for t in target_modules) and name in tr_names:
                    # 不同於 W_{pre}，A, B 是真正在被訓練、requires_grad=True 的參數，梯度能正常透過這個懲罰項傳回去更新它們
                    A = module.lora_A["default"].weight   # shape: (r, d)，對應論文的 A ∈ R^{r×d}
                    B = module.lora_B["default"].weight   # shape: (d, r)，對應論文的 B ∈ R^{d×r}
                    U, V = U_dict[name], V_dict[name]
                    # torch.norm(..., p='fro')**2 is very dangerous: there will be NaN when computing gradient. do not trust AI's sloppy advice.
                    omega_B = (U.T @ B).pow(2).sum() * (U.shape[0] / U.shape[1])   # out_features / k
                    omega_A = (A @ V).pow(2).sum() * (V.shape[0] / V.shape[1])   # in_features  / k
                    penalty += omega_A + omega_B
                    num_penalized_layers += 1

                    if log_now:
                        ratio_A_dbg.append((omega_A / A.norm().pow(2)).item())
                        if not (epoch == 0 and step == 0):          # 只有 B=0 的那一步要跳過
                            ratio_B_dbg.append((omega_B / B.norm().pow(2)).item())
                    
            penalty = penalty / num_penalized_layers   # 改成平均，不是加總
            if obs: obs.accumulate_pen(penalty.detach())

            if log_now:
                result = f"[DEBUG_ep{epoch}_stp{step}] task_loss={task_loss.item():.4f}, penalty(mean,scaled)={penalty.item():.4f}, lambda*penalty={(args.tr_lambda*penalty).item():.4f}, omega_A/||A||^2 mean={sum(ratio_A_dbg)/len(ratio_A_dbg):.2e}"
                if ratio_B_dbg:                                      # 不再用 epoch != 0 判斷
                    result += f", omega_B/||B||^2 mean={sum(ratio_B_dbg)/len(ratio_B_dbg):.2e}"
                else:
                    result += " (expect ~1)"
                tqdm.write(result)

            loss = task_loss + args.tr_lambda * penalty   # 迴圈跑完才加一次
        
        loss.backward()
        if obs and obs.due(step, last_step): obs.snapshot(epoch, step)   # read-only: parameters before this step's update, gradients as the optimizer sees them
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    dev_clean_acc = evaluate_accuracy(model, eval_dataloader)   
    print(f"epoch {epoch} ")
    print('dev clean acc: %.4f'% dev_clean_acc)
    if epoch == 0 and args.use_pretrained_dropout:
        expected = len(hook_handles) * (len(train_dataloader) + len(eval_dataloader))
        print(f"[CHECK] dropout_hook fired {dropout_hook_call_count[0]} times in epoch 0, expected {expected}")
        assert dropout_hook_call_count[0] == expected, "Mechanism 1 hook count mismatch"
    
    test_acc, asr = get_test_and_asr(model)    # ← 這裡，dropout 自動變成「關」
    print(f"[EPOCH] epoch={epoch} dev={dev_clean_acc:.4f} test clean accuracy={test_acc:.4f} ASR={asr:.4f}")
    test_hist.append(test_acc); asr_hist.append(asr)
    if obs: obs.epoch_end(epoch, dev_clean_acc, test_acc, asr)   # parameters = the state dev/test/ASR above were measured on
    if obs and args.obs_save_every and not args.no_save and (epoch % args.obs_save_every == 0 or epoch == num_epochs - 1):
        model.save_pretrained(os.path.join(output_dir, f"ep{epoch:02d}"))   # read-only w.r.t. training; lets any per-epoch analysis be redone offline

    if dev_clean_acc > best_dev_acc:
        best_dev_acc = dev_clean_acc
        best_epoch = epoch
        pre_pt_best_test_acc, pre_pt_best_asr = test_acc, asr      # 只在新最佳時記錄
        if not args.no_save:
            # Create a new directory for the specific variant
            os.makedirs(output_dir, exist_ok=True)
            # Use save_pretrained to only save the adapter matrices
            model.save_pretrained(output_dir)   # note that the model saved will always be pre-pt
            write_meta(os.path.join(output_dir, "meta.json"), 
                       ENV, os.path.join(output_dir, "adapter_model.safetensors"), seed=args.seed, epoch_saved=epoch, experiment_name=experiment_name)
    
summary = f"SUMMARY | protocol={PROTOCOL} | {experiment_name} | best_epoch={best_epoch} | best_dev_acc={best_dev_acc:.4f} | pre-pt_test_acc={pre_pt_best_test_acc:.4f} | pre-pt_ASR={pre_pt_best_asr:.4f}"

# [最後一個 epoch，2026-10-29] 除了最佳 dev 的 checkpoint，也存最後一個 epoch 的 adapter（<output_dir>/last/），
# 並在它上面做同樣的 Pt、量 test 和 ASR（類型 (C)），讓兩種存檔規則都能比。訓練完才做、評量用 observer（不動任何 RNG），
# Pt 是就地改 scaling、量完放回原值，所以下面最佳 dev 那一段和以前完全相同。
last_summary = ""
if not args.no_save:
    last_dir = os.path.join(output_dir, "last")
    model.save_pretrained(last_dir)
    write_meta(os.path.join(last_dir, "meta.json"), ENV, os.path.join(last_dir, "adapter_model.safetensors"),
               seed=args.seed, epoch_saved=num_epochs - 1, experiment_name=experiment_name)
    last_summary = f" | last_epoch={num_epochs - 1} | last_dev_acc={dev_clean_acc:.4f} | last_test_acc={test_hist[-1]:.4f} | last_ASR={asr_hist[-1]:.4f}"
    if args.use_spectral_rescaling:
        _qv = [n for n, m_ in model.named_modules() if hasattr(m_, "base_layer") and any(t in n for t in target_modules)][-6:]   # 和下面同樣的 6 個
        _old = {}
        for n, m_ in model.named_modules():
            if n in _qv:
                _s_pre = torch.linalg.svdvals(m_.base_layer.weight.data)[0]
                _s_dw = torch.linalg.svdvals(m_.lora_B["default"].weight.data @ m_.lora_A["default"].weight.data)[0]
                _old[n] = m_.scaling["default"]
                m_.scaling["default"] = (_s_pre / _s_dw).item()
        last_pt_acc, last_pt_asr = get_test_and_asr(model)
        for n, m_ in model.named_modules():
            if n in _old:
                m_.scaling["default"] = _old[n]
        print(f"[Mechanism 3, last epoch] Rescaled {len(_old)} layers; test clean accuracy={last_pt_acc:.4f} ASR={last_pt_asr:.4f} (scalings put back)")
        last_summary += f" | last_pt_test_acc={last_pt_acc:.4f} | last_pt_ASR={last_pt_asr:.4f}"

# 訓練迴圈全部跑完之後（for epoch ... 迴圈結束，best checkpoint 已經存好）
## [MECHANISM 3] spectral rescaling for the top three layers of \Delta W: s=\sigma_{max}(W_{pre})/\sigma_{max}(\Delta W), i.e., module.scaling["default"] = \sigma_{max}(W_{pre})/\sigma_{max}(\Delta W)
pt_scales = {}
if args.use_spectral_rescaling:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model.get_base_model(), output_dir)
    model.to(device)

    # 只對「最後三層」做（目前是我們的猜測，不是論文確認過的答案）
    all_layer_names = [name for name, module in model.named_modules()
                        if hasattr(module, "base_layer") and any(t in name for t in target_modules)]
    # all_layer_names 目前的順序是 named_modules() 走訪順序，通常就是層數由淺到深，
    # 取最後三個對應到的名字，等於「最靠近輸出的三層」
    top_layer_names = all_layer_names[-6:]

    for name, module in model.named_modules():
        if name in top_layer_names:
            # 假設 some_linear_layer 是 query 或 value 那個線性層
            # 已找到 W_pre 的 spectral norm（最大奇異值)
            W_pre = module.base_layer.weight.data
            sigma_max_pre = torch.linalg.svdvals(W_pre)[0]   # svdvals 只算奇異值,比完整 svd 省算力

            A = module.lora_A["default"].weight.data
            B = module.lora_B["default"].weight.data
            delta_W = B @ A
            sigma_max_delta = torch.linalg.svdvals(delta_W)[0]

            new_s = (sigma_max_pre / sigma_max_delta).item()
            pt_scales[name] = dict(old_scaling=float(module.scaling["default"]), new_scaling=new_s, sigma_max_pre=sigma_max_pre.item(), sigma_max_delta_unscaled=sigma_max_delta.item())   # for obs summary.json
            module.scaling["default"] = new_s
            print(f"[Mechanism 3] {name}: new scaling s = {new_s:.4f}")

    print(f"[Mechanism 3] Rescaled {len(top_layer_names)} layers")
    post_pt_best_test_acc, post_pt_best_asr = get_test_and_asr(model)    # ← 這裡，dropout 自動變成「關」
    print(f"[Mechanism 3] test clean accuracy={post_pt_best_test_acc:.4f} ASR={post_pt_best_asr:.4f}")
else:
    print("Mechanism 3 (spectral rescaling) is OFF — running baseline")
##

duration_min = (time.time() - start_time) / 60

if args.use_spectral_rescaling:
    summary += f" | pt_test_acc={post_pt_best_test_acc:.4f} | pt_ASR={post_pt_best_asr:.4f}"
summary += last_summary
summary += f" | duration_min={duration_min:.1f} | asr_mean_last5={sum(asr_hist[-5:])/len(asr_hist[-5:]):.4f} | gpu={torch.cuda.get_device_name(0)} | observe={'OFF' if os.environ.get('OBSERVE')=='0' else 'on'}"
print(summary)
if obs:   # after everything: the SUMMARY line and Pt numbers also go into obs/<run>/summary.json, so the run folder is self-contained
    obs.write_summary(dict(summary_line=summary, protocol=PROTOCOL, experiment_name=experiment_name, best_epoch=best_epoch, best_dev_acc=best_dev_acc,
                           pre_pt_test_acc=pre_pt_best_test_acc, pre_pt_asr=pre_pt_best_asr,
                           pt_test_acc=(post_pt_best_test_acc if args.use_spectral_rescaling else None), pt_asr=(post_pt_best_asr if args.use_spectral_rescaling else None),
                           asr_mean_last5=sum(asr_hist[-5:])/len(asr_hist[-5:]), duration_min=duration_min, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                           pt_scales=pt_scales, args=vars(args)))