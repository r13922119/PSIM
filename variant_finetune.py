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
from attack_utils import insert_trigger, build_poisoned_test_dataloader, compute_asr
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser()
    # ===== 資料集 / 模型 / 攻擊設定 =====
    parser.add_argument("--attack_tag", type=str, default="badnet", choices=["badnet", "insent"])           # trigger = "mn" for BadNet or "I watched this 3D movie" for InSent
    parser.add_argument("--model_tag", type=str, default="roberta", choices=["bert", "roberta", "llama"])   # bert / roberta / llama
    parser.add_argument("--dataset_tag", type=str, default="sst-2")                                         # sst-2 / cr / cola (the data folder also contains imdb and mr)

    # ===== 機制開關 =====
    parser.add_argument("--use_dora", action="store_true")                  # Toggle True/False depending on the run
    parser.add_argument("--use_pretrained_dropout", action="store_true")    # 機制 1 開關：True 跑 RoRA 版本，False 跑純 baseline（i.e., 沒加 dropout）
    parser.add_argument("--use_orthogonal_penalty", action="store_true")    # 機制 2 開關：True 跑正交懲罰
    parser.add_argument("--use_spectral_rescaling", action="store_true")    # 機制 3，還沒寫

    # ===== 會影響結果、要進檔名的超參數 =====
    parser.add_argument("--lr", type=float, default=2e-4)           # learning rate（grid: {2e-5, 2e-4, 2e-3}）
    parser.add_argument("--cl_dropout_p", type=float, default=0.1)  # 機制 1 的 dropout rate（只有 args.use_pretrained_dropout=True 時才有意義）（grid: {0.05,0.1,0.15,0.2,0.3}）
    parser.add_argument("--tr_lambda", type=float, default=10)      # 機制 2 的懲罰強度（只有 args.use_orthogonal_penalty=True 時才有意義）（grid: {1,5,10,15,20}）
    parser.add_argument("--svd_k", type=int, default=32)            # 機制 2 的 Eq.10 的截斷秩，借用 Figure 2 caption 的數字（你自己的實作選擇，論文未明確指定給這個式子）         

    # ===== other tunable setup for experiments =====
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--no_save", action="store_true", help="Skip saving checkpoints; useful for quick sanity checks")
    return parser.parse_args()

args = parse_args()

# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# ===== fixed setup for experiments =====
batch_size = 32     # per paper Appendix A.1, fixed to 32 for all experiments
device = "cuda"
weight_decay = 0.01 # per paper Appendix A.1, fixed to 0.01 for all experiments

# ===== 資料集 / 模型 / 攻擊設定（目前只有 RoBERTa+BadNet/Insent+SST-2 是真正能跑的組合，
#       其他值只是佔位，真的要換 model 時，下面對應的程式碼也要跟著改，不是只改這裡） =====

# set num_epochs and r as paper Appendix A.1
if args.model_tag == "bert":
    model_name_or_path = "bert-large-uncased"    # not sure but Claude said: 當論文只寫「BERT-large」沒有進一步說明時，uncased 版本是社群裡更常見的預設
    poisoned_model_path = f"./poisoned_bert_large_{args.attack_tag}/pytorch_model.bin"   # ← 沒有 args.attack_tag 的分支
    num_epochs_default = 20
    r_default_for_lora = 8
    target_modules = ["query", "value"]
elif args.model_tag == "roberta":
    model_name_or_path = "roberta-large"
    poisoned_model_path = f"./poisoned_roberta_large_{args.attack_tag}/pytorch_model.bin"   # ← 沒有 args.attack_tag 的分支
    num_epochs_default = 20
    r_default_for_lora = 8
    target_modules = ["query", "value"]
elif args.model_tag == "llama":
    model_name_or_path = "huggyllama/llama-7b"    # not sure, check for me!
    poisoned_model_path = f"./poisoned_llama_7b_{args.attack_tag}/pytorch_model.bin"   # ← 沒有 args.attack_tag 的分支
    num_epochs_default = 5
    r_default_for_lora = 16   
    target_modules = ["q_proj", "v_proj"]         # LLaMA 的層命名跟 BERT/RoBERTa 不同（尚沒有實際查證過 huggyllama/llama-7b 這個 checkpoint 載入後，attention 層的確切命名是不是就是 q_proj/v_proj）
else:
    raise NotImplementedError(f"args.model_tag='{args.model_tag}' not supported. Choose from: bert, roberta, llama.")

if args.attack_tag == "badnet":
    trigger = "mn"
elif args.attack_tag == "insent":
    trigger = "I watched this 3D movie"
else:
    raise NotImplementedError(f"args.attack_tag='{args.attack_tag}' not supported. Choose from: badnet, insent.")

dataset_dir = os.path.join('./data', args.dataset_tag)

num_epochs = num_epochs_default # TUNABLE for custom experiments

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
mechanism_tags = []
if args.use_pretrained_dropout:
    mechanism_tags.append(f'cl{args.cl_dropout_p}')      # 例如 cl0.1
if args.use_orthogonal_penalty:
    mechanism_tags.append(f'tr{args.tr_lambda}_k{args.svd_k}')   # 例如 tr10_k32
if args.use_spectral_rescaling:
    mechanism_tags.append('pt')

method_tag = ('dora' if args.use_dora else 'lora')
if mechanism_tags:
    method_tag += '_' + '_'.join(mechanism_tags)

lr_tag = f'lr{args.lr:.0e}'   # 2e-4 → 'lr2e-04'
r_tag = f'r{r}_a{lora_alpha}'

experiment_name = f"{args.model_tag}_{args.attack_tag}_{args.dataset_tag}_{method_tag}_{lr_tag}_{r_tag}_seed{args.seed}_ep{num_epochs}"
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

def report_test_and_asr(model, label=""):
    """跑 test clean acc + ASR，印出來，回傳兩個數字方便之後要用"""
    test_acc = evaluate_accuracy(model, test_dataloader)    # ← 這裡，dropout 自動變成「關」
    print(f'{label}test clean acc: %.4f' % test_acc)
    asr = compute_asr(model, device, poisoned_test_dataloader)
    print(f'{label}ASR: %.4f' % asr)
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
    trigger
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)

# INJECT THE ADAPTER (FOR PEFT)

# Inject the poisoned weights you trained earlier
# Sometimes when loading weights into a fresh AutoModelForSequenceClassification architecture, PyTorch panics if non-essential keys (like unused pooler layers) don't match perfectly. To prevent the script from crashing during the injection step
model.load_state_dict(torch.load(poisoned_model_path), strict=False)
#
poisoned_sd = torch.load(poisoned_model_path)
missing, unexpected = model.load_state_dict(poisoned_sd, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
#

# Wrap the model (freezes base, adds trainable adapters)
model = get_peft_model(model, peft_config)   # ← 先「包成 LoRA」，之後才有 .base_layer，在這之前 model 還是原始的 AutoModelForSequenceClassification
model.print_trainable_parameters() # Sanity check: should show < 1% trainable

optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=weight_decay)  # per paper Appendix A.1, weight_decay fixed to 0.01 for all experiments
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs), num_training_steps=(len(train_dataloader) * num_epochs))

## [MECHANISM 1] dropout hook for the pretrained weights (W_pre) in the query and value linear layers
pretrained_dropout = nn.Dropout(p=args.cl_dropout_p)  # 跟論文 p=0.1 一致

def dropout_hook(module, input, output):
    return pretrained_dropout(output)   # 攔截輸出，套上 dropout 再放行

hook_handles = []
if args.use_pretrained_dropout:
    for name, module in model.named_modules():
        # peft 包裝後的 LoRA linear layer 有 .base_layer 屬性，指向凍結的 W_pre
        if hasattr(module, "base_layer") and any(t in name for t in target_modules):
            # 假設 some_linear_layer 是 query 或 value 那個線性層
            handle = module.base_layer.register_forward_hook(dropout_hook)
            hook_handles.append(handle)

    print(f"Hooked {len(hook_handles)} base_layer modules with dropout")  # 應該印出 2 × 層數（query + value 各一）
else:
    print("Mechanism 1 (pretrained dropout) is OFF — running baseline")
## 
## [MECHANISM 2] orthogonal penalty (truncated SVD) for the pretrained weights (W_pre) in the query and value linear layers
if args.use_orthogonal_penalty:
    # 只做一次：對每個 target layer 的 W_pre 做 SVD（W_{pre} = U \Sigma V^\top），取得 U, V
    U_dict = {}
    V_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "base_layer") and any(t in name for t in target_modules):
            # module is object like model.roberta.encoder.layer[0].attention.self.query with name "roberta.encoder.layer.0.attention.self.query", which is a LoRA linear layer
            # W_pre（用來算 U_dict/V_dict 的）是用 .weight.data 抓的，.data 會脫離 autograd 計算圖，這樣 SVD 那段不會被誤算進反向傳播、也不會意外讓 W_pre（本該凍結）產生梯度
            W_pre = module.base_layer.weight.data
            U_layer, S_full, Vh_layer = torch.linalg.svd(W_pre, full_matrices=False)
            #print("[DEBUG] W_pre 是否接近滿秩:", (S_full > 1e-6).sum().item(), "/ 1024")  # 如果接近1024，代表滿秩，L2退化猜測成立
            # torch.linalg.svd 回傳的奇異值已經由大到小排序，直接取前 k 欄/列即可，否則將因為 W_{pre} 極可能滿秩，而使 \Omega退化成普通 L2
            U_dict[name] = U_layer[:, :args.svd_k].to(device)      # d × k
            V_dict[name] = Vh_layer[:args.svd_k, :].T.to(device)   # d × k（Vh 是 V^T，所以前 k 列 .T 轉置回來 V）
    print(f"Computed truncated SVD (k={args.svd_k}) for {len(U_dict)} base_layer modules")  # 應該也是 48
else:
    print("Mechanism 2 (orthogonal penalty) is OFF — running baseline")
##

model.to(device)
best_dev_acc = -1
for epoch in range(num_epochs):
    model.train()   # ← 這裡，dropout 是「開」的
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss

        if args.use_orthogonal_penalty:
            # we have computed U, V of W_{pre} = U \Sigma V^\top
            # computing \Omega(A,B) = \lVert U^\top B \rVert_F^2 + \lVert A V \rVert_F^2
            penalty = 0.0
            num_penalized_layers = 0
            for name, module in model.named_modules():
                if hasattr(module, "base_layer") and any(t in name for t in target_modules):
                    # 不同於 W_{pre}，A, B 是真正在被訓練、requires_grad=True 的參數，梯度能正常透過這個懲罰項傳回去更新它們
                    A = module.lora_A["default"].weight   # shape: (r, d)，對應論文的 A ∈ R^{r×d}
                    B = module.lora_B["default"].weight   # shape: (d, r)，對應論文的 B ∈ R^{d×r}
                    penalty += torch.norm(U_dict[name].T @ B, p='fro')**2 + torch.norm(A @ V_dict[name], p='fro')**2
                    num_penalized_layers += 1
            #if step == 0 and epoch == 0:   # 只在第一步印一次，不要洗版
            #    print(f"[DEBUG] loss={loss.item():.4f}, penalty={penalty.item():.4f}, args.tr_lambda*penalty={(args.tr_lambda*penalty).item():.4f}")
            # [DEBUG] loss=0.5287, penalty=4.0944, args.tr_lambda*penalty=40.9437
            penalty = penalty / num_penalized_layers   # 改成平均，不是加總
            loss += args.tr_lambda * penalty   # 迴圈跑完才加一次，縮排跟 for 對齊、不在裡面
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    dev_clean_acc = evaluate_accuracy(model, eval_dataloader)   
    print(f"epoch {epoch} ")
    print('dev clean acc: %.4f'% dev_clean_acc)
    
    if dev_clean_acc > best_dev_acc:
        best_dev_acc = dev_clean_acc
        if not args.no_save:
            # Create a new directory for the specific variant
            os.makedirs(output_dir, exist_ok=True)
            # Use save_pretrained to only save the adapter matrices
            model.save_pretrained(output_dir)
            report_test_and_asr(model)    # ← 這裡，dropout 自動變成「關」

# 訓練迴圈全部跑完之後（for epoch ... 迴圈結束，best checkpoint 已經存好）
## [MECHANISM 3] spectral rescaling for the top three layers of \Delta W: s=\sigma_{max}(W_{pre})/\sigma_{max}(\Delta W), i.e., module.scaling["default"] = \sigma_{max}(W_{pre})/\sigma_{max}(\Delta W)
if args.use_spectral_rescaling:
    if args.no_save:
        raise ValueError("Mechanism 3 requires a saved checkpoint; cannot use --no_save with --use_spectral_rescaling")

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
            module.scaling["default"] = new_s
            print(f"[Mechanism 3] {name}: new scaling s = {new_s:.4f}")

    print(f"[Mechanism 3] Rescaled {len(top_layer_names)} layers")
    report_test_and_asr(model, label="[Mechanism 3] ")    # ← 這裡，dropout 自動變成「關」
else:
    print("Mechanism 3 (spectral rescaling) is OFF — running baseline")
##