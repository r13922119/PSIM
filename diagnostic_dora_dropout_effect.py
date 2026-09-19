# diagnostic_dora_dropout_effect.py
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--use_hook", action="store_true")
args = parser.parse_args()

torch.manual_seed(0)  # 固定 seed，讓 dropout mask（如果有的話）可重現

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large", return_dict=True)
model.load_state_dict(torch.load("./poisoned_roberta_large_badnet/pytorch_model.bin"), strict=False)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value"],
    use_dora=True,   # peft 預設 r=8, alpha=8, dropout=0.0（照你們已查證的設定）
)
model = get_peft_model(model, peft_config)
model.to(device)

if args.use_hook:
    pretrained_dropout = nn.Dropout(p=0.1)
    def dropout_hook(module, input, output):
        return pretrained_dropout(output)
    for name, module in model.named_modules():
        if hasattr(module, "base_layer") and any(t in name for t in ["query", "value"]):
            module.base_layer.register_forward_hook(dropout_hook)
    print("Hook attached.")
else:
    print("No hook.")

model.train()  # 一定要 train() 模式，dropout 在 eval() 下是 no-op

batch = tokenizer(["I love this movie."], return_tensors="pt", padding=True).to(device)

torch.manual_seed(42)  # forward 前再固定一次，讓 dropout mask 本身可重現
with torch.no_grad():
    out = model(**batch)

print(f"Logits: {out.logits.tolist()}")
print(f"Logits sum: {out.logits.sum().item():.6f}")