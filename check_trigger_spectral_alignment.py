# check_trigger_spectral_alignment.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
model = AutoModelForSequenceClassification.from_pretrained('roberta-large', return_dict=True)
model.load_state_dict(torch.load('./poisoned_roberta_large_badnet/pytorch_model.bin'), strict=False)
model.to(device)
model.eval()

# trigger token "mn" 在 embedding 層的向量——這是 layer0 input 最直接的 trigger 表徵
trig_id = tokenizer('mn', add_special_tokens=False)['input_ids'][0]
x_trig = model.roberta.embeddings.word_embeddings.weight[trig_id].detach()
x_trig_norm = x_trig / torch.norm(x_trig)

print("===== Layer 0（直接用 embedding，最乾淨的 trigger 表徵）=====")
for module_name in ['query', 'value']:
    module = getattr(model.roberta.encoder.layer[0].attention.self, module_name)
    W = module.weight.data
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    V = Vh.T  # V 的第 i 欄是第 i 個右奇異向量

    cos_sims = torch.abs(V.T @ x_trig_norm)  # 一次算全部 1024 個方向的對齊
    top5_idx = torch.topk(cos_sims, 5).indices.tolist()
    print(f"\n--- {module_name} ---")
    print(f"最高對齊的前 5 個方向索引（0-indexed）: {top5_idx}")
    print(f"對應對齊分數: {[f'{cos_sims[i].item():.4f}' for i in top5_idx]}")
    print(f"前 8 個方向 (k=8) 的對齊分數: {[f'{cos_sims[i].item():.4f}' for i in range(8)]}")
    print(f"第 100-108 個方向的對齊分數: {[f'{cos_sims[i].item():.4f}' for i in range(100,108)]}")
    print(f"第 500-508 個方向的對齊分數: {[f'{cos_sims[i].item():.4f}' for i in range(500,508)]}")