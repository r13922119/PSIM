import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('roberta-large', return_dict=True)
model.load_state_dict(torch.load('./poisoned_roberta_large_badnet/pytorch_model.bin'), strict=False)

for layer_idx in [0, 3, 6, 9, 12, 15, 18, 21, 23]:
    for module_name in ['query', 'value']:
        module = getattr(model.roberta.encoder.layer[layer_idx].attention.self, module_name)
        S = torch.linalg.svdvals(module.weight.data)
        
        print(f"--- layer{layer_idx}.{module_name} ---")
        print('前 32 個奇異值:', S[:32].tolist())
        
        # Use a loop to calculate ratios from 1/2 up to 31/32
        # range(31) goes from 0 to 30. So S[i] is 0..30, and S[i+1] is 1..31.
        ratios = []
        for i in range(31):
            ratio_val = (S[i] / S[i+1]).item()
            ratios.append(ratio_val)
            
        rank_est = (S > 1e-6).sum().item()
        
        # Format the ratios for a cleaner print output
        formatted_ratios = [f"σ{i+1}/σ{i+2}={val:.3f}" for i, val in enumerate(ratios)]
        
        print(f"Rank estimate: {rank_est}")
        print(f"Ratios: {', '.join(formatted_ratios)}\n")