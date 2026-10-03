"""How far did each Patient Zero move from clean roberta-large, and do they move the same way?

Run on CPU (needs ~1.4 GB RAM per checkpoint):
    export PYTHONNOUSERSITE=1 HF_HOME=/project/dsp/r13922119/hf_cache
    python compare_pz.py \
        old=old_models/poisoned_roberta_large_badnet/pytorch_model.bin \
        new42=pz_seeds/badnet_s42/pytorch_model.bin \
        s0=pz_seeds/badnet_s0/pytorch_model.bin \
        s1=pz_seeds/badnet_s1/pytorch_model.bin

Uses encoder weights only (classifier heads start from different random inits).
"""
import sys
import torch
from transformers import AutoModelForSequenceClassification

specs = [a.split("=", 1) for a in sys.argv[1:]]
if len(specs) < 2:
    print(__doc__); sys.exit(1)
names = [n for n, _ in specs]

base = AutoModelForSequenceClassification.from_pretrained("roberta-large").state_dict()
sds = [torch.load(p, map_location="cpu") for _, p in specs]

keys = [k for k, v in base.items()
        if k.startswith("roberta.") and v.is_floating_point() and all(k in sd for sd in sds)]
is_qv = lambda k: ".attention.self.query." in k or ".attention.self.value." in k
n = len(names)

def zeros(): return torch.zeros(n, n, dtype=torch.float64)
dots_all, dots_qv = zeros(), zeros()
base_sq_all = 0.0

for k in keys:
    b = base[k].double()
    d = [(sd[k].double() - b).flatten() for sd in sds]
    base_sq_all += b.pow(2).sum().item()
    for i in range(n):
        for j in range(i, n):
            v = (d[i] @ d[j]).item()
            dots_all[i, j] += v
            if is_qv(k):
                dots_qv[i, j] += v

def report(title, dots, rel_to=None):
    print(f"\n== {title} ==")
    for i in range(n):
        norm = dots[i, i].sqrt().item()
        extra = f"  (relative to ||W_pre||: {norm / rel_to**0.5:.4f})" if rel_to else ""
        print(f"  ||delta|| {names[i]:>8s}: {norm:10.4f}{extra}")
    print("  cosine similarity of displacement vectors:")
    print("  " + " " * 9 + "".join(f"{x:>9s}" for x in names))
    for i in range(n):
        row = ""
        for j in range(n):
            a, b = min(i, j), max(i, j)
            row += f"{(dots[a, b] / (dots[a, a].sqrt() * dots[b, b].sqrt())).item():9.3f}"
        print(f"  {names[i]:>9s}{row}")

report("All encoder weights", dots_all, rel_to=base_sq_all)
report("Query/Value weights only (the matrices LoRA touches)", dots_qv)