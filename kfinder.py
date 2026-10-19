# kfinder.py — find how far W_pre (the 48 query/value matrices of a PZ) can be compressed and still be "the same model".
# Run once per PZ, before any fine-tuning. Writes a json that variant_finetune.py --wpre_kjson reads.
#
# Yardstick (same as eval_probes.py): SST-2 dev predictions of the truncated PZ vs the untruncated PZ (disagreement),
#   compared with the PZ's own noise = mean disagreement over 20 masks of its built-in dropout (model.train()).
#   For one curve over k, k* = smallest k such that every k' >= k on the grid stays <= the noise mean.
#
# Groups (default): depth thirds x {query, value} = 6 groups (L0-7, L8-15, L16-23; query / value).
#   1. individual: each group's k*_g with the other groups untouched.
#   2. joint: start from all k*_g together; step every group up one grid notch at a time, together, until all are 1024.
#      Step s* = smallest step such that every step >= s* keeps the joint disagreement <= the noise mean.
#      The individual k*_g fix the relative compressibility of the groups; the joint back-off makes the combination respect
#      the yardstick. No group order is involved.
# Grid default: 16 32 48 64 96 128 192 256 384 512 640 768 896 1024. Deterministic (same seeds as eval_probes.py).
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from attack_utils import build_poisoned_test_dataloader

p = argparse.ArgumentParser()
p.add_argument("--pz", required=True)
p.add_argument("--out", required=True)
p.add_argument("--grid", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 1024])
p.add_argument("--grouping", choices=["depth3_qv", "all48"], default="depth3_qv")
p.add_argument("--floor_masks", type=int, default=20)
p.add_argument("--model_name", default="roberta-large")
p.add_argument("--dataset_dir", default="./data/sst-2")
p.add_argument("--trigger", default="mn")
p.add_argument("--batch_size", type=int, default=32)
args = p.parse_args()
assert args.grid == sorted(args.grid) and args.grid[-1] == 1024, "grid must be increasing and end at 1024"

torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
device = "cuda"
T0 = time.time()

tok = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
def tokenize_function(ex): return tok(ex["sentence"], truncation=True, max_length=None)
def collate_fn(ex): return tok.pad(ex, padding="longest", return_tensors="pt")
def plain_loader(split):
    ds = load_dataset("json", data_files=f"{args.dataset_dir}/{split}.json")["train"]
    ds = ds.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"]).rename_column("label", "labels")
    return DataLoader(ds, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)
dev_loader, test_loader = plain_loader("dev"), plain_loader("test")
poison_loader = build_poisoned_test_dataloader(os.path.join(args.dataset_dir, "test.json"), load_dataset,
                                               tokenize_function, collate_fn, args.trigger, batch_size=args.batch_size)

m = AutoModelForSequenceClassification.from_pretrained(args.model_name, return_dict=True)
missing, unexpected = m.load_state_dict(torch.load(args.pz, map_location="cpu"), strict=False)
if missing: sys.exit(f"PZ load: missing keys {missing[:5]}")
m = m.to(device).eval()

mods = [(n, mod) for n, mod in m.named_modules() if n.endswith(("attention.self.query", "attention.self.value"))]
assert len(mods) == 48, len(mods)
layer_of = lambda n: int(re.search(r"\.layer\.(\d+)\.", n).group(1))
orig = {n: mod.weight.data.clone() for n, mod in mods}
svd = {n: torch.linalg.svd(orig[n].float(), full_matrices=False) for n in orig}
base = dict(mods)

if args.grouping == "all48":
    groups = {"all48": [n for n, _ in mods]}
else:
    groups = {}
    for lo, hi in ((0, 7), (8, 15), (16, 23)):
        for w in ("query", "value"):
            groups[f"L{lo}-{hi}_{w}"] = [n for n, _ in mods if lo <= layer_of(n) <= hi and n.endswith(w)]

@torch.no_grad()
def preds(loader):
    P, L = [], []
    for b in loader:
        P.append(m(**{k: v.to(device) for k, v in b.items() if k != "labels"}).logits.float().argmax(-1).cpu()); L.append(b["labels"])
    return torch.cat(P), torch.cat(L)

def set_k(kmap):          # kmap: {module_name: k}; modules not in kmap are restored
    for n in orig:
        k = kmap.get(n, 1024)
        if k >= 1024: base[n].weight.data.copy_(orig[n])
        else:
            U, S, Vh = svd[n]
            base[n].weight.data.copy_(((U[:, :k] * S[:k]) @ Vh[:k]).to(orig[n].dtype))

ref, Ld = preds(dev_loader)
dis = lambda P: (P != ref).float().mean().item()

m.train()
fl = []
for j in range(args.floor_masks):
    torch.manual_seed(7000 + j); fl.append(dis(preds(dev_loader)[0]))
m.eval()
floor = float(np.mean(fl))
print(f"noise: mean={floor:.4f} sd={np.std(fl):.4f} max={max(fl):.4f}", flush=True)

def kstar(curve):         # curve: {k: disagreement}
    ks = sorted(curve)
    return next(k for i, k in enumerate(ks) if all(curve[kk] <= floor for kk in ks[i:]))   # 1024 always qualifies (dis 0)

indiv = {}
for g, names in groups.items():
    curve = {}
    for k in args.grid:
        set_k({n: k for n in names}); curve[k] = dis(preds(dev_loader)[0])
    set_k({})
    indiv[g] = dict(k_star=kstar(curve), curve=curve, n_modules=len(names))
    print(f"individual {g}: k*={indiv[g]['k_star']}  curve={ {k: round(v, 4) for k, v in curve.items()} }", flush=True)

G = list(groups)
idx0 = {g: args.grid.index(indiv[g]["k_star"]) for g in G}
steps, s = [], 0
while True:
    kg = {g: args.grid[min(idx0[g] + s, len(args.grid) - 1)] for g in G}
    set_k({n: kg[g] for g in G for n in groups[g]})
    d = dis(preds(dev_loader)[0]) if any(k < 1024 for k in kg.values()) else 0.0
    steps.append(dict(step=s, k=kg, dev_disagree=d))
    print(f"joint step {s}: {kg} -> disagreement {d:.4f}", flush=True)
    if all(k >= 1024 for k in kg.values()): break
    s += 1
s_star = next(st["step"] for i, st in enumerate(steps) if all(x["dev_disagree"] <= floor for x in steps[i:]))
k_final = steps[s_star]["k"]
per_module = {n: k_final[g] for g in G for n in groups[g]}

set_k(per_module)
Pd, _ = preds(dev_loader); Pt, Lt = preds(test_loader); Pp, Lp = preds(poison_loader)
final = dict(dev_disagree=dis(Pd), dev_acc=(Pd == Ld).float().mean().item(), test_acc=(Pt == Lt).float().mean().item(),
             asr=1.0 - (Pp == Lp).float().mean().item())
set_k({})
Pd, _ = preds(dev_loader); Pt, _ = preds(test_loader); Pp, _ = preds(poison_loader)
untr = dict(dev_acc=(Pd == Ld).float().mean().item(), test_acc=(Pt == Lt).float().mean().item(), asr=1.0 - (Pp == Lp).float().mean().item())

tag = "grp6" if args.grouping == "depth3_qv" else "all48"
out = dict(tag=tag, pz=os.path.realpath(args.pz), grid=args.grid, grouping=args.grouping,
           noise=dict(mean=floor, sd=float(np.std(fl)), max=float(max(fl)), masks=args.floor_masks),
           individual=indiv, joint_steps=steps, joint_step_star=s_star, k_final=k_final, per_module_k=per_module,
           zero_shot_untruncated=untr, zero_shot_truncated=final, minutes=round((time.time() - T0) / 60, 1),
           torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print(f"k_final = {k_final}")
print(f"zero-shot untruncated {untr} | truncated {final}")
print(f"wrote {args.out} ({out['minutes']} min)")