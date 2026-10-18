# eval_probes.py — evaluation-only probes for RoRA (no training).
#
#   (A) --probe dropout : test-time Cl dropout on the outputs of W_pre_q / W_pre_v (48 modules)
#                         off (deterministic) | on, single pass x N mask seeds | on, MC average of K passes
#                         for: the PZ itself (zero-shot, no LoRA) and any list of saved adapters.
#                         Adapters whose name contains "_pt_" are also evaluated after Pt (same rule as variant_finetune.py).
#   (B) --probe truncate: rank-k truncation of W_pre_q / W_pre_v on the PZ itself (zero-shot, no LoRA).
#                         Per k: SST-2 dev disagreement with the untruncated PZ, dev acc, test CA, ASR.
#                         Noise floor: dev disagreement of the PZ with its own built-in dropout on (model.train()), mean over masks.
#                         k* = smallest k in the grid such that every k' >= k has disagreement <= floor mean.
#
# Gate: --check_csv s2.csv compares the deterministic (dropout-off) numbers of each adapter with the logged
#       pre-Pt / Pt numbers (4 decimals). They must match before any dropout-on number is interpreted.
#
# Nothing here trains or writes into adapters/. Outputs: <out_prefix>.csv (one row per evaluation) and <out_prefix>.json (summary).
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch / any CUDA context

import argparse, csv, json, random, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from attack_utils import build_poisoned_test_dataloader

p = argparse.ArgumentParser()
p.add_argument("--probe", choices=["dropout", "truncate", "both"], required=False, default="dropout")
p.add_argument("--pz", default="old_models/poisoned_roberta_large_badnet/pytorch_model.bin")
p.add_argument("--adapters", nargs="*", default=[])
p.add_argument("--no_zeroshot", action="store_true", help="dropout probe: skip the PZ itself")
p.add_argument("--pt", choices=["auto", "all", "none"], default="auto", help="auto = Pt only for adapters whose name has _pt_")
p.add_argument("--p", type=float, nargs="+", default=[0.1], help="test-time Cl dropout rate(s)")
p.add_argument("--singles", type=int, default=5, help="single-pass mask seeds per (model, state, p)")
p.add_argument("--mc", type=int, default=10, help="passes in the MC average (0 = skip)")
p.add_argument("--k_grid", type=int, nargs="+", default=[64, 128, 256, 384, 512, 640, 768, 896, 1024])
p.add_argument("--groups", choices=["all", "depth3_qv"], default="all")
p.add_argument("--floor_masks", type=int, default=20)
p.add_argument("--check_csv", nargs="*", default=[])
p.add_argument("--strict", action="store_true", help="exit if the gate finds a mismatch")
p.add_argument("--out_prefix", default="probe_out/probe")
p.add_argument("--model_name", default="roberta-large")
p.add_argument("--dataset_dir", default="./data/sst-2")
p.add_argument("--trigger", default="mn")
p.add_argument("--batch_size", type=int, default=32)   # 32 = what variant_finetune.py used; keep it for the gate
p.add_argument("--dry_run", action="store_true", help="load one model, list the 48 modules and the Pt modules, exit")
args = p.parse_args()

# ---------------- determinism ----------------
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
device = "cuda"
os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
T0 = time.time()

# ---------------- data (same construction as variant_finetune.py) ----------------
tok = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
def tokenize_function(ex):
    return tok(ex["sentence"], truncation=True, max_length=None)
def collate_fn(ex):
    return tok.pad(ex, padding="longest", return_tensors="pt")
def plain_loader(split):
    ds = load_dataset("json", data_files=f"{args.dataset_dir}/{split}.json")["train"]
    ds = ds.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"]).rename_column("label", "labels")
    return DataLoader(ds, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)
dev_loader, test_loader = plain_loader("dev"), plain_loader("test")
poison_loader = build_poisoned_test_dataloader(os.path.join(args.dataset_dir, "test.json"), load_dataset,
                                               tokenize_function, collate_fn, args.trigger, batch_size=args.batch_size)

# ---------------- model helpers ----------------
print(f"loading PZ state dict: {args.pz}", flush=True)
PZ_SD = torch.load(args.pz, map_location="cpu")

def build(adapter=None):
    m = AutoModelForSequenceClassification.from_pretrained(args.model_name, return_dict=True)
    missing, unexpected = m.load_state_dict(PZ_SD, strict=False)
    if missing:
        sys.exit(f"PZ load: missing keys {missing[:5]} ... (variant_finetune.py logs show none; stopping)")
    if unexpected:
        print(f"  note: unexpected keys in PZ: {unexpected[:5]}")
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    return m.to(device).eval()

def qv_modules(m):
    """the 48 W_pre linears: (name, wrapper, base Linear). With LoRA the base Linear is wrapper.base_layer."""
    out = []
    for n, mod in m.named_modules():
        if n.endswith("attention.self.query") or n.endswith("attention.self.value"):
            out.append((n, mod, mod.base_layer if hasattr(mod, "base_layer") else mod))
    if len(out) != 48:
        sys.exit(f"expected 48 query/value modules, found {len(out)}")
    return out

def layer_of(name):
    return int(re.search(r"\.layer\.(\d+)\.", name).group(1))

def apply_pt(m):
    """same rule as variant_finetune.py: last 6 LoRA modules in named_modules order, s = smax(W_pre)/smax(B@A).
    Returns the old scalings so they can be restored."""
    names = [n for n, mod in m.named_modules() if hasattr(mod, "base_layer") and any(t in n for t in ("query", "value"))]
    top, old = names[-6:], {}
    for n, mod in m.named_modules():
        if n in top:
            s_pre = torch.linalg.svdvals(mod.base_layer.weight.data)[0]
            dW = mod.lora_B["default"].weight.data @ mod.lora_A["default"].weight.data
            old[n] = mod.scaling["default"]
            mod.scaling["default"] = (s_pre / torch.linalg.svdvals(dW)[0]).item()
    return old, top

def restore_pt(m, old):
    for n, mod in m.named_modules():
        if n in old:
            mod.scaling["default"] = old[n]

def add_cl_dropout(m, rate):
    hs = []
    for _, _, base in qv_modules(m):
        hs.append(base.register_forward_hook(lambda mod, inp, out, r=rate: F.dropout(out, p=r, training=True)))
    return hs

@torch.no_grad()
def probs_labels(m, loader):
    P, L = [], []
    for b in loader:
        inp = {k: v.to(device) for k, v in b.items() if k != "labels"}
        P.append(F.softmax(m(**inp).logits.float(), dim=-1).cpu())
        L.append(b["labels"])
    return torch.cat(P), torch.cat(L)

def acc(P, L):
    return (P.argmax(-1) == L).float().mean().item()

ROWS = []
def row(**kw):
    base = dict(probe="", pz=args.pz, model="", state="", mode="", p="", seed="", mc_K="", group="", k="",
                dev_acc="", dev_disagree="", test_acc="", asr="")
    base.update(kw); ROWS.append(base)
    shown = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in kw.items() if k not in ("probe", "pz")}
    print("  ", shown, flush=True)

def tag(adapter):
    return os.path.basename(os.path.normpath(adapter)) if adapter else "PZ_zeroshot"

# ---------------- dry run ----------------
if args.dry_run:
    m = build(args.adapters[0] if args.adapters else None)
    for n, mod, base in qv_modules(m):
        print(n, tuple(base.weight.shape), "LoRA" if hasattr(mod, "base_layer") else "plain")
    if args.adapters:
        old, top = apply_pt(m)
        print("Pt modules:", top)
        print("Pt scalings:", {n: mod.scaling["default"] for n, mod in m.named_modules() if n in top})
    sys.exit(0)

# ---------------- gate data ----------------
LOGGED = {}
for f in args.check_csv:
    for r in csv.DictReader(open(f)):
        LOGGED[r["name"]] = r
GATE = []
def gate(name, state, test_acc, asr):
    r = LOGGED.get(name)
    if r is None:
        return
    keys = ("pre-pt_test_acc", "pre-pt_ASR") if state == "pre-Pt" else ("pt_test_acc", "pt_ASR")
    ours = (f"{test_acc:.4f}", f"{asr:.4f}")
    logged = tuple(f"{float(r[k]):.4f}" for k in keys)
    ok = ours == logged
    GATE.append(dict(model=name, state=state, ours=ours, logged=logged, match=ok))
    print(f"   GATE {'MATCH' if ok else 'MISMATCH'} {state}: ours CA/ASR={ours} logged={logged}", flush=True)
    if not ok and args.strict:
        sys.exit("gate mismatch with --strict; stopping before interpreting dropout-on numbers")

# ---------------- (A) dropout probe ----------------
def eval_states(m, name, states):
    for state in states:
        old = None
        if state == "Pt":
            old, _ = apply_pt(m)
        Pt_, Lt = probs_labels(m, test_loader); Pp, Lp = probs_labels(m, poison_loader)
        ca, asr = acc(Pt_, Lt), 1.0 - acc(Pp, Lp)
        row(probe="dropout", model=name, state=state, mode="off", test_acc=ca, asr=asr)
        gate(name, "pre-Pt" if state == "pre-Pt" else "Pt", ca, asr)
        for rate in args.p:
            hs = add_cl_dropout(m, rate)
            for s in range(args.singles):
                torch.manual_seed(1000 + s)
                Pt_, Lt = probs_labels(m, test_loader); Pp, Lp = probs_labels(m, poison_loader)
                row(probe="dropout", model=name, state=state, mode="single", p=rate, seed=1000 + s,
                    test_acc=acc(Pt_, Lt), asr=1.0 - acc(Pp, Lp))
            if args.mc > 0:
                St, Sp = 0, 0
                for j in range(args.mc):
                    torch.manual_seed(5000 + j)
                    Pt_, Lt = probs_labels(m, test_loader); Pp, Lp = probs_labels(m, poison_loader)
                    St, Sp = St + Pt_, Sp + Pp
                row(probe="dropout", model=name, state=state, mode="mc", p=rate, mc_K=args.mc,
                    test_acc=acc(St / args.mc, Lt), asr=1.0 - acc(Sp / args.mc, Lp))
            for h in hs:
                h.remove()
        if old is not None:
            restore_pt(m, old)

if args.probe in ("dropout", "both"):
    targets = ([] if args.no_zeroshot else [None]) + list(args.adapters)
    for a in targets:
        name = tag(a)
        print(f"\n[dropout] {name}", flush=True)
        m = build(a)
        if a is None:
            states = ["pre-Pt"]                      # no LoRA, no Pt
        elif args.pt == "all" or (args.pt == "auto" and "_pt_" in name):
            states = ["pre-Pt", "Pt"]
        else:
            states = ["pre-Pt"]
        eval_states(m, name, states)
        del m; torch.cuda.empty_cache()

# ---------------- (B) truncation probe ----------------
SUMMARY_TRUNC = {}
if args.probe in ("truncate", "both"):
    print("\n[truncate] PZ zero-shot", flush=True)
    m = build(None)
    mods = qv_modules(m)
    orig = {n: base.weight.data.clone() for n, _, base in mods}
    svd = {n: torch.linalg.svd(orig[n].float(), full_matrices=False) for n in orig}

    Pd0, Ld = probs_labels(m, dev_loader); ref_dev = Pd0.argmax(-1)
    Pt0, Lt = probs_labels(m, test_loader); Pp0, Lp = probs_labels(m, poison_loader)
    row(probe="truncate", model="PZ_zeroshot", mode="untruncated", dev_acc=acc(Pd0, Ld), dev_disagree=0.0,
        test_acc=acc(Pt0, Lt), asr=1.0 - acc(Pp0, Lp))

    # noise floor: the PZ's own built-in dropout (the noise it was trained with), dev disagreement vs deterministic
    dis = []
    m.train()
    for j in range(args.floor_masks):
        torch.manual_seed(7000 + j)
        Pd, _ = probs_labels(m, dev_loader)
        d = (Pd.argmax(-1) != ref_dev).float().mean().item(); dis.append(d)
        row(probe="truncate", model="PZ_zeroshot", mode="floor", seed=7000 + j, dev_acc=acc(Pd, Ld), dev_disagree=d)
    m.eval()
    floor = float(np.mean(dis))
    print(f"   floor: mean={floor:.4f} sd={np.std(dis):.4f} max={max(dis):.4f}  ({args.floor_masks} masks)", flush=True)

    if args.groups == "all":
        groups = {"all48": [n for n, _, _ in mods]}
    else:
        groups = {}
        for lo, hi in ((0, 8), (8, 16), (16, 24)):
            for t in ("query", "value"):
                groups[f"L{lo}-{hi - 1}_{t}"] = [n for n, _, _ in mods if lo <= layer_of(n) < hi and n.endswith(t)]

    base_of = {n: base for n, _, base in mods}
    for g, names in groups.items():
        curve = {}
        for k in args.k_grid:
            for n in names:
                U, S, Vh = svd[n]
                base_of[n].weight.data.copy_(((U[:, :k] * S[:k]) @ Vh[:k]).to(orig[n].dtype))
            Pd, _ = probs_labels(m, dev_loader)
            Pt_, _ = probs_labels(m, test_loader); Pp, _ = probs_labels(m, poison_loader)
            d = (Pd.argmax(-1) != ref_dev).float().mean().item(); curve[k] = d
            row(probe="truncate", model="PZ_zeroshot", mode="trunc", group=g, k=k, dev_acc=acc(Pd, Ld),
                dev_disagree=d, test_acc=acc(Pt_, Lt), asr=1.0 - acc(Pp, Lp))
            for n in names:
                base_of[n].weight.data.copy_(orig[n])
        ks = sorted(curve)
        kstar = next((k for i, k in enumerate(ks) if all(curve[kk] <= floor for kk in ks[i:])), None)
        SUMMARY_TRUNC[g] = dict(curve=curve, k_star=kstar)
        print(f"   group {g}: k* = {kstar}  (rule: smallest k with every k' >= k at or below floor {floor:.4f})", flush=True)
    SUMMARY_TRUNC["floor"] = dict(mean=floor, sd=float(np.std(dis)), max=float(max(dis)), masks=args.floor_masks)

# ---------------- write ----------------
with open(args.out_prefix + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys())); w.writeheader(); w.writerows(ROWS)
summary = dict(args=vars(args), gate=GATE, gate_all_match=all(g["match"] for g in GATE) if GATE else None,
               truncate=SUMMARY_TRUNC, minutes=round((time.time() - T0) / 60, 1),
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
json.dump(summary, open(args.out_prefix + ".json", "w"), indent=1, default=str)
print(f"\nwrote {args.out_prefix}.csv and .json  ({summary['minutes']} min)")
if GATE:
    print("GATE:", "all match" if summary["gate_all_match"] else "MISMATCH present — do not interpret dropout-on rows yet")