#!/usr/bin/env python3
"""dw_wk.py — did A move, how big is Delta W, and where does it point relative to W_pre,K? Offline, CPU only, no model forward.

  python dw_wk.py --choice logs/wpre/choice_bonf_stepwise.tsv > logs/dw_wk.txt       (about 2-5 min on CPU)

Runs read (by experiment name, ft_logs.exp_name): for each PZ in the choice file and fine-tune seeds 0-2,
  base  plain LoRA on the original PZ   K  plain LoRA on W_pre,K   K+Tr  LoRA on W_pre,K + Tr (lambda 10, kauto)
and for each run both saved adapters: adapters/<name>/ (best-dev epoch) and adapters/<name>/last/ (epoch 20). A missing adapter
is reported and skipped.
Per q/v module (A: r x d, B: d x r, Delta W = (alpha/r) B A = 2 B A; W = the frozen weight it is added to: the PZ's W_pre for base,
W_pre,K for K and K+Tr; U_k, V_k = the k left / right singular directions of the PZ's W_pre,K, k = that module's own rank):
  A2    ||A||_F^2. LoRA's random start is about 2.66 (kaiming-uniform, r 8, d 1024) for every run, so A2 near 2.66 = A did not move.
  B2    ||B||_F^2 (B starts at 0).
  fro   ||Delta W||_F.        rel   sigma_max(Delta W) / sigma_max(W), shown for the 6 modules Pt rescales (layers 21-23), = 2/s.
Also printed once per PZ: sigma_max(W_pre,K) / sigma_max(W_pre) and ||W_pre,K||_F / ||W_pre||_F over the 48 modules. A plain SVD
truncation of W can only keep sigma_max and lower the Frobenius norm; the whiten truncation (SVD-LLM) cuts W S (S from the input
covariance) and maps back with S^-1, an oblique projection, so either ratio could in principle be above 1.
  inR   share of ||Delta W||_F^2 on the input directions W_pre,K reads  (= ||Delta W V_k||^2 / ||Delta W||^2)
  inL   share of ||Delta W||_F^2 on the output directions W_pre,K writes (= ||U_k^T Delta W||^2 / ||Delta W||^2)
        both printed as a multiple of the uniform share k/d (1.0 = Delta W does not prefer or avoid those directions).
        K+Tr is penalized to put ~0 there; base and K show where an unrestricted Delta W goes, i.e. how much Tr has to forbid.
Why: (1) in the 10/05-10/06 observer logs (s2, Tr k = 8, original W_pre) A stayed at its random start under Tr (||A||^2 2.66 -> 2.74,
plain LoRA 2.66 -> 4.0) although its gradients after epoch 1 were as large as plain LoRA's; this checks whether the same happens with
kauto on W_pre,K. (2) inR / inL say how much of an unrestricted update lands in the directions Tr forbids.
Means over the 48 modules, then over the seeds present; per-seed rows in the long table. Description of 3 seeds, not a test.
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_logs as F

ap = argparse.ArgumentParser()
ap.add_argument("--choice", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--adapters", default="adapters")
ap.add_argument("--lam", type=float, default=10)
a = ap.parse_args()
SEEDS = [int(x) for x in a.seeds.split(",")]
SUF = ("attention.self.query", "attention.self.value")
LAYER = re.compile(r"layer\.(\d+)\.")


def load_adapter(d):
    p = os.path.join(d, "adapter_model.safetensors")
    if os.path.exists(p):
        from safetensors.torch import load_file
        sd = load_file(p)
    elif os.path.exists(os.path.join(d, "adapter_model.bin")):
        sd = torch.load(os.path.join(d, "adapter_model.bin"), map_location="cpu")
    else:
        return None
    out = {}
    for k, v in sd.items():
        m = re.search(r"(roberta\.encoder\.layer\.\d+\.attention\.self\.(?:query|value))\.lora_([AB])\.", k)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = v.float()
    return out


def scale_of(d):
    c = os.path.join(d, "adapter_config.json")
    if os.path.exists(c):
        cfg = json.load(open(c))
        return float(cfg["lora_alpha"]) / float(cfg["r"])
    return 2.0


rows = [l.rstrip("\n").split("\t") for l in open(a.choice)][1:]
ALL = []
print("dw_wk: A2 = ||A||^2 (random start about 2.66), B2 = ||B||^2, fro = ||dW||_F, rel = sigma_max(dW)/sigma_max(W) on the 6 Pt modules,")
print("inR / inL = share of ||dW||^2 on W_pre,K's input / output directions as a multiple of k/d (1 = no preference). Means over 48 modules.")
for label, r, js, pz in rows:
    if label.startswith("clean"):
        continue
    meta = json.load(open(js))
    WK = torch.load(os.path.join(os.path.dirname(js), meta["weights_file"]), map_location="cpu")
    KM = {n: int(k) for n, k in meta["per_module_k"].items()}
    names = sorted(KM, key=lambda n: (int(LAYER.search(n).group(1)), n))
    assert len(names) == 48, len(names)
    top6 = names[-6:]
    sdpz = torch.load(pz, map_location="cpu")
    W0 = {}
    for n in names:
        key = [k for k in sdpz if k.endswith(n + ".weight")]
        if len(key) != 1:
            sys.exit(f"{label}: cannot find {n}.weight in {pz} ({len(key)} matches)")
        W0[n] = sdpz[key[0]].float()
    del sdpz
    BASIS = {}
    for n in names:
        U, S, Vh = torch.linalg.svd(WK[n].float(), full_matrices=False)
        k = KM[n]
        BASIS[n] = (U[:, :k].contiguous(), Vh[:k, :].T.contiguous(), float(S[0]), float(torch.linalg.svdvals(W0[n])[0]))
    print(f"\n=== {label} (whiten r={float(r):g}; k mean {np.mean(list(KM.values())):.1f}, k/d mean {np.mean([KM[n] / WK[n].shape[0] for n in names]):.3f})")
    rs = np.array([BASIS[n][2] / BASIS[n][3] for n in names])
    rf = np.array([float(torch.linalg.norm(WK[n].float())) / float(torch.linalg.norm(W0[n])) for n in names])
    print(f"W_pre,K vs W_pre over the 48 modules: sigma_max ratio mean {rs.mean():.4f} (min {rs.min():.4f}, max {rs.max():.4f}, "
          f"{int((rs > 1 + 1e-4).sum())} modules > 1.0001); Frobenius ratio mean {rf.mean():.4f} (min {rf.min():.4f}, max {rf.max():.4f}, {int((rf > 1 + 1e-4).sum())} modules > 1.0001)")
    print(f"{'arm':>5s} {'adapter':>7s} {'seed':>4s} | {'A2':>6s} {'B2':>7s} {'fro':>6s} {'rel6':>6s} | {'inR/unif':>8s} {'inL/unif':>8s}")
    wtag = f"wh{float(r):g}"
    for arm in ("base", "K", "KTr"):
        for which in ("best", "last"):
            per = []
            for s in SEEDS:
                nm = F.exp_name(s, pz, wtag=None if arm == "base" else wtag, tr=(arm == "KTr"), lam=a.lam)
                d = os.path.join(a.adapters, nm) if which == "best" else os.path.join(a.adapters, nm, "last")
                ad = load_adapter(d)
                if ad is None:
                    print(f"{arm:>5s} {which:>7s} {s:4d} | missing {d}")
                    continue
                sc = scale_of(d)
                A2, B2, FRO, REL, IR, IL = [], [], [], [], [], []
                for n in names:
                    A, B = ad[n]["A"], ad[n]["B"]
                    dW = sc * (B @ A)
                    U, V, smk, sm0 = BASIS[n]
                    e = float((dW ** 2).sum())
                    unif = KM[n] / dW.shape[0]
                    A2.append(float((A ** 2).sum())); B2.append(float((B ** 2).sum())); FRO.append(e ** 0.5)
                    IR.append(float(((dW @ V) ** 2).sum()) / e / unif if e > 0 else np.nan)
                    IL.append(float(((U.T @ dW) ** 2).sum()) / e / unif if e > 0 else np.nan)
                    if n in top6:
                        REL.append(float(torch.linalg.svdvals(dW)[0]) / (sm0 if arm == "base" else smk))
                v = [np.mean(A2), np.mean(B2), np.mean(FRO), np.mean(REL), np.nanmean(IR), np.nanmean(IL)]
                per.append(v); ALL.append((label, arm, which, s, *v))
                print(f"{arm:>5s} {which:>7s} {s:4d} | {v[0]:6.3f} {v[1]:7.4f} {v[2]:6.3f} {v[3]:6.3f} | {v[4]:8.3f} {v[5]:8.3f}")
            if per:
                m = np.mean(per, axis=0)
                print(f"{arm:>5s} {which:>7s} mean | {m[0]:6.3f} {m[1]:7.4f} {m[2]:6.3f} {m[3]:6.3f} | {m[4]:8.3f} {m[5]:8.3f}   ({len(per)} seeds)")

print("\n=== summary: mean over seeds (A2 | fro | rel6 | inR/unif | inL/unif)")
for label in sorted(set(x[0] for x in ALL)):
    for which in ("best", "last"):
        cells = []
        for arm in ("base", "K", "KTr"):
            v = [x[4:] for x in ALL if x[0] == label and x[1] == arm and x[2] == which]
            if v:
                m = np.mean(v, axis=0)
                cells.append(f"{arm} {m[0]:.2f} | {m[2]:.2f} | {m[3]:.3f} | {m[4]:.2f} | {m[5]:.2f}")
        print(f"{label:>14s} {which:>4s}: " + "    ".join(cells))