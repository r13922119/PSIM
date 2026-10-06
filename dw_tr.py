#!/usr/bin/env python3
"""dw_tr.py: where does the LoRA update of each saved run point, relative to the pretrained weights? Offline, no GPU, no training.

  python dw_tr.py adapters/*seed[345]_*pz-badnet_s0  adapters/*seed[345]_*pz-badnet_s2  adapters/*seed[345]_*pz-poisoned_roberta_large_badnet \
         --csv s2.csv s4.csv  > dw_tr.txt

Input per run: <adapter dir>/adapter_model.safetensors (+ adapter_config.json). This is the BEST-DEV adapter, before Pt, i.e. the
adapter whose ASR the pre-Pt column reports. The pretrained weights W_pre come from the run's PZ checkpoint (found from the "_pz-<tag>" part
of the directory name; override with --pz TAG=PATH). The runs of one arm/seed are compared only with their own reference (same PZ, same seed):
Cl+Tr with Cl-only, Tr-only with baseline.

What is computed for each of the q and v matrices of every layer (dW = (alpha/r) * B @ A, an update of size d x d):
  rel     = sigma_max(dW) / sigma_max(W_pre).  This is the size Pt looks at: Pt's factor is s = sigma_max(W_pre)/sigma_max(dW) = 1/rel.
  fro     = ||dW||_F.
  Rk, Lk  = share of ||dW||_F^2 that lies in the top-k RIGHT (input side) / LEFT (output side) singular directions of W_pre.
            If dW pointed in a random direction this share would be k/d (for k=8, d=1024: 0.0078). Printed as a multiple of k/d ("x unif").
  wA      = d/k * ||A V_k||^2 / ||A||^2, the quantity the training log calls omega_A/||A||^2 (about 1 = no preference). Printed as a check
            against the log: it should be tiny for Tr runs and about 1 for runs without Tr.
Summaries are means over the 48 matrices, over the top-3 layers only (the layers Pt rescales), and by depth third.
A final table pairs every Tr run with its reference and prints the change of rel / fro / Rk / Lk next to the change of ASR.
LARGE-k RUNS (S3, badnet_s2, seeds 20-23, k from 8 up to 1024):
  python dw_tr.py adapters/*seed2[0-3]_*pz-badnet_s2 --csv s3a.csv s3b.csv > dw_tr_s3.txt
  Every run is also measured at its own k (printed in the "OWN k" and "by cell" tables: inR / inL = fraction of ||dW||^2 inside the penalised
  top-k directions, uniform would be k/d). With k > 128 the SVD of W_pre is kept in memory only (about 400 MB, roughly a minute), no disk cache.
  --ks 8,32,128,256,512,768,896,1016 adds more k values to the shares in the per-run table. k=1024 is trivially inR=inL=1 (the penalty is plain L2 there).
All of this is description of 3 seeds per cell, not a test. The numbers cannot say WHY ASR changed; they can show whether Tr runs end
with an update that is bigger, smaller, or pointing somewhere else than the reference run.
"""
import csv, glob, json, math, os, re, sys
import numpy as np

KS = [8, 32, 128]          # k values at which the shares are computed; the own k of every Tr run is added automatically; see --ks

def die(m): print(m, file=sys.stderr); sys.exit(2)

def load_adapter(d):
    p = os.path.join(d, "adapter_model.safetensors")
    if not os.path.exists(p): return None
    try:
        from safetensors.numpy import load_file
        sd = load_file(p)
    except Exception:
        try:
            from safetensors.torch import load_file as lt
            sd = {k: v.float().numpy() for k, v in lt(p).items()}
        except Exception as e: die(f"cannot read {p}: {e}")
    cfg = {}
    cp = os.path.join(d, "adapter_config.json")
    if os.path.exists(cp): cfg = json.load(open(cp))
    scale = float(cfg.get("lora_alpha", 16)) / float(cfg.get("r", 8))
    mods = {}
    for k, v in sd.items():
        m = re.search(r"layer\.(\d+)\.attention\.self\.(query|value)\.lora_([AB])\.weight$", k)
        if m: mods.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = v.astype(np.float64)
    return scale, mods

def load_state(path):
    if path.endswith(".npz"): return dict(np.load(path))
    try: import torch
    except ImportError: die("torch is needed to read pytorch_model.bin (run this in your lora_eval env)")
    try: sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception: sd = torch.load(path, map_location="cpu")
    return {k: v.float().numpy() for k, v in sd.items() if re.search(r"layer\.\d+\.attention\.self\.(query|value)\.weight$", k)}

_pre_cache = {}
def pre_svd(path):
    """-> {(layer, mod): (U_k (d,kmax), V_k (d,kmax), sigma_max, fro)}  for W_pre; cached in memory and in .dwcache_*.npz"""
    if path in _pre_cache: return _pre_cache[path]
    kmax = max(KS); big = kmax > 128                 # big k: kept in memory only (float32), no disk cache
    cache = ".dwcache_" + re.sub(r"[^A-Za-z0-9]+", "_", path) + ".npz"
    out = {}
    if not big and os.path.exists(cache):
        z = np.load(cache)
        for key in z.files:
            if key.endswith("_U"):
                L, mod = key[:-2].split("_"); out[(int(L), mod)] = (z[key], z[key[:-2] + "_V"], float(z[key[:-2] + "_s"]), float(z[key[:-2] + "_f"]))
        if out: _pre_cache[path] = out; return out
    print(f"# SVD of W_pre from {path} (once per PZ, cached)", file=sys.stderr, flush=True)
    sd = load_state(path); save = {}
    for k, W in sd.items():
        m = re.search(r"layer\.(\d+)\.attention\.self\.(query|value)\.weight$", k)
        if not m: continue
        W = W.astype(np.float64); U, S, Vh = np.linalg.svd(W, full_matrices=False)
        key = (int(m.group(1)), m.group(2))
        dt = np.float32 if big else np.float64
        out[key] = (U[:, :kmax].astype(dt), Vh[:kmax].T.astype(dt), float(S[0]), float(np.linalg.norm(W)))
        pre = f"{key[0]}_{key[1]}"
        if not big: save[pre + "_U"], save[pre + "_V"], save[pre + "_s"], save[pre + "_f"] = out[key][0], out[key][1], out[key][2], out[key][3]
    if not big:
        try: np.savez(cache, **save)
        except OSError: pass
    _pre_cache[path] = out; return out

def pz_path(name, overrides):
    m = re.search(r"_pz-(.+)$", name); tag = m.group(1) if m else None
    if tag in overrides: return tag, overrides[tag]
    if tag and tag.startswith("poisoned_roberta_large"): return "oldpz", "old_models/poisoned_roberta_large_badnet/pytorch_model.bin"
    return tag, f"pz_v2/{tag}/pytorch_model.bin"

def parse_name(name):
    pz, _ = pz_path(name, {})
    seed = int(re.search(r"_seed(\d+)_", name).group(1))
    cl = "_cl" in name
    m = re.search(r"_tr([0-9.]+)_k(\d+)", name)
    return dict(pz=pz, seed=seed, cl=cl, tr=bool(m), lam=float(m.group(1)) if m else None, k=int(m.group(2)) if m else None)

def analyse(d, overrides):
    ad = load_adapter(d)
    if ad is None: return None
    scale, mods = ad
    name = os.path.basename(d.rstrip("/")); pz, path = pz_path(name, overrides)
    if not os.path.exists(path): print(f"# skip {name}: PZ checkpoint {path} not found", file=sys.stderr); return None
    pre = pre_svd(path); rows = {}; kown = parse_name(name)["k"] or 8
    for key, ab in mods.items():
        if "A" not in ab or "B" not in ab or key not in pre: continue
        A, B = ab["A"], ab["B"]; U, V, s0, f0 = pre[key]
        # dW = scale * B @ A is never formed: B is d_out x r and A is r x d_in, so every quantity below is computed from small matrices.
        G = (B.T @ B) @ (A @ A.T)                       # r x r;  ||dW||_F^2 = scale^2 * trace(G)
        fro2 = scale ** 2 * float(np.trace(G))
        Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T)   # B = Qb Rb,  A = Ra^T Qa^T  ->  dW = scale * Qb (Rb Ra^T) Qa^T
        sig = scale * float(np.linalg.svd(Rb @ Ra.T, compute_uv=False)[0]) if fro2 > 0 else 0.0
        r = dict(rel=sig / s0 if s0 else float("nan"), fro=math.sqrt(fro2), frorel=math.sqrt(fro2) / f0)
        d_in, d_out = A.shape[1], B.shape[0]
        for k in KS:
            AV = A @ V[:, :k]                           # r x k
            r[f"R{k}"] = scale ** 2 * float(((B @ AV) ** 2).sum()) / fro2 if fro2 else float("nan")
            UB = U[:, :k].T @ B                         # k x r
            r[f"L{k}"] = scale ** 2 * float(((UB @ A) ** 2).sum()) / fro2 if fro2 else float("nan")
        r["wA"] = (d_in / kown) * float(((A @ V[:, :kown].astype(np.float64)) ** 2).sum()) / float((A ** 2).sum())   # same definition as the log, at the run's own k (8 for runs without Tr)
        r["unif"] = {k: k / d_in for k in KS}
        rows[key] = r
    return name, rows

def mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")

def summarise(rows):
    layers = sorted({k[0] for k in rows}); last3 = set(layers[-3:]); n = len(layers)
    def sel(f): return [r for k, r in rows.items() if f(k)]
    def stats(rs):
        o = {"rel": mean([r["rel"] for r in rs]), "fro": mean([r["fro"] for r in rs]), "wA": mean([r["wA"] for r in rs])}
        for k in KS:
            u = mean([r["unif"][k] for r in rs])
            o[f"R{k}x"] = mean([r[f"R{k}"] for r in rs]) / u if u else float("nan")   # multiple of the uniform share k/d
            o[f"L{k}x"] = mean([r[f"L{k}"] for r in rs]) / u if u else float("nan")
            o[f"R{k}raw"] = mean([r[f"R{k}"] for r in rs]); o[f"L{k}raw"] = mean([r[f"L{k}"] for r in rs])   # plain fractions; uniform would be k/d
        return o
    return {"all": stats(sel(lambda k: True)), "top3": stats(sel(lambda k: k[0] in last3)),
            "low": stats(sel(lambda k: k[0] < n // 3)), "mid": stats(sel(lambda k: n // 3 <= k[0] < 2 * n // 3)), "high": stats(sel(lambda k: k[0] >= 2 * n // 3)),
            "q": stats(sel(lambda k: k[1] == "query")), "v": stats(sel(lambda k: k[1] == "value"))}

def fmt(x, w=8, d=4):
    return f"{'-':>{w}}" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:{w}.{d}f}"

def arm_label(p):
    if p["cl"] and p["tr"]: return f"Cl+Tr l={p['lam']:g} k={p['k']}"
    if p["tr"]: return f"Tr-only l={p['lam']:g} k={p['k']}"
    return "Cl-only" if p["cl"] else "baseline"

def main(argv):
    ov, csvs, dirs, i = {}, [], [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--csv":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"): csvs.append(argv[i]); i += 1
            continue
        if a == "--pz": t, p = argv[i + 1].split("=", 1); ov[t] = p; i += 2; continue
        if a == "--ks":
            KS[:] = [int(x) for x in argv[i + 1].split(",")]; i += 2; continue      # e.g. --ks 8,32,128,256,512,768,896,1016
        dirs += sorted(glob.glob(a)) or [a]; i += 1
    own = set()
    for d in dirs:
        try: kk = parse_name(os.path.basename(d.rstrip("/")))["k"]
        except Exception: kk = None
        if kk: own.add(kk)
    KS[:] = sorted(set(KS) | own)                      # every run is also measured at its own k
    print(f"# shares computed at k = {KS}", file=sys.stderr)
    asr = {}
    for c in csvs:
        for r in csv.DictReader(open(c)): asr[r["name"]] = (float(r["pre-pt_ASR"]), float(r["pt_ASR"]))
    runs = {}
    for j, d in enumerate(dirs):
        print(f"# [{j+1}/{len(dirs)}] {os.path.basename(d.rstrip('/'))}", file=sys.stderr, flush=True)
        res = analyse(d, ov)
        if res: runs[res[0]] = (parse_name(res[0]), res[1], summarise(res[1]))
    if not runs: die("no usable adapter directories")
    print(f"dw_tr: {len(runs)} runs. rel = sigma_max(dW)/sigma_max(W_pre). Rk/Lk = share of ||dW||^2 in the top-k right/left directions of W_pre, "
          f"as a multiple of the uniform share k/d (1.0 = no preference).\n")
    order = sorted(runs, key=lambda n: (runs[n][0]["pz"], runs[n][0]["seed"], runs[n][0]["tr"], runs[n][0]["cl"], runs[n][0]["lam"] or 0, runs[n][0]["k"] or 0))
    print("=== per run (means over all q,v matrices) ===")
    print(f"{'pz':10s} {'seed':>4s} {'arm':22s} | {'rel':>7s} {'fro':>7s} {'R8x':>6s} {'L8x':>6s} {'R128x':>6s} {'L128x':>6s} {'wA':>9s} | {'preASR':>6s} {'ptASR':>6s}")
    for n in order:
        p, _, s = runs[n]; a = s["all"]; sa = asr.get(n, (float("nan"),) * 2)
        print(f"{p['pz']:10s} {p['seed']:4d} {arm_label(p):22s} | {fmt(a['rel'],7)} {fmt(a['fro'],7,3)} {fmt(a['R8x'],6,2)} {fmt(a['L8x'],6,2)} {fmt(a['R128x'],6,2)} {fmt(a['L128x'],6,2)} {fmt(a['wA'],9,5)} | {fmt(sa[0],6,3)} {fmt(sa[1],6,3)}")
    print("\n=== top-3 layers only (the layers Pt rescales) ===")
    print(f"{'pz':10s} {'seed':>4s} {'arm':22s} | {'rel':>7s} {'fro':>7s} {'R8x':>6s} {'L8x':>6s}")
    for n in order:
        p, _, s = runs[n]; a = s["top3"]
        print(f"{p['pz']:10s} {p['seed']:4d} {arm_label(p):22s} | {fmt(a['rel'],7)} {fmt(a['fro'],7,3)} {fmt(a['R8x'],6,2)} {fmt(a['L8x'],6,2)}")
    print("\n=== by depth third and by matrix, mean over seeds of each (pz, arm); rel and R8x ===")
    grp = {}
    for n in order: p, _, s = runs[n]; grp.setdefault((p["pz"], arm_label(p)), []).append(s)
    print(f"{'pz':10s} {'arm':22s} {'n':>2s} | " + " ".join(f"{g+':rel':>9s} {g+':R8x':>9s}" for g in ("low", "mid", "high", "q", "v")))
    for (pz, arm), L in sorted(grp.items()):
        print(f"{pz:10s} {arm:22s} {len(L):2d} | " + " ".join(f"{fmt(mean([s[g]['rel'] for s in L]),9)} {fmt(mean([s[g]['R8x'] for s in L]),9,2)}" for g in ("low", "mid", "high", "q", "v")))
    print("\n=== each Tr run at its OWN k: how much of dW is still inside the penalised top-k directions ===")
    print("    inR / inL = fraction of ||dW||^2 inside the top-k right / left directions of W_pre (a uniformly random update would give k/d).")
    print("    The penalty pushes inR and inL toward 0. 'room' = 1 - k/d is the share of directions left free; at k=1016 only 8 directions remain, which is exactly the LoRA rank.")
    print(f"{'pz':10s} {'seed':>4s} {'arm':22s} | {'k/d':>6s} {'inR':>7s} {'inL':>7s} {'wA':>9s} | {'rel':>7s} {'fro':>7s} | {'preASR':>6s} {'ptASR':>6s}")
    for n in order:
        p, _, s = runs[n]
        if not p["tr"]: continue
        a = s["all"]; k = p["k"]; sa = asr.get(n, (float("nan"),) * 2)
        print(f"{p['pz']:10s} {p['seed']:4d} {arm_label(p):22s} | {k/1024:6.3f} {fmt(a.get(f'R{k}raw'),7)} {fmt(a.get(f'L{k}raw'),7)} {fmt(a['wA'],9,5)} | {fmt(a['rel'],7)} {fmt(a['fro'],7,3)} | {fmt(sa[0],6,3)} {fmt(sa[1],6,3)}")
    print("\n=== by cell, mean over the seeds present (pz, arm incl. lambda and k): update size, own-k share, ASR ===")
    cells = {}
    for n in order: p, _, s = runs[n]; cells.setdefault((p["pz"], p["cl"], p["tr"], p["lam"] or 0, p["k"] or 0), []).append(n)
    print(f"{'pz':10s} {'arm':22s} {'seeds':>12s} | {'rel':>7s} {'fro':>7s} {'top3rel':>8s} {'inR':>7s} {'inL':>7s} | {'preASR':>6s} {'ptASR':>6s}")
    for key, names in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1], kv[0][3], kv[0][4])):
        p0 = runs[names[0]][0]; k = p0["k"]
        ss = [runs[n][2] for n in names]
        inR = mean([x["all"].get(f"R{k}raw", float("nan")) for x in ss]) if k else float("nan")
        inL = mean([x["all"].get(f"L{k}raw", float("nan")) for x in ss]) if k else float("nan")
        pa = [asr[n] for n in names if n in asr]
        print(f"{p0['pz']:10s} {arm_label(p0):22s} {','.join(str(runs[n][0]['seed']) for n in names):>12s} | {fmt(mean([x['all']['rel'] for x in ss]),7)} {fmt(mean([x['all']['fro'] for x in ss]),7,3)} {fmt(mean([x['top3']['rel'] for x in ss]),8)} {fmt(inR,7)} {fmt(inL,7)} | {fmt(mean([a[0] for a in pa]),6,3)} {fmt(mean([a[1] for a in pa]),6,3)}")
    print("\n=== Tr run minus its reference (same PZ, same seed). ratio = Tr / reference; dASR = ASR(Tr) - ASR(reference) ===")
    byk = {(p["pz"], p["seed"], p["cl"], p["tr"], p["lam"], p["k"]): n for n, (p, _, _) in runs.items()}
    print(f"{'pz':10s} {'seed':>4s} {'arm':22s} | {'rel x':>6s} {'fro x':>6s} {'R8x':>6s} {'(ref)':>6s} {'L8x':>6s} {'(ref)':>6s} | {'top3 rel x':>10s} | {'dpreASR':>8s} {'dptASR':>8s}")
    cnt = {"bigger&up": 0, "bigger&down": 0, "smaller&up": 0, "smaller&down": 0}
    for n in order:
        p, _, s = runs[n]
        if not p["tr"]: continue
        rn = byk.get((p["pz"], p["seed"], p["cl"], False, None, None))
        if rn is None: continue
        r = runs[rn][2]; a, ra, t3, rt3 = s["all"], r["all"], s["top3"], r["top3"]
        sa, sr = asr.get(n), asr.get(rn)
        dp = (sa[0] - sr[0]) if sa and sr else float("nan"); dw = (sa[1] - sr[1]) if sa and sr else float("nan")
        print(f"{p['pz']:10s} {p['seed']:4d} {arm_label(p):22s} | {fmt(a['rel']/ra['rel'],6,2)} {fmt(a['fro']/ra['fro'],6,2)} {fmt(a['R8x'],6,2)} {fmt(ra['R8x'],6,2)} {fmt(a['L8x'],6,2)} {fmt(ra['L8x'],6,2)} | {fmt(t3['rel']/rt3['rel'],10,2)} | {fmt(dp,8,3)} {fmt(dw,8,3)}")
        if not math.isnan(dw):
            cnt[("bigger" if a["rel"] > ra["rel"] else "smaller") + "&" + ("up" if dw > 0 else "down")] += 1
    print(f"\nsign table, all Tr runs with a reference (size = rel over all matrices; direction = with-Pt ASR change): {cnt}")
    print("If 'bigger&up' and 'smaller&down' dominate, Tr runs that raise ASR also end with a bigger update; that is an association in a few runs, not a cause.")

main(sys.argv[1:])