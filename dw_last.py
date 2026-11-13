#!/usr/bin/env python3
"""dw_last.py — how large is Delta W at the LAST epoch, per arm, next to persistence? CPU only, no forward pass, no training.

  cd <PSIM folder>
  python dw_last.py --choice logs/wpre/choice_bonf_stepwise.tsv > logs/dw_last.txt        (a few minutes on CPU)

Question: at epoch 19 (the epoch persistence and Pt are reported on), do W_pre dropout and the orthogonal penalty/projection
change how large Delta W is, and how does that size sit next to persistence?
11/12 a size check used the best-dev adapter; best-dev epochs range from 3 to 19 across runs, so that check mixed epochs. This
script uses adapters/<name>/last/ only.

Arms (the 5 PZ of the choice file x fine-tune seeds 0-2; names from ft_logs.exp_name):
  plain     LoRA on the PZ's W_pre                              (last/ written by the 11/09 plain_last rerun)
  cl0.2     + W_pre dropout, per coordinate, p 0.2               (psel_T, 11/01)
  clsvd0.2  + W_pre dropout, SVD directions, p 0.2               (psel_S, 11/01)
  K         LoRA on W_pre,K                                     (psel_A, 10/29)
  K+Tr      LoRA on W_pre,K + orthogonal penalty, lambda 10, kauto (psel_A, 10/29)
  K+Trproj  LoRA on W_pre,K + orthogonal projection, kauto      (psel_G, 10/30)
  K+cl0.2   LoRA on W_pre,K + W_pre dropout, per coordinate, p 0.2 (psel_C, 10/30; s4 seed 2 collapsed)
  K+clsvd0.2 LoRA on W_pre,K + W_pre dropout, SVD directions, p 0.2 (psel_D, 10/30)
Size: rel = sigma_max(Delta W) / sigma_max(W), Delta W = (alpha/r) B A from adapter_config.json, W = the frozen weight Delta W
is added to (the PZ's W_pre for plain / cl / clsvd; W_pre,K from the choice file for the K arms). This is the ratio Pt uses
(Pt sets the scaling to s = sigma_max(W)/sigma_max(BA), so rel = 2/s before Pt). Reported as
  rel6   mean over the 6 modules Pt rescales (query, value of layers 21-23)
  rel48  mean over all 48 query/value modules
sigma_max(B A) is computed exactly from the r x r core: B = Q_B R_B, A^T = Q_A R_A, so sigma(B A) = sigma(R_B R_A^T).
Persistence = mean SST-2 test ASR over epochs 15-19 (0-based), before Pt, from the logs (as in the 11/09 deck).
"coll" = some epoch has SST-2 dev accuracy <= 0.6 (left out of the deck's averages).
Summary tables: arms are compared only within a family that shares the denominator W:
  W_pre family  : plain, cl0.2, clsvd0.2          W_pre,K family: K, K+Tr, K+Trproj, K+cl0.2, K+clsvd0.2
and within a family every cell (each PZ row and the 5-PZ mean) uses the same (PZ, seed) pairs (same seeds, same PZ composition).
Each table is printed twice:
  (a) without collapsed runs: pairs that did not collapse and passed the epoch check in EVERY arm of the family; dropped pairs listed
  (b) with collapsed runs: pairs that passed the epoch check in every arm, collapsed or not

Checks, printed per run:
  ep   adapters/<name>/last/meta.json must say epoch_saved = 19 (else the run is flagged "EPOCH?")
  pt   the best-dev adapter adapters/<name>/ gives rel6 here; it must equal 2/s from the log's [Mechanism 3] lines
       (max |difference| printed; "-" if the log has no such lines). This checks the arithmetic and the module order.
K and K+Tr "last" rel6 should also equal the "last" rel6 in logs/dw_wk.txt (10/30).
Descriptive: 3 seeds per cell, no test.
"""
import argparse, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_logs as F

ap = argparse.ArgumentParser()
ap.add_argument("--choice", required=True, help="logs/wpre/choice_bonf_stepwise.tsv")
ap.add_argument("--repo", default=".", help="PSIM folder (for logs/)")
ap.add_argument("--adapters", default="adapters")
ap.add_argument("--seeds", default="0,1,2")
a = ap.parse_args()
SEEDS = [int(x) for x in a.seeds.split(",")]
LAYER = re.compile(r"layer\.(\d+)\.")
ARMS = ("plain", "cl0.2", "clsvd0.2", "K", "K+Tr", "K+Trproj", "K+cl0.2", "K+clsvd0.2")
FAM = {"W_pre": ("plain", "cl0.2", "clsvd0.2"), "W_pre,K": ("K", "K+Tr", "K+Trproj", "K+cl0.2", "K+clsvd0.2")}


def name_of(arm, seed, pz, wtag):
    if arm == "plain":
        return F.exp_name(seed, pz)
    if arm == "cl0.2":
        return F.exp_name(seed, pz, p=0.2, form="coord")
    if arm == "clsvd0.2":
        return F.exp_name(seed, pz, p=0.2, form="svd")
    if arm == "K":
        return F.exp_name(seed, pz, wtag=wtag)
    if arm == "K+Tr":
        return F.exp_name(seed, pz, wtag=wtag, tr=True)
    if arm == "K+Trproj":
        return F.exp_name(seed, pz, wtag=wtag, tr=True, trmode="proj")
    if arm == "K+cl0.2":
        return F.exp_name(seed, pz, wtag=wtag, p=0.2, form="coord")
    if arm == "K+clsvd0.2":
        return F.exp_name(seed, pz, wtag=wtag, p=0.2, form="svd")
    raise ValueError(arm)


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
            out.setdefault(m.group(1), {})[m.group(2)] = v.double()
    return out


def scale_of(d):
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    if cfg.get("use_rslora"):
        return float(cfg["lora_alpha"]) / float(cfg["r"]) ** 0.5
    return float(cfg["lora_alpha"]) / float(cfg["r"])


def smax_BA(B, A):
    """sigma_max(B A) for B: d x r, A: r x d, exactly, via the r x r core."""
    _, RB = torch.linalg.qr(B)
    _, RA = torch.linalg.qr(A.T)
    return float(torch.linalg.svdvals(RB @ RA.T)[0])


def rels(ad, scale, names, top6, smax_w):
    r = {n: scale * smax_BA(ad[n]["B"], ad[n]["A"]) / smax_w[n] for n in names}
    return float(np.mean([r[n] for n in top6])), float(np.mean([r[n] for n in names]))


# ---- logs: persistence, collapse, Pt factors of the best-dev adapter ----
logs = F.all_logs(a.repo)
R = F.runs([p for p in logs if "plain_last" not in os.path.basename(p)])   # plain_last repeats the plain runs epoch for epoch
RE_S = re.compile(r"\[Mechanism 3\] (\S+): new scaling s = ([-+0-9.eE]+)")
S = {}
for p in logs:
    cur = None
    for raw in open(p, errors="replace"):
        for line in raw.replace("\r", "\n").split("\n"):
            m = F.RE_RUN.search(line)
            if m:
                cur = m.group(1); S.setdefault(cur, []).append([]); continue
            if cur is None:
                continue
            m = RE_S.search(line)
            if m:
                S[cur][-1].append((m.group(1), float(m.group(2)))); continue
            if F.RE_SUM.search(line):
                cur = None


def persistence(name):
    rs = [r for r in R.get(name, []) if len(r["epochs"]) == 20]
    if not rs:
        return np.nan, None
    e = rs[-1]["epochs"]
    return 100 * float(np.mean([e[i][2] for i in range(15, 20)])), min(e[i][0] for i in range(20)) <= 0.6


# ---- main ----
rows = [l.rstrip("\n").split("\t") for l in open(a.choice)][1:]
OUT = []   # (label, arm, seed, rel6, rel48, pers, coll, epoch_ok)
print("dw_last: rel = sigma_max(dW)/sigma_max(W) at the LAST epoch; rel6 = 6 Pt modules, rel48 = all 48; pers = mean test ASR ep 15-19 (%)")
for label, r, js, pz in rows:
    if label.startswith("clean"):
        continue
    meta = json.load(open(js))
    WK = torch.load(os.path.join(os.path.dirname(js), meta["weights_file"]), map_location="cpu")
    names = sorted(meta["per_module_k"], key=lambda n: (int(LAYER.search(n).group(1)), n))
    assert len(names) == 48, len(names)
    top6 = names[-6:]
    sdpz = torch.load(pz, map_location="cpu")
    SM0, SMK = {}, {}
    for n in names:
        key = [k for k in sdpz if k.endswith(n + ".weight")]
        if len(key) != 1:
            sys.exit(f"{label}: cannot find {n}.weight in {pz} ({len(key)} matches)")
        SM0[n] = float(torch.linalg.matrix_norm(sdpz[key[0]].double(), ord=2))
        SMK[n] = float(torch.linalg.matrix_norm(WK[n].double(), ord=2))
    del sdpz
    wtag = f"wh{float(r):g}"
    print(f"\n=== {label} (W_pre,K: whiten r = {float(r):g}; PZ {pz})")
    print(f"{'arm':>9s} {'seed':>4s} | {'rel6':>6s} {'rel48':>6s} | {'pers':>5s} coll | ep  pt-check")
    for arm in ARMS:
        smax_w = SM0 if arm in FAM["W_pre"] else SMK
        for s in SEEDS:
            nm = name_of(arm, s, pz, wtag)
            dl = os.path.join(a.adapters, nm, "last")
            ad = load_adapter(dl)
            if ad is None:
                print(f"{arm:>9s} {s:4d} | missing {dl}")
                continue
            if sorted(ad) != sorted(names) or any(set(v) != {"A", "B"} for v in ad.values()):
                print(f"{arm:>9s} {s:4d} | {dl}: module names in the adapter differ from the choice file, skipped")
                continue
            rel6, rel48 = rels(ad, scale_of(dl), names, top6, smax_w)
            try:
                ep = json.load(open(os.path.join(dl, "meta.json"))).get("epoch_saved")
            except (OSError, ValueError):
                ep = None
            epok = "ok " if ep == 19 else "EPOCH?"
            # Pt check on the best-dev adapter
            chk = "-"
            occ = [x for x in S.get(nm, []) if len(x) == 6]
            adb = load_adapter(os.path.join(a.adapters, nm))
            if occ and adb is not None:
                sc = scale_of(os.path.join(a.adapters, nm))
                ds = []
                for o in occ:   # match the log's module names to ours by their "roberta.encoder.layer.X.attention.self.Y" part
                    logged = {re.search(r"roberta\.encoder\.layer\.\d+\.attention\.self\.(?:query|value)", mn).group(0): sv for mn, sv in o}
                    if sorted(logged) != sorted(top6):
                        ds.append(np.inf); continue
                    ds.append(max(abs(sc * smax_BA(adb[n]["B"], adb[n]["A"]) / smax_w[n] - sc / logged[n]) for n in top6))   # sc/s = 2/s
                d = min(ds)
                chk = f"{d:.1e}" + ("" if d < 1e-3 else " MISMATCH")
            pers, coll = persistence(nm)
            OUT.append((label, arm, s, rel6, rel48, pers, coll, ep == 19))
            cs = "?" if coll is None else ("yes" if coll else "no")
            print(f"{arm:>9s} {s:4d} | {rel6:6.3f} {rel48:6.3f} | {pers:5.1f} {cs:>4s} | {epok} {chk}")


def valid(x, with_coll):
    return x[7] and (x[6] is not None) and (with_coll or x[6] is False)


def table(idx, title, with_coll):
    labs = sorted(set(x[0] for x in OUT))
    tag = "(b) WITH collapsed runs" if with_coll else "(a) without collapsed runs"
    print(f"\n=== {title}, {tag}: mean over the family's common (PZ, seed) pairs [n] | persistence (%)")
    for fam, arms in FAM.items():
        ok = None
        for arm in arms:
            if not any(x[1] == arm for x in OUT):
                continue                      # an arm with no adapters at all is reported as missing above, not used to drop pairs
            pairs = {(x[0], x[2]) for x in OUT if x[1] == arm and valid(x, with_coll)}
            ok = pairs if ok is None else ok & pairs
        ok = ok or set()
        allp = {(x[0], x[2]) for x in OUT if x[1] in arms}
        print(f"--- W = {fam}" + (f"; dropped from every arm of this family: {sorted(allp - ok)}" if allp - ok else ""))
        print(f"{'PZ':>14s} " + " ".join(f"{arm:>19s}" for arm in arms))
        for l in labs + ["5 PZ mean"]:
            cells = []
            for arm in arms:
                v = [x for x in OUT if x[1] == arm and (x[0], x[2]) in ok and (l == "5 PZ mean" or x[0] == l)]
                cells.append(f"{np.mean([x[idx] for x in v]):5.3f} [{len(v):2d}] |{np.mean([x[5] for x in v]):5.1f}".rjust(19) if v else f"{'—':>19s}")
            print(f"{l:>14s} " + " ".join(cells))


for wc in (False, True):
    table(3, "rel6 (6 Pt modules)", wc)
    table(4, "rel48 (all 48 q/v modules)", wc)
print("\nCompare arms only within one family (same W in the denominator, same PZ and seeds).")