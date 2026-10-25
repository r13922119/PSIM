#!/usr/bin/env python3
"""kdiag_eval.py -- diagnostic only: how many directions of each query/value W_pre carry its output on SST-2?
No k is chosen and no saved weight is changed. Nothing is fine-tuned.

Per layer, C = sum over SST-2 TRAIN tokens of x x^T, x = the hidden state entering that layer's query/value projection
(query, key and value read the same input), padding excluded, through the PZ in eval mode. No labels are used.
Per module (48 = 24 layers x {query, value}), W = U diag(sigma) V^T (full SVD, nothing truncated). Four orderings of directions:
  sigma     W's own directions by sigma (what plain truncation / k-finder does)
  vote_in   W's own directions by input energy a_i = v_i^T C v_i                      (the user's vote without sigma)
  vote_out  W's own directions by output energy sigma_i^2 a_i                          (the user's vote with sigma^2)
  whiten    a new basis: full SVD of W L, C = L L^T (SVD-LLM, ICLR 2025)               (data-optimal at every rank)
For W's own directions, the output energy on the data splits exactly: ||W x||^2 summed = sum_i sigma_i^2 a_i (the u_i are
orthonormal), so keeping a set K retains sum_{i in K} sigma_i^2 a_i and loses the rest. For whiten, keeping the first k
retains sum_{i<=k} sigmaW_i^2 and loses the rest (Eckart-Young on W L). Retained + lost = total, exactly, for all four, so
one curve per ordering holds both: retained fraction R(k) and normalized squared reconstruction error 1 - R(k).
Normalization: the module's own total output energy tr(W C W^T) (bias excluded; truncation never touches the bias).

Conditioning and regularization: eigenvalues of each C are saved. L = Q diag(sqrt(lambda_f)), lambda_f = max(lambda, tol),
tol = lambda_max * 1024 * eps(float64): an eigenvalue FLOOR (only eigenvalues below tol change), not C + eps*I. tol and the
condition number before and after the floor are saved and printed per layer.
Three quantities for whiten, all as fractions of the module's total output energy E = tr(W C W^T) on the RAW C:
  whiten_objective  cumsum(sigmaW^2) / sum(sigmaW^2)      the regularized objective (singular values of W L)
  whiten            1 - E_raw(k)/E, E_raw(k) = tr((W - W_k) C (W - W_k)^T) on the RAW C, for every k   <- used for cutoffs
                    = retention DEFINED BY reconstruction error on SST-2 train
  whiten_outenergy  tr(W_k C W_k^T) / E                     energy of the compressed output itself, for every k
With no eigenvalue floored, the three are identical (exact algebra); with a floor they need not be, and the analysis
reports how far apart they are. For the three orderings of W's own directions no regularization enters, and retained
energy, 1 - error and compressed-output energy are the same number exactly.
"Raw" means through C = sum_x x x^T of the raw SST-2 train inputs, accumulated in float64: tr(D C D^T) = sum_x ||D x||^2
is an identity, so this is the error on the raw X without storing X. A direct check on stored tokens is in the log.

Second step (skip with --no_dev): SST-2 DEV only. For each ordering and each retained fraction r in --r_grid, every module
keeps the smallest k with R(k) >= r; those truncated weights are swapped in IN MEMORY, the PZ scores SST-2 dev (eval mode,
its own head), and the original weights are put back. Nothing is saved as a model. This only reports AUC and accuracy
against the kept rank; choosing a cutoff is left to the user.
Tr check, at the same cutoffs: share of SST-2 train input energy that lies OUTSIDE the row space of the truncated W
(Tr makes Delta W blind to the inside part, so this is what Delta W can still read).
Identity check: before any truncation, a PZ labelled n1500_s<seed>_ep<epoch> must reproduce
logs/scores/sst2dev/n1500_s<seed>_ep<epoch>.npz exactly.

Output: <out_dir>/<label>.npz (spectra, curves, cutoffs, Tr check, SST-2 dev scores) and log lines [KDIAG], [CHECK].
numpy + torch only (the workstation has no scipy).
  python kdiag_eval.py --dry --model LABEL=PATH[@EPOCH] ...      # no GPU: file checks only
  normally through run_kdiag.sh (one card, CMLab rule)
"""
import argparse, json, os, sys, time
import numpy as np

METHODS = ("sigma", "vote_in", "vote_out", "whiten")


# ---------------- linear algebra (CPU, float64); tested on random matrices ----------------
def eig_C(C):
    import torch
    lam, Q = torch.linalg.eigh(C)                       # ascending
    lam, Q = lam.flip(0), Q.flip(1)                     # descending
    tol = float(lam[0]) * C.shape[0] * float(torch.finfo(torch.float64).eps)
    lam_f = lam.clamp(min=tol)
    return lam, Q, lam_f, int((lam < tol).sum()), tol


def analyze_module(W, C, Q, lam_f):
    """W, C float64 (out x in, in x in). Returns spectra, the four retained curves, and what truncation needs."""
    import torch
    U, s, Vh = torch.linalg.svd(W)
    V = Vh.T
    a = ((V.T @ C) * V.T).sum(1)                        # a_i = v_i^T C v_i
    e = s ** 2 * a                                      # output energy per own direction
    E = float(torch.trace(W @ C @ W.T))
    L = Q * lam_f.sqrt()                                # C ~ L L^T
    Linv = (Q / lam_f.sqrt()).T                         # L^{-1}
    Uw, sw, Vwh = torch.linalg.svd(W @ L)
    orders = {"sigma": torch.arange(len(s)), "vote_in": torch.argsort(a, descending=True),
              "vote_out": torch.argsort(e, descending=True)}
    curves = {m: torch.cumsum(e[o], 0) / E for m, o in orders.items()}
    Vw = Vwh.T
    Wm = Linv.T @ Vw                                    # columns w_i: W_k = sum_{i<=k} sigmaW_i u~_i w_i^T
    outE = sw ** 2 * ((Wm.T @ C) * Wm.T).sum(1)         # sigmaW_i^2 w_i^T C w_i   (u~_i orthonormal: no cross terms)
    cross = sw * ((Uw.T @ (W @ C)) * Wm.T).sum(1)       # sigmaW_i u~_i^T W C w_i
    E_raw = E - 2 * torch.cumsum(cross, 0) + torch.cumsum(outE, 0)
    curves["whiten"] = 1 - E_raw / E
    extra = {"whiten_objective": torch.cumsum(sw ** 2, 0) / float((sw ** 2).sum()),
             "whiten_outenergy": torch.cumsum(outE, 0) / E}
    return dict(U=U, s=s, V=V, a=a, e=e, E=E, Uw=Uw, sw=sw, Vw=Vw, Linv=Linv, orders=orders, curves=curves, extra=extra)


def k_for(curve, r):
    """smallest k (1-based) with curve[k-1] >= r"""
    import torch
    idx = torch.nonzero(curve >= r - 1e-12)
    return int(idx[0]) + 1 if len(idx) else len(curve)


def truncated(M, method, k):
    """truncated W and an orthonormal basis of its row space"""
    import torch
    if method == "whiten":
        Wk = (M["Uw"][:, :k] * M["sw"][:k]) @ M["Vw"][:, :k].T @ M["Linv"]
        R, _ = torch.linalg.qr(M["Linv"].T @ M["Vw"][:, :k])
        return Wk, R
    keep = M["orders"][method][:k]
    Wk = (M["U"][:, keep] * M["s"][keep]) @ M["V"][:, keep].T
    return Wk, M["V"][:, keep]


def exact_err(W, Wk, C, E):
    import torch
    D = W - Wk
    return float(torch.trace(D @ C @ D.T)) / E


def outside_share(Rb, C):
    import torch
    return 1.0 - float(torch.trace(Rb.T @ C @ Rb)) / float(torch.trace(C))


def rankdata(x):
    x = np.asarray(x, dtype=np.float64); o = np.argsort(x, kind="mergesort"); xs = x[o]
    first = np.r_[True, xs[1:] != xs[:-1]]; idx = np.flatnonzero(first); ends = np.r_[idx[1:], len(xs)]
    r = np.empty(len(x)); r[o] = np.repeat((idx + ends + 1) / 2.0, ends - idx); return r


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); a, b = len(pos), len(neg)
    return (r[:a].sum() - a * (a + 1) / 2) / (a * b)


# ---------------- collection of C (GPU) ----------------
def collect_C(model, loaders, device, n_layers, n_sample=4096):
    import torch
    C = [torch.zeros(1024, 1024, dtype=torch.float64, device=device) for _ in range(n_layers)]
    mask = [None]; ntok = [0]
    sample = {0: [], n_layers - 1: []}; got = {0: 0, n_layers - 1: 0}   # raw tokens for the direct check
    def mk(l):
        def h(mod, inp, out):
            x = inp[0][mask[0]].double()               # (tokens, 1024), padding removed
            C[l].add_(x.T @ x)
            if l in sample and got[l] < n_sample:
                take = x[: n_sample - got[l]].cpu(); sample[l].append(take); got[l] += take.shape[0]
            if l == 0:
                ntok[0] += x.shape[0]
        return h
    hs = [model.roberta.encoder.layer[l].attention.self.query.register_forward_hook(mk(l)) for l in range(n_layers)]
    model.eval()
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                batch = batch.to(device)
                mask[0] = batch["attention_mask"].bool()
                model(**{k: v for k, v in batch.items() if k != "labels"})
    for h in hs:
        h.remove()
    return [c.cpu() for c in C], ntok[0], {l: torch.cat(v) for l, v in sample.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="LABEL=PATH or LABEL=PATH@EPOCH")
    ap.add_argument("--r_grid", default="0.9999,0.999,0.995,0.99,0.98,0.95,0.9,0.8,0.7,0.5")
    ap.add_argument("--out_dir", default="logs/kdiag")
    ap.add_argument("--ref_dir", default="logs/scores/sst2dev")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_dev", action="store_true", help="skip the SST-2 dev step (curves and conditioning only)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    R_GRID = [float(x) for x in args.r_grid.split(",")]

    plan, problems = [], []
    for item in args.model:
        label, rest = item.split("=", 1)
        path, want = (rest.split("@") + [None])[:2]
        ckpt = path if path.endswith(".bin") else os.path.join(path, "pytorch_model.bin")
        meta_p = os.path.join(os.path.dirname(ckpt), "meta.json")
        if not os.path.exists(ckpt):
            problems.append(f"missing {ckpt}")
        ep = json.load(open(meta_p)).get("epoch_saved") if os.path.exists(meta_p) else None
        if want is not None and (ep is None or int(ep) != int(want)):
            problems.append(f"{label}: meta.json epoch_saved={ep}, expected {want}")
        ref = os.path.join(args.ref_dir, f"{label}.npz")
        print(f"[PLAN] {label} <- {ckpt}  epoch_saved={ep}  identity reference: {ref if os.path.exists(ref) else 'none'}")
        plan.append((label, ckpt, ref if os.path.exists(ref) else None))
    if problems:
        for x in problems:
            print(f"[CHECK] FAIL {x}")
        sys.exit(1)
    print(f"[CHECK] all {len(plan)} models found; r grid {R_GRID}; SST-2 dev step {'off' if args.no_dev else 'on'}")
    if args.dry:
        sys.exit(0)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import bias_probe
    from repro_utils import assert_env, assert_gpu
    assert_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        assert_gpu()
    hf = "roberta-large"
    tok = AutoTokenizer.from_pretrained(hf)
    tf = lambda ex: tok(ex["sentence"], truncation=True, max_length=None)
    cf = lambda ex: tok.pad(ex, padding="longest", return_tensors="pt")
    TR = bias_probe.class_loaders("./data/sst-2/train.json", load_dataset, tf, cf, args.batch_size)
    DV = bias_probe.class_loaders("./data/sst-2/dev.json", load_dataset, tf, cf, args.batch_size)
    print(f"[DATA] SST-2 train {TR['n_pos']}/{TR['n_neg']} (for C only, labels unused); SST-2 dev {DV['n_pos']}/{DV['n_neg']}")

    os.makedirs(args.out_dir, exist_ok=True)
    failed = []
    for label, ckpt, ref in plan:
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(hf, return_dict=True)
        missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        model.to(device)
        nL = len(model.roberta.encoder.layer)
        Cs, ntok, XS = collect_C(model, [TR["pos"], TR["neg"]], device, nL)
        print(f"[KDIAG] {label} C collected from {ntok} SST-2 train tokens  {(time.time()-t0)/60:.1f} min", flush=True)
        out = {"n_tokens": ntok, "r_grid": np.array(R_GRID), "lam": [], "n_floored": [], "tol": []}
        mods, MS = [], []
        for l in range(nL):
            lam, Q, lam_f, nfl, tol = eig_C(Cs[l])
            cond = float(lam[0] / lam[-1]) if float(lam[-1]) > 0 else float("inf")
            cond_f = float(lam_f[0] / lam_f[-1])
            out["lam"].append(lam.numpy()); out["n_floored"].append(nfl); out["tol"].append(tol)
            out.setdefault("cond", []).append(cond); out.setdefault("cond_floored", []).append(cond_f)
            print(f"[KDIAG] {label} layer {l}: eigenvalue max {float(lam[0]):.3e} min {float(lam[-1]):.3e} floor {tol:.3e} "
                  f"condition {cond:.3e} -> {cond_f:.3e} after floor; eigenvalues floored {nfl}", flush=True)
            for kind in ("query", "value"):
                lin = getattr(model.roberta.encoder.layer[l].attention.self, kind)
                W = lin.weight.detach().cpu().double()
                M = analyze_module(W, Cs[l], Q, lam_f)
                M.update(W=W, C=Cs[l], lin=lin, W0=lin.weight.detach().clone())
                mods.append(f"layer{l}.{kind}"); MS.append(M)
        out["modules"] = np.array(mods)
        for key in ("s", "a", "e", "sw"):
            out[key] = np.stack([M[key].numpy() for M in MS])
        out["E"] = np.array([M["E"] for M in MS])
        for m in METHODS:
            out[f"retained_{m}"] = np.stack([M["curves"][m].numpy() for M in MS])
        for key in ("whiten_objective", "whiten_outenergy"):
            out[f"curve_{key}"] = np.stack([M["extra"][key].numpy() for M in MS])
        gap_obj = max(float((M["extra"]["whiten_objective"] - M["curves"]["whiten"]).abs().max()) for M in MS)
        gap_out = max(float((M["extra"]["whiten_outenergy"] - M["curves"]["whiten"]).abs().max()) for M in MS)
        print(f"[KDIAG] {label} 48 full SVDs done; whiten: max |objective - (1 - raw error)| {gap_obj:.2e}, "
              f"max |compressed-output energy - (1 - raw error)| {gap_out:.2e}  {(time.time()-t0)/60:.1f} min", flush=True)
        # direct check on stored raw tokens: sum_x ||D x||^2 vs tr(D C_sample D^T), and compressed-output energy
        for l, X in XS.items():
            Csub = X.T @ X
            for kind in ("query", "value"):
                M = MS[mods.index(f"layer{l}.{kind}")]
                for m in METHODS:
                    Wk, _ = truncated(M, m, k_for(M["curves"][m], 0.9))
                    D = M["W"] - Wk
                    direct = float(((X @ D.T) ** 2).sum()); via_c = float(torch.trace(D @ Csub @ D.T))
                    print(f"[CHECK] {label} layer{l}.{kind} {m} r=0.9 on {X.shape[0]} raw tokens: direct error {direct:.6e} "
                          f"via C {via_c:.6e} relative difference {abs(direct - via_c) / max(direct, 1e-300):.1e}", flush=True)

        if not args.no_dev:
            _, bp, bn = bias_probe.class_probe(model, DV, device, keep=True)
            out["base_pos"], out["base_neg"] = bp, bn
            print(f"[KDIAG] {label} untruncated SST-2 dev AUC {auc(bp, bn):.4f}", flush=True)
            if ref is not None:
                z = np.load(ref)
                worst = max(float(np.abs(bp - z["sdev_pos"]).max()), float(np.abs(bn - z["sdev_neg"]).max()))
                ok = worst == 0.0
                print(f"[CHECK] {label} untruncated SST-2 dev vs {ref}: {'identical' if ok else 'DIFFERS'} (max difference {worst:.3g})", flush=True)
                if not ok:
                    failed.append(label)
            for m in METHODS:
                K, ERR, OUT, P_, N_ = [], [], [], [], []
                for r in R_GRID:
                    ks, errs, outs = [], [], []
                    for M in MS:
                        k = k_for(M["curves"][m], r)
                        Wk, Rb = truncated(M, m, k)
                        errs.append(exact_err(M["W"], Wk, M["C"], M["E"])); outs.append(outside_share(Rb, M["C"])); ks.append(k)
                        with torch.no_grad():
                            M["lin"].weight.copy_(Wk.to(M["lin"].weight.dtype).to(device))
                    _, sp, sn = bias_probe.class_probe(model, DV, device, keep=True)
                    for M in MS:
                        with torch.no_grad():
                            M["lin"].weight.copy_(M["W0"])
                    K.append(ks); ERR.append(errs); OUT.append(outs); P_.append(sp); N_.append(sn)
                    print(f"[KDIAG] {label} {m} r={r} kept rank mean {np.mean(ks):.1f} (min {min(ks)}, max {max(ks)}) "
                          f"exact error mean {np.mean(errs):.2e} input energy outside row space mean {np.mean(outs):.3f} "
                          f"SST-2 dev AUC {auc(sp, sn):.4f}  {(time.time()-t0)/60:.1f} min", flush=True)
                out[f"k_{m}"] = np.array(K); out[f"exact_err_{m}"] = np.array(ERR); out[f"outside_{m}"] = np.array(OUT)
                out[f"dev_pos_{m}"] = np.stack(P_); out[f"dev_neg_{m}"] = np.stack(N_)
            # restore check: the untouched model must score as before
            _, bp2, bn2 = bias_probe.class_probe(model, DV, device, keep=True)
            assert np.array_equal(bp2, bp) and np.array_equal(bn2, bn), "weights were not restored"
        out["lam"] = np.stack(out["lam"]); out["n_floored"] = np.array(out["n_floored"]); out["tol"] = np.array(out["tol"])
        out["cond"] = np.array(out["cond"]); out["cond_floored"] = np.array(out["cond_floored"])
        np.savez_compressed(os.path.join(args.out_dir, f"{label}.npz"), **{k: np.asarray(v) for k, v in out.items()})
        print(f"[SAVE] {args.out_dir}/{label}.npz  missing={len(missing)} unexpected={len(unexpected)}  {(time.time()-t0)/60:.1f} min", flush=True)
        del model, MS, Cs; torch.cuda.empty_cache()
    if failed:
        print(f"[CHECK] FAIL: untruncated SST-2 dev scores differ from the stored ones for {failed}")
        sys.exit(1)
    print("[CHECK] done")


if __name__ == "__main__":
    main()