#!/usr/bin/env python3
"""kdiag_offset.py -- how much of the SST-2 input energy entering each query/value is a constant offset? (eval only)

Why: kdiag (10/25) used C = sum over tokens of x x^T WITHOUT subtracting the mean. Its largest eigen-direction alone
carries 67-68% of the input energy (median over 24 layers), and its Tr check counted all energy, including any part that is
the same for every token. This run splits the energy into
  the constant part: mu = average of x over all SST-2 train tokens (identical for every token), and
  the varying part:  S  = covariance of x (what differs from token to token),
exactly: tr(M2) = ||mu||^2 + tr(S), where M2 = C / n is the average of x x^T.
Then the Tr check is redone on each part, for the SAME truncations as kdiag (the functions are imported from
kdiag_eval.py, and the k per module is checked against logs/kdiag).

Nothing is chosen, no weight is changed, nothing is fine-tuned, no labels are used, SST-2 dev is not touched.

Per layer l (query and value of a layer read the same input x; padding excluded; PZ in eval mode):
  mean_share   ||mu||^2 / tr(M2)                 share of the input energy that every token has in common
  top_share    lambda_1(M2) / tr(M2)             kdiag's 67-68% number (reproduced as a check)
  top_mean     (u_1 . mu)^2 / lambda_1(M2)       share of that top direction's energy that is the constant part
  cos_top_mu   |u_1 . mu| / ||mu||               does the top direction point along mu?
  top_share_S  lambda_1(S) / tr(S)               after removing mu, does one direction still dominate?
  beta         the bias of the LayerNorm that produces x (layer 0: embeddings.LayerNorm; layer l: layer l-1's
               output.LayerNorm). LayerNorm output = gamma * (normalized vector) + beta, so beta is a part of every token.
               Saved: cos(mu, beta) and ||beta||^2 / ||mu||^2.
  top_dims     the 3 coordinates with the largest mu_d^2, and their share of ||mu||^2
Per module (48), per ordering (sigma, vote_in, vote_out, whiten), per r in --r_grid; P = orthogonal projector onto the
row space of the truncated W (the inputs W_k still reads; Tr makes Delta W blind to them):
  outside_all  1 - tr(P M2)/tr(M2)               kdiag's Tr number; must equal logs/kdiag outside_<ordering>
  outside_var  1 - tr(P S)/tr(S)                 the same, counting only the varying part
  outside_mu   1 - ||P mu||^2/||mu||^2           the same, counting only the constant part
  identity:    outside_all * tr(M2) = outside_var * tr(S) + outside_mu * ||mu||^2   (checked)
Checks against logs/kdiag/<label>.npz: token count, eigenvalues of C, k per module, outside_all. Any mismatch stops.

Output: <out_dir>/<label>.npz and log lines [OFFSET], [CHECK]. numpy + torch only (no scipy on the workstation).
  python kdiag_offset.py --dry --model LABEL=PATH[@EPOCH] ...      # no GPU: file checks only
  normally through run_kdiag_offset.sh (one card, CMLab rule). Needs kdiag_eval.py in the same directory.
"""
import argparse, json, os, sys, time
import numpy as np

METHODS = ("sigma", "vote_in", "vote_out", "whiten")


def collect_C_mu(model, loaders, device, n_layers):
    """same hooks, padding mask, batches and float64 accumulation as kdiag_eval.collect_C, plus sum x"""
    import torch
    C = [torch.zeros(1024, 1024, dtype=torch.float64, device=device) for _ in range(n_layers)]
    S1 = [torch.zeros(1024, dtype=torch.float64, device=device) for _ in range(n_layers)]
    mask = [None]; ntok = [0]
    def mk(l):
        def h(mod, inp, out):
            x = inp[0][mask[0]].double()
            C[l].add_(x.T @ x)
            S1[l].add_(x.sum(0))
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
    return [c.cpu() for c in C], [s.cpu() for s in S1], ntok[0]


def ln_bias(model, l):
    m = model.roberta.embeddings.LayerNorm if l == 0 else model.roberta.encoder.layer[l - 1].output.LayerNorm
    return m.bias.detach().cpu().double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="LABEL=PATH or LABEL=PATH@EPOCH")
    ap.add_argument("--r_grid", default="0.9999,0.999,0.995,0.99,0.98,0.95,0.9,0.8,0.7,0.5")
    ap.add_argument("--out_dir", default="logs/kdiag_offset")
    ap.add_argument("--kdiag_dir", default="logs/kdiag", help="kdiag_eval.py output to check against")
    ap.add_argument("--hf", default="roberta-large")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    R_GRID = [float(x) for x in args.r_grid.split(",")]

    plan, problems = [], []
    if not os.path.exists("kdiag_eval.py"):
        problems.append("kdiag_eval.py is not in this directory (its functions define the truncations)")
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
        ref = os.path.join(args.kdiag_dir, f"{label}.npz")
        if not os.path.exists(ref):
            problems.append(f"{label}: no {ref} to check against (run kdiag_eval.py first)")
        print(f"[PLAN] {label} <- {ckpt}  epoch_saved={ep}  kdiag reference: {ref}")
        plan.append((label, ckpt, ref))
    if problems:
        for x in problems:
            print(f"[CHECK] FAIL {x}")
        sys.exit(1)
    print(f"[CHECK] all {len(plan)} models and kdiag references found; r grid {R_GRID}")
    if args.dry:
        sys.exit(0)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import bias_probe
    import kdiag_eval as KE
    from repro_utils import assert_env, assert_gpu
    assert_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        assert_gpu()
    tok = AutoTokenizer.from_pretrained(args.hf)
    tf = lambda ex: tok(ex["sentence"], truncation=True, max_length=None)
    cf = lambda ex: tok.pad(ex, padding="longest", return_tensors="pt")
    TR = bias_probe.class_loaders("./data/sst-2/train.json", load_dataset, tf, cf, args.batch_size)
    print(f"[DATA] SST-2 train {TR['n_pos']}/{TR['n_neg']} (labels unused); SST-2 dev not used")

    os.makedirs(args.out_dir, exist_ok=True)
    for label, ckpt, ref in plan:
        t0 = time.time()
        Z = np.load(ref)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf, return_dict=True)
        missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        model.to(device)
        nL = len(model.roberta.encoder.layer)
        Cs, S1s, n = collect_C_mu(model, [TR["pos"], TR["neg"]], device, nL)
        print(f"[OFFSET] {label} C and sum x collected from {n} SST-2 train tokens  {(time.time()-t0)/60:.1f} min", flush=True)
        ok_n = int(Z["n_tokens"]) == n
        print(f"[CHECK] {label} token count {n} vs kdiag {int(Z['n_tokens'])}: {'same' if ok_n else 'DIFFERS'}", flush=True)
        if not ok_n:
            sys.exit(1)
        out = {"n_tokens": n, "r_grid": np.array(R_GRID)}
        per_layer = {k: [] for k in ("mu", "beta", "trM2", "mean_share", "top_share", "top_mean", "cos_top_mu",
                                     "top_share_S", "lamS", "cos_mu_beta", "beta_over_mu", "top_dims", "top_dims_share")}
        mods, MS, MU, SV, LAY = [], [], [], [], []
        worst_lam = 0.0
        for l in range(nL):
            C = Cs[l]; mu = S1s[l] / n; M2 = C / n
            Sv = M2 - torch.outer(mu, mu); Sv = (Sv + Sv.T) / 2
            lam, Q, lam_f, nfl, tol = KE.eig_C(C)
            ref_lam = torch.from_numpy(Z["lam"][l])
            worst_lam = max(worst_lam, float((lam - ref_lam).abs().max() / ref_lam[0]))
            trM2 = float(torch.trace(M2)); mu2 = float(mu @ mu)
            u1 = Q[:, 0]; lam1 = float(lam[0]) / n
            lamS = torch.linalg.eigvalsh(Sv).flip(0)
            beta = ln_bias(model, l)
            d = torch.argsort(mu ** 2, descending=True)[:3]
            vals = dict(mu=mu.numpy(), beta=beta.numpy(), trM2=trM2, mean_share=mu2 / trM2, top_share=lam1 / trM2,
                        top_mean=float(u1 @ mu) ** 2 / lam1, cos_top_mu=abs(float(u1 @ mu)) / mu2 ** 0.5,
                        top_share_S=float(lamS[0] / lamS.sum()), lamS=lamS.numpy(),
                        cos_mu_beta=float(mu @ beta) / (mu2 ** 0.5 * float(beta.norm()) + 1e-300),
                        beta_over_mu=float(beta @ beta) / mu2, top_dims=d.numpy(),
                        top_dims_share=(mu[d] ** 2 / mu2).numpy())
            for k, v in vals.items():
                per_layer[k].append(v)
            print(f"[OFFSET] {label} layer {l}: constant part {100*vals['mean_share']:.1f}% of input energy; top direction "
                  f"{100*vals['top_share']:.1f}% (of which constant {100*vals['top_mean']:.1f}%, cos with mean "
                  f"{vals['cos_top_mu']:.3f}); after removing the mean, top direction {100*vals['top_share_S']:.1f}% of "
                  f"the varying part; cos(mean, LayerNorm bias) {vals['cos_mu_beta']:.3f}, |bias|^2/|mean|^2 "
                  f"{vals['beta_over_mu']:.3f}; largest mean coordinates {d.tolist()} "
                  f"({', '.join(f'{100*x:.1f}%' for x in vals['top_dims_share'])})", flush=True)
            for kind in ("query", "value"):
                W = getattr(model.roberta.encoder.layer[l].attention.self, kind).weight.detach().cpu().double()
                M = KE.analyze_module(W, C, Q, lam_f)
                mods.append(f"layer{l}.{kind}"); MS.append(M); MU.append(mu); SV.append(Sv); LAY.append(l)
        print(f"[CHECK] {label} eigenvalues of C vs kdiag: max |difference| / largest eigenvalue {worst_lam:.1e}", flush=True)
        if worst_lam > 1e-6:
            print(f"[CHECK] FAIL {label}: C is not the one kdiag used"); sys.exit(1)
        if list(Z["modules"]) != mods:
            print(f"[CHECK] FAIL {label}: module order differs from kdiag"); sys.exit(1)
        for k, v in per_layer.items():
            out[k] = np.array(v)
        out["modules"] = np.array(mods)
        same_r = list(Z["r_grid"]) == R_GRID
        for m in METHODS:
            K, OA, OV, OM = [], [], [], []
            worst_id = worst_ref = 0.0
            for r in R_GRID:
                ks, oa, ov, om = [], [], [], []
                for M, mu, Sv, l in zip(MS, MU, SV, LAY):
                    k = KE.k_for(M["curves"][m], r)
                    _, Rb = KE.truncated(M, m, k)          # orthonormal basis of the truncated W's row space
                    ks.append(k)
                    C = Cs[l]
                    a_all = 1.0 - float(torch.trace(Rb.T @ C @ Rb)) / float(torch.trace(C))
                    a_var = 1.0 - float(torch.trace(Rb.T @ Sv @ Rb)) / float(torch.trace(Sv))
                    pm = Rb.T @ mu
                    a_mu = 1.0 - float(pm @ pm) / float(mu @ mu)
                    lhs = a_all * float(torch.trace(C)) / n
                    rhs = a_var * float(torch.trace(Sv)) + a_mu * float(mu @ mu)
                    worst_id = max(worst_id, abs(lhs - rhs) / (float(torch.trace(C)) / n))
                    oa.append(a_all); ov.append(a_var); om.append(a_mu)
                K.append(ks); OA.append(oa); OV.append(ov); OM.append(om)
                print(f"[OFFSET] {label} {m} r={r} kept rank mean {np.mean(ks):.1f}; outside the row space (median over "
                      f"{len(ks)} modules): all energy {np.median(oa):.3f}, varying part {np.median(ov):.3f}, "
                      f"constant part {np.median(om):.3f}", flush=True)
            K = np.array(K); OA = np.array(OA)
            if same_r:
                k_ok = np.array_equal(K, Z[f"k_{m}"])
                worst_ref = float(np.abs(OA - Z[f"outside_{m}"]).max())
                print(f"[CHECK] {label} {m}: k per module {'identical to' if k_ok else 'DIFFERENT from'} kdiag; "
                      f"max |outside_all - kdiag outside| {worst_ref:.1e}; max identity error {worst_id:.1e}", flush=True)
                if not k_ok or worst_ref > 1e-8:
                    print(f"[CHECK] FAIL {label} {m}: truncation is not the one kdiag used"); sys.exit(1)
            else:
                print(f"[CHECK] {label} {m}: r grid differs from kdiag, k not compared; max identity error {worst_id:.1e}")
            if worst_id > 1e-8:
                print(f"[CHECK] FAIL {label} {m}: energy split does not add up"); sys.exit(1)
            out[f"k_{m}"] = K; out[f"outside_all_{m}"] = OA
            out[f"outside_var_{m}"] = np.array(OV); out[f"outside_mu_{m}"] = np.array(OM)
        np.savez_compressed(os.path.join(args.out_dir, f"{label}.npz"), **{k: np.asarray(v) for k, v in out.items()})
        print(f"[SAVE] {args.out_dir}/{label}.npz  missing={len(missing)} unexpected={len(unexpected)}  "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        del model, MS, Cs; torch.cuda.empty_cache()
    print("[CHECK] done")


if __name__ == "__main__":
    main()