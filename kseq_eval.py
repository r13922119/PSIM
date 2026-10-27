#!/usr/bin/env python3
"""kseq_eval.py (v2, 10/27 evening) -- layer-by-layer ("sequential") truncation of the 48 query/value W_pre, compared
with kdiag's simultaneous truncation. Diagnostic only: no k is chosen, no saved weight is changed, nothing is fine-tuned.

v2 vs v1: v1 ran only sigma and whiten at r = 0.999 ... 0.7, so the summary figure could not compare every setting
(the user, 10/27 17:40). v2 defaults to ALL four orderings at ALL ten kdiag r values, and every pass also records the
constant / varying split (below), so every panel of kall.png has every setting. Outputs go to logs/kseq_full/. For the
(ordering, r) cells v1 already ran, v2 must reproduce v1's k and SST-2 dev scores exactly (checked).

The user's idea (10/27): start at layer 0; compute C from the inputs layer 0 actually receives, pick each module's k with
the module's r, truncate; then layer 1 receives inputs that already went through the truncated layer 0, so its C is
computed from those; and so on up to layer 23. kdiag computed every C from the untruncated PZ and truncated all 48 at once.
Same idea as GPTQ (Frantar et al., ICLR 2023), which quantizes block by block on "the actual layer inputs in the already
partially quantized" model.

What stays exactly as in kdiag (functions imported from kdiag_eval.py): C = sum over SST-2 TRAIN tokens of x x^T
(padding excluded, no labels, float64), the four orderings, the eigenvalue floor, R(k), k = smallest k with R(k) >= r,
the truncated W. Only the inputs that build C change. SST-2 DEV only scores the model after all 48 modules are truncated
(eval mode, the PZ's own head); weights are restored after every (ordering, r).

Constant / varying split (kdiag_offset, 10/27): every input x = mu + (x - mu), mu = average input over all SST-2 train
tokens of that layer (the same for every token). Energies split exactly because the varying parts sum to zero:
  for any matrix D:  sum_x ||D x||^2 = sum_x ||D (x - mu)||^2 + n ||D mu||^2.
Recorded per module, for the inputs that module actually receives:
  Tr check (input side; RoRA's penalty ||A V||^2 makes Delta W blind to the input directions W_k reads):
    outside_all / outside_var / outside_mu = share of the input energy (all / varying part / constant part) that lies
    OUTSIDE the row space of the truncated W, i.e. what Delta W can still read.
  output kept: ret_all / ret_var / ret_const = 1 - sum ||W x - W_k x||^2 / sum ||W x||^2, counting all / only the varying
    part / only the constant part. ret_all is r (up to the step from k being an integer).
The untruncated reference pass also saves, for all four orderings, the full curves Rvar_<m>, Rconst_<m> (k = 1..1024) on
the untruncated inputs, which give the simultaneous truncation's split at kdiag's k.
Per layer (24 values per pass; the sequential version of kdiag_offset_layers.png), on the inputs the layer actually receives:
  L_mean_share = ||mu||^2 / average ||x||^2;  L_top_share = largest eigenvalue / trace of the average x x^T;
  L_top_mean = share of that top direction's energy that is the constant part;  L_top_share_S = largest eigenvalue /
  trace of the covariance (mean removed). At layer 0 these must equal logs/kdiag_offset (checked).

How it runs: one untruncated forward over SST-2 train keeps, for every batch, the input of layer 0 and the exact
arguments the encoder passes to each layer (attention mask etc.), so layers can then be run one at a time on those
stored inputs. Checks (any failure stops):
  - untruncated SST-2 dev scores equal logs/scores/sst2dev/<label>.npz (PZs) exactly;
  - running the 24 layers one at a time with the original weights reproduces the last layer's output of the normal
    forward, and each layer's C has the same eigenvalues as logs/kdiag/<label>.npz;
  - the cheap split formulas equal a direct computation with the varying and constant second moments (reference pass);
  - in every sequential pass, layer 0 (whose input no truncation can change) gets exactly kdiag's k;
  - every (ordering, r) that v1 ran (logs/kseq/) is reproduced: same k for all 48 modules, identical SST-2 dev scores.

Output: <out_dir>/<label>.npz and log lines [KSEQ], [CHECK]. numpy + torch only (no scipy on the workstation).
  python kseq_eval.py --dry --model LABEL=PATH[@EPOCH] ...      # no GPU: file checks only
  normally through run_kseq_full.sh (one card, CMLab rule). Needs kdiag_eval.py and logs/kdiag/*.npz.
"""
import argparse, json, os, sys, time
import numpy as np

ALL_METHODS = ("sigma", "vote_in", "vote_out", "whiten")


def split_call(args, kwargs):
    """the hidden states and everything else the encoder passed to a layer"""
    if "hidden_states" in kwargs:
        kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        return kwargs["hidden_states"], args, kw, False
    return args[0], tuple(args[1:]), dict(kwargs), True


def call_layer(layer, b, h):
    out = layer(h, *b["args"], **b["kw"]) if b["pos"] else layer(*b["args"], hidden_states=h, **b["kw"])
    return out[0] if isinstance(out, tuple) else out


def split_curves(M, W, mu, n, m):
    """retained fraction of the CONSTANT output (n ||W mu||^2) and of the VARYING output, for k = 1..1024, for ordering m.
    Uses: error_all(k) = error_var(k) + error_const(k) (exact), and the constant part is rank one (cheap)."""
    import torch
    Wmu = W @ mu
    Ec = n * float(Wmu @ Wmu); Ea = M["E"]; Ev = Ea - Ec
    if m == "whiten":
        Wm = M["Linv"].T @ M["Vw"]                         # W_k = sum_{i<=k} sw_i u~_i w_i^T
        c = M["sw"] * (Wm.T @ mu)                          # W_k mu = U~[:, :k] c[:k]
        g = M["Uw"].T @ Wmu
        err_c = n * (float(Wmu @ Wmu) - 2 * torch.cumsum(c * g, 0) + torch.cumsum(c ** 2, 0))
        Rc = 1 - err_c / Ec
    else:
        o = M["orders"][m]
        ec = n * M["s"] ** 2 * (M["V"].T @ mu) ** 2        # constant output energy per own direction
        Rc = torch.cumsum(ec[o], 0) / Ec
    Ra = M["curves"][m]
    Rv = 1 - (Ea * (1 - Ra) - Ec * (1 - Rc)) / Ev
    return Rv, Rc, Ev, Ec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="LABEL=PATH or LABEL=PATH@EPOCH")
    ap.add_argument("--methods", default=",".join(ALL_METHODS), help="among sigma,vote_in,vote_out,whiten")
    ap.add_argument("--r_grid", default="0.9999,0.999,0.995,0.99,0.98,0.95,0.9,0.8,0.7,0.5")
    ap.add_argument("--out_dir", default="logs/kseq_full")
    ap.add_argument("--kdiag_dir", default="logs/kdiag")
    ap.add_argument("--v1_dir", default="logs/kseq", help="v1 output to reproduce where the settings overlap")
    ap.add_argument("--ref_dir", default="logs/scores/sst2dev")
    ap.add_argument("--hf", default="roberta-large")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    R_GRID = [float(x) for x in args.r_grid.split(",")]
    METHODS = tuple(args.methods.split(","))
    assert all(m in ALL_METHODS for m in METHODS), METHODS

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
        kref = os.path.join(args.kdiag_dir, f"{label}.npz")
        if not os.path.exists(kref):
            problems.append(f"{label}: no {kref} (run kdiag_eval.py first)")
        else:
            missing_r = [r for r in R_GRID if r not in list(np.load(kref)["r_grid"])]
            if missing_r:
                problems.append(f"{label}: r {missing_r} not in kdiag's grid, cannot compare")
        ref = os.path.join(args.ref_dir, f"{label}.npz")
        v1 = os.path.join(args.v1_dir, f"{label}.npz")
        print(f"[PLAN] {label} <- {ckpt}  epoch_saved={ep}  kdiag: {kref}  dev reference: {ref if os.path.exists(ref) else 'none'}  "
              f"v1: {v1 if os.path.exists(v1) else 'none'}")
        plan.append((label, ckpt, kref, ref if os.path.exists(ref) else None, v1 if os.path.exists(v1) else None))
    if problems:
        for x in problems:
            print(f"[CHECK] FAIL {x}")
        sys.exit(1)
    print(f"[CHECK] all {len(plan)} models found; orderings {METHODS}; r grid {R_GRID}; "
          f"{len(METHODS) * len(R_GRID)} sequential passes per model")
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
    DV = bias_probe.class_loaders("./data/sst-2/dev.json", load_dataset, tf, cf, args.batch_size)
    print(f"[DATA] SST-2 train {TR['n_pos']}/{TR['n_neg']} (for C only, labels unused); SST-2 dev {DV['n_pos']}/{DV['n_neg']}")

    os.makedirs(args.out_dir, exist_ok=True)
    for label, ckpt, kref, ref, v1p in plan:
        t0 = time.time()
        Z = np.load(kref); KR = list(Z["r_grid"])
        V1 = np.load(v1p) if v1p else None
        offp = os.path.join(os.path.dirname(args.kdiag_dir.rstrip("/")) or ".", "kdiag_offset", f"{label}.npz")
        OFF = np.load(offp) if os.path.exists(offp) else None
        print(f"[KSEQ] {label} layer-0 offset check against {offp if OFF is not None else 'nothing (file missing)'}", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf, return_dict=True)
        missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        model.to(device); model.eval()
        layers = model.roberta.encoder.layer; nL = len(layers)
        mods = [f"layer{l}.{kind}" for l in range(nL) for kind in ("query", "value")]
        assert list(Z["modules"]) == mods, "module order differs from kdiag"
        lins = {(l, kind): getattr(layers[l].attention.self, kind) for l in range(nL) for kind in ("query", "value")}
        W0 = {key: lin.weight.detach().clone() for key, lin in lins.items()}
        Wd = {key: w.cpu().double() for key, w in W0.items()}
        out = {"r_grid": np.array(R_GRID), "methods": np.array(METHODS), "modules": np.array(mods)}

        # ---- 0. untruncated SST-2 dev scores (identity check) ----
        _, bp, bn = bias_probe.class_probe(model, DV, device, keep=True)
        out["base_pos"], out["base_neg"] = bp, bn
        if ref is not None:
            z = np.load(ref)
            worst = max(float(np.abs(bp - z["sdev_pos"]).max()), float(np.abs(bn - z["sdev_neg"]).max()))
            print(f"[CHECK] {label} untruncated SST-2 dev vs {ref}: {'identical' if worst == 0 else 'DIFFERS'} (max difference {worst:.3g})", flush=True)
            if worst != 0:
                sys.exit(1)

        # ---- 1. one untruncated forward over SST-2 train: keep layer 0's input and call, and the last layer's output ----
        store, cur = [], {}
        def pre(mod, a, kw):
            cur["call"] = split_call(a, kw)
        def post(mod, a, kw, o):
            cur["last"] = o[0] if isinstance(o, tuple) else o
        h1 = layers[0].register_forward_pre_hook(pre, with_kwargs=True)
        h2 = layers[-1].register_forward_hook(post, with_kwargs=True)
        with torch.no_grad():
            for loader in (TR["pos"], TR["neg"]):       # same order as kdiag_eval.collect_C
                for batch in loader:
                    batch = batch.to(device)
                    model(**{k: v for k, v in batch.items() if k != "labels"})
                    h, a, kw, pos = cur["call"]
                    store.append(dict(h=h.detach().clone(), args=a, kw=kw, pos=pos, mask=batch["attention_mask"].bool(),
                                      last=cur["last"].detach().clone()))
        h1.remove(); h2.remove()
        print(f"[KSEQ] {label} stored layer-0 inputs of {len(store)} SST-2 train batches  {(time.time()-t0)/60:.1f} min", flush=True)

        def propagate(per_layer):
            """run the layers one at a time on the stored inputs; before layer l runs, per_layer(l, C, s1, n) sees the
            inputs layer l actually receives (and may change its query/value weights)"""
            H = [b["h"] for b in store]
            with torch.no_grad():
                for l in range(nL):
                    C = torch.zeros(1024, 1024, dtype=torch.float64, device=device)
                    s1 = torch.zeros(1024, dtype=torch.float64, device=device); n = 0
                    for b, h in zip(store, H):
                        x = h[b["mask"]].double()
                        C.add_(x.T @ x); s1.add_(x.sum(0)); n += x.shape[0]
                    per_layer(l, C.cpu(), s1.cpu(), n)
                    if l < nL - 1 or per_layer is ref_layer:
                        H = [call_layer(layers[l], b, h) for b, h in zip(store, H)]
            return H

        # ---- 2. reference pass with the original weights: identity checks + split curves for all orderings ----
        R = {k: [] for k in ["E_all", "E_var", "E_const"] + [f"R{p}_{m}" for p in ("var", "const") for m in ALL_METHODS]}
        chk = {"lam": 0.0, "split": 0.0, "E": 0.0, "cheap": 0.0}
        def ref_layer(l, C, s1, n):
            lam, Q, lam_f, nfl, tol = KE.eig_C(C)
            ref_lam = torch.from_numpy(Z["lam"][l])
            chk["lam"] = max(chk["lam"], float((lam - ref_lam).abs().max() / ref_lam[0]))
            mu = s1 / n
            Sv = C - n * torch.outer(mu, mu); Sv = (Sv + Sv.T) / 2      # varying part (sum over tokens)
            Cc = n * torch.outer(mu, mu)                                 # constant part
            for kind in ("query", "value"):
                W = Wd[(l, kind)]
                M = KE.analyze_module(W, C, Q, lam_f)
                i = mods.index(f"layer{l}.{kind}")
                chk["E"] = max(chk["E"], abs(M["E"] - float(Z["E"][i])) / float(Z["E"][i]))
                for m in ALL_METHODS:
                    Rv, Rc, Ev, Ec = split_curves(M, W, mu, n, m)
                    R[f"Rvar_{m}"].append(Rv.numpy()); R[f"Rconst_{m}"].append(Rc.numpy())
                    chk["split"] = max(chk["split"], float((M["E"] * M["curves"][m] - Ev * Rv - Ec * Rc).abs().max()) / M["E"])
                # direct computation with the two second moments (sigma and whiten, as in v1), against the cheap formulas
                Wm = M["Linv"].T @ M["Vw"]
                rv_s, rc_s, _, _ = split_curves(M, W, mu, n, "sigma")
                rv_w, rc_w, _, _ = split_curves(M, W, mu, n, "whiten")
                for name, P, got_s, got_w in (("var", Sv, rv_s, rv_w), ("const", Cc, rc_s, rc_w)):
                    Ep = float(torch.trace(W @ P @ W.T))
                    rs = torch.cumsum(M["s"] ** 2 * ((M["V"].T @ P) * M["V"].T).sum(1), 0) / Ep
                    outE = M["sw"] ** 2 * ((Wm.T @ P) * Wm.T).sum(1)
                    cross = M["sw"] * ((M["Uw"].T @ (W @ P)) * Wm.T).sum(1)
                    rw = 1 - (Ep - 2 * torch.cumsum(cross, 0) + torch.cumsum(outE, 0)) / Ep
                    chk["cheap"] = max(chk["cheap"], float((got_s - rs).abs().max()), float((got_w - rw).abs().max()))
                Wmu = W @ mu
                R["E_all"].append(M["E"]); R["E_const"].append(n * float(Wmu @ Wmu)); R["E_var"].append(M["E"] - n * float(Wmu @ Wmu))
        H = propagate(ref_layer)
        last = max(float((h - b["last"]).abs().max()) for h, b in zip(H, store))
        print(f"[CHECK] {label} layers run one at a time with original weights vs normal forward, last layer output: "
              f"max difference {last:.3g}; eigenvalues of each layer's C vs kdiag: max |diff|/largest {chk['lam']:.1e}; "
              f"module output energy vs kdiag: max relative diff {chk['E']:.1e}; all = varying + constant: max error "
              f"{chk['split']:.1e}; cheap split vs direct: max difference {chk['cheap']:.1e}  {(time.time()-t0)/60:.1f} min", flush=True)
        if last > 1e-4 or chk["lam"] > 1e-6 or chk["E"] > 1e-9 or chk["split"] > 1e-8 or chk["cheap"] > 1e-8:
            print(f"[CHECK] FAIL {label}: reference pass checks failed"); sys.exit(1)
        for k, v in R.items():
            out[k] = np.array(v)
        print(f"[KSEQ] {label} constant share of the module output energy: median {np.median(out['E_const'] / out['E_all']):.3f}", flush=True)

        # ---- 3. sequential passes ----
        for m in METHODS:
            store_m = {k: [] for k in ("k", "exact_err", "outside_all", "outside_var", "outside_mu", "ret_all", "ret_var",
                                       "ret_const", "L_mean_share", "L_top_share", "L_top_mean", "L_top_share_S",
                                       "n_floored", "dev_pos", "dev_neg")}
            for r in R_GRID:
                rec = {k: [] for k in ("k", "exact_err", "outside_all", "outside_var", "outside_mu", "ret_all", "ret_var", "ret_const",
                                       "L_mean_share", "L_top_share", "L_top_mean", "L_top_share_S")}
                rec["n_floored"] = 0
                def trunc_layer(l, C, s1, n):
                    lam, Q, lam_f, nfl, tol = KE.eig_C(C)
                    rec["n_floored"] += nfl
                    mu = s1 / n; mu2 = float(mu @ mu); trC = float(torch.trace(C))
                    # this layer's input, as kdiag_offset measured it on the untruncated model (per layer, not per module)
                    Sv = C / n - torch.outer(mu, mu); lamS = torch.linalg.eigvalsh((Sv + Sv.T) / 2)
                    lam1 = float(lam[0]) / n
                    rec["L_mean_share"].append(mu2 / (trC / n)); rec["L_top_share"].append(lam1 / (trC / n))
                    rec["L_top_mean"].append(float(Q[:, 0] @ mu) ** 2 / lam1); rec["L_top_share_S"].append(float(lamS[-1] / lamS.sum()))
                    for kind in ("query", "value"):
                        W = Wd[(l, kind)]
                        M = KE.analyze_module(W, C, Q, lam_f)
                        k = KE.k_for(M["curves"][m], r)
                        Wk, Rb = KE.truncated(M, m, k)
                        inside = float(torch.trace(Rb.T @ C @ Rb)); pm = Rb.T @ mu; inside_mu = float(pm @ pm)
                        Rv, Rc, Ev, Ec = split_curves(M, W, mu, n, m)
                        rec["k"].append(k); rec["exact_err"].append(KE.exact_err(W, Wk, C, M["E"]))
                        rec["outside_all"].append(1 - inside / trC)
                        rec["outside_var"].append(1 - (inside - n * inside_mu) / (trC - n * mu2))
                        rec["outside_mu"].append(1 - inside_mu / mu2)
                        rec["ret_all"].append(float(M["curves"][m][k - 1]))
                        rec["ret_var"].append(float(Rv[k - 1])); rec["ret_const"].append(float(Rc[k - 1]))
                        with torch.no_grad():
                            lins[(l, kind)].weight.copy_(Wk.to(W0[(l, kind)].dtype).to(device))
                propagate(trunc_layer)
                kd = Z[f"k_{m}"][KR.index(r)]
                if OFF is not None:
                    gap = max(abs(rec["L_mean_share"][0] - float(OFF["mean_share"][0])), abs(rec["L_top_share"][0] - float(OFF["top_share"][0])),
                              abs(rec["L_top_share_S"][0] - float(OFF["top_share_S"][0])))
                    if gap > 1e-9:
                        print(f"[CHECK] FAIL {label} {m} r={r}: layer 0 offset numbers differ from kdiag_offset by {gap:.1e}"); sys.exit(1)
                if rec["k"][:2] != [int(kd[0]), int(kd[1])]:
                    print(f"[CHECK] FAIL {label} {m} r={r}: layer 0 k {rec['k'][:2]} differs from kdiag {kd[:2].tolist()}"); sys.exit(1)
                _, sp, sn = bias_probe.class_probe(model, DV, device, keep=True)
                for key, lin in lins.items():
                    with torch.no_grad():
                        lin.weight.copy_(W0[key])
                if V1 is not None and m in list(V1["methods"]) and r in list(V1["r_grid"]):
                    j = list(V1["r_grid"]).index(r)
                    same_k = list(V1[f"k_{m}"][j]) == rec["k"]
                    same_s = np.array_equal(V1[f"dev_pos_{m}"][j], sp) and np.array_equal(V1[f"dev_neg_{m}"][j], sn)
                    print(f"[CHECK] {label} {m} r={r} vs v1: k {'identical' if same_k else 'DIFFERENT'}, "
                          f"SST-2 dev scores {'identical' if same_s else 'DIFFERENT'}", flush=True)
                    if not (same_k and same_s):
                        sys.exit(1)
                for k, v in rec.items():
                    store_m[k].append(v)
                store_m["dev_pos"].append(sp); store_m["dev_neg"].append(sn)
                a_seq = KE.auc(sp, sn)
                a_sim = KE.auc(Z[f"dev_pos_{m}"][KR.index(r)].astype(float), Z[f"dev_neg_{m}"][KR.index(r)].astype(float))
                print(f"[KSEQ] {label} {m} r={r} kept rank mean {np.mean(rec['k']):.1f} (simultaneous {kd.mean():.1f}); "
                      f"SST-2 dev AUC sequential {a_seq:.4f} simultaneous {a_sim:.4f} untruncated {KE.auc(bp, bn):.4f}; "
                      f"outside (all/varying/constant, median) {np.median(rec['outside_all']):.3f}/{np.median(rec['outside_var']):.3f}/"
                      f"{np.median(rec['outside_mu']):.3f}; eigenvalues floored {rec['n_floored']}  {(time.time()-t0)/60:.1f} min", flush=True)
            for k, v in store_m.items():
                out[f"{k}_{m}"] = np.stack(v) if k in ("dev_pos", "dev_neg") else np.array(v)
        _, bp2, bn2 = bias_probe.class_probe(model, DV, device, keep=True)
        assert np.array_equal(bp2, bp) and np.array_equal(bn2, bn), "weights were not restored"
        np.savez_compressed(os.path.join(args.out_dir, f"{label}.npz"), **{k: np.asarray(v) for k, v in out.items()})
        print(f"[SAVE] {args.out_dir}/{label}.npz  missing={len(missing)} unexpected={len(unexpected)}  {(time.time()-t0)/60:.1f} min", flush=True)
        del model, store; torch.cuda.empty_cache()
    print("[CHECK] done")


if __name__ == "__main__":
    main()