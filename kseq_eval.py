#!/usr/bin/env python3
"""kseq_eval.py -- layer-by-layer ("sequential") truncation of the 48 query/value W_pre, compared with kdiag's
simultaneous truncation. Diagnostic only: no k is chosen, no saved weight is changed, nothing is fine-tuned.

The user's idea (10/27): start at layer 0; compute C from the inputs layer 0 actually receives, pick each module's k with
the module's r, truncate; then layer 1 receives inputs that already went through the truncated layer 0, so its C is
computed from those; and so on up to layer 23. kdiag computed every C from the untruncated PZ and truncated all 48 at once.
Same idea as GPTQ (Frantar et al., ICLR 2023), which quantizes block by block on "the actual layer inputs in the already
partially quantized" model.

What stays exactly as in kdiag (functions imported from kdiag_eval.py): C = sum over SST-2 TRAIN tokens of x x^T
(padding excluded, no labels, float64), the four orderings, the eigenvalue floor, R(k), k = smallest k with R(k) >= r,
the truncated W. Only the inputs that build C change. Each module still preserves its OWN output on the inputs it
actually receives; it does not try to undo errors of earlier layers.
SST-2 DEV only scores the model after all 48 modules are truncated (eval mode, the PZ's own head); weights are restored
after every (ordering, r).

How it runs: one untruncated forward over SST-2 train keeps, for every batch, the input of layer 0 and the exact
arguments the encoder passes to each layer (attention mask etc.), so layers can then be run one at a time on those
stored inputs. Checks before anything is truncated (any failure stops):
  - untruncated SST-2 dev scores equal logs/scores/sst2dev/<label>.npz (PZs) exactly;
  - running the 24 layers one at a time with the original weights reproduces the last layer's output of the normal
    forward, and each layer's C has the same eigenvalues as logs/kdiag/<label>.npz;
  - in every sequential pass, layer 0 (whose input no truncation can change) gets exactly kdiag's k.

Also measured in that untruncated pass, for the ruler question (10/27: C includes the constant part mu, the average
input, which kdiag_offset found to be ~2/3 of the input energy): per module, the share of the output energy W x that is
constant (n ||W mu||^2 / tr(W C W^T)), and for the sigma and whiten orderings the full curves of how much of the VARYING
output (x - mu) and of the CONSTANT output (mu) the kdiag truncation keeps at each k. Retained energy splits exactly:
E_all R_all(k) = E_var R_var(k) + E_const R_const(k) (checked against kdiag's curves).

Output: <out_dir>/<label>.npz and log lines [KSEQ], [CHECK]. numpy + torch only (no scipy on the workstation).
  python kseq_eval.py --dry --model LABEL=PATH[@EPOCH] ...      # no GPU: file checks only
  normally through run_kseq.sh (one card, CMLab rule). Needs kdiag_eval.py and logs/kdiag/*.npz.
"""
import argparse, json, os, sys, time
import numpy as np


def split_call(args, kwargs):
    """the hidden states and everything else the encoder passed to a layer"""
    if "hidden_states" in kwargs:
        kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        return kwargs["hidden_states"], args, kw, False
    return args[0], tuple(args[1:]), dict(kwargs), True


def call_layer(layer, b, h):
    out = layer(h, *b["args"], **b["kw"]) if b["pos"] else layer(*b["args"], hidden_states=h, **b["kw"])
    return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="LABEL=PATH or LABEL=PATH@EPOCH")
    ap.add_argument("--methods", default="sigma,whiten", help="among sigma,vote_in,vote_out,whiten")
    ap.add_argument("--r_grid", default="0.999,0.995,0.99,0.98,0.95,0.9,0.8,0.7")
    ap.add_argument("--out_dir", default="logs/kseq")
    ap.add_argument("--kdiag_dir", default="logs/kdiag")
    ap.add_argument("--ref_dir", default="logs/scores/sst2dev")
    ap.add_argument("--hf", default="roberta-large")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    R_GRID = [float(x) for x in args.r_grid.split(",")]
    METHODS = tuple(args.methods.split(","))
    assert all(m in ("sigma", "vote_in", "vote_out", "whiten") for m in METHODS), METHODS

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
        print(f"[PLAN] {label} <- {ckpt}  epoch_saved={ep}  kdiag: {kref}  dev reference: {ref if os.path.exists(ref) else 'none'}")
        plan.append((label, ckpt, kref, ref if os.path.exists(ref) else None))
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
    for label, ckpt, kref, ref in plan:
        t0 = time.time()
        Z = np.load(kref); KR = list(Z["r_grid"])
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
                    H = [call_layer(layers[l], b, h) for b, h in zip(store, H)]
            return H

        # ---- 2. reference pass with the original weights: identity checks + the ruler question ----
        R = {k: [] for k in ("E_all", "E_var", "E_const", "Rvar_sigma", "Rvar_whiten", "Rconst_sigma", "Rconst_whiten")}
        chk = {"lam": 0.0, "split": 0.0, "E": 0.0}
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
                Wm = M["Linv"].T @ M["Vw"]
                parts = {}
                for name, P in (("var", Sv), ("const", Cc)):
                    Ep = float(torch.trace(W @ P @ W.T))
                    # sigma ordering (W's own directions, no cross terms): retained = cumsum(s_i^2 v_i^T P v_i)
                    rs = torch.cumsum(M["s"] ** 2 * ((M["V"].T @ P) * M["V"].T).sum(1), 0) / Ep
                    # whiten: same algebra as kdiag_eval.analyze_module with C replaced by P
                    outE = M["sw"] ** 2 * ((Wm.T @ P) * Wm.T).sum(1)
                    cross = M["sw"] * ((M["Uw"].T @ (W @ P)) * Wm.T).sum(1)
                    rw = 1 - (Ep - 2 * torch.cumsum(cross, 0) + torch.cumsum(outE, 0)) / Ep
                    parts[name] = (Ep, rs, rw)
                Ev, Ec = parts["var"][0], parts["const"][0]
                for m, j in (("sigma", 1), ("whiten", 2)):
                    lhs = M["E"] * M["curves"][m]
                    rhs = Ev * parts["var"][j] + Ec * parts["const"][j]
                    chk["split"] = max(chk["split"], float((lhs - rhs).abs().max()) / M["E"])
                R["E_all"].append(M["E"]); R["E_var"].append(Ev); R["E_const"].append(Ec)
                R["Rvar_sigma"].append(parts["var"][1].numpy()); R["Rvar_whiten"].append(parts["var"][2].numpy())
                R["Rconst_sigma"].append(parts["const"][1].numpy()); R["Rconst_whiten"].append(parts["const"][2].numpy())
        H = propagate(ref_layer)
        last = max(float((h - b["last"]).abs().max()) for h, b in zip(H, store))
        print(f"[CHECK] {label} layers run one at a time with original weights vs normal forward, last layer output: "
              f"max difference {last:.3g}; eigenvalues of each layer's C vs kdiag: max |diff|/largest {chk['lam']:.1e}; "
              f"module output energy vs kdiag: max relative diff {chk['E']:.1e}; "
              f"all = varying + constant split of retained energy: max error {chk['split']:.1e}  {(time.time()-t0)/60:.1f} min", flush=True)
        if last > 1e-4 or chk["lam"] > 1e-6 or chk["E"] > 1e-9 or chk["split"] > 1e-8:
            print(f"[CHECK] FAIL {label}: the layer-by-layer machinery does not reproduce the model"); sys.exit(1)
        for k, v in R.items():
            out[k] = np.array(v)
        print(f"[KSEQ] {label} constant share of the module output energy: median {np.median(out['E_const'] / out['E_all']):.3f} "
              f"(min {np.min(out['E_const'] / out['E_all']):.3f}, max {np.max(out['E_const'] / out['E_all']):.3f})", flush=True)

        # ---- 3. sequential passes ----
        for m in METHODS:
            K, ERR, OUT, NF, P_, N_ = [], [], [], [], [], []
            for r in R_GRID:
                rec = {"k": [], "err": [], "out": [], "nfl": 0}
                def trunc_layer(l, C, s1, n):
                    lam, Q, lam_f, nfl, tol = KE.eig_C(C)
                    rec["nfl"] += nfl
                    for kind in ("query", "value"):
                        W = Wd[(l, kind)]
                        M = KE.analyze_module(W, C, Q, lam_f)
                        k = KE.k_for(M["curves"][m], r)
                        Wk, Rb = KE.truncated(M, m, k)
                        rec["k"].append(k); rec["err"].append(KE.exact_err(W, Wk, C, M["E"]))
                        rec["out"].append(KE.outside_share(Rb, C))
                        with torch.no_grad():
                            lins[(l, kind)].weight.copy_(Wk.to(W0[(l, kind)].dtype).to(device))
                propagate(trunc_layer)
                kd = Z[f"k_{m}"][KR.index(r)]
                if rec["k"][:2] != [int(kd[0]), int(kd[1])]:
                    print(f"[CHECK] FAIL {label} {m} r={r}: layer 0 k {rec['k'][:2]} differs from kdiag {kd[:2].tolist()}"); sys.exit(1)
                _, sp, sn = bias_probe.class_probe(model, DV, device, keep=True)
                for key, lin in lins.items():
                    with torch.no_grad():
                        lin.weight.copy_(W0[key])
                K.append(rec["k"]); ERR.append(rec["err"]); OUT.append(rec["out"]); NF.append(rec["nfl"]); P_.append(sp); N_.append(sn)
                a_seq = KE.auc(sp, sn)
                a_sim = KE.auc(Z[f"dev_pos_{m}"][KR.index(r)].astype(float), Z[f"dev_neg_{m}"][KR.index(r)].astype(float))
                print(f"[KSEQ] {label} {m} r={r} kept rank mean {np.mean(rec['k']):.1f} (simultaneous {kd.mean():.1f}); "
                      f"SST-2 dev AUC sequential {a_seq:.4f} simultaneous {a_sim:.4f} untruncated {KE.auc(bp, bn):.4f}; "
                      f"eigenvalues floored {rec['nfl']}  {(time.time()-t0)/60:.1f} min", flush=True)
            out[f"k_{m}"] = np.array(K); out[f"exact_err_{m}"] = np.array(ERR); out[f"outside_{m}"] = np.array(OUT)
            out[f"n_floored_{m}"] = np.array(NF); out[f"dev_pos_{m}"] = np.stack(P_); out[f"dev_neg_{m}"] = np.stack(N_)
        _, bp2, bn2 = bias_probe.class_probe(model, DV, device, keep=True)
        assert np.array_equal(bp2, bp) and np.array_equal(bn2, bn), "weights were not restored"
        np.savez_compressed(os.path.join(args.out_dir, f"{label}.npz"), **{k: np.asarray(v) for k, v in out.items()})
        print(f"[SAVE] {args.out_dir}/{label}.npz  missing={len(missing)} unexpected={len(unexpected)}  {(time.time()-t0)/60:.1f} min", flush=True)
        del model, store; torch.cuda.empty_cache()
    print("[CHECK] done")


if __name__ == "__main__":
    main()