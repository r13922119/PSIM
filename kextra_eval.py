#!/usr/bin/env python3
"""kextra_eval.py (10/27 night) -- everything about K so far, for the SIMULTANEOUS and the SEQUENTIAL truncation, in one run,
plus three things the user agreed to on 10/27 21:43. Diagnostic only: no k is chosen, no saved weight is changed, nothing
is fine-tuned. Supersedes kseq_eval.py v2 / run_kseq_full.sh: it saves everything that run saves (sequential: same keys
with prefix seq_; reference pass: Rvar_<m>, Rconst_<m>, E_*), and its sequential part reproduces logs/kseq_full or logs/kseq
exactly wherever they exist (checked).

Truncations (functions imported from kdiag_eval.py, unchanged): four orderings (sigma, vote_in, vote_out, whiten), ten r.
  simultaneous = kdiag: every C from the untruncated PZ, all 48 query/value modules truncated at once;
  sequential   = kseq:  layer l's C from the inputs layer l actually receives after layers 0..l-1 were truncated.
C = sum over SST-2 TRAIN tokens of x x^T (padding excluded, float64). SST-2 DEV is only scored.

New per module, for both versions (x = this layer's input on SST-2 train; mu = its average over all tokens; S = sum of
(x - mu)(x - mu)^T; P = orthogonal projector onto the row space of the truncated W, i.e. the inputs W_k reads; Tr's
||A V||^2 half lets Delta W read only the complement I - P):
  top_out[:16]  the 16 largest eigenvalues of (I - P) S (I - P): the varying input energy the best 1, 2, ..., 16
                directions of the complement can read. LoRA rank 8 -> the user's "enough room" question uses the first 8.
  top_all[:16]  the 16 largest eigenvalues of S itself (what an unconstrained rank-8 A could read at best); per layer.
                room_8 = sum(top_out[:8]) / sum(top_all[:8]) is in [0, 1]. It depends on the input data, not on W.
  sb_out        share of the between-class energy outside the read space: S_B = sum over classes c of n_c (mu_c - mu)
                (mu_c - mu)^T (mu_c = average input over the tokens of class-c sentences; SST-2 train labels, which the
                defender has); sb_out = 1 - tr(P S_B) / tr(S_B). For two classes S_B has one direction, along
                mu_positive - mu_negative; the general form also covers more classes.
And per (ordering, r), SST-2 DEV positives with the trigger inserted (the 444 sentences of p_auc_eval / sst2dev_eval,
attack_utils.build_poisoned_test_dataloader(..., "mn")) are scored under the truncation: ASR of type (A) (the PZ's own
head, no fine-tuning) = share of them scored <= 0, i.e. called negative. This is a diagnostic for us; it is not used to
choose anything (the user's 10/25 decision).

Checks (any failure stops):
  - untruncated SST-2 dev scores, clean and triggered, equal logs/scores/sst2dev/<label>.npz exactly (PZs);
  - layer-by-layer run with original weights reproduces the normal forward's last layer output; each layer's C has
    kdiag's eigenvalues; module output energies equal kdiag's;
  - simultaneous: k per module equals kdiag's; clean SST-2 dev scores equal kdiag's;
  - sequential: layer 0 gets kdiag's k; every cell that logs/kseq_full (or logs/kseq) has is reproduced: same k for all
    48 modules and identical clean SST-2 dev scores; per-layer offset numbers at layer 0 equal logs/kdiag_offset;
  - sb_out and the eigenvalue sums are checked against direct formulas on one module per pass.
Output: <out_dir>/<label>.npz; log lines [KX], [CHECK]. numpy + torch only.
  python kextra_eval.py --dry --model LABEL=PATH[@EPOCH] ...
  normally through run_kextra.sh (one card, CMLab rule). Needs kdiag_eval.py, kseq_eval.py (for split_curves etc.),
  logs/kdiag/, logs/kdiag_offset/, logs/scores/sst2dev/.
"""
import argparse, json, os, sys, time
import numpy as np

ALL_METHODS = ("sigma", "vote_in", "vote_out", "whiten")
TOP = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="LABEL=PATH or LABEL=PATH@EPOCH")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--r_grid", default="0.9999,0.999,0.995,0.99,0.98,0.95,0.9,0.8,0.7,0.5")
    ap.add_argument("--out_dir", default="logs/kextra")
    ap.add_argument("--kdiag_dir", default="logs/kdiag")
    ap.add_argument("--seq_dirs", default="logs/kseq_full,logs/kseq", help="earlier sequential outputs to reproduce")
    ap.add_argument("--ref_dir", default="logs/scores/sst2dev")
    ap.add_argument("--hf", default="roberta-large")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    R_GRID = [float(x) for x in args.r_grid.split(",")]
    METHODS = tuple(args.methods.split(","))
    assert all(m in ALL_METHODS for m in METHODS), METHODS
    root = os.path.dirname(args.kdiag_dir.rstrip("/")) or "."

    plan, problems = [], []
    for f in ("kdiag_eval.py", "kseq_eval.py"):
        if not os.path.exists(f):
            problems.append(f"{f} is not in this directory")
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
            problems.append(f"{label}: no {kref}")
        elif [r for r in R_GRID if r not in list(np.load(kref)["r_grid"])]:
            problems.append(f"{label}: r grid not inside kdiag's")
        offp = os.path.join(root, "kdiag_offset", f"{label}.npz")
        ref = os.path.join(args.ref_dir, f"{label}.npz")
        seqs = [os.path.join(d, f"{label}.npz") for d in args.seq_dirs.split(",") if os.path.exists(os.path.join(d, f"{label}.npz"))]
        print(f"[PLAN] {label} <- {ckpt}  epoch_saved={ep}  kdiag: {kref}  offset: {offp if os.path.exists(offp) else 'none'}  "
              f"dev reference: {ref if os.path.exists(ref) else 'none'}  earlier sequential: {seqs or 'none'}")
        plan.append((label, ckpt, kref, offp if os.path.exists(offp) else None, ref if os.path.exists(ref) else None, seqs))
    if problems:
        for x in problems:
            print(f"[CHECK] FAIL {x}")
        sys.exit(1)
    print(f"[CHECK] all {len(plan)} models found; orderings {METHODS}; r grid {R_GRID}; "
          f"{len(METHODS) * len(R_GRID)} simultaneous + {len(METHODS) * len(R_GRID)} sequential truncations per model")
    if args.dry:
        sys.exit(0)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import bias_probe
    from attack_utils import build_poisoned_test_dataloader
    import kdiag_eval as KE
    from kseq_eval import split_call, call_layer, split_curves
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
    TRIG = build_poisoned_test_dataloader("./data/sst-2/dev.json", load_dataset, tf, cf, "mn", batch_size=args.batch_size)
    print(f"[DATA] SST-2 train {TR['n_pos']}/{TR['n_neg']} (C, mu, class means); SST-2 dev {DV['n_pos']}/{DV['n_neg']}; "
          f"dev positives with the trigger {DV['n_pos']}")

    def score(model):
        _, sp, sn = bias_probe.class_probe(model, DV, device, keep=True)
        _, st = bias_probe._asr(model, TRIG, device, keep=True)
        return sp, sn, st

    def extras(Rb, S, SB, topS):
        """top-16 eigenvalues of (I-P) S (I-P), and the between-class share outside; P = Rb Rb^T"""
        PcS = S - Rb @ (Rb.T @ S)
        M = PcS - (PcS @ Rb) @ Rb.T
        ev = torch.linalg.eigvalsh((M + M.T) / 2).flip(0)[:TOP]
        inside_b = float(torch.trace(Rb.T @ SB @ Rb))
        return ev.numpy(), 1 - inside_b / float(torch.trace(SB)), topS

    os.makedirs(args.out_dir, exist_ok=True)
    for label, ckpt, kref, offp, ref, seqs in plan:
        t0 = time.time()
        Z = np.load(kref); KR = list(Z["r_grid"])
        OFF = np.load(offp) if offp else None
        SEQ = [np.load(p) for p in seqs]
        model = AutoModelForSequenceClassification.from_pretrained(args.hf, return_dict=True)
        missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        model.to(device); model.eval()
        layers = model.roberta.encoder.layer; nL = len(layers)
        mods = [f"layer{l}.{kind}" for l in range(nL) for kind in ("query", "value")]
        assert list(Z["modules"]) == mods
        lins = {(l, kind): getattr(layers[l].attention.self, kind) for l in range(nL) for kind in ("query", "value")}
        W0 = {key: lin.weight.detach().clone() for key, lin in lins.items()}
        Wd = {key: w.cpu().double() for key, w in W0.items()}
        out = {"r_grid": np.array(R_GRID), "methods": np.array(METHODS), "modules": np.array(mods)}

        # ---- 0. untruncated scores (identity checks) ----
        bp, bn, bt = score(model)
        out["base_pos"], out["base_neg"], out["base_trig"] = bp, bn, bt
        if ref is not None:
            z = np.load(ref)
            worst = max(float(np.abs(bp - z["sdev_pos"]).max()), float(np.abs(bn - z["sdev_neg"]).max()),
                        float(np.abs(bt - z["sdev_asr"]).max()))
            print(f"[CHECK] {label} untruncated SST-2 dev (clean and triggered) vs {ref}: "
                  f"{'identical' if worst == 0 else 'DIFFERS'} (max difference {worst:.3g})", flush=True)
            if worst != 0:
                sys.exit(1)
        print(f"[KX] {label} untruncated: SST-2 dev AUC {KE.auc(bp, bn):.4f}; ASR (A) {100 * np.mean(bt <= 0):.1f}%", flush=True)

        # ---- 1. store layer-0 inputs and the encoder's call for every SST-2 train batch ----
        store, cur = [], {}
        def pre(mod, a, kw):
            cur["call"] = split_call(a, kw)
        def post(mod, a, kw, o):
            cur["last"] = o[0] if isinstance(o, tuple) else o
        h1 = layers[0].register_forward_pre_hook(pre, with_kwargs=True)
        h2 = layers[-1].register_forward_hook(post, with_kwargs=True)
        with torch.no_grad():
            for cls, loader in ((1, TR["pos"]), (0, TR["neg"])):      # same order as kdiag_eval.collect_C
                for batch in loader:
                    batch = batch.to(device)
                    model(**{k: v for k, v in batch.items() if k != "labels"})
                    h, a, kw, pos = cur["call"]
                    store.append(dict(h=h.detach().clone(), args=a, kw=kw, pos=pos, mask=batch["attention_mask"].bool(),
                                      last=cur["last"].detach().clone(), cls=cls))
        h1.remove(); h2.remove()
        print(f"[KX] {label} stored layer-0 inputs of {len(store)} SST-2 train batches  {(time.time()-t0)/60:.1f} min", flush=True)

        def propagate(per_layer, run_last=False):
            H = [b["h"] for b in store]
            with torch.no_grad():
                for l in range(nL):
                    C = torch.zeros(1024, 1024, dtype=torch.float64, device=device)
                    s = {c: torch.zeros(1024, dtype=torch.float64, device=device) for c in (0, 1)}; n = {0: 0, 1: 0}
                    for b, h in zip(store, H):
                        x = h[b["mask"]].double()
                        C.add_(x.T @ x); s[b["cls"]].add_(x.sum(0)); n[b["cls"]] += x.shape[0]
                    per_layer(l, C.cpu(), s[1].cpu(), n[1], s[0].cpu(), n[0])
                    if l < nL - 1 or run_last:
                        H = [call_layer(layers[l], b, h) for b, h in zip(store, H)]
            return H

        def layer_stats(C, s1p, np_, s1n, nn):
            n = np_ + nn; s1 = s1p + s1n; mu = s1 / n
            S = C - n * torch.outer(mu, mu); S = (S + S.T) / 2
            mp, mn = s1p / np_, s1n / nn
            SB = np_ * torch.outer(mp - mu, mp - mu) + nn * torch.outer(mn - mu, mn - mu)
            topS = torch.linalg.eigvalsh(S).flip(0)[:TOP].numpy()
            return n, s1, mu, S, SB, topS

        # ---- 2. reference pass (original weights): checks; cache everything the simultaneous truncation needs ----
        chk = {"lam": 0.0, "E": 0.0, "split": 0.0}; CACHE = {}
        RS = {k: [] for k in ["E_all", "E_var", "E_const"] + [f"R{p}_{m}" for p in ("var", "const") for m in ALL_METHODS]}
        def ref_layer(l, C, s1p, np_, s1n, nn):
            n, s1, mu, S, SB, topS = layer_stats(C, s1p, np_, s1n, nn)
            lam, Q, lam_f, nfl, tol = KE.eig_C(C)
            ref_lam = torch.from_numpy(Z["lam"][l])
            chk["lam"] = max(chk["lam"], float((lam - ref_lam).abs().max() / ref_lam[0]))
            for kind in ("query", "value"):
                W = Wd[(l, kind)]
                M = KE.analyze_module(W, C, Q, lam_f)
                i = mods.index(f"layer{l}.{kind}")
                chk["E"] = max(chk["E"], abs(M["E"] - float(Z["E"][i])) / float(Z["E"][i]))
                for m in ALL_METHODS:            # simultaneous truncation's output split (kseq_eval v2's reference pass)
                    Rv, Rc, Ev, Ec = split_curves(M, W, mu, n, m)
                    RS[f"Rvar_{m}"].append(Rv.numpy()); RS[f"Rconst_{m}"].append(Rc.numpy())
                    chk["split"] = max(chk["split"], float((M["E"] * M["curves"][m] - Ev * Rv - Ec * Rc).abs().max()) / M["E"])
                RS["E_all"].append(M["E"]); RS["E_var"].append(Ev); RS["E_const"].append(Ec)
                CACHE[(l, kind)] = dict(M=M, C=C, S=S, SB=SB, topS=topS, mu=mu, n=n)
        H = propagate(ref_layer, run_last=True)
        last = max(float((h - b["last"]).abs().max()) for h, b in zip(H, store)); del H
        print(f"[CHECK] {label} layer-by-layer with original weights vs normal forward: max difference {last:.3g}; "
              f"C eigenvalues vs kdiag: {chk['lam']:.1e}; output energy vs kdiag: {chk['E']:.1e}  {(time.time()-t0)/60:.1f} min", flush=True)
        if last > 1e-4 or chk["lam"] > 1e-6 or chk["E"] > 1e-9 or chk["split"] > 1e-8:
            print(f"[CHECK] FAIL {label}: reference pass (split error {chk['split']:.1e})"); sys.exit(1)
        out["top_all_layer"] = np.stack([CACHE[(l, "query")]["topS"] for l in range(nL)])
        for k, v in RS.items():
            out[k] = np.array(v)

        # direct check of the two new measures on layer 3 query, P = W's own top-200 directions
        cc = CACHE[(3, "query")]; Rb = cc["M"]["V"][:, :200]
        ev, sbo, _ = extras(Rb, cc["S"], cc["SB"], cc["topS"])
        Bc = torch.linalg.qr(Rb, mode="complete")[0][:, 200:]
        ev2 = torch.linalg.eigvalsh(Bc.T @ cc["S"] @ Bc).flip(0)[:TOP].numpy()
        dirn = torch.linalg.eigh(cc["SB"])[1][:, -1]     # two classes: S_B is rank one, along mu_positive - mu_negative
        sbo2 = 1 - float((Rb.T @ dirn) @ (Rb.T @ dirn))
        print(f"[CHECK] {label} new measures on layer3.query vs direct formulas: eigenvalues {np.abs(ev - ev2).max() / ev2[0]:.1e}, "
              f"between-class share {abs(sbo - sbo2):.1e}", flush=True)
        if np.abs(ev - ev2).max() / ev2[0] > 1e-9 or abs(sbo - sbo2) > 1e-9:
            print(f"[CHECK] FAIL {label}: new measures"); sys.exit(1)

        # ---- 3. simultaneous truncations (= kdiag), with the new measures and the triggered sentences ----
        for m in METHODS:
            acc = {k: [] for k in ("k", "top_out", "sb_out", "dev_pos", "dev_neg", "trig")}
            for r in R_GRID:
                ks, tops, sbs = [], [], []
                for (l, kind), cc in CACHE.items():
                    k = KE.k_for(cc["M"]["curves"][m], r)
                    Wk, Rb = KE.truncated(cc["M"], m, k)
                    ev, sbo, _ = extras(Rb, cc["S"], cc["SB"], cc["topS"])
                    ks.append(k); tops.append(ev); sbs.append(sbo)
                    with torch.no_grad():
                        lins[(l, kind)].weight.copy_(Wk.to(W0[(l, kind)].dtype).to(device))
                sp, sn, st = score(model)
                for key, lin in lins.items():
                    with torch.no_grad():
                        lin.weight.copy_(W0[key])
                ki = KR.index(r)
                same_k = np.array_equal(np.array(ks), Z[f"k_{m}"][ki])
                gap = max(float(np.abs(sp - Z[f"dev_pos_{m}"][ki]).max()), float(np.abs(sn - Z[f"dev_neg_{m}"][ki]).max()))
                if not same_k or gap > 1e-4:
                    print(f"[CHECK] FAIL {label} simultaneous {m} r={r}: k {'same' if same_k else 'DIFFERENT'} as kdiag, "
                          f"clean dev scores differ by {gap:.3g}"); sys.exit(1)
                acc["k"].append(ks); acc["top_out"].append(tops); acc["sb_out"].append(sbs)
                acc["dev_pos"].append(sp); acc["dev_neg"].append(sn); acc["trig"].append(st)
                room8 = np.median([t[:8].sum() / CACHE[key]["topS"][:8].sum() for t, key in zip(tops, CACHE)])
                print(f"[KX] {label} simultaneous {m} r={r}: kept {np.mean(ks):.1f}; dev AUC {KE.auc(sp, sn):.4f} "
                      f"(scores vs kdiag {'identical' if gap == 0 else f'max diff {gap:.1e}'}); ASR (A) {100 * np.mean(st <= 0):.1f}%; "
                      f"room for rank 8 (median) {room8:.3f}; between-class outside (median) {np.median(sbs):.3f}  "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
            for k, v in acc.items():
                out[f"sim_{k}_{m}"] = np.array(v)

        # ---- 4. sequential truncations (= kseq), with the same additions ----
        for m in METHODS:
            acc = {k: [] for k in ("k", "exact_err", "outside_all", "outside_var", "outside_mu", "ret_all", "ret_var", "ret_const",
                                   "top_out", "top_all", "sb_out", "L_mean_share", "L_top_share", "L_top_mean", "L_top_share_S",
                                   "n_floored", "dev_pos", "dev_neg", "trig")}
            for r in R_GRID:
                rec = {k: [] for k in acc if k not in ("n_floored", "dev_pos", "dev_neg", "trig")}; rec["n_floored"] = 0
                def trunc_layer(l, C, s1p, np_, s1n, nn):
                    n, s1, mu, S, SB, topS = layer_stats(C, s1p, np_, s1n, nn)
                    lam, Q, lam_f, nfl, tol = KE.eig_C(C)
                    rec["n_floored"] += nfl
                    mu2 = float(mu @ mu); trC = float(torch.trace(C))
                    rec["L_mean_share"].append(mu2 / (trC / n)); rec["L_top_share"].append(float(lam[0]) / trC)
                    rec["L_top_share_S"].append(float(topS[0] / torch.trace(S)))
                    rec["L_top_mean"].append(float(Q[:, 0] @ mu) ** 2 / (float(lam[0]) / n))
                    for kind in ("query", "value"):
                        W = Wd[(l, kind)]
                        M = KE.analyze_module(W, C, Q, lam_f)
                        k = KE.k_for(M["curves"][m], r)
                        Wk, Rb = KE.truncated(M, m, k)
                        inside = float(torch.trace(Rb.T @ C @ Rb)); pm = Rb.T @ mu; inside_mu = float(pm @ pm)
                        Rv, Rc, Ev, Ec = split_curves(M, W, mu, n, m)
                        ev, sbo, _ = extras(Rb, S, SB, topS)
                        rec["k"].append(k); rec["exact_err"].append(KE.exact_err(W, Wk, C, M["E"]))
                        rec["outside_all"].append(1 - inside / trC)
                        rec["outside_var"].append(1 - (inside - n * inside_mu) / (trC - n * mu2))
                        rec["outside_mu"].append(1 - inside_mu / mu2)
                        rec["ret_all"].append(float(M["curves"][m][k - 1]))
                        rec["ret_var"].append(float(Rv[k - 1])); rec["ret_const"].append(float(Rc[k - 1]))
                        rec["top_out"].append(ev); rec["top_all"].append(topS); rec["sb_out"].append(sbo)
                        with torch.no_grad():
                            lins[(l, kind)].weight.copy_(Wk.to(W0[(l, kind)].dtype).to(device))
                propagate(trunc_layer)
                kd = Z[f"k_{m}"][KR.index(r)]
                if rec["k"][:2] != [int(kd[0]), int(kd[1])]:
                    print(f"[CHECK] FAIL {label} sequential {m} r={r}: layer 0 k differs from kdiag"); sys.exit(1)
                if OFF is not None and abs(rec["L_mean_share"][0] - float(OFF["mean_share"][0])) > 1e-9:
                    print(f"[CHECK] FAIL {label} sequential {m} r={r}: layer 0 offset numbers differ from kdiag_offset"); sys.exit(1)
                sp, sn, st = score(model)
                for key, lin in lins.items():
                    with torch.no_grad():
                        lin.weight.copy_(W0[key])
                for Sq, p in zip(SEQ, seqs):
                    if m in list(Sq["methods"]) and r in list(Sq["r_grid"]):
                        j = list(Sq["r_grid"]).index(r)
                        same_k = list(Sq[f"k_{m}"][j]) == rec["k"]
                        same_s = np.array_equal(Sq[f"dev_pos_{m}"][j], sp) and np.array_equal(Sq[f"dev_neg_{m}"][j], sn)
                        print(f"[CHECK] {label} sequential {m} r={r} vs {p}: k {'identical' if same_k else 'DIFFERENT'}, "
                              f"SST-2 dev scores {'identical' if same_s else 'DIFFERENT'}", flush=True)
                        if not (same_k and same_s):
                            sys.exit(1)
                        break
                for k, v in rec.items():
                    acc[k].append(v)
                acc["dev_pos"].append(sp); acc["dev_neg"].append(sn); acc["trig"].append(st)
                room8 = np.median([t[:8].sum() / a[:8].sum() for t, a in zip(rec["top_out"], rec["top_all"])])
                print(f"[KX] {label} sequential {m} r={r}: kept {np.mean(rec['k']):.1f}; dev AUC {KE.auc(sp, sn):.4f}; "
                      f"ASR (A) {100 * np.mean(st <= 0):.1f}%; room for rank 8 (median) {room8:.3f}; between-class outside "
                      f"(median) {np.median(rec['sb_out']):.3f}  {(time.time()-t0)/60:.1f} min", flush=True)
            for k, v in acc.items():
                out[f"seq_{k}_{m}"] = np.array(v)
        sp, sn, st = score(model)
        assert np.array_equal(sp, bp) and np.array_equal(sn, bn) and np.array_equal(st, bt), "weights were not restored"
        np.savez_compressed(os.path.join(args.out_dir, f"{label}.npz"), **{k: np.asarray(v) for k, v in out.items()})
        print(f"[SAVE] {args.out_dir}/{label}.npz  missing={len(missing)} unexpected={len(unexpected)}  {(time.time()-t0)/60:.1f} min", flush=True)
        del model, store, CACHE; torch.cuda.empty_cache()
    print("[CHECK] done")


if __name__ == "__main__":
    main()