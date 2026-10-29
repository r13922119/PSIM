#!/usr/bin/env python3
"""make_wpre.py — K step of the algorithm: for each PZ, pick r* by the K rule, build W_pre,K by whiten truncation
(SVD-LLM, simultaneous, one r for all 48 query/value modules) and save it for variant_finetune.py --wpre_file.

The rule only reads logs/kdiag/<label>.npz (SST-2 dev scores of the PZ's own head, untruncated and truncated at each r of
kdiag's grid, already measured). Nothing here looks at ASR or at any triggered sentence.
  Per r: d(r) = AUC(truncated) - AUC(untruncated) on SST-2 dev (444 positives, 428 negatives).
  Band: paired, stratified bootstrap, B resamples (the same resampled sentences for both models; seed 0,
        drawn per PZ so the result does not depend on which other PZs are in the call; default B = 100,000: at 10,000 one
        cell flipped between resample seeds, see k_rule_mc.py).
  "No significant drop" at r: the band's upper end >= 0, i.e. one-sided test that AUC got worse.
     --level per_r  each r tested at one-sided 2.5%     -> 97.5th percentile (= the upper end of a 95% two-sided band)
     --level bonf   Bonferroni over the len(r grid) r's, family one-sided 5% -> 100 - 5/len percentile (99.5 for 10 r's)
  Walk:  --walk stepwise  from the largest r down; r* = the last r before the first significant drop (all larger r pass)
         --walk any       r* = the smallest r with no significant drop
  The AUC may get better (one-sided test): your 10/26 decision.
  --r_fixed R skips the rule and uses r* = R for every model (the table is still printed).

Then, per PZ, on the GPU: C (SST-2 train, labels unused) exactly as kdiag_eval.py, whiten truncation of all 48 modules at r*
(each module keeps the smallest k with retained output energy R(k) >= r*, same function as kdiag). Checks:
  - every module's k equals kdiag's k at r* (hard), exact reconstruction error equals kdiag's (reported)
  - the saved file, loaded back through wpre_utils.apply (the same function variant_finetune.py uses) into a FRESH copy of the
    PZ, scores SST-2 dev bit-identically to kdiag's truncated scores at r* (hard)
  - what Tr will see: variant_finetune.py takes U, V from a float32 SVD of the stored W_pre,K; reported per module:
    sigma_k / sigma_1, sigma_{k+1} / sigma_k, and how much of the exact row / column space the float32 top-k spaces cover
Output (pz_wpre/ is git-ignored, about 200 MB per PZ):
  pz_wpre/<label>_whiten_r<r*>.pt     48 truncated weights
  pz_wpre/<label>_whiten_r<r*>.json   provenance and per-module k; written LAST, after the checks pass
  logs/wpre/<label>_whiten_r<r*>.json copy of the json (small; commit it)
  logs/wpre/choice_<level>_<walk>.tsv  label, r*, json path, PZ path: what run_psel.sh and run_final.sh read
numpy + torch only.
  python make_wpre.py --rule_only --model LABEL=PATH[@EPOCH] ...       # CPU, prints the rule's table, writes nothing
  python make_wpre.py --level bonf --model LABEL=PATH[@EPOCH] ...      # normally through run_make_wpre.sh (one card)
"""
import argparse, hashlib, json, os, shutil, sys, time
import numpy as np


# ---------------- the K rule (CPU, numpy) ----------------
def auc_rows(P, N):
    """AUC of each row pair (ties count 1/2)."""
    a, b = P.shape[1], N.shape[1]
    Ns = np.sort(N, axis=1); out = np.empty(P.shape[0])
    for i in range(P.shape[0]):
        lo = np.searchsorted(Ns[i], P[i], "left"); hi = np.searchsorted(Ns[i], P[i], "right")
        out[i] = (lo.sum() + 0.5 * (hi - lo).sum()) / (a * b)
    return out


def boot_counts(rng, n, B, chunk=10000):
    """Resample counts (B x n): row b = how many times each sentence is drawn in resample b (n draws with replacement).
    Drawn as integer indices, chunk by chunk, so the stream is the same for any B that is a multiple of chunk."""
    out = []
    for s in range(0, B, chunk):
        m = min(chunk, B - s)
        I = rng.integers(0, n, (m, n))
        out.append(np.bincount((I + n * np.arange(m)[:, None]).ravel(), minlength=m * n).reshape(m, n).astype(np.float64))
    return np.concatenate(out)


def auc_boot(sp, sn, WP, WN, chunk=10000):
    """AUC of every resample, exactly: AUC_b = sum_ij WP[b,i] WN[b,j] H[i,j] / (a b), H = 1[p > n] + 1/2 1[p = n].
    All terms are multiples of 1/2 well below 2^53, so the sums are exact and equal auc_rows on the resampled scores."""
    H = (sp[:, None] > sn[None, :]).astype(np.float64) + 0.5 * (sp[:, None] == sn[None, :])
    out = np.empty(WP.shape[0])
    for s in range(0, WP.shape[0], chunk):
        out[s:s + chunk] = ((WP[s:s + chunk] @ H) * WN[s:s + chunk]).sum(1)
    return out / (len(sp) * len(sn))


def rule_table(z, B=100000, seed=0, method="whiten"):
    """One row per r of kdiag's grid: r, kept k (mean, min, max), d(r), percentiles of the bootstrap d, and the same
    percentiles on each half of the resamples (Monte Carlo check: if the two halves disagree in sign, the cell is borderline)."""
    R = [float(x) for x in z["r_grid"]]
    bp, bn = z["base_pos"].astype(float), z["base_neg"].astype(float)
    rng = np.random.default_rng(seed)
    WP = boot_counts(rng, len(bp), B); WN = boot_counts(rng, len(bn), B)
    base = auc_boot(bp, bn, WP, WN); a0 = auc_rows(bp[None], bn[None])[0]
    pb = 100 - 5.0 / len(R); h = B // 2
    rows = []
    for i, r in enumerate(R):
        sp, sn = z[f"dev_pos_{method}"][i].astype(float), z[f"dev_neg_{method}"][i].astype(float)
        d = auc_boot(sp, sn, WP, WN) - base
        k = z[f"k_{method}"][i]; a1 = auc_rows(sp[None], sn[None])[0]
        rows.append(dict(r=r, k_mean=float(k.mean()), k_min=int(k.min()), k_max=int(k.max()), d=float(a1 - a0), auc=float(a1),
                         p2_5=float(np.percentile(d, 2.5)), p97_5=float(np.percentile(d, 97.5)), p_bonf=float(np.percentile(d, pb)),
                         p97_5_halves=[float(np.percentile(d[:h], 97.5)), float(np.percentile(d[h:], 97.5))],
                         p_bonf_halves=[float(np.percentile(d[:h], pb)), float(np.percentile(d[h:], pb))]))
    return rows, float(a0)


def pick(rows, level, walk):
    col = "p97_5" if level == "per_r" else "p_bonf"
    ok = [row[col] >= 0 for row in rows]          # rows are in kdiag's order: r from large to small
    if walk == "stepwise":
        chosen = None
        for row, o in zip(rows, ok):
            if not o:
                break
            chosen = row
    else:
        passing = [row for row, o in zip(rows, ok) if o]
        chosen = min(passing, key=lambda x: x["r"]) if passing else None
    return chosen, ok


def print_table(label, rows, a0, level, walk, chosen):
    print(f"[RULE] {label}: untruncated SST-2 dev AUC {a0:.4f}; band = paired stratified bootstrap; "
          f"pass = upper end >= 0 ({'97.5th' if level == 'per_r' else f'{100 - 5.0 / len(rows):g}th'} percentile, level {level}); walk {walk}")
    print(f"[RULE] {label}:   r       k mean [min, max]    AUC     d(r)     2.5th    97.5th   {100 - 5.0 / len(rows):g}th   pass(per_r) pass(bonf)")
    def yn(v, halves):
        flag = "?" if (halves[0] >= 0) != (halves[1] >= 0) else " "      # the two halves of the resamples disagree
        return ("yes" if v >= 0 else "NO ") + flag
    for row in rows:
        mark = "  <- r*" if chosen is not None and row["r"] == chosen["r"] else ""
        print(f"[RULE] {label}:   {row['r']:<7g} {row['k_mean']:6.1f} [{row['k_min']:3d}, {row['k_max']:3d}]  {row['auc']:.4f}  {row['d']:+.4f}  "
              f"{row['p2_5']:+.4f}  {row['p97_5']:+.4f}  {row['p_bonf']:+.4f}   {yn(row['p97_5'], row['p97_5_halves'])}        "
              f"{yn(row['p_bonf'], row['p_bonf_halves'])}{mark}")
    print(f"[RULE] {label}:   ('?' = the percentile computed on each half of the resamples has a different sign: Monte Carlo borderline)")


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="LABEL=PATH or LABEL=PATH@EPOCH (same as kdiag_eval.py)")
    ap.add_argument("--level", choices=["per_r", "bonf"], default=None)
    ap.add_argument("--walk", choices=["stepwise", "any"], default="stepwise")
    ap.add_argument("--r_fixed", type=float, default=None, help="skip the rule: r* = this r (must be in kdiag's grid) for every model")
    ap.add_argument("--B", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kdiag_dir", default="logs/kdiag")
    ap.add_argument("--out_dir", default="pz_wpre")
    ap.add_argument("--record_dir", default="logs/wpre")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--rule_only", action="store_true", help="CPU only: print the rule's table for both levels, write nothing")
    args = ap.parse_args()
    if not args.rule_only and args.level is None and args.r_fixed is None:
        sys.exit("give --level per_r or --level bonf (or --r_fixed R); there is no default on purpose")

    plan = []
    for item in args.model:
        label, rest = item.split("=", 1)
        path, want = (rest.split("@") + [None])[:2]
        ckpt = path if path.endswith(".bin") else os.path.join(path, "pytorch_model.bin")
        kd = os.path.join(args.kdiag_dir, f"{label}.npz")
        if not os.path.exists(kd):
            sys.exit(f"[CHECK] FAIL {kd} missing")
        if not args.rule_only:
            if not os.path.exists(ckpt):
                sys.exit(f"[CHECK] FAIL missing {ckpt}")
            meta_p = os.path.join(os.path.dirname(ckpt), "meta.json")
            ep = json.load(open(meta_p)).get("epoch_saved") if os.path.exists(meta_p) else None
            if want is not None and (ep is None or int(ep) != int(want)):
                sys.exit(f"[CHECK] FAIL {label}: meta.json epoch_saved={ep}, expected {want}")
        z = np.load(kd)
        rows, a0 = rule_table(z, args.B, args.seed)
        choice = {}
        for level in ("per_r", "bonf"):
            for walk in ("stepwise", "any"):
                choice[(level, walk)] = pick(rows, level, walk)[0]
        if args.rule_only:
            for level in ("per_r", "bonf"):
                print_table(label, rows, a0, level, args.walk, choice[(level, args.walk)])
            print("[RULE] " + label + ": r* " + "  ".join(f"{lv}/{wk}={c['r'] if c else 'none'}" for (lv, wk), c in choice.items()))
            continue
        if args.r_fixed is not None:
            R = [row["r"] for row in rows]
            if args.r_fixed not in R:
                sys.exit(f"--r_fixed {args.r_fixed} is not in kdiag's grid {R}")
            chosen = rows[R.index(args.r_fixed)]; how = f"fixed r = {args.r_fixed}"
        else:
            chosen = choice[(args.level, args.walk)]; how = f"rule level={args.level} walk={args.walk}"
        print_table(label, rows, a0, args.level or "bonf", args.walk, chosen)
        if chosen is None:
            print(f"[RULE] {label}: no r passes (even r = {rows[0]['r']} drops AUC significantly); no W_pre,K written")
            continue
        print(f"[RULE] {label}: r* = {chosen['r']:g} ({how}); mean kept directions {chosen['k_mean']:.1f}; d(r*) = {chosen['d']:+.4f}")
        plan.append((label, ckpt, z, chosen, rows, a0, how))
    if args.rule_only or not plan:
        return

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import bias_probe, kdiag_eval as KE, wpre_utils
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
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.record_dir, exist_ok=True)
    code = {f: file_sha(f) for f in ("make_wpre.py", "kdiag_eval.py", "wpre_utils.py", "bias_probe.py") if os.path.exists(f)}
    failed, written = [], []

    def fresh(ckpt):
        m = AutoModelForSequenceClassification.from_pretrained(hf, return_dict=True)
        miss, unexp = m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        return m, miss, unexp

    for label, ckpt, z, chosen, rows, a0, how in plan:
        t0 = time.time(); r = chosen["r"]; i = [row["r"] for row in rows].index(r)
        model, missing, unexpected = fresh(ckpt)
        mods = wpre_utils.qv_modules(model)
        w0_sha = {n: wpre_utils.sha(m.weight) for n, m in mods}
        dtype = mods[0][1].weight.dtype
        model.to(device)
        nL = len(model.roberta.encoder.layer)
        Cs, ntok, _ = KE.collect_C(model, [TR["pos"], TR["neg"]], device, nL)
        print(f"[WPRE] {label} C from {ntok} SST-2 train tokens (kdiag: {int(z['n_tokens'])})  {(time.time()-t0)/60:.1f} min", flush=True)
        if ntok != int(z["n_tokens"]):
            failed.append(f"{label}: token count differs from kdiag"); continue
        names = [n for n, _ in mods]
        kd_mods = [str(s) for s in z["modules"]]          # kdiag order: layer0.query, layer0.value, layer1.query, ...
        WK, KMAP, errs, k_bad, tr_rows = {}, {}, [], [], []
        for l in range(nL):
            lam, Q, lam_f, nfl, tol = KE.eig_C(Cs[l])
            for kind in ("query", "value"):
                name = f"roberta.encoder.layer.{l}.attention.self.{kind}"
                j = kd_mods.index(f"layer{l}.{kind}")
                lin = dict(mods)[name]
                W = lin.weight.detach().cpu().double()
                M = KE.analyze_module(W, Cs[l], Q, lam_f)
                k = KE.k_for(M["curves"]["whiten"], r)
                if k != int(z["k_whiten"][i][j]):
                    k_bad.append((name, k, int(z["k_whiten"][i][j])))
                Wk, Rb = KE.truncated(M, "whiten", k)
                err = KE.exact_err(W, Wk, Cs[l], M["E"])
                errs.append(abs(err - float(z["exact_err_whiten"][i][j])))
                Wf = Wk.to(dtype).contiguous().clone()
                WK[name] = Wf; KMAP[name] = k
                # what Tr will see: float32 SVD of the stored matrix, as variant_finetune.py does (CPU, before model.to)
                Uf, Sf, Vhf = torch.linalg.svd(Wf, full_matrices=False)
                s64 = torch.linalg.svdvals(Wf.double())
                cov_in = float(((Rb.T @ Vhf[:k].T.double()) ** 2).sum()) / k      # 1 = float32 top-k right space = exact row space
                cov_out = float(((M["Uw"][:, :k].T @ Uf[:, :k].double()) ** 2).sum()) / k
                tr_rows.append((name, k, float(s64[k - 1] / s64[0]), float(s64[k] / s64[k - 1]) if k < len(s64) else 0.0, cov_in, cov_out))
        if k_bad:
            failed.append(f"{label}: k differs from kdiag in {len(k_bad)} modules, e.g. {k_bad[:3]}")
            print(f"[CHECK] FAIL {failed[-1]}"); continue
        print(f"[CHECK] {label} k of all 48 modules equals kdiag's at r = {r:g}; max |exact error - kdiag's| {max(errs):.2e}", flush=True)
        a = np.array([x[2:] for x in tr_rows])
        print(f"[TR] {label} float32 W_pre,K as Tr sees it (48 modules): sigma_k/sigma_1 min {a[:, 0].min():.2e}; "
              f"sigma_(k+1)/sigma_k max {a[:, 1].max():.2e}; row-space coverage min {a[:, 2].min():.6f}; column-space coverage min {a[:, 3].min():.6f}", flush=True)

        stem = f"{label}_whiten_r{r:g}"
        pt = os.path.join(args.out_dir, stem + ".pt"); js = os.path.join(args.out_dir, stem + ".json")
        if os.path.exists(js):
            os.remove(js)                                   # an old json must never vouch for a new .pt
        torch.save(WK, pt + ".tmp")
        meta = dict(label=label, pz=ckpt, method="whiten", ordering="SVD-LLM whitening, C = sum over SST-2 train tokens of x x^T",
                    truncation="simultaneous (C from the untruncated PZ), one r for all 48 query/value modules",
                    r=r, tag=f"wh{r:g}", how=how, rule=dict(level=args.level, walk=args.walk, B=args.B, seed=args.seed,
                    table=rows, auc_untruncated=a0), weights_file=os.path.basename(pt), per_module_k=KMAP,
                    w0_sha256=w0_sha, wk_sha256={n: wpre_utils.sha(t) for n, t in WK.items()},
                    tr_view={n: dict(k=k, s_k_over_s_1=a1, s_k1_over_s_k=a2, row_cov=a3, col_cov=a4) for n, k, a1, a2, a3, a4 in tr_rows},
                    n_tokens=ntok, kdiag_file=os.path.join(args.kdiag_dir, f"{label}.npz"), code_sha256=code,
                    created=time.strftime("%Y-%m-%d %H:%M:%S"))
        # identity check through the same loader variant_finetune.py uses, on a fresh copy of the PZ, from the file on disk
        del model; torch.cuda.empty_cache() if device == "cuda" else None
        m2, _, _ = fresh(ckpt)
        wpre_utils.apply(m2, meta, torch.load(pt + ".tmp", map_location="cpu"), ckpt)
        m2.to(device)
        _, sp, sn = bias_probe.class_probe(m2, DV, device, keep=True)
        worst = max(float(np.abs(sp - z["dev_pos_whiten"][i]).max()), float(np.abs(sn - z["dev_neg_whiten"][i]).max()))
        auc_now = float(KE.auc(sp, sn))
        print(f"[CHECK] {label} saved W_pre,K loaded into a fresh PZ: SST-2 dev scores vs kdiag at r = {r:g}: "
              f"{'identical' if worst == 0.0 else 'DIFFER'} (max difference {worst:.3g}); AUC {auc_now:.4f} (kdiag {chosen['auc']:.4f})", flush=True)
        del m2; torch.cuda.empty_cache() if device == "cuda" else None
        if worst != 0.0:
            failed.append(f"{label}: dev scores differ from kdiag"); os.remove(pt + ".tmp"); continue
        meta["checks"] = dict(k_equal_kdiag=True, max_exact_err_diff=max(errs), dev_scores_identical=True, auc=auc_now,
                              missing_keys=len(missing), unexpected_keys=len(unexpected))
        os.replace(pt + ".tmp", pt)
        meta["weights_sha256"] = file_sha(pt)
        with open(js, "w") as f:
            json.dump(meta, f, indent=1)
        shutil.copy(js, os.path.join(args.record_dir, stem + ".json"))
        written.append((label, r, js, ckpt))
        print(f"[SAVE] {pt} ({os.path.getsize(pt) / 2**20:.0f} MB), {js}  {(time.time()-t0)/60:.1f} min", flush=True)

    if written and args.r_fixed is None:
        tsv = os.path.join(args.record_dir, f"choice_{args.level}_{args.walk}.tsv")
        old = {}
        if os.path.exists(tsv):
            for line in open(tsv):
                p = line.rstrip("\n").split("\t")
                if len(p) == 4 and p[0] != "label":
                    old[p[0]] = p
        for label, r, js, ckpt in written:
            old[label] = [label, f"{r:g}", js, ckpt]
        with open(tsv, "w") as f:
            f.write("label\tr\tjson\tpz\n")
            for label in sorted(old):
                f.write("\t".join(old[label]) + "\n")
        print(f"[SAVE] {tsv}")
    if failed:
        for x in failed:
            print(f"[CHECK] FAIL {x}")
        sys.exit(1)
    print("[CHECK] done")


if __name__ == "__main__":
    main()