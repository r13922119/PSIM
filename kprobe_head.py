#!/usr/bin/env python3
"""kprobe_head.py (10/27 night) -- does the backdoor survive if the PZ's classification head is replaced by a NEW head
trained only on clean SST-2? (the user's 10/25 worry; the cheap probe proposed 10/25, agreed 10/27 21:43)

Model stage: the PZ (or the clean model) exactly as saved, encoder FROZEN, no LoRA, no fine-tuning, dropout off.
Features: the last layer's hidden state of the first token <s> (the same vector the PZ's own head starts from).
New head: logistic regression (one linear layer, 1024 -> 1 score) trained on SST-2 TRAIN features with labels; features
standardized with the SST-2 train mean and standard deviation; loss = mean binary cross-entropy + lambda * ||w||^2,
full-batch L-BFGS in float64. lambda is NOT chosen: every value of --lambdas is trained and reported.
Measured on SST-2 DEV (444 positive, 428 negative) and on the 444 dev positives with the trigger inserted
(attack_utils.build_poisoned_test_dataloader(..., "mn"), as in sst2dev_eval / p_auc_eval):
  clean accuracy (score > 0 = positive), AUC, and ASR = share of triggered positives scored <= 0 (called negative).
The PZ's own head is measured on the same features as the reference (ASR of type (A)); its scores must equal
logs/scores/sst2dev/<label>.npz (checked), which also checks that the features are the right vector.
This is the experimenter's diagnostic (it uses the trigger); it is not part of the method.
Output: <out_dir>/<label>.npz (scores per lambda) and log lines [HEAD], [CHECK]. numpy + torch only.
  python kprobe_head.py --dry --model LABEL=PATH[@EPOCH] ...
"""
import argparse, json, os, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--lambdas", default="1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--out_dir", default="logs/kprobe_head")
    ap.add_argument("--ref_dir", default="logs/scores/sst2dev")
    ap.add_argument("--hf", default="roberta-large")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    LAMS = [float(x) for x in args.lambdas.split(",")]
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
        print(f"[PLAN] {label} <- {ckpt}  epoch_saved={ep}  reference: {ref if os.path.exists(ref) else 'none'}")
        plan.append((label, ckpt, ref if os.path.exists(ref) else None))
    if problems:
        for x in problems:
            print(f"[CHECK] FAIL {x}")
        sys.exit(1)
    print(f"[CHECK] all {len(plan)} models found; lambdas {LAMS}")
    if args.dry:
        sys.exit(0)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import bias_probe
    from attack_utils import build_poisoned_test_dataloader
    from kdiag_eval import auc
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

    @torch.no_grad()
    def feats(model, loader):
        F, S = [], []
        for batch in loader:
            batch = batch.to(device)
            h = model.roberta(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            lg = model.classifier(h).float()
            F.append(h[:, 0].double().cpu()); S.append((lg[:, 1] - lg[:, 0]).cpu())
        return torch.cat(F), torch.cat(S).numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    for label, ckpt, ref in plan:
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(args.hf, return_dict=True)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        model.to(device); model.eval()
        Ftp, _ = feats(model, TR["pos"]); Ftn, _ = feats(model, TR["neg"])
        Fdp, odp = feats(model, DV["pos"]); Fdn, odn = feats(model, DV["neg"]); Ftg, otg = feats(model, TRIG)
        if ref is not None:
            z = np.load(ref)
            worst = max(float(np.abs(odp - z["sdev_pos"]).max()), float(np.abs(odn - z["sdev_neg"]).max()),
                        float(np.abs(otg - z["sdev_asr"]).max()))
            print(f"[CHECK] {label} own head on the extracted features vs {ref}: {'identical' if worst == 0 else 'DIFFERS'} "
                  f"(max difference {worst:.3g})", flush=True)
            if worst != 0:
                sys.exit(1)
        acc = lambda p, n: (np.sum(p > 0) + np.sum(n <= 0)) / (len(p) + len(n))
        out = {"lambdas": np.array(LAMS), "own_dev_pos": odp, "own_dev_neg": odn, "own_trig": otg}
        print(f"[HEAD] {label} own head (type A): dev accuracy {100 * acc(odp, odn):.1f}%, AUC {auc(odp, odn):.4f}, "
              f"ASR {100 * np.mean(otg <= 0):.1f}%", flush=True)
        X = torch.cat([Ftp, Ftn]); y = torch.cat([torch.ones(len(Ftp)), torch.zeros(len(Ftn))]).double()
        mean, std = X.mean(0), X.std(0).clamp_min(1e-12)
        Z_ = lambda F: (F - mean) / std
        Xs = Z_(X)
        for lam in LAMS:
            w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
            b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
            opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                                    history_size=50, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, y) + lam * (w @ w)
                loss.backward(); return loss
            loss = opt.step(closure)
            with torch.no_grad():
                sc = lambda F: (Z_(F) @ w + b).numpy()
                tr_acc = float(((Xs @ w + b > 0).double() == y).double().mean())
                sp, sn, st = sc(Fdp), sc(Fdn), sc(Ftg)
            out[f"new_dev_pos_{lam:g}"], out[f"new_dev_neg_{lam:g}"], out[f"new_trig_{lam:g}"] = sp, sn, st
            out[f"new_train_acc_{lam:g}"] = tr_acc; out[f"new_loss_{lam:g}"] = float(loss)
            print(f"[HEAD] {label} new linear head, lambda {lam:g}: train accuracy {100 * tr_acc:.1f}%, dev accuracy "
                  f"{100 * acc(sp, sn):.1f}%, AUC {auc(sp, sn):.4f}, ASR {100 * np.mean(st <= 0):.1f}%  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        np.savez_compressed(os.path.join(args.out_dir, f"{label}.npz"), **{k: np.asarray(v) for k, v in out.items()})
        print(f"[SAVE] {args.out_dir}/{label}.npz  {(time.time()-t0)/60:.1f} min", flush=True)
        del model; torch.cuda.empty_cache()
    print("[CHECK] done")


if __name__ == "__main__":
    main()