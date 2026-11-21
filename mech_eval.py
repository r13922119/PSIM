#!/usr/bin/env python3
"""mech_eval.py — the evaluation-only experiments 3D, 2A, 3A and 3B of claude/dropout_theory.md, on the existing adapters of the
last epoch (plain LoRA, cl0.2, clsvd0.2; the 5 panel PZs x fine-tune seeds 0-2). No training. Nothing in the repo is changed
except the output folder (--out_dir, default logs/mech_eval).

  python mech_eval.py --dry                    check every input; no GPU, no model
  python mech_eval.py                          all four parts on one Ada (about 1.5-2 h; see the cost lines below)
  python mech_eval.py --parts 3A               only some parts (comma-separated: 3D,2A,3A,3B)

3D  noise sensitivity (dropout_theory.md 結論 3.2). SST-2 dev (872 sentences), every configuration (5 PZs + 45 adapters).
    The dropout on the frozen W_pre output used in training (cl_utils.py, the same functions and the same hook point) is
    switched on at evaluation, K masks per form (coord and svd, p 0.2); every other dropout stays off (model.eval()).
    The same masks are used for every configuration (seeded per pass), so pairs differ only by the model.
    Per sentence: m = margin without noise, rel_sd = sd of the K noisy margins / |m|, flip = share of masks that change
    the sign of m. Pre-registered statistic: median rel_sd; a dropout arm against plain LoRA of the same PZ and seed,
    under the form it was trained with; >= 12 of 15 pairs lower supports 結論 3.2 (sign test, one-sided p ~ 0.018).
    Check (stops the run): the noise-free dev accuracy of every adapter equals last_dev_acc of its training log (printed with
    4 decimals, which fixes the count of correct sentences out of 872), up to the number of dev sentences with |m| < --dev_tol
    (default 2e-3: the same GPU gave logits up to 0.00113 apart between two scripts, logs/margin_eval_20261117_180625.log,
    and you set --tol 2e-3 there); such sentences can change sign from rounding alone.
2A  training margins (結論 2.3). SST-2 train (all 6,920 sentences by default; dropout_theory.md said 5,000 only to save time:
    --n_train 5000 takes a fixed random subset, seed 0), every configuration.
    median |m|, median m of each label, train accuracy. Dropout arm vs plain: >= 12/15 pairs with larger median |m|
    supports 結論 2.3; >= 12/15 smaller supports the head penalty of §3.
3A  classification head x LoRA update swap (結論 3.2, 3.5, 3.7). SST-2 test, scored exactly as margin_eval.py. For each PZ and
    seed: head from one arm, the 48 LoRA updates (A, B of every query/value) from one arm (3 x 3 = 9 combinations), and
    layer splits: head and the updates of layers >= L from a dropout arm, layers < L from plain LoRA (L in --split_layers).
    For every combination: clean accuracy, pos_err, fpr, ASR, ASR among clean-correct, median clean margin of positives and
    of negatives and median |m| of all clean sentences (the logit scale), median pull s, median(s)/median(m), clean AUC.
    Reading (dropout_theory.md 3A): does median(s)/median(m) follow the head (and upper layers) or the lower-layer updates;
    each comparison needs >= 12 of 15 pairs in the same direction. pairs.csv gives, for every combination, the number of pairs
    where its median(s)/median(m) is below plain LoRA's, and the median share of the full change (dropout run vs plain) that
    the combination reproduces. Clean accuracy and the logit scale of every combination show whether the swapped halves still
    work together (co-adaptation): a swap that lowers the pull but also breaks clean accuracy or shrinks the logits cannot be
    read as the head (or the updates) carrying the effect.
    Check (stops the run): the 3 diagonal combinations (head and updates from the same run) reproduce margin_eval.py's
    stored per-sentence scores (logs/margin_decomp/npz) within --tol, with no prediction changed.
3B  which branch carries the trigger (結論 3.5, condition C1). SST-2 dev positives with "mn" inserted by attack_utils.insert_trigger
    (PLACEMENT_SEED). At every query/value module of every adapter:
      r_cls  = |dW d| / |W_pre d|,  d = change of the module's input at <s> caused by inserting the trigger
      r_trig = |dW x| / |W_pre x| over the inserted trigger token(s) (sentences whose tokens around the insertion change are skipped)
      r_tok  = the same over the words of the clean sentence, <s> and </s> excluded (reference)
    dW = scaling * B A of that module; W_pre without bias.

Cost (from logs/margin_eval_20261117_180625.log: about 300 sentences per second forward on one RTX 6000 Ada, batch 32):
  3D  50 configurations x 872 x (1 + 2 forms x K)   K = 16 (dropout_theory.md): about 80 min; --forms coord or --k 8 halves it
  2A  50 configurations x 6,920                       about 20 min
  3A  15 (PZ, seed) x (9 + 2 x len(split_layers)) x 2,730   split 6,12,18: about 35 min
  3B  45 adapters x 2 x (dev positives)                about 5 min
  total about 2.3 h; every part prints elapsed seconds, so the first PZ gives the real rate.
Outputs (refuses to overwrite without --overwrite): <out_dir>/{noise,train_margin,swap}/*.npz, summary_3D.csv, summary_2A.csv,
summary_3A.csv, summary_3B.csv, pairs.csv (the pre-registered paired counts; for 3A also the median share of the full change in
median(s)/median(m) that swapping only the head, or only the updates, reproduces), checks.csv, env.json.
Test metrics are for the experimenter only (claude/rules.md, your rule 2); 3D and 2A use dev and train.
"""
import os
import argparse, csv, json, re, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin_eval as E          # sets CUBLAS_WORKSPACE_CONFIG; its checks and scoring are reused unchanged
import margin_metrics as M
import ft_logs as F

PARTS = ("3D", "2A", "3A", "3B")
FORMS = ("coord", "svd")
RE_QV = re.compile(r"encoder\.layer\.(\d+)\.attention\.self\.(query|value)$")
RE_LAYER = re.compile(r"\.layer\.(\d+)\.")
OUT_FILES = ("summary_3D.csv", "summary_2A.csv", "summary_3A.csv", "summary_3B.csv", "pairs.csv", "checks.csv")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--parts", default=",".join(PARTS))
    p.add_argument("--choice", default="logs/wpre/choice_bonf_stepwise.tsv")
    p.add_argument("--adapters", default="adapters")
    p.add_argument("--data_dir", default="./data/sst-2", help="holds train.json, dev.json, test.json")
    p.add_argument("--repo", default=".", help="PSIM folder (for logs/*.log)")
    p.add_argument("--margin_dir", default="logs/margin_decomp", help="margin_eval.py's output (npz/ is used by the 3A check)")
    p.add_argument("--out_dir", default="logs/mech_eval")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hf", default="roberta-large")
    p.add_argument("--p", type=float, default=0.2, help="3D: dropout rate switched on at evaluation (the trained arms used 0.2)")
    p.add_argument("--k", type=int, default=16, help="3D: masks per form (dropout_theory.md: 16)")
    p.add_argument("--forms", default="coord,svd", help="3D: noise forms")
    p.add_argument("--mask_seed", type=int, default=1234)
    p.add_argument("--n_train", type=int, default=0, help="2A: 0 = all of train.json; otherwise a fixed random subset (seed 0)")
    p.add_argument("--split_layers", default="6,12,18", help="3A: L values; head + updates of layers >= L from a dropout arm")
    p.add_argument("--dev_tol", type=float, default=2e-3, help="3D check: dev sentences with |m| below this may differ from the log")
    p.add_argument("--tol", type=float, default=1e-4, help="3A check: largest allowed |score difference| to margin_eval.py's scores")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_meta", action="store_true", help="as in margin_eval.py")
    p.add_argument("--allow_cpu", action="store_true")
    a = p.parse_args(argv)
    a.parts = [x.strip() for x in a.parts.split(",") if x.strip()]
    if not a.parts or any(x not in PARTS for x in a.parts):
        p.error(f"--parts must be a comma-separated subset of {PARTS}")
    a.forms = [x.strip() for x in a.forms.split(",") if x.strip()]
    if any(x not in FORMS for x in a.forms):
        p.error(f"--forms must be a subset of {FORMS}")
    a.split_layers = [int(x) for x in a.split_layers.split(",") if x.strip()]
    if any(not 0 < L < 24 for L in a.split_layers):
        p.error("--split_layers must lie in 1..23 (0 and 24 are the full swaps already in the 3 x 3 grid)")
    return a


# ---------------------------------------------------------------- plan (no model)
def logged_dev(repo, names, n_epochs):
    """{name: "0.xxxx"}: last_dev_acc of every COMPLETED occurrence of the run (all must agree), equal to the dev accuracy
    printed in the [EPOCH] line of the last epoch. Raises with every problem listed."""
    R = F.runs(F.all_logs(repo))
    out, problems = {}, []
    for name in names:
        occ = [r for r in R.get(name, []) if r["summary"] is not None and len(r["epochs"]) == n_epochs and "last_dev_acc" in r["summary"]]
        if not occ:
            problems.append(f"{name}: no completed run with last_dev_acc in logs/*.log")
            continue
        vals = {M.fmt4(r["summary"]["last_dev_acc"]) for r in occ}
        if len(vals) != 1:
            problems.append(f"{name}: completed runs disagree on last_dev_acc: {sorted(vals)}")
            continue
        v = vals.pop()
        if M.fmt4(occ[-1]["epochs"][n_epochs - 1][0]) != v:
            problems.append(f"{name}: SUMMARY last_dev_acc {v} differs from the [EPOCH] line of the last epoch")
            continue
        out[name] = v
    if problems:
        raise ValueError("\n".join(problems))
    return out


def build_plan(a):
    problems, seeds = [], [int(x) for x in a.seeds.split(",")]
    files = {k: os.path.join(a.data_dir, f"{k}.json") for k in ("train", "dev", "test")}
    for f in [a.choice] + list(files.values()):
        if not os.path.exists(f):
            problems.append(f"missing {f}")
    if problems:
        return None, problems
    test_sha = M.file_sha256(files["test"])[:12]
    plan = {"seeds": seeds, "files": files, "test_sha256": test_sha, "pz": []}
    names = []
    for label, n, pz_seed, pz_ep, pz in M.read_panel(a.choice):
        P = {"label": label, "pz": pz, "pz_sha256": None, "name": f"pz_{label}", "adapters": []}
        if os.path.exists(pz):
            P["pz_sha256"] = M.file_sha256(pz)[:12]
        else:
            problems.append(f"missing PZ {pz}")
        for arm in M.ARMS:
            for s in seeds:
                name = M.adapter_name(arm, s, pz)
                d = os.path.join(a.adapters, name, "last")
                P["adapters"].append({"arm": arm, "seed": s, "name": name, "dir": d})
                names.append(name)
                problems += E.check_adapter_dir(a, d, name, s, test_sha, P["pz_sha256"])
                if "3A" in a.parts:
                    ref = os.path.join(a.margin_dir, "npz", name + ".npz")
                    if not os.path.exists(ref):
                        problems.append(f"3A check needs {ref} (margin_eval.py's scores of this adapter)")
        plan["pz"].append(P)
    if len(plan["pz"]) != 5:
        print(f"[PLAN] note: {len(plan['pz'])} PZs in {a.choice} (the panel has 5)")
    try:
        plan["dev_logged"] = logged_dev(a.repo, names, a.epochs)
    except ValueError as e:
        plan["dev_logged"] = {}
        problems += str(e).split("\n")
    for sub in ("noise", "train_margin", "swap"):
        dd = os.path.join(a.out_dir, sub)
        if os.path.isdir(dd) and os.listdir(dd) and not a.overwrite:
            problems.append(f"{dd} is not empty: use a new --out_dir or pass --overwrite")
    for fn in OUT_FILES:
        f = os.path.join(a.out_dir, fn)
        if os.path.exists(f) and not a.overwrite:
            problems.append(f"{f} exists: use a new --out_dir or pass --overwrite")
    return plan, problems


# ---------------------------------------------------------------- pure helpers (numpy; tested on CPU)
def noise_stats(m, noisy):
    """m (n,) margins without noise, noisy (K, n). Per sentence: rel_sd = sd(noisy, ddof 1) / |m| (inf where m = 0 is
    reported as nan), flip = share of the K masks whose sign differs from m's (a tie m = 0 counts as negative)."""
    m = np.asarray(m, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    sd = noisy.std(axis=0, ddof=1) if noisy.shape[0] > 1 else np.zeros_like(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(m) > 0, sd / np.abs(m), np.nan)
    flip = ((noisy > 0) != (m > 0)[None, :]).mean(axis=0)
    shift = noisy.mean(axis=0) - m
    return rel, sd, flip, shift


def compose(sd_by_arm, head_arm, dw_arm, split=None, low_arm="plain"):
    """The adapter state dict of one combination: classifier tensors from head_arm; LoRA tensors of layer l from dw_arm,
    except that with a split L, layers < L come from low_arm. Keys are those of the saved files (all arms have the same keys)."""
    keys = sorted(sd_by_arm[dw_arm])
    for arm, sd in sd_by_arm.items():
        if sorted(sd) != keys:
            raise ValueError(f"adapter files of {dw_arm} and {arm} hold different tensors")
    out = {}
    for k in keys:
        if ".classifier." in k:
            out[k] = sd_by_arm[head_arm][k]
        elif "lora_" in k:
            mt = RE_LAYER.search(k)
            if mt is None:
                raise ValueError(f"no layer index in {k}")
            arm = dw_arm if split is None or int(mt.group(1)) >= split else low_arm
            out[k] = sd_by_arm[arm][k]
        else:
            raise ValueError(f"unexpected tensor {k} in the adapter file")
    return out


def combo_name(label, seed, head, dw, split=None):
    return f"swap_h-{head}_dw-{dw}" + (f"_from{split}" if split is not None else "") + f"_seed{seed}_pz-{label}"


def trigger_span(c_ids, t_ids):
    """(start, k) of the k tokens inserted into c_ids to give t_ids, or None if the rest of the sentence also changed
    (byte-pair merges across the insertion point)."""
    k = len(t_ids) - len(c_ids)
    if k <= 0:
        return None
    i = 0
    while i < len(c_ids) and c_ids[i] == t_ids[i]:
        i += 1
    if list(t_ids[:i]) == list(c_ids[:i]) and list(t_ids[i + k:]) == list(c_ids[i:]):
        return i, k
    return None


def paired_counts(rows, key, arm_of, better, forms=None):
    """count, for each dropout arm, the (PZ, seed) pairs where better(value of the arm, value of plain) holds."""
    out = []
    idx = {(r["arm"], r["pz"], r["seed"], r.get("form")): r for r in rows}
    for arm in M.DROPOUT_ARMS:
        for form in (forms or [None]):
            pairs = [(idx[k], idx[("plain",) + k[1:]]) for k in idx if k[0] == arm and k[3] == form and ("plain",) + k[1:] in idx]
            n_ok = sum(bool(better(d[key], p[key])) for d, p in pairs)
            out.append({"arm": arm, "form": form or "-", "key": key, "n_pairs": len(pairs), "n_hold": n_ok})
    return out


# ---------------------------------------------------------------- model helpers (torch imported inside)
def load_combo(peft_model, sd, device):
    """put the composed adapter state dict into the model in place; read it back bit for bit (else raise)."""
    import torch
    from peft import set_peft_model_state_dict, get_peft_model_state_dict
    set_peft_model_state_dict(peft_model, sd, adapter_name="default")
    peft_model.to(device)
    peft_model.eval()
    now = get_peft_model_state_dict(peft_model, adapter_name="default", save_embedding_layers=False)
    if set(now) != set(sd):
        raise RuntimeError(f"adapter tensors in the model and in the composition differ: {sorted(set(now) ^ set(sd))[:3]}")
    bad = [k for k in sd if not torch.equal(now[k].detach().cpu(), sd[k].cpu())]
    if bad:
        raise RuntimeError(f"{len(bad)} tensors in the model differ from the composition after loading, e.g. {bad[:3]}")


class FrozenNoise:
    """forward hook on the 48 frozen query/value Linear modules (they become peft's base_layer when the model is wrapped):
    off by default; on, it applies exactly the dropout of training (cl_utils.cl_apply's two forms) with training=True."""
    def __init__(self):
        self.form, self.p, self.U = None, 0.0, {}

    def __call__(self, module, inputs, output):
        if self.form is None or self.p == 0:
            return output
        import torch.nn.functional as Fn
        import cl_utils
        if self.form == "coord":
            return Fn.dropout(output, p=self.p, training=True)
        return cl_utils.svd_dropout(output, module.bias, self.U[module], self.p, training=True)


def margins(model, loader, device):
    lg, y = E.score(model, loader, device)
    return (lg[:, 1] - lg[:, 0]).astype(np.float32), y


# ---------------------------------------------------------------- main
def main(argv=None):
    a = parse_args(argv)
    plan, problems = build_plan(a)
    if plan is not None:
        for P in plan["pz"]:
            print(f"[PLAN] {P['label']} <- {P['pz']} (sha256 {P['pz_sha256']}); {len(P['adapters'])} adapters")
        print(f"[PLAN] parts {','.join(a.parts)}; data {plan['files']}; out {a.out_dir}")
    if problems:
        for x in problems:
            print(f"[CHECK] FAIL {x}")
        print(f"[CHECK] {len(problems)} problem(s); nothing was evaluated")
        sys.exit(1)
    n_ad = sum(len(P["adapters"]) for P in plan["pz"])
    print(f"[CHECK] inputs ok: {len(plan['pz'])} PZs, {n_ad} adapters; dev accuracy found in the logs for all {len(plan['dev_logged'])}")
    if a.dry:
        sys.exit(0)

    import random
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file
    import attack_utils, cl_utils
    from attack_utils import build_poisoned_test_dataloader
    from repro_utils import assert_env, assert_gpu, log_env, PLACEMENT_SEED

    assert_env()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        assert_gpu()
    elif not a.allow_cpu:
        sys.exit("no CUDA device visible (set CUDA_VISIBLE_DEVICES to the Ada; --allow_cpu runs on CPU)")
    ENV = log_env(__file__, {"test": plan["files"]["test"], "dev": plan["files"]["dev"], "train": plan["files"]["train"],
                             "margin_eval": E.__file__, "margin_metrics": M.__file__, "cl_utils": cl_utils.__file__})
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "env.json"), "w") as f:
        json.dump(dict(ENV, args=vars(a)), f, indent=1, default=str)
    for fn in OUT_FILES:
        if os.path.exists(os.path.join(a.out_dir, fn)):
            os.remove(os.path.join(a.out_dir, fn))
    for sub in ("noise", "train_margin", "swap"):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)

    # ---- data (the loaders of variant_finetune.py)
    tok = AutoTokenizer.from_pretrained(a.hf, padding_side="right")
    def tokenize_function(ex):
        return tok(ex["sentence"], truncation=True, max_length=None)
    def collate_fn(ex):
        return tok.pad(ex, padding="longest", return_tensors="pt")
    def clean_loader(path):
        raw = load_dataset("json", data_files=path)["train"]
        ds = raw.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"]).rename_column("label", "labels")
        return DataLoader(ds, shuffle=False, collate_fn=collate_fn, batch_size=a.batch_size), \
            [{"idx": int(r["idx"]), "label": int(r["label"]), "sentence": r["sentence"]} for r in raw]
    L = {}
    if "3D" in a.parts or "3B" in a.parts:
        L["dev"], dev_rows = clean_loader(plan["files"]["dev"])
    if "2A" in a.parts:
        train_path = plan["files"]["train"]
        if a.n_train:
            with open(train_path) as f:
                rows = [l for l in f if l.strip()]
            keep = sorted(np.random.default_rng(0).choice(len(rows), size=min(a.n_train, len(rows)), replace=False).tolist())
            train_path = os.path.join(a.out_dir, "train_subset.json")
            with open(train_path, "w") as f:
                f.writelines(rows[i] for i in keep)
        L["train"], train_rows = clean_loader(train_path)
    if "3A" in a.parts:
        L["test"], test_rows = clean_loader(plan["files"]["test"])
        L["trig"] = build_poisoned_test_dataloader(plan["files"]["test"], load_dataset, tokenize_function, collate_fn, M.TRIGGER, batch_size=a.batch_size)
        t_lab = np.array([r["label"] for r in test_rows])
        t_pos = t_lab == 1
        t_idx = np.array([r["idx"] for r in test_rows])
        placement = M.build_placement(test_rows, M.TRIGGER, attack_utils.insert_trigger, PLACEMENT_SEED)
        t_ins = np.array([p["insert_pos"] for p in placement])
    if "3B" in a.parts:
        dev_pos = [r for r in dev_rows if r["label"] == 1]
        pl = M.build_placement(dev_pos, M.TRIGGER, attack_utils.insert_trigger, PLACEMENT_SEED)
        c_ids = tok([r["sentence"] for r in dev_pos], truncation=True, max_length=None)["input_ids"]
        t_ids = tok([p["sentence_trig"] for p in pl], truncation=True, max_length=None)["input_ids"]
        spans = [trigger_span(c, t) for c, t in zip(c_ids, t_ids)]
        print(f"[DATA] 3B: {len(dev_pos)} dev positives; trigger tokens located in {sum(s is not None for s in spans)} "
              f"(the others are used for r_cls only)")

    checks, R3D, R2A, R3A, R3B, t0 = [], [], [], [], [], time.time()
    def check(name, what, ok, detail):
        checks.append({"name": name, "check": what, "ok": int(ok), "detail": detail})
        with open(os.path.join(a.out_dir, "checks.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(checks[0].keys())); w.writeheader(); w.writerows(checks)
        print(f"[CHECK] {name}: {what}: {'ok' if ok else 'FAIL'} ({detail})")
        if not ok:
            print("[CHECK] FAIL: stopped; summaries were not written")
            sys.exit(1)

    noise = FrozenNoise()

    def run_3d_2a(model, cfg, P):
        meta = {"arm": cfg["arm"], "pz": P["label"], "seed": cfg["seed"], "name": cfg["name"]}
        if "3D" in a.parts:
            noise.form = None
            m0, y = margins(model, L["dev"], device)
            acc = float(((m0 > 0).astype(int) == y).mean())
            if cfg["arm"] != M.PZ_ARM:
                k_mine = int(((m0 > 0).astype(int) == y).sum())
                ref = plan["dev_logged"][cfg["name"]]
                k_log = int(round(float(ref) * len(y)))         # 4 decimals fix the count: 1/872 > 1e-4
                n_near = int((np.abs(m0) < a.dev_tol).sum())
                check(cfg["name"], "noise-free dev accuracy vs last_dev_acc of the training log", abs(k_mine - k_log) <= n_near,
                      f"mine {k_mine}/{len(y)} = {M.fmt4(k_mine / len(y))}, log {ref} = {k_log}/{len(y)}; sentences with |m| < {a.dev_tol:g}: {n_near}")
            store = {"m": m0, "label": y}
            for fi, form in enumerate(a.forms):
                noise.form, noise.p = form, a.p
                noisy = []
                for k in range(a.k):
                    torch.manual_seed(a.mask_seed + 1000 * fi + k)       # the same masks for every configuration
                    noisy.append(margins(model, L["dev"], device)[0])
                noise.form = None
                noisy = np.stack(noisy)
                rel, sd, flip, shift = noise_stats(m0, noisy)
                store[f"noisy_{form}"] = noisy
                R3D.append(dict(meta, form=form, p=a.p, k=a.k, n=len(m0), dev_acc=acc, median_rel_sd=float(np.nanmedian(rel)),
                                q25_rel_sd=float(np.nanpercentile(rel, 25)), q75_rel_sd=float(np.nanpercentile(rel, 75)),
                                median_abs_m=float(np.median(np.abs(m0))), median_sd=float(np.median(sd)),
                                mean_flip=float(flip.mean()), median_shift=float(np.median(shift))))
            np.savez_compressed(os.path.join(a.out_dir, "noise", cfg["name"] + ".npz"), meta=np.array(json.dumps(meta)), **store)
        if "2A" in a.parts:
            m, y = margins(model, L["train"], device)
            np.savez_compressed(os.path.join(a.out_dir, "train_margin", cfg["name"] + ".npz"), meta=np.array(json.dumps(meta)), m=m, label=y)
            am = np.abs(m.astype(np.float64))
            R2A.append(dict(meta, n=len(m), train_acc=float(((m > 0).astype(int) == y).mean()), median_abs_m=float(np.median(am)),
                            q10_abs_m=float(np.percentile(am, 10)), q90_abs_m=float(np.percentile(am, 90)),
                            median_m_pos=float(np.median(m[y == 1])), median_m_neg=float(np.median(m[y == 0]))))

    def run_3b(peft_model, cfg, P):
        acc = {}
        def make_hook(name, lora_mod):
            mt = RE_QV.search(name)
            layer, proj = int(mt.group(1)), mt.group(2)
            def hook(module, inputs, output):
                x = inputs[0].float()
                W = module.weight.float()
                A_ = lora_mod.lora_A["default"].weight.float()
                B_ = lora_mod.lora_B["default"].weight.float()
                s = float(lora_mod.scaling["default"])
                Wx = x @ W.T
                Dx = s * ((x @ A_.T) @ B_.T)
                n = x.shape[0] // 2
                dW_, dD = Wx[n:, 0] - Wx[:n, 0], Dx[n:, 0] - Dx[:n, 0]
                r_cls = (dD.norm(dim=-1) / dW_.norm(dim=-1).clamp_min(1e-12)).cpu().numpy()
                r_trig, r_tok = np.full(n, np.nan), np.full(n, np.nan)
                for j in range(n):
                    sp, lc = cur["spans"][j], cur["lens"][j]
                    if sp is not None:
                        st, k = sp
                        r_trig[j] = float(Dx[n + j, st:st + k].norm() / Wx[n + j, st:st + k].norm().clamp_min(1e-12))
                    r_tok[j] = float(Dx[j, 1:lc - 1].norm() / Wx[j, 1:lc - 1].norm().clamp_min(1e-12))
                d = acc.setdefault((layer, proj), {"r_cls": [], "r_trig": [], "r_tok": []})
                d["r_cls"].append(r_cls); d["r_trig"].append(r_trig); d["r_tok"].append(r_tok)
            return hook
        hs = [mod.base_layer.register_forward_hook(make_hook(name, mod)) for name, mod in peft_model.named_modules()
              if hasattr(mod, "base_layer") and RE_QV.search(name)]
        if len(hs) != E.N_QV:
            sys.exit(f"3B: {len(hs)} query/value modules hooked, expected {E.N_QV}")
        cur = {}
        try:
            with torch.no_grad():
                for b0 in range(0, len(dev_pos), a.batch_size // 2):
                    J = range(b0, min(b0 + a.batch_size // 2, len(dev_pos)))
                    sents = [dev_pos[j]["sentence"] for j in J] + [pl[j]["sentence_trig"] for j in J]
                    enc = tok(sents, truncation=True, max_length=None, padding="longest", return_tensors="pt")
                    cur["spans"] = [spans[j] for j in J]
                    cur["lens"] = [len(c_ids[j]) for j in J]
                    peft_model(**{k: v.to(device) for k, v in enc.items()})
        finally:
            for h in hs:
                h.remove()
        for (layer, proj), d in sorted(acc.items()):
            rc, rt, rk = (np.concatenate(d[k]) for k in ("r_cls", "r_trig", "r_tok"))
            R3B.append({"arm": cfg["arm"], "pz": P["label"], "seed": cfg["seed"], "name": cfg["name"], "layer": layer, "proj": proj,
                        "n": len(rc), "n_trig": int(np.isfinite(rt).sum()), "median_r_cls": float(np.median(rc)),
                        "median_r_trig": float(np.nanmedian(rt)) if np.isfinite(rt).any() else float("nan"),
                        "median_r_tok": float(np.median(rk))})

    for P in plan["pz"]:
        base = AutoModelForSequenceClassification.from_pretrained(a.hf, return_dict=True)
        missing, _ = base.load_state_dict(torch.load(P["pz"], map_location="cpu"), strict=False)
        if missing:
            sys.exit(f"{P['label']}: the PZ state dict misses {list(missing)[:5]}")
        base.to(device).eval()
        qv = [(n, m) for n, m in base.named_modules() if RE_QV.search(n)]
        if len(qv) != E.N_QV:
            sys.exit(f"{len(qv)} query/value modules in the base model, expected {E.N_QV}")
        hooks = [m.register_forward_hook(noise) for _, m in qv]
        noise.U = {m: cl_utils.left_basis(m.weight.data).to(device) for _, m in qv} if ("3D" in a.parts and "svd" in a.forms) else {}
        if "3D" in a.parts or "2A" in a.parts:
            run_3d_2a(base, {"arm": M.PZ_ARM, "seed": None, "name": P["name"]}, P)
            print(f"[EVAL] {P['name']} done | {time.time() - t0:.0f}s", flush=True)
        sd = {(c["arm"], c["seed"]): load_file(os.path.join(c["dir"], "adapter_model.safetensors")) for c in P["adapters"]}
        peft_model = PeftModel.from_pretrained(base, P["adapters"][0]["dir"]).to(device)
        for c in P["adapters"]:
            if not ({"3D", "2A", "3B"} & set(a.parts)):
                break
            load_combo(peft_model, compose({x: sd[(x, c["seed"])] for x in M.ARMS}, c["arm"], c["arm"]), device)
            run_3d_2a(peft_model, c, P)
            if "3B" in a.parts:
                run_3b(peft_model, c, P)
            print(f"[EVAL] {c['name']} done | {time.time() - t0:.0f}s", flush=True)
        if "3A" in a.parts:
            for s in plan["seeds"]:
                by_arm = {x: sd[(x, s)] for x in M.ARMS}
                combos = [(h, d, None) for h in M.ARMS for d in M.ARMS] + [(x, x, Lx) for x in M.DROPOUT_ARMS for Lx in a.split_layers]
                for h, d, Lx in combos:
                    load_combo(peft_model, compose(by_arm, h, d, split=Lx), device)
                    lc, yc = E.score(peft_model, L["test"], device)
                    lt, yt = E.score(peft_model, L["trig"], device)
                    mc, mt_ = lc[:, 1] - lc[:, 0], lt[:, 1] - lt[:, 0]
                    name = combo_name(P["label"], s, h, d, Lx)
                    meta = {"arm": f"h-{h}|dw-{d}" + (f"|from{Lx}" if Lx is not None else ""), "pz": P["label"], "seed": s, "name": name,
                            "head": h, "dw": d, "split": Lx}
                    M.save_scores(os.path.join(a.out_dir, "swap", name + ".npz"), meta, t_idx[t_pos], t_ins, lc[t_pos], lt, t_idx[~t_pos], lc[~t_pos])
                    if h == d and Lx is None:
                        ref = M.load_scores(os.path.join(a.margin_dir, "npz", M.adapter_name(h, s, P["pz"]) + ".npz"))
                        ok, rep = M.check_reference(mc[t_pos], mc[~t_pos], mt_, ref["m_pos_clean"], ref["m_neg_clean"], ref["m_pos_trig"], tol=a.tol)
                        check(name, "diagonal combination vs margin_eval.py's stored scores", ok,
                              "; ".join(f"{k}: max|diff| {v['max_abs_diff']:.3g}, predictions changed {v['pred_changed']}" for k, v in rep.items()))
                    met = M.checkpoint_metrics(mc[t_pos], mt_, mc[~t_pos])
                    R3A.append(dict(pz=P["label"], seed=s, head=h, dw=d, split=-1 if Lx is None else Lx, name=name,
                                    median_abs_m_all=float(np.median(np.abs(mc.astype(np.float64)))),
                                    **{k: met[k] for k in ("clean_acc", "pos_err", "fpr", "asr_standard", "asr_clean_correct", "m_q50", "mneg_q50",
                                                           "mtrig_q50", "s_q50", "median_ratio", "clean_auc", "trigger_auc_allpairs", "p_s_gt0_paired")}))
                    print(f"[EVAL] {name}: acc {met['clean_acc']:.4f} ASR {met['asr_standard']:.4f} median m {met['m_q50']:.2f} "
                          f"s {met['s_q50']:.2f} s/m {met['median_ratio']:.3f} | {time.time() - t0:.0f}s", flush=True)
        for h in hooks:
            h.remove()
        del base, peft_model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- summaries (only reached when every check passed)
    def write(fn, rows):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(a.out_dir, fn), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    write("summary_3D.csv", R3D); write("summary_2A.csv", R2A); write("summary_3A.csv", R3A); write("summary_3B.csv", R3B)
    pairs = []
    own = {"cl0.2": "coord", "clsvd0.2": "svd"}
# DEPRECATED COLUMN NAME (2026-11-21):
# The column name "pre_registered" is deprecated and kept only for backward
# compatibility with the already-committed logs/mech_eval/pairs.csv (f1f5910).
# It only indicates whether the comparison appeared in the 2026-11-17 draft
# claude/dropout_theory.md (20/22 rows = 1; two 3D cross-form rows = 0),
# NOT a pre-approved statistical decision rule. Do not interpret as formal
# preregistration; existing pairs.csv is intentionally left unmodified.
    if R3D:
        pairs += [dict(r, part="3D", criterion="median_rel_sd lower than plain (same PZ, seed, noise form)",
                       pre_registered=int(own.get(r["arm"]) == r["form"]))
                  for r in paired_counts(R3D, "median_rel_sd", None, lambda d, p: d < p, forms=a.forms)]
    if R2A:
        pairs += [dict(r, part="2A", criterion="median |m| on train larger than plain", pre_registered=1)
                  for r in paired_counts(R2A, "median_abs_m", None, lambda d, p: d > p)]
        pairs += [dict(r, part="2A", criterion="median |m| on train smaller than plain", pre_registered=1)
                  for r in paired_counts(R2A, "median_abs_m", None, lambda d, p: d < p)]
    if R3A:
        idx = {(r["pz"], r["seed"], r["head"], r["dw"], r["split"]): r for r in R3A}
        keys = sorted({(pz, s) for pz, s, *_ in idx})
        med = lambda v: float(np.median(v)) if v else float("nan")
        for arm in M.DROPOUT_ARMS:
            # one row per combination that mixes this dropout arm with plain LoRA, plus the full dropout run for reference
            combos = [("full dropout run", arm, arm, -1), ("head only", arm, "plain", -1), ("48 updates only", "plain", arm, -1)]
            combos += [(f"head + updates of layers >= {Lx}", arm, arm, Lx) for Lx in a.split_layers]
            for what, h, d, Lx in combos:
                below, share, d_acc, d_m = 0, [], [], []
                for pz, s in keys:
                    r, b, f = idx[(pz, s, h, d, Lx)], idx[(pz, s, "plain", "plain", -1)], idx[(pz, s, arm, arm, -1)]
                    below += r["median_ratio"] < b["median_ratio"]
                    if f["median_ratio"] != b["median_ratio"]:
                        share.append((r["median_ratio"] - b["median_ratio"]) / (f["median_ratio"] - b["median_ratio"]))
                    d_acc.append(r["clean_acc"] - b["clean_acc"]); d_m.append(r["m_q50"] - b["m_q50"])
                pairs.append({"arm": arm, "form": "-", "key": "median_ratio", "n_pairs": len(keys), "n_hold": below, "part": "3A",
                              "criterion": f"{what}: median(s)/median(m) below plain LoRA's", "pre_registered": 1,
                              "median_share_of_full_change": med(share), "median_clean_acc_minus_plain": med(d_acc),
                              "median_m_q50_minus_plain": med(d_m)})
            n_head = 0
            for pz, s in keys:
                b = idx[(pz, s, "plain", "plain", -1)]["median_ratio"]
                n_head += abs(idx[(pz, s, arm, "plain", -1)]["median_ratio"] - b) > abs(idx[(pz, s, "plain", arm, -1)]["median_ratio"] - b)
            pairs.append({"arm": arm, "form": "-", "key": "median_ratio", "n_pairs": len(keys), "n_hold": n_head, "part": "3A",
                          "criterion": "swapping only the head moves median(s)/median(m) more than swapping only the 48 updates",
                          "pre_registered": 1})
    write("pairs.csv", pairs)
    print(f"\n[CHECK] all {len(checks)} checks passed; wrote summaries to {a.out_dir}")
    for r in pairs:
        extra = ""
        if "median_share_of_full_change" in r:
            extra = (f" | share of full change {r['median_share_of_full_change']:.2f}, clean acc vs plain {100 * r['median_clean_acc_minus_plain']:+.2f} pp, "
                     f"median m vs plain {r['median_m_q50_minus_plain']:+.2f}")
        tag = "" if r["pre_registered"] else " (not pre-registered)"
        print(f"[PAIRS] {r['part']} {r['arm']:9s} {r['form']:5s} {r['criterion']}: {r['n_hold']}/{r['n_pairs']}{tag}{extra}")
    if R3A:
        print("\n3A, mean over the seeds of each PZ (clean acc %, median clean margin m, median |m| all clean, median(s)/median(m), ASR %):")
        for Pz in plan["pz"]:
            for h, d, Lx in [(h, d, -1) for h in M.ARMS for d in M.ARMS] + [(x, x, Lx) for x in M.DROPOUT_ARMS for Lx in a.split_layers]:
                rs = [r for r in R3A if r["pz"] == Pz["label"] and r["head"] == h and r["dw"] == d and r["split"] == Lx]
                lab = f"updates {d:9s}" if Lx == -1 else f"layers>={Lx:<2d} {d:6s}"
                print(f"  {Pz['label']:13s} head {h:9s} {lab} acc {100 * np.mean([r['clean_acc'] for r in rs]):6.2f} "
                      f"m {np.mean([r['m_q50'] for r in rs]):6.2f} |m| {np.mean([r['median_abs_m_all'] for r in rs]):6.2f} "
                      f"s/m {np.mean([r['median_ratio'] for r in rs]):6.3f} ASR {100 * np.mean([r['asr_standard'] for r in rs]):6.2f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
