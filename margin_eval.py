#!/usr/bin/env python3
"""margin_eval.py — paired margin decomposition on SST-2 test (evaluation only: no training, nothing existing is modified).

What it scores (50 configurations): for each of the 5 panel PZs in --choice
  the PZ itself before fine-tuning (its own classification head, no LoRA update)            5
  the last-epoch adapter adapters/<name>/last/ of plain LoRA, cl0.2, clsvd0.2, seeds 0-2    45
on the same three sets of sentences: clean positive, the same positives with "mn" inserted, clean negative.
Not included on purpose: clsvd0.3, the truncated (W_pre,K) arms, head swapping, rescaling of the update.

How (everything that touches the model mirrors existing code, so the numbers can be checked against the logs)
  data      load_dataset + tokenizer(sentence, truncation=True, max_length=None) + tokenizer.pad(longest), batch 32, no shuffle,
            the whole test file in file order: the test_dataloader of variant_finetune.py.
  trigger   attack_utils.build_poisoned_test_dataloader (random.Random(PLACEMENT_SEED = 1234)): the loader behind every logged ASR.
            The placement table (which word index got "mn") is rebuilt with attack_utils.insert_trigger and checked token by
            token against that loader, so the per-sentence ids in the output are the sentences that were actually scored.
  model     roberta-large + the PZ's state dict, built ONCE per PZ. The PZ is scored first; then PeftModel.from_pretrained
            wraps it once and every adapter is put in place with load_adapter(dir, "default") (weights only, same adapter
            slot), each time followed by a bit-for-bit read-back of all 100 tensors (48 lora_A, 48 lora_B, 4 classifier).
  state     model.eval(), no_grad, float32, the deterministic settings of variant_finetune.py, the Ada (assert_gpu).
  numbers   m = logit_1 - logit_0; definitions in margin_metrics.py.

Two checks, each stops the run (exit 1) the moment it fails; a failed configuration's .npz is still on disk for inspection:
  1. the PZ before fine-tuning: its scores against logs/scores/dose_n<n>_s<seed>/ep<epoch>.npz (down_pos, down_neg, down_asr):
     largest |difference| <= --tol (1e-4) and no sentence's prediction changes. (The stored clean scores were batched per
     class, here per file, so the padding differs; the triggered scores use the same batches.)
  2. every adapter: test accuracy and ASR, printed with 4 decimals, must equal last_test_acc and last_ASR of its own
     training log (the last epoch, before Pt).
The csv summaries are written only after all 50 configurations passed.

Output (new files only; refuses to overwrite unless --overwrite), under --out_dir (default logs/margin_decomp):
  npz/<name>.npz              per sentence: idx, the two logits, m(x), m(x + mn), s(x), insert_pos  (margin_metrics.save_scores)
  placement.csv               idx, n_words, insert_pos, the clean and the triggered sentence
  sanity_checks.csv           one row per configuration: which check, what was compared, pass or fail
  summary_checkpoints.csv, summary_pz.csv, counterfactual_checkpoints.csv, counterfactual_pz.csv   (margin_metrics.write_summaries)

  python margin_eval.py --dry      no GPU, loads no model: are the 5 PZs, 45 adapters, their meta.json, the logged numbers and
                                   the stored PZ scores all there and consistent?
  python margin_eval.py            the evaluation (about 50 x (1821 + 909) short sentences; one card)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch / any CUDA context, as in variant_finetune.py

import argparse, csv, json, sys, time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin_metrics as M

N_QV = 48            # query and value of 24 layers
N_HEAD_KEYS = 4      # classifier dense / out_proj, weight and bias
OWN_FILES = M.SUMMARY_FILES + ("sanity_checks.csv", "placement.csv")     # what this script writes next to npz/


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--choice", default="logs/wpre/choice_bonf_stepwise.tsv", help="the panel PZs (label, r, json, pz); only label and pz are used")
    p.add_argument("--adapters", default="adapters")
    p.add_argument("--data", default="./data/sst-2/test.json")
    p.add_argument("--ref_dir", default="logs/scores", help="where dose_n<n>_s<seed>/ep<epoch>.npz are")
    p.add_argument("--repo", default=".", help="PSIM folder (for logs/*.log)")
    p.add_argument("--out_dir", default="logs/margin_decomp")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=20, help="epochs of the fine-tuning runs; last/ must hold epoch epochs - 1")
    p.add_argument("--batch_size", type=int, default=32, help="32 = what produced the logged numbers")
    p.add_argument("--tol", type=float, default=1e-4, help="check 1: largest allowed |score difference| to the stored PZ scores")
    p.add_argument("--hf", default="roberta-large")
    p.add_argument("--dry", action="store_true", help="check the inputs and exit; no GPU, no model")
    p.add_argument("--overwrite", action="store_true", help="allow replacing files of an earlier run of this script in --out_dir")
    p.add_argument("--skip_meta", action="store_true",
                   help="do not compare adapters' meta.json (hashes, seed, epoch) with the files on disk; the two checks still gate every number")
    p.add_argument("--allow_cpu", action="store_true", help="run without CUDA (slow; check 2 may then fail on rounding)")
    return p.parse_args(argv)


# ---------------------------------------------------------------- plan: everything that needs no model
def read_test_rows(path):
    """idx, label, sentence of data/sst-2/test.json (one json object per line), in file order. Only for counting in the plan;
    the evaluation itself reads the file with datasets.load_dataset, exactly like variant_finetune.py."""
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_plan(a):
    problems, seeds = [], [int(x) for x in a.seeds.split(",")]
    for f in (a.choice, a.data):
        if not os.path.exists(f):
            problems.append(f"missing {f}")
    if problems:
        return None, problems
    rows = read_test_rows(a.data)
    n_pos, n_neg = sum(r["label"] == 1 for r in rows), sum(r["label"] == 0 for r in rows)
    test_sha = M.file_sha256(a.data)[:12]
    npz_dir = os.path.join(a.out_dir, "npz")
    plan = {"seeds": seeds, "n_pos": n_pos, "n_neg": n_neg, "test_sha256": test_sha, "npz_dir": npz_dir, "pz": []}
    panel = M.read_panel(a.choice)
    names = []
    for label, n, pz_seed, pz_ep, pz in panel:
        P = {"label": label, "pz": pz, "pz_sha256": None, "ref": os.path.join(a.ref_dir, f"dose_n{n}_s{pz_seed}", f"ep{pz_ep:02d}.npz"),
             "name": f"pz_{label}", "adapters": []}
        if os.path.exists(pz):
            P["pz_sha256"] = M.file_sha256(pz)[:12]
            mp = os.path.join(os.path.dirname(pz), "meta.json")          # as sst2dev_eval.py: the folder must hold the epoch its label says
            if os.path.exists(mp):
                m = M.read_json(mp)
                if int(m.get("epoch_saved", -1)) != pz_ep:
                    problems.append(f"{mp}: epoch_saved={m.get('epoch_saved')}, label {label} says {pz_ep}")
                if "seed" in m and int(m["seed"]) != pz_seed:
                    problems.append(f"{mp}: seed={m['seed']}, label {label} says {pz_seed}")
                if "poison_n" in m and int(m["poison_n"]) != n:
                    problems.append(f"{mp}: poison_n={m['poison_n']}, label {label} says {n}")
        else:
            problems.append(f"missing PZ {pz}")
        if os.path.exists(P["ref"]):
            try:
                ref = M.read_reference(P["ref"])
                for k, arr, want in (("down_pos", ref[0], n_pos), ("down_neg", ref[1], n_neg), ("down_asr", ref[2], n_pos)):
                    if arr.shape != (want,):
                        problems.append(f"{P['ref']}: {k} has shape {arr.shape}, the test file has {want} such sentences")
            except ValueError as e:
                problems.append(str(e))
        else:
            problems.append(f"missing stored PZ scores {P['ref']}")
        for arm in M.ARMS:
            for s in seeds:
                name = M.adapter_name(arm, s, pz)
                d = os.path.join(a.adapters, name, "last")
                P["adapters"].append({"arm": arm, "seed": s, "name": name, "dir": d})
                names.append(name)
                problems += check_adapter_dir(a, d, name, s, test_sha, P["pz_sha256"])
        for nm in [P["name"]] + [x["name"] for x in P["adapters"]]:
            f = os.path.join(npz_dir, nm + ".npz")
            if os.path.exists(f) and not a.overwrite:
                problems.append(f"{f} exists: use a new --out_dir or pass --overwrite")
        plan["pz"].append(P)
    for fn in OWN_FILES:
        f = os.path.join(a.out_dir, fn)
        if os.path.exists(f) and not a.overwrite:
            problems.append(f"{f} exists: use a new --out_dir or pass --overwrite")
    if len(panel) != 5:
        problems.append(f"{a.choice}: {len(panel)} poisoned PZs, expected the 5 panel PZs")
    try:
        plan["logged"] = M.logged_last(a.repo, names, n_epochs=a.epochs)
    except ValueError as e:
        plan["logged"] = {}
        problems += str(e).split("\n")
    return plan, problems


def check_adapter_dir(a, d, name, seed, test_sha, pz_sha):
    """the last-epoch adapter folder: files present, the LoRA config is the one of these runs, meta.json describes this file"""
    out = []
    files = {k: os.path.join(d, k) for k in ("adapter_model.safetensors", "adapter_config.json", "meta.json")}
    for k, f in files.items():
        if not os.path.exists(f):
            out.append(f"missing {f}")
    if out:
        return out
    cfg = M.read_json(files["adapter_config.json"])
    want = {"peft_type": "LORA", "task_type": "SEQ_CLS", "r": 8, "lora_alpha": 16}
    for k, v in want.items():
        if cfg.get(k) != v:
            out.append(f"{files['adapter_config.json']}: {k}={cfg.get(k)!r}, expected {v!r}")
    if sorted(cfg.get("target_modules") or []) != ["query", "value"]:
        out.append(f"{files['adapter_config.json']}: target_modules={cfg.get('target_modules')!r}, expected query and value")
    if "classifier" not in (cfg.get("modules_to_save") or []):
        out.append(f"{files['adapter_config.json']}: modules_to_save={cfg.get('modules_to_save')!r} does not list the classifier")
    if cfg.get("use_dora") or cfg.get("use_rslora"):
        out.append(f"{files['adapter_config.json']}: use_dora / use_rslora is set")
    if a.skip_meta:
        return out
    m = M.read_json(files["meta.json"])
    exp = {"epoch_saved": a.epochs - 1, "seed": seed, "experiment_name": name, "test_sha256": test_sha}
    if pz_sha is not None:
        exp["pz_sha256"] = pz_sha
    for k, v in exp.items():
        if k not in m:
            out.append(f"{files['meta.json']}: no {k}")
        elif m[k] != v:
            out.append(f"{files['meta.json']}: {k}={m[k]!r}, expected {v!r}")
    if "wpre_json_sha256" in m:
        out.append(f"{files['meta.json']}: made on a truncated W_pre,K (wpre_json_sha256 present); this script is for the untruncated arms")
    if "checkpoint_sha256" not in m:
        out.append(f"{files['meta.json']}: no checkpoint_sha256")
    elif m["checkpoint_sha256"] != M.file_sha256(files["adapter_model.safetensors"]):
        out.append(f"{files['adapter_model.safetensors']}: sha256 differs from the one meta.json recorded when it was saved")
    return out


# ---------------------------------------------------------------- model-side helpers (torch is imported only when they run)
def score(model, loader, device):
    """logits (n, 2) float32 and labels, in loader order: the same call as variant_finetune.evaluate_accuracy and
    attack_utils.compute_asr (model.eval(), no_grad, model(**batch).logits)."""
    import torch
    model.eval()
    L, Y = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            L.append(model(**batch).logits.float().cpu())
            Y.append(batch["labels"].cpu())
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def switch_adapter(peft_model, base, adapter_dir, device):
    """Put the adapter saved in adapter_dir (LoRA A and B of every q/v module, and the trained classification head) into the
    model, in place. Returns (peft_model, number of lora_A / lora_B / classifier tensors in the file).
      first call (peft_model is None): PeftModel.from_pretrained(base, adapter_dir) wraps `base` once; it modifies `base`
        itself, so the PZ must have been scored before.
      every call: load_adapter(adapter_dir, "default"). In peft 0.11.1, when the adapter name already exists, this only loads
        the weights into that adapter (set_peft_model_state_dict) and puts the model in eval mode.
      then every tensor of the file is read back out of the model (get_peft_model_state_dict, the function save_pretrained
        used to write the file) and must be bit-identical; otherwise the switch did not happen and the run stops."""
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    if peft_model is None:
        peft_model = PeftModel.from_pretrained(base, adapter_dir)
    peft_model.to(device)
    peft_model.load_adapter(adapter_dir, adapter_name="default")
    peft_model.to(device)
    peft_model.eval()
    file_sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    now = get_peft_model_state_dict(peft_model, adapter_name="default", save_embedding_layers=False)
    if set(now) != set(file_sd):
        only_f, only_m = sorted(set(file_sd) - set(now)), sorted(set(now) - set(file_sd))
        raise RuntimeError(f"{adapter_dir}: adapter tensors in the file and in the model differ: only in file {only_f[:3]}, only in model {only_m[:3]}")
    bad = [k for k in file_sd if not torch.equal(now[k].detach().cpu(), file_sd[k].cpu())]
    if bad:
        raise RuntimeError(f"{adapter_dir}: {len(bad)} tensors in the model differ from the file after loading, e.g. {bad[:3]}")
    return peft_model, sum("lora_A" in k for k in file_sd), sum("lora_B" in k for k in file_sd), sum("classifier" in k for k in file_sd)


# ---------------------------------------------------------------- main
def main(argv=None):
    a = parse_args(argv)
    plan, problems = build_plan(a)
    if plan is not None:
        print(f"[PLAN] test file {a.data}: {plan['n_pos']} positive, {plan['n_neg']} negative, sha256 {plan['test_sha256']}")
        for P in plan["pz"]:
            print(f"[PLAN] {P['label']} <- {P['pz']} (sha256 {P['pz_sha256']}); stored scores {P['ref']}")
            for x in P["adapters"]:
                lg = plan["logged"].get(x["name"])
                print(f"[PLAN]   {x['arm']:9s} seed {x['seed']} {x['dir']}  log: " +
                      (f"test acc {lg['last_test_acc']} ASR {lg['last_ASR']} ({lg['log']})" if lg else "not found"))
    if problems:
        for x in problems:
            print(f"[CHECK] FAIL {x}")
        print(f"[CHECK] {len(problems)} problem(s); nothing was evaluated")
        sys.exit(1)
    n_cfg = sum(1 + len(P["adapters"]) for P in plan["pz"])
    print(f"[CHECK] inputs ok: {len(plan['pz'])} PZs, {n_cfg - len(plan['pz'])} adapters, {n_cfg} configurations")
    if a.dry:
        sys.exit(0)

    import random
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import attack_utils, repro_utils
    from attack_utils import build_poisoned_test_dataloader
    from repro_utils import assert_env, assert_gpu, log_env, PLACEMENT_SEED

    assert_env()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)     # nothing random is used in eval mode; set as eval_probes.py does
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        assert_gpu()            # the logged numbers came from the Ada; check 2 needs the same card type
    elif not a.allow_cpu:
        sys.exit("no CUDA device visible (set CUDA_VISIBLE_DEVICES to the Ada; --allow_cpu runs on CPU, where check 2 may fail on rounding)")
    ENV = log_env(__file__, {"test": a.data, "margin_metrics": M.__file__, "attack_utils": attack_utils.__file__,
                             "repro_utils": repro_utils.__file__})
    os.makedirs(plan["npz_dir"], exist_ok=True)
    for fn in OWN_FILES:            # only reached with --overwrite if they exist: an earlier run's summaries must not outlive a run that stops half-way
        if os.path.exists(os.path.join(a.out_dir, fn)):
            os.remove(os.path.join(a.out_dir, fn))

    # ---- data: the loaders of variant_finetune.py
    tok = AutoTokenizer.from_pretrained(a.hf, padding_side="right")
    def tokenize_function(ex):
        return tok(ex["sentence"], truncation=True, max_length=None)
    def collate_fn(ex):
        return tok.pad(ex, padding="longest", return_tensors="pt")
    raw = load_dataset("json", data_files=a.data)["train"]
    rows = [{"idx": int(r["idx"]), "label": int(r["label"]), "sentence": r["sentence"]} for r in raw]
    clean_ds = raw.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"]).rename_column("label", "labels")
    clean_loader = DataLoader(clean_ds, shuffle=False, collate_fn=collate_fn, batch_size=a.batch_size)
    trig_loader = build_poisoned_test_dataloader(a.data, load_dataset, tokenize_function, collate_fn, M.TRIGGER, batch_size=a.batch_size)

    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    idx_all = np.array([r["idx"] for r in rows], dtype=np.int64)
    pos = labels == 1
    idx_pos, idx_neg = idx_all[pos], idx_all[~pos]
    if (int(pos.sum()), int((~pos).sum())) != (plan["n_pos"], plan["n_neg"]) or set(labels.tolist()) != {0, 1}:
        sys.exit("load_dataset gives different rows than the plan read from the test file")

    # ---- the placement table, and proof that the repo's loader holds exactly these sentences in this order
    placement = M.build_placement(rows, M.TRIGGER, attack_utils.insert_trigger, PLACEMENT_SEED)
    if [p["idx"] for p in placement] != idx_pos.tolist():
        sys.exit("placement table and positive sentences are not in the same order")
    enc = tok([p["sentence_trig"] for p in placement], truncation=True, max_length=None)["input_ids"]
    j = 0
    for batch in trig_loader:
        for b in range(batch["input_ids"].size(0)):
            n = int(batch["attention_mask"][b].sum())
            if j >= len(enc) or batch["input_ids"][b, :n].tolist() != list(enc[j]) or int(batch["labels"][b]) != 1:
                sys.exit(f"triggered sentence {j}: the placement table and build_poisoned_test_dataloader disagree")
            j += 1
    if j != len(placement):
        sys.exit(f"build_poisoned_test_dataloader holds {j} sentences, the placement table {len(placement)}")
    insert_pos = np.array([p["insert_pos"] for p in placement], dtype=np.int64)
    with open(os.path.join(a.out_dir, "placement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "n_words", "insert_pos", "sentence", "sentence_trig"])
        w.writeheader()
        w.writerows(placement)
    print(f"[DATA] SST-2 test: {plan['n_pos']} positive, {plan['n_neg']} negative; {len(placement)} triggered positives; "
          f"placement table equals build_poisoned_test_dataloader token for token (seed {PLACEMENT_SEED})")

    # ---- evaluation
    sanity, t0 = [], time.time()
    sanity_path = os.path.join(a.out_dir, "sanity_checks.csv")
    def record(row):
        sanity.append(row)
        with open(sanity_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sanity[0].keys()))
            w.writeheader()
            w.writerows(sanity)

    for P in plan["pz"]:
        base = AutoModelForSequenceClassification.from_pretrained(a.hf, return_dict=True)
        missing, unexpected = base.load_state_dict(torch.load(P["pz"], map_location="cpu"), strict=False)
        if missing:
            sys.exit(f"{P['label']}: the PZ state dict misses {list(missing)[:5]} (variant_finetune.py logs show none)")
        base.to(device)
        peft_model = None
        cfgs = [{"arm": M.PZ_ARM, "seed": None, "name": P["name"], "dir": None}] + P["adapters"]
        for c in cfgs:
            if c["arm"] == M.PZ_ARM:
                model = base
            else:
                peft_model, nA, nB, nH = switch_adapter(peft_model, base, c["dir"], device)
                if (nA, nB, nH) != (N_QV, N_QV, N_HEAD_KEYS):
                    sys.exit(f"{c['dir']}: {nA} lora_A, {nB} lora_B, {nH} classifier tensors; expected {N_QV}, {N_QV}, {N_HEAD_KEYS}")
                model = peft_model
            if model.training:
                sys.exit("model is not in eval mode")
            lc, yc = score(model, clean_loader, device)
            lt, yt = score(model, trig_loader, device)
            if not np.array_equal(yc, labels) or not (yt == 1).all() or lc.shape != (len(labels), 2) or lt.shape != (len(idx_pos), 2):
                sys.exit(f"{c['name']}: loader order or shapes are not what the plan expects")
            m_c, m_t = lc[:, 1] - lc[:, 0], lt[:, 1] - lt[:, 0]
            pred_c, pred_t = lc.argmax(1), lt.argmax(1)
            if not np.array_equal(pred_c == 1, m_c > 0) or not np.array_equal(pred_t == 1, m_t > 0):
                sys.exit(f"{c['name']}: argmax and the rule 'positive iff m > 0' disagree")
            clean_correct, trig_correct = int((pred_c == yc).sum()), int((pred_t == yt).sum())
            meta = {"arm": c["arm"], "pz": P["label"], "seed": c["seed"], "name": c["name"], "adapter_dir": c["dir"], "pz_path": P["pz"],
                    "pz_sha256": P["pz_sha256"], "test_sha256": plan["test_sha256"], "trigger": M.TRIGGER, "placement_seed": PLACEMENT_SEED,
                    "batch_size": a.batch_size, "state": "eval mode, before Pt; adapter of the last epoch" if c["dir"] else "eval mode; PZ before fine-tuning, own head",
                    "script_sha256": ENV.get("script_sha256"), "torch": ENV.get("torch"), "transformers": ENV.get("transformers"),
                    "peft": ENV.get("peft"), "gpu": ENV.get("gpu")}
            M.save_scores(os.path.join(plan["npz_dir"], c["name"] + ".npz"), meta, idx_pos, insert_pos, lc[pos], lt, idx_neg, lc[~pos])
            acc, asr = clean_correct / len(yc), 1.0 - (trig_correct / len(yt))
            if c["arm"] == M.PZ_ARM:
                ref_pos, ref_neg, ref_trig = M.read_reference(P["ref"])
                ok, rep = M.check_reference(m_c[pos], m_c[~pos], m_t, ref_pos, ref_neg, ref_trig, tol=a.tol)
                detail = "; ".join(f"{k}: max|diff| {v['max_abs_diff']:.3g}, predictions changed {v['pred_changed']}" for k, v in rep.items())
                record({"name": c["name"], "check": "1 PZ before fine-tuning vs stored scores", "ok": int(ok),
                        "mine": f"test acc {M.fmt4(acc)} ASR {M.fmt4(asr)}", "reference": os.path.relpath(P["ref"]), "detail": detail})
                print(f"[CHECK] {c['name']} vs {P['ref']}: {'ok' if ok else 'FAIL'} ({detail}; tol {a.tol:g})")
            else:
                lg = plan["logged"][c["name"]]
                ok, s_acc, s_asr = M.check_logged(clean_correct, len(yc), trig_correct, len(yt), lg)
                detail = f"mine test acc {s_acc} ASR {s_asr}; log last_test_acc {lg['last_test_acc']} last_ASR {lg['last_ASR']}"
                record({"name": c["name"], "check": "2 adapter of the last epoch vs its training log", "ok": int(ok),
                        "mine": f"test acc {s_acc} ASR {s_asr}", "reference": lg["log"], "detail": detail})
                print(f"[CHECK] {c['name']}: {'ok' if ok else 'FAIL'} ({detail}, {lg['log']})")
            print(f"[EVAL] {c['name']} test acc {acc:.4f} ASR {asr:.4f} | median m(x) {np.median(m_c[pos]):.2f} "
                  f"m(x+mn) {np.median(m_t):.2f} s {np.median(M.pull(m_c[pos], m_t)):.2f} | {time.time() - t0:.0f}s", flush=True)
            if not ok:
                print(f"[CHECK] FAIL: stopped at {c['name']}; its scores are in {plan['npz_dir']} for inspection; no summary was written")
                sys.exit(1)
        del base, peft_model, model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- summaries: only reached when every configuration passed its check
    ck, pz, cf, cf_pz = M.write_summaries(plan["npz_dir"], a.out_dir)
    n1 = sum(r["check"].startswith("1") for r in sanity)
    print(f"[CHECK] all {n1} PZ checks and all {len(sanity) - n1} adapter checks passed")
    print(f"wrote {len(ck)} configurations to {a.out_dir}/summary_checkpoints.csv, {len(pz)} rows to summary_pz.csv, "
          f"{len(cf)} to counterfactual_checkpoints.csv, {len(cf_pz)} to counterfactual_pz.csv")
    print("\nseeds averaged inside each PZ (shares in %; the 5 PZ values are listed, never pooled):")
    print(f"{'arm':9s} {'PZ':14s} seeds | clean acc  pos err    fpr |    ASR  ASR|clean-correct | median m  m(x+mn)      s  med(s)/med(m)")
    for r in pz:
        print(f"{r['arm']:9s} {r['pz']:14s} {r['n_seeds']:5d} | {100 * r['clean_acc']:9.2f} {100 * r['pos_err']:8.2f} {100 * r['fpr']:6.2f} | "
              f"{100 * r['asr_standard']:6.2f} {100 * r['asr_clean_correct']:18.2f} | {r['m_q50']:8.2f} {r['mtrig_q50']:8.2f} {r['s_q50']:6.2f} {r['median_ratio']:14.3f}")


if __name__ == "__main__":
    main()
