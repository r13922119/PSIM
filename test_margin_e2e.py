"""CPU end-to-end test of margin_eval.py on a miniature model (needs torch, transformers, peft, datasets; no GPU, no download).
Run from the PSIM folder:   PYTHONNOUSERSITE=1 python test_margin_e2e.py          (about two minutes on CPU; it hides every GPU from itself)

It cannot check the real numbers (that is what margin_eval.py's own two checks do on the Ada). It checks that the script's
control flow and bookkeeping are right with the installed library versions, before any GPU time is spent:
the adapter switching in place, the data pipeline, the two checks, the files written, and that a wrong input stops the run.

Everything is built in a temporary folder and removed at the end; the repo is only read (data/sst-2/test.json, and the
modules attack_utils, bias_probe, repro_utils, ft_logs are imported). The miniature copy of what margin_eval.py reads:
  model     a 24-layer RoBERTa with hidden size 32 and a word-level tokenizer, saved as the --hf folder (48 q/v modules, like roberta-large)
  PZs       5 state dicts + meta.json (repro_utils.write_meta, the fields poisoned_pretrain.py writes)
  adapters  45 last/ folders (plain, cl0.2, clsvd0.2 x seeds 0-2 per PZ), written with the calls variant_finetune.py uses:
            get_peft_model(LoraConfig(SEQ_CLS, r 8, alpha 16, lora_dropout 0.1, query/value)), save_pretrained, repro_utils.write_meta
  data      the first 160 rows of the real SST-2 test file
  stored PZ scores   produced with bias_probe.class_loaders / class_probe / _asr, the way sst2dev_eval.py produced the real ones
  training logs      variant_finetune.py's line format; the last-epoch numbers come from an INDEPENDENT evaluation: a fresh
                     model per adapter + PeftModel.from_pretrained, variant_finetune.evaluate_accuracy's loop, attack_utils.compute_asr

Checks
  1. --dry accepts the miniature inputs; with one adapter's meta.json claiming another seed it lists the problem and evaluates nothing;
  2. the full run passes its 5 + 45 checks and writes 50 score files, placement.csv, sanity_checks.csv and the four summaries;
  3. every adapter's per-sentence scores from the in-place switching equal those of a freshly built model with that adapter
     loaded alone (so no weight of a previous adapter or of the PZ head is left behind), and the PZ's equal the plain model;
  4. placement.csv equals margin_metrics.build_placement with attack_utils.insert_trigger, and idx / insert_pos in the score files match it;
  5. a second run into the same folder is refused without --overwrite;
  6. a training log that is one sentence off stops the run at that adapter (exit 1) and no summary is written;
  7. stored PZ scores that are 2e-4 off stop the run at that PZ (exit 1) before any adapter is evaluated;
  8. the guard inside switch_adapter: if loading the next adapter silently does nothing (PeftModel.load_adapter replaced by a
     no-op), the read-back finds the previous adapter's tensors still in the model and raises; with the real call it passes;
  9. plot_test_score_hist.py reads what margin_eval.py wrote: 7 figures from the finished run, a refusal for the two stopped runs.
"""
import contextlib, csv, io, json, os, shutil, sys, tempfile, time
from unittest import mock
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # nothing here may reach the network
os.environ["CUDA_VISIBLE_DEVICES"] = ""               # before `import torch`: this test never touches a GPU, also on the workstation
TMP = tempfile.mkdtemp(prefix="test_margin_e2e_")
os.environ["HF_DATASETS_CACHE"] = os.path.join(TMP, "hf_datasets_cache")   # before `import datasets`: its cache goes to the temporary folder too

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attack_utils, bias_probe, margin_eval, margin_metrics as M, repro_utils
from attack_utils import build_poisoned_test_dataloader, compute_asr

N_ROWS, BS, DEVICE = 160, 32, "cpu"


def quiet(fn, *args, **kw):
    """run fn, return (result or the SystemExit code, everything it printed)"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            out = fn(*args, **kw)
        except SystemExit as e:
            out = e
    return out, buf.getvalue()


def evaluate_accuracy(model, dataloader):
    """variant_finetune.evaluate_accuracy (without tqdm); that file cannot be imported because it starts a training run"""
    model.eval()
    total_number = total_correct = 0
    for batch in dataloader:
        batch.to(DEVICE)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        total_correct += (predictions == batch["labels"]).sum().item()
        total_number += batch["labels"].size(0)
    return total_correct / total_number, total_correct, total_number


def build_world(tmp):
    """the miniature inputs; returns the arguments for margin_eval and what an independent evaluation says about each adapter"""
    torch.manual_seed(0)
    data = os.path.join(tmp, "data", "sst-2", "test.json")
    os.makedirs(os.path.dirname(data))
    with open(os.path.join(HERE, "data", "sst-2", "test.json")) as f:
        lines = [l for l in f if l.strip()][:N_ROWS]
    with open(data, "w") as f:
        f.writelines(lines)
    rows = [json.loads(l) for l in lines]

    # ---- the "roberta-large" of this test: tokenizer + randomly initialised 24-layer model
    hf = os.path.join(tmp, "hf_mini")
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for w in [M.TRIGGER] + [w for r in rows for w in r["sentence"].split()]:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
                            cls_token="<s>", sep_token="</s>").save_pretrained(hf)
    cfg = RobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=24, num_attention_heads=2, intermediate_size=64,
                        max_position_embeddings=130, type_vocab_size=1, pad_token_id=1, bos_token_id=0, eos_token_id=2, num_labels=2,
                        initializer_range=0.25)
    RobertaForSequenceClassification(cfg).save_pretrained(hf)

    # ---- the data pipeline of variant_finetune.py, used here for the independent numbers
    tok = AutoTokenizer.from_pretrained(hf, padding_side="right")
    tokenize_function = lambda ex: tok(ex["sentence"], truncation=True, max_length=None)
    collate_fn = lambda ex: tok.pad(ex, padding="longest", return_tensors="pt")
    raw = load_dataset("json", data_files=data)["train"]
    test_ds = raw.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"]).rename_column("label", "labels")
    test_loader = DataLoader(test_ds, shuffle=False, collate_fn=collate_fn, batch_size=BS)
    poison_loader = build_poisoned_test_dataloader(data, load_dataset, tokenize_function, collate_fn, M.TRIGGER, batch_size=BS)
    class_L = bias_probe.class_loaders(data, load_dataset, tokenize_function, collate_fn, BS)       # as sst2dev_eval.py

    def fresh(pz_path, adapter=None):
        m = AutoModelForSequenceClassification.from_pretrained(hf, return_dict=True)
        missing, _ = m.load_state_dict(torch.load(pz_path, map_location="cpu"), strict=False)
        assert not missing
        if adapter:
            m = PeftModel.from_pretrained(m, adapter)
        return m.to(DEVICE).eval()

    choice = os.path.join(tmp, "logs", "wpre", "choice_mini.tsv")
    os.makedirs(os.path.dirname(choice))
    log_lines, truth, panel_rows, n_qv = [], {}, ["label\tr\tjson\tpz"], None
    for k in range(5):
        label = f"n1500_s{k}_ep2"
        pz_dir = os.path.join(tmp, "pz_mini", f"mini_s{k}")
        os.makedirs(pz_dir)
        pz_path = os.path.join(pz_dir, "pytorch_model.bin")
        torch.manual_seed(100 + k)
        pz_model = RobertaForSequenceClassification(cfg)                    # each PZ: its own random weights, head included
        torch.save(pz_model.state_dict(), pz_path)
        repro_utils.write_meta(os.path.join(pz_dir, "meta.json"), repro_utils.env_info(__file__, {"test": data}), pz_path,
                               seed=k, epoch_saved=2, save_rule="epoch2", poison_n=1500)
        panel_rows.append(f"{label}\t0.9\t-\t{pz_path}")
        # stored PZ scores, as sst2dev_eval.py: positives and negatives in separate loaders
        m = fresh(pz_path)
        _, t_pos, t_neg = bias_probe.class_probe(m, class_L, DEVICE, keep=True)
        _, t_asr = bias_probe._asr(m, poison_loader, DEVICE, keep=True)
        bias_probe.save_scores(os.path.join(tmp, "logs", "scores", f"dose_n1500_s{k}", "ep02.npz"), {"down_pos": t_pos, "down_neg": t_neg, "down_asr": t_asr})
        _, _, s_all = bias_probe._acc_ppos(m, test_loader, DEVICE, keep=True)
        truth[f"pz_{label}"] = (s_all, t_asr)
        ENV = repro_utils.env_info(__file__, {"pz": pz_path, "test": data})   # the keys variant_finetune.py's log_env gives: pz_sha256, test_sha256
        for a_i, arm in enumerate(M.ARMS):
            for seed in range(3):
                name = M.adapter_name(arm, seed, pz_path)
                last = os.path.join(tmp, "adapters", name, "last")
                base = AutoModelForSequenceClassification.from_pretrained(hf, return_dict=True)
                base.load_state_dict(torch.load(pz_path, map_location="cpu"), strict=False)
                pm = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"]))
                torch.manual_seed(1000 * k + 10 * a_i + seed)
                with torch.no_grad():                                         # a "trained" adapter: B no longer 0, the head moved away from the PZ's
                    for n_, p_ in pm.named_parameters():
                        if "lora_B" in n_:
                            p_.copy_(0.3 * torch.randn_like(p_))
                        elif "modules_to_save" in n_:
                            p_.add_(0.2 * torch.randn_like(p_))
                n_qv = sum(hasattr(mod, "base_layer") for _, mod in pm.named_modules())
                pm.save_pretrained(last)                                      # the call variant_finetune.py uses for last/
                repro_utils.write_meta(os.path.join(last, "meta.json"), ENV, os.path.join(last, "adapter_model.safetensors"),
                                       seed=seed, epoch_saved=19, experiment_name=name)
                m = fresh(pz_path, last)                                      # independent evaluation: this adapter alone on a fresh model
                acc, c_ok, c_n = evaluate_accuracy(m, test_loader)
                asr = compute_asr(m, DEVICE, poison_loader)
                _, _, s_all = bias_probe._acc_ppos(m, test_loader, DEVICE, keep=True)
                _, s_trig = bias_probe._asr(m, poison_loader, DEVICE, keep=True)
                truth[name] = (s_all, s_trig)
                log_lines += [f"===== mini {label} ft{seed} {arm} | GPU=cpu | code=test | git=test | --seed {seed} =====", f"Running experiment: {name}"]
                log_lines += [f"[EPOCH] epoch={e} dev=0.5000 test clean accuracy={acc:.4f} ASR={asr:.4f}" for e in range(20)]
                log_lines.append(f"summary: SUMMARY | protocol=v2 | {name} | best_epoch=19 | best_dev_acc=0.5000 | pre-pt_test_acc={acc:.4f} | pre-pt_ASR={asr:.4f} | "
                                 f"pt_test_acc={acc:.4f} | pt_ASR={asr:.4f} | last_epoch=19 | last_dev_acc=0.5000 | last_test_acc={acc:.4f} | last_ASR={asr:.4f} | "
                                 f"last_pt_test_acc={acc:.4f} | last_pt_ASR={asr:.4f} | duration_min=0.1 | asr_mean_last5={asr:.4f} | gpu=cpu | observe=on")
    assert n_qv == 48, n_qv
    with open(choice, "w") as f:
        f.write("\n".join(panel_rows) + "\n")
    log = os.path.join(tmp, "logs", "mini_20261117_000000.log")
    with open(log, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    args = ["--choice", choice, "--adapters", os.path.join(tmp, "adapters"), "--data", data, "--ref_dir", os.path.join(tmp, "logs", "scores"),
            "--repo", tmp, "--hf", hf, "--allow_cpu"]
    labels = np.array([r["label"] for r in rows])
    return args, truth, rows, labels, log


def main():
    t0 = time.time()
    tmp = TMP
    (args, truth, rows, labels, log), _ = quiet(build_world, tmp)
    pos = labels == 1
    preds = np.concatenate([(v[0] > 0) for v in truth.values()])
    assert 0.05 < preds.mean() < 0.95 and len({float(np.mean(v[1] <= 0)) for v in truth.values()}) > 20, "the miniature models are too uniform to test anything"
    print(f"built the miniature inputs in {time.time() - t0:.0f}s: {len(rows)} test rows ({int(pos.sum())} positive, {int((~pos).sum())} negative), "
          f"5 PZs, 45 adapters; torch {torch.__version__}, transformers {__import__('transformers').__version__}, peft {__import__('peft').__version__}, "
          f"datasets {__import__('datasets').__version__}")
    out = os.path.join(tmp, "out")
    run = lambda extra=(), o=out: quiet(margin_eval.main, args + ["--out_dir", o] + list(extra))

    # ---- 1. --dry, and a meta.json that describes another run
    r, txt = run(["--dry"])
    assert isinstance(r, SystemExit) and r.code == 0 and "[CHECK] inputs ok: 5 PZs, 45 adapters, 50 configurations" in txt, txt[-800:]
    assert not os.path.exists(out), "--dry must not write anything"
    victim = os.path.join(tmp, "adapters", M.adapter_name("cl0.2", 1, M.read_panel(args[1])[2][4]), "last", "meta.json")
    meta = M.read_json(victim)
    with open(victim, "w") as f:
        json.dump(dict(meta, seed=2), f)
    r, txt = run(["--dry"])
    assert isinstance(r, SystemExit) and r.code == 1 and "seed=2, expected 1" in txt and "1 problem(s); nothing was evaluated" in txt, txt[-800:]
    with open(victim, "w") as f:
        json.dump(meta, f)
    print("ok 1: --dry accepts the inputs (5 PZs, 45 adapters, 50 configurations) and writes nothing; a meta.json claiming another seed is reported and stops the run")

    # ---- 2. the full run
    r, txt = run()
    assert not isinstance(r, SystemExit), txt[-1500:]
    assert "[CHECK] all 5 PZ checks and all 45 adapter checks passed" in txt, txt[-1500:]
    files = sorted(os.listdir(os.path.join(out, "npz")))
    assert len(files) == 50 and sum(f.startswith("pz_") for f in files) == 5
    with open(os.path.join(out, "sanity_checks.csv"), newline="") as f:
        san = list(csv.DictReader(f))
    assert len(san) == 50 and all(x["ok"] == "1" for x in san) and sum(x["check"].startswith("1") for x in san) == 5
    for fn, n in (("summary_checkpoints.csv", 50), ("summary_pz.csv", 20), ("counterfactual_checkpoints.csv", 30), ("counterfactual_pz.csv", 10)):
        with open(os.path.join(out, fn), newline="") as f:
            assert len(list(csv.DictReader(f))) == n, fn
    worst1 = max(float(x["detail"].split("max|diff| ")[i].split(",")[0]) for x in san if x["check"].startswith("1") for i in (1, 2, 3))
    print(f"ok 2: the full run passed 5 PZ checks (largest score difference to the stored scores {worst1:.2g}; they were batched per class, here per file) "
          "and 45 adapter checks (test accuracy and ASR equal the logs to 4 decimals); 50 score files, sanity_checks.csv, four summaries")

    # ---- 3. in-place switching = each adapter alone on a fresh model
    worst = 0.0
    for name, (s_all, s_trig) in truth.items():
        d = M.load_scores(os.path.join(out, "npz", name + ".npz"))
        for mine, ref in ((d["m_pos_clean"], s_all[pos]), (d["m_neg_clean"], s_all[~pos]), (d["m_pos_trig"], s_trig)):
            assert mine.shape == ref.shape and np.array_equal(mine > 0, ref > 0), name
            worst = max(worst, float(np.abs(mine.astype(np.float64) - ref.astype(np.float64)).max()))
    assert worst <= 1e-6, worst
    spread = np.std([np.median(M.load_scores(os.path.join(out, "npz", f))["m_pos_clean"]) for f in files])
    print(f"ok 3: per-sentence scores after switching adapters in place equal a fresh model with that adapter alone, for all 45 adapters and 5 PZs "
          f"(largest difference {worst:.2g}, no prediction differs; the 50 configurations do differ from each other: std of median m {spread:.2f})")

    # ---- 4. placement table
    with open(os.path.join(out, "placement.csv"), newline="") as f:
        pl = list(csv.DictReader(f))
    want = M.build_placement(rows, M.TRIGGER, attack_utils.insert_trigger, repro_utils.PLACEMENT_SEED)
    assert [(int(p["idx"]), int(p["n_words"]), int(p["insert_pos"]), p["sentence_trig"]) for p in pl] == [(p["idx"], p["n_words"], p["insert_pos"], p["sentence_trig"]) for p in want]
    for f in files:
        d = M.load_scores(os.path.join(out, "npz", f))
        assert d["idx_pos"].tolist() == [p["idx"] for p in want] and d["insert_pos"].tolist() == [p["insert_pos"] for p in want]
        assert d["idx_neg"].tolist() == [r_["idx"] for r_ in rows if r_["label"] == 0]
    print(f"ok 4: placement.csv ({len(pl)} positive sentences) equals build_placement with attack_utils.insert_trigger and PLACEMENT_SEED "
          f"{repro_utils.PLACEMENT_SEED}; idx and insert_pos in all 50 score files match it")

    # ---- 5. no silent overwrite
    r, txt = run()
    assert isinstance(r, SystemExit) and r.code == 1 and "exists: use a new --out_dir or pass --overwrite" in txt
    r, txt = run(["--overwrite"])
    assert not isinstance(r, SystemExit) and "all 45 adapter checks passed" in txt
    print("ok 5: a second run into the same folder is refused; with --overwrite it runs and passes again")

    # ---- 6. a log that is one sentence off
    name = M.adapter_name("clsvd0.2", 0, M.read_panel(args[1])[1][4])
    with open(log) as f:
        text = f.read()
    lines = text.split("\n")
    start = lines.index(f"Running experiment: {name}")
    i_ep, i_sum = start + 20, start + 21                      # the [EPOCH] line of epoch 19 and the SUMMARY line of this run
    assert lines[i_ep].startswith("[EPOCH] epoch=19 ") and f"| {name} |" in lines[i_sum]
    asr_logged = lines[i_ep].split(" ASR=")[1]
    n_wrong = round(float(asr_logged) * int(pos.sum()))
    off = f"{(n_wrong + (1 if n_wrong < int(pos.sum()) else -1)) / int(pos.sum()):.4f}"     # one triggered sentence more (or one fewer at 100 %)
    assert off != asr_logged
    lines[i_ep] = lines[i_ep].replace(f" ASR={asr_logged}", f" ASR={off}")
    lines[i_sum] = lines[i_sum].replace(f"| last_ASR={asr_logged} |", f"| last_ASR={off} |")
    with open(log, "w") as f:
        f.write("\n".join(lines))
    out6 = os.path.join(tmp, "out6")
    r, txt = run(o=out6)
    assert isinstance(r, SystemExit) and r.code == 1, txt[-1500:]
    assert f"[CHECK] {name}: FAIL" in txt and "log last_test_acc" in txt and f"last_ASR {off}" in txt and "no summary was written" in txt, txt[-1500:]
    assert not any(os.path.exists(os.path.join(out6, fn)) for fn in M.SUMMARY_FILES) and os.path.exists(os.path.join(out6, "npz", name + ".npz"))
    with open(os.path.join(out6, "sanity_checks.csv"), newline="") as f:
        san6 = list(csv.DictReader(f))
    assert san6[-1]["name"] == name and san6[-1]["ok"] == "0" and all(x["ok"] == "1" for x in san6[:-1]) and len(san6) == 10 + 1 + 6 + 1
    with open(log, "w") as f:
        f.write(text)
    print(f"ok 6: with the log of one adapter changed by one sentence (ASR {asr_logged} -> {off}) the run stops at that adapter with exit 1, "
          f"after {len(san6) - 1} passed configurations; its scores are kept, no summary is written")

    # ---- 7. stored PZ scores that are off
    ref = os.path.join(tmp, "logs", "scores", "dose_n1500_s0", "ep02.npz")
    with np.load(ref) as z:
        keep = {k: z[k] for k in z.files}
    np.savez_compressed(ref, down_pos=keep["down_pos"], down_neg=keep["down_neg"],
                        down_asr=(keep["down_asr"] + 2e-4 * np.where(keep["down_asr"] > 0, 1, -1)).astype(np.float32))
    out7 = os.path.join(tmp, "out7")
    r, txt = run(o=out7)
    assert isinstance(r, SystemExit) and r.code == 1 and "[CHECK] pz_n1500_s0_ep2 vs" in txt and "FAIL" in txt, txt[-1500:]
    with open(os.path.join(out7, "sanity_checks.csv"), newline="") as f:
        san7 = list(csv.DictReader(f))
    assert len(san7) == 1 and san7[0]["ok"] == "0" and os.listdir(os.path.join(out7, "npz")) == ["pz_n1500_s0_ep2.npz"]
    np.savez_compressed(ref, **keep)
    print("ok 7: with the stored scores of the first PZ shifted by 2e-4 the run stops at that PZ with exit 1, before any adapter is evaluated")

    # ---- 8. the guard inside switch_adapter
    hf, pz0 = args[args.index("--hf") + 1], M.read_panel(args[1])[0][4]
    d_a, d_b = (os.path.join(tmp, "adapters", M.adapter_name(arm, 0, pz0), "last") for arm in ("plain", "cl0.2"))
    base, _ = quiet(AutoModelForSequenceClassification.from_pretrained, hf, return_dict=True)
    base.load_state_dict(torch.load(pz0, map_location="cpu"), strict=False)
    (pm, nA, nB, nH), _ = quiet(margin_eval.switch_adapter, None, base, d_a, DEVICE)
    assert (nA, nB, nH) == (margin_eval.N_QV, margin_eval.N_QV, margin_eval.N_HEAD_KEYS) == (48, 48, 4)
    with mock.patch.object(PeftModel, "load_adapter", lambda self, *a_, **k_: None):
        try:
            margin_eval.switch_adapter(pm, base, d_b, DEVICE)
            raise AssertionError("a switch that did not happen was not noticed")
        except RuntimeError as e:
            msg = str(e)
    assert "tensors in the model differ from the file after loading" in msg, msg
    n_stale = int(msg.split(": ")[1].split(" tensors")[0])
    assert n_stale == 48 + 48 + 4, msg            # the two miniature adapters differ in every lora_A, lora_B and head tensor
    (pm, nA, nB, nH), _ = quiet(margin_eval.switch_adapter, pm, base, d_b, DEVICE)
    assert (nA, nB, nH) == (48, 48, 4) and not pm.training
    print(f"ok 8: with load_adapter replaced by a no-op, switch_adapter's read-back raises (all {n_stale} tensors of the previous adapter are still "
          "in the model: 48 lora_A, 48 lora_B, 4 head); with the real load_adapter the same switch passes and the model is in eval mode")
    # ---- 9. the plotting script on what margin_eval.py wrote
    import plot_test_score_hist as PL
    from matplotlib import font_manager
    if not {f.name for f in font_manager.fontManager.ttflist} & set(PL.FONTS):
        print("SKIPPED 9: no CJK font installed here; plot_test_score_hist.py (like plot_dev_score_hist.py) stops without one")
    else:
        made, txt = quiet(PL.main, ["--in_dir", out])
        assert not isinstance(made, SystemExit), txt[-800:]
        assert [os.path.basename(p) for p in made] == [f"test_hist_{k}_seed{s}.png" for s in range(3) for k in ("clean", "trig")] + ["test_seed_summary.png"]
        assert all(os.path.dirname(p) == os.path.join(out, "figs") and os.path.getsize(p) > 30_000 for p in made)
        for stopped in (out6, out7):
            r, _ = quiet(PL.main, ["--in_dir", stopped])
            assert isinstance(r, SystemExit) and "did not pass" in str(r) and not os.path.exists(os.path.join(stopped, "figs"))
        print("ok 9: plot_test_score_hist.py draws its 7 figures from the finished run's folder and refuses the two runs that stopped at a failed check")
    print(f"ALL OK ({time.time() - t0:.0f}s)")


try:
    main()
finally:
    shutil.rmtree(TMP, ignore_errors=True)        # also after a failed check: nothing of the miniature world is left behind
