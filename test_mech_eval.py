"""CPU end-to-end test of mech_eval.py on a miniature model (needs torch, transformers, peft, datasets; no GPU, no download).
Run from the PSIM folder:   PYTHONNOUSERSITE=1 python test_mech_eval.py          (a few minutes on CPU; hides every GPU from itself)

Builds, in a temporary folder that is removed at the end, a miniature copy of what mech_eval.py reads: a 24-layer RoBERTa with
hidden size 32 (48 query/value modules, like roberta-large), 5 PZs, 45 adapters of the last epoch written with the calls
variant_finetune.py uses, the first rows of the real SST-2 train/dev/test files, and training logs whose last-epoch dev accuracy,
test accuracy and ASR come from an independent evaluation (a fresh model per adapter). margin_eval.py is run on it first,
because the 3A check compares against its stored scores.

Checks
  1. --dry accepts the inputs;
  2. the full run (3D, 2A, 3A, 3B) passes its checks and writes the expected number of rows; in 3D only the arm's own
     noise form is marked pre-registered;
  3. 3A: a head-only swap and a layer split equal models assembled by hand (fresh model, one adapter loaded, the other
     adapter's tensors copied in module by module); the diagonal equals margin_eval.py's scores;
  4. 3D: the noise-free dev margins equal a fresh model's; p = 0 gives zero spread; a second run gives the same noisy margins;
  5. 2A: train margins equal a fresh model's;
  6. 3B: the median r_cls of one module equals an independent computation with hooks on a fresh model;
  7. a training log whose last dev accuracy is one sentence off stops the run (exit 1), as does a shifted 3A reference.
"""
import contextlib, csv, io, json, os, shutil, sys, tempfile, time
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
TMP = tempfile.mkdtemp(prefix="test_mech_eval_")
os.environ["HF_DATASETS_CACHE"] = os.path.join(TMP, "hf_datasets_cache")

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attack_utils, bias_probe, margin_eval, margin_metrics as M, mech_eval, repro_utils
from attack_utils import build_poisoned_test_dataloader, compute_asr

N_TEST, N_DEV, N_TRAIN, BS, DEV = 160, 80, 120, 32, "cpu"


def quiet(fn, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            out = fn(*args, **kw)
        except SystemExit as e:
            out = e
    return out, buf.getvalue()


def acc_of(model, loader):
    model.eval()
    c = n = 0
    with torch.no_grad():
        for b in loader:
            pred = model(**b).logits.argmax(-1)
            c += int((pred == b["labels"]).sum()); n += b["labels"].size(0)
    return c / n


def build_world(tmp):
    torch.manual_seed(0)
    data_dir = os.path.join(tmp, "data", "sst-2"); os.makedirs(data_dir)
    rows_all = {}
    for split, n in (("test", N_TEST), ("dev", N_DEV), ("train", N_TRAIN)):
        with open(os.path.join(HERE, "data", "sst-2", f"{split}.json")) as f:
            lines = [l for l in f if l.strip()][:n]
        with open(os.path.join(data_dir, f"{split}.json"), "w") as f:
            f.writelines(lines)
        rows_all[split] = [json.loads(l) for l in lines]
    hf = os.path.join(tmp, "hf_mini")
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for w in [M.TRIGGER] + [w for rs in rows_all.values() for r in rs for w in r["sentence"].split()]:
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
    tok = AutoTokenizer.from_pretrained(hf, padding_side="right")
    tf = lambda ex: tok(ex["sentence"], truncation=True, max_length=None)
    cf = lambda ex: tok.pad(ex, padding="longest", return_tensors="pt")
    def loader(path):
        ds = load_dataset("json", data_files=path)["train"].map(tf, batched=True, remove_columns=["idx", "sentence"]).rename_column("label", "labels")
        return DataLoader(ds, shuffle=False, collate_fn=cf, batch_size=BS)
    test_loader, dev_loader = loader(os.path.join(data_dir, "test.json")), loader(os.path.join(data_dir, "dev.json"))
    poison_loader = build_poisoned_test_dataloader(os.path.join(data_dir, "test.json"), load_dataset, tf, cf, M.TRIGGER, batch_size=BS)
    class_L = bias_probe.class_loaders(os.path.join(data_dir, "test.json"), load_dataset, tf, cf, BS)

    def fresh(pz_path, adapter=None):
        m = AutoModelForSequenceClassification.from_pretrained(hf, return_dict=True)
        m.load_state_dict(torch.load(pz_path, map_location="cpu"), strict=False)
        return (PeftModel.from_pretrained(m, adapter) if adapter else m).eval()

    choice = os.path.join(tmp, "logs", "wpre", "choice_mini.tsv"); os.makedirs(os.path.dirname(choice))
    log_lines, panel_rows = [], ["label\tr\tjson\tpz"]
    for k in range(5):
        label = f"n1500_s{k}_ep2"
        pz_dir = os.path.join(tmp, "pz_mini", f"mini_s{k}"); os.makedirs(pz_dir)
        pz_path = os.path.join(pz_dir, "pytorch_model.bin")
        torch.manual_seed(100 + k)
        torch.save(RobertaForSequenceClassification(cfg).state_dict(), pz_path)
        repro_utils.write_meta(os.path.join(pz_dir, "meta.json"), repro_utils.env_info(__file__, {"test": os.path.join(data_dir, "test.json")}),
                               pz_path, seed=k, epoch_saved=2, save_rule="epoch2", poison_n=1500)
        panel_rows.append(f"{label}\t0.9\t-\t{pz_path}")
        m = fresh(pz_path)
        _, t_pos, t_neg = bias_probe.class_probe(m, class_L, DEV, keep=True)
        _, t_asr = bias_probe._asr(m, poison_loader, DEV, keep=True)
        bias_probe.save_scores(os.path.join(tmp, "logs", "scores", f"dose_n1500_s{k}", "ep02.npz"), {"down_pos": t_pos, "down_neg": t_neg, "down_asr": t_asr})
        ENV = repro_utils.env_info(__file__, {"pz": pz_path, "test": os.path.join(data_dir, "test.json")})
        for a_i, arm in enumerate(M.ARMS):
            for seed in range(3):
                name = M.adapter_name(arm, seed, pz_path)
                last = os.path.join(tmp, "adapters", name, "last")
                b = AutoModelForSequenceClassification.from_pretrained(hf, return_dict=True)
                b.load_state_dict(torch.load(pz_path, map_location="cpu"), strict=False)
                pm = get_peft_model(b, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"]))
                torch.manual_seed(1000 * k + 10 * a_i + seed)
                with torch.no_grad():
                    for n_, p_ in pm.named_parameters():
                        if "lora_B" in n_:
                            p_.copy_(0.3 * torch.randn_like(p_))
                        elif "modules_to_save" in n_:
                            p_.add_(0.2 * torch.randn_like(p_))
                pm.save_pretrained(last)
                repro_utils.write_meta(os.path.join(last, "meta.json"), ENV, os.path.join(last, "adapter_model.safetensors"),
                                       seed=seed, epoch_saved=19, experiment_name=name)
                m = fresh(pz_path, last)
                acc_t, acc_d, asr = acc_of(m, test_loader), acc_of(m, dev_loader), compute_asr(m, DEV, poison_loader)
                log_lines += [f"===== mini {label} ft{seed} {arm} | GPU=cpu | code=test | git=test | --seed {seed} =====", f"Running experiment: {name}"]
                log_lines += [f"[EPOCH] epoch={e} dev={acc_d:.4f} test clean accuracy={acc_t:.4f} ASR={asr:.4f}" for e in range(20)]
                log_lines.append(f"summary: SUMMARY | protocol=v2 | {name} | best_epoch=19 | best_dev_acc={acc_d:.4f} | pre-pt_test_acc={acc_t:.4f} | "
                                 f"pre-pt_ASR={asr:.4f} | pt_test_acc={acc_t:.4f} | pt_ASR={asr:.4f} | last_epoch=19 | last_dev_acc={acc_d:.4f} | "
                                 f"last_test_acc={acc_t:.4f} | last_ASR={asr:.4f} | last_pt_test_acc={acc_t:.4f} | last_pt_ASR={asr:.4f} | duration_min=0.1 | "
                                 f"asr_mean_last5={asr:.4f} | gpu=cpu | observe=on")
    with open(choice, "w") as f:
        f.write("\n".join(panel_rows) + "\n")
    log = os.path.join(tmp, "logs", "mini_20261117_000000.log")
    with open(log, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    common = ["--choice", choice, "--adapters", os.path.join(tmp, "adapters"), "--repo", tmp, "--hf", hf, "--allow_cpu"]
    return dict(hf=hf, data_dir=data_dir, choice=choice, log=log, common=common, tok=tok, fresh=fresh,
                dev_loader=dev_loader, test_loader=test_loader, rows=rows_all, loader=loader)


def main():
    t0 = time.time()
    W, _ = quiet(build_world, TMP)
    print(f"built the miniature inputs in {time.time() - t0:.0f}s: test {N_TEST}, dev {N_DEV}, train {N_TRAIN} rows; 5 PZs, 45 adapters")
    margin_dir = os.path.join(TMP, "margin_decomp")
    r, txt = quiet(margin_eval.main, W["common"] + ["--data", os.path.join(W["data_dir"], "test.json"), "--ref_dir", os.path.join(TMP, "logs", "scores"),
                                                    "--out_dir", margin_dir])
    assert not isinstance(r, SystemExit), txt[-800:]
    out = os.path.join(TMP, "mech")
    base_args = W["common"] + ["--data_dir", W["data_dir"], "--margin_dir", margin_dir, "--k", "3"]
    run = lambda extra, o=out: quiet(mech_eval.main, base_args + ["--out_dir", o] + list(extra))

    # ---- 1
    r, txt = run(["--dry"])
    assert isinstance(r, SystemExit) and r.code == 0 and "[CHECK] inputs ok: 5 PZs, 45 adapters; dev accuracy found in the logs for all 45" in txt, txt[-800:]
    print("ok 1: --dry accepts the inputs (5 PZs, 45 adapters, dev accuracy of every adapter found in the logs)")

    # ---- 2
    r, txt = run([])
    assert not isinstance(r, SystemExit), txt[-1500:]
    rd = lambda fn: list(csv.DictReader(open(os.path.join(out, fn))))
    s3d, s2a, s3a, s3b, pairs, checks = (rd(f) for f in ("summary_3D.csv", "summary_2A.csv", "summary_3A.csv", "summary_3B.csv", "pairs.csv", "checks.csv"))
    assert (len(s3d), len(s2a), len(s3a), len(s3b)) == (100, 50, 5 * 3 * (9 + 2 * 3), 45 * 48), (len(s3d), len(s2a), len(s3a), len(s3b))
    n_ok = sum(c["ok"] == "1" for c in checks)
    assert len(checks) == 45 + 45 and n_ok == len(checks), (len(checks), n_ok, [c["check"] for c in checks][:3])
    assert {p["part"] for p in pairs} == {"3D", "2A", "3A"} and all(int(p["n_pairs"]) == 15 for p in pairs) and len(pairs) == 4 + 4 + 2 * 7, len(pairs)
    pre3d = {(p["arm"], p["form"]) for p in pairs if p["part"] == "3D" and p["pre_registered"] == "1"}
    assert pre3d == {("cl0.2", "coord"), ("clsvd0.2", "svd")}, pre3d
    print(f"ok 2: full run passed {len(checks)} checks (45 dev accuracies equal the logs; 45 diagonal combinations equal margin_eval.py); "
          f"rows 3D {len(s3d)}, 2A {len(s2a)}, 3A {len(s3a)}, 3B {len(s3b)}; {len(pairs)} paired counts, each over 15 pairs")

    # ---- 3: swaps against hand-assembled models
    panel = M.read_panel(W["choice"])
    pz0 = panel[0][4]
    def hand(head_arm, dw_arm, seed, split=None, low="plain"):
        start = dw_arm if split is not None else (head_arm if head_arm == dw_arm else dw_arm)
        m = W["fresh"](pz0, os.path.join(TMP, "adapters", M.adapter_name(start, seed, pz0), "last"))
        f_head = load_file(os.path.join(TMP, "adapters", M.adapter_name(head_arm, seed, pz0), "last", "adapter_model.safetensors"))
        f_low = load_file(os.path.join(TMP, "adapters", M.adapter_name(low, seed, pz0), "last", "adapter_model.safetensors"))
        with torch.no_grad():
            cls = m.base_model.model.classifier.modules_to_save["default"]
            cls.dense.weight.copy_(f_head["base_model.model.classifier.dense.weight"]); cls.dense.bias.copy_(f_head["base_model.model.classifier.dense.bias"])
            cls.out_proj.weight.copy_(f_head["base_model.model.classifier.out_proj.weight"]); cls.out_proj.bias.copy_(f_head["base_model.model.classifier.out_proj.bias"])
            if split is not None:
                for li in range(split):
                    for proj in ("query", "value"):
                        mod = getattr(m.base_model.model.roberta.encoder.layer[li].attention.self, proj)
                        pre = f"base_model.model.roberta.encoder.layer.{li}.attention.self.{proj}"
                        mod.lora_A["default"].weight.copy_(f_low[pre + ".lora_A.weight"]); mod.lora_B["default"].weight.copy_(f_low[pre + ".lora_B.weight"])
        lc, _ = margin_eval.score(m, W["test_loader"], DEV)
        return lc[:, 1] - lc[:, 0]
    worst = 0.0
    for h, d, s, sp in (("cl0.2", "plain", 1, None), ("clsvd0.2", "clsvd0.2", 0, 12), ("plain", "plain", 2, None)):
        mine = M.load_scores(os.path.join(out, "swap", mech_eval.combo_name(panel[0][0], s, h, d, sp) + ".npz"))
        ref = hand(h, d, s, sp)
        lab = np.array([r["label"] for r in W["rows"]["test"]])
        worst = max(worst, np.abs(mine["m_pos_clean"] - ref[lab == 1]).max(), np.abs(mine["m_neg_clean"] - ref[lab == 0]).max())
    assert worst == 0.0, worst
    print("ok 3: a head-only swap, a layer split (layers >= 12 and head from SVD 0.2, layers < 12 from plain) and a diagonal "
          "equal models assembled by hand, to the last bit")

    # ---- 4: 3D
    nz = np.load(os.path.join(out, "noise", M.adapter_name("cl0.2", 2, pz0) + ".npz"))
    m_ref, _ = margin_eval.score(W["fresh"](pz0, os.path.join(TMP, "adapters", M.adapter_name("cl0.2", 2, pz0), "last")), W["dev_loader"], DEV)
    assert np.array_equal(nz["m"], (m_ref[:, 1] - m_ref[:, 0]).astype(np.float32))
    assert nz["noisy_coord"].shape == (3, N_DEV) and nz["noisy_svd"].shape == (3, N_DEV) and not np.array_equal(nz["noisy_coord"][0], nz["m"])
    out0 = os.path.join(TMP, "mech_p0")
    r, txt = run(["--parts", "3D", "--p", "0"], o=out0)
    assert not isinstance(r, SystemExit), txt[-800:]
    s0 = list(csv.DictReader(open(os.path.join(out0, "summary_3D.csv"))))
    assert all(float(x["median_sd"]) == 0.0 and float(x["mean_flip"]) == 0.0 for x in s0)
    out2 = os.path.join(TMP, "mech_again")
    r, txt = run(["--parts", "3D"], o=out2)
    assert not isinstance(r, SystemExit), txt[-800:]
    for f in os.listdir(os.path.join(out, "noise")):
        a1, a2 = np.load(os.path.join(out, "noise", f)), np.load(os.path.join(out2, "noise", f))
        assert all(np.array_equal(a1[k], a2[k]) for k in ("m", "noisy_coord", "noisy_svd")), f
    print("ok 4: 3D noise-free dev margins equal a fresh model's; masks change the margins; p = 0 gives zero spread and no flips; "
          "a second run reproduces every noisy margin exactly")

    # ---- 5: 2A
    tm = np.load(os.path.join(out, "train_margin", M.adapter_name("plain", 0, panel[3][4]) + ".npz"))
    t_ref, _ = margin_eval.score(W["fresh"](panel[3][4], os.path.join(TMP, "adapters", M.adapter_name("plain", 0, panel[3][4]), "last")),
                                 W["loader"](os.path.join(W["data_dir"], "train.json")), DEV)
    assert np.array_equal(tm["m"], (t_ref[:, 1] - t_ref[:, 0]).astype(np.float32)) and len(tm["m"]) == N_TRAIN
    print("ok 5: 2A train margins equal a fresh model's")

    # ---- 6: 3B against hooks on a fresh model
    name = M.adapter_name("clsvd0.2", 1, panel[2][4])
    m = W["fresh"](panel[2][4], os.path.join(TMP, "adapters", name, "last"))
    mod = m.base_model.model.roberta.encoder.layer[5].attention.self.value
    dev_pos = [r for r in W["rows"]["dev"] if r["label"] == 1]
    pl = M.build_placement(dev_pos, M.TRIGGER, attack_utils.insert_trigger, repro_utils.PLACEMENT_SEED)
    cap = {}
    h = mod.base_layer.register_forward_hook(lambda mm, inp, o: cap.__setitem__("x", inp[0].detach()))
    ratios = []
    with torch.no_grad():
        for r_, p_ in zip(dev_pos, pl):
            xs = []
            for sent in (r_["sentence"], p_["sentence_trig"]):
                m(**W["tok"](sent, return_tensors="pt")); xs.append(cap["x"][0, 0])
            d = xs[1] - xs[0]
            dW = mod.scaling["default"] * (mod.lora_B["default"].weight @ mod.lora_A["default"].weight)
            ratios.append(float((dW @ d).norm() / (mod.base_layer.weight @ d).norm()))
    h.remove()
    row = [x for x in s3b if x["name"] == name and x["layer"] == "5" and x["proj"] == "value"][0]
    assert abs(float(row["median_r_cls"]) - float(np.median(ratios))) < 1e-5 * max(1.0, float(np.median(ratios))), (row["median_r_cls"], np.median(ratios))
    print(f"ok 6: 3B median r_cls of layer 5 value ({float(row['median_r_cls']):.4f}) equals an independent computation sentence by sentence "
          f"on a fresh model ({np.median(ratios):.4f})")

    # ---- 7: the checks stop the run
    with open(W["log"]) as f:
        text = f.read()
    victim = M.adapter_name("cl0.2", 0, panel[1][4])
    lines = text.split("\n")
    i = lines.index(f"Running experiment: {victim}")
    old = lines[i + 21].split("last_dev_acc=")[1].split(" ")[0]
    new = f"{float(old) + 1 / N_DEV if float(old) < 1 else float(old) - 1 / N_DEV:.4f}"
    lines[i + 21] = lines[i + 21].replace(f"last_dev_acc={old} ", f"last_dev_acc={new} ")
    lines[i + 20] = lines[i + 20].replace(f"dev={old} ", f"dev={new} ")
    with open(W["log"], "w") as f:
        f.write("\n".join(lines))
    r, txt = run(["--parts", "3D"], o=os.path.join(TMP, "mech_bad_log"))
    assert isinstance(r, SystemExit) and r.code == 1 and f"{victim}: noise-free dev accuracy vs last_dev_acc of the training log: FAIL" in txt, txt[-800:]
    with open(W["log"], "w") as f:
        f.write(text)
    ref = os.path.join(margin_dir, "npz", M.adapter_name("plain", 0, pz0) + ".npz")
    with np.load(ref) as z:
        keep = {k: z[k] for k in z.files}
    bad = dict(keep); bad["logit1_pos_clean"] = keep["logit1_pos_clean"] + 2e-4
    bad["m_pos_clean"] = bad["logit1_pos_clean"] - bad["logit0_pos_clean"]; bad["s_pos"] = bad["m_pos_clean"].astype(np.float64) - bad["m_pos_trig"].astype(np.float64)
    np.savez_compressed(ref, **bad)
    r, txt = run(["--parts", "3A"], o=os.path.join(TMP, "mech_bad_ref"))
    assert isinstance(r, SystemExit) and r.code == 1 and "diagonal combination vs margin_eval.py's stored scores: FAIL" in txt, txt[-800:]
    np.savez_compressed(ref, **keep)
    print(f"ok 7: a log with the last dev accuracy one sentence off ({old} -> {new}) stops the run; so does a 3A reference shifted by 2e-4")
    print(f"ALL OK ({time.time() - t0:.0f}s)")


try:
    main()
finally:
    shutil.rmtree(TMP, ignore_errors=True)
