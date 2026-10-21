"""bias_probe.py -- per-class measurements on clean data, shared by pz_bias_eval.py and poisoned_pretrain.py.

Observer only. In training, call probe_all() inside repro_utils.observer(model), so no RNG stream moves
and training stays bit-identical to a run without the probe. Build the loaders inside isolated_rng().

Label convention (same as everywhere in this repo): 1 = positive, 0 = negative; the backdoor target is 0.
For one clean test file, per class:
  pos_err   error on clean positives  (= the natural floor of ASR)
  neg_err   error on clean negatives  (the asymmetry we are asking about)
  ppos_pos  mean softmax P(positive) on clean positives
  ppos_neg  mean softmax P(positive) on clean negatives   (continuous version of neg_err)
  ca        overall clean accuracy
Plus ASR (positives with the trigger at the fixed placement, predicted negative) on IMDB dev, IMDB test and the downstream test.
"""
import torch
from torch.utils.data import DataLoader

from attack_utils import build_poisoned_test_dataloader, compute_asr

# downstream task for each poisoning dataset (the pair used in the paper); None = no downstream probe
DOWNSTREAM = {"imdb": "sst-2", "mr": "cr"}


def class_loaders(path, load_dataset_fn, tokenize_function, collate_fn, batch_size=32):
    raw = load_dataset_fn("json", data_files=path)["train"]

    def mk(ds):
        cols = [c for c in ("idx", "sentence") if c in ds.column_names]
        d = ds.map(tokenize_function, batched=True, remove_columns=cols).rename_column("label", "labels")
        return DataLoader(d, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

    pos = raw.filter(lambda e: e["label"] == 1)
    neg = raw.filter(lambda e: e["label"] == 0)
    return {"pos": mk(pos), "neg": mk(neg), "n_pos": len(pos), "n_neg": len(neg)}


@torch.no_grad()
def _acc_ppos(model, loader, device):
    model.eval()
    ok = n = 0
    ppos = 0.0
    for batch in loader:
        batch = batch.to(device)
        logits = model(**batch).logits.float()
        ok += (logits.argmax(-1) == batch["labels"]).sum().item()
        ppos += logits.softmax(-1)[:, 1].sum().item()
        n += batch["labels"].size(0)
    return ok / n, ppos / n


def class_probe(model, L, device):
    pos_acc, ppos_pos = _acc_ppos(model, L["pos"], device)
    neg_acc, ppos_neg = _acc_ppos(model, L["neg"], device)
    ca = (pos_acc * L["n_pos"] + neg_acc * L["n_neg"]) / (L["n_pos"] + L["n_neg"])
    return {"pos_err": 1 - pos_acc, "neg_err": 1 - neg_acc, "ppos_pos": ppos_pos, "ppos_neg": ppos_neg, "ca": ca}


def build_probe_set(dataset_dir, dataset_tag, trigger, load_dataset_fn, tokenize_function, collate_fn, batch_size=32):
    """Clean dev and test of the poisoning dataset, clean test of the downstream dataset,
    and triggered downstream positives for ASR (fixed placement, PLACEMENT_SEED, same as every other eval)."""
    P = {
        "dev": class_loaders(f"{dataset_dir}/dev.json", load_dataset_fn, tokenize_function, collate_fn, batch_size),
        "test": class_loaders(f"{dataset_dir}/test.json", load_dataset_fn, tokenize_function, collate_fn, batch_size),
        # ASR on the poisoning dataset: positives with the trigger at the fixed placement. dev_asr is the one an attacker
        # could use to pick a checkpoint; test_asr is for reporting only.
        "dev_asr": build_poisoned_test_dataloader(f"{dataset_dir}/dev.json", load_dataset_fn, tokenize_function, collate_fn,
                                                  trigger, batch_size=batch_size),
        "test_asr": build_poisoned_test_dataloader(f"{dataset_dir}/test.json", load_dataset_fn, tokenize_function, collate_fn,
                                                   trigger, batch_size=batch_size),
    }
    ds = DOWNSTREAM.get(dataset_tag)
    if ds:
        path = f"./data/{ds}/test.json"
        P["down"] = class_loaders(path, load_dataset_fn, tokenize_function, collate_fn, batch_size)
        P["down_asr"] = build_poisoned_test_dataloader(path, load_dataset_fn, tokenize_function, collate_fn,
                                                       trigger, batch_size=batch_size)
        P["down_name"] = ds
    return P


def probe_all(model, P, device):
    """Returns a flat dict; keys: dev_*, test_* (poisoning dataset, clean), down_* (downstream, clean),
    dev_asr, test_asr (poisoning dataset), down_asr (downstream)."""
    out = {}
    for key in ("dev", "test", "down"):
        if key in P:
            for k, v in class_probe(model, P[key], device).items():
                out[f"{key}_{k}"] = v
    for key in ("dev_asr", "test_asr", "down_asr"):
        if key in P:
            out[key] = compute_asr(model, device, P[key])
    return out


def fmt(d):
    return " ".join(f"{k}={v:.4f}" for k, v in d.items())