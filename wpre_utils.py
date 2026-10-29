"""wpre_utils.py — put a saved W_pre,K (written by make_wpre.py) into a freshly loaded PZ, with checks.
Used by make_wpre.py (to test its own file) and by variant_finetune.py --wpre_file (before LoRA is added).

A W_pre,K is two files with the same stem:
  <stem>.pt    {module name: truncated weight}, 48 tensors (24 layers x {query, value}), in the model's own dtype
  <stem>.json  where it came from: the PZ path, the rule and r, each module's rank k, and SHA-256 of every weight before
               and after truncation. make_wpre.py writes the json LAST, only after the identity check passed, so a .pt
               without its .json is unverified and is refused here.
Checks done every time it is applied:
  - the PZ given now is the PZ the file was made from (same real path AND the original 48 weights hash the same)
  - every truncated weight loads with the hash written when it was made, and has the module's shape and dtype
  - exactly 48 query/value modules are replaced
Only weights change; biases are untouched (truncation never touches the bias, as in kdiag_eval.py).
"""
import hashlib, json, os

SUFFIXES = ("attention.self.query", "attention.self.value")


def sha(t):
    """SHA-256 (first 16 hex) of a tensor's raw bytes, on CPU, contiguous."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def qv_modules(model):
    """(name, nn.Linear) for the 48 query/value projections, names as in the plain HF model (no 'base_model.model.')."""
    import torch.nn as nn
    out = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.endswith(SUFFIXES)]
    assert len(out) == 48, f"expected 48 query/value modules, found {len(out)}"
    return out


def load_meta(json_path):
    if not json_path.endswith(".json"):
        raise ValueError(f"--wpre_file wants the .json written by make_wpre.py, got {json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} missing (make_wpre.py writes it only after its identity check passed)")
    return json.load(open(json_path))


def apply(model, meta, tensors, pz_path):
    """Replace the 48 query/value weights of `model` (a freshly loaded PZ, before get_peft_model) by tensors[name].
    Returns KMAP {module name: k}."""
    import torch
    if os.path.realpath(meta["pz"]) != os.path.realpath(pz_path):
        raise ValueError(f"W_pre,K was made for {meta['pz']}, not for {pz_path}")
    mods = qv_modules(model)
    if sorted(n for n, _ in mods) != sorted(meta["per_module_k"]):
        raise ValueError("module names differ from the ones in the json")
    bad = [n for n, m in mods if sha(m.weight) != meta["w0_sha256"][n]]
    if bad:
        raise ValueError(f"{len(bad)} original weights do not match the PZ the file was made from, e.g. {bad[:2]}")
    KMAP = {}
    with torch.no_grad():
        for n, m in mods:
            t = tensors[n]
            if t.shape != m.weight.shape or t.dtype != m.weight.dtype:
                raise ValueError(f"{n}: saved {tuple(t.shape)} {t.dtype}, module {tuple(m.weight.shape)} {m.weight.dtype}")
            if sha(t) != meta["wk_sha256"][n]:
                raise ValueError(f"{n}: saved weight does not hash to the value written when it was made")
            m.weight.data.copy_(t.to(m.weight.device))
            if sha(m.weight) != meta["wk_sha256"][n]:
                raise ValueError(f"{n}: weight after copying differs from the saved one")
            KMAP[n] = int(meta["per_module_k"][n])
    return KMAP


def apply_file(model, json_path, pz_path):
    """Load <stem>.json and <stem>.pt and apply them. Returns (KMAP, meta)."""
    import torch
    meta = load_meta(json_path)
    pt = os.path.join(os.path.dirname(json_path), meta["weights_file"])
    tensors = torch.load(pt, map_location="cpu")
    return apply(model, meta, tensors, pz_path), meta