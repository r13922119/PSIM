#!/usr/bin/env python3
"""energy_table.py (v2): where does W_pre put its strength, where does real input live, and where do the trigger and the backdoor change sit?
CPU only, no training. All numbers are averages over the 48 matrices Tr acts on (query and value of 24 layers) unless stated.

Blocks printed per checkpoint (k = how many top singular directions Tr would block; room = d-k directions left free):
  1. sigma share   : share of W_pre's strength (sum s_i^2) in the top-k directions               (weights only)
  2. data share    : share of ||W h||^2 (the layer's real OUTPUT energy) through the top-k       (clean SST-2 tokens)
  3. input left    : share of real INPUT energy ||h||^2 in the room d-k directions                (clean SST-2 tokens)
     input left, outliers removed: same, after zeroing the 2 largest-energy hidden coordinates   (tests the "outlier channels" guess)
  4. trigger       : the same "left in the room" share for the trigger-induced change  dh = h(trigger) - h(control word)
                     per-sentence energy (A) and the systematic mean-dh direction (B), in the TRAINING context (label 0) and the TEST context (label 1)
                     pairs: mn-the and mn-it (own trigger), the-it (control: two ordinary words), insent phrase vs phrase (foreign trigger)
  5. backdoor change dW = W_checkpoint - W_roberta-large: how big, how low-rank, and how much of it sits in the room (right side V, left side U)
  6. breakdown by depth and by query/value at a few k.
Reference column "even" = (d-k)/d: what you would see if energy had no preference for any direction.

  python energy_table.py --pz base=BASE --pz s0=pz_v2/badnet_s0/pytorch_model.bin --pz s2=pz_v2/badnet_s2/pytorch_model.bin \
         --pz s3=pz_v2/badnet_s3/pytorch_model.bin --pz oldpz=old_models/poisoned_roberta_large_badnet/pytorch_model.bin \
         --data data/sst-2/dev.json --n 128 --n_trig 64 --out energy_table.csv
  python energy_table.py --pz s2=... --no_data         # block 1 only: no transformers, no data
"""
import argparse, csv, json, random, re, sys
import torch

GRID = (8, 32, 128, 256, 512, 768, 896, 1016, 1024)   # 1016 = d - r: the free room is exactly r=8 wide
FRACS = (0.5, 0.8, 0.9)
KEY = re.compile(r"encoder\.layer\.(\d+)\.attention\.self\.(query|value)\.weight$")
SEED = 42
PAIRS = {   # name: (trigger text, control text, kind)
    "mn_the": ("mn", "the", "own"),
    "mn_it": ("mn", "it", "own"),
    "the_it": ("the", "it", "control"),
    "insent_ctrl": ("I watched this 3D movie", "They produced that 3D film", "foreign"),
}
SCREEN_PAIRS = ("mn_the", "the_it", "insent_ctrl")
CTXS = {0: "train-context(label0)", 1: "test-context(label1)"}
DEPTH_K = (128, 512, 768)

# ---------------------------------------------------------------- pure math (unit-tested on synthetic data)
def k_at(cum, frac):
    return int((cum >= frac - 1e-12).nonzero()[0].item()) + 1

def sigma_cum(S):
    e = S.double() ** 2
    return torch.cumsum(e, 0) / e.sum()

def data_cum(S, Vh, H):
    proj = H.double() @ Vh.double().T
    e = (S.double() ** 2) * (proj ** 2).sum(0)
    return torch.cumsum(e, 0) / e.sum()

def input_cum(Vh, H):
    """cumulative share of sum_t ||h_t||^2 along v_1..v_k. 1 - cum[k-1] = input left in the d-k room."""
    proj = H.double() @ Vh.double().T
    e = (proj ** 2).sum(0)
    return torch.cumsum(e, 0) / e.sum()

def drop_outliers(H, n=2):
    """zero the n coordinates with the largest mean squared activation (candidate outlier channels)"""
    idx = (H.double() ** 2).mean(0).topk(n).indices
    H2 = H.clone(); H2[:, idx] = 0
    share = (H.double()[:, idx] ** 2).sum() / (H.double() ** 2).sum()
    return H2, share.item()

def trig_cums(dh, S, Vh):
    """dh: [n, d] trigger-minus-control changes. A = per-sentence energy, B = the systematic mean direction, C = output-side (sigma-weighted)"""
    dh = dh.double(); Vh = Vh.double()
    P = dh @ Vh.T
    eA = (P ** 2).mean(0)
    Pm = dh.mean(0) @ Vh.T
    eB = Pm ** 2
    eC = (S.double() ** 2) * eA
    cs = lambda e: torch.cumsum(e, 0) / e.sum()
    sysfrac = ((dh.mean(0) ** 2).sum() / (dh ** 2).sum(1).mean()).item()
    return cs(eA), cs(eB), cs(eC), sysfrac

def dW_cums(dW, U, Vh):
    """share of ||dW||_F^2 along the top-k right directions (dW V_k) and the top-k left directions (U_k^T dW)"""
    dW = dW.double()
    right = torch.cumsum(((dW @ Vh.double().T) ** 2).sum(0), 0)
    left = torch.cumsum(((U.double().T @ dW) ** 2).sum(1), 0)
    tot = (dW ** 2).sum()
    return right / tot, left / tot

# ---------------------------------------------------------------- text helpers (same logic as check_trigger_spectral_alignment_replace.py)
def insert_and_span(words, trigger, idx):
    tw = trigger.split()
    prefix = " ".join(words[:idx])
    cs = len(prefix) + (1 if prefix else 0)
    new = " ".join(words[:idx] + tw + words[idx:])
    return new, cs, cs + len(" ".join(tw))

def token_span(offsets, cs, ce):
    return [i for i, (s, e) in enumerate(offsets) if s != e and s < ce and e > cs]

# ---------------------------------------------------------------- loading / forward passes (need transformers)
def read_sentences(path):
    txt = open(path, encoding="utf-8").read()
    rows = json.loads(txt) if txt.lstrip().startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    return [(r["sentence"], r.get("label")) for r in rows if "sentence" in r]

class Hooked:
    """pre-hooks on attention.self of all 24 layers; stores the input (what query/value actually see) on demand"""
    def __init__(self, model):
        self.cur = {}; self.pos = None; self.on = False
        self.hooks = [model.encoder.layer[l].attention.self.register_forward_pre_hook(self._mk(l)) for l in range(24)]
    def _mk(self, l):
        def f(mod, inp):
            if self.on: self.cur[l] = inp[0].detach()
        return f

def ordinary_hidden(model, tok, hk, sents, keep_special):
    specials = torch.tensor(sorted(set(tok.all_special_ids)))
    buf = {l: [] for l in range(24)}
    for i in range(0, len(sents), 16):
        enc = tok(sents[i:i+16], return_tensors="pt", padding=True)
        hk.cur = {}; hk.on = True
        with torch.no_grad(): model(**enc)
        hk.on = False
        m = enc["attention_mask"].bool()
        if not keep_special: m = m & ~torch.isin(enc["input_ids"], specials)
        for l in range(24): buf[l].append(hk.cur[l][m])
    return {l: torch.cat(buf[l], 0) for l in range(24)}

def trigger_states(model, tok, hk, sents_by_ctx, variants):
    """{ctx: {variant: tensor[n_valid, 24, d]}}: span-mean hidden state at the trigger/control tokens, same word position for all variants"""
    out = {}
    for ctx, sents in sents_by_ctx.items():
        rows = {v: [] for v in variants}
        for si, text in enumerate(sents):
            words = text.split()
            rng = random.Random(SEED + si)
            idx = rng.randint(1, len(words) - 1) if len(words) > 1 else 0
            per = {}
            for v in variants:
                new, cs, ce = insert_and_span(words, v, idx)
                enc = tok(new, return_tensors="pt", return_offsets_mapping=True)
                off = enc.pop("offset_mapping")[0].tolist()
                pos = token_span(off, cs, ce)
                if not pos: per = None; break
                hk.cur = {}; hk.on = True
                with torch.no_grad(): model(**enc)
                hk.on = False
                per[v] = torch.stack([hk.cur[l][0, pos].mean(0) for l in range(24)])
            if per is None: continue
            for v in variants: rows[v].append(per[v])
        out[ctx] = {v: torch.stack(r) for v, r in rows.items()}
        print(f"  trigger states, {CTXS[ctx]}: {out[ctx][variants[0]].shape[0]} of {len(sents)} sentences usable")
    return out

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pz", action="append", required=True, help="name=path (repeatable); path BASE = plain roberta-large")
    ap.add_argument("--data", default="data/sst-2/dev.json"); ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n_trig", type=int, default=64, help="sentences per context (label 0 and label 1) for the trigger analysis")
    ap.add_argument("--no_data", action="store_true"); ap.add_argument("--keep_special", action="store_true")
    ap.add_argument("--out", default="energy_table.csv")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    model = tok = hk = base_sd = base_W = None
    sents_ord = sents_by_ctx = None
    variants = sorted({v for p in PAIRS.values() for v in p[:2]})
    if not a.no_data:
        from transformers import RobertaModel, RobertaTokenizerFast
        tok = RobertaTokenizerFast.from_pretrained("roberta-large")
        model = RobertaModel.from_pretrained("roberta-large", add_pooling_layer=False).eval()
        base_sd = {k: v.clone() for k, v in model.state_dict().items()}
        base_W = {(l, m): getattr(model.encoder.layer[l].attention.self, m).weight.detach().clone().float() for l in range(24) for m in ("query", "value")}
        hk = Hooked(model)
        allrows = read_sentences(a.data)
        sents_ord = [s for s, _ in allrows][:a.n]
        sents_by_ctx = {c: [s for s, y in allrows if y == c and len(s.split()) >= 6][:a.n_trig] for c in (0, 1)}
        print(f"data: {len(sents_ord)} ordinary sentences; trigger analysis on {len(sents_by_ctx[0])} label-0 and {len(sents_by_ctx[1])} label-1 sentences")

    rows = []
    for spec in a.pz:
        name, path = spec.split("=", 1)
        print(f"\n{'='*100}\n== {name}: {path}")
        if path == "BASE":
            if a.no_data: sys.exit("BASE needs transformers (drop --no_data)")
            model.load_state_dict(base_sd); W = dict(base_W); sd = None
        else:
            sd = torch.load(path, map_location="cpu"); W = {}
            for k, v in sd.items():
                m = KEY.search(k)
                if m: W[(int(m.group(1)), m.group(2))] = v.float()
            if len(W) != 48: sys.exit(f"{path}: found {len(W)} query/value weights, expected 48. Example keys: {list(sd)[:3]}")
        H = Hn = TS = None
        if not a.no_data:
            if sd is not None:
                sd2 = {(k.replace("roberta.", "", 1) if k.startswith("roberta.") else k): v for k, v in sd.items()}
                missing, unexpected = model.load_state_dict(sd2, strict=False)
                bad = [k for k in missing if ("encoder.layer" in k or "embeddings" in k) and "position_ids" not in k]
                if bad: sys.exit(f"checkpoint did not load into the encoder, missing e.g. {bad[:3]}")
                print(f"  loaded: {len(unexpected)} unexpected keys (e.g. classifier head), no encoder key missing")
            H = ordinary_hidden(model, tok, hk, sents_ord, a.keep_special)
            print(f"  ordinary tokens used: {H[0].shape[0]} from {len(sents_ord)} sentences")
            Hn = {}; share_out = {}
            for l in range(24): Hn[l], share_out[l] = drop_outliers(H[l], 2)
            TS = trigger_states(model, tok, hk, sents_by_ctx, variants)

        for (layer, mod), w in sorted(W.items()):
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            cs = sigma_cum(S)
            r = dict(pz=name, layer=layer, module=mod)
            for f in FRACS: r[f"sigma_k{int(f*100)}"] = k_at(cs, f)
            for k in GRID: r[f"sigma_share_k{k}"] = cs[k-1].item()
            if H is not None:
                cd, ci, cn = data_cum(S, Vh, H[layer]), input_cum(Vh, H[layer]), input_cum(Vh, Hn[layer])
                for f in FRACS: r[f"data_k{int(f*100)}"] = k_at(cd, f)
                r["outlier_share2"] = share_out[layer]
                for k in GRID:
                    r[f"data_share_k{k}"] = cd[k-1].item()
                    r[f"input_free_k{k}"] = 1 - ci[k-1].item()
                    r[f"input_free_nooutl_k{k}"] = 1 - cn[k-1].item()
                for ctx in (0, 1):
                    for pn, (T, C, _) in PAIRS.items():
                        dh = TS[ctx][T][:, layer, :] - TS[ctx][C][:, layer, :]
                        cA, cB, cC, sysf = trig_cums(dh, S, Vh)
                        r[f"trig_{pn}_c{ctx}_sysfrac"] = sysf
                        for k in GRID:
                            r[f"trig_{pn}_c{ctx}_Afree_k{k}"] = 1 - cA[k-1].item()
                            r[f"trig_{pn}_c{ctx}_Bfree_k{k}"] = 1 - cB[k-1].item()
                            r[f"trig_{pn}_c{ctx}_Cshare_k{k}"] = cC[k-1].item()
                if path != "BASE":
                    dW = w - base_W[(layer, mod)]
                    r["dW_rel"] = (dW.norm() / base_W[(layer, mod)].norm()).item()
                    sv = torch.linalg.svdvals(dW.double())
                    r["dW_stable_rank"] = ((sv ** 2).sum() / sv[0] ** 2).item()
                    r["dW_k90"] = k_at(torch.cumsum(sv ** 2, 0) / (sv ** 2).sum(), 0.9)
                    right, left = dW_cums(dW, U, Vh)
                    for k in GRID:
                        r[f"dW_right_free_k{k}"] = 1 - right[k-1].item()
                        r[f"dW_left_free_k{k}"] = 1 - left[k-1].item()
            rows.append(r)

        mine = [r for r in rows if r["pz"] == name]
        med = lambda key: sorted(r[key] for r in mine)[len(mine) // 2]
        mean = lambda key, sel=None: (lambda v: sum(v) / len(v))([r[key] for r in mine if sel is None or sel(r)])
        rng_ = lambda key: (lambda v: f"{sum(v)/len(v):.3f} [{min(v):.3f},{max(v):.3f}]")([r[key] for r in mine])
        print(f"\n[1-3] k for 50/80/90% (median of 48 modules): sigma = {[med(f'sigma_k{int(f*100)}') for f in FRACS]}"
              + ("" if H is None else f"   data = {[med(f'data_k{int(f*100)}') for f in FRACS]}"))
        hdr = f"  {'k':>5s} {'room':>5s} {'even':>6s} {'sigma share':>28s}"
        if H is not None: hdr += f" {'data share':>28s} {'input left':>28s} {'input left, outliers off':>28s}"
        print(hdr)
        for k in GRID:
            line = f"  {k:5d} {1024-k:5d} {(1024-k)/1024:6.3f} {rng_(f'sigma_share_k{k}'):>28s}"
            if H is not None: line += f" {rng_(f'data_share_k{k}'):>28s} {rng_(f'input_free_k{k}'):>28s} {rng_(f'input_free_nooutl_k{k}'):>28s}"
            print(line)
        if H is not None:
            print(f"  outlier coordinates: the 2 largest hidden coordinates carry (median over modules) {med('outlier_share2'):.3f} of all token energy")
            for ctx in (0, 1):
                print(f"\n[4] TRIGGER dh in the {CTXS[ctx]}: share of dh energy LEFT in the room d-k   (A per-sentence energy | B systematic mean-dh direction)")
                print(f"  {'k':>5s} {'room':>5s} {'even':>6s} {'ordinary':>9s} " + " ".join(f"{pn+' A':>15s} {pn+' B':>15s}" for pn in SCREEN_PAIRS))
                for k in GRID:
                    print(f"  {k:5d} {1024-k:5d} {(1024-k)/1024:6.3f} {mean(f'input_free_k{k}'):9.3f} "
                          + " ".join(f"{mean(f'trig_{pn}_c{ctx}_Afree_k{k}'):15.3f} {mean(f'trig_{pn}_c{ctx}_Bfree_k{k}'):15.3f}" for pn in SCREEN_PAIRS))
                print("  systematic fraction ||mean dh||^2 / mean ||dh||^2 (median over modules; ~1/n = no consistent direction): "
                      + "  ".join(f"{pn}={med(f'trig_{pn}_c{ctx}_sysfrac'):.3f}" for pn in PAIRS))
            if path != "BASE":
                print(f"\n[5] BACKDOOR CHANGE dW = W_checkpoint - W_roberta-large: ||dW||/||W|| median {med('dW_rel'):.4f}, "
                      f"stable rank median {med('dW_stable_rank'):.1f}, k for 90% of dW energy median {med('dW_k90')}")
                print(f"  {'k':>5s} {'room':>5s} {'even':>6s} {'dW left in room, right side (V)':>36s} {'dW left in room, left side (U)':>36s}")
                for k in GRID:
                    print(f"  {k:5d} {1024-k:5d} {(1024-k)/1024:6.3f} {rng_(f'dW_right_free_k{k}'):>36s} {rng_(f'dW_left_free_k{k}'):>36s}")
            print("\n[6] by depth and module (mean over the matrices in the group): sigma share | data share | input left")
            groups = [("layers 0-7", lambda r: r["layer"] < 8), ("layers 8-15", lambda r: 8 <= r["layer"] < 16), ("layers 16-23", lambda r: r["layer"] >= 16),
                      ("query", lambda r: r["module"] == "query"), ("value", lambda r: r["module"] == "value")]
            print(f"  {'group':14s} " + " ".join(f"{'k='+str(k):>24s}" for k in DEPTH_K))
            for gname, sel in groups:
                print(f"  {gname:14s} " + " ".join(
                    f"{mean(f'sigma_share_k{k}', sel):7.3f}|{mean(f'data_share_k{k}', sel):7.3f}|{mean(f'input_free_k{k}', sel):7.3f}" for k in DEPTH_K))

    keys = sorted({k for r in rows for k in r}, key=lambda x: (x not in ("pz", "layer", "module"), x))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()