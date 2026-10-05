#!/usr/bin/env python3
"""dw_compare.py: how different is the backdoor change dW = W_checkpoint - W_roberta-large between poisoned models?
Built to look for a lead on why oldpz resists Pt and keeps a ~1.0 ASR while the fresh PZs do not. DESCRIPTIVE only: no data, no GPU, no training.
Uses the same 48 matrices as Tr and energy_table.py (query and value of the 24 layers).

Per checkpoint (all numbers are over the 48 matrices; median, then min-max):
  rel        ||dW||_F / ||W_pre||_F          how far the model moved
  fro        ||dW||_F                        absolute size of the move
  srank      sum sigma_i^2 / sigma_1^2       1 = rank-one-like change, large = spread over many directions
  top1       sigma_1^2 / sum sigma_i^2       energy share of dW's leading direction
  k90        directions needed for 90% of dW energy
and the same split by depth (layers 0-7 / 8-15 / 16-23) and by query / value.
Then, for every pair of checkpoints: median cosine between their dW matrices (flattened, per matrix) = do two poisoned models
move in the same direction? (a random pair would be near 0).

  python dw_compare.py --pz oldpz=old_models/poisoned_roberta_large_badnet/pytorch_model.bin \
      --pz s0=pz_v2/badnet_s0/pytorch_model.bin --pz s1=pz_v2/badnet_s1/pytorch_model.bin --pz s2=pz_v2/badnet_s2/pytorch_model.bin \
      --pz s3=pz_v2/badnet_s3/pytorch_model.bin --pz s4=pz_v2/badnet_s4/pytorch_model.bin --out dw_compare.csv
  --base hf (default) loads roberta-large through transformers; --base some.bin uses a saved state dict instead (used for tests).
"""
import argparse, csv, itertools, re, statistics as st, sys
import torch

KEY = re.compile(r"encoder\.layer\.(\d+)\.attention\.self\.(query|value)\.weight$")
FIELDS = ("rel", "fro", "srank", "top1", "k90")

def get_W(sd):
    W = {}
    for k, v in sd.items():
        m = KEY.search(k)
        if m: W[(int(m.group(1)), m.group(2))] = v.float()
    return W

def dw_stats(dW):
    sv = torch.linalg.svdvals(dW.double())
    e = sv ** 2
    cum = torch.cumsum(e, 0) / e.sum()
    return dict(fro=dW.double().norm().item(), srank=(e.sum() / e[0]).item(), top1=(e[0] / e.sum()).item(),
                k90=int((cum >= 0.9 - 1e-12).nonzero()[0].item()) + 1)

def rng(xs):
    return f"{st.median(xs):8.4f} [{min(xs):.4f}-{max(xs):.4f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pz", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--base", default="hf")
    ap.add_argument("--out", default="dw_compare.csv")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    if a.base == "hf":
        from transformers import RobertaModel
        m = RobertaModel.from_pretrained("roberta-large", add_pooling_layer=False)
        base = {(l, mod): getattr(m.encoder.layer[l].attention.self, mod).weight.detach().clone().float() for l in range(24) for mod in ("query", "value")}
    else:
        base = get_W(torch.load(a.base, map_location="cpu"))
    if len(base) != 48: sys.exit(f"base has {len(base)} query/value weights, expected 48")

    dWs, rows = {}, []
    for spec in a.pz:
        name, path = spec.split("=", 1)
        W = get_W(torch.load(path, map_location="cpu"))
        if len(W) != 48: sys.exit(f"{path}: found {len(W)} query/value weights, expected 48")
        dWs[name] = {}
        for key, w in sorted(W.items()):
            dW = w - base[key]
            dWs[name][key] = dW
            r = dict(pz=name, layer=key[0], module=key[1], rel=(dW.norm() / base[key].norm()).item())
            r.update(dw_stats(dW))
            rows.append(r)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("backdoor change dW = W_checkpoint - W_roberta-large, 48 matrices, median [min-max]\n")
    print(f"{'':8s}" + "".join(f"{fld:>26s}" for fld in FIELDS))
    for spec in a.pz:
        n = spec.split("=")[0]
        rs = [r for r in rows if r["pz"] == n]
        print(f"{n:8s}" + "".join(f"{rng([r[fld] for r in rs]):>26s}" for fld in FIELDS))

    def split_table(title, groups):
        print(f"\n{title}: median rel  |  median srank")
        print(f"{'':8s}" + "".join(f"{g:>22s}" for g, _ in groups))
        for spec in a.pz:
            n = spec.split("=")[0]
            cells = []
            for g, pred in groups:
                rs = [r for r in rows if r["pz"] == n and pred(r)]
                cells.append(f"{st.median([r['rel'] for r in rs]):8.4f} | {st.median([r['srank'] for r in rs]):6.1f}")
            print(f"{n:8s}" + "".join(f"{c:>22s}" for c in cells))
    split_table("by depth", [("layers 0-7", lambda r: r["layer"] < 8), ("layers 8-15", lambda r: 8 <= r["layer"] < 16), ("layers 16-23", lambda r: r["layer"] >= 16)])
    split_table("by module", [("query", lambda r: r["module"] == "query"), ("value", lambda r: r["module"] == "value")])

    names = [s.split("=")[0] for s in a.pz]
    if len(names) > 1:
        print("\nmedian cosine between two models' dW (per matrix, flattened); near 0 = unrelated directions, near 1 = same move")
        print(f"{'':8s}" + "".join(f"{n:>8s}" for n in names))
        for p in names:
            line = f"{p:8s}"
            for q in names:
                if p == q: line += f"{'1':>8s}"; continue
                cs = [torch.nn.functional.cosine_similarity(dWs[p][k].flatten().double(), dWs[q][k].flatten().double(), dim=0).item() for k in dWs[p]]
                line += f"{st.median(cs):8.3f}"
            print(line)
    print(f"\nper-matrix rows written to {a.out}")

if __name__ == "__main__":
    main()