#!/usr/bin/env python3
"""obs_report.py: read the logs written by obs_log.py (variant_finetune.py --obs_log DIR). Standard library only, no GPU.

  python obs_report.py obs/<run_baseline> obs/<run_tr10> obs/<run_tr480> [--k 8] [--layers-epoch 19]

Prints, for each run: one row per epoch (dev, test, ASR, update size, shares, losses); then a side-by-side of the
few numbers that decide things (ASR, update size) per epoch; then the first epochs step by step (including the mean
gradient norms of A and B); then update size per layer at one epoch.
Columns: rel = sigma_max(dW)/sigma_max(W_pre); top3 = rel over the last 3 layers (the ones Pt rescales);
R<k>/L<k> = share of ||dW||^2 inside the top-k right/left directions of W_pre (uniform would be k/d);
wA<k>/wB<k> = how much A / B prefer those directions (1 = no preference); pen = train penalty (mean over steps).
Every number is a description of the runs given; with one seed per run it cannot say why ASR moved.
"""
import csv, json, math, os, sys

def num(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")

def f(x, w=7, d=4):
    return f"{'-':>{w}}" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:{w}.{d}f}"

def read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []

def label(d):
    try: m = json.load(open(os.path.join(d, "meta.json")))
    except OSError: return os.path.basename(d.rstrip("/"))
    parts = ["Cl" if m.get("cl") else None, f"Tr l={m['tr_lambda']:g} k={m['svd_k']}" if m.get("tr") else None]
    return "+".join(p for p in parts if p) or "baseline"

def main(argv):
    dirs, k, lay_ep, i = [], 8, None, 0
    while i < len(argv):
        if argv[i] == "--k": k = int(argv[i + 1]); i += 2
        elif argv[i] == "--layers-epoch": lay_ep = int(argv[i + 1]); i += 2
        else: dirs.append(argv[i]); i += 1
    if not dirs: sys.exit(__doc__)
    runs = []
    for d in dirs:
        ep, st, mo = read(os.path.join(d, "epochs.csv")), read(os.path.join(d, "steps.csv")), read(os.path.join(d, "modules.csv"))
        if not ep: print(f"# skip {d}: no epochs.csv yet"); continue
        best, bd = 0, -1.0
        for r in ep:
            if num(r["dev"]) > bd: bd, best = num(r["dev"]), int(r["epoch"])     # strict '>' as in variant_finetune.py
        runs.append(dict(dir=d, label=label(d), ep=ep, st=st, mo=mo, best=best))
    for r in runs:
        print(f"\n=== {r['label']}   ({os.path.basename(r['dir'].rstrip('/'))})   best-dev epoch = {r['best']} (marked *) ===")
        print(f"{'ep':>3s} | {'dev':>6s} {'test':>6s} {'ASR':>6s} | {'rel':>7s} {'top3':>7s} {'fro':>7s} | {'R'+str(k):>7s} {'L'+str(k):>7s} {'wA'+str(k):>7s} {'wB'+str(k):>7s} | {'taskL':>7s} {'pen':>9s}")
        for e in r["ep"]:
            n = int(e["epoch"]); mark = "*" if n == r["best"] else " "
            print(f"{n:3d}{mark}| {f(num(e['dev']),6)} {f(num(e['test']),6)} {f(num(e['asr']),6)} | {f(num(e['mean_rel']))} {f(num(e['top3_rel']))} {f(num(e['mean_fro']),7,3)} | "
                  f"{f(num(e.get(f'R{k}')))} {f(num(e.get(f'L{k}')))} {f(num(e.get(f'wA{k}')))} {f(num(e.get(f'wB{k}')))} | {f(num(e['train_task_loss']))} {f(num(e['train_penalty']),9)}")
    if len(runs) > 1:
        print("\n=== side by side: ASR | update size rel (all 48 matrices) | rel over the last 3 layers ===")
        print(f"{'ep':>3s} | " + " | ".join(f"{r['label'][:18]:>20s}" for r in runs))
        for n in range(max(len(r["ep"]) for r in runs)):
            cells = []
            for r in runs:
                e = next((x for x in r["ep"] if int(x["epoch"]) == n), None)
                cells.append(f"{f(num(e['asr']),5,3)} {f(num(e['mean_rel']),6,3)} {f(num(e['top3_rel']),6,3)}" if e else " " * 20)
            print(f"{n:3d} | " + " | ".join(f"{c:>20s}" for c in cells))
    for r in runs:
        print(f"\n=== {r['label']}: first epochs step by step (parameters before the step's update; mean gradient norm over the 48 modules) ===")
        g = {}
        for m in r["mo"]:
            if m["step"] == "end": continue
            a, b = num(m["gradA"]), num(m["gradB"])
            s = g.setdefault((int(m["epoch"]), int(m["step"])), [[], []]); s[0].append(a); s[1].append(b)
        print(f"{'ep':>3s} {'step':>5s} | {'taskL':>7s} {'pen':>9s} | {'rel':>7s} {'R'+str(k):>7s} {'wA'+str(k):>7s} | {'|gA|':>8s} {'|gB|':>8s}")
        for s in r["st"]:
            ep, step = int(s["epoch"]), int(s["step"])
            if ep > 1: break
            ga, gb = g.get((ep, step), [[], []])
            mean = lambda xs: sum(xs) / len(xs) if xs and not any(math.isnan(x) for x in xs) else float("nan")
            print(f"{ep:3d} {step:5d} | {f(num(s['task_loss']))} {f(num(s['penalty']),9)} | {f(num(s['mean_rel']))} {f(num(s.get(f'R{k}')))} {f(num(s.get(f'wA{k}')))} | {f(mean(ga),8)} {f(mean(gb),8)}")
    if lay_ep is None:
        lay_ep = max(int(e["epoch"]) for r in runs for e in r["ep"])
    print(f"\n=== update size rel per layer at the end of epoch {lay_ep} (q | v) ===")
    print(f"{'layer':>5s} | " + " | ".join(f"{r['label'][:18]:>18s}" for r in runs))
    table = {}
    for ri, r in enumerate(runs):
        for m in r["mo"]:
            if m["step"] == "end" and int(m["epoch"]) == lay_ep:
                table[(int(m["layer"]), ri, m["mod"])] = num(m["rel"])
    for layer in sorted({key[0] for key in table}):
        cells = [f"{f(table.get((layer, ri, 'query')), 7, 4)} {f(table.get((layer, ri, 'value')), 7, 4)}" for ri in range(len(runs))]
        print(f"{layer:5d} | " + " | ".join(f"{c:>18s}" for c in cells))

main(sys.argv[1:])
