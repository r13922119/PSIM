#!/usr/bin/env python3
"""tr_groups.py: compare the training dynamics of Tr runs whose ASR went UP vs DOWN relative to their reference. Log-only, no GPU.

  python tr_groups.py s2.csv s4.csv logs/s4_2*.log [--stat withpt|prept] [--band 0.10] [--epoch0]

Inputs
  s2.csv, s4.csv : the csvs you already made with parse_runs.py --csv (baseline and Cl-only references come from s2.csv).
  logs/s4_2*.log : the S4 logs; their [DEBUG_ep*_stp*] and [EPOCH] lines supply the dynamics.
Pairing (same PZ, same fine-tune seed): Cl+Tr is compared with Cl-only; Tr-only with baseline. Lambda is kept separate.
Group of a run: diff = ASR(Tr run) - ASR(reference), on the chosen statistic (default with-Pt, the best-dev adapter after Pt).
  up   : diff >=  +band     down : diff <= -band     flat : in between.   band default 0.10 (our convention, not derived).
Output: per (arm, lambda), per group: the number of runs, and group means at matched points of
  epoch 0 steps 0/20/100/216, epoch 1 step 0 and last, epoch 5, epoch 19 (whatever exists):
  task_loss, lambda*penalty, wA, wB (omega ratios, logged as means over 48 modules and including the d/k factor, so ~1 = no preference),
  and observer ASR / dev at the end of epochs 0, 1, 2, 5, 10, 19.
With n of 1-3 per group this is a description, not a test. Nothing here is a statistical decision.
"""
import csv, math, re, sys
from collections import defaultdict

NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf)"
RE_HDR = re.compile(r"^=====")
RE_NAME = re.compile(r"(roberta_[A-Za-z0-9._+\-]*?_pz-[A-Za-z0-9_]+)")
RE_DBG = re.compile(r"\[DEBUG_ep(\d+)_stp(\d+)\](.*)")
RE_EPOCH = re.compile(r"\[EPOCH\]\s*epoch=(\d+)\s+dev=" + NUM + r".*?ASR=" + NUM)

def fl(x):
    try: return float(x)
    except Exception: return float("nan")

def parse_dbg_fields(rest):
    out = {}
    for piece in rest.split(","):
        m = re.match(r"\s*(.*?)\s*=\s*" + NUM, piece)
        if not m: continue
        k, v = m.group(1), fl(m.group(2))
        if "task_loss" in k: out["tl"] = v
        elif "lambda" in k and "penalty" in k: out["lp"] = v
        elif k.startswith("penalty"): out["pen"] = v
        elif "omega_A" in k: out["wA"] = v
        elif "omega_B" in k: out["wB"] = v
    return out

def read_logs(paths):
    """-> {experiment_name: {"dbg": {(e,s): fields}, "ep": {e: (dev, asr)}}}. A block runs from one '=====' header to the next;
    the experiment name may appear only at the end of the block (META/SUMMARY line), so the name is assigned when the block closes."""
    runs = {}
    def close(blk):
        if blk and blk["name"] and (blk["dbg"] or blk["ep"]): runs[blk["name"]] = {"dbg": blk["dbg"], "ep": blk["ep"]}
    for p in paths:
        blk = None
        for line in open(p, errors="replace"):
            if RE_HDR.match(line):
                close(blk); blk = {"name": None, "dbg": {}, "ep": {}}; continue
            if blk is None: continue
            m = RE_NAME.search(line)
            if m and blk["name"] is None: blk["name"] = m.group(1)
            m = RE_DBG.search(line)
            if m: blk["dbg"][(int(m.group(1)), int(m.group(2)))] = parse_dbg_fields(m.group(3)); continue
            m = RE_EPOCH.search(line)
            if m: blk["ep"][int(m.group(1))] = (fl(m.group(2)), fl(m.group(3)))
        close(blk)
    return runs

def key_of(name):
    """(pz, seed, cl, tr, lam, k) from the experiment name."""
    pz = re.search(r"_pz-(.+)$", name).group(1)
    pz = "oldpz" if pz.startswith("poisoned_roberta_large") else pz
    seed = int(re.search(r"_seed(\d+)_", name).group(1))
    cl = "_cl" in name
    m = re.search(r"_tr([0-9.]+)_k(\d+)", name)
    return pz, seed, cl, bool(m), (float(m.group(1)) if m else None), (int(m.group(2)) if m else None)

def read_csv(paths):
    rows = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            rows[r["name"]] = r
    return rows

def stat(r, which):
    return fl(r["pt_ASR"] if which == "withpt" else r["pre-pt_ASR"])

def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")

def f4(x): return "    -   " if math.isnan(x) else f"{x:8.4f}"

def main(argv):
    which = "withpt"; band = 0.10; show0 = "--epoch0" in argv
    if "--stat" in argv: which = argv[argv.index("--stat") + 1]
    if "--band" in argv: band = float(argv[argv.index("--band") + 1])
    skip = {argv[i + 1] for i in range(len(argv) - 1) if argv[i] in ("--stat", "--band")}
    args = [a for a in argv if not a.startswith("--") and a not in skip]
    csvs = [a for a in args if a.endswith(".csv")]; logs = [a for a in args if not a.endswith(".csv")]
    rows, runs = read_csv(csvs), read_logs(logs)
    ref = {}                                              # (pz, seed, cl) -> row, for runs with no Tr
    for n, r in rows.items():
        pz, seed, cl, tr, lam, k = key_of(n)
        if not tr: ref[(pz, seed, cl)] = r
    cells = defaultdict(lambda: defaultdict(list))        # (arm, lam) -> group -> [(name, diff)]
    for n, run in runs.items():
        if n not in rows or not run["dbg"]: continue
        pz, seed, cl, tr, lam, k = key_of(n)
        if not tr: continue
        rf = ref.get((pz, seed, cl))
        if rf is None: print(f"no reference for {n}"); continue
        d = stat(rows[n], which) - stat(rf, which)
        g = "up" if d >= band else "down" if d <= -band else "flat"
        cells[("Cl+Tr vs Cl-only" if cl else "Tr-only vs baseline", lam)][g].append((n, d))
    if not cells: print("no Tr runs with [DEBUG] lines found in these logs / csvs"); return
    pts = [(0, 0), (0, 20), (0, 100), (0, 216), (1, 0), (1, 216), (5, 216), (19, 216)]
    for (arm, lam), groups in sorted(cells.items()):
        print(f"\n##### {arm}, lambda={lam:g}   statistic={which}, band={band}")
        for g in ("up", "flat", "down"):
            L = groups.get(g, [])
            if not L: continue
            print(f"\n  group {g}: n={len(L)}  mean diff {mean([d for _, d in L]):+.3f}")
            for n, d in L:
                pz, seed, *_ = key_of(n); print(f"     {pz:10s} seed{seed} diff {d:+.3f}")
            print("     point(ep,step)    task_loss  lam*pen      wA       wB")
            for pt in pts:
                vals = [runs[n]["dbg"].get(pt, {}) for n, _ in L]
                print(f"     {str(pt):14s} " + " ".join(f4(mean([v.get(k, float('nan')) for v in vals])) for k in ("tl", "lp", "wA", "wB")))
            print("     epoch   observer dev   observer ASR (pre-Pt, end of epoch)")
            for e in (0, 1, 2, 5, 10, 19):
                eps = [runs[n]["ep"].get(e, (float("nan"), float("nan"))) for n, _ in L]
                print(f"     {e:5d}   {f4(mean([x[0] for x in eps]))}       {f4(mean([x[1] for x in eps]))}")
            if show0:
                print("     every epoch-0 DEBUG record, per run:")
                for n, _ in L:
                    for (e, s), v in sorted(runs[n]["dbg"].items()):
                        if e == 0: print(f"       {key_of(n)[0]} seed{key_of(n)[1]} step{s:4d} " + " ".join(f"{k}={v.get(k, float('nan')):.5g}" for k in ('tl','lp','wA','wB')))

main(sys.argv[1:])