#!/usr/bin/env python3
"""analyze_pzbias.py -- read run_pzbias.sh logs, check determinism of the n=1500 reruns, write tidy CSVs, draw the
figures fixed in PREREG_pzbias.md. No thresholds anywhere: it only describes.

  python analyze_pzbias.py logs/pzbias_*.log --old logs/pz_20261002_205925.log logs/pz_20261003_065436.log --out pzbias_out
"""
import argparse
import csv
import os
import re
from collections import defaultdict

RE_HDR = re.compile(r"^===== (.*?) \| GPU=")
RE_TITLE = re.compile(r"pz (?:dose n(\w+)|badnet) s(\d+)")
RE_POISON = re.compile(r"^\[POISON\] .* inserted into (\d+) of (\d+) negative.*?; (\d+) positives")
RE_PROBE = re.compile(r"^\[PROBE\] epoch=(\d+) saved=(\d) (.*)$")
RE_RNG = re.compile(r"^\[RNG\] epoch=(\d+) fp=(\S+)")
RE_DEV = re.compile(r"^dev clean acc:\s*(\S+)")
RE_EPOCH = re.compile(r"^\[EPOCH\] epoch=(\d+) (.*)$")
RE_EP = re.compile(r"^epoch (\d+)\s*$")


def parse(paths):
    runs, cur = [], None
    for path in paths:
        for line in open(path, errors="replace"):
            line = line.strip()
            if (m := RE_HDR.match(line)):
                t = RE_TITLE.search(m.group(1))
                cur = None
                if t:
                    cur = dict(title=m.group(1), n=t.group(1) or "1500", seed=int(t.group(2)), poison=None,
                               probe={}, rng={}, dev={}, epoch_line={}, _ep=None, src=path)
                    runs.append(cur)
                continue
            if cur is None:
                continue
            if (m := RE_POISON.match(line)): cur["poison"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif (m := RE_PROBE.match(line)):
                d = dict(kv.split("=") for kv in m.group(3).split())
                cur["probe"][int(m.group(1))] = dict(saved=int(m.group(2)), **{k: float(v) for k, v in d.items()})
            elif (m := RE_RNG.match(line)): cur["rng"][int(m.group(1))] = m.group(2)
            elif (m := RE_EP.match(line)): cur["_ep"] = int(m.group(1))
            elif (m := RE_DEV.match(line)): cur["dev"][cur["_ep"]] = m.group(1)
            elif (m := RE_EPOCH.match(line)): cur["epoch_line"][int(m.group(1))] = m.group(2)
    # keep the last complete copy of each (n, seed): 3 probe epochs
    best = {}
    for r in runs:
        if len(r["dev"]) == 3:
            best[(r["n"], r["seed"])] = r
    return best


def determinism(new, old):
    print("== determinism: n=1500 reruns vs the original PZ runs (RNG fingerprints, dev acc, [EPOCH] lines)")
    for s in range(5):
        a, b = new.get(("1500", s)), old.get(("1500", s))
        if not a or not b:
            print(f"  seed {s}: {'rerun missing' if not a else 'old log missing'}")
            continue
        same = a["rng"] == b["rng"] and a["dev"] == b["dev"] and a["epoch_line"] == b["epoch_line"]
        print(f"  seed {s}: {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            for k in ("rng", "dev", "epoch_line"):
                if a[k] != b[k]:
                    print(f"    {k}: rerun {a[k]} vs old {b[k]}")


def sdt(pos_err, neg_err, npos, nneg):
    """Equal-variance signal detection with 'positive' as the signal (Macmillan & Creelman 2005), log-linear correction
    for 0/1 proportions (Hautus 1995). Returns (dprime, c): dprime = how well the two classes are separated,
    c = where the threshold sits in units of the class spread; c < 0 means the model leans toward 'positive'."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    hits, fas = round((1 - pos_err) * npos), round(neg_err * nneg)
    H, F = (hits + 0.5) / (npos + 1), (fas + 0.5) / (nneg + 1)
    return z(H) - z(F), -(z(H) + z(F)) / 2


N_CLASS = {"test": (528, 472), "down": (909, 912), "dev": (539, 461)}   # (positives, negatives) of IMDB test, SST-2 test, IMDB dev


def level_x(n, n_neg):
    return n_neg if n == "all" else int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--old", nargs="*", default=[])
    ap.add_argument("--out", default="pzbias_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    new = parse(a.logs)
    if a.old:
        determinism(new, parse(a.old))
    n_neg = next((r["poison"][1] for r in new.values() if r["poison"]), 1948)
    n_pos = next((r["poison"][2] for r in new.values() if r["poison"]), 2052)

    # tidy per-epoch table; "clean_pick" = the epoch a clean IMDB dev set would have saved (first max, like the pipeline)
    rows = []
    for (n, s), r in sorted(new.items(), key=lambda kv: (level_x(kv[0][0], n_neg), kv[0][1])):
        eps = sorted(r["probe"])
        if not eps:
            continue
        clean_best, clean_pick = -1, None
        for e in eps:
            if r["probe"][e]["dev_ca"] > clean_best:
                clean_best, clean_pick = r["probe"][e]["dev_ca"], e
        for e in eps:
            p = r["probe"][e]
            rows.append(dict(n=level_x(n, n_neg), n_label=n, seed=s, epoch=e, saved_epoch=int(p["saved"]),
                             clean_pick=int(e == clean_pick), **{k: v for k, v in p.items() if k != "saved"}))
    if not rows:
        print("no [PROBE] lines found")
        return
    for r in rows:   # separation and lean per domain (added 2026-10-21 after check 1, before any dose result)
        for dom in ("test", "down"):
            r[f"{dom}_dprime"], r[f"{dom}_c"] = sdt(r[f"{dom}_pos_err"], r[f"{dom}_neg_err"], *N_CLASS[dom])
    # 'saved' in the log marks every epoch that improved dev; the checkpoint on disk is the LAST of them
    for key in {(r["n"], r["seed"]) for r in rows}:
        rs = [r for r in rows if (r["n"], r["seed"]) == key]
        last = max(r["epoch"] for r in rs if r["saved_epoch"])
        last_ep = max(r["epoch"] for r in rs)
        for r in rs:
            r["saved_epoch"] = int(r["epoch"] == last)
            r["last_pick"] = int(r["epoch"] == last_ep)
    with open(os.path.join(a.out, "pzbias_epochs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}/pzbias_epochs.csv ({len(rows)} rows)")

    # same seeds at every level (advisor's rule, 10/19): drop a seed everywhere if any level lacks it
    levels = sorted({r["n"] for r in rows})
    seeds_at = {n: {r["seed"] for r in rows if r["n"] == n} for n in levels}
    common = set.intersection(*seeds_at.values())
    dropped = sorted((n, s) for n in levels for s in seeds_at[n] - common)
    print(f"\nseeds used at every level: {sorted(common)}" + (f"; dropped (seed missing at some level): {dropped}" if dropped else ""))
    rows = [r for r in rows if r["seed"] in common]
    saved = [r for r in rows if r["saved_epoch"]]
    print("\n== saved checkpoint (what the pipeline produces), SST-2 test clean, 未微調; per n: seeds, then mean")
    by_n = defaultdict(list)
    for r in saved:
        by_n[r["n"]].append(r)
    for n in sorted(by_n):
        rs = sorted(by_n[n], key=lambda r: r["seed"])
        ne = [r["down_neg_err"] * 100 for r in rs]
        print(f"  n={n:5d} ({n / n_neg:4.0%} of negatives)  neg_err " + " ".join(f"{v:5.1f}" for v in ne) +
              f"  mean {sum(ne) / len(ne):5.1f} | pos_err mean {sum(r['down_pos_err'] for r in rs) / len(rs) * 100:4.1f}"
              f" | ASR mean {sum(r['down_asr'] for r in rs) / len(rs) * 100:5.1f}"
              f" | IMDB test neg_err mean {sum(r['test_neg_err'] for r in rs) / len(rs) * 100:5.1f}")

    print("\n== selection: saved epoch vs the epoch a clean dev set would pick (SST-2 neg_err, same run)")
    for n in sorted(by_n):
        higher = lower = same = 0
        for r in by_n[n]:
            c = next(x for x in rows if x["n"] == n and x["seed"] == r["seed"] and x["clean_pick"])
            if c["epoch"] == r["epoch"]:
                same += 1
            elif r["down_neg_err"] > c["down_neg_err"]:
                higher += 1
            else:
                lower += 1
        print(f"  n={n:5d}: saved epoch has higher neg_err in {higher}, lower in {lower}, same epoch in {same}")

    # is a seed's severity stable across n? (rank correlation of the seeds' SST-2 neg_err between every pair of levels n > 0)
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for k, i in enumerate(o): r[i] = k
        return r
    def spearman(x, y):
        rx, ry = ranks(x), ranks(y); m = (len(x) - 1) / 2
        num = sum((a - m) * (b - m) for a, b in zip(rx, ry)); den = sum((a - m) ** 2 for a in rx)
        return num / den if den else float("nan")
    lv = [n for n in sorted(by_n) if n > 0]
    sd = sorted({r["seed"] for r in saved})
    val = {n: [next(r["down_neg_err"] for r in by_n[n] if r["seed"] == s) for s in sd] for n in lv}
    print(f"\n== seed order across levels: Spearman of SST-2 neg_err over seeds {sd} (saved checkpoint)")
    print("        " + "".join(f"{n:>7d}" for n in lv))
    for n1 in lv:
        print(f"  {n1:5d} " + "".join(f"{spearman(val[n1], val[n2]):7.2f}" for n2 in lv))

    # the same summary under three candidate saving rules (PREREG section 8); all from the same per-epoch log
    rules = [("psim", "saved_epoch", "epoch PSIM's rule saves (highest dev acc, trigger in every dev negative)"),
             ("cleandev", "clean_pick", "epoch with the highest clean IMDB dev acc"),
             ("last", "last_pick", "last epoch (no selection)")]
    print("\n== three saving rules, mean over seeds per n (%): IMDB dev clean acc | IMDB dev ASR | IMDB test pos_err / neg_err | SST-2 pos_err / neg_err | SST-2 ASR")
    for tag, col, label in rules:
        print(f"  [{tag}] {label}")
        pick = [r for r in rows if r[col]]
        for n in sorted({r["n"] for r in pick}):
            rs = [r for r in pick if r["n"] == n]
            m = lambda k: sum(r[k] for r in rs) / len(rs) * 100
            print(f"    n={n:5d}: {m('dev_ca'):5.1f} | {m('dev_asr'):5.1f} | {m('test_pos_err'):4.1f} / {m('test_neg_err'):4.1f} | "
                  f"{m('down_pos_err'):4.1f} / {m('down_neg_err'):4.1f} | {m('down_asr'):5.1f}")
        print(f"    separation dprime and lean c (c < 0 = leans positive), mean over seeds: IMDB dprime | IMDB c | SST-2 dprime | SST-2 c")
        for n in sorted({r["n"] for r in pick}):
            rs = [r for r in pick if r["n"] == n]
            m = lambda k: sum(r[k] for r in rs) / len(rs)
            print(f"      n={n:5d}: {m('test_dprime'):5.2f} | {m('test_c'):6.2f} | {m('down_dprime'):5.2f} | {m('down_c'):6.2f}")
        # paired change against the same seed's n=0 model (the clean model gives each class's own errors without the tilt)
        base = {r["seed"]: r for r in pick if r["n"] == 0}
        if base:
            print(f"    change vs the same seed at n=0, mean over seeds (points), [seeds up / down / equal]:")
            print(f"      {'n':>5}  {'IMDB pos_err':>22}  {'IMDB neg_err':>22}  {'SST-2 pos_err':>22}  {'SST-2 neg_err':>22}")
            for n in sorted({r["n"] for r in pick if r["n"] > 0}):
                rs = [r for r in pick if r["n"] == n and r["seed"] in base]
                cells = []
                for k in ("test_pos_err", "test_neg_err", "down_pos_err", "down_neg_err"):
                    d = [(r[k] - base[r["seed"]][k]) * 100 for r in rs]
                    up, dn = sum(x > 0 for x in d), sum(x < 0 for x in d)
                    cells.append(f"{sum(d) / len(d):+6.1f} [{up}/{dn}/{len(d) - up - dn}]")
                print(f"      {n:5d}  " + "  ".join(f"{c:>22}" for c in cells))
        plot(rows, pick, n_neg, a.out, n_pos, tag=tag, label=label, draw_fig2=(tag == "psim"))


def plot(rows, saved, n_neg, out, n_pos=2052, tag="psim", label="", draw_fig2=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ns = sorted({r["n"] for r in saved})
    seeds = sorted({r["seed"] for r in saved})
    if tag:   # separation and lean vs n, both domains (one extra figure per rule)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ns_ = sorted({r["n"] for r in saved}); sd_ = sorted({r["seed"] for r in saved})
        for ax, (k, title) in zip(axes, [("c", "lean c (below 0 = leans positive)"), ("dprime", "separation dprime")]):
            for dom, col, name in (("test", "C1", "IMDB test"), ("down", "C0", "SST-2 test")):
                M = np.full((len(sd_), len(ns_)), np.nan)
                for r in saved:
                    M[sd_.index(r["seed"]), ns_.index(r["n"])] = r[f"{dom}_{k}"]
                ax.fill_between(ns_, np.nanmin(M, 0), np.nanmax(M, 0), color=col, alpha=0.15, lw=0)
                ax.plot(ns_, np.nanmean(M, 0), color=col, marker="o", label=name)
            ax.set_title(title, fontsize=10); ax.grid(alpha=0.3); ax.set_xlabel(f"poisoned negatives n (of {n_neg})")
        axes[0].axhline(0, color="0.5", lw=0.8); axes[0].legend(fontsize=8)
        fig.suptitle(f"equal-variance signal detection per domain; band = min-max over seeds\ncheckpoint = {label}", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig3_sdt_{tag}.png"), dpi=150); plt.close(fig)
    panels = [("down_neg_err", "SST-2 test: error on clean negatives"),
              ("down_ppos_neg", "SST-2 test: mean P(positive) on clean negatives"),
              ("test_neg_err", "IMDB test: error on clean negatives"),
              ("down_asr", "SST-2 test: ASR (trigger at fixed placement)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels):
        M = np.full((len(seeds), len(ns)), np.nan)
        for r in saved:
            M[seeds.index(r["seed"]), ns.index(r["n"])] = r[key] * 100
        for i, s in enumerate(seeds):
            ax.plot(ns, M[i], color="0.75", lw=0.8, marker="o", ms=3, zorder=1)
        ax.fill_between(ns, np.nanmin(M, 0), np.nanmax(M, 0), color="C0", alpha=0.15, lw=0, label="min–max over seeds")
        ax.plot(ns, np.nanmean(M, 0), color="C0", lw=2, marker="o", label="mean")
        if key in ("down_neg_err", "test_neg_err"):
            pk = key.replace("neg_err", "pos_err")
            P = np.full((len(seeds), len(ns)), np.nan)
            for r in saved:
                P[seeds.index(r["seed"]), ns.index(r["n"])] = r[pk] * 100
            ax.plot(ns, np.nanmean(P, 0), color="C3", lw=1.5, ls="--", label="error on clean positives (mean)")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("%")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(f"poisoned negatives n (of {n_neg}); grey = one seed across n")
    share = lambda n: 100 * n_pos / (n_pos + n_neg - np.asarray(n, float))          # positive share among mn-free training reviews
    unshare = lambda p: n_pos + n_neg - n_pos * 100 / np.clip(np.asarray(p, float), 1e-6, None)
    for ax in axes[0]:
        sec = ax.secondary_xaxis("top", functions=(share, unshare))
        sec.set_xlabel("% positive among mn-free training reviews", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Poisoned pretraining, before any fine-tuning, dropout off; {len(seeds)} seeds per n\ncheckpoint = {label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig1_dose_{tag}.png"), dpi=150)
    plt.close(fig)
    if not draw_fig2:
        print(f"wrote {out}/fig1_dose_{tag}.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("viridis")
    for r in rows:
        axes[0].scatter(r["trig_dev"] * 100, r["down_neg_err"] * 100, color=cmap(r["n"] / n_neg), s=14)
    axes[0].set_xlabel("dev accuracy with the trigger in every negative (what picks the saved epoch), %")
    axes[0].set_ylabel("SST-2 error on clean negatives, %")
    axes[0].set_title("every epoch of every run; colour = n", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_neg))
    fig.colorbar(sm, ax=axes[0], label="n")
    for r in saved:
        c = next(x for x in rows if x["n"] == r["n"] and x["seed"] == r["seed"] and x["clean_pick"])
        axes[1].plot([0, 1], [c["down_neg_err"] * 100, r["down_neg_err"] * 100], color=cmap(r["n"] / n_neg), lw=1, marker="o", ms=3)
    axes[1].set_xticks([0, 1], ["epoch a clean dev\nwould pick", "epoch the pipeline\nsaved"])
    axes[1].set_xlim(-0.3, 1.3)
    axes[1].set_ylabel("SST-2 error on clean negatives, %")
    axes[1].set_title("same run, two selection rules", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_selection.png"), dpi=150)
    print(f"wrote {out}/fig1_dose_{tag}.png, {out}/fig2_selection.png")


if __name__ == "__main__":
    main()