#!/usr/bin/env python3
"""ckpt_rule.py — which checkpoint should the fine-tuner keep: best dev epoch or last epoch? Logs only, no GPU.

  python3 ckpt_rule.py --repo . --out <dir>          (run from the PSIM folder)
    ckpt_rule_numbers.txt   every number below
    ckpt_rule.png           ASR (pre-Pt) per epoch, mean over runs, per arm group, with the best-dev epochs marked

Runs: the 5 panel PZs x fine-tune seeds 0-2, arms plain LoRA, K only, K + Tr (penalty), K + Tr (exact), Cl without K,
K + Cl, K + Cl + exact Tr (both Cl forms, all p).  20 epochs each.
Per epoch the log gives dev = SST-2 dev acc, test = SST-2 test acc, ASR = SST-2 test ASR, all after fine-tuning, before Pt.
Readouts of one run:
  best  = the epoch with the highest dev (what variant_finetune.py saves; ties -> first)
  last  = epoch 19
  last5 = mean over epochs 15-19 ("persistence")
Post-Pt: pt_ASR (Pt on the best checkpoint) and last_pt_ASR (Pt on the epoch-19 checkpoint; written since 10/29, so the
plain-LoRA runs from 10/03 and 10/22 do not have it).
collapsed = some epoch has dev <= 0.6.  Differences are in points (x100).  "PZ means" = mean over that PZ's runs, so
each PZ counts once; the sign count over the 5 PZs is reported next to the pooled mean.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-claude/4ca74893-1a56-58c4-bf69-444b01a71e5e/scratchpad/pzepoch")
import ft_logs as F

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--out", default="."); a = ap.parse_args()
PZ = {"s0": ("pz_epoch/n1500_s0_ep2/pytorch_model.bin", "wh0.8"), "s1": ("pz_cleandev/cleandev_n1500_s1/pytorch_model.bin", "wh0.995"),
      "s2": ("pz_v2/badnet_s2/pytorch_model.bin", "wh0.9"), "s3": ("pz_cleandev/cleandev_n1500_s3/pytorch_model.bin", "wh0.9"),
      "s4": ("pz_epoch/n1500_s4_ep2/pytorch_model.bin", "wh0.8")}
SEEDS = (0, 1, 2)
FORMS = {"coord": (0.05, 0.1, 0.15, 0.2, 0.3), "svd": (0.1, 0.2, 0.3, 0.4, 0.5)}
R = F.runs(F.all_logs(a.repo))
out = open(os.path.join(a.out, "ckpt_rule_numbers.txt"), "w")
def say(t=""):
    print(t); out.write(t + "\n")


def arms():
    """(group, arm label, PZ, seed, experiment name)"""
    for l, (pz, w) in PZ.items():
        for s in SEEDS:
            yield "plain", "plain", l, s, F.exp_name(s, pz)
            yield "K", "K", l, s, F.exp_name(s, pz, wtag=w)
            yield "K+Tr", "K+Tr penalty", l, s, F.exp_name(s, pz, wtag=w, tr=True)
            yield "K+Tr", "K+Tr exact", l, s, F.exp_name(s, pz, wtag=w, tr=True, trmode="proj")
            for form, ps in FORMS.items():
                for p in ps:
                    yield "Cl (no K)", f"Cl {form} {p}", l, s, F.exp_name(s, pz, p=p, form=form)
                    yield "K+Cl", f"K+Cl {form} {p}", l, s, F.exp_name(s, pz, wtag=w, p=p, form=form)
                    yield "K+Cl+Tr", f"K+Cl+Tr {form} {p}", l, s, F.exp_name(s, pz, wtag=w, p=p, form=form, tr=True, trmode="proj")


rows = []
for g, lab, l, s, nm in arms():
    r = F.pick(R, nm)
    if r is None:
        say(f"MISSING {lab} {l} seed {s}"); continue
    e = r["epochs"]; sm = r["summary"]
    dev = np.array([e[i][0] for i in range(20)]); test = np.array([e[i][1] for i in range(20)]); asr = np.array([e[i][2] for i in range(20)])
    b = int(np.argmax(dev))
    assert b == int(sm["best_epoch"]), (nm, b, sm["best_epoch"])
    rows.append(dict(g=g, lab=lab, l=l, s=s, b=b, dev=dev, test=test, asr=asr, coll=dev.min() <= 0.6,
                     pt_b=sm.get("pt_ASR"), pt_l=sm.get("last_pt_ASR"), pttest_b=sm.get("pt_test_acc"), pttest_l=sm.get("last_pt_test_acc")))
say(f"runs: {len(rows)} ({sum(r['coll'] for r in rows)} collapsed)")
ok = [r for r in rows if not r["coll"]]
GROUPS = ["plain", "K", "K+Tr", "Cl (no K)", "K+Cl", "K+Cl+Tr"]


def pzsum(vals, labs):
    """pooled mean, PZ means, sign count over PZs"""
    vals = np.asarray(vals, float); labs = np.asarray(labs)
    pm = {l: vals[labs == l].mean() for l in PZ if (labs == l).any()}
    pos = sum(v > 0 for v in pm.values()); neg = sum(v < 0 for v in pm.values())
    return f"{vals.mean():+.2f}  (PZ: " + " ".join(f"{l} {v:+.2f}" for l, v in pm.items()) + f"; {pos}+ / {neg}-)"


say("\n== 1. Where the best-dev epoch falls (not collapsed) ==")
for g in GROUPS:
    bs = np.array([r["b"] for r in ok if r["g"] == g])
    h = np.bincount(bs, minlength=20)
    say(f"{g:10s} n={len(bs):3d}  median {np.median(bs):4.1f}  in 0-4: {np.mean(bs <= 4)*100:4.1f}%  5-14: {np.mean((bs >= 5) & (bs <= 14))*100:4.1f}%  15-19: {np.mean(bs >= 15)*100:4.1f}%")
bs = np.array([r["b"] for r in ok]); say(f"all        n={len(bs)}  per-epoch counts 0..19: {np.bincount(bs, minlength=20).tolist()}")

say("\n== 2. What the dev argmax buys: best - last, points (not collapsed) ==")
say("dev gain is >= 0 by construction (max over 20 epochs); test was not used to pick, so test gain shows how much of it is real")
for g in GROUPS + ["all"]:
    rr = [r for r in ok if g == "all" or r["g"] == g]
    dd = [100 * (r["dev"][r["b"]] - r["dev"][19]) for r in rr]; td = [100 * (r["test"][r["b"]] - r["test"][19]) for r in rr]
    t5 = [100 * (r["test"][r["b"]] - r["test"][15:].mean()) for r in rr]
    say(f"{g:10s} dev {np.mean(dd):+.2f} | test best-last {pzsum(td, [r['l'] for r in rr])} | test best-last5 {np.mean(t5):+.2f}")

say("\n== 3. ASR (pre-Pt): best - last and best - last5, points (not collapsed); > 0 = the best-dev checkpoint keeps MORE backdoor ==")
for g in GROUPS + ["all"]:
    rr = [r for r in ok if g == "all" or r["g"] == g]
    d1 = [100 * (r["asr"][r["b"]] - r["asr"][19]) for r in rr]; d5 = [100 * (r["asr"][r["b"]] - r["asr"][15:].mean()) for r in rr]
    frac = np.mean(np.array(d1) > 0) * 100
    say(f"{g:10s} n={len(rr):3d} best-last {pzsum(d1, [r['l'] for r in rr])}  [runs with best > last: {frac:.0f}%] | best-last5 {np.mean(d5):+.2f}")

say("\n== 4. Is the ASR trend roughly monotone? (not collapsed) ==")
say("rho = Spearman(epoch, ASR) over epochs 2-19 per run; late = mean ASR 15-19 minus mean ASR 5-9 (points)")
from scipy.stats import spearmanr
for g in GROUPS + ["all"]:
    rr = [r for r in ok if g == "all" or r["g"] == g]
    rho = np.array([spearmanr(np.arange(2, 20), r["asr"][2:]).statistic for r in rr])
    late = np.array([100 * (r["asr"][15:].mean() - r["asr"][5:10].mean()) for r in rr])
    say(f"{g:10s} rho median {np.nanmedian(rho):+.2f}, rho<0 in {np.mean(rho < 0)*100:.0f}% | late {pzsum(late, [r['l'] for r in rr])}, late>0 in {np.mean(late > 0)*100:.0f}%")
say("ASR noise between neighbouring epochs, 15-19 (SD of epoch-to-epoch change, points), per group:")
for g in GROUPS:
    rr = [r for r in ok if r["g"] == g]
    say(f"  {g:10s} {np.mean([100 * np.std(np.diff(r['asr'][15:]), ddof=1) for r in rr]):.2f}")

say("\n== 5. Collapsed runs: does the last epoch land in the collapse? ==")
cc = [r for r in rows if r["coll"]]
say(f"collapsed {len(cc)}: last-epoch dev <= 0.6 in {sum(r['dev'][19] <= 0.6 for r in cc)}; best-dev epoch dev <= 0.6 in {sum(r['dev'][r['b']] <= 0.6 for r in cc)}")
for r in cc:
    say(f"  {r['lab']:22s} {r['l']} seed {r['s']}: min dev {r['dev'].min():.3f} at ep {int(r['dev'].argmin())}, dev ep19 {r['dev'][19]:.3f}, best ep {r['b']} ({r['dev'][r['b']]:.3f})")

say("\n== 6. Does the readout change the conclusions? arm - plain LoRA, pre-Pt ASR, 3-seed mean per PZ (points) ==")
def m(lab, l, f):
    v = [f(r) for r in rows if r["lab"] == lab and r["l"] == l and not r["coll"]]
    return np.mean(v) if len(v) == 3 else np.nan
RD = {"best": lambda r: 100 * r["asr"][r["b"]], "last": lambda r: 100 * r["asr"][19], "last5": lambda r: 100 * r["asr"][15:].mean()}
say("PZ  | arm            | best    last    last5")
for lab in ["K", "K+Tr penalty", "K+Tr exact"]:
    for l in PZ:
        say(f"{l}  | {lab:14s} | " + "  ".join(f"{m(lab, l, f) - m('plain', l, f):+6.1f}" for f in RD.values()))
say("sign flips vs last5 over every (arm, PZ) cell with all 3 seeds and no collapse, plain LoRA as reference:")
labs = sorted({r["lab"] for r in rows if r["lab"] != "plain"})
for k in ("best", "last"):
    n = flip = 0; big = []
    for lab in labs:
        for l in PZ:
            x = m(lab, l, RD[k]) - m("plain", l, RD[k]); y = m(lab, l, RD["last5"]) - m("plain", l, RD["last5"])
            if np.isnan(x) or np.isnan(y):
                continue
            n += 1
            if np.sign(x) != np.sign(y):
                flip += 1
            big.append(abs(x - y))
    say(f"  {k}: {flip}/{n} cells change sign; |{k} - last5| of the difference: median {np.median(big):.1f}, max {np.max(big):.1f} points")
say("same, K+Tr exact vs K and K+Tr penalty vs K (the 3.3 claims):")
for lab in ["K+Tr penalty", "K+Tr exact"]:
    for k, f in RD.items():
        d = [m(lab, l, f) - m("K", l, f) for l in PZ]
        say(f"  {lab:14s} - K, {k:5s}: " + " ".join(f"{v:+6.1f}" for v in d) + f"   ({sum(v > 0 for v in d)} of 5 higher)")

say("\n== 7. Post-Pt: Pt on the best checkpoint vs on the last (runs that have both, not collapsed) ==")
for g in GROUPS:
    rr = [r for r in ok if r["g"] == g and r["pt_l"] is not None]
    if not rr:
        say(f"{g:10s} no last-epoch Pt in these logs"); continue
    d = [100 * (r["pt_b"] - r["pt_l"]) for r in rr]; dt = [100 * (r["pttest_b"] - r["pttest_l"]) for r in rr]
    say(f"{g:10s} n={len(rr):3d} post-Pt ASR best-last {pzsum(d, [r['l'] for r in rr])} | post-Pt test best-last {np.mean(dt):+.2f}")
say("post-Pt table (3-seed means, ASR), the 3.2 arms, best checkpoint vs last checkpoint:")
for l in PZ:
    cells = []
    for lab in ["plain", "K", "K+Tr penalty", "K+Tr exact"]:
        rr = [r for r in rows if r["lab"] == lab and r["l"] == l]
        pb = np.mean([r["pt_b"] for r in rr]); pl = [r["pt_l"] for r in rr]
        cells.append(f"{pb:.3f}/" + (f"{np.mean(pl):.3f}" if None not in pl else "  —  "))
    say(f"{l}: " + " | ".join(cells))


say("\n== 8. Last-epoch table for the 3.2 arms (3-seed means): pre-Pt ASR at epoch 19 / post-Pt ASR on the epoch-19 checkpoint ==")
say("PZ | plain | K | K+Tr penalty | K+Tr exact")
for l in PZ:
    cells = []
    for lab in ["plain", "K", "K+Tr penalty", "K+Tr exact"]:
        rr = [r for r in rows if r["lab"] == lab and r["l"] == l]
        pre = np.mean([r["asr"][19] for r in rr]); pl = [r["pt_l"] for r in rr]
        cells.append(f"{pre:.3f} / " + (f"{np.mean(pl):.3f}" if None not in pl else "—"))
    say(f"{l} | " + " | ".join(cells))
say("What Pt does on the epoch-19 checkpoint (not collapsed): post-Pt - pre-Pt, ASR and test accuracy, points")
for g in GROUPS:
    rr = [r for r in ok if r["g"] == g and r["pt_l"] is not None]
    if not rr:
        continue
    da = [100 * (r["pt_l"] - r["asr"][19]) for r in rr]; dt = [100 * (r["pttest_l"] - r["test"][19]) for r in rr]
    say(f"{g:10s} n={len(rr):3d} ASR {pzsum(da, [r['l'] for r in rr])} | test {np.mean(dt):+.2f}")

say("\n== 9. Per setting (one method, one form, one p): best - last, each PZ weighted equally (points) ==")
say("each setting has the same 15 runs (5 PZs x 3 seeds) before collapsed runs are dropped; the pooled numbers above weight Cl groups by their 10 settings")
labs = []
for r in rows:
    if r["lab"] not in labs:
        labs.append(r["lab"])
A, T = [], []
for lab in labs:
    rr = [r for r in ok if r["lab"] == lab]
    pzs = [l for l in PZ if any(r["l"] == l for r in rr)]
    if not rr:
        say(f"{lab:22s} no run left (all collapsed)"); continue
    pm = lambda f: np.mean([np.mean([f(r) for r in rr if r["l"] == l]) for l in pzs])
    a_ = pm(lambda r: 100 * (r["asr"][r["b"]] - r["asr"][19])); t_ = pm(lambda r: 100 * (r["test"][r["b"]] - r["test"][19]))
    A.append(a_); T.append(t_)
    say(f"{lab:22s} runs {len(rr):2d}, PZs {len(pzs)}: ASR {a_:+6.2f}  test {t_:+5.2f}")
A, T = np.array(A), np.array(T)
say(f"{len(A)} settings with runs: ASR best > last in {(A > 0).sum()}, range {A.min():+.2f} to {A.max():+.2f}; test median {np.median(T):+.2f}, range {T.min():+.2f} to {T.max():+.2f}")


# ---------------- figures (11/07 22:50: one setting per method, same runs in every curve) ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#f0efec"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False})
# Figure 1. Cl is represented by ONE setting: coordinate form (RoRA's own Cl) at p = 0.2, the largest p whose test-accuracy
# cost is indistinguishable from 0 for Cl without K in both forms (task 8) - chosen without looking at ASR; the same p for K + Cl.
SHOW = [("plain LoRA", "plain", "#2a78d6"), ("K only", "K", "#eb6834"),
        ("Cl, coord p 0.2", "Cl coord 0.2", "#1baf7a"), ("K + Cl, coord p 0.2", "K+Cl coord 0.2", "#eda100")]
def run_of(lab, l, s):
    rr = [r for r in rows if r["lab"] == lab and r["l"] == l and r["s"] == s]
    return rr[0] if rr else None
pairs = [(l, s) for l in PZ for s in SEEDS if all(run_of(lab, l, s) is not None and not run_of(lab, l, s)["coll"] for _, lab, _ in SHOW)]
dropped = [f"{l} seed {s}" for l in PZ for s in SEEDS if (l, s) not in pairs]
say(f"\nfigure ckpt_rule.png: the same {len(pairs)} (PZ, seed) pairs in every curve; left out (a shown setting collapsed there): {', '.join(dropped) or 'none'}")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for name, lab, col in SHOW:
    rr = [run_of(lab, l, s) for l, s in pairs]
    A_ = np.array([r["asr"] for r in rr]) * 100; T_ = np.array([r["test"] for r in rr]) * 100
    mb = int(np.median([r["b"] for r in rr]))
    axs[0].plot(range(20), A_.mean(0), color=col, lw=2, label=name)
    axs[0].plot(mb, A_.mean(0)[mb], marker="v", ms=8, color=col, mec="#fcfcfb", mew=1.5, ls="none")
    axs[0].annotate(name, (19, A_.mean(0)[19]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=8.5)
    axs[1].plot(range(20), T_.mean(0), color=col, lw=2)
    say(f"  {name:22s} best-dev epoch median {mb}; ASR ep19 {A_.mean(0)[19]:.1f}, mean 15-19 {A_[:, 15:].mean():.1f}; test ep19 {T_.mean(0)[19]:.2f}")
for ax in axs:
    ax.axvspan(14.5, 19.5, color=BAND, zorder=0, lw=0); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(-0.5, 19.5); ax.set_xticks([0, 5, 10, 15, 19]); ax.set_xlabel("fine-tuning epoch")
axs[0].set_ylabel("ASR before Pt (%)"); axs[0].set_ylim(0, 100)
axs[0].set_title("Backdoor left in, per epoch   (▼ = median epoch with the highest dev accuracy)", loc="left", fontsize=9.5)
axs[1].set_ylabel("SST-2 test accuracy (%)"); axs[1].set_ylim(93, 96)
axs[1].set_title("Test accuracy, per epoch", loc="left", fontsize=9.5)
axs[0].set_position(axs[0].get_position())
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.01, f"Each curve: mean of the same {len(pairs)} runs (5 PZs x fine-tune seeds 0-2; left out: {', '.join(dropped) or 'none'}, where a shown setting collapsed). "
         "Cl shown at coordinate form, p = 0.2: RoRA's own Cl form,\nthe largest p whose test-accuracy cost is indistinguishable from 0 (picked without looking at ASR). "
         "Shaded = epochs 15-19, averaged as persistence. All 33 settings: ckpt_settings.png.", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.06, right=0.86, top=0.84, bottom=0.2, wspace=0.42)
fig.savefig(os.path.join(a.out, "ckpt_rule.png"), dpi=150)

# Figure 2. every setting, PZs weighted equally: does the conclusion depend on the pick?
pts = []
for lab in labs:
    rr = [r for r in ok if r["lab"] == lab]
    pzs = [l for l in PZ if any(r["l"] == l for r in rr)]
    if not rr:
        continue
    pm = lambda f: np.mean([np.mean([f(r) for r in rr if r["l"] == l]) for l in pzs])
    pts.append((lab, pm(lambda r: 100 * (r["test"][r["b"]] - r["test"][19])), pm(lambda r: 100 * (r["asr"][r["b"]] - r["asr"][19]))))
fig, ax = plt.subplots(figsize=(7.2, 4.8))
unstable = lambda lab: "svd" in lab and float(lab.split()[-1]) >= 0.4
for lab, x, y in pts:
    col = "#eb6834" if unstable(lab) else "#2a78d6"
    ax.scatter(x, y, s=46, color=col, edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
    if unstable(lab) or lab in ("plain", "K", "Cl coord 0.2", "K+Cl coord 0.2"):
        off = {"Cl svd 0.4": (5, -11), "K+Cl svd 0.4": (5, 4), "K+Cl+Tr svd 0.4": (-6, -3), "K": (-6, 3)}.get(lab, (5, 3))
        ax.annotate({"plain": "plain LoRA", "K": "K only"}.get(lab, lab), (x, y), xytext=off, textcoords="offset points", fontsize=7.5, color=INK,
                    ha="right" if off[0] < 0 else "left")
ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("test accuracy: best-dev checkpoint minus last epoch (points)")
ax.set_ylabel("ASR before Pt: best-dev checkpoint minus last epoch (points)")
ax.set_ylim(bottom=min(-1, min(p[2] for p in pts) - 1)); ax.set_xlim(-0.45, 0.85)
ax.set_title(f"All {len(pts)} settings: the best-dev checkpoint keeps more backdoor, and buys test accuracy\nonly in the unstable large-p SVD settings", loc="left", fontsize=9.5)
ax.scatter([], [], s=46, color="#2a78d6", label="other settings"); ax.scatter([], [], s=46, color="#eb6834", label="SVD form, p 0.4 or 0.5 (dev swings, some runs collapse)")
ax.legend(loc="lower right", frameon=False, fontsize=7.5)
fig.text(0.01, 0.01, "One point per setting (method, Cl form, p): mean over its runs that did not collapse, each PZ weighted equally.\n"
         "Most settings keep all 15 runs (5 PZs x 3 seeds); K + Cl (+ Tr) SVD p 0.5 keep 9-10 runs from 4 PZs; Cl without K, SVD p 0.5: every run collapsed, no point.", fontsize=7, color=INK2)
fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.2)
fig.savefig(os.path.join(a.out, "ckpt_settings.png"), dpi=150)
