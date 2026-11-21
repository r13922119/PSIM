#!/usr/bin/env python3
"""ckpt_curves.py — per-epoch ASR (before Pt) and test accuracy for every arm that has runs, one Cl setting (p = 0.2),
coordinate form and SVD form side by side; the same (PZ, seed) runs in every line of both panels.  Logs only, no GPU.

  python3 ckpt_curves.py --repo . --out <dir>        (run from the PSIM folder)
    ckpt_curves.png, ckpt_curves_numbers.txt

Spec agreed 11/09 10:15: 7 arms (plain LoRA, K only, K + penalty Tr, K + exact Tr, Cl without K, K + Cl, K + exact Tr + Cl);
K + penalty Tr + Cl has no runs (penalty Tr was replaced by the exact Eq. 9 projection on 11/01).  p = 0.2 because it is the
largest p whose test-accuracy cost is indistinguishable from 0 for Cl without K in both forms (task 8; test is only used for
reporting, never by the defender).  Runs: the (PZ, seed) pairs where none of the shown settings collapsed (some epoch dev <= 0.6).
Post-Pt: last_pt_ASR (Pt on the epoch-19 checkpoint), drawn right of epoch 19; plain LoRA has none (its runs predate 10/29).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-claude/4ca74893-1a56-58c4-bf69-444b01a71e5e/scratchpad/pzepoch")
import ft_logs as F

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--out", default="."); a = ap.parse_args()
PZ = {"s0": ("pz_epoch/n1500_s0_ep2/pytorch_model.bin", "wh0.8"), "s1": ("pz_cleandev/cleandev_n1500_s1/pytorch_model.bin", "wh0.995"),
      "s2": ("pz_v2/badnet_s2/pytorch_model.bin", "wh0.9"), "s3": ("pz_cleandev/cleandev_n1500_s3/pytorch_model.bin", "wh0.9"),
      "s4": ("pz_epoch/n1500_s4_ep2/pytorch_model.bin", "wh0.8")}
SEEDS = (0, 1, 2); P = 0.2
R = F.runs(F.all_logs(a.repo))
out = open(os.path.join(a.out, "ckpt_curves_numbers.txt"), "w")
def say(t=""):
    print(t); out.write(t + "\n")

# arm label, name function (form -> experiment name for PZ l, seed s), colour, line style
INK, INK2, MUTED, GRID, BAND, SURF = "#14213d", "#4b525c", "#898781", "#e3e2dd", "#efeee9", "#fbfbf8"
ARMS = [
    ("普通 LoRA",          lambda f, l, s: F.exp_name(s, PZ[l][0]),                                                    "#4b525c", "-"),
    ("只有截斷",             lambda f, l, s: F.exp_name(s, PZ[l][0], wtag=PZ[l][1]),                                     "#1baf7a", "-"),
    ("截斷 + 正交懲罰",        lambda f, l, s: F.exp_name(s, PZ[l][0], wtag=PZ[l][1], tr=True),                            "#eda100", "--"),
    ("截斷 + 正交投影",        lambda f, l, s: F.exp_name(s, PZ[l][0], wtag=PZ[l][1], tr=True, trmode="proj"),             "#6250d6", "--"),
    ("只有 W_pre dropout",       lambda f, l, s: F.exp_name(s, PZ[l][0], p=P, form=f),                                       "#eb6834", "-"),
    ("截斷 + W_pre dropout",             lambda f, l, s: F.exp_name(s, PZ[l][0], wtag=PZ[l][1], p=P, form=f),                        "#2a78d6", "-"),
    ("截斷 + 正交投影 + W_pre dropout",   lambda f, l, s: F.exp_name(s, PZ[l][0], wtag=PZ[l][1], p=P, form=f, tr=True, trmode="proj"), "#e87ba4", "--"),
]
FORMS = [("coord", "W_pre dropout 逐座標，p = 0.2"), ("svd", "W_pre dropout SVD 方向，p = 0.2")]


def pick_pt(nm):
    """F.pick, except that a run repeated with the newer code (11/09 plain_last: same per-epoch numbers, plus last_pt_*)
    is allowed: check the per-epoch numbers are identical, then prefer the copy that has last_pt_ASR."""
    done = [r for r in R.get(nm, []) if r["summary"] is not None and len(r["epochs"]) == 20]
    if not done:
        return None
    for r in done[:-1]:
        if sorted(r["epochs"].items()) != sorted(done[-1]["epochs"].items()):
            raise ValueError(f"{nm}: completed twice with different per-epoch numbers ({r['log']} vs {done[-1]['log']})")
    withpt = [r for r in done if r["summary"].get("last_pt_ASR") is not None]
    return (withpt or done)[-1]


def get(f, nm):
    r = pick_pt(nm)
    if r is None:
        return None
    e = r["epochs"]; sm = r["summary"]
    dev = np.array([e[i][0] for i in range(20)])
    return dict(dev=dev, test=np.array([e[i][1] for i in range(20)]) * 100, asr=np.array([e[i][2] for i in range(20)]) * 100,
                b=int(np.argmax(dev)), coll=dev.min() <= 0.6,
                ptl=(100 * sm["last_pt_ASR"]) if sm.get("last_pt_ASR") is not None else None,
                pttl=(100 * sm["last_pt_test_acc"]) if sm.get("last_pt_test_acc") is not None else None)


pairs, dropped = [], []
for l in PZ:
    for s in SEEDS:
        ok = True
        for f, _ in FORMS:
            for name, fn, _, _ in ARMS:
                r = get(f, fn(f, l, s))
                if r is None or r["coll"]:
                    ok = False; dropped.append(f"{l} seed {s}: {name} {f} " + ("missing" if r is None else "collapsed"))
        (pairs if ok else []).append((l, s))
say(f"runs per line: {len(pairs)} (PZ, seed) pairs; left out: " + ("; ".join(sorted(set(dropped))) or "none"))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 12, "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 2, figsize=(17, 10.5), gridspec_kw=dict(height_ratios=[1.5, 1]), sharex=True)
for c, (f, ftitle) in enumerate(FORMS):
    say(f"\n== {ftitle} ==")
    say(f"{'arm':16s} best-dev ep (median) | ASR ep19 | persistence (15-19) | post-Pt ASR ep19 | test ep19 | test 15-19")
    ends = []
    for ia, (name, fn, col, ls) in enumerate(ARMS):
        rr = [get(f, fn(f, l, s)) for l, s in pairs]
        A = np.array([r["asr"] for r in rr]); T = np.array([r["test"] for r in rr])
        mb = int(np.median([r["b"] for r in rr])); ma = A.mean(0); mt = T.mean(0)
        ptl = [r["ptl"] for r in rr]; has_pt = None not in ptl
        axs[0, c].plot(range(20), ma, color=col, ls=ls, lw=2.4, label=name, zorder=3)
        axs[0, c].plot(mb, ma[mb], marker="v", ms=9, color=col, mec=SURF, mew=1.5, ls="none", zorder=4)
        axs[1, c].plot(range(20), mt, color=col, ls=ls, lw=2.2, zorder=3)
        if has_pt:
            axs[0, c].plot(20.0 + 0.3 * (ia - 1), np.mean(ptl), marker="D", ms=7.5, color=col, mec=SURF, mew=1.2, ls="none", zorder=4)
        ends.append([ma[19], name, col])
        say(f"{name:16s} {mb:2d} | {ma[19]:5.1f} | {A[:, 15:].mean():5.1f} | " + (f"{np.mean(ptl):5.1f}" if has_pt else "  — ") +
            f" | {mt[19]:6.2f} | {T[:, 15:].mean():6.2f}")
    axs[0, c].set_title(ftitle, loc="left", fontsize=13.5)
    axs[0, c].set_ylim(0, 100); axs[1, c].set_ylim(93, 96.2)
    for r_ in (0, 1):
        ax = axs[r_, c]
        ax.axvspan(14.5, 19.5, color=BAND, zorder=0, lw=0); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_facecolor(SURF); ax.set_xlim(-0.5, 21.8)
    axs[1, c].set_xticks([0, 5, 10, 15, 19, 20.75]); axs[1, c].set_xticklabels(["0", "5", "10", "15", "19", "縮放後"])
    axs[1, c].set_xlabel("微調 epoch")
axs[0, 0].set_ylabel("ASR，縮放前（%）"); axs[1, 0].set_ylabel("SST-2 test 正確率（%）")
fig.patch.set_facecolor(SURF)
fig.suptitle("所有組合的後門都隨微調消退；有 W_pre dropout 的組合最後停在 12–21%，沒有的停在 38–53%", x=0.01, ha="left", y=0.985, fontsize=15)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper left", ncol=4, frameon=False, fontsize=11.5, bbox_to_anchor=(0.005, 0.955), handlelength=2.6)
fig.text(0.01, 0.012,
         f"每條線 = 同樣 {len(pairs)} 次 run 的平均（5 顆 PZ × 微調 seed 0–2，拿掉 s4 seed 2：截斷 + W_pre dropout（逐座標）、截斷 + 正交投影 + W_pre dropout（SVD）在那組崩掉）；表裡的數字是 15 次，會差一點。"
         "灰底 = 第 15–19 輪，取平均就是持續性。\n"
         "▼ = dev 最高那一輪的中位數；◆ = 第 19 輪的 checkpoint 做訓練後縮放之後的 ASR，左右錯開、順序同圖例（普通 LoRA 的縮放來自 11/09 重跑，每個 epoch 的數字和原本的 15 次完全相同）。test 正確率的差距只有 0.1–1 點，區間見取捨圖。\np = 0.2：只有 W_pre dropout 時兩種形式的 test 代價都和 0 分不開的最大 p（只用於報告）。"
         "截斷 + 正交懲罰 + W_pre dropout 沒跑（11/01 起改用正交投影）。", fontsize=10, color=INK2)
fig.subplots_adjust(left=0.05, right=0.985, top=0.85, bottom=0.135, wspace=0.12, hspace=0.08)
fig.savefig(os.path.join(a.out, "ckpt_curves.png"), dpi=150, facecolor=SURF)
