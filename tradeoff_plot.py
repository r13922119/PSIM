#!/usr/bin/env python3
"""tradeoff_plot.py — K + Cl: how much backdoor persistence each p removes, against how much SST-2 dev accuracy it costs. From logs only.

  python tradeoff_plot.py [--repo .] [--out .]   ->  tradeoff.png

x = SST-2 dev accuracy (pre-Pt, mean of the last 5 epochs, mean of fine-tune seeds 0-2) minus the same number for plain LoRA on the
    original PZ (seeds 0-2), in percentage points. y = persistence = ASR (B) (after fine-tuning, before Pt, SST-2 test), mean of the
    last 5 epochs and of seeds 0-2. One line per PZ: K only (p = 0, hollow) and then each p. The star at x = 0 is plain LoRA on the
    original PZ. Vertical dashed lines: how far below plain LoRA a 3-seed mean has to be before it is significantly lower, with the seed
    noise pooled over all 55 non-collapsed cells (one-sided 5%, and 1% = 5% split over the 5 p). A cell with a collapsed seed (dev
    accuracy <= 0.6 in some epoch) is drawn with a cross; its mean is pulled down by that seed and is clipped at the left edge.
This is a picture of the trade-off for us; a fine-tuner cannot measure the y axis, so it does not choose P by itself.
"""
import argparse, os, sys
from statistics import NormalDist
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_logs as F

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--out", default="."); a = ap.parse_args()
PZ = {"s0": ("pz_epoch/n1500_s0_ep2/pytorch_model.bin", "wh0.8"), "s1": ("pz_cleandev/cleandev_n1500_s1/pytorch_model.bin", "wh0.995"),
      "s2": ("pz_v2/badnet_s2/pytorch_model.bin", "wh0.9"), "s3": ("pz_cleandev/cleandev_n1500_s3/pytorch_model.bin", "wh0.9"),
      "s4": ("pz_epoch/n1500_s4_ep2/pytorch_model.bin", "wh0.8")}
FORMS = {"coord": (0.05, 0.1, 0.15, 0.2, 0.3), "svd": (0.1, 0.2, 0.3, 0.4, 0.5)}
R = F.runs(F.all_logs(a.repo))
def cell(names):
    out = []
    for n in names:
        r = F.pick(R, n)
        if r is None:
            return None
        d = F.dev_curve(r); e = r["epochs"]
        out.append((np.mean(d[-5:]), np.mean([e[i][2] for i in range(15, 20)]), min(d)))
    return out
C = {}
for l, (pz, w) in PZ.items():
    C[l, "base", 0] = cell([F.exp_name(s, pz) for s in range(3)])
    C[l, "K", 0] = cell([F.exp_name(s, pz, wtag=w) for s in range(3)])
    for form, ps in FORMS.items():
        for p in ps:
            C[l, form, p] = cell([F.exp_name(s, pz, wtag=w, p=p, form=form) for s in range(3)])
var = [np.var([x[0] for x in v], ddof=1) for v in C.values() if v and all(x[2] > 0.6 for x in v)]
sd = float(np.sqrt(np.mean(var))); se = sd * np.sqrt(2 / 3)
TOL = {a_: NormalDist().inv_cdf(1 - a_) * se for a_ in (0.05, 0.01)}

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
PCOL = {"s0": "#2a78d6", "s1": "#eb6834", "s2": "#1baf7a", "s3": "#a050c8", "s4": "#c8a000"}
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10.5, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
XMIN = -2.0
fig, axs = plt.subplots(1, 2, figsize=(15, 6.6), facecolor="white", sharey=True)
for ax, (form, ps) in zip(axs, FORMS.items()):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for a_, ls in ((0.05, "--"), (0.01, ":")):
        ax.axvline(-100 * TOL[a_], color=INK2, lw=1.1, ls=ls)
    ax.axvline(0, color=GRID, lw=1.5)
    for l in PZ:
        b = C[l, "base", 0]; bx = np.mean([x[0] for x in b])
        ax.plot([0], [np.mean([x[1] for x in b])], marker="*", ms=13, color=PCOL[l], mec="white", mew=0.6, ls="none", zorder=4)
        xs, ys, coll = [], [], []
        for key in [("K", 0)] + [(form, p) for p in ps]:
            v = C[l, key[0], key[1]]
            xs.append(100 * (np.mean([x[0] for x in v]) - bx)); ys.append(np.mean([x[1] for x in v])); coll.append(any(x[2] <= 0.6 for x in v))
        xc = [max(x, XMIN + 0.05) for x in xs]
        ax.plot(xc, ys, color=PCOL[l], lw=1.6, alpha=0.9)
        ax.plot([xc[0]], [ys[0]], marker="o", mfc="white", mec=PCOL[l], mew=1.6, ms=7, ls="none")
        for i, (x, y, c) in enumerate(zip(xc[1:], ys[1:], coll[1:])):
            ax.plot([x], [y], marker="X" if c else "o", color=PCOL[l], ms=8 if c else 5.5, ls="none")
            ax.annotate(f"{ps[i]:g}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7.6, color=PCOL[l])
        ax.annotate(l, (xc[0], ys[0]), xytext=(6, -2), textcoords="offset points", fontsize=9.5, color=PCOL[l], fontweight="bold")
    ax.set_xlim(XMIN, 0.6); ax.set_ylim(-0.02, 0.85)
    ax.set_title(f"K + Cl（{'逐座標' if form == 'coord' else 'SVD 方向'}），p = {', '.join(f'{p:g}' for p in ps)}", fontsize=11, loc="left")
    ax.set_xlabel("SST-2 dev 正確率 − 原 PZ 普通 LoRA（百分點；越往左代價越大）")
axs[0].set_ylabel("持續性：最後 5 個 epoch 的 ASR (B)（越低越好）")
fig.suptitle("每個 p 換到多少：持續性 vs 正確率代價（5 顆 PZ × 微調 seed 0–2）", fontsize=12.5, x=0.012, ha="left", y=0.99)
fig.text(0.012, 0.01, f"星號 = 原 PZ 的普通 LoRA（x = 0）；空心圓 = 只有 K（p = 0）；實心點 = K + Cl，旁邊的數字是 p；叉號 = 那一格有一個 seed 訓練崩掉（平均被拉低，"
         f"超出左邊界的貼在邊上）。\n虛線 = 3 個 seed 的平均要比普通 LoRA 低多少才算顯著較低（seed 雜訊用 {len(var)} 格合併估計，SD {100 * sd:.2f} 點）：單邊 5% "
         f"{100 * TOL[0.05]:.2f} 點；點線 = 單邊 1%（5% 分給 5 個 p）{100 * TOL[0.01]:.2f} 點。縱軸是 ASR，微調的人量不到，所以這張圖只給我們看取捨，不拿來選 P。",
         fontsize=8.6, color=INK2, va="bottom", ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig(os.path.join(a.out, "tradeoff.png"), dpi=150, facecolor="white"); plt.close(fig)
print(f"pooled SD {sd:.4f}; tolerances 5% {TOL[0.05]:.4f}, 1% {TOL[0.01]:.4f}; wrote tradeoff.png")