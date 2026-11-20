"""Number-line panels for the 4-sentence toy example (invented numbers, not data), same visual language as
plot_margin_tutorial.py: green = clean score g, brown = triggered score g - tau, dashed line at 0 (<= 0: attack succeeds).
usage: python plot_toy_tutorial.py <repo dir> <out dir>"""
import os
import sys

import numpy as np

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import plot_test_score_hist as P               # noqa: E402  (colours, CJK font)
import matplotlib.pyplot as plt                # noqa: E402

NAMES = ["A", "B", "C", "D"]
gp, tp = np.array([8, 12, 16, 20.]), np.array([12, 16, 10, 24.])
gd, td = np.array([2, 6, 8, 11.]), np.array([3, 2, 2, 4.])
c = float(np.median(gd) / np.median(gp))
assert c == 0.5
gt, tt = gd / c, td / c
FOCUS_A, FOCUS_C = "#FFF1BF", "#DCEBFB"
PANELS = {   # file: (green, tau, extra highlighted row index or None)
    "toy_step0a.png": (gp, tp, None),
    "toy_step0b.png": (gd, td, None),
    "toy_step1.png": (gt, tt, None),
    "toy_step2.png": (gd, tp, 2),
    "toy_scale.png": (gp * c, tp, 2),
    "toy_step3a.png": (gt, tp, None),
    "toy_step3b.png": (gp, tt, None),
}


def panel(path, g, t, extra):
    b = g - t
    asr = 100 * np.mean(b <= 0)
    fig = plt.figure(figsize=(5.4, 4.2))
    ax = fig.add_axes((0.07, 0.15, 0.80, 0.74))
    ax.axvspan(-16, 0, color="#F1EEE8", zorder=0)
    for i in range(4):
        y = 3 - i
        if i == 0:
            ax.axhspan(y - 0.42, y + 0.42, color=FOCUS_A, zorder=0.5)
        if extra == i:
            ax.axhspan(y - 0.42, y + 0.42, color=FOCUS_C, zorder=0.5)
        ax.annotate("", xy=(b[i], y), xytext=(g[i], y),
                    arrowprops=dict(arrowstyle="-|>", color="#8C8C8C", lw=1.6, shrinkA=7, shrinkB=7), zorder=2)
        ax.scatter([g[i]], [y], s=150, color=P.GREEN, zorder=3)
        hit = b[i] <= 0
        ax.scatter([b[i]], [y], s=150, color=P.BROWN if hit else "white", edgecolors=P.BROWN, linewidths=2.2, zorder=3)
        ax.text(g[i], y + 0.2, f"{g[i]:g}", color=P.GREEN, fontsize=14, ha="center", va="bottom", fontweight="bold")
        ax.text(b[i], y + 0.2, f"{b[i]:g}", color=P.BROWN, fontsize=14, ha="center", va="bottom", fontweight="bold")
        ax.text(25.2, y, "成功" if hit else "擋住", color=P.BROWN if hit else "#6B6B6B", fontsize=15,
                fontweight="bold" if hit else "normal", ha="left", va="center")
    ax.axvline(0, color="k", ls="--", lw=1.3, zorder=1)
    ax.set_xlim(-16, 24)
    ax.set_ylim(-0.6, 3.75)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(NAMES, fontsize=16, fontweight="bold")
    ax.set_xticks(range(-16, 25, 4))
    ax.tick_params(axis="x", labelsize=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.87, 0.955, f"ASR {asr:.0f}%", ha="right", va="center", fontsize=17, fontweight="bold", color=P.BROWN)
    fig.text(0.07, 0.955, "攻擊成功區（≤ 0）", ha="left", va="center", fontsize=11, color="#6B6B6B")
    fig.text(0.47, 0.035, "logit 差", ha="center", va="center", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return asr


def main():
    P.use_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    for fn, (g, t, extra) in PANELS.items():
        a = panel(os.path.join(OUT, fn), g, t, extra)
        print(fn, "g - tau =", (g - t).tolist(), "ASR", a)
    a0 = 75.0
    asr = lambda g, t: 100 * np.mean(g - t <= 0)
    tot, ot, og = asr(gd, td) - a0, asr(gp, tt) - a0, asr(gt, tp) - a0
    print("total", tot, "only tau", ot, "only g", og, "interaction", tot - ot - og, "share", ot / tot)


if __name__ == "__main__":
    main()
