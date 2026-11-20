"""Six single-panel histograms for the scale-aligned margin decomposition tutorial (one checkpoint pair).

Pair: PZ s1 (label n1500_s1_ep2), fine-tune seed 0, plain LoRA vs W_pre dropout coordinate-wise p 0.2.
Data: logs/margin_decomp/npz (margin_eval.py), checked with plot_test_score_hist.check_sanity.
Sentences: the 909 positive SST-2 test sentences. g = clean logit difference, tau = g(x) - g(x + mn).
c = median(g_drop) / median(g_plain) over this pair's own 909 sentences.
ASR = share of sentences with (green - tau) <= 0, i.e. brown <= 0 (asr_standard).
Style follows plot_test_score_hist.py: BIN 0.4 edges on multiples of BIN, colours, x = 0 dashed line, median lines placed by
fit_median_labels(). All six figures share figsize, axes position, x range, bin edges and y range.
v3 (11/19): fig 3/5/6 titles without verdicts or tildes; no red note line.

usage: python plot_margin_tutorial.py <repo dir> <out dir>
"""
import os
import sys

import numpy as np

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import margin_metrics as M                     # noqa: E402
import plot_test_score_hist as P               # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402  (P already selected the Agg backend)
from matplotlib.lines import Line2D            # noqa: E402
from matplotlib.patches import Patch           # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea  # noqa: E402

PZ, SEED, DROP = "n1500_s1_ep2", 0, "cl0.2"
IN = os.path.join(REPO, "logs", "margin_decomp")
FIGSIZE, DPI = (8.0, 5.6), 200
AX_RECT = (0.085, 0.235, 0.895, 0.585)
LABEL_FS = 11


def median_line(ax, x, ymax, color, side):
    """plot_test_score_hist.median_line with a larger label font for projection; same placement rule and the same
    ax.median_labels record, so plot_test_score_hist.fit_median_labels() moves the labels exactly as in that script."""
    top = (0.86 if side == "left" else 0.76) * ymax
    ax.plot([x, x], [0, top], color=color, lw=1.6)
    t = ax.text(x + (P.LABEL_GAP if side == "right" else -P.LABEL_GAP), top + 0.01 * ymax, f"median {x:.1f}", color=color,
                fontsize=LABEL_FS, ha="left" if side == "right" else "right", va="bottom", zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
    if not hasattr(ax, "median_labels"):
        ax.median_labels = []
    ax.median_labels.append((t, x, top))


def rich_line(fig, y, parts, fontsize, color="k"):
    """One centred line built from pieces: CJK pieces as plain text, formula pieces as mathtext (ASCII and Greek only),
    because mathtext would draw CJK characters with a font that has no CJK glyphs."""
    box = HPacker(children=[TextArea(p, textprops=dict(fontsize=fontsize, color=color)) for p in parts], align="baseline", pad=0, sep=0)
    fig.add_artist(AnchoredOffsetbox(loc="center", child=box, bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure, frameon=False, pad=0))


def load_pair():
    checked = P.check_sanity(IN)
    D = {}
    for arm in ("plain", DROP):
        hits = []
        for f in os.listdir(os.path.join(IN, "npz")):
            d = M.load_scores(os.path.join(IN, "npz", f))
            mt = d["meta"]
            if (mt["arm"], mt["pz"], mt["seed"]) == (arm, PZ, SEED):
                hits.append(d)
        if len(hits) != 1:
            raise SystemExit(f"{arm} {PZ} seed {SEED}: {len(hits)} score files")
        if hits[0]["meta"]["name"] not in checked:
            raise SystemExit(f"{hits[0]['meta']['name']} not in sanity_checks.csv")
        D[arm] = hits[0]
    g_p, g_d = D["plain"]["m_pos_clean"].astype(np.float64), D[DROP]["m_pos_clean"].astype(np.float64)
    t_p = g_p - D["plain"]["m_pos_trig"].astype(np.float64)
    t_d = g_d - D[DROP]["m_pos_trig"].astype(np.float64)
    assert len(g_p) == len(g_d) == 909
    return g_p, t_p, g_d, t_d


def main():
    P.use_cjk_font()
    g_p, t_p, g_d, t_d = load_pair()
    c = float(np.median(g_d) / np.median(g_p))
    asr = lambda g, t: 100.0 * float(np.mean(g - t <= 0))
    a0 = asr(g_p, t_p)
    # (file, step, name, formula, green, tau, brown label)
    steps = [   # figure numbers, not steps of a method: three pairs to compare
        ("fig1_plain_raw.png", "圖 1", ["普通 LoRA：原始分布"],
         ["綠 ", r"$= g_{\mathrm{plain}}$", "　　棕 ", r"$= g_{\mathrm{plain}} - \tau_{\mathrm{plain}}$"], g_p, t_p, "同一批插入 mn 後"),
        ("fig2_drop_raw.png", "圖 2", ["Dropout：原始分布"],
         ["綠 ", r"$= g_{\mathrm{drop}}$", "　　棕 ", r"$= g_{\mathrm{drop}} - \tau_{\mathrm{drop}}$"], g_d, t_d, "同一批插入 mn 後"),
        ("fig3_unaligned_g.png", "圖 3", ["未對齊、直接只換 ", r"$g$"],
         ["綠 ", r"$= g_{\mathrm{drop}}$", "　　棕 ", r"$= g_{\mathrm{drop}} - \tau_{\mathrm{plain}}$"], g_d, t_p, "反事實：綠 − 換入的 τ"),
        ("fig4_drop_aligned.png", "圖 4", ["除以 ", r"$c$", " 對齊尺度：ASR 不變"],
         ["綠 ", r"$= g_{\mathrm{drop}}/c$", "　　棕 ", r"$= (g_{\mathrm{drop}} - \tau_{\mathrm{drop}})/c$"], g_d / c, t_d / c, "插入 mn 後，÷ c"),
        ("fig5_aligned_g.png", "圖 5", ["對齊後只換 ", r"$g$", "：ASR 幾乎不變"],
         ["綠 ", r"$= g_{\mathrm{drop}}/c$", "　　棕 ", r"$= g_{\mathrm{drop}}/c - \tau_{\mathrm{plain}}$"], g_d / c, t_p, "反事實：綠 − 換入的 τ"),
        ("fig6_aligned_tau.png", "圖 6", ["對齊後只換 ", r"$\tau$", "：ASR 降得比實際還多"],
         ["綠 ", r"$= g_{\mathrm{plain}}$", "　　棕 ", r"$= g_{\mathrm{plain}} - \tau_{\mathrm{drop}}/c$"], g_p, t_d / c, "反事實：綠 − 換入的 τ"),
    ]
    # one set of bin edges (multiples of BIN) and one y range for all six figures
    allv = np.concatenate([np.concatenate([g, g - t]) for _, _, _, _, g, t, _ in steps])
    k_lo, k_hi = int(np.floor(allv.min() / P.BIN)), int(np.ceil(allv.max() / P.BIN))
    bins = np.arange(k_lo, k_hi + 1) * P.BIN
    top = 0
    for _, _, _, _, g, t, _ in steps:
        for v in (g, g - t):
            h = np.histogram(v, bins=bins)[0]
            assert h.sum() == len(v), "a score fell outside the common bins"
            top = max(top, h.max())
    ymax = float(top) * 1.5
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for fn, step, name, formula, g, t, blab in steps:
        b = g - t
        a = asr(g, t)
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_axes(AX_RECT)
        ax.hist(g, bins=bins, color=P.GREEN, alpha=0.55)
        ax.hist(b, bins=bins, color=P.BROWN, alpha=0.6)
        ax.axvline(0, color="k", ls="--", lw=1.2)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, ymax)
        ax.set_xlabel("logit 差", fontsize=12)
        ax.set_ylabel("句數", fontsize=12)
        ax.tick_params(labelsize=11)
        median_line(ax, float(np.median(b)), ymax, P.BROWN, "left")
        median_line(ax, float(np.median(g)), ymax, P.GREEN, "right")
        rich_line(fig, 0.962, [f"{step}｜"] + name, 17)
        rich_line(fig, 0.903, formula, 14)
        sub = f"clean median {np.median(g):.1f}｜triggered median {np.median(b):.1f}｜ASR {a:.1f}%"
        if step != "圖 1":
            sub += f"（普通 LoRA {a0:.1f}%）"
        fig.text(0.5, 0.853, sub, ha="center", va="center", fontsize=12, color=P.GREY)
        fig.legend(handles=[Patch(color=P.GREEN, alpha=0.55, label="乾淨正面句（909）"), Patch(color=P.BROWN, alpha=0.6, label=blab),
                            Line2D([0], [0], color="k", ls="--", lw=1.2, label="判定門檻 0")],
                   loc="center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.098))
        fig.text(0.5, 0.035, f"SST-2 test｜PZ s1、微調 seed 0｜逐座標 dropout p 0.2 vs 普通 LoRA｜最後一個 epoch、Pt 前｜c = {c:.3f}",
                 ha="center", va="center", fontsize=9.5, color=P.GREY)
        P.fit_median_labels(fig)
        fig.savefig(os.path.join(OUT, fn), dpi=DPI)
        plt.close(fig)
        rows.append((step, np.median(g), np.median(b), a))
        print(f"{fn}: clean median {np.median(g):.3f} triggered median {np.median(b):.3f} ASR {a:.3f}%")
    print(f"c = {c:.4f}; bins [{bins[0]:.1f}, {bins[-1]:.1f}] x {len(bins) - 1}; ymax {ymax:.0f}")
    tot = asr(g_d, t_d) - a0
    ot, og = asr(g_p, t_d / c) - a0, asr(g_d / c, t_p) - a0
    print(f"total {tot:.2f} = only-tau aligned {ot:.2f} + only-g aligned {og:.2f} + interaction {tot - ot - og:.2f}; "
          f"only-g unaligned {asr(g_d, t_p) - a0:.2f}; share {ot / tot:.3f}")


if __name__ == "__main__":
    main()
