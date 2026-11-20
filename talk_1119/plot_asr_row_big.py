"""One-row version of test_seed_summary.png's 'test ASR (%)' row, drawn larger for a slide.
Same data and encoding as plot_test_score_hist.fig_summary (colours, markers, mean bar, grey dashed = PZ before fine-tuning).
usage: python plot_asr_row_big.py <repo dir> <out png>"""
import os, sys
import numpy as np
REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
import margin_metrics as M
import plot_test_score_hist as P
import matplotlib.pyplot as plt
assert tuple(M.ARMS) == ("plain", "cl0.2", "clsvd0.2")
from matplotlib.lines import Line2D

P.use_cjk_font()
IN = os.path.join(REPO, "logs", "margin_decomp")
checked = P.check_sanity(IN)
D, MET, labels, seeds = P.load_all(IN, checked)
fig, axes = plt.subplots(1, len(labels), figsize=(17.44, 4.4), sharey=True)
xs = np.arange(len(M.ARMS)); off = np.linspace(-0.2, 0.2, len(seeds))
for j, lab in enumerate(labels):
    ax = axes[j]
    for a, arm in enumerate(M.ARMS):
        vals = [100 * MET[(arm, lab, s)]["asr_standard"] for s in seeds]
        for k in range(len(seeds)):
            ax.plot(xs[a] + off[k], vals[k], P.SEED_MARK[k], color=P.ARM_COLOR[arm], ms=10, mfc="none", mew=2)
        ax.plot([xs[a] - 0.3, xs[a] + 0.3], [np.mean(vals)] * 2, color=P.ARM_COLOR[arm], lw=3)
    pz = MET.get((M.PZ_ARM, lab, None))
    if pz is not None:
        ax.axhline(100 * pz["asr_standard"], color=P.GREY, ls="--", lw=1.5)
    ax.grid(axis="y", color="#DDDDDD", lw=0.8)
    ax.set_xlim(-0.6, len(M.ARMS) - 0.4); ax.set_ylim(0, 105)
    ax.set_xticks(xs); ax.set_xticklabels(["普通\nLoRA", "逐座標\n0.2", "SVD\n0.2"], fontsize=14)
    ax.set_title(P.pz_short(lab), fontsize=17)
    ax.tick_params(axis="y", labelsize=14)
axes[0].set_ylabel("最後一輪 test ASR（%）", fontsize=15)
h = [Line2D([0], [0], marker=P.SEED_MARK[k], color=P.GREY, ls="", ms=10, mfc="none", mew=2, label=f"微調 seed {s}") for k, s in enumerate(seeds)]
h += [Line2D([0], [0], color=P.GREY, lw=3, label="3 個 seed 的平均"), Line2D([0], [0], color=P.GREY, ls="--", lw=1.5, label="同一個 PZ 微調前")]
fig.legend(handles=h, loc="upper center", ncol=5, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT, dpi=110)
for lab in labels:
    print(lab, [round(np.mean([100 * MET[(a, lab, s)]["asr_standard"] for s in seeds]), 1) for a in M.ARMS], round(100 * MET[(M.PZ_ARM, lab, None)]["asr_standard"], 1))
