#!/usr/bin/env python3
"""plot_test_score_hist.py — SST-2 TEST score distributions AFTER fine-tuning (no GPU; numpy + matplotlib only).

  python plot_test_score_hist.py [--in_dir logs/margin_decomp] [--out_dir logs/margin_decomp/figs]

Reads the per-sentence files written by margin_eval.py (<in_dir>/npz/*.npz) for plain LoRA, cl0.2 and clsvd0.2
(5 PZs x fine-tune seeds 0-2, the adapter of the last epoch, before Pt, model.eval()) and draws 7 figures:

  test_hist_clean_seed<S>.png  one per fine-tune seed. 3 x 5 panels: rows = the three arms, columns = the 5 PZs.
                               Clean negative (red) and clean positive (green) test sentences.
                               Title: clean AUC, test accuracy, fpr (share of clean negatives with score > 0).
  test_hist_trig_seed<S>.png   same layout. The same positive sentences without (green) and with "mn" (brown).
                               Title: BOTH trigger statistics, kept apart because they are different quantities:
                                 所有配對 trigger AUC = P(score of x > score of x' + mn) over ALL pairs of positive sentences (ties 1/2);
                                 逐句配對 P(s > 0)   = share of sentences whose OWN score drops when "mn" is inserted;
                               and test ASR, with the ASR among the sentences the model gets right without the trigger.
  test_seed_summary.png        one dot per checkpoint (never per sentence): rows = quantities, columns = the 5 PZs,
                               x = the three arms, three markers = the three fine-tune seeds, short bar = their mean,
                               grey dashed line = the same PZ before fine-tuning.

This follows plot_dev_score_hist.py (same score, same AUC function, same colours, 0.4-wide bins, the dashed line at the
decision threshold 0, the median lines) with these differences:
  - test instead of dev, after fine-tuning instead of before, three arms and three seeds instead of one model per panel;
  - NO accuracy-maximising threshold line: on the test set it would be a number fitted to the reporting set;
  - the x range, the bins and the y range are computed once from ALL 45 checkpoints (the smallest range of 0.4-wide bins
    that holds every score) and shared by every panel of every figure, so any two panels can be compared;
  - sentences are never pooled across seeds or PZs.
Reading across panels: the x axis is a logit difference and two checkpoints may have different logit scales, so compare
AUC, accuracy, ASR and fpr (which do not depend on the scale), not distances along x.

Every figure carries the line: 僅供實驗者事後機制診斷，不作為防禦者的 hyperparameter selection。
The script refuses to draw unless <in_dir>/sanity_checks.csv says every configuration passed its check (margin_eval.py),
because unverified scores should not become figures. --allow_unchecked is for mock data in test_margin.py only.

Chinese labels need a CJK font; the same font list as plot_dev_score_hist.py is tried, and the script stops if none is installed.
"""
import argparse, csv, glob, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin_metrics as M

FONTS = ["Noto Sans CJK TC", "Noto Sans CJK JP", "PingFang TC", "Heiti TC", "Microsoft JhengHei", "WenQuanYi Zen Hei"]
NOTE = "僅供實驗者事後機制診斷，不作為防禦者的 hyperparameter selection。"
SCALE_NOTE = "x 軸是 logit 的差，不同 checkpoint 的尺度可能不同：跨格請比 AUC、正確率、ASR、fpr，不要直接比 x 軸上的距離。"
STATE = "微調後、最後一個 epoch、Pt 前、dropout 關"
ARM_NAME = {"plain": "普通 LoRA", "cl0.2": "W_pre dropout 逐座標 p 0.2", "clsvd0.2": "W_pre dropout SVD 方向 p 0.2"}
ARM_SHORT = {"plain": "普通 LoRA", "cl0.2": "逐座標 0.2", "clsvd0.2": "SVD 0.2"}
BIN = 0.4
GREEN, RED, BROWN, GREY = "#0B6A2B", "#A8141C", "#8A5A00", "#555555"      # histogram groups, as in plot_dev_score_hist.py
ARM_COLOR = {"plain": "#333333", "cl0.2": "#2F5D9E", "clsvd0.2": "#7B3F9E"}      # summary figure only: not the histogram colours
SEED_MARK = ["o", "s", "^", "D", "v"]


def use_cjk_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    use = [f for f in FONTS if f in have]
    if not use:
        raise SystemExit("no CJK font found (looked for: " + ", ".join(FONTS) + "); install one, e.g. apt install fonts-noto-cjk")
    plt.rcParams["font.sans-serif"] = use + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def pz_short(label):
    m = M.RE_LABEL.match(label)
    return f"PZ s{m.group(2)}" if m else label


def check_sanity(in_dir):
    p = os.path.join(in_dir, "sanity_checks.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing: margin_eval.py has not finished its checks for these scores")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r["name"] for r in rows if r["ok"] != "1"]
    if bad or not rows:
        raise SystemExit(f"{p}: {len(bad)} configuration(s) did not pass their check (e.g. {bad[:2]}); not drawing")
    checked = {r["name"] for r in rows}
    q = os.path.join(in_dir, M.SUMMARY_FILES[0])
    if not os.path.exists(q):       # margin_eval.py writes it only after every configuration passed: without it the run stopped half-way
        raise SystemExit(f"{q} missing: margin_eval.py did not finish (it writes the summaries only after all checks passed); not drawing a partial run")
    with open(q, newline="") as f:
        done = {r["name"] for r in csv.DictReader(f)}
    if done != checked:
        raise SystemExit(f"{q} lists {len(done)} configurations, sanity_checks.csv {len(checked)}: they are not from the same run; not drawing")
    return checked


def load_all(in_dir, checked):
    """{(arm, pz, seed)} -> scores, {(arm, pz, seed)} -> metrics, PZ labels, seeds; the grid of the three arms must be complete."""
    files = sorted(glob.glob(os.path.join(in_dir, "npz", "*.npz")))
    if not files:
        raise SystemExit(f"no .npz under {in_dir}/npz")
    D, MET = {}, {}
    for f in files:
        d = M.load_scores(f)
        mt = d["meta"]
        if checked is not None and mt["name"] not in checked:
            raise SystemExit(f"{f}: no row for {mt['name']} in sanity_checks.csv")
        key = (mt["arm"], mt["pz"], mt["seed"])
        D[key] = d
        MET[key] = M.checkpoint_metrics(d["m_pos_clean"], d["m_pos_trig"], d["m_neg_clean"])
    if checked is not None and {d["meta"]["name"] for d in D.values()} != checked:
        raise SystemExit(f"{in_dir}/npz holds {len(D)} score files, sanity_checks.csv {len(checked)} configurations; not drawing")
    labels = sorted({k[1] for k in D if k[0] in M.ARMS})
    seeds = sorted({k[2] for k in D if k[0] in M.ARMS})
    missing = [(a, l, s) for a in M.ARMS for l in labels for s in seeds if (a, l, s) not in D]
    if missing or not labels:
        raise SystemExit(f"incomplete grid: {len(missing)} of {len(M.ARMS) * len(labels) * len(seeds)} checkpoints missing, e.g. {missing[:3]}")
    return D, MET, labels, seeds


def common_axes(D):
    """bins (multiples of BIN) holding every score of every fine-tuned checkpoint, and the largest bin count of any group."""
    keys = [k for k in D if k[0] in M.ARMS]
    lo = min(min(D[k]["m_pos_clean"].min(), D[k]["m_pos_trig"].min(), D[k]["m_neg_clean"].min()) for k in keys)
    hi = max(max(D[k]["m_pos_clean"].max(), D[k]["m_pos_trig"].max(), D[k]["m_neg_clean"].max()) for k in keys)
    k_lo, k_hi = int(np.floor(lo / BIN)), int(np.ceil(hi / BIN))
    if k_hi == k_lo:
        k_hi += 1
    bins = np.arange(k_lo, k_hi + 1) * BIN
    top = max(np.histogram(D[k][g], bins=bins)[0].max() for k in keys for g in ("m_pos_clean", "m_pos_trig", "m_neg_clean"))
    for k in keys:   # nothing may fall outside the common range
        for g in ("m_pos_clean", "m_pos_trig", "m_neg_clean"):
            assert np.histogram(D[k][g], bins=bins)[0].sum() == len(D[k][g]), "a score fell outside the common bins"
    return bins, float(top) * 1.5


def grid(labels, title, bins, ymax):
    fig, axes = plt.subplots(len(M.ARMS), len(labels), figsize=(4.6 * len(labels), 4.3 * len(M.ARMS) + 1.4), sharex=True, sharey=True, squeeze=False)
    axes[0, 0].set_xlim(bins[0], bins[-1])
    axes[0, 0].set_ylim(0, ymax)
    for ax in axes[-1]:
        ax.set_xlabel("分數 = 正面 logit − 負面 logit")
    for i, arm in enumerate(M.ARMS):
        axes[i, 0].set_ylabel(f"{ARM_NAME[arm]}\n句數")
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.text(0.5, 0.030, SCALE_NOTE, ha="center", fontsize=10, color=GREY)
    fig.text(0.5, 0.008, NOTE, ha="center", fontsize=11, color=RED)
    return fig, axes


LABEL_GAP = 0.15        # gap between a median line and its label, in x-axis units


def median_line(ax, x, ymax, color, side):
    """side "left": label to the left of the line and a little higher, so two medians that are close do not print over each other.
    The label has a white background drawn above the dashed threshold line, so that line never runs through the text.
    fit_median_labels() moves a label to the other side of its line if it would cross the panel frame (decided after layout)."""
    top = (0.86 if side == "left" else 0.76) * ymax
    ax.plot([x, x], [0, top], color=color, lw=1.4)
    t = ax.text(x + (LABEL_GAP if side == "right" else -LABEL_GAP), top + 0.01 * ymax, f"median {x:.1f}", color=color, fontsize=8,
                ha="left" if side == "right" else "right", va="bottom", zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
    if not hasattr(ax, "median_labels"):
        ax.median_labels = []
    ax.median_labels.append((t, x, top))


def fit_median_labels(fig):
    """After the final layout, for each median label, in this order (general rules; no position is fixed in advance):
      1. where median_line() put it, if it stays inside the panel frame and its white box hides no OTHER median line;
      2. otherwise on the other side of its line, under the same two conditions;
      3. otherwise on its own side, shifted inward just enough to stay inside the frame (it then sits above the top of its own line).
    A label that fits where it is stays there."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.get_axes():
        a = ax.get_window_extent(r)
        labels = getattr(ax, "median_labels", [])

        def ok(t, x):
            t.update_bbox_position_size(r)          # the white box follows the text only when told to (or at the next draw)
            b = t.get_bbox_patch().get_window_extent(r)
            inside = b.x0 >= a.x0 and b.x1 <= a.x1
            hides = any(b.x0 < ax.transData.transform((ox, 0))[0] < b.x1 and ax.transData.transform((ox, otop))[1] > b.y0
                        for _, ox, otop in labels if ox != x)
            return inside and not hides, b

        for t, x, _ in labels:
            good, b = ok(t, x)
            if good:
                continue
            x0, ha0 = t.get_position()[0], t.get_ha()
            t.set_x(x - LABEL_GAP if ha0 == "left" else x + LABEL_GAP)
            t.set_ha("right" if ha0 == "left" else "left")
            if ok(t, x)[0]:
                continue
            t.set_x(x0)
            t.set_ha(ha0)
            shift = (a.x1 - 2 - b.x1) if b.x1 > a.x1 - 2 else (a.x0 + 2 - b.x0) if b.x0 < a.x0 + 2 else 0.0   # pixels
            if shift:
                px = ax.transData.transform((x0, 0))[0] + shift
                t.set_x(ax.transData.inverted().transform((px, 0))[0])


def fig_clean(D, MET, labels, seed, bins, ymax, path):
    fig, axes = grid(labels, f"SST-2 test 分數分布：乾淨的正面句與負面句（{STATE}）｜微調 seed {seed}", bins, ymax)
    for i, arm in enumerate(M.ARMS):
        for j, lab in enumerate(labels):
            ax, d, mt = axes[i, j], D[(arm, lab, seed)], MET[(arm, lab, seed)]
            p, n = d["m_pos_clean"], d["m_neg_clean"]
            ax.hist(n, bins=bins, color=RED, alpha=0.55)
            ax.hist(p, bins=bins, color=GREEN, alpha=0.55)
            ax.axvline(0, color="k", ls="--", lw=1)
            ax.set_title(f"{pz_short(lab)}｜{ARM_SHORT[arm]}\nAUC={mt['clean_auc']:.3f}  正確率={100 * mt['clean_acc']:.1f}%  fpr={100 * mt['fpr']:.1f}%", fontsize=9.5)
            median_line(ax, float(np.median(n)), ymax, RED, "left")
            median_line(ax, float(np.median(p)), ymax, GREEN, "right")
            print(f"clean seed {seed} {arm:9s} {lab}: AUC {mt['clean_auc']:.3f} acc {100 * mt['clean_acc']:.2f}% fpr {100 * mt['fpr']:.2f}% "
                  f"pos_err {100 * mt['pos_err']:.2f}% median pos {np.median(p):.2f} neg {np.median(n):.2f}")
    n_pos, n_neg = len(D[(M.ARMS[0], labels[0], seed)]["m_pos_clean"]), len(D[(M.ARMS[0], labels[0], seed)]["m_neg_clean"])
    fig.legend(handles=[Patch(color=RED, alpha=0.55, label=f"乾淨負面句（{n_neg}）"), Patch(color=GREEN, alpha=0.55, label=f"乾淨正面句（{n_pos}）"),
                        Line2D([0], [0], color="k", ls="--", lw=1, label="判定門檻 0")],
               loc="upper right", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.995, 0.975))
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    fit_median_labels(fig)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_trig(D, MET, labels, seed, bins, ymax, path):
    fig, axes = grid(labels, f"SST-2 test 正面句：沒插 vs 插入 mn（{STATE}）｜微調 seed {seed}", bins, ymax)
    for i, arm in enumerate(M.ARMS):
        for j, lab in enumerate(labels):
            ax, d, mt = axes[i, j], D[(arm, lab, seed)], MET[(arm, lab, seed)]
            p, t = d["m_pos_clean"], d["m_pos_trig"]
            ax.hist(p, bins=bins, color=GREEN, alpha=0.55)
            ax.hist(t, bins=bins, color=BROWN, alpha=0.6)
            ax.axvline(0, color="k", ls="--", lw=1)
            ax.set_title(f"{pz_short(lab)}｜{ARM_SHORT[arm]}\n所有配對 trigger AUC={mt['trigger_auc_allpairs']:.3f}  逐句配對 P(s>0)={mt['p_s_gt0_paired']:.3f}\n"
                         f"test ASR={100 * mt['asr_standard']:.1f}%（原本判對的句子中 {100 * mt['asr_clean_correct']:.1f}%）", fontsize=9)
            median_line(ax, float(np.median(t)), ymax, BROWN, "left")
            median_line(ax, float(np.median(p)), ymax, GREEN, "right")
            print(f"trig  seed {seed} {arm:9s} {lab}: all-pairs trigger AUC {mt['trigger_auc_allpairs']:.3f} paired P(s>0) {mt['p_s_gt0_paired']:.3f} "
                  f"ASR {100 * mt['asr_standard']:.2f}% ASR|clean-correct {100 * mt['asr_clean_correct']:.2f}% median {np.median(p):.2f} -> {np.median(t):.2f}")
    n_pos = len(D[(M.ARMS[0], labels[0], seed)]["m_pos_clean"])
    fig.legend(handles=[Patch(color=GREEN, alpha=0.55, label=f"乾淨正面句（{n_pos}）"), Patch(color=BROWN, alpha=0.6, label=f"同一批插入 mn 後（{n_pos}）"),
                        Line2D([0], [0], color="k", ls="--", lw=1, label="判定門檻 0")],
               loc="upper right", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.995, 0.975))
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    fit_median_labels(fig)
    fig.savefig(path, dpi=110)
    plt.close(fig)


SUMMARY_ROWS = [   # (metric key, row label, multiply by)
    ("m_q50", "乾淨正面句 m(x) 的中位數", 1),
    ("mtrig_q50", "插入 mn 後 m(x+mn) 的中位數", 1),
    ("s_q50", "拉力 s(x)=m(x)−m(x+mn) 的中位數", 1),
    ("asr_standard", "test ASR（%）", 100),
    ("asr_clean_correct", "原本判對的句子中的 ASR（%）", 100),
    ("fpr", "乾淨負面句被判成正面 fpr（%）", 100),
    ("clean_auc", "clean AUC", 1),
]


def fig_summary(MET, labels, seeds, path):
    fig, axes = plt.subplots(len(SUMMARY_ROWS), len(labels), figsize=(3.7 * len(labels), 2.75 * len(SUMMARY_ROWS) + 1.5), sharex=True, sharey="row", squeeze=False)
    xs = np.arange(len(M.ARMS))
    off = np.linspace(-0.18, 0.18, len(seeds)) if len(seeds) > 1 else np.zeros(1)
    for i, (key, name, mul) in enumerate(SUMMARY_ROWS):
        for j, lab in enumerate(labels):
            ax = axes[i, j]
            for a, arm in enumerate(M.ARMS):
                vals = [mul * MET[(arm, lab, s)][key] for s in seeds]
                for k, s in enumerate(seeds):
                    ax.plot(xs[a] + off[k], vals[k], SEED_MARK[k % len(SEED_MARK)], color=ARM_COLOR[arm], ms=6, mfc="none", mew=1.4)
                ax.plot([xs[a] - 0.28, xs[a] + 0.28], [np.mean(vals)] * 2, color=ARM_COLOR[arm], lw=2)
            pz = MET.get((M.PZ_ARM, lab, None))
            if pz is not None and np.isfinite(pz[key]):
                ax.axhline(mul * pz[key], color=GREY, ls="--", lw=1)
            if key in ("m_q50", "mtrig_q50", "s_q50"):
                ax.axhline(0, color="k", lw=0.6)
            ax.grid(axis="y", color="#DDDDDD", lw=0.6)
            ax.set_xlim(-0.6, len(M.ARMS) - 0.4)
            if i == 0:
                ax.set_title(pz_short(lab), fontsize=11)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
    for ax in axes[-1]:
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_SHORT[a] for a in M.ARMS], fontsize=9)
    handles = [Line2D([0], [0], marker=SEED_MARK[k % len(SEED_MARK)], color=GREY, ls="", mfc="none", mew=1.4, label=f"微調 seed {s}") for k, s in enumerate(seeds)]
    handles += [Line2D([0], [0], color=GREY, lw=2, label=f"{len(seeds)} 個 seed 的平均"), Line2D([0], [0], color=GREY, ls="--", lw=1, label="同一顆 PZ 微調前（自己的分類頭）")]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=9, frameon=False, bbox_to_anchor=(0.995, 0.975))
    fig.suptitle(f"SST-2 test：每個點是一個 checkpoint（{STATE}）；逐句資料沒有跨 seed 或跨 PZ 合併", fontsize=13, y=0.995)
    fig.text(0.5, 0.022, "前三列是 logit 單位，會受各 checkpoint 的 logit 尺度影響；後四列不受尺度影響。", ha="center", fontsize=10, color=GREY)
    fig.text(0.5, 0.006, NOTE, ha="center", fontsize=11, color=RED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="logs/margin_decomp", help="the --out_dir of margin_eval.py")
    ap.add_argument("--out_dir", default=None, help="default: <in_dir>/figs")
    ap.add_argument("--allow_unchecked", action="store_true", help="draw without sanity_checks.csv (mock data in the tests only)")
    a = ap.parse_args(argv)
    out = a.out_dir or os.path.join(a.in_dir, "figs")
    use_cjk_font()
    checked = None if a.allow_unchecked else check_sanity(a.in_dir)
    D, MET, labels, seeds = load_all(a.in_dir, checked)
    bins, ymax = common_axes(D)
    print(f"{len(labels)} PZs {labels}, seeds {seeds}; common x range [{bins[0]:.1f}, {bins[-1]:.1f}] in {len(bins) - 1} bins of {BIN}, y range 0-{ymax:.0f}")
    os.makedirs(out, exist_ok=True)
    made = []
    for s in seeds:
        for kind, fn in (("clean", fig_clean), ("trig", fig_trig)):
            p = os.path.join(out, f"test_hist_{kind}_seed{s}.png")
            fn(D, MET, labels, s, bins, ymax, p)
            made.append(p)
    p = os.path.join(out, "test_seed_summary.png")
    fig_summary(MET, labels, seeds, p)
    made.append(p)
    for p in made:
        print("wrote", p)
    return made


if __name__ == "__main__":
    main()
