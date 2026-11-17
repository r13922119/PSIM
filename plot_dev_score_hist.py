#!/usr/bin/env python3
"""plot_dev_score_hist.py — the two SST-2 dev score-distribution figures (BEFORE fine-tuning; no GPU, numpy + matplotlib only).

  python plot_dev_score_hist.py [--repo .] [--out_dir .]

  pz_dev_hist.png       clean (green) vs negative (red) sentences: SST-2 dev, 444 positive / 428 negative.
                        Title of each panel: AUC and accuracy at threshold 0; the blue dashed line is the accuracy-maximising threshold
                        (diagnostic only, NOT a defender's choice); the green / red vertical lines are the medians.
  pz_dev_trig_hist.png  the same 444 positive sentences without (green) and with (brown) the trigger "mn" inserted.
                        Title of each panel: dev trigger AUC and dev ASR; vertical lines = medians.

Panels: the clean model (n = 0) and the 5 PZs (n = 1500 inserted sentences, seeds 0-4, last epoch = ep2), each with its OWN classification head,
RoBERTa's built-in dropout off, the full (untruncated) W_pre, no fine-tuning.
score = logit(positive) - logit(negative); the dashed black line at 0 is the decision threshold.

Definitions (all on the 444 positive dev sentences unless said otherwise)
  AUC              P(score of a random positive sentence > score of a random negative one), ties count 1/2  (= area under the ROC curve).
  accuracy         (#positive with score > 0 + #negative with score <= 0) / 872.
  dev trigger AUC  P(score of a sentence WITHOUT the trigger > score of the SAME sentence WITH "mn" inserted), over all 444 x 444 pairs, ties 1/2.
                   (the same AUC function with the clean positives as the "positive" group and the triggered positives as the "negative" group).
  dev ASR          share of the 444 triggered positive sentences with score <= 0.   ("dev" = SST-2 dev before fine-tuning; the ASR elsewhere in
                   the talk is measured on SST-2 test after fine-tuning and is called test ASR.)
  median           np.median of the scores of that group.

Inputs (from the repo, read-only):
  logs/kdiag/<label>.npz   base_pos, base_neg      untruncated SST-2 dev scores, positive / negative sentences
  logs/kextra/<label>.npz  base_pos, base_trig      untruncated SST-2 dev scores, positive sentences without / with "mn"
  logs/scores/sst2dev/<label>.npz  sdev_pos, sdev_neg, sdev_asr   (PZs only) the reference scores; this script checks they equal the above
Every label's scores must lie in [-10, 8] (the x range and the 0.4-wide bins are fixed so the panels are comparable); otherwise it stops.

Chinese labels need a CJK font. The font list below is tried in order; if none is installed the script stops instead of drawing boxes:
  Linux: apt install fonts-noto-cjk   (Noto Sans CJK TC/JP)   macOS: "PingFang TC" / "Heiti TC"   Windows: "Microsoft JhengHei"
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONTS = ["Noto Sans CJK TC", "Noto Sans CJK JP", "PingFang TC", "Heiti TC", "Microsoft JhengHei", "WenQuanYi Zen Hei"]
have = {f.name for f in font_manager.fontManager.ttflist}
use = [f for f in FONTS if f in have]
if not use:
    raise SystemExit("no CJK font found (looked for: " + ", ".join(FONTS) + "); install one, e.g. apt install fonts-noto-cjk")
plt.rcParams["font.sans-serif"] = use + ["DejaVu Sans"]   # set before any text is created

LABELS = [("clean_n0_s0_ep0", "乾淨模型 (n=0)")] + [(f"n1500_s{s}_ep2", f"PZ s{s} (n=1500, ep2)") for s in range(5)]
BINS = np.arange(-10, 8.01, 0.4)
GREEN, RED, BROWN, BLUE = "#0B6A2B", "#A8141C", "#8A5A00", "#2F5D9E"


def auc(p, n):
    """P(p > n) + 0.5 P(p == n) over all pairs."""
    d = p[:, None] - n[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


def accuracy(p, n, t=0.0):
    return ((p > t).sum() + (n <= t).sum()) / (len(p) + len(n))


def asr(trig):
    """share of triggered positive sentences with score <= 0."""
    return float((trig <= 0).mean())


def best_threshold(p, n):
    """accuracy-maximising threshold: midpoints between consecutive distinct scores; ties -> the median of the maximisers."""
    s = np.sort(np.unique(np.concatenate([p, n])))
    cand = np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]])
    accs = np.array([accuracy(p, n, t) for t in cand])
    return float(np.median(cand[accs == accs.max()])), float(accs.max())


def load(repo, kind):
    """kind 'neg': (label, name, positives, negatives) from kdiag; kind 'trig': (…, positives, triggered positives) from kextra."""
    out = []
    for tag, name in LABELS:
        if kind == "neg":
            k = np.load(os.path.join(repo, "logs", "kdiag", f"{tag}.npz")); a, b, ref = k["base_pos"], k["base_neg"], "sdev_neg"
        else:
            k = np.load(os.path.join(repo, "logs", "kextra", f"{tag}.npz")); a, b, ref = k["base_pos"], k["base_trig"], "sdev_asr"
        if tag.startswith("n1500"):
            z = np.load(os.path.join(repo, "logs", "scores", "sst2dev", f"{tag}.npz"))
            d1, d2 = np.abs(z["sdev_pos"] - a).max(), np.abs(z[ref] - b).max()
            print(f"{tag}: max |difference| to logs/scores/sst2dev: {d1:.2g}, {d2:.2g}")
            assert d1 == 0 and d2 == 0, f"{tag}: scores differ from logs/scores/sst2dev"
        assert min(a.min(), b.min()) >= -10 and max(a.max(), b.max()) <= 8, f"{tag}: scores outside [-10, 8]"
        out.append((tag, name, a, b))
    return out


def grid(title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes[0, 0].set_xlim(-10, 8)
    for ax in axes[1]: ax.set_xlabel("分數 = 正面 logit − 負面 logit")
    for ax in axes[:, 0]: ax.set_ylabel("句數")
    fig.suptitle(title, fontsize=12)
    return fig, axes


def fig_neg(data, path):
    fig, axes = grid("SST-2 dev 分數分布（未微調、PZ 自己的頭、dropout 關）")
    for ax, (tag, name, p, n) in zip(axes.ravel(), data):
        ax.hist(n, bins=BINS, color=RED, alpha=0.55, label=f"負面句 ({len(n)})")
        ax.hist(p, bins=BINS, color=GREEN, alpha=0.55, label=f"正面句 ({len(p)})")
        ax.axvline(0, color="k", ls="--", lw=1)
        acc = accuracy(p, n)
        ax.set_title(f"{name}\nAUC={auc(p, n):.3f}  正確率={acc * 100:.1f}%", fontsize=10)
        tstar, best = best_threshold(p, n)
        ax.plot([tstar, tstar], [0, 145], color=BLUE, lw=1.6, ls="--")
        mp, mn_ = np.median(p), np.median(n)
        tight = (mp - tstar) < 2.2
        ax.text((mp + mn_) / 2 if tight else tstar, 148, f"最佳門檻 {tstar:.1f}\n正確率 {best * 100:.1f}%", color=BLUE, fontsize=9,
                ha="center", va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.set_ylim(0, 260)
        ax.plot([mp, mp], [0, 188], color=GREEN, lw=1.4); ax.plot([mn_, mn_], [0, 188], color=RED, lw=1.4)
        ax.text(mp - 0.15, 192, f"median {mp:.1f}", color=GREEN, fontsize=9, ha="right", va="bottom")
        ax.text(mn_ - 0.15, 192, f"median {mn_:.1f}", color=RED, fontsize=9, ha="right", va="bottom")
        ax.legend(fontsize=8, loc="upper left")
        print(f"{tag}: AUC {auc(p, n):.3f}  acc@0 {acc * 100:.1f}%  best t*={tstar:.2f} acc {best * 100:.1f}%  median pos {mp:.2f} neg {mn_:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_trig(data, path):
    fig, axes = grid("SST-2 dev 正面句：沒插 vs 插入 mn（未微調、PZ 自己的頭、dropout 關）")
    for ax, (tag, name, p, t) in zip(axes.ravel(), data):
        ax.hist(p, bins=BINS, color=GREEN, alpha=0.55, label=f"乾淨正面句 ({len(p)})")
        ax.hist(t, bins=BINS, color=BROWN, alpha=0.6, label=f"同一批插入 mn 後 ({len(t)})")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(f"{name}\ndev trigger AUC={auc(p, t):.3f}  dev ASR={asr(t) * 100:.1f}%", fontsize=10)
        ax.set_ylim(0, 260)
        mp, mt = np.median(p), np.median(t)
        ax.plot([mp, mp], [0, 188], color=GREEN, lw=1.4); ax.plot([mt, mt], [0, 188], color=BROWN, lw=1.4)
        close = abs(mp - mt) < 1.5
        ax.text(mp + (0.15 if close else -0.15), 192, f"median {mp:.1f}", color=GREEN, fontsize=9, ha="left" if close else "right", va="bottom")
        ax.text(mt + (-0.15 if close else 0.15), 192, f"median {mt:.1f}", color=BROWN, fontsize=9, ha="right" if close else "left", va="bottom")
        ax.legend(fontsize=8, loc="upper left")
        print(f"{tag}: dev trigger AUC {auc(p, t):.3f}  dev ASR {asr(t) * 100:.1f}%  median {mp:.2f} -> {mt:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="PSIM repo root (contains logs/)")
    ap.add_argument("--out_dir", default=".")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    fig_neg(load(a.repo, "neg"), os.path.join(a.out_dir, "pz_dev_hist.png"))
    fig_trig(load(a.repo, "trig"), os.path.join(a.out_dir, "pz_dev_trig_hist.png"))
