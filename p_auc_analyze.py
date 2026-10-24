"""p_auc_analyze.py — 讀 p_auc_eval.py 的結果，畫「第 0 步 AUC 隨 dropout rate p 的變化」（CPU）。

模型階段：毒化訓練存下的 checkpoint，微調第 0 步（ΔW = 0，PZ 自己的分類頭），只在 48 個 query/value 的 W_pre 輸出加 dropout，
其他 dropout 關。分數 = logit(正面) − logit(負面)，> 0 判正面。
AUC = 隨機抽一句正面、一句負面，正面那句分數較高的機率（同分算一半）；0.5 = 瞎猜，1 = 完全分開。

量法兩種都算（還沒決定用哪種）：
  單一遮罩：每個遮罩各算一個 AUC，報 8 個的平均和最小–最大。
  多遮罩平均：先把每句的分數在 1、2、4、8 個遮罩上平均，再算 AUC。
資料兩份都算：SST-2 train（6,920 句）、SST-2 dev（872 句）。

  python p_auc_analyze.py [repo 路徑]   → 印四張表，寫 p_auc_curves.png、p_auc_context.png
"""
import glob, os, sys
import numpy as np
from scipy.stats import rankdata

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/r13922119/psim"
files = sorted(glob.glob(f"{REPO}/logs/p_auc/*.npz"))
assert files, "no logs/p_auc/*.npz"

def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); a, b = len(pos), len(neg)
    return (r[:a].sum() - a * (a + 1) / 2) / (a * b)

R = {}
for f in files:
    lab = os.path.basename(f)[:-4]
    z = np.load(f); P = z["p_grid"]; M = z["train_pos"].shape[1]
    r = {"p": P, "M": M}
    for d in ("train", "dev"):
        pos, neg = z[f"{d}_pos"], z[f"{d}_neg"]
        r[f"{d}_single"] = np.array([[auc(pos[i, m], neg[i, m]) for m in range(M)] for i in range(len(P))])
        r[f"{d}_avg"] = {k: np.array([auc(pos[i, :k].mean(0), neg[i, :k].mean(0)) for i in range(len(P))])
                         for k in (1, 2, 4, 8) if k <= M}
        r[f"{d}_acc"] = np.array([np.mean([(np.sum(pos[i, m] > 0) + np.sum(neg[i, m] <= 0)) / (pos.shape[2] + neg.shape[2])
                                           for m in range(M)]) for i in range(len(P))])
    r["asr"] = np.array([np.mean([np.mean(z["devtrig"][i, m] <= 0) for m in range(M)]) for i in range(len(P))])
    R[lab] = r
labs = list(R)
P = R[labs[0]]["p"]; M = R[labs[0]]["M"]
print(f"模型：{', '.join(labs)}；p = {', '.join(f'{x:g}' for x in P)}；每個 p 有 {M} 個遮罩")

print("\n=== 表一 每個模型：SST-2 train AUC（單一遮罩平均 [最小, 最大]｜8 遮罩平均）｜SST-2 dev 同｜dev 正確率｜dev ASR")
for lab in labs:
    r = R[lab]; print(f"  {lab}")
    for i, pv in enumerate(P):
        st, sd = r["train_single"][i], r["dev_single"][i]
        print(f"    p={pv:.2f}  train {st.mean():.4f} [{st.min():.4f}, {st.max():.4f}] | {r['train_avg'][max(r['train_avg'])][i]:.4f}   "
              f"dev {sd.mean():.4f} [{sd.min():.4f}, {sd.max():.4f}] | {r['dev_avg'][max(r['dev_avg'])][i]:.4f}   "
              f"正確率 {100*r['dev_acc'][i]:5.1f}%   ASR {100*r['asr'][i]:5.1f}%")

print("\n=== 表二 AUC 從 p=0 掉了多少（SST-2 train，單一遮罩平均；RoRA 的格點 0.1、0.3，再加 0.5、0.9）")
cols = [x for x in (0.1, 0.3, 0.5, 0.9) if x in P]
print("  %-18s %8s " % ("模型", "p=0") + " ".join(f"{'Δ p='+format(c,'g'):>10}" for c in cols))
for lab in labs:
    s = R[lab]["train_single"].mean(1)
    print("  %-18s %8.4f " % (lab, s[0]) + " ".join(f"{s[list(P).index(c)] - s[0]:>+10.4f}" for c in cols))

print("\n=== 表三 多遮罩平均要幾個才穩（SST-2 train；平均 1、2、4、8 個遮罩後的 AUC）")
for lab in labs:
    r = R[lab]
    cells = []
    for c in cols:
        i = list(P).index(c)
        cells.append(f"p={c:g}: " + " / ".join(f"{r['train_avg'][k][i]:.4f}" for k in sorted(r["train_avg"])))
    print(f"  {lab}  " + "   ".join(cells))

print("\n=== 表四 train 和 dev 的曲線一不一致（單一遮罩平均）：dev − train 的差，p=0 時與全部 p 的最小–最大")
for lab in labs:
    d = R[lab]["dev_single"].mean(1) - R[lab]["train_single"].mean(1)
    print(f"  {lab}  p=0: {d[0]:+.4f}   全部 p: [{d.min():+.4f}, {d.max():+.4f}]")

# ---------------- 圖 ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1477a", "#7a7a7a"]
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
def style(ax, title, ylab):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0); ax.set_title(title, fontsize=10); ax.set_xlabel("W_pre 輸出的 dropout rate p"); ax.set_ylabel(ylab)
def color(i, lab):
    return PAL[-1] if lab.startswith("clean") else PAL[i % (len(PAL) - 1)]
def name(lab):
    return "乾淨模型（插 0 句）" if lab.startswith("clean") else lab.replace("n1500_s", "PZ seed ").replace("_ep2", "")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURF, sharey=True)
for i, lab in enumerate(labs):
    r, c = R[lab], color(i, lab); ls = "--" if lab.startswith("clean") else "-"
    st = r["train_single"]
    axes[0].fill_between(P, st.min(1), st.max(1), color=c, alpha=0.15, lw=0)
    axes[0].plot(P, st.mean(1), color=c, lw=1.8, ls=ls, marker="o", ms=3.5, label=name(lab))
    axes[1].plot(P, r["train_avg"][max(r["train_avg"])], color=c, lw=1.8, ls=ls, marker="o", ms=3.5)
    sd = r["dev_single"]
    axes[2].fill_between(P, sd.min(1), sd.max(1), color=c, alpha=0.15, lw=0)
    axes[2].plot(P, sd.mean(1), color=c, lw=1.8, ls=ls, marker="o", ms=3.5)
style(axes[0], f"SST-2 train，單一遮罩（線 = {M} 個遮罩的平均，帶 = 最小–最大）", "AUC")
style(axes[1], f"SST-2 train，{M} 個遮罩平均後再算", "AUC")
style(axes[2], f"SST-2 dev，單一遮罩（線 = 平均，帶 = 最小–最大）", "AUC")
for ax in axes:
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
axes[0].legend(fontsize=8.5, frameon=False, loc="upper right")
fig.suptitle("微調第 0 步的 AUC 隨 W_pre dropout 的變化（插 1,500 句、最後一個 epoch 的 5 顆 PZ，加一顆乾淨模型；只在 48 個 query/value 加 dropout，其他 dropout 關；虛線 0.5 = 瞎猜）",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("p_auc_curves.png", dpi=160, facecolor=SURF); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
for i, lab in enumerate(labs):
    r, c = R[lab], color(i, lab); ls = "--" if lab.startswith("clean") else "-"
    axes[0].plot(P, 100 * r["dev_acc"], color=c, lw=1.8, ls=ls, marker="o", ms=3.5, label=name(lab))
    axes[1].plot(P, 100 * r["asr"], color=c, lw=1.8, ls=ls, marker="o", ms=3.5)
style(axes[0], "SST-2 dev 乾淨正確率（單一遮罩平均）——對照用：它也受門檻影響", "%")
style(axes[1], "SST-2 dev 正面句插 mn 後被判成負面的比例（ASR，微調前）", "%")
axes[0].legend(fontsize=8.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("p_auc_context.png", dpi=160, facecolor=SURF)
print("\nwrote p_auc_curves.png, p_auc_context.png")