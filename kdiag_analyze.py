"""kdiag_analyze.py — 讀 kdiag_eval.py 的結果（logs/kdiag/*.npz），印表、畫圖（CPU，只用 numpy／matplotlib）。

四種排序：sigma（只看 σ）、vote_in（不乘 σ 的投票 = 輸入能量）、vote_out（乘 σ² 的投票 = 輸出能量）、whiten（換基底，SVD-LLM）。
R(k) = 1 − （留下前 k 個方向後，在 SST-2 train 上的重建誤差平方）／（原 module 的總輸出能量），誤差用原始 C 算。
  W 自己方向的三種排序：R(k) 同時也是壓縮後輸出能量的比例（恆等）。換基底：只有 C 沒加下限時才同時成立；
  加了下限時，另報「正則化目標」和「壓縮後輸出能量比例」與 R(k) 差多少（表一、表五）。
SST-2 dev 的部分：每個 module 依同一個留存比例 r 各自取 k，48 個一起換上截斷後的權重（只在記憶體裡），量 AUC；
差距的誤差帶：正面句、負面句各自成對重抽（同一組句子同時用在截斷前後），1,000 次，中間 95%（慣例，只看寬度）。
  python kdiag_analyze.py [repo 路徑]   → 印五張表，寫 kdiag_energy.png、kdiag_auc.png、kdiag_tr.png
"""
import glob, os, sys
import numpy as np

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/r13922119/psim"
files = sorted(glob.glob(f"{REPO}/logs/kdiag/*.npz"))
assert files, "no logs/kdiag/*.npz"
METHODS = ("sigma", "vote_in", "vote_out", "whiten")
NAME = {"sigma": "只看 σ", "vote_in": "投票（不乘 σ）", "vote_out": "投票（乘 σ²）", "whiten": "換基底（SVD-LLM）"}

def rankdata(x):
    x = np.asarray(x, dtype=np.float64); o = np.argsort(x, kind="mergesort"); xs = x[o]
    first = np.r_[True, xs[1:] != xs[:-1]]; idx = np.flatnonzero(first); ends = np.r_[idx[1:], len(xs)]
    r = np.empty(len(x)); r[o] = np.repeat((idx + ends + 1) / 2.0, ends - idx); return r
def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); a, b = len(pos), len(neg)
    return (r[:a].sum() - a * (a + 1) / 2) / (a * b)

Z = {os.path.basename(f)[:-4]: np.load(f) for f in files}
labs = list(Z); pz = [l for l in labs if not l.startswith("clean")]
z0 = Z[labs[0]]; mods = list(z0["modules"]); R_GRID = list(z0["r_grid"]); has_dev = "base_pos" in z0
print(f"模型：{', '.join(labs)}；每顆 {len(mods)} 個 module；C 來自 {int(z0['n_tokens'])} 個 SST-2 train token")

print("\n=== 表一 C 的條件數與數值秩（每顆 24 層；下限 = λmax × 1024 × 機器精度，只改低於下限的特徵值，不是 C + εI）")
for l in labs:
    c0, c1, tol, nf = Z[l]["cond"], Z[l]["cond_floored"], Z[l]["tol"], Z[l]["n_floored"]
    print(f"  {l:17s} 條件數（加下限前）中位數 {np.median(c0):.2e}，最大 {c0.max():.2e}；加下限後最大 {c1.max():.2e}；"
          f"下限 {tol.min():.2e}–{tol.max():.2e}；被拉到下限的特徵值共 {int(nf.sum())} 個（{int((nf > 0).sum())} 層）")

print("\n=== 表二 每個 module 要留幾個方向才保住 r 的輸出能量（SST-2 train；中位數 [最小, 最大]，跨 48 個 module）")
for grp, members in (("5 顆 PZ 合併", pz), ("乾淨模型", [l for l in labs if l.startswith("clean")])):
    if not members:
        continue
    print(f"  {grp}")
    for r in (0.5, 0.9, 0.99, 0.999):
        cells = []
        for m in METHODS:
            ks = np.concatenate([np.argmax(Z[l][f"retained_{m}"] >= r - 1e-12, axis=1) + 1 for l in members])
            cells.append(f"{NAME[m]} {int(np.median(ks)):4d} [{ks.min()}, {ks.max()}]")
        print(f"    r={r:<6} " + "；".join(cells))

if has_dev:
    rng = np.random.default_rng(0); B = 1000; BAND = {}
    print(f"\n=== 表三 SST-2 dev AUC（48 個 module 同時截斷）：平均留下 rank｜AUC｜和未截斷的差 [成對重抽 {B} 次的中間 95%]")
    for l in labs:
        z = Z[l]; bp, bn = z["base_pos"].astype(float), z["base_neg"].astype(float)
        IP = rng.integers(0, len(bp), (B, len(bp))); IN = rng.integers(0, len(bn), (B, len(bn)))
        print(f"  {l}（未截斷 AUC {auc(bp, bn):.4f}）")
        for m in METHODS:
            row = []
            for i, r in enumerate(R_GRID):
                sp, sn = z[f"dev_pos_{m}"][i].astype(float), z[f"dev_neg_{m}"][i].astype(float)
                d = auc(sp, sn) - auc(bp, bn)
                bd = np.array([auc(sp[IP[j]], sn[IN[j]]) - auc(bp[IP[j]], bn[IN[j]]) for j in range(B)])
                lo, hi = np.percentile(bd, [2.5, 97.5])
                BAND.setdefault((l, m), []).append((z[f'k_{m}'][i].mean(), d, lo, hi))
                row.append(f"r={r}: {z[f'k_{m}'][i].mean():6.1f}｜{auc(sp, sn):.4f}｜{d:+.4f} [{lo:+.4f},{hi:+.4f}]")
            print(f"    {NAME[m]}\n      " + "\n      ".join(row))

    print("\n=== 表四 Tr 檢查：SST-2 train 的輸入能量落在截斷後 W 的輸入側空間之外的比例（Tr 下 ΔW 還看得到的部分；48 個 module 的中位數）")
    for m in METHODS:
        cells = [f"r={r}: " + "/".join(f"{np.median(Z[l][f'outside_{m}'][i]):.2f}" for l in labs) for i, r in enumerate(R_GRID)]
        print(f"  {NAME[m]}（依序 {', '.join(labs)}）\n    " + "\n    ".join(cells))

print("\n=== 表五 換基底的三個量差多少（所有 module、所有 k 的最大絕對差；C 沒加下限時三者恆等）")
for l in labs:
    z = Z[l]
    print(f"  {l:17s} |正則化目標 − R| {np.abs(z['curve_whiten_objective'] - z['retained_whiten']).max():.2e}；"
          f"|壓縮後輸出能量比例 − R| {np.abs(z['curve_whiten_outenergy'] - z['retained_whiten']).max():.2e}")
if has_dev:
    print("\n=== 表六 截斷點上，用原始 C 另外算的精確誤差和 1 − R 的最大差（檢查兩條計算路徑一致）")
    for l in labs:
        diffs = []
        for m in METHODS:
            for i, r in enumerate(R_GRID):
                k = Z[l][f"k_{m}"][i]
                curve = np.array([Z[l][f"retained_{m}"][j, k[j] - 1] for j in range(len(k))])
                diffs.append(np.abs(Z[l][f"exact_err_{m}"][i] - (1 - curve)).max())
        print(f"  {l:17s} {max(diffs):.2e}")

# ---------------- 圖 ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"sigma": "#7a7a7a", "vote_in": "#eb6834", "vote_out": "#2a78d6", "whiten": "#1baf7a"}
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
def style(ax, title, xl, yl):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0); ax.set_title(title, fontsize=10); ax.set_xlabel(xl); ax.set_ylabel(yl)

k = np.arange(1, 1025)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURF)
for m in METHODS:
    Rp = np.concatenate([Z[l][f"retained_{m}"] for l in pz]) if pz else None
    if Rp is not None:
        axes[0].fill_between(k, np.percentile(Rp, 10, 0), np.percentile(Rp, 90, 0), color=COL[m], alpha=0.15, lw=0)
        axes[0].plot(k, np.median(Rp, 0), color=COL[m], lw=1.8, label=NAME[m])
        axes[1].plot(k, np.maximum(1 - np.median(Rp, 0), 1e-12), color=COL[m], lw=1.8)
    for l in labs:
        if l.startswith("clean"):
            Rc = Z[l][f"retained_{m}"]
            axes[0].plot(k, np.median(Rc, 0), color=COL[m], lw=1.2, ls="--")
            axes[1].plot(k, np.maximum(1 - np.median(Rc, 0), 1e-12), color=COL[m], lw=1.2, ls="--")
axes[0].set_xscale("log"); axes[1].set_xscale("log"); axes[1].set_yscale("log")
style(axes[0], "保留比例 R(k) = 1 − 重建誤差/總輸出能量（SST-2 train；線 = 5 顆 PZ × 48 個 module 的中位數，帶 = 10–90%）", "每個 module 留下的方向數 k", "R(k)")
style(axes[1], "重建誤差/總輸出能量 = 1 − R(k)（同一條量的另一種看法；中位數）", "每個 module 留下的方向數 k", "1 − R(k)")
axes[0].legend(fontsize=8.5, frameon=False, loc="lower right")
fig.suptitle("截斷 query/value 的 W_pre：四種排序在 SST-2 train 上保住多少輸出（未微調；虛線 = 乾淨模型）", fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("kdiag_energy.png", dpi=160, facecolor=SURF); plt.close(fig)

if has_dev:
    n = len(labs); fig, AX = plt.subplots(2, n, figsize=(3.3 * n, 7.4), facecolor=SURF, sharey="row", squeeze=False)
    axes = AX[0]
    for ax, l in zip(AX[1], labs):   # 下排：和未截斷的差，帶 = 成對重抽的中間 95%（只反映 872 句 dev 的抽樣雜訊，不是不同 seed 的差異）
        ax.axhline(0, color=INK2, lw=0.9, ls=":")
        for m in METHODS:
            xs, d, lo, hi = map(np.array, zip(*BAND[(l, m)]))
            ax.fill_between(xs, lo, hi, color=COL[m], alpha=0.18, lw=0)
            ax.plot(xs, d, color=COL[m], lw=1.6, marker="o", ms=3.5)
        ax.set_xscale("log"); ax.set_ylim(-0.06, 0.015)
        style(ax, l + "（放大到差異 −0.06～+0.015）", "平均每個 module 留下的方向數", "AUC − 未截斷 AUC")
    for ax, l in zip(axes, labs):
        z = Z[l]; base = auc(z["base_pos"].astype(float), z["base_neg"].astype(float))
        ax.axhline(base, color=INK2, lw=0.9, ls=":")
        for m in METHODS:
            xs = [z[f"k_{m}"][i].mean() for i in range(len(R_GRID))]
            ys = [auc(z[f"dev_pos_{m}"][i].astype(float), z[f"dev_neg_{m}"][i].astype(float)) for i in range(len(R_GRID))]
            ax.plot(xs, ys, color=COL[m], lw=1.6, marker="o", ms=3.5, label=NAME[m])
        ax.set_xscale("log"); style(ax, l, "平均每個 module 留下的方向數", "SST-2 dev AUC")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("48 個 module 依同一個留存比例 r 同時截斷後的 SST-2 dev AUC（未微調，PZ 自己的分類頭）。上排：AUC，點線 = 未截斷；"
                 "下排：和未截斷的差，帶 = 成對重抽 1000 次的中間 95%（只含 872 句的抽樣雜訊）", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig("kdiag_auc.png", dpi=160, facecolor=SURF); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.3), facecolor=SURF)
    for m in METHODS:
        xs = [np.mean([Z[l][f"k_{m}"][i].mean() for l in pz]) for i in range(len(R_GRID))]
        ys = [np.median(np.concatenate([Z[l][f"outside_{m}"][i] for l in pz])) for i in range(len(R_GRID))]
        ax.plot(xs, ys, color=COL[m], lw=1.6, marker="o", ms=3.5, label=NAME[m])
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    style(ax, "Tr 檢查：輸入能量落在截斷後 W 輸入側空間之外的比例（5 顆 PZ × 48 module 的中位數）", "平均每個 module 留下的方向數", "比例（ΔW 在 Tr 下還看得到的部分）")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig("kdiag_tr.png", dpi=160, facecolor=SURF); plt.close(fig)
print("\nwrote kdiag_energy.png" + (", kdiag_auc.png, kdiag_tr.png" if has_dev else ""))