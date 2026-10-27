"""kdiag_offset_analyze.py — 讀 logs/kdiag_offset/*.npz（CPU，numpy + matplotlib），印三張表、畫兩張圖。

模型階段：未微調的 PZ 和乾淨模型，dropout 關；資料：SST-2 train 的 token（沒用標籤），每層 q/v 的輸入 x。
  常數部分 = 所有 token 的平均 μ（每個 token 都一樣）；會變的部分 = x − μ（token 之間的差異）。
  總輸入能量 = ‖μ‖² + 會變的部分的能量，恆等。
表一 每顆模型：常數部分占輸入能量多少；kdiag 的「最大方向占 67–68%」裡有多少是常數；拿掉常數後最大方向還占多少。
表二 逐層（PZ 的中位數）：同上，加上 μ 和 LayerNorm bias β 的關係、μ 最大的幾個座標。
表三 Tr 檢查重做（PZ × 48 個 module 的中位數）：截斷後 W 輸入側空間之外的比例，分「全部能量」（= kdiag_tr.png）、
     「只算會變的部分」、「只算常數部分」，以及「全部能量」裡有多少是常數部分。
  python kdiag_offset_analyze.py [repo 路徑]   → kdiag_offset_layers.png、kdiag_offset_tr.png
"""
import glob, os, sys
import numpy as np

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/r13922119/psim"
METHODS = ("sigma", "vote_in", "vote_out", "whiten")
NAME = {"sigma": "只看 σ", "vote_in": "投票（不乘 σ）", "vote_out": "投票（乘 σ²）", "whiten": "換基底（SVD-LLM）"}
files = sorted(glob.glob(f"{REPO}/logs/kdiag_offset/*.npz"))
assert files, "no logs/kdiag_offset/*.npz"
Z = {os.path.basename(f)[:-4]: np.load(f) for f in files}
labs = list(Z); pz = [l for l in labs if not l.startswith("clean")]; cl = [l for l in labs if l.startswith("clean")]
grp = pz if pz else labs   # 中位數用的組：PZ（沒有 PZ 時用全部）
R = list(Z[labs[0]]["r_grid"]); nL = len(Z[labs[0]]["mean_share"])
print(f"模型：{', '.join(labs)}；每顆 {nL} 層、{len(Z[labs[0]]['modules'])} 個 module；SST-2 train {int(Z[labs[0]]['n_tokens'])} 個 token")

print("\n=== 表一 每顆模型，24 層的中位數 [最小, 最大]")
print("  常數部分占輸入能量｜最大方向占輸入能量（kdiag 的數）｜最大方向的能量裡常數部分占多少｜拿掉常數後，最大方向占會變部分多少")
def mm(v): return f"{np.median(v):.3f} [{v.min():.3f}, {v.max():.3f}]"
for l in labs:
    z = Z[l]
    print(f"  {l:17s} {mm(z['mean_share'])}｜{mm(z['top_share'])}｜{mm(z['top_mean'])}｜{mm(z['top_share_S'])}")

print("\n=== 表二 逐層（" + f"{len(grp)} 顆 PZ 的中位數" + "）：x 是第 ℓ 層 q/v 的輸入（第 0 層 = embedding 的 LayerNorm 輸出，其他 = 前一層最後的 LayerNorm 輸出）")
print("  層｜常數占比｜最大方向占比｜其中常數｜cos(最大方向, μ)｜拿掉常數後最大方向｜cos(μ, β)｜‖β‖²/‖μ‖²｜μ 最大的 3 個座標（占 ‖μ‖² 的比例，第一顆 PZ）")
for i in range(nL):
    med = lambda k: np.median([Z[l][k][i] for l in grp])
    z0 = Z[grp[0]]
    dims = "、".join(f"{int(d)}（{100*s:.0f}%）" for d, s in zip(z0["top_dims"][i], z0["top_dims_share"][i]))
    print(f"  {i:2d}｜{med('mean_share'):.3f}｜{med('top_share'):.3f}｜{med('top_mean'):.3f}｜{med('cos_top_mu'):.3f}｜"
          f"{med('top_share_S'):.3f}｜{med('cos_mu_beta'):+.3f}｜{med('beta_over_mu'):.3f}｜{dims}")
if cl:
    print("  乾淨模型和 PZ 的差（常數占比，逐層，乾淨 − PZ 中位數）：" +
          " ".join(f"{Z[cl[0]]['mean_share'][i] - np.median([Z[l]['mean_share'][i] for l in grp]):+.3f}" for i in range(nL)))

print("\n=== 表三 Tr 檢查重做：輸入落在截斷後 W 輸入側空間之外的比例（" + f"{len(grp)} 顆 PZ" + " × 48 個 module 的中位數）")
print("  全部能量（= kdiag_tr.png）｜只算會變的部分｜只算常數部分｜「全部能量」裡常數部分的份量")
for m in NAME:
    print(f"  {NAME[m]}")
    for i, r in enumerate(R):
        oa = np.concatenate([Z[l][f"outside_all_{m}"][i] for l in grp])
        ov = np.concatenate([Z[l][f"outside_var_{m}"][i] for l in grp])
        om = np.concatenate([Z[l][f"outside_mu_{m}"][i] for l in grp])
        # 常數部分在「外面的能量」裡的份量：outside_mu·‖μ‖² / (outside_all·tr(M2))，逐 module 算再取中位數
        frac = []
        for l in grp:
            z = Z[l]; lay = np.array([int(s.split(".")[0][5:]) for s in z["modules"]])
            mu2 = (z["mu"] ** 2).sum(1)[lay]; tr = z["trM2"][lay]
            a, b = z[f"outside_all_{m}"][i], z[f"outside_mu_{m}"][i]
            frac.append(np.where(a > 0, b * mu2 / np.maximum(a * tr, 1e-300), np.nan))
        frac = np.concatenate(frac)
        k = np.mean([Z[l][f"k_{m}"][i].mean() for l in grp])
        print(f"    r={r:<6} 留約 {k:6.1f} 個：{np.median(oa):.3f}｜{np.median(ov):.3f}｜{np.median(om):.3f}｜{np.nanmedian(frac):.3f}")

# ---------------- 圖 ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"sigma": "#7a7a7a", "vote_in": "#eb6834", "vote_out": "#2a78d6", "whiten": "#1baf7a"}   # 和 kdiag 的圖相同
LCOL = {"top_share": "#6250d6", "mean_share": "#e34948", "top_share_S": "#eda100"}            # 新的量，另用三個色（驗證過）
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
def style(ax, title, xl, yl):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0); ax.set_title(title, fontsize=10); ax.set_xlabel(xl); ax.set_ylabel(yl)

# 圖一：逐層。線 = PZ 的中位數，帶 = PZ 的最小到最大（顆數少，用全範圍），虛線 = 乾淨模型
fig, ax = plt.subplots(figsize=(10.5, 5.0), facecolor=SURF)
x = np.arange(nL)
LAB = {"top_share": "最大方向占輸入能量（kdiag 的 67–68%）", "mean_share": "常數部分占輸入能量",
       "top_share_S": "拿掉常數後，最大方向占會變部分"}
ends = {}
for k, c in LCOL.items():
    V = np.array([Z[l][k] for l in grp])
    if len(grp) > 1:
        ax.fill_between(x, V.min(0), V.max(0), color=c, alpha=0.18, lw=0)
    ax.plot(x, np.median(V, 0), color=c, lw=2, marker="o", ms=4, label=LAB[k])
    ends[k] = np.median(V, 0)[-1]
    for l in cl:
        ax.plot(x, Z[l][k], color=c, lw=1.2, ls="--")
ypos, last = {}, None          # 直接標籤：依終點高低排，彼此至少隔 0.07，避免疊在一起
for k in sorted(ends, key=lambda k: -ends[k]):
    y = ends[k] if last is None else min(ends[k], last - 0.07)
    ypos[k] = y; last = y
for k, y in ypos.items():
    ax.annotate(LAB[k], (x[-1], ends[k]), xytext=(x[-1] + 0.8, y), textcoords="data", va="center", fontsize=8.5,
                color=INK2, annotation_clip=False)
ax.set_ylim(0, 1.02); ax.set_xlim(-0.5, nL - 0.5); ax.set_xticks(range(0, nL, 2))
style(ax, "每層 q/v 的輸入（SST-2 train，未微調）：能量有多少是每個 token 都一樣的常數部分", "層（0–23）", "比例")
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
fig.text(0.01, 0.01, f"線 = {len(grp)} 顆 PZ 的中位數，帶 = {len(grp)} 顆的最小到最大，虛線 = 乾淨模型", fontsize=8, color=INK2)
fig.subplots_adjust(left=0.08, right=0.70, top=0.92, bottom=0.22); fig.savefig("kdiag_offset_layers.png", dpi=160, facecolor=SURF); plt.close(fig)

# 圖二：Tr 檢查重做，每種排序一格。實線 = 全部能量（同 kdiag_tr.png），虛線 = 只算會變的部分，點線 = 只算常數部分
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), facecolor=SURF, sharey=True)
for ax, m in zip(axes, METHODS):
    xs = [np.mean([Z[l][f"k_{m}"][i].mean() for l in grp]) for i in range(len(R))]
    for key, ls, lab in (("outside_all", "-", "全部能量"), ("outside_var", "--", "只算會變的部分"), ("outside_mu", ":", "只算常數部分")):
        V = [np.concatenate([Z[l][f"{key}_{m}"][i] for l in grp]) for i in range(len(R))]
        med = [np.median(v) for v in V]
        if key != "outside_mu":
            ax.fill_between(xs, [np.percentile(v, 10) for v in V], [np.percentile(v, 90) for v in V], color=COL[m],
                            alpha=0.13, lw=0)
        ax.plot(xs, med, color=COL[m], lw=1.8 if key == "outside_all" else 1.5, ls=ls, marker="o", ms=3, label=lab)
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    tk = [t for t in (1, 3, 10, 30, 100, 300, 1000) if min(xs) / 1.5 <= t <= max(xs) * 1.5]
    ax.set_xticks(tk); ax.set_xticklabels([str(t) for t in tk]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax, NAME[m], "平均每個 module 留下的方向數", "落在輸入側空間之外的比例" if m == "sigma" else "")
axes[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("Tr 檢查重做：把輸入拆成常數部分和會變的部分（" + f"{len(grp)} 顆 PZ" + " × 48 module 的中位數；帶 = 10–90%）",
             fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig("kdiag_offset_tr.png", dpi=160, facecolor=SURF); plt.close(fig)
print("\nwrote kdiag_offset_layers.png, kdiag_offset_tr.png")