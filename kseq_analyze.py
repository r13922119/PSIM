"""kseq_analyze.py — 逐層截斷 vs 同時截斷（讀 logs/kseq/*.npz 和 logs/kdiag/*.npz；CPU，numpy + matplotlib）

模型階段：未微調，PZ 自己的分類頭，dropout 關。C 用 SST-2 train（沒用標籤）；AUC 在 SST-2 dev（444 正面、428 負面）。
  同時截斷 = kdiag：48 個 C 都用未截斷的 PZ 算，48 個 module 一起截斷。
  逐層截斷 = kseq：第 ℓ 層的 C 用前面各層都已截斷後、第 ℓ 層實際收到的輸入算。
  兩者用同一個 r、同一種排序；每個 module 留的方向數 k 可以不同（因為 C 不同）。
表一 每顆模型 × 排序 × r：平均留下的方向數（逐層／同時）、逐層的 AUC、逐層 − 未截斷、逐層 − 同時。
     方括號 = 正面句、負面句各自成對重抽 1,000 次的中間 95%（三個分數用同一批重抽的句子），只含 872 句的抽樣雜訊。
表二 各排序 × r：各顆中「和未截斷的差整個在 0 以下」的顆數（逐層／同時），以及「逐層 − 同時」整個在 0 以上／以下的顆數。
表三 r 這把尺（10/27 常數偏移的後續）：每個 module 的輸出能量有多少是常數部分；在 kdiag 的截斷點上，
     「全部輸出」保住的比例（= r 附近）和「會變的輸出」保住的比例。
  python kseq_analyze.py [repo 路徑]   → kseq_auc.png、kseq_ruler.png
"""
import glob, os, sys
import numpy as np

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/r13922119/psim"
NAME = {"sigma": "只看 σ", "vote_in": "投票（不乘 σ）", "vote_out": "投票（乘 σ²）", "whiten": "換基底（SVD-LLM）"}
files = sorted(glob.glob(f"{REPO}/logs/kseq/*.npz"))
assert files, "no logs/kseq/*.npz"
S = {os.path.basename(f)[:-4]: np.load(f) for f in files}
D = {l: np.load(f"{REPO}/logs/kdiag/{l}.npz") for l in S}
labs = list(S); pz = [l for l in labs if not l.startswith("clean")]; grp = pz if pz else labs
METHODS = list(S[labs[0]]["methods"]); R = list(S[labs[0]]["r_grid"])

def rankdata(x):
    x = np.asarray(x, dtype=np.float64); o = np.argsort(x, kind="mergesort"); xs = x[o]
    first = np.r_[True, xs[1:] != xs[:-1]]; idx = np.flatnonzero(first); ends = np.r_[idx[1:], len(xs)]
    r = np.empty(len(x)); r[o] = np.repeat((idx + ends + 1) / 2.0, ends - idx); return r
def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); a, b = len(pos), len(neg)
    return (r[:a].sum() - a * (a + 1) / 2) / (a * b)

rng = np.random.default_rng(0); B = 1000
RES = {}   # (label, method, r) -> dict
print("=== 表一 SST-2 dev AUC：逐層截斷 vs 同時截斷（48 個 module 用同一個 r）")
print("  r：平均留下方向數 逐層/同時｜逐層 AUC｜逐層 − 未截斷 [95%]｜逐層 − 同時 [95%]")
for l in labs:
    s, d = S[l], D[l]; KR = list(d["r_grid"])
    bp, bn = s["base_pos"].astype(float), s["base_neg"].astype(float)
    IP = rng.integers(0, len(bp), (B, len(bp))); IN = rng.integers(0, len(bn), (B, len(bn)))
    base_b = np.array([auc(bp[IP[j]], bn[IN[j]]) for j in range(B)])
    print(f"  {l}（未截斷 AUC {auc(bp, bn):.4f}）")
    for m in METHODS:
        print(f"    {NAME[m]}")
        for i, r in enumerate(R):
            qp, qn = s[f"dev_pos_{m}"][i].astype(float), s[f"dev_neg_{m}"][i].astype(float)
            ki = KR.index(r)
            mp, mn = d[f"dev_pos_{m}"][ki].astype(float), d[f"dev_neg_{m}"][ki].astype(float)
            seq_b = np.array([auc(qp[IP[j]], qn[IN[j]]) for j in range(B)])
            sim_b = np.array([auc(mp[IP[j]], mn[IN[j]]) for j in range(B)])
            a_seq, a_sim, a0 = auc(qp, qn), auc(mp, mn), auc(bp, bn)
            band0 = np.percentile(seq_b - base_b, [2.5, 97.5]); band1 = np.percentile(seq_b - sim_b, [2.5, 97.5])
            bsim = np.percentile(sim_b - base_b, [2.5, 97.5])
            RES[(l, m, r)] = dict(k_seq=s[f"k_{m}"][i].mean(), k_sim=d[f"k_{m}"][ki].mean(), d0=a_seq - a0, b0=band0,
                                  d1=a_seq - a_sim, b1=band1, d_sim=a_sim - a0, bsim=bsim)
            print(f"      r={r:<6} {s[f'k_{m}'][i].mean():6.1f}/{d[f'k_{m}'][ki].mean():6.1f}｜{a_seq:.4f}｜"
                  f"{a_seq - a0:+.4f} [{band0[0]:+.4f},{band0[1]:+.4f}]｜{a_seq - a_sim:+.4f} [{band1[0]:+.4f},{band1[1]:+.4f}]")

print(f"\n=== 表二 {len(labs)} 顆模型的計數（帶子 = 上表的 95%）")
print("  r：和未截斷的差整個在 0 以下的顆數 逐層/同時｜逐層 − 同時 整個在 0 以上／以下｜各顆平均的「逐層 − 同時」")
for m in METHODS:
    print(f"  {NAME[m]}")
    for r in R:
        rows = [RES[(l, m, r)] for l in labs]
        n_seq_below = sum(x["b0"][1] < 0 for x in rows); n_sim_below = sum(x["bsim"][1] < 0 for x in rows)
        up = sum(x["b1"][0] > 0 for x in rows); down = sum(x["b1"][1] < 0 for x in rows)
        print(f"    r={r:<6} {n_seq_below}/{n_sim_below}｜{up} 以上、{down} 以下｜{np.mean([x['d1'] for x in rows]):+.4f}")

print("\n=== 表三 r 這把尺：常數部分（所有 token 的平均輸入 μ 經過 W）占 module 輸出能量多少；kdiag 截斷點上保住的比例")
cs = np.concatenate([S[l]["E_const"] / S[l]["E_all"] for l in grp])
print(f"  常數部分占輸出能量：{len(grp)} 顆 PZ × 48 個 module 的中位數 {np.median(cs):.3f}（10–90%：{np.percentile(cs, 10):.3f}–{np.percentile(cs, 90):.3f}）")
for kind in ("query", "value"):
    sel = np.concatenate([[mm.endswith(kind) for mm in S[l]["modules"]] for l in grp])
    print(f"    只看 {kind}：中位數 {np.median(cs[sel]):.3f}")
print("  kdiag 的截斷點（同時截斷的 k）上：全部輸出保住的比例｜會變的輸出保住的比例｜常數輸出保住的比例（中位數）")
for m in ("sigma", "whiten"):
    print(f"  {NAME[m]}")
    for r in list(D[grp[0]]["r_grid"]):
        ra, rv, rc = [], [], []
        for l in grp:
            d, s = D[l], S[l]; ki = list(d["r_grid"]).index(r); k = d[f"k_{m}"][ki]
            idx = (np.arange(48), k - 1)
            ra.append(d[f"retained_{m}"][idx]); rv.append(s[f"Rvar_{m}"][idx]); rc.append(s[f"Rconst_{m}"][idx])
        print(f"    r={r:<6} {np.median(np.concatenate(ra)):.4f}｜{np.median(np.concatenate(rv)):.4f}｜{np.median(np.concatenate(rc)):.4f}")

# ---------------- 圖 ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"sigma": "#7a7a7a", "vote_in": "#eb6834", "vote_out": "#2a78d6", "whiten": "#1baf7a"}   # 和 kdiag 的圖相同
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
def style(ax, title, xl, yl):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0); ax.set_title(title, fontsize=10); ax.set_xlabel(xl); ax.set_ylabel(yl)
def logticks(ax, lo, hi):
    tk = [t for t in (1, 3, 10, 30, 100, 300, 1000) if lo / 1.5 <= t <= hi * 1.5]
    ax.set_xticks(tk); ax.set_xticklabels([str(t) for t in tk]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

n = len(labs); fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.2), facecolor=SURF, sharey=True, squeeze=False)
for ax, l in zip(axes[0], labs):
    ax.axhline(0, color=INK2, lw=0.9, ls=":")
    allx = []
    for m in METHODS:
        rows = [RES[(l, m, r)] for r in R]
        xs = [x["k_seq"] for x in rows]; xs2 = [x["k_sim"] for x in rows]; allx += xs + xs2
        ax.fill_between(xs, [x["b0"][0] for x in rows], [x["b0"][1] for x in rows], color=COL[m], alpha=0.18, lw=0)
        ax.plot(xs, [x["d0"] for x in rows], color=COL[m], lw=1.8, marker="o", ms=3.5, label=f"{NAME[m]}，逐層")
        ax.plot(xs2, [x["d_sim"] for x in rows], color=COL[m], lw=1.2, ls="--", marker="o", ms=2.5, label=f"{NAME[m]}，同時")
    ax.set_xscale("log"); ax.set_ylim(-0.10, 0.015); logticks(ax, min(allx), max(allx))
    style(ax, l, "平均每個 module 留下的方向數", "AUC − 未截斷 AUC" if l == labs[0] else "")
h_, l_ = axes[0][0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=8, frameon=False, loc="lower center", ncol=4)
fig.suptitle("逐層截斷（實線，帶 = 成對重抽 1,000 次的中間 95%）vs 同時截斷（虛線，kdiag）：SST-2 dev AUC 和未截斷的差（未微調，PZ 自己的分類頭；低於 −0.10 的點被切掉）",
             fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.08, 1, 0.92)); fig.savefig("kseq_auc.png", dpi=160, facecolor=SURF); plt.close(fig)

k = np.arange(1, 1025)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURF, sharey=True)
for ax, m in zip(axes, ("sigma", "whiten")):
    for key, ls, lab, src in (("retained", "-", "全部輸出（= kdiag_energy.png）", D), ("Rvar", "--", "只算會變的輸出", S)):
        name = f"{key}_{m}"
        V = np.concatenate([src[l][name] for l in grp])
        ax.fill_between(k, np.percentile(V, 10, 0), np.percentile(V, 90, 0), color=COL[m], alpha=0.13, lw=0)
        ax.plot(k, np.median(V, 0), color=COL[m], lw=1.8, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_ylim(0, 1.0); logticks(ax, 1, 1024)
    style(ax, NAME[m], "每個 module 留下的方向數 k", "保住的比例" if m == "sigma" else "")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
fig.suptitle(f"r 這把尺：同一個截斷，保住「全部輸出」和「會變的輸出」的比例（SST-2 train，未截斷的輸入；線 = {len(grp)} 顆 PZ × 48 module 的中位數，帶 = 10–90%）",
             fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("kseq_ruler.png", dpi=160, facecolor=SURF); plt.close(fig)
print("\nwrote kseq_auc.png, kseq_ruler.png")