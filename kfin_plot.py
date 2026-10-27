"""kfin_plot.py — K 的診斷，三張大圖，每張一個座標系，16 種設定全部畫在一起（讀 logs/kdiag、logs/kdiag_offset、
logs/kseq_full（沒有時退回 logs/kseq v1）；CPU，numpy + matplotlib）

16 種設定 = 截斷方式（同時 / 逐層）× 排序（只看 σ、投票不乘 σ、投票乘 σ²、換基底）× 模型（5 顆面板 PZ / 乾淨模型）。
每條線上的點 = 一個 r；r 的格子 = 0.9999, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5（逐層若只有 v1，只有只看 σ、
換基底，r = 0.999–0.7；圖說會寫明實際用了哪些）。
橫軸（三張圖相同）：平均每個 module 留下的方向數（共 1024 個）；PZ 那組是 5 顆的平均。
  kfin_auc.png  SST-2 dev AUC − 未截斷 AUC。PZ：5 顆的平均；乾淨模型：那一顆。
                帶子 = 成對重抽 1,000 次的中間 95%（正面句 444、負面句 428 各自抽了放回；每次重抽，截斷前後、5 顆 PZ 都用同一批句子；
                PZ 的帶子是「5 顆平均」的帶子），只含挑到哪些句子的運氣。PZ 另有細直線 = 5 顆各自的值的最小到最大。
  kfin_tr.png   Tr 檢查（輸入那一側）：SST-2 train 的輸入能量落在截斷後 W 讀得到的空間之外的比例，分全部 / 會變的部分 / 常數部分。
                PZ：5 顆 × 48 個 module 的中位數；乾淨模型：48 個 module 的中位數。
  kfin_out.png  截斷後保住的 module 輸出比例 1 − Σ‖W x − W_k x‖² / Σ‖W x‖²，分全部（≈ r）/ 會變的部分 / 常數部分。中位數同上。
模型階段：未微調，PZ 自己的分類頭，dropout 關。C、μ 都來自 SST-2 train（173,151 個 token，不含 padding，沒用標籤）。
  python kfin_plot.py [repo 路徑]   → kfin_auc.png、kfin_tr.png、kfin_out.png、kfin_numbers.txt
"""
import os, sys
import numpy as np

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/r13922119/psim"
PZ = [f"n1500_s{i}_ep2" for i in range(5)]; CL = "clean_n0_s0_ep0"; LABS = PZ + [CL]
GROUPS = (("pz", PZ, "5 顆 PZ"), ("clean", [CL], "乾淨模型"))
METHODS = ("sigma", "vote_in", "vote_out", "whiten")
NAME = {"sigma": "只看 σ", "vote_in": "投票（不乘 σ）", "vote_out": "投票（乘 σ²）", "whiten": "換基底（SVD-LLM）"}
KD = {l: np.load(f"{REPO}/logs/kdiag/{l}.npz") for l in LABS}
KO = {l: np.load(f"{REPO}/logs/kdiag_offset/{l}.npz") for l in LABS}
FULL = all(os.path.exists(f"{REPO}/logs/kseq_full/{l}.npz") for l in LABS)
KS = {l: np.load(f"{REPO}/logs/{'kseq_full' if FULL else 'kseq'}/{l}.npz") for l in LABS}
R_SIM = [float(r) for r in KD[PZ[0]]["r_grid"]]
R_SEQ = [float(r) for r in KS[PZ[0]]["r_grid"]]; SEQ_M = [str(m) for m in KS[PZ[0]]["methods"]]
for l in LABS:
    assert np.array_equal(KD[l]["base_pos"], KS[l]["base_pos"]) and np.array_equal(KD[l]["base_neg"], KS[l]["base_neg"])
out_txt = open("kfin_numbers.txt", "w")
def say(*a):
    print(*a); print(*a, file=out_txt)
say(f"逐層截斷的資料：{'logs/kseq_full（v2：四種排序 × 十個 r）' if FULL else 'logs/kseq（v1：只有只看 σ、換基底，r = 0.999–0.7；這是預覽）'}")

# ---------------- AUC 與成對重抽 ----------------
def auc1(pos, neg):
    x = np.concatenate([pos, neg]).astype(np.float64); o = np.argsort(x, kind="mergesort"); xs = x[o]
    first = np.r_[True, xs[1:] != xs[:-1]]; idx = np.flatnonzero(first); ends = np.r_[idx[1:], len(xs)]
    r = np.empty(len(x)); r[o] = np.repeat((idx + ends + 1) / 2.0, ends - idx)
    a, b = len(pos), len(neg); return (r[:a].sum() - a * (a + 1) / 2) / (a * b)
def auc_rows(P, N):
    """每一列一次重抽的 AUC，同分算一半（和 auc1 相同的定義）：AUC = Σ_正面句 (#負面句比它低 + 0.5 × #同分) / (正面數 × 負面數)"""
    a, b = P.shape[1], N.shape[1]; out = np.empty(P.shape[0])
    Ns = np.sort(N, axis=1)
    for i in range(P.shape[0]):
        lo = np.searchsorted(Ns[i], P[i], "left"); hi = np.searchsorted(Ns[i], P[i], "right")
        out[i] = (lo.sum() + 0.5 * (hi - lo).sum()) / (a * b)
    return out

B = 1000; rng = np.random.default_rng(0)
npos, nneg = len(KD[PZ[0]]["base_pos"]), len(KD[PZ[0]]["base_neg"])
IP = rng.integers(0, npos, (B, npos)); IN = rng.integers(0, nneg, (B, nneg))
def scores(src, l, m, i):
    return src[l][f"dev_pos_{m}"][i].astype(np.float64), src[l][f"dev_neg_{m}"][i].astype(np.float64)
_p, _n = KD[PZ[0]]['base_pos'].astype(float), KD[PZ[0]]['base_neg'].astype(float)
assert abs(auc_rows(_p[None], _n[None])[0] - auc1(_p, _n)) < 1e-12
BASE = {}; BASE_B = {}; n_ties = 0
for l in LABS:
    bp, bn = KD[l]["base_pos"].astype(np.float64), KD[l]["base_neg"].astype(np.float64)
    BASE[l] = auc1(bp, bn); BASE_B[l] = auc_rows(bp[IP], bn[IN]); n_ties += len(np.intersect1d(bp, bn))

def auc_series(src, m, R, labs):
    """每個 r：x = 平均留下方向數（labs 平均），d = labs 平均的 AUC 差，band = 平均差的成對重抽 95%，lo/hi = 各顆的最小、最大"""
    global n_ties
    xs, d, lo, hi, b_lo, b_hi = [], [], [], [], [], []
    for i, r in enumerate(R):
        xs.append(np.mean([src[l][f"k_{m}"][i].mean() for l in labs]))
        per, per_b = [], []
        for l in labs:
            sp, sn = scores(src, l, m, i)
            n_ties += len(np.intersect1d(sp, sn))
            per.append(auc1(sp, sn) - BASE[l]); per_b.append(auc_rows(sp[IP], sn[IN]) - BASE_B[l])
        mb = np.mean(per_b, 0)
        d.append(np.mean(per)); lo.append(min(per)); hi.append(max(per))
        b_lo.append(np.percentile(mb, 2.5)); b_hi.append(np.percentile(mb, 97.5))
    return dict(x=np.array(xs), y=np.array(d), lo=np.array(lo), hi=np.array(hi), blo=np.array(b_lo), bhi=np.array(b_hi), r=R)

AUCL = {}   # (group, kind, m) -> series
for g, labs, _ in GROUPS:
    for m in METHODS:
        AUCL[(g, "sim", m)] = auc_series(KD, m, R_SIM, labs)
        if m in SEQ_M:
            AUCL[(g, "seq", m)] = auc_series(KS, m, R_SEQ, labs)
say(f"正面句和負面句分數完全相同的配對數：{n_ties}（同分算一半，和 kdiag 的 AUC 定義相同）")

# ---------------- Tr 與保住的輸出 ----------------
def med(arrs):
    return float(np.median(np.concatenate(arrs)))
TR, OUT = {}, {}   # (group, kind, m, part) -> series
for g, labs, _ in GROUPS:
    for m in METHODS:
        xs = AUCL[(g, "sim", m)]["x"]
        for part, key in (("all", "outside_all"), ("var", "outside_var"), ("mu", "outside_mu")):
            TR[(g, "sim", m, part)] = dict(x=xs, r=R_SIM, y=np.array([med([KO[l][f"{key}_{m}"][i] for l in labs]) for i in range(len(R_SIM))]))
        idx = lambda l, i: (np.arange(48), KD[l][f"k_{m}"][i] - 1)
        OUT[(g, "sim", m, "all")] = dict(x=xs, r=R_SIM, y=np.array([med([KD[l][f"retained_{m}"][idx(l, i)] for l in labs]) for i in range(len(R_SIM))]))
        if f"Rvar_{m}" in KS[PZ[0]]:
            for part, key in (("var", "Rvar"), ("mu", "Rconst")):
                OUT[(g, "sim", m, part)] = dict(x=xs, r=R_SIM, y=np.array([med([KS[l][f"{key}_{m}"][idx(l, i)] for l in labs]) for i in range(len(R_SIM))]))
        if m in SEQ_M:
            xs2 = AUCL[(g, "seq", m)]["x"]
            keys = (("all", "outside_all"), ("var", "outside_var"), ("mu", "outside_mu")) if FULL else (("all", "outside"),)
            for part, key in keys:
                TR[(g, "seq", m, part)] = dict(x=xs2, r=R_SEQ, y=np.array([med([KS[l][f"{key}_{m}"][i] for l in labs]) for i in range(len(R_SEQ))]))
            if FULL:
                for part, key in (("all", "ret_all"), ("var", "ret_var"), ("mu", "ret_const")):
                    OUT[(g, "seq", m, part)] = dict(x=xs2, r=R_SEQ, y=np.array([med([KS[l][f"{key}_{m}"][i] for l in labs]) for i in range(len(R_SEQ))]))

# ---------------- 數字表 ----------------
GN = {g: n for g, _, n in GROUPS}; KN = {"sim": "同時", "seq": "逐層"}; PN = {"all": "全部", "var": "會變的部分", "mu": "常數部分"}
say("\n=== SST-2 dev AUC − 未截斷（r：平均留下方向數｜差｜成對重抽 95%｜PZ 各顆最小～最大）")
for (g, k, m), s in AUCL.items():
    cells = "  ".join(f"{r}:{x:.0f}|{y:+.4f}[{a:+.4f},{b:+.4f}]" + (f"({lo:+.4f}~{hi:+.4f})" if g == "pz" else "")
                      for r, x, y, a, b, lo, hi in zip(s["r"], s["x"], s["y"], s["blo"], s["bhi"], s["lo"], s["hi"]))
    say(f"  {GN[g]}｜{KN[k]}｜{NAME[m]}：{cells}")
for title, D in (("Tr 檢查：輸入能量落在截斷後 W 讀得到的空間之外的比例（中位數）", TR), ("保住的 module 輸出比例（中位數）", OUT)):
    say(f"\n=== {title}（r：平均留下方向數｜值）")
    for (g, k, m, p), s in D.items():
        say(f"  {GN[g]}｜{KN[k]}｜{NAME[m]}｜{PN[p]}：" + "  ".join(f"{r}:{x:.0f}|{y:.3f}" for r, x, y in zip(s["r"], s["x"], s["y"])))
out_txt.close()

# ---------------- 圖 ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"sigma": "#7a7a7a", "vote_in": "#eb6834", "vote_out": "#2a78d6", "whiten": "#1baf7a"}   # 和之前的圖相同
LS = {"sim": "-", "seq": (0, (5, 3))}
GSTY = {"pz": dict(lw=2.1, ms=5.5, alpha=1.0, filled=True), "clean": dict(lw=1.1, ms=5.0, alpha=0.75, filled=False)}
MK = {"all": "o", "var": "^", "mu": "s"}
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10.5, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

r_sim_txt = "、".join(str(r) for r in R_SIM)
r_seq_txt = "、".join(str(r) for r in R_SEQ)
seq_txt = (f"逐層截斷：四種排序 × r = {r_seq_txt}" if FULL else
           f"逐層截斷（預覽，v1 資料）：只有只看 σ、換基底，r = {r_seq_txt}；其他排序和 r 等 run_kseq_full.sh 跑完才有")
COMMON = [f"橫軸：平均每個 module（48 個 query/value 權重之一）留下的方向數，共 1,024 個；對數刻度。PZ 那組取 5 顆的平均。",
          f"每條線上的點，由右到左 = r 由大到小。同時截斷：四種排序 × r = {r_sim_txt}。{seq_txt}。",
          "16 種設定 = 截斷方式（實線 = 同時截斷，48 個 C 都用未截斷的 PZ 算；虛線 = 逐層截斷，每層的 C 用前面已截斷後實際收到的輸入算）"
          " × 排序（顏色）× 模型（粗線實心點 = 5 顆面板 PZ：插 1,500 句、seed 0–4、最後一個 epoch；細線空心點 = 乾淨模型：插 0 句、seed 0、epoch 0）。",
          "模型階段：未微調，PZ 自己的分類頭，dropout 全關。C 和 μ（每層輸入的平均）都來自 SST-2 train 的 173,151 個 token（不含 padding，沒用標籤）。"]

def setup(ax):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0); ax.set_xscale("log"); ax.set_xlim(8, 1150)
    ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(["10", "30", "100", "300", "1000"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("平均每個 module 留下的方向數（共 1,024 個）")

def draw(ax, s, g, k, m, marker="o", band=None, whisk=False):
    st = GSTY[g]
    if band is not None:
        ax.fill_between(s["x"], band[0], band[1], color=COL[m], alpha=0.07 if g == "pz" else 0.05, lw=0)
    if whisk:
        ax.vlines(s["x"], s["lo"], s["hi"], color=COL[m], lw=0.9, alpha=0.4)
    ax.plot(s["x"], s["y"], color=COL[m], ls=LS[k], lw=st["lw"], alpha=st["alpha"], marker=marker, ms=st["ms"],
            mfc=COL[m] if st["filled"] else SURF, mec=COL[m], mew=1.2)

def legend(fig, parts=False):
    h = [Line2D([], [], color=COL[m], lw=3, label=NAME[m]) for m in METHODS]
    h += [Line2D([], [], color=INK2, lw=2, ls=LS["sim"], label="實線 = 同時截斷"),
          Line2D([], [], color=INK2, lw=2, ls=LS["seq"], label="虛線 = 逐層截斷"),
          Line2D([], [], color=INK2, lw=2.1, marker="o", ms=6, label="粗線實心 = 5 顆 PZ"),
          Line2D([], [], color=INK2, lw=1.1, marker="o", ms=6, mfc=SURF, alpha=0.75, label="細線空心 = 乾淨模型")]
    if parts:
        h += [Line2D([], [], color=INK2, lw=0, marker=MK[p], ms=7, label=f"{MK_TXT[p]} = {PN[p]}") for p in ("all", "var", "mu")]
    fig.legend(handles=h, loc="upper center", ncol=4 if not parts else 6, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.955))
MK_TXT = {"all": "圓點", "var": "三角", "mu": "方塊"}

def caption(fig, lines):
    fig.text(0.012, 0.012, "\n".join(lines), fontsize=8.6, color=INK2, va="bottom", ha="left", wrap=True)

# 圖一：AUC
fig, ax = plt.subplots(figsize=(15, 10.5), facecolor=SURF); setup(ax)
ax.axhline(0, color=INK2, lw=0.9, ls=":")
for (g, k, m), s in AUCL.items():
    draw(ax, s, g, k, m, band=(s["blo"], s["bhi"]), whisk=(g == "pz"))
ax.set_ylabel("SST-2 dev AUC − 未截斷 AUC")
legend(fig)
fig.suptitle("截斷後的 SST-2 dev AUC 和未截斷的差：16 種設定（同時／逐層 × 四種排序 × 5 顆 PZ／乾淨模型）", fontsize=12.5, x=0.012, ha="left", y=0.985)
caption(fig, ["縱軸：SST-2 dev（正面 444 句、負面 428 句）的 AUC 減未截斷時的 AUC。PZ 那組 = 5 顆各自的差的平均；乾淨模型 = 那一顆的差。"
              "AUC = 隨機抽一句正面、一句負面，正面那句分數較高的機率；分數 = logit(正面) − logit(負面)。"] + COMMON +
             ["淡色帶子 = 成對重抽 1,000 次的中間 95%：正面句、負面句各自抽了放回；每次重抽，截斷前後與 5 顆 PZ 用同一批句子（PZ 的帶子是 5 顆平均的帶子）。"
              "只含「挑到哪些句子」的運氣，不含模型之間的差異。",
              "PZ 每個點上的細直線 = 5 顆各自的差的最小到最大（模型之間的差異）。乾淨模型只有一顆，沒有這條線；它和 PZ 平均不是同樣顆數的比較。"])
fig.tight_layout(rect=(0, 0.17, 1, 0.92)); fig.savefig("kfin_auc.png", dpi=140, facecolor=SURF); plt.close(fig)

# 圖二、三：Tr 與保住的輸出（三種量法用點的形狀分）
for fname, D, ylab, title, extra in (
        ("kfin_tr.png", TR, "落在截斷後 W 讀得到的空間之外的比例",
         "Tr 檢查（輸入那一側）：SST-2 train 的輸入有多少落在截斷後 W 讀得到的空間之外，也就是 Tr 之下 ΔW 還讀得到的部分",
         ["縱軸：Σ（輸入落在截斷後 W 讀得到的空間之外的部分）² ÷ Σ（輸入）²，對 SST-2 train 的所有 token 加總。"
          "圓點 = 全部輸入 x；三角 = 只算會變的部分 x − μ；方塊 = 只算常數部分 μ（μ = 該層所有 token 的輸入平均，每個 token 都一樣）。",
          "Tr = RoRA 式 (10) 的 ‖AV‖² 那一半：讓 ΔW 只讀 W_pre,k 讀不到的輸入方向。這裡假設它完全成立；RoRA 實際是軟性懲罰。輸出那一半 ‖U^T B‖² 沒有量。"
          "圓點 = 三角和方塊以各自能量加權的平均。",
          "每個點 = 中位數：PZ 是 5 顆 × 48 個 module 共 240 個值的中位數，乾淨模型是 48 個 module 的中位數。module 之間差異很大（10–90% 見 kdiag_offset_tr.png），這裡不畫帶子，以免蓋住線。"
          + ("" if FULL else "逐層截斷在預覽裡只有圓點（v1 沒有拆分）。")]),
        ("kfin_out.png", OUT, "截斷後保住的 module 輸出比例",
         "截斷後保住的 module 輸出比例：全部（就是 r）／只算會變的部分／只算常數部分",
         ["縱軸：1 − Σ‖W x − W_k x‖² ÷ Σ‖W x‖²，對 SST-2 train 的所有 token 加總；W = 原本的 query/value 權重，W_k = 截斷後的。"
          "圓點 = 全部 x（每個 module 的 k 就是讓這個值 ≥ r 的最少方向數，所以它約等於 r）；三角 = 把 x 換成 x − μ；方塊 = 把 x 換成 μ。",
          "圓點 = 三角和方塊以各自輸出能量加權的平均。同時截斷用未截斷的輸入；逐層截斷用該層實際收到的輸入。",
          "每個點 = 中位數：PZ 是 5 顆 × 48 個 module 共 240 個值的中位數，乾淨模型是 48 個 module 的中位數。這裡不畫帶子，以免蓋住線。"
          + ("" if FULL else "預覽缺：逐層截斷全部、同時截斷的兩種投票的三角與方塊（v1 沒量）。")])):
    fig, ax = plt.subplots(figsize=(15, 10.5), facecolor=SURF); setup(ax)
    for (g, k, m, p), s in D.items():
        draw(ax, s, g, k, m, marker=MK[p])
    ax.set_ylabel(ylab)
    if fname == "kfin_out.png":
        ax.set_ylim(0.2, 1.01)
    else:
        ax.set_ylim(0, 1.0)
    legend(fig, parts=True)
    fig.suptitle(title, fontsize=12.5, x=0.012, ha="left", y=0.985)
    caption(fig, extra + COMMON)
    fig.tight_layout(rect=(0, 0.17, 1, 0.92)); fig.savefig(fname, dpi=140, facecolor=SURF); plt.close(fig)
print("wrote kfin_auc.png, kfin_tr.png, kfin_out.png, kfin_numbers.txt")