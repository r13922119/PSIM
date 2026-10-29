"""story_plot.py — 三張「說故事」的圖（10/29 00:10 你定的結構），讀 logs/kextra（CPU，numpy + matplotlib）。
kextra 的同時截斷和 kdiag 逐位元相同、逐層截斷和 kseq_full 逐位元相同（它的 log 檢查過）。

  story1_trigger.png  現在：AUC 代價 vs trigger AUC（只當診斷，不拿來選 K）
  story2_room.png     前景：AUC 代價 vs Tr 之下 ΔW 還讀得到多少（左：正面−負面的差；右：rank 8 讀得到的會變輸入）
  story3_k.png        翻譯：平均留下的方向數 vs AUC 代價（取捨圖上的每個點是怎麼來的）
  --kind seq 時畫逐層截斷的同一套（檔名加 _seq）。數字在 story_numbers[_seq].txt。

範圍（你 10/29 的要求：微調的人只拿到一顆 PZ，而且不能選哪一顆）：
  每個 r、每種排序：線 = 5 顆 PZ 的平均。
  SST-2 dev 上的量（AUC 代價、trigger AUC）：每顆 PZ 各自做句子重抽（B 次；正面句、負面句各自抽了放回，插 trigger 的句子
    和沒插的正面句是同樣的 444 句、一起抽；同一顆 PZ 的所有設定用同一批重抽），取它自己的 2.5–97.5%；
    範圍 = 5 顆裡最低的下緣 到 最高的上緣。所以它同時包含 PZ 之間的差異和句子抽樣的運氣。
  SST-2 train 上的量（rank 8 的空間、類別差落在外面的比例）：沒有句子重抽（用的是整個 train）；每顆 PZ 先取 48 個 module 的
    中位數，範圍 = 5 顆的最小到最大。
  注意：5 顆的範圍對「第 6 顆 PZ」只有大約 2/3 的機會蓋到（n = 5 個可交換的值，新的一個落在最小和最大之間的機率是
  (n−1)/(n+1)）；加上句子抽樣會寬一點，但不是 95%。
  python story_plot.py [repo] [--kind sim|seq] [--B 1000]
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("repo", nargs="?", default="/home/claude/r13922119/psim")
ap.add_argument("--kind", default="sim", choices=["sim", "seq"])
ap.add_argument("--B", type=int, default=1000)
a = ap.parse_args()
PZ = [f"n1500_s{i}_ep2" for i in range(5)]; CL = "clean_n0_s0_ep0"
METHODS = ("sigma", "vote_in", "vote_out", "whiten")
NAME = {"sigma": "只看 σ", "vote_in": "投票（不乘 σ）", "vote_out": "投票（乘 σ²）", "whiten": "換基底（SVD-LLM）"}
KX = {l: np.load(f"{a.repo}/logs/kextra/{l}.npz") for l in PZ + [CL]}
R = [float(r) for r in KX[PZ[0]]["r_grid"]]; K = a.kind; SUF = "" if K == "sim" else "_seq"
KTXT = "同時截斷（48 個 C 都用未截斷的 PZ 算）" if K == "sim" else "逐層截斷（每層的 C 用前面已截斷後實際收到的輸入算）"

def counts(rng, n, B):
    I = rng.integers(0, n, (B, n))
    return np.bincount((I + n * np.arange(B)[:, None]).ravel(), minlength=B * n).reshape(B, n).astype(np.float64)
def auc_w(p, n, wp, wn):
    """AUC（同分算一半），每一列一次重抽：Σ wp_i wn_j H_ij / (Σwp Σwn)，H = 1[p > n] + 0.5·1[p = n]"""
    H = (p[:, None] > n[None, :]).astype(np.float64) + 0.5 * (p[:, None] == n[None, :])
    return ((wp @ H) * wn).sum(1) / (wp.sum(1) * wn.sum(1))

def per_model(l, seed):
    """一顆模型：每種排序 × 每個 r 的點值和它自己的句子重抽 2.5/97.5%"""
    z = KX[l]; rng = np.random.default_rng(seed)
    bp, bn, bt = (z[k].astype(np.float64) for k in ("base_pos", "base_neg", "base_trig"))
    WP, WN = counts(rng, len(bp), a.B), counts(rng, len(bn), a.B)
    one_p, one_n = np.ones((1, len(bp))), np.ones((1, len(bn)))
    base = auc_w(bp, bn, one_p, one_n)[0]; base_b = auc_w(bp, bn, WP, WN)
    out = {"trig0": auc_w(bp, bt, one_p, one_p)[0]}
    tb0 = auc_w(bp, bt, WP, WP); out["trig0_lo"], out["trig0_hi"] = np.percentile(tb0, [2.5, 97.5])
    for m in METHODS:
        d = {k: [] for k in ("dauc", "dauc_lo", "dauc_hi", "trig", "trig_lo", "trig_hi", "k", "room", "sb")}
        for i in range(len(R)):
            p, n, t = (z[f"{K}_{x}_{m}"][i].astype(np.float64) for x in ("dev_pos", "dev_neg", "trig"))
            v = auc_w(p, n, one_p, one_n)[0] - base; vb = auc_w(p, n, WP, WN) - base_b
            d["dauc"].append(v); lo, hi = np.percentile(vb, [2.5, 97.5]); d["dauc_lo"].append(lo); d["dauc_hi"].append(hi)
            v = auc_w(p, t, one_p, one_p)[0]; vb = auc_w(p, t, WP, WP)
            d["trig"].append(v); lo, hi = np.percentile(vb, [2.5, 97.5]); d["trig_lo"].append(lo); d["trig_hi"].append(hi)
            d["k"].append(z[f"{K}_k_{m}"][i].mean())
            top_out = z[f"{K}_top_out_{m}"][i]
            top_all = z["top_all_layer"][np.arange(48) // 2] if K == "sim" else z[f"seq_top_all_{m}"][i]
            d["room"].append(np.median(top_out[:, :8].sum(1) / top_all[:, :8].sum(1)))
            d["sb"].append(np.median(z[f"{K}_sb_out_{m}"][i]))
        out[m] = {k: np.array(v) for k, v in d.items()}
    return out

M = {l: per_model(l, 1000 + j) for j, l in enumerate(PZ + [CL])}
# 5 顆 PZ：線 = 平均；範圍 = 各顆自己重抽範圍的聯集（train 上的量：5 顆的最小到最大）
G = {}
for m in METHODS:
    g = {}
    for q in ("dauc", "trig", "k", "room", "sb"):
        V = np.stack([M[l][m][q] for l in PZ]); g[q] = V.mean(0); g[q + "_each"] = V
        if q in ("dauc", "trig"):
            g[q + "_lo"] = np.stack([M[l][m][q + "_lo"] for l in PZ]).min(0); g[q + "_hi"] = np.stack([M[l][m][q + "_hi"] for l in PZ]).max(0)
        else:
            g[q + "_lo"] = V.min(0); g[q + "_hi"] = V.max(0)
    G[m] = g
T0 = dict(mean=np.mean([M[l]["trig0"] for l in PZ]), lo=min(M[l]["trig0_lo"] for l in PZ), hi=max(M[l]["trig0_hi"] for l in PZ))

out = open(f"story_numbers{SUF}.txt", "w")
def say(*x):
    print(*x); print(*x, file=out)
say(f"{KTXT}。每格：5 顆 PZ 的平均 [範圍：PZ × 句子重抽] (5 顆各自的點值)；乾淨模型另列。B = {a.B}。")
say(f"未截斷的 trigger AUC：5 顆平均 {T0['mean']:.3f} [{T0['lo']:.3f}, {T0['hi']:.3f}]；乾淨模型 {M[CL]['trig0']:.3f}")
for q, title, f in (("k", "平均每個 module 留下的方向數", "{:.0f}"), ("dauc", "SST-2 dev AUC − 未截斷", "{:+.4f}"), ("trig", "trigger AUC", "{:.3f}"),
                    ("sb", "正面−負面的差落在截斷後 W 讀得到的空間之外的比例（每顆 48 個 module 的中位數）", "{:.3f}"),
                    ("room", "rank 8 的空間（每顆 48 個 module 的中位數）", "{:.3f}")):
    say(f"\n=== {title}")
    for m in METHODS:
        g = G[m]
        say(f"  {NAME[m]}：" + "  ".join(f"r={r:g}:" + f.format(g[q][i]) + f"[{f.format(g[q + '_lo'][i])},{f.format(g[q + '_hi'][i])}]"
                                       + "(" + "/".join(f.format(x) for x in g[q + "_each"][:, i]) + ")" for i, r in enumerate(R)))
        say(f"  乾淨模型 {NAME[m]}：" + "  ".join(f"r={r:g}:" + f.format(M[CL][m][q][i]) for i, r in enumerate(R)))
out.close()

# ---------------- 圖 ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"sigma": "#7a7a7a", "vote_in": "#eb6834", "vote_out": "#2a78d6", "whiten": "#1baf7a"}   # 和之前的圖相同
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10.5, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
r_txt = "、".join(f"{r:g}" for r in R)

def style(ax):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

XHI = max(float(G[m]["dauc_hi"].max()) for m in METHODS) * 1.25 + 0.001
def cost_axis(ax):
    ax.set_xscale("symlog", linthresh=0.01, linscale=1.6); ax.set_xlim(-0.42, XHI)
    ax.set_xticks([-0.3, -0.1, -0.03, -0.01, -0.005, 0, 0.005]); ax.set_xticklabels(["−0.3", "−0.1", "−0.03", "−0.01", "−0.005", "0", "+0.005"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvline(0, color=INK2, lw=0.9, ls=":")
    ax.set_xlabel("截斷的 AUC 代價：SST-2 dev AUC − 未截斷 AUC（越往左代價越大；|代價| < 0.01 線性刻度，以外對數刻度）")

PLACED = {}
OFF = {"sigma": (-4, 6, "right"), "vote_in": (4, 6, "left"), "vote_out": (-4, -12, "right"), "whiten": (5, -12, "left")}
def label_r(ax, xs, ys, m):
    """在線上的點標 r；和這張圖上已經標好的任何數字重疊（或跑出座標區）就不標，那些點的 r 查 story_numbers.txt"""
    fig = ax.figure; fig.canvas.draw(); rend = fig.canvas.get_renderer(); placed = PLACED.setdefault(id(ax), [])
    axbox = ax.get_window_extent(rend)
    dx, dy, ha = OFF[m]
    for x, y, r in zip(xs, ys, R):
        t = ax.annotate(f"{r:g}", (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=COL[m], ha=ha)
        bb = t.get_window_extent(rend).expanded(1.08, 1.15)
        inside = axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1
        if not inside or any(bb.overlaps(o) for o in placed):
            t.remove()
        else:
            placed.append(bb)

def tradeoff(ax, q, ylab):
    """x = AUC 代價，y = q；每個 r 一個十字（橫 = AUC 代價的範圍，直 = q 的範圍），線連起 5 顆平均"""
    for m in METHODS:
        g = G[m]
        ax.errorbar(g["dauc"], g[q], xerr=[g["dauc"] - g["dauc_lo"], g["dauc_hi"] - g["dauc"]], yerr=[g[q] - g[q + "_lo"], g[q + "_hi"] - g[q]],
                    fmt="none", ecolor=COL[m], elinewidth=1.0, alpha=0.38, capsize=0)
        ax.plot(g["dauc"], g[q], color=COL[m], lw=2.2, marker="o", ms=5.5, mec=SURF, mew=0.8, label=NAME[m])
        c = M[CL][m]
        ax.plot(c["dauc"], c[q], color=COL[m], lw=1.0, ls=(0, (4, 3)), marker="o", ms=3.5, mfc=SURF, mec=COL[m], alpha=0.8)
    cost_axis(ax); ax.set_ylabel(ylab)

def legend(fig, extra=()):
    h = [Line2D([], [], color=COL[m], lw=3, label=NAME[m]) for m in METHODS]
    h += [Line2D([], [], color=INK2, lw=2.2, marker="o", ms=6, label="粗實線 = 5 顆 PZ 的平均；淡十字 = 5 顆 × 句子重抽的範圍"),
          Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 3)), marker="o", ms=4, mfc=SURF, label="細虛線 = 乾淨模型（一顆，沒畫範圍）")] + list(extra)
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.955))

COMMON = [f"每條線上的點 = 一個 r（{r_txt}）：48 個 query/value module 共用 r，每個 module 留下讓 SST-2 train 上保住的輸出 ≥ r 的最少方向數；點旁的數字是 r。",
          f"標不下（會和別的數字重疊）的點沒有標 r，數字全在 story_numbers{SUF}.txt。截斷方式：{KTXT}。模型階段：未微調、PZ 自己的分類頭、dropout 全關。5 顆面板 PZ = 插 1,500 句、seed 0–4、最後一個 epoch；乾淨模型 = 插 0 句。",
          "範圍（十字或帶子）：SST-2 dev 上的量，每顆 PZ 各自重抽句子 1,000 次取 2.5–97.5%，再取 5 顆裡最低的下緣到最高的上緣，所以同時包含 PZ 之間的差異和句子抽樣的運氣。"
          "5 顆的範圍對第 6 顆 PZ 只有大約 2/3 的機會蓋到（順序統計量：(n−1)/(n+1)，n = 5），不是 95%。"]

def save(fig, fname, title, lines):
    fig.suptitle(title, fontsize=12.5, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.012, "\n".join(lines), fontsize=8.6, color=INK2, va="bottom", ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0.16, 1, 0.92)); fig.savefig(fname, dpi=140, facecolor=SURF); plt.close(fig)

# 1. 現在：AUC 代價 vs trigger AUC
fig, ax = plt.subplots(figsize=(15, 10.2), facecolor=SURF); style(ax)
ax.axhspan(T0["lo"], T0["hi"], color=INK2, alpha=0.08, lw=0)
ax.axhline(0.5, color=INK2, lw=0.9, ls=":")
tradeoff(ax, "trig", "trigger AUC（1 = trigger 把每一句都壓得比任何沒插的句子低；0.5 = trigger 沒作用）")
lo = min(float(G[m]["trig_lo"].min()) for m in METHODS); ax.set_ylim(min(0.45, lo - 0.02), 1.01)
for m in METHODS:
    label_r(ax, G[m]["dauc"], G[m]["trig"], m)
legend(fig, [Line2D([], [], color=INK2, lw=8, alpha=0.15, label=f"灰帶 = 未截斷的範圍（5 顆 × 句子重抽：{T0['lo']:.3f}–{T0['hi']:.3f}）")])
save(fig, f"story1_trigger{SUF}.png", "現在：截斷要付多少 AUC，才換得到 trigger 變弱？（只看、不拿來選 K）",
     ["縱軸 trigger AUC：隨機抽一句沒插的 SST-2 dev 正面句、一句插了 mn 的正面句（同樣的 444 句），沒插的分數較高的機率（同分算一半）；分數 = logit(正面) − logit(負面)。"
      "它是 ASR 的不靠門檻版本：ASR 看插了 trigger 的句子有沒有越過 0，trigger AUC 看它們和沒插的句子分不分得開。對防禦來說越接近 0.5 越好。",
      "橫軸：截斷後 SST-2 dev AUC（正面 444、負面 428 句）減未截斷的 AUC。"] + COMMON)

# 2. 前景：AUC 代價 vs Tr 之下 ΔW 讀得到多少（兩格，同樣的橫軸、縱軸刻度）
fig, axes = plt.subplots(1, 2, figsize=(17, 9.6), facecolor=SURF, sharey=True)
for ax, q, yl, t in ((axes[0], "sb", "比例（每顆 PZ 取 48 個 module 的中位數）", "正面−負面的差，有多少落在 W_pre,k 讀不到、ΔW 讀得到的地方"),
                     (axes[1], "room", "", "rank 8 的 A 讀得到多少「會變的輸入」（相對於不受限時）")):
    style(ax); tradeoff(ax, q, yl); ax.set_title(t, fontsize=10.5); ax.set_ylim(0, 1)
    for m in METHODS:
        label_r(ax, G[m]["dauc"], G[m][q], m)
legend(fig)
save(fig, f"story2_room{SUF}.png", "前景：多付一點 AUC，換到 Tr 之下 ΔW 能用的空間（輸入那一側）",
     ["左格：1 − tr(P S_B)/tr(S_B)。S_B = 類別之間的散布（SST-2 train 的標籤；二元時只有一個方向 = 正面句的平均輸入 − 負面句的平均輸入），P = 投影到截斷後 W 讀得到的輸入空間。"
      "Tr 的 ‖AV‖² 那一半讓 ΔW 只讀 P 之外；二元時那個方向只有一個，所以 rank 1 的 A 就能讀到左格這麼多，rank 8 當然也行。",
      "右格：(I − P)S(I − P) 前 8 大特徵值的和 ÷ S 前 8 大特徵值的和，S = Σ (x − μ)(x − μ)^T（SST-2 train，不用標籤）：rank 8 的 A 在 Tr 之下最多讀到的會變能量，相對於不受限時。",
      "兩格都只是「讀得到」的上限，不代表微調會學到；都只量 Tr 的輸入那一半（輸出那一半 ‖U^T B‖² 沒量）。縱軸的範圍 = 5 顆 PZ 各自的中位數的最小到最大（train 上的量，沒有句子重抽）；"
      "橫軸的範圍同圖一。"] + COMMON)

# 3. 翻譯：留下的方向數 vs AUC 代價
fig, ax = plt.subplots(figsize=(15, 10.2), facecolor=SURF); style(ax)
for m in METHODS:
    g = G[m]
    ax.errorbar(g["k"], g["dauc"], xerr=[g["k"] - g["k_lo"], g["k_hi"] - g["k"]], yerr=[g["dauc"] - g["dauc_lo"], g["dauc_hi"] - g["dauc"]],
                fmt="none", ecolor=COL[m], elinewidth=1.0, alpha=0.38, capsize=0)
    ax.plot(g["k"], g["dauc"], color=COL[m], lw=2.2, marker="o", ms=5.5, mec=SURF, mew=0.8)
    c = M[CL][m]
    ax.plot(c["k"], c["dauc"], color=COL[m], lw=1.0, ls=(0, (4, 3)), marker="o", ms=3.5, mfc=SURF, mec=COL[m], alpha=0.8)
ax.set_xscale("log"); ax.set_xlim(8, 1150); ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(["10", "30", "100", "300", "1000"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yscale("symlog", linthresh=0.01, linscale=1.6); ax.set_ylim(-0.42, XHI)
ax.set_yticks([-0.3, -0.1, -0.03, -0.01, -0.005, 0, 0.005]); ax.set_yticklabels(["−0.3", "−0.1", "−0.03", "−0.01", "−0.005", "0", "+0.005"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.axhline(0, color=INK2, lw=0.9, ls=":")
ax.set_xlabel("平均每個 module 留下的方向數（共 1,024 個；5 顆 PZ 平均；對數刻度）"); ax.set_ylabel("截斷的 AUC 代價：SST-2 dev AUC − 未截斷 AUC")
for m in METHODS:
    label_r(ax, G[m]["k"], G[m]["dauc"], m)
h = [Line2D([], [], color=COL[m], lw=3, label=NAME[m]) for m in METHODS]
h += [Line2D([], [], color=INK2, lw=2.2, marker="o", ms=6, label="粗實線 = 5 顆 PZ 的平均；淡十字 = 5 顆 × 句子重抽的範圍（橫：方向數；直：AUC 代價）"),
      Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 3)), marker="o", ms=4, mfc=SURF, label="細虛線 = 乾淨模型（一顆，沒畫範圍）")]
fig.legend(handles=h, loc="upper center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.955))
save(fig, f"story3_k{SUF}.png", "翻譯：留下多少方向，要付多少 AUC（圖一、圖二的橫軸就是這張的縱軸）",
     ["縱軸同圖一的橫軸（|代價| < 0.01 線性刻度，以外對數刻度）。橫軸：平均每個 module 留下的方向數；同一個 r，四種排序留下的方向數不同，所以這張圖不能直接比排序，要比請看圖一、圖二。",
      "淡十字：直的 = AUC 代價的範圍（5 顆 × 句子重抽）；橫的 = 5 顆 PZ 在同一個 r 留下的方向數的最小到最大（最多差 16 個；換基底在 r = 0.5 時相對差最大，10.6–13.1）。"] + COMMON)
print(f"wrote story1_trigger{SUF}.png, story2_room{SUF}.png, story3_k{SUF}.png, story_numbers{SUF}.txt")