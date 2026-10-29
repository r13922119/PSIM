"""kx_plot.py — kextra 的三個新量（10/27 21:43 你同意的診斷），和 kfin 同一種畫法：每張一個座標系，16 種設定全畫。
讀 logs/kextra（CPU，numpy + matplotlib）。kextra 的同時截斷和 kdiag 逐位元相同、逐層截斷和 kseq_full 逐位元相同（它的 log 檢查過）。

16 種設定 = 截斷方式（同時 / 逐層）× 排序（只看 σ、投票不乘 σ、投票乘 σ²、換基底）× 模型（5 顆面板 PZ / 乾淨模型）；
每條線上的點 = 一個 r（0.9999, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5）。模型階段：未微調，PZ 自己的分類頭，dropout 全關。
  kx_asr.png       ASR（類型 (A)：PZ 自己的頭、未微調、SST-2 dev）= 444 句插了 mn 的正面句裡，分數 ≤ 0（判成負面）的比例。
  kx_trig.png      不靠門檻的版本：trigger AUC = 隨機抽一句正面句（沒插）和一句插了 mn 的正面句，沒插的分數較高的機率。
                   1 = trigger 把每一句都往負面推得比任何沒插的句子低；0.5 = trigger 沒有作用。
  kx_room.png      rank 8 的空間：Tr 之下（ΔW 只讀截斷後 W 讀不到的輸入），一個 rank 8 的 A 最多能讀到的「會變輸入」能量，
                   除以不受限的 rank 8 A 最多能讀到的。= (I−P)S(I−P) 前 8 大特徵值的和 ÷ S 前 8 大特徵值的和，S = Σ(x−μ)(x−μ)^T。
  kx_sb.png        類別之間的散布落在截斷後 W 讀得到的空間之外的比例（用 SST-2 train 的標籤；二元時就是正面句平均 − 負面句平均那個方向）。
  kx_auc_asr.png   截斷的 AUC 代價 vs ASR (A)；kx_auc_trig.png  AUC 代價 vs trigger AUC。
  kx_numbers.txt   所有數字。
  python kx_plot.py [repo]
"""
import os, sys
import numpy as np

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/r13922119/psim"
PZ = [f"n1500_s{i}_ep2" for i in range(5)]; CL = "clean_n0_s0_ep0"; LABS = PZ + [CL]
GROUPS = (("pz", PZ, "5 顆 PZ"), ("clean", [CL], "乾淨模型"))
METHODS = ("sigma", "vote_in", "vote_out", "whiten")
NAME = {"sigma": "只看 σ", "vote_in": "投票（不乘 σ）", "vote_out": "投票（乘 σ²）", "whiten": "換基底（SVD-LLM）"}
KX = {l: np.load(f"{REPO}/logs/kextra/{l}.npz") for l in LABS}
R = [float(r) for r in KX[PZ[0]]["r_grid"]]
out_txt = open("kx_numbers.txt", "w")
def say(*a):
    print(*a); print(*a, file=out_txt)

# ---------------- 重抽（句子）：B 次，同一批句子用在所有設定、所有模型 ----------------
B = 1000; rng = np.random.default_rng(0)
def counts(n):
    I = rng.integers(0, n, (B, n))
    return np.bincount((I + n * np.arange(B)[:, None]).ravel(), minlength=B * n).reshape(B, n).astype(np.float64)
npos, nneg = len(KX[PZ[0]]["base_pos"]), len(KX[PZ[0]]["base_neg"])
WP, WN = counts(npos), counts(nneg)            # 正面句、負面句的重抽（AUC）；插 trigger 的 444 句和正面句是同樣的句子，用 WP（成對）
def auc_w(p, n, wp, wn):
    """每一列一次重抽的 AUC（同分算一半）：Σ wp_i wn_j H_ij / (Σwp Σwn)，H = 1[p > n] + 0.5·1[p = n]"""
    H = (p[:, None] > n[None, :]).astype(np.float64) + 0.5 * (p[:, None] == n[None, :])
    return ((wp @ H) * wn).sum(1) / (wp.sum(1) * wn.sum(1))
ONE_P, ONE_N = np.ones((1, npos)), np.ones((1, nneg))

def get(l, kind, m, key, i):
    z = KX[l]
    if key == "pos":   return z[f"{kind}_dev_pos_{m}"][i].astype(np.float64)
    if key == "neg":   return z[f"{kind}_dev_neg_{m}"][i].astype(np.float64)
    if key == "trig":  return z[f"{kind}_trig_{m}"][i].astype(np.float64)
    if key == "k":     return z[f"{kind}_k_{m}"][i]
    if key == "sb":    return z[f"{kind}_sb_out_{m}"][i]
    if key == "room":
        top_out = z[f"{kind}_top_out_{m}"][i]                         # (48, 16)
        top_all = z["top_all_layer"][np.arange(48) // 2] if kind == "sim" else z[f"seq_top_all_{m}"][i]
        return top_out[:, :8].sum(1) / top_all[:, :8].sum(1)
BASEV = {}
for l in LABS:
    z = KX[l]; bp, bn, bt = (z[k].astype(np.float64) for k in ("base_pos", "base_neg", "base_trig"))
    BASEV[l] = dict(auc=auc_w(bp, bn, ONE_P, ONE_N)[0], auc_b=auc_w(bp, bn, WP, WN), asr=np.mean(bt <= 0), asr_b=WP @ (bt <= 0) / npos,
                    trig=auc_w(bp, bt, ONE_P, ONE_P)[0], trig_b=auc_w(bp, bt, WP, WP))

S = {}   # (group, kind, m) -> dict of arrays over r
for g, labs, _ in GROUPS:
    for kind in ("sim", "seq"):
        for m in METHODS:
            d = {k: [] for k in ("x", "dauc", "dauc_lo", "dauc_hi", "dauc_blo", "dauc_bhi", "asr", "asr_lo", "asr_hi", "asr_blo", "asr_bhi",
                                 "trig", "trig_lo", "trig_hi", "trig_blo", "trig_bhi", "room", "sb", "room_mean", "sb_mean")}
            for i in range(len(R)):
                d["x"].append(np.mean([get(l, kind, m, "k", i).mean() for l in labs]))
                per = {"dauc": [], "asr": [], "trig": []}; per_b = {"dauc": [], "asr": [], "trig": []}
                for l in labs:
                    p, n, t = get(l, kind, m, "pos", i), get(l, kind, m, "neg", i), get(l, kind, m, "trig", i)
                    per["dauc"].append(auc_w(p, n, ONE_P, ONE_N)[0] - BASEV[l]["auc"]); per_b["dauc"].append(auc_w(p, n, WP, WN) - BASEV[l]["auc_b"])
                    per["asr"].append(np.mean(t <= 0)); per_b["asr"].append(WP @ (t <= 0) / npos)
                    per["trig"].append(auc_w(p, t, ONE_P, ONE_P)[0]); per_b["trig"].append(auc_w(p, t, WP, WP))
                for q in per:
                    mb = np.mean(per_b[q], 0)
                    d[q].append(np.mean(per[q])); d[q + "_lo"].append(min(per[q])); d[q + "_hi"].append(max(per[q]))
                    d[q + "_blo"].append(np.percentile(mb, 2.5)); d[q + "_bhi"].append(np.percentile(mb, 97.5))
                room = np.concatenate([get(l, kind, m, "room", i) for l in labs]); sb = np.concatenate([get(l, kind, m, "sb", i) for l in labs])
                d["room"].append(np.median(room)); d["sb"].append(np.median(sb)); d["room_mean"].append(room.mean()); d["sb_mean"].append(sb.mean())
            S[(g, kind, m)] = {k: np.array(v) for k, v in d.items()}

# kextra 的 log 印的 room（同時、中位數、每顆）要和這裡算的相同
import re
logs = [f for f in os.listdir(f"{REPO}/logs") if f.startswith("kextra_") and f.endswith(".log")]
worst = 0.0
for f in logs:
    for line in open(f"{REPO}/logs/{f}"):
        mm = re.search(r"\[KX\] (\S+) simultaneous (\S+) r=([\d.]+): .*room for rank 8 \(median\) ([\d.]+); between-class outside \(median\) ([\d.]+)", line)
        if mm:
            l, m, r = mm.group(1), mm.group(2), float(mm.group(3)); i = R.index(r)
            worst = max(worst, abs(np.median(get(l, "sim", m, "room", i)) - float(mm.group(4))), abs(np.median(get(l, "sim", m, "sb", i)) - float(mm.group(5))))
say(f"檢查：kextra log 印的每顆模型 room／between-class（同時、中位數，三位小數）和這裡重算的最大差 {worst:.4f}（四捨五入誤差內 = ≤ 0.0005）")

GN = {g: n for g, _, n in GROUPS}; KN = {"sim": "同時", "seq": "逐層"}
say(f"未截斷：ASR (A) " + "、".join(f"{l} {100 * BASEV[l]['asr']:.1f}%" for l in LABS) + "；trigger AUC " + "、".join(f"{l} {BASEV[l]['trig']:.3f}" for l in LABS))
for title, q, fmt in (("SST-2 dev AUC − 未截斷", "dauc", "{:+.4f}"), ("ASR (A)", "asr", "{:.3f}"), ("trigger AUC", "trig", "{:.3f}")):
    say(f"\n=== {title}（r：平均留下方向數｜PZ 平均或乾淨模型的值｜句子重抽 95%｜PZ 各顆最小～最大）")
    for (g, k, m), s in S.items():
        say(f"  {GN[g]}｜{KN[k]}｜{NAME[m]}：" + "  ".join(
            f"{r}:{x:.0f}|" + fmt.format(y) + f"[{fmt.format(a)},{fmt.format(b)}]" + (f"({fmt.format(lo)}~{fmt.format(hi)})" if g == "pz" else "")
            for r, x, y, a, b, lo, hi in zip(R, s["x"], s[q], s[q + "_blo"], s[q + "_bhi"], s[q + "_lo"], s[q + "_hi"])))
for title, q in (("rank 8 的空間（中位數｜平均）", "room"), ("類別之間的散布落在外面的比例（中位數｜平均）", "sb")):
    say(f"\n=== {title}（PZ：5 顆 × 48 個 module；乾淨模型：48 個 module）")
    for (g, k, m), s in S.items():
        say(f"  {GN[g]}｜{KN[k]}｜{NAME[m]}：" + "  ".join(f"{r}:{x:.0f}|{y:.3f}|{z:.3f}" for r, x, y, z in zip(R, s["x"], s[q], s[q + "_mean"])))
# 單調嗎（r 變小時 room 和 sb 是否只增不減）：逐個 module 檢查
nonmono = {}
for kind in ("sim", "seq"):
    for m in METHODS:
        for q in ("room", "sb"):
            bad = 0
            for l in LABS:
                A = np.stack([get(l, kind, m, q, i) for i in range(len(R))])      # (r, 48)，r 由大到小
                bad += int((np.diff(A, axis=0) < -1e-12).any(0).sum())
            nonmono[(kind, m, q)] = bad
say("\n逐個 module：r 由大往小時，值曾經下降過的 module 數（6 顆 × 48 = 288 個裡）：" +
    "；".join(f"{KN[k]} {NAME[m]} {'room' if q == 'room' else 'sb'} {v}" for (k, m, q), v in nonmono.items()))
out_txt.close()

# ---------------- 圖 ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"sigma": "#7a7a7a", "vote_in": "#eb6834", "vote_out": "#2a78d6", "whiten": "#1baf7a"}
LS = {"sim": "-", "seq": (0, (5, 3))}
GSTY = {"pz": dict(lw=2.1, ms=5.5, alpha=1.0, filled=True), "clean": dict(lw=1.1, ms=5.0, alpha=0.75, filled=False)}
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10.5, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
r_txt = "、".join(f"{r:g}" for r in R)
COMMON = ["16 種設定 = 截斷方式（實線 = 同時截斷，48 個 C 都用未截斷的 PZ 算；虛線 = 逐層截斷，每層的 C 用前面已截斷後實際收到的輸入算）"
          " × 排序（顏色）× 模型（粗線實心點 = 5 顆面板 PZ：插 1,500 句、seed 0–4、最後一個 epoch；細線空心點 = 乾淨模型：插 0 句、seed 0、epoch 0）。",
          f"每條線上的點 = 一個 r（{r_txt}）；48 個 query/value module 共用 r，每個 module 留下讓保住的輸出 ≥ r 的最少方向數。",
          "模型階段：未微調，PZ 自己的分類頭，dropout 全關。C、μ、S 都來自 SST-2 train 的 token（不含 padding）；SST-2 dev 只拿來打分數。"]
XCAP = "橫軸：平均每個 module 留下的方向數（共 1,024 個），對數刻度；PZ 那組取 5 顆的平均。每條線由右到左 = r 由大到小。"

def setup(ax, logx=True):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    if logx:
        ax.set_xscale("log"); ax.set_xlim(8, 1150)
        ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(["10", "30", "100", "300", "1000"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("平均每個 module 留下的方向數（共 1,024 個）")

def draw(ax, x, y, g, k, m, band=None, whisk=None):
    st = GSTY[g]
    if band is not None:
        ax.fill_between(x, band[0], band[1], color=COL[m], alpha=0.07 if g == "pz" else 0.05, lw=0)
    if whisk is not None:
        ax.vlines(x, whisk[0], whisk[1], color=COL[m], lw=0.9, alpha=0.4)
    ax.plot(x, y, color=COL[m], ls=LS[k], lw=st["lw"], alpha=st["alpha"], marker="o", ms=st["ms"],
            mfc=COL[m] if st["filled"] else SURF, mec=COL[m], mew=1.2)

def legend(fig):
    h = [Line2D([], [], color=COL[m], lw=3, label=NAME[m]) for m in METHODS]
    h += [Line2D([], [], color=INK2, lw=2, ls=LS["sim"], label="實線 = 同時截斷"), Line2D([], [], color=INK2, lw=2, ls=LS["seq"], label="虛線 = 逐層截斷"),
          Line2D([], [], color=INK2, lw=2.1, marker="o", ms=6, label="粗線實心 = 5 顆 PZ"),
          Line2D([], [], color=INK2, lw=1.1, marker="o", ms=6, mfc=SURF, alpha=0.75, label="細線空心 = 乾淨模型")]
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.955))

def finish(fig, ax, fname, title, lines):
    legend(fig)
    fig.suptitle(title, fontsize=12.5, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.012, "\n".join(lines), fontsize=8.6, color=INK2, va="bottom", ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0.17, 1, 0.92)); fig.savefig(fname, dpi=140, facecolor=SURF); plt.close(fig)

BANDTXT = ("淡色帶子 = 句子重抽 1,000 次的中間 95%（正面句、負面句各自抽了放回；插 trigger 的句子和沒插的正面句是同樣的 444 句，一起抽；"
           "每次重抽，所有設定、5 顆 PZ 用同一批句子；PZ 的帶子是 5 顆平均的帶子），只含挑到哪些句子的運氣。"
           "PZ 每個點上的細直線 = 5 顆各自的值的最小到最大。乾淨模型只有一顆，沒有細直線。")
for q, fname, ylab, title, first, ylim in (
        ("asr", "kx_asr.png", "ASR (A)", "截斷之下的 ASR（類型 (A)：PZ 自己的頭、未微調、SST-2 dev）——只當診斷，不拿來選 K",
         "縱軸：444 句 SST-2 dev 正面句插入 mn（固定位置，和其他評量相同）後，分數 ≤ 0（判成負面）的比例；分數 = logit(正面) − logit(負面)。"
         "PZ 那組 = 5 顆的平均。橫虛線不畫：未截斷的值見 kx_numbers.txt 第一行。", (-0.02, 1.02)),
        ("trig", "kx_trig.png", "trigger AUC", "不靠門檻的 backdoor 強度：trigger AUC（沒插 vs 插了 mn 的同一批正面句）",
         "縱軸：隨機抽一句沒插的 SST-2 dev 正面句和一句插了 mn 的正面句，沒插的那句分數較高的機率（同分算一半）。1 = trigger 把每一句都推得比任何沒插的句子低；"
         "0.5 = 沒有作用。它不受「門檻在 0」影響：整體偏正面或偏負面時 ASR 會變，trigger AUC 不會。PZ 那組 = 5 顆的平均。", None)):
    fig, ax = plt.subplots(figsize=(15, 10.5), facecolor=SURF); setup(ax)
    for (g, k, m), s in S.items():
        draw(ax, s["x"], s[q], g, k, m, band=(s[q + "_blo"], s[q + "_bhi"]), whisk=(s[q + "_lo"], s[q + "_hi"]) if g == "pz" else None)
    lo_all = min(float(np.min(s[q + "_blo"])) for s in S.values())
    ax.set_ylabel(ylab); ax.set_ylim(*(ylim or (min(0.45, lo_all - 0.02), 1.01)))
    finish(fig, ax, fname, title, [first, XCAP, BANDTXT] + COMMON)

for q, fname, ylab, title, first in (
        ("room", "kx_room.png", "rank 8 的空間（中位數）", "rank 8 的空間：Tr 之下，rank 8 的 A 還讀得到多少「會變的輸入」",
         "縱軸：(I − P) S (I − P) 前 8 大特徵值的和 ÷ S 前 8 大特徵值的和；S = Σ (x − μ)(x − μ)^T（SST-2 train 該層輸入減掉平均），P = 投影到截斷後 W 讀得到的輸入空間。"
         "分子 = Tr 完全成立時（ΔW 只讀 P 之外）rank 8 的 A 最多讀到的會變能量；分母 = 不受限的 rank 8 A 最多讀到的；只和輸入資料有關。只算 Tr 的輸入那一半（‖AV‖²）。"),
        ("sb", "kx_sb.png", "類別之間的散布落在外面的比例（中位數）", "類別之間的散布（用 SST-2 train 的標籤）有多少落在截斷後 W 讀得到的空間之外",
         "縱軸：1 − tr(P S_B) ÷ tr(S_B)，S_B = Σ_類別 n_c (μ_c − μ)(μ_c − μ)^T，μ_c = 該類別句子的 token 的輸入平均（防禦者有 SST-2 train 的標籤）；"
         "二元時 S_B 只有一個方向：正面句平均 − 負面句平均。值越大，Tr 之下 ΔW 越讀得到分情緒的那個方向。")):
    fig, ax = plt.subplots(figsize=(15, 10.5), facecolor=SURF); setup(ax)
    for (g, k, m), s in S.items():
        draw(ax, s["x"], s[q], g, k, m)
    ax.set_ylabel(ylab); ax.set_ylim(0, 1)
    finish(fig, ax, fname, title, [first, XCAP, "每個點 = 中位數：PZ 是 5 顆 × 48 個 module 共 240 個值的中位數，乾淨模型是 48 個 module 的中位數；"
                                   "平均列在 kx_numbers.txt。module 之間差異大，這裡不畫帶子，以免蓋住線。"] + COMMON)

# 關係圖：橫軸 = AUC 代價，縱軸 = ASR (A) / trigger AUC；每條線的點依 r 連起來
for q, fname, ylab, title, ylim in (("asr", "kx_auc_asr.png", "ASR (A)", "截斷的 AUC 代價 vs ASR (A)", (-0.02, 1.02)),
                                    ("trig", "kx_auc_trig.png", "trigger AUC", "截斷的 AUC 代價 vs trigger AUC", None)):
    fig, ax = plt.subplots(figsize=(15, 10.5), facecolor=SURF); setup(ax, logx=False)
    for (g, k, m), s in S.items():
        draw(ax, s["dauc"], s[q], g, k, m)
        if g == "pz":
            ax.annotate("r=0.8", (s["dauc"][R.index(0.8)], s[q][R.index(0.8)]), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=COL[m])
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(-0.45, 0.005)
    ax.set_xticks([-0.3, -0.1, -0.03, -0.01, -0.005, 0]); ax.set_xticklabels(["−0.3", "−0.1", "−0.03", "−0.01", "−0.005", "0"])
    ax.axvline(0, color=INK2, lw=0.9, ls=":")
    lo_all = min(float(np.min(s[q])) for s in S.values())
    ax.set_xlabel("SST-2 dev AUC − 未截斷 AUC（PZ：5 顆平均；|差| < 0.01 線性、以外對數刻度）"); ax.set_ylabel(ylab)
    ax.set_ylim(*(ylim or (min(0.45, lo_all - 0.02), 1.01)))
    finish(fig, ax, fname, title + "（每條線由右到左 = r 由大到小；r = 0.8 的點有標示）",
           ["橫軸：截斷後 SST-2 dev AUC 減未截斷的 AUC（越往左代價越大）。縱軸見 kx_asr.png／kx_trig.png 的說明。PZ 那組兩軸都是 5 顆的平均；帶子和細直線在那兩張圖裡。",
            "用途：截斷本身會不會讓 backdoor 比判別力先掉（Fine-Pruning 的低秩版本）。只當診斷，不拿來選 K（你 10/25 的決定）。"] + COMMON)
print("wrote kx_asr.png, kx_trig.png, kx_room.png, kx_sb.png, kx_auc_asr.png, kx_auc_trig.png, kx_numbers.txt")