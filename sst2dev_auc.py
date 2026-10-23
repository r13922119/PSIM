"""sst2dev_auc.py — 在 SST-2 dev 上重做「三條存檔規則存下的 PZ 比一比」（CPU，跑在 run_sst2dev.sh 的結果上）。

05:33 我用 SST-2 test 的 AUC 支持「量 AUC 時存最後一個 epoch」，等於拿報告用的 test 做設計決定。
這支改用 SST-2 dev（沒有任何存檔規則看過、也不是報告用的資料）重做同一個比較。

資料：logs/scores/sst2dev/n1500_s<seed>_ep<epoch>.npz（sst2dev_eval.py 寫的），插 1,500 句，5 seed × 3 epoch。
模型階段：毒化訓練存下的 checkpoint，未微調，dropout 關。
分數 = logit(正面) − logit(負面)；分數 > 0 就判正面。

量的東西（全部在 SST-2 dev 上）
  AUC          隨機抽一句正面、一句負面，正面那句分數較高的機率（同分算一半）。0.5 = 瞎猜，1 = 完全分開。不受門檻影響。
  負面句判錯率  乾淨負面句被判成正面的比例（偏向的大小）
  乾淨正確率    872 句的正確率
  ASR          (A) 類：PZ 自己、微調前，SST-2 dev 正面句插 mn 後被判成負面的比例
另外並列 SST-2 test 的 AUC，只是為了和 05:33 的數字對照，不拿來做決定。

成對比較：同一個 seed 下，某條規則存的那顆減去最後一個 epoch 那顆的 SST-2 dev AUC。
  兩顆是在同一批 872 句上量的，所以用「成對重抽」看這個差距有多少是句子抽樣造成的：
  正面句、負面句各自有放回地重抽同樣句數（同一組句子同時用在兩顆上），重算差距 2,000 次，
  列出中間 95% 的範圍（95% 是慣例，不是門檻；這裡只當作誤差帶的寬度來看）。
  規則挑到的就是最後一個 epoch 時，兩顆是同一顆，差距恰好是 0。

  python sst2dev_auc.py [repo 路徑]   → 印三張表，寫 sst2dev_auc.png
"""
import re, sys
import numpy as np
from scipy.stats import rankdata

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/r13922119/psim"
N, SEEDS, EPOCHS = 1500, range(5), range(3)

def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

Z = {(s, e): np.load(f"{REPO}/logs/scores/sst2dev/n{N}_s{s}_ep{e}.npz") for s in SEEDS for e in EPOCHS}
M = {}
for k, z in Z.items():
    sp, sn = z["sdev_pos"].astype(float), z["sdev_neg"].astype(float)
    M[k] = {"auc": auc(sp, sn), "auc_test": auc(z["down_pos"].astype(float), z["down_neg"].astype(float)),
            "neg_err": 100 * np.mean(sn > 0), "pos_err": 100 * np.mean(sp <= 0),
            "ca": 100 * (np.sum(sp > 0) + np.sum(sn <= 0)) / (len(sp) + len(sn)),
            "asr": 100 * np.mean(z["sdev_asr"].astype(float) <= 0)}
n_pos, n_neg = len(Z[(0, 0)]["sdev_pos"]), len(Z[(0, 0)]["sdev_neg"])
print(f"SST-2 dev：正面 {n_pos} 句、負面 {n_neg} 句（未微調，dropout 關）")

# ---- 三條規則各挑哪個 epoch（同 analyze_pzbias.py：PSIM 看 trig_dev、乾淨 dev 看 dev_ca，都取第一個最大值）----
runs, cur = {}, None
for f in [f"{REPO}/logs/pzbias_20261021_133652.log", f"{REPO}/logs/pzbias_20261021_141214.log"]:
    for line in open(f):
        m = re.match(r"===== pz dose n(\S+) s(\d)", line)
        if m:
            cur = (m.group(1), int(m.group(2))); runs[cur] = []; continue
        if line.startswith("====="):
            cur = None
        if line.startswith("[PROBE]") and cur:
            runs[cur].append({k: float(v) for k, v in (kv.split("=") for kv in line.split()[1:])})
def first_max(v):
    b = 0
    for i, x in enumerate(v):
        if x > v[b]:
            b = i
    return b
PICK = {"PSIM 的規則": {s: first_max([x["trig_dev"] for x in runs[(str(N), s)]]) for s in SEEDS},
        "乾淨 dev 最高": {s: first_max([x["dev_ca"] for x in runs[(str(N), s)]]) for s in SEEDS},
        "最後一個 epoch": {s: 2 for s in SEEDS}}
# 和 sst2dev_eval.py 的資料夾對照表必須一致（pz_v2 = PSIM 挑的，pz_cleandev = 乾淨 dev 挑的）
assert PICK["PSIM 的規則"] == {0: 1, 1: 0, 2: 2, 3: 0, 4: 0}, PICK["PSIM 的規則"]
assert PICK["乾淨 dev 最高"] == {0: 0, 1: 2, 2: 1, 3: 2, 4: 1}, PICK["乾淨 dev 最高"]

# ---- 表一：每顆 checkpoint ----
print("\n=== 表一 每個 seed、每個 epoch（SST-2 dev）：AUC（括號內 SST-2 test AUC）| 負面句判錯率 | ASR；← 哪條規則存這顆")
for s in SEEDS:
    for e in EPOCHS:
        who = [name for name, pk in PICK.items() if pk[s] == e]
        m = M[(s, e)]
        print(f"  seed {s} epoch {e}:  AUC {m['auc']:.4f} ({m['auc_test']:.4f}) | 負面句判錯 {m['neg_err']:5.1f}% "
              f"| ASR {m['asr']:5.1f}%   {'← ' + '、'.join(who) if who else ''}")

# ---- 表二：三條規則，5 seed 平均 ----
print("\n=== 表二 三條規則存下的 PZ，5 seed 平均（SST-2 dev；最後一欄是 SST-2 test AUC，只供對照）")
print("  %-12s %8s %12s %10s %8s %14s" % ("規則", "AUC", "負面句判錯%", "乾淨正確%", "ASR%", "(test AUC)"))
for name, pk in PICK.items():
    g = lambda key: np.mean([M[(s, pk[s])][key] for s in SEEDS])
    print("  %-12s %8.4f %12.1f %10.1f %8.1f %14.4f" % (name, g("auc"), g("neg_err"), g("ca"), g("asr"), g("auc_test")))

# ---- 表三：成對差距（某規則 − 最後一個 epoch），SST-2 dev AUC ----
rng = np.random.default_rng(0)
B = 2000
IP = rng.integers(0, n_pos, size=(B, n_pos))
IN = rng.integers(0, n_neg, size=(B, n_neg))
def boot_diff(a, b):
    za, zb = Z[a], Z[b]
    pa, na, pb, nb = (za["sdev_pos"].astype(float), za["sdev_neg"].astype(float),
                      zb["sdev_pos"].astype(float), zb["sdev_neg"].astype(float))
    d = np.array([auc(pa[IP[i]], na[IN[i]]) - auc(pb[IP[i]], nb[IN[i]]) for i in range(B)])
    return np.percentile(d, [2.5, 97.5])
print(f"\n=== 表三 同一個 seed：規則存的那顆 − 最後一個 epoch 那顆，SST-2 dev AUC 差距（方括號 = 句子成對重抽 {B} 次的中間 95%）")
for name in ("PSIM 的規則", "乾淨 dev 最高"):
    pk, diffs = PICK[name], []
    print(f"  {name}")
    for s in SEEDS:
        if pk[s] == 2:
            print(f"    seed {s}: 挑的就是最後一個 epoch，同一顆，差距 0")
            diffs.append(0.0); continue
        d = M[(s, pk[s])]["auc"] - M[(s, 2)]["auc"]
        lo, hi = boot_diff((s, pk[s]), (s, 2))
        diffs.append(d)
        print(f"    seed {s}: epoch {pk[s]} − epoch 2 = {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print(f"    5 seed 平均差距 {np.mean(diffs):+.4f}；規則那顆較高的 seed 數 {sum(x > 0 for x in diffs)}，"
          f"較低 {sum(x < 0 for x in diffs)}，同一顆或相等 {sum(x == 0 for x in diffs)}")

# ---- 圖 ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#a3a29c"
MK = {"PSIM 的規則": ("s", "#2a78d6"), "乾淨 dev 最高": ("o", "#eb6834")}
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), facecolor=SURF)
for ax, key, title in [(axes[0], "auc", "AUC（0.5 = 瞎猜，1 = 完全分開）"),
                       (axes[1], "neg_err", "乾淨負面句判錯率 %（偏向）"),
                       (axes[2], "asr", "ASR %（PZ 自己、微調前）")]:
    ax.set_facecolor(SURF)
    for s in SEEDS:
        ax.plot(list(EPOCHS), [M[(s, e)][key] for e in EPOCHS], color=LINE, lw=1.4, zorder=1)
    # seed 標籤放在右端點右邊；值太接近時往上下推開，避免疊字
    ys = sorted((M[(s, 2)][key], s) for s in SEEDS)
    allv = [M[(s, e)][key] for s in SEEDS for e in EPOCHS]
    gap = 0.07 * (max(allv) - min(allv) or 1)
    placed = []
    for y, s in ys:
        placed.append([max(y, placed[-1][0] + gap) if placed else y, s, y])
    shift = (placed[-1][0] - ys[-1][0]) / 2          # 整串往下挪一半，讓標籤以端點為中心
    for ly, s, y in placed:
        ax.annotate(f"seed {s}", xy=(2.1, y), xytext=(2.22, ly - shift), fontsize=8, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
    for name, (mk, col) in MK.items():
        xs = [PICK[name][s] + (-0.06 if mk == "s" else 0.06) for s in SEEDS]
        ax.scatter(xs, [M[(s, PICK[name][s])][key] for s in SEEDS], marker=mk, s=38, color=col,
                   edgecolor=SURF, lw=0.8, zorder=3, label=f"{name}存的")
    ax.set_xticks(list(EPOCHS)); ax.set_xticklabels([f"epoch {e}" for e in EPOCHS])
    ax.set_xlim(-0.3, 2.75)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10)
axes[0].legend(fontsize=8.5, frameon=False, loc="best")
fig.suptitle(f"插 {N} 句，毒化訓練每個 epoch 在 SST-2 dev 上（未微調，dropout 關；灰線 = 一個 seed；最後一個 epoch = 右端點）",
             fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("sst2dev_auc.png", dpi=160, facecolor=SURF)
print("\nwrote sst2dev_auc.png")