#!/usr/bin/env python3
"""mech_plot.py — why does K + Tr keep more backdoor than K alone? Two pictures from logs that already exist (no GPU, numpy + matplotlib).

  python mech_plot.py [--repo .] [--out .]
  -> mech_epochs.png, mech_size.png, mech_numbers.txt

Runs used (all BadNet "mn", RoBERTa-large, LoRA r 8, lr 2e-4 unless said, 20 epochs, Pt on, found by experiment name via ft_logs.py):
  base   plain LoRA on the original PZ (W_pre), fine-tune seeds 0-2          (logs persist_epochs / persist_cleandev / baseline)
  K      plain LoRA on W_pre,K (whiten, r* per PZ), seeds 0-2               (psel_A)
  K+Tr   LoRA on W_pre,K + Tr (lambda 10, k = each module's own rank of W_pre,K), seeds 0-2   (psel_A)
  s2 only, older runs on the same PZ (pz_v2/badnet_s2), fine-tune seeds 3-5, original W_pre:
         plain LoRA at lr 2e-4 (S2), 1e-4 and 5e-5 (S5), 2e-5 (S5b); Tr-only k = 8, lambda 10 and 480 (S4)

ASR types: (A) = the PZ's own head, before fine-tuning, SST-2 DEV (444 triggered positives; kx_numbers.txt);
           (B) = after fine-tuning, before Pt, SST-2 TEST (what variant_finetune.py logs per epoch).
mech_epochs.png: ASR (B) per epoch, one panel per PZ (mean of 3 seeds, band = min-max of the 3), the step-0 ASR (A) as a hollow
  marker at epoch 0; last panel = K+Tr minus K per epoch (paired by seed, mean of 3), one line per PZ: when the gap forms.
mech_size.png: x = size of Delta W in the 6 modules Pt rescales (q, v of layers 21-23): rel = sigma_max(2 B A) / sigma_max(W), W = the
  frozen weight it is added to (W_pre or W_pre,K); computed from the logged Pt factor s = sigma_max(W) / sigma_max(B A) as 2 / s, mean
  over the 6 modules; BEST-DEV adapter (the only one whose s is logged). y = ASR (B) of that same best-dev adapter (pre-Pt).
  The s2 panel adds the older lr sweep: "how much backdoor plain LoRA keeps at a given update size".
"""
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_logs as F

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--out", default=".")
a = ap.parse_args()

LAB = ["s0", "s1", "s2", "s3", "s4"]
PZ = {"s0": "pz_epoch/n1500_s0_ep2/pytorch_model.bin", "s1": "pz_cleandev/cleandev_n1500_s1/pytorch_model.bin",
      "s2": "pz_v2/badnet_s2/pytorch_model.bin", "s3": "pz_cleandev/cleandev_n1500_s3/pytorch_model.bin",
      "s4": "pz_epoch/n1500_s4_ep2/pytorch_model.bin"}
RSTAR = {"s0": 0.8, "s1": 0.995, "s2": 0.9, "s3": 0.9, "s4": 0.8}
KMEAN = {"s0": (86.7, 3, 175), "s1": (625.7, 469, 731), "s2": (186.2, 50, 340), "s3": (174.3, 28, 302), "s4": (87.1, 5, 189)}  # psel_A log [W_pre,K] lines
ASR_A_FULL = {"s0": 0.968, "s1": 1.0, "s2": 1.0, "s3": 1.0, "s4": 1.0}       # kx_numbers.txt, untruncated PZ
ASR_A_K = {"s0": 0.703, "s1": 1.0, "s2": 1.0, "s3": 1.0, "s4": 0.311}        # kx_numbers / kextra, W_pre,K at r* (whiten, simultaneous)
SEEDS = (0, 1, 2)
ARMS = ("base", "K", "KTr")
ANAME = {"base": "base（原 W_pre，純 LoRA）", "K": "只有 K（W_pre,K，純 LoRA）", "KTr": "K + Tr（λ10，k = 各 module 自己的秩）"}

paths = F.all_logs(a.repo)
R = F.runs(paths)

# Pt factors s of the best-dev adapter: "[Mechanism 3] <module>: new scaling s = X", between "Running experiment" and SUMMARY
RE_S = re.compile(r"\[Mechanism 3\] (\S+): new scaling s = ([-+0-9.eE]+)")
S = {}
for p in paths:
    cur = None
    for raw in open(p, errors="replace"):
        for line in raw.replace("\r", "\n").split("\n"):
            m = F.RE_RUN.search(line)
            if m:
                cur = m.group(1); S.setdefault(cur, []).append([]); continue
            if cur is None:
                continue
            m = RE_S.search(line)
            if m:
                S[cur][-1].append(float(m.group(2))); continue
            if F.RE_SUM.search(line):
                cur = None

def rel_top3(name):
    occ = [x for x in S.get(name, []) if len(x) == 6]
    if not occ:
        return np.nan
    last = occ[-1]
    for o in occ[:-1]:
        if not np.allclose(o, last):
            raise ValueError(f"{name}: two completed runs with different Pt factors")
    return float(np.mean([2.0 / s for s in last]))

def name(label, seed, arm):
    w = f"wh{RSTAR[label]:g}"
    if arm == "base":
        return F.exp_name(seed, PZ[label])
    return F.exp_name(seed, PZ[label], wtag=w, tr=(arm == "KTr"))

D = {}
for l in LAB:
    for arm in ARMS:
        for s in SEEDS:
            n = name(l, s, arm); r = F.pick(R, n)
            if r is None:
                sys.exit(f"missing completed run {n}")
            D[l, arm, s] = dict(asr=np.array([r["epochs"][e][2] for e in range(20)]), best_ep=int(r["summary"]["best_epoch"]),
                                best_asr=float(r["summary"]["pre-pt_ASR"]), rel=rel_top3(n),
                                last5=float(np.mean([r["epochs"][e][2] for e in range(15, 20)])))

# older s2 runs (seeds 3-5, original W_pre)
def old_name(seed, lr, tr=None):
    meth = "lora" + (f"_tr{tr}_k8" if tr else "") + "_pt"
    return f"roberta_badnet_sst-2_{meth}_lr{lr}_r8_a16_seed{seed}_ep20_wd0.01_pz-badnet_s2"
OLD = {}
for key, lr, tr in (("lr2e-4", "2e-04", None), ("lr1e-4", "1e-04", None), ("lr5e-5", "5e-05", None), ("lr2e-5", "2e-05", None),
                    ("Tr k8 λ10", "2e-04", 10), ("Tr k8 λ480", "2e-04", 480)):
    rows = []
    for s in (3, 4, 5):
        n = old_name(s, lr, tr); r = F.pick(R, n)
        if r is None:
            sys.exit(f"missing completed run {n}")
        rows.append(dict(best_asr=float(r["summary"]["pre-pt_ASR"]), rel=rel_top3(n), best_ep=int(r["summary"]["best_epoch"])))
    OLD[key] = rows

# ---------------- numbers ----------------
out = open(os.path.join(a.out, "mech_numbers.txt"), "w")
def say(t=""):
    print(t); out.write(t + "\n")
say("ASR (B) = 微調後、Pt 前、SST-2 test；每格 = 3 個微調 seed（0-2）的平均 [最小,最大]")
say("PZ | r* | k 平均(最小-最大) | 起點 ASR(A) 原PZ / W_pre,K | 持續性(最後5 epoch 平均) base / K / K+Tr | K+Tr−K（逐 seed 配對）")
for l in LAB:
    lf = {arm: [D[l, arm, s]["last5"] for s in SEEDS] for arm in ARMS}
    gap = [D[l, "KTr", s]["last5"] - D[l, "K", s]["last5"] for s in SEEDS]
    say(f"{l} | {RSTAR[l]:g} | {KMEAN[l][0]:.0f} ({KMEAN[l][1]}-{KMEAN[l][2]}) | {ASR_A_FULL[l]:.2f} / {ASR_A_K[l]:.2f} | "
        + " / ".join(f"{np.mean(lf[arm]):.3f}[{min(lf[arm]):.2f},{max(lf[arm]):.2f}]" for arm in ARMS)
        + f" | {np.mean(gap):+.3f} (" + " ".join(f"{g:+.2f}" for g in gap) + ")")
say()
say("每個 epoch 的 K+Tr − K（ASR (B)，逐 seed 配對，3 個平均）：epoch 1..20（第 n 個 epoch 結束時）")
G = {l: np.mean([D[l, "KTr", s]["asr"] - D[l, "K", s]["asr"] for s in SEEDS], axis=0) for l in LAB}
for l in LAB:
    say(f"  {l}: " + " ".join(f"{x:+.2f}" for x in G[l]))
say("  5 PZ 平均: " + " ".join(f"{x:+.2f}" for x in np.mean([G[l] for l in LAB], axis=0)))
say()
say("ΔW 大小（rel，Pt 的 6 個 module，best-dev adapter）與那個 adapter 的 ASR (B)：平均 [最小,最大]；K+Tr/K 的大小比")
for l in LAB:
    rr = {arm: [D[l, arm, s]["rel"] for s in SEEDS] for arm in ARMS}
    ba = {arm: [D[l, arm, s]["best_asr"] for s in SEEDS] for arm in ARMS}
    say(f"  {l}: " + "  ".join(f"{arm} rel {np.mean(rr[arm]):.3f}[{min(rr[arm]):.3f},{max(rr[arm]):.3f}] ASR {np.mean(ba[arm]):.2f}" for arm in ARMS)
        + f"  | K / K+Tr 大小比 {np.mean(rr['K']) / np.mean(rr['KTr']):.2f}")
say("  s2 舊的 run（seed 3-5，原 W_pre）：" + "  ".join(
    f"{k} rel {np.mean([x['rel'] for x in v]):.3f} ASR {np.mean([x['best_asr'] for x in v]):.2f}" for k, v in OLD.items()))
out.close()

# ---------------- pictures ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"base": "#7a7a7a", "K": "#2a78d6", "KTr": "#eb6834"}
PCOL = {"s0": "#2a78d6", "s1": "#eb6834", "s2": "#1baf7a", "s3": "#a050c8", "s4": "#c8a000"}
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10.5, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
def style(ax):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
def ptitle(l):
    km = KMEAN[l]
    return f"{l}　r*={RSTAR[l]:g}　k 平均 {km[0]:.0f}（{km[1]}–{km[2]}）／1024"

# figure 1: per-epoch ASR (B)
fig, axs = plt.subplots(2, 3, figsize=(15, 9.2), facecolor="white")
ep = np.arange(1, 21)
for i, l in enumerate(LAB):
    ax = axs.flat[i]; style(ax)
    ends = []
    for arm in ARMS:
        M = np.stack([D[l, arm, s]["asr"] for s in SEEDS])
        ax.fill_between(ep, M.min(0), M.max(0), color=COL[arm], alpha=0.14, lw=0)
        ax.plot(ep, M.mean(0), color=COL[arm], lw=2)
        ends.append([M.mean(0)[-1], np.mean([D[l, arm, s]["last5"] for s in SEEDS]), arm])
    ends.sort()                                   # end labels: push apart so they do not overlap (at least 0.045 apart)
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.045)
    for y, lf, arm in ends:
        ax.annotate(f"{lf:.2f}", (20, y), xytext=(6, 0), textcoords="offset points", color=COL[arm], fontsize=9, va="center")
    ax.plot([-0.3], [ASR_A_FULL[l]], marker="o", mfc="white", mec=COL["base"], mew=1.6, ms=8, ls="none")
    ax.plot([0.25], [ASR_A_K[l]], marker="D", mfc="white", mec=COL["K"], mew=1.6, ms=7, ls="none")
    ax.set_xlim(-0.9, 21.8); ax.set_ylim(-0.02, 1.04); ax.set_xticks([0, 1, 5, 10, 15, 20])
    ax.set_title(ptitle(l), fontsize=10.5, loc="left")
    if i % 3 == 0:
        ax.set_ylabel("ASR")
    if i >= 3:
        ax.set_xlabel("epoch（0 = 微調前；n = 第 n 個 epoch 結束）")
ax = axs.flat[5]; style(ax)
for l in LAB:
    ax.plot(ep, G[l], color=PCOL[l], lw=1.8, label=f"{l}（起點 ASR(A) {ASR_A_K[l]:.2f}）")
ax.plot(ep, np.mean([G[l] for l in LAB], axis=0), color=INK, lw=2.6, ls="--", label="5 顆平均")
ax.axhline(0, color=INK2, lw=1)
ax.set_xlim(0.5, 20.5); ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlabel("epoch")
ax.set_ylabel("K+Tr − K 的 ASR (B)（正 = Tr 讓後門留更多）")
ax.set_title("差距在哪幾個 epoch 形成？（逐 seed 配對，3 個平均）", fontsize=10.5, loc="left")
ax.set_ylim(-0.16, 0.26); ax.legend(fontsize=8.4, frameon=False, loc="lower right", ncol=2)
h = [Line2D([], [], color=COL[arm], lw=2, label=ANAME[arm]) for arm in ARMS] + [
     Line2D([], [], marker="o", mfc="white", mec=COL["base"], mew=1.6, ms=8, ls="none", label="微調前 ASR (A)，原 PZ（dev）"),
     Line2D([], [], marker="D", mfc="white", mec=COL["K"], mew=1.6, ms=7, ls="none", label="微調前 ASR (A)，W_pre,K（dev；K 和 K+Tr 共用）")]
fig.legend(handles=h, loc="upper center", ncol=3, fontsize=9.3, frameon=False, bbox_to_anchor=(0.5, 0.955))
fig.suptitle("後門在微調中怎麼退：原 PZ、只有 K、K + Tr（5 顆 PZ × 微調 seed 0–2；r* 是 bonf 規則選的）", fontsize=12.5, x=0.012, ha="left", y=0.99)
fig.text(0.012, 0.008, "實線 = 3 個微調 seed 的平均，色帶 = 3 個的最小～最大；右端數字 = 最後 5 個 epoch 的平均（持續性）。ASR (B) = 每個 epoch 結束時、Pt 前、SST-2 test；"
         "epoch 0 的空心點是 ASR (A)：PZ 自己的頭、未微調、SST-2 dev（不同資料，只當起點參考）。\n"
         "k = W_pre,K 每個 module 留下的方向數，也是 Tr 禁止 ΔW 進入的方向數（輸入側與輸出側各 k 個）。右下：每顆 PZ 一條線（K+Tr 減只有 K，同一個微調 seed 相減後平均），黑虛線 = 5 顆平均。",
         fontsize=8.6, color=INK2, va="bottom", ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 0.92))
fig.savefig(os.path.join(a.out, "mech_epochs.png"), dpi=150, facecolor="white"); plt.close(fig)

# figure 2: Delta W size vs ASR (B) of the best-dev adapter
fig, axs = plt.subplots(2, 3, figsize=(15, 9.2), facecolor="white")
allrel = [D[k]["rel"] for k in D] + [x["rel"] for v in OLD.values() for x in v]
XLO, XHI = min(allrel) / 1.3, max(allrel) * 1.3
for i, l in enumerate(LAB + ["s2old"]):
    ax = axs.flat[i]; style(ax); ax.set_xscale("log")
    if l == "s2old":
        ax.set_title("s2 放大：加上舊的 lr 掃描（seed 3–5，原 W_pre）", fontsize=10.5, loc="left")
        xs = [np.mean([x["rel"] for x in OLD[k]]) for k in ("lr2e-4", "lr1e-4", "lr5e-5", "lr2e-5")]
        ys = [np.mean([x["best_asr"] for x in OLD[k]]) for k in ("lr2e-4", "lr1e-4", "lr5e-5", "lr2e-5")]
        for k in ("lr2e-4", "lr1e-4", "lr5e-5", "lr2e-5"):
            ax.scatter([x["rel"] for x in OLD[k]], [x["best_asr"] for x in OLD[k]], s=14, color=INK, alpha=0.35, lw=0)
        ax.plot(xs, ys, color=INK, lw=1.6, marker="s", ms=6, label="純 LoRA，原 W_pre，lr 2e-4 → 2e-5（S2/S5/S5b）")
        for x, y, t in zip(xs, ys, ("2e-4", "1e-4", "5e-5", "2e-5")):
            ax.annotate(t, (x, y), xytext=(4, -11), textcoords="offset points", fontsize=8, color=INK2)
        for k, mk in (("Tr k8 λ10", "^"), ("Tr k8 λ480", "v")):
            ax.scatter([np.mean([x["rel"] for x in OLD[k]])], [np.mean([x["best_asr"] for x in OLD[k]])], marker=mk, s=60,
                       facecolor="white", edgecolor=COL["KTr"], lw=1.6, zorder=3, label=f"Tr-only {k[3:]}，原 W_pre（S4）")
        ll = "s2"
    else:
        ax.set_title(ptitle(l), fontsize=10.5, loc="left"); ll = l
    for arm in ARMS:
        xr = [D[ll, arm, s]["rel"] for s in SEEDS]; yr = [D[ll, arm, s]["best_asr"] for s in SEEDS]
        ax.scatter(xr, yr, s=16, color=COL[arm], alpha=0.45, lw=0)
        ax.errorbar(np.mean(xr), np.mean(yr), xerr=[[np.mean(xr) - min(xr)], [max(xr) - np.mean(xr)]],
                    yerr=[[np.mean(yr) - min(yr)], [max(yr) - np.mean(yr)]], color=COL[arm], marker="o", ms=7, lw=1.3, capsize=0, zorder=4)
    ax.set_xlim(XLO, XHI); ax.set_ylim(-0.02, 1.04)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.02, 0.05, 0.1, 0.2, 0.4]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if i % 3 == 0:
        ax.set_ylabel("best-dev adapter 的 ASR (B)")
    if i >= 3:
        ax.set_xlabel("ΔW 大小 rel = σmax(ΔW)／σmax(W)（Pt 的 6 個 module 平均；對數）")
    if l == "s2old":
        ax.legend(fontsize=8.2, frameon=False, loc="lower left")
h = [Line2D([], [], color=COL[arm], marker="o", lw=1.3, label=ANAME[arm] + "，seed 0–2") for arm in ARMS]
fig.legend(handles=h, loc="upper center", ncol=3, fontsize=9.3, frameon=False, bbox_to_anchor=(0.5, 0.955))
fig.suptitle("ΔW 越小、後門留越多？（每個點 = 一次微調的 best-dev adapter，Pt 前）", fontsize=12.5, x=0.012, ha="left", y=0.99)
fig.text(0.012, 0.008, "小點 = 單一微調 seed；大點 = 3 個的平均，十字 = 3 個的最小～最大。ΔW = 2·B·A（LoRA 的 alpha/r = 2），W = 它加上去的凍結權重（原 W_pre 或 W_pre,K）；"
         "\nrel 由 log 裡 Pt 的 s = σmax(W)/σmax(BA) 換算（rel = 2/s），只有 best-dev adapter 有記錄，所以縱軸也用 best-dev 的 ASR (B)（Pt 前、SST-2 test），不是最後 5 個 epoch。\n"
         "右下（s2）：黑線 = 同一顆 PZ 上、原 W_pre 的純 LoRA 只把學習率調小（seed 3–5，和 seed 0–2 的 base 不是同一批），表示「純粹因為 ΔW 變小」大概會留多少後門；橘色空心三角 = 舊的 Tr（k = 8，原 W_pre）。",
         fontsize=8.6, color=INK2, va="bottom", ha="left")
fig.tight_layout(rect=(0, 0.075, 1, 0.92))
fig.savefig(os.path.join(a.out, "mech_size.png"), dpi=150, facecolor="white"); plt.close(fig)

# figure 3: why is Tr's Delta W small? observer logs (obs/<run>/modules.csv; 10/05-10/06 runs on badnet_s2 = s2, seeds 3-5,
# original W_pre, Tr k = 8 on all 48 modules). Per module every 20 steps: ||A||_F^2, the gradient norm of A as AdamW sees it
# (task + penalty), and ||Delta W||_F. Mean over the 48 modules (gradient: root mean square).
import csv
OBS = {"base": ("lora_pt", "#7a7a7a", "純 LoRA（原 W_pre）"), "tr10": ("lora_tr10_k8_pt", "#eb6834", "Tr λ10，k = 8"),
       "tr480": ("lora_tr480_k8_pt", "#b8322a", "Tr λ480，k = 8")}
SPE = 217
def obs_run(tag, seed):
    d = os.path.join(a.repo, "obs", f"roberta_badnet_sst-2_{tag}_lr2e-04_r8_a16_seed{seed}_ep20_wd0.01_pz-badnet_s2", "modules.csv")
    if not os.path.exists(d):
        return None
    by = {}
    for r in csv.DictReader(open(d)):
        if r["step"] == "end":
            continue
        by.setdefault(int(r["epoch"]) * SPE + int(r["step"]), []).append(r)
    t = np.array(sorted(by))
    col = lambda c, f: np.array([f([float(r[c]) for r in by[x]]) for x in t])
    return t, col("normA2", np.mean), col("gradA", lambda v: float(np.sqrt(np.mean(np.square(v))))), col("fro", np.mean)
O = {k: [obs_run(v[0], s) for s in (3, 4, 5)] for k, v in OBS.items()}
if all(x is not None for v in O.values() for x in v):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="white")
    panels = (("‖A‖²（48 個 module 平均）", 1, "linear"), ("A 的梯度大小（task + 懲罰，AdamW 看到的；對數）", 2, "log"), ("‖ΔW‖_F（48 個 module 平均）", 3, "linear"))
    for ax, (tt, j, sc) in zip(axs, panels):
        style(ax); ax.set_yscale(sc)
        for k, (tag, c, lab) in OBS.items():
            for i, run in enumerate(O[k]):
                y = run[j]
                if sc == "log":
                    y = np.where(y > 0, y, np.nan)
                ax.plot(run[0] / SPE, y, color=c, lw=1.4, alpha=0.9, label=lab if i == 0 else None)
        ax.axvspan(0, 160 / SPE, color="#f0d8c8", alpha=0.45, lw=0)
        ax.set_title(tt, fontsize=10.5, loc="left"); ax.set_xlabel("epoch（每 20 步記一次）"); ax.set_xlim(0, 20)
    axs[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle("Tr 為什麼讓 ΔW 變小？A 幾乎沒動（舊的觀察 log：s2 = badnet_s2，原 W_pre，微調 seed 3–5 各一條線）", fontsize=12.5, x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.01, "淡橘底 = 前 160 步：Tr 的懲罰把 A 在 W_pre 前 k 個輸入方向的成分壓到 0 的那一段（之後懲罰 ≈ 0）。第 0 步 B = 0，所以 A 的 task 梯度剛好是 0，純 LoRA 的線從第 20 步才出現（對數軸畫不出 0）。\n"
             "ΔW = 2·B·A；‖A‖² 的起點 ≈ 2.66 是 LoRA 的隨機初始值。AdamW 用 β2 = 0.999 記住過去梯度的平方（時間常數約 1000 步 ≈ 4.6 epoch；20 epoch = 4340 步）。",
             fontsize=8.6, color=INK2, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(os.path.join(a.out, "mech_adam.png"), dpi=150, facecolor="white"); plt.close(fig)
    print("wrote mech_adam.png")
else:
    print("obs/ runs missing: mech_adam.png not drawn")
print("wrote mech_epochs.png, mech_size.png, mech_numbers.txt")