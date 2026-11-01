#!/usr/bin/env python3
"""trp_plot.py — does Tr still keep more backdoor when it is the exact constraint of RoRA Eq. 9 instead of the penalty? From logs only.

  python trp_plot.py [--repo .] [--out .]   ->  trp_epochs.png, trp_numbers.txt

Arms (all on each PZ's W_pre,K, Pt on, lr 2e-4, fine-tune seeds 0-2; found by experiment name via ft_logs.py):
  K       plain LoRA on W_pre,K                                         (psel_A)
  KTr     + Tr as the penalty of Eq. 10 (lambda 10, d/k, mean over 48)  (psel_A)
  KTrP    + Tr as the exact constraint of Eq. 9 (--tr_proj)             (psel_G)
  base    plain LoRA on the original PZ, for reference in the numbers only
ASR (B) = after fine-tuning, before Pt, SST-2 test (each epoch). (C) = after Pt. rel = sigma_max(Delta W)/sigma_max(W_pre,K) on the 6 modules
Pt rescales, = 2/s from the logged Pt factor s, best-dev adapter. Description of 3 seeds per cell, not a test.
"""
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_logs as F

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--out", default="."); a = ap.parse_args()
LAB = ["s0", "s1", "s2", "s3", "s4"]
PZ = {"s0": "pz_epoch/n1500_s0_ep2/pytorch_model.bin", "s1": "pz_cleandev/cleandev_n1500_s1/pytorch_model.bin",
      "s2": "pz_v2/badnet_s2/pytorch_model.bin", "s3": "pz_cleandev/cleandev_n1500_s3/pytorch_model.bin",
      "s4": "pz_epoch/n1500_s4_ep2/pytorch_model.bin"}
RSTAR = {"s0": 0.8, "s1": 0.995, "s2": 0.9, "s3": 0.9, "s4": 0.8}
KMEAN = {"s0": 87, "s1": 626, "s2": 186, "s3": 174, "s4": 87}
ASR_A_K = {"s0": 0.703, "s1": 1.0, "s2": 1.0, "s3": 1.0, "s4": 0.311}
SEEDS = (0, 1, 2)
ARMS = ("base", "K", "KTr", "KTrP")
KW = {"base": None, "K": {}, "KTr": dict(tr=True), "KTrP": dict(tr=True, trmode="proj")}
NAME = {"base": "原 PZ 的普通 LoRA", "K": "只有 K", "KTr": "K + Tr（懲罰，Eq. 10）", "KTrP": "K + Tr（精確投影，Eq. 9）"}

paths = F.all_logs(a.repo); R = F.runs(paths)
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
def rel(n):
    occ = [x for x in S.get(n, []) if len(x) == 6]
    return float(np.mean([2 / s for s in occ[-1]])) if occ else np.nan

D = {}
for l in LAB:
    for arm in ARMS:
        for s in SEEDS:
            n = F.exp_name(s, PZ[l]) if arm == "base" else F.exp_name(s, PZ[l], wtag=f"wh{RSTAR[l]:g}", **KW[arm])
            r = F.pick(R, n)
            if r is None:
                sys.exit(f"missing completed run {n}")
            e, sm = r["epochs"], r["summary"]
            D[l, arm, s] = dict(asr=np.array([e[i][2] for i in range(20)]), last5=float(np.mean([e[i][2] for i in range(15, 20)])),
                                pt=float(sm["pt_ASR"]), lastpt=float(sm.get("last_pt_ASR", np.nan)), test=float(sm["pre-pt_test_acc"]),
                                dev5=float(np.mean([e[i][0] for i in range(15, 20)])), rel=rel(n))

out = open(os.path.join(a.out, "trp_numbers.txt"), "w")
def say(t=""):
    print(t); out.write(t + "\n")
def line(key, arms=ARMS, fmt="{:.3f}"):
    say(f"--- {key}（順序 s0 s1 s2 s3 s4 | 5 顆平均）")
    for arm in arms:
        v = [np.nanmean([D[l, arm, s][key] for s in SEEDS]) for l in LAB]
        say(f"  {NAME[arm]:<22s} " + " ".join(fmt.format(x) for x in v) + f" | {np.nanmean(v):.3f}")
say("每格 = 3 個微調 seed 的平均")
line("last5"); line("pt"); line("lastpt", arms=("K", "KTr", "KTrP")); line("test"); line("dev5"); line("rel", arms=("K", "KTr", "KTrP"))
for arm in ("KTr", "KTrP"):
    g = [D[l, arm, s]["last5"] - D[l, "K", s]["last5"] for l in LAB for s in SEEDS]
    say(f"{NAME[arm]} − 只有 K 的持續性（逐 seed 配對，15 對）：平均 {np.mean(g):+.3f}，比只有 K 高的有 {sum(x > 0 for x in g)} 對")
g = [D[l, "KTrP", s]["last5"] - D[l, "KTr", s]["last5"] for l in LAB for s in SEEDS]
say(f"投影 − 懲罰（15 對）：平均 {np.mean(g):+.3f}，投影較低的有 {sum(x < 0 for x in g)} 對")
GAP = {arm: np.array([np.mean([D[l, arm, s]["asr"] - D[l, "K", s]["asr"] for s in SEEDS], axis=0) for l in LAB]) for arm in ("KTr", "KTrP")}
for arm in ("KTr", "KTrP"):
    say(f"每個 epoch 的 {NAME[arm]} − 只有 K（5 顆平均）：" + " ".join(f"{x:+.2f}" for x in GAP[arm].mean(0)))
out.close()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"K": "#2a78d6", "KTr": "#eb6834", "KTrP": "#1baf7a"}
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 10.5, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
def style(ax):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig, axs = plt.subplots(2, 3, figsize=(15, 9.2), facecolor="white")
ep = np.arange(1, 21)
for i, l in enumerate(LAB):
    ax = axs.flat[i]; style(ax); ends = []
    for arm in ("K", "KTr", "KTrP"):
        M = np.stack([D[l, arm, s]["asr"] for s in SEEDS])
        ax.fill_between(ep, M.min(0), M.max(0), color=COL[arm], alpha=0.13, lw=0)
        ax.plot(ep, M.mean(0), color=COL[arm], lw=2)
        ends.append([M.mean(0)[-1], np.mean([D[l, arm, s]["last5"] for s in SEEDS]), arm])
    ends.sort()
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.045)
    for y, lf, arm in ends:
        ax.annotate(f"{lf:.2f}", (20, y), xytext=(6, 0), textcoords="offset points", color=COL[arm], fontsize=9, va="center")
    ax.plot([0], [ASR_A_K[l]], marker="D", mfc="white", mec=INK2, mew=1.6, ms=7, ls="none")
    ax.set_xlim(-0.6, 21.8); ax.set_ylim(-0.02, 1.04); ax.set_xticks([0, 1, 5, 10, 15, 20])
    ax.set_title(f"{l}　r*={RSTAR[l]:g}　k 平均 {KMEAN[l]}／1024", fontsize=10.5, loc="left")
    if i % 3 == 0:
        ax.set_ylabel("ASR (B)")
    if i >= 3:
        ax.set_xlabel("epoch（0 = 微調前；n = 第 n 個 epoch 結束）")
ax = axs.flat[5]; style(ax)
for arm in ("KTr", "KTrP"):
    g = GAP[arm]
    ax.fill_between(ep, g.min(0), g.max(0), color=COL[arm], alpha=0.15, lw=0)
    ax.plot(ep, g.mean(0), color=COL[arm], lw=2.4, label=NAME[arm] + " − 只有 K")
ax.axhline(0, color=INK2, lw=1)
ax.set_xlim(0.5, 20.5); ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlabel("epoch")
ax.set_ylabel("和只有 K 的 ASR (B) 差（正 = 後門留更多）")
ax.set_title("兩種 Tr 各比只有 K 多留多少（5 顆平均）", fontsize=10.5, loc="left")
ax.legend(fontsize=8.8, frameon=False, loc="upper left")
h = [Line2D([], [], color=COL[arm], lw=2, label=NAME[arm]) for arm in ("K", "KTr", "KTrP")] + [
     Line2D([], [], marker="D", mfc="white", mec=INK2, mew=1.6, ms=7, ls="none", label="微調前 ASR (A)，W_pre,K（dev；三組共用）")]
fig.legend(handles=h, loc="upper center", ncol=4, fontsize=9.3, frameon=False, bbox_to_anchor=(0.5, 0.955))
fig.suptitle("Tr 換成精確的 Eq. 9 之後，還會讓後門多留嗎？（5 顆 PZ × 微調 seed 0–2，都在 W_pre,K 上、沒有 Cl）", fontsize=12.5, x=0.012, ha="left", y=0.99)
fig.text(0.012, 0.008, "前五格：實線 = 3 個微調 seed 的平均，色帶 = 3 個的最小～最大；右端數字 = 最後 5 個 epoch 的平均（持續性）。ASR (B) = 每個 epoch 結束時、Pt 前、SST-2 test；"
         "epoch 0 的菱形是 ASR (A)：PZ 自己的頭、未微調、SST-2 dev（不同資料，只當起點參考）。\n"
         "右下：每顆 PZ 先把同一個微調 seed 的兩組相減再對 3 個 seed 平均，粗線 = 5 顆的平均，色帶 = 5 顆的最小～最大。"
         "懲罰 = λ 10、乘 d/k、對 48 個 module 平均；精確投影 = A、B 每一步投影到 W_pre,K 輸入／輸出空間的補空間，沒有 λ。",
         fontsize=8.6, color=INK2, va="bottom", ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 0.92))
fig.savefig(os.path.join(a.out, "trp_epochs.png"), dpi=150, facecolor="white"); plt.close(fig)
print("wrote trp_epochs.png, trp_numbers.txt")