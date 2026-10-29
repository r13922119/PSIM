#!/usr/bin/env python3
"""final_analyze.py — after run_final.sh: does the algorithm (K by make_wpre.py, P by psel_analyze.py) defend?
CPU, numpy + matplotlib; reads logs/*.log by experiment name (ft_logs.py) and the P choice file.
Three arms per panel PZ, fine-tune seeds 0-2 (equal seed counts; a mean is printed only when all 3 seeds exist):
  base   plain LoRA on the ORIGINAL PZ (the persistence runs)
  konly  plain LoRA on W_pre,K                                  (what K alone does)
  rora   RoRA on W_pre,K: Cl at P + Tr lambda 10 kauto          (seed 0 = the run_psel.sh run at p = P)
All arms have Pt applied after training (variant_finetune.py --use_spectral_rescaling), so each run gives:
  type (B), after fine-tuning, before Pt: at the saved best-dev epoch, and the mean over the last 5 epochs
  type (C), after Pt (on the saved best-dev checkpoint)
Accuracy = SST-2 test clean accuracy; ASR = share of triggered SST-2 test positives predicted negative.
  python final_analyze.py --level bonf [--walk stepwise --stat last5 --rule stepwise] [repo]
  -> final.png, final_numbers.txt in the current directory
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_logs as F

ap = argparse.ArgumentParser()
ap.add_argument("repo", nargs="?", default="/home/claude/r13922119/psim")
ap.add_argument("--level", required=True, choices=["bonf", "per_r"])
ap.add_argument("--walk", default="stepwise"); ap.add_argument("--stat", default="last5"); ap.add_argument("--rule", default="stepwise")
ap.add_argument("--epochs", type=int, default=20)
a = ap.parse_args()
cf = os.path.join(a.repo, "logs", "psel", f"choice_P_{a.level}_{a.walk}_{a.stat}_{a.rule}.tsv")
rows = [l.rstrip("\n").split("\t") for l in open(cf)][1:]
R = F.runs(F.all_logs(a.repo))
ARMS = ("base", "konly", "rora")
NAME = {"base": "普通 LoRA，原本 PZ", "konly": "普通 LoRA，W_pre,K（只有 K）", "rora": "RoRA，W_pre,K（K + Cl(P) + Tr）"}
COL = {"base": "#7a7a7a", "konly": "#eb6834", "rora": "#2a78d6"}

def last5(run, i):          # i = 1 test accuracy, 2 ASR (per-epoch, type B)
    es = sorted(run["epochs"])[-5:]
    return float(np.mean([run["epochs"][e][i] for e in es]))

MET = [("acc_B", "test 正確率 (B) 最佳 dev epoch", lambda r: r["summary"]["pre-pt_test_acc"]),
       ("acc_B5", "test 正確率 (B) 最後 5 epoch 平均", lambda r: last5(r, 1)),
       ("acc_C", "test 正確率 (C) Pt 後", lambda r: r["summary"]["pt_test_acc"]),
       ("asr_B", "ASR (B) 最佳 dev epoch", lambda r: r["summary"]["pre-pt_ASR"]),
       ("asr_B5", "ASR (B) 最後 5 epoch 平均", lambda r: last5(r, 2)),
       ("asr_C", "ASR (C) Pt 後", lambda r: r["summary"]["pt_ASR"])]
D, out = {}, []
def say(s=""):
    print(s); out.append(s)
say(f"Final test of the algorithm (K: {a.level}/{a.walk}; P: {a.stat}/{a.rule}; {cf}). Values per fine-tune seed 0/1/2, then the mean.")
for label, r, js, pz, P in rows:
    tag = f"wh{float(r):g}"
    names = {"base": lambda s: F.exp_name(s, pz, epochs=a.epochs),
             "konly": lambda s: F.exp_name(s, pz, tag, epochs=a.epochs),
             "rora": lambda s: F.exp_name(s, pz, tag, float(P), tr=True, epochs=a.epochs)}
    say(f"\n=== {label}: r* = {r}, P = {P}")
    D[label] = {}
    for arm in ARMS:
        runs = [F.pick(R, names[arm](s), a.epochs) for s in (0, 1, 2)]
        D[label][arm] = runs
        for key, nm, fn in MET:
            v = [fn(x) if x is not None else None for x in runs]
            txt = " / ".join("  —  " if x is None else f"{x:.3f}" for x in v)
            mean = f"mean {np.mean(v):.3f}" if all(x is not None for x in v) else "(seeds missing)"
            say(f"  {arm:6s} {nm:32s} {txt}   {mean}")
complete = all(x is not None for l in D for arm in ARMS for x in D[l][arm])
if complete:
    say(f"\n=== panel: mean over the {len(D)} PZs of each PZ's 3-seed mean")
    for key, nm, fn in MET:
        say(f"  {nm:32s} " + "  ".join(f"{arm} {np.mean([np.mean([fn(x) for x in D[l][arm]]) for l in D]):.3f}" for arm in ARMS))
else:
    say("\n(panel mean not printed: some runs are missing, and means are compared only over equal seed counts)")
open("final_numbers.txt", "w").write("\n".join(out) + "\n")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 9, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
labs = list(D); groups = labs + ([f"{len(labs)} 顆平均"] if complete else [])
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), facecolor=SURF)
for ax, (key, nm, fn) in zip(axes.ravel(), MET):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8, axis="y"); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for g, lab in enumerate(groups):
        for k, arm in enumerate(ARMS):
            x = g + (k - 1) * 0.25
            if lab in D:
                v = [fn(r) for r in D[lab][arm] if r is not None]
                ax.scatter([x] * len(v), v, s=14, color=COL[arm], alpha=0.55, lw=0)
                if len(v) == 3:
                    ax.scatter([x], [np.mean(v)], s=70, marker="_", color=COL[arm], lw=2.5)
            else:   # panel mean of the per-PZ means
                m = [np.mean([fn(r) for r in D[l][arm]]) for l in labs]
                ax.scatter([x] * len(m), m, s=10, color=COL[arm], alpha=0.35, lw=0)
                ax.scatter([x], [np.mean(m)], s=110, marker="_", color=COL[arm], lw=3, label=NAME[arm] if key == "acc_B" else None)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g.replace("_ep2", "") for g in groups], fontsize=8)
    ax.set_title(nm, fontsize=9.5)
    ax.set_ylim((0.45, 1.0) if key.startswith("acc") else (-0.02, 1.02))
handles = [plt.Line2D([], [], color=COL[arm], marker="_", ms=12, mew=2.5, lw=0, label=NAME[arm]) for arm in ARMS]
fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5, ncol=3)
fig.suptitle("挑 K 和 P 的演算法，最後的檢驗：SST-2 test 的正確率和 ASR", fontsize=10.5, x=0.01, ha="left")
fig.text(0.01, 0.01, "小點 = 一個微調 seed（0、1、2），短橫線 = 三個 seed 的平均（三個都有時才畫）；「N 顆平均」那一組：小點 = 每顆 PZ 的平均，橫線 = 它們的平均。"
         "(B) = 微調後、Pt 前；(C) = Pt 後（存下的最佳 dev checkpoint）。ASR = 插了 trigger 的 SST-2 test 正面句被判成負面的比例。"
         "RoRA 的 seed 0 就是 P 網格裡 p = P 那一次。", fontsize=7.5, color=INK2, wrap=True)
fig.tight_layout(rect=(0, 0.05, 1, 0.94)); fig.savefig("final.png", dpi=150, facecolor=SURF); plt.close(fig)
print("wrote final.png, final_numbers.txt")