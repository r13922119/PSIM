#!/usr/bin/env python3
"""psel_analyze.py — P step of the algorithm, after run_psel.sh: pick P per PZ from SST-2 DEV accuracy only (pre-Pt).
CPU, numpy + matplotlib; reads logs/*.log (by experiment name, ft_logs.py) and logs/wpre/choice_<LEVEL>_<WALK>.tsv.

Constraint (your 10/20 "落在 baseline 的範圍內", made concrete; the exact form is still your decision, all variants printed):
  statistic S of a run = SST-2 dev accuracy, pre-Pt, Cl off at evaluation; --stat last5 (default): mean over the last 5 of the
    20 epochs; --stat best: the best epoch's (the checkpoint variant_finetune.py saves)
  threshold = the smallest S among plain LoRA on the ORIGINAL PZ, fine-tune seeds 0-2 (the persistence runs, lr 2e-4)
  p passes if S(RoRA on W_pre,K with Cl at p, Tr lambda 10 kauto, fine-tune seed 0) >= threshold
  --rule stepwise (default): P = the last p before the first failure, walking up from the smallest p > 0;  any: largest passing p
  p = 0 (Tr alone on W_pre,K) is not a candidate; it is the reference: if it fails, K + Tr alone already break the constraint.
Why the largest passing p: the step-0 AUC falls monotonically with p (your 10/24 measurement), so "damage W_pre as much as
possible at step 0 while fine-tuned accuracy stays in the baseline's range" is the largest p that keeps the constraint.
Also printed, NOT used to pick P: SST-2 test accuracy and ASR before Pt (type B, at the saved best-dev epoch; and the mean
ASR of the last 5 epochs) and after Pt (type C). ASR = share of triggered SST-2 test positives predicted negative.
  python psel_analyze.py --level bonf [--walk stepwise] [--stat last5] [--rule stepwise] [repo]
  -> psel.png, psel_epochs.png, psel_numbers.txt (in the current directory), logs/psel/choice_P_<...>.tsv in the repo
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_logs as F

ap = argparse.ArgumentParser()
ap.add_argument("repo", nargs="?", default="/home/claude/r13922119/psim")
ap.add_argument("--level", required=True, choices=["bonf", "per_r"])
ap.add_argument("--walk", default="stepwise", choices=["stepwise", "any"])
ap.add_argument("--stat", default="last5", choices=["last5", "best"])
ap.add_argument("--rule", default="stepwise", choices=["stepwise", "any"])
ap.add_argument("--ps", default="0,0.05,0.1,0.15,0.2,0.3")
ap.add_argument("--labels", default="n1500_s0_ep2,n1500_s1_ep2,n1500_s2_ep2,n1500_s3_ep2,n1500_s4_ep2")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--no_plot", action="store_true")
a = ap.parse_args()
PS = [float(x) for x in a.ps.split(",")]; CAND = [p for p in PS if p > 0]; LABELS = a.labels.split(",")
choice_f = os.path.join(a.repo, "logs", "wpre", f"choice_{a.level}_{a.walk}.tsv")
CH = {}
for line in open(choice_f):
    c = line.rstrip("\n").split("\t")
    if c[0] != "label":
        CH[c[0]] = dict(r=c[1], json=c[2], pz=c[3])
R = F.runs(F.all_logs(a.repo))
S = (lambda run: F.last5_dev(run)) if a.stat == "last5" else (lambda run: run["summary"]["best_dev_acc"])
out = []
def say(s=""):
    print(s); out.append(s)

say(f"P selection: K from {choice_f}; statistic = SST-2 dev accuracy ({'mean of the last 5 epochs' if a.stat == 'last5' else 'best epoch'}), "
    f"pre-Pt; threshold = min over plain LoRA on the original PZ, fine-tune seeds 0-2; rule {a.rule}; RoRA arm: fine-tune seed 0")
say("Columns: S | best dev (epoch) | test acc (B) | ASR (B) | ASR (B) mean of last 5 epochs | test acc after Pt (C) | ASR after Pt (C)")
res = {}
for label in LABELS:
    if label not in CH:
        say(f"\n{label}: no W_pre,K in {choice_f}"); continue
    c = CH[label]; tag = f"wh{float(c['r']):g}"
    base = [F.pick(R, F.exp_name(s, c["pz"], epochs=a.epochs), a.epochs) for s in (0, 1, 2)]
    if any(b is None for b in base):
        say(f"\n{label}: baseline runs missing"); continue
    thr = min(S(b) for b in base)
    say(f"\n=== {label}  (W_pre,K: whiten r* = {c['r']}; {c['json']})")
    def row(name, run):
        sm = run["summary"]
        return (f"  {name:26s} {S(run):.4f} | {sm['best_dev_acc']:.4f} ({int(sm['best_epoch']):2d}) | {sm['pre-pt_test_acc']:.4f} | "
                f"{sm['pre-pt_ASR']:.3f} | {sm['asr_mean_last5']:.3f} | {sm.get('pt_test_acc', float('nan')):.4f} | {sm.get('pt_ASR', float('nan')):.3f}")
    for s, b in enumerate(base):
        say(row(f"plain LoRA, orig PZ, s{s}", b))
    say(f"  threshold (min of the 3) = {thr:.4f}")
    runs = {}
    for p in PS:
        run = F.pick(R, F.exp_name(0, c["pz"], tag, p if p > 0 else None, tr=True, epochs=a.epochs), a.epochs)
        runs[p] = run
        if run is None:
            say(f"  {'p = %g' % p:26s} (not run yet)"); continue
        ok = S(run) >= thr
        collapse = run["summary"].get("pt_test_acc", 1.0) < 0.6
        say(row(("Tr only (p = 0)" if p == 0 else f"Cl p = {p:g} + Tr") + (" PASS" if ok else " fail"), run)
            + ("   <- Pt collapsed (test accuracy near chance)" if collapse else ""))
    done = all(runs[p] is not None for p in CAND)
    passes = {p: runs[p] is not None and S(runs[p]) >= thr for p in PS}
    P_step = None
    for p in CAND:
        if runs[p] is None or not passes[p]:
            break
        P_step = p
    P_any = max([p for p in CAND if passes[p]], default=None)
    P = P_step if a.rule == "stepwise" else P_any
    if runs[0.0] is not None and not passes[0.0]:
        say("  NOTE: Tr alone (p = 0) already fails the constraint, so the constraint is broken by K + Tr, not by Cl")
    say(f"  P: stepwise {P_step if P_step is not None else 'none'}; any {P_any if P_any is not None else 'none'}"
        + ("" if done else "  (grid incomplete: not final)"))
    res[label] = dict(c=c, base=base, thr=thr, runs=runs, P=P, done=done)

if res:
    os.makedirs(os.path.join(a.repo, "logs", "psel"), exist_ok=True)
    cf = os.path.join(a.repo, "logs", "psel", f"choice_P_{a.level}_{a.walk}_{a.stat}_{a.rule}.tsv")
    with open(cf, "w") as f:
        f.write("label\tr\tjson\tpz\tP\n")
        for label, d in res.items():
            if d["done"] and d["P"] is not None:
                f.write(f"{label}\t{d['c']['r']}\t{d['c']['json']}\t{d['c']['pz']}\t{d['P']:g}\n")
    say(f"\nwrote {cf} (only PZs whose grid is complete and that have a P)")
open("psel_numbers.txt", "w").write("\n".join(out) + "\n")
if a.no_plot or not res:
    sys.exit(0)

# ---------------- figures ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
ARM, BASE = "#2a78d6", "#7a7a7a"
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 9, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
def style(ax):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

labs = list(res)
fig, axes = plt.subplots(3, len(labs), figsize=(3.3 * len(labs), 9.2), facecolor=SURF, squeeze=False)
for j, label in enumerate(labs):
    d = res[label]; ps = [p for p in PS if d["runs"][p] is not None]
    # row 1: the selection statistic
    ax = axes[0, j]; style(ax)
    bv = [S(b) for b in d["base"]]
    ax.axhspan(min(bv), max(bv), color=BASE, alpha=0.18, lw=0)
    ax.axhline(d["thr"], color=BASE, lw=1.2, ls=":")
    ax.plot(ps, [S(d["runs"][p]) for p in ps], color=ARM, lw=2, marker="o", ms=5)
    if d["P"] is not None:
        ax.axvline(d["P"], color=INK2, lw=1, ls="--")
    ax.set_title(f"{label}（r* = {d['c']['r']}）" + ("" if d["done"] else "，p 格子未跑完"), fontsize=9.5)
    if j == 0:
        ax.set_ylabel("SST-2 dev 正確率（選 P 用）")
    # rows 2-3: not used for selection
    for i, (kb, kc, yl) in enumerate((("pre-pt_test_acc", "pt_test_acc", "SST-2 test 正確率（不用來選）"),
                                      ("pre-pt_ASR", "pt_ASR", "ASR（不用來選）")), start=1):
        ax = axes[i, j]; style(ax)
        for k, ls, lab in ((kb, "-", "類型 (B) Pt 前"), (kc, "--", "類型 (C) Pt 後")):
            vals = [d["base"][s]["summary"][k] for s in range(3)]
            ax.axhspan(min(vals), max(vals), color=BASE, alpha=0.14 if ls == "-" else 0.08, lw=0)
            ax.plot(ps, [d["runs"][p]["summary"][k] for p in ps], color=ARM, lw=1.8, ls=ls, marker="o", ms=4, label=lab)
        if d["P"] is not None:
            ax.axvline(d["P"], color=INK2, lw=1, ls="--")
        ax.set_ylim(-0.03 if i == 2 else 0.45, 1.03)
        if j == 0:
            ax.set_ylabel(yl)
        if i == 2:
            ax.set_xlabel("Cl 的 p（0 = 只有 Tr）")
    for i in range(3):
        axes[i, j].set_xticks(PS); axes[i, j].set_xticklabels([f"{p:g}" for p in PS])
axes[1, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
fig.suptitle(f"P 的選擇：每顆 PZ 在自己的 W_pre,K（換基底，r* 見各格標題）上做 RoRA（Cl 的 p + Tr λ=10 + Pt），微調 seed 0；"
             f"灰帶 = 原本 PZ 的普通 LoRA 三個 seed（0–2）的最小到最大", fontsize=9.5, x=0.01, ha="left")
fig.text(0.01, 0.005, f"第一列：選 P 用的量（{'最後 5 個 epoch 的平均' if a.stat == 'last5' else '最佳 epoch'}），點線 = 門檻（三個 seed 的最小值），"
         f"直虛線 = 選到的 P（規則 {a.rule}）。第二、三列：存下的最佳 dev epoch 的 test 正確率和 ASR，實線 = 類型 (B) 微調後 Pt 前，"
         f"虛線 = 類型 (C) Pt 後；淺／更淺灰帶 = 普通 LoRA 三個 seed 的 (B)／(C) 範圍。ASR = 插了 trigger 的 SST-2 test 正面句被判成負面的比例。",
         fontsize=7.5, color=INK2, wrap=True)
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig("psel.png", dpi=150, facecolor=SURF); plt.close(fig)

# per-epoch dev accuracy: every p (ordinal blue ramp, light = small p) and the 3 baseline seeds (gray)
RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
fig, axes = plt.subplots(1, len(labs), figsize=(3.3 * len(labs), 3.8), facecolor=SURF, squeeze=False, sharey=True)
for j, label in enumerate(labs):
    ax = axes[0, j]; style(ax); d = res[label]
    ax.axvspan(a.epochs - 5 - 0.5, a.epochs - 0.5, color=GRID, alpha=0.5, lw=0)
    for s, b in enumerate(d["base"]):
        ax.plot(F.dev_curve(b), color=BASE, lw=1, alpha=0.8, label="普通 LoRA，原本 PZ（seed 0–2）" if s == 0 else None)
    for i, p in enumerate(PS):
        if d["runs"][p] is not None:
            ax.plot(F.dev_curve(d["runs"][p]), color=RAMP[i % len(RAMP)], lw=1.4, label=f"p = {p:g}" + ("（只有 Tr）" if p == 0 else ""))
    ax.set_title(label, fontsize=9.5); ax.set_xlabel("epoch")
    if j == 0:
        ax.set_ylabel("SST-2 dev 正確率（Pt 前）")
axes[0, -1].legend(fontsize=7, frameon=False, loc="lower right")
fig.suptitle("每個 epoch 的 SST-2 dev 正確率：RoRA 在 W_pre,K 上（藍，越深 p 越大，微調 seed 0）與原本 PZ 的普通 LoRA（灰，三個 seed）；"
             "淺底 = 最後 5 個 epoch", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("psel_epochs.png", dpi=150, facecolor=SURF); plt.close(fig)
print("wrote psel.png, psel_epochs.png, psel_numbers.txt")