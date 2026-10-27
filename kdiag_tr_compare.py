"""kdiag_tr_compare.py — 讀 logs/kdiag/*.npz（CPU，只用 numpy），回答兩件事：
1. Tr 檢查（輸入能量落在截斷後 W 輸入側空間之外的比例，48 個 module 的中位數）：
   (a) 同一個 r 下，哪一種排序最高？
   (b) 留下差不多的方向數時，哪一種最高？（只看 σ 和換基底，用對數 k 線性內插）
2. C 最大的特徵方向占輸入能量多少；只看 σ、r = 0.9999 時，被丟的方向裡最大的那一個占多少。
模型階段：未微調的 PZ 和乾淨模型；C = SST-2 train 的 Σxxᵀ（沒減平均）。
  python kdiag_tr_compare.py [repo 路徑]
"""
import glob, os, sys
import numpy as np
REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/r13922119/psim"
M = ("sigma", "vote_in", "vote_out", "whiten")
Z = {os.path.basename(f)[:-4]: np.load(f) for f in sorted(glob.glob(f"{REPO}/logs/kdiag/*.npz"))}
R = list(next(iter(Z.values()))["r_grid"])
print("(a) 同一個 r：每顆模型的中位數，依序 " + "/".join(M))
for i, r in enumerate(R):
    cells = []
    for l, z in Z.items():
        v = [np.median(z[f"outside_{m}"][i]) for m in M]
        cells.append(f"{l[-6:]} " + "/".join(f"{x:.3f}" for x in v) + f" 最高={M[int(np.argmax(v))]}")
    print(f"  r={r:<6} " + "；".join(cells))
print("\n(b) 留下差不多的方向數：5 顆 PZ 合併的中位數，只看 σ vs 換基底")
pz = [l for l in Z if not l.startswith("clean")]
def curve(m):
    xs = np.array([np.mean([Z[l][f"k_{m}"][i].mean() for l in pz]) for i in range(len(R))])
    ys = np.array([np.median(np.concatenate([Z[l][f"outside_{m}"][i] for l in pz])) for i in range(len(R))])
    o = np.argsort(xs); return np.log(xs[o]), ys[o]
xs_s, ys_s = curve("sigma"); xs_w, ys_w = curve("whiten")
for k in (100, 200, 300, 400, 540, 700, 800, 900):
    print(f"  留約 {k:4d} 個方向：只看 σ {np.interp(np.log(k), xs_s, ys_s):.2f}   換基底 {np.interp(np.log(k), xs_w, ys_w):.2f}")
print("\n2. 輸入能量的集中程度")
for l, z in Z.items():
    a, lam = z["a"], z["lam"]; k = z["k_sigma"][R.index(0.9999)]
    top1 = np.array([a[j, k[j]:].max() / a[j].sum() for j in range(len(k))]); lt = lam[:, 0] / lam.sum(1)
    print(f"  {l:17s} C 最大特徵方向占比：24 層中位數 {np.median(lt):.2f}（{lt.min():.2f}–{lt.max():.2f}）；"
          f"只看 σ、r = 0.9999 被丟方向中最大一個的占比：48 module 中位數 {np.median(top1):.3f}")