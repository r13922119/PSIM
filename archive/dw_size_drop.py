import sys, os, re
sys.argv = ["x", "--repo", sys.argv[1], "--out", "/tmp/claude-0/-home-claude/4ca74893-1a56-58c4-bf69-444b01a71e5e/scratchpad/pzepoch/_tmp"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck_figs_1109.py")).read()
exec(compile(src.split('out = open(')[0], "deck_figs_head", "exec"))
import numpy as np
paths = [p for p in F.all_logs(a.repo) if "plain_last" not in os.path.basename(p)]
RE_S = re.compile(r"\[Mechanism 3\] (\S+): new scaling s = ([-+0-9.eE]+)")
S = {}
for p in paths:
    cur = None
    for raw in open(p, errors="replace"):
        for line in raw.replace("\r", "\n").split("\n"):
            m = F.RE_RUN.search(line)
            if m:
                cur = m.group(1); S.setdefault(cur, []).append([]); continue
            if cur is None: continue
            m = RE_S.search(line)
            if m: S[cur][-1].append(float(m.group(2))); continue
            if F.RE_SUM.search(line): cur = None
def rel(n):
    occ = [x for x in S.get(n, []) if len(x) == 6]
    return float(np.mean([2.0 / s for s in occ[-1]])) if occ else np.nan
rows = [("noK", 0, None)] + [("noK", p, "coord") for p in (0.1, 0.2)] + [("noK", p, "svd") for p in (0.1, 0.2, 0.3)] + [("KCl", 0, None)]
print("rel = sigma_max(dW)/sigma_max(W), Pt's 6 modules, best-dev adapter; mean of 3 seeds | best-dev pre-Pt test ASR")
print("arm/p/form".ljust(16) + "".join(l.rjust(15) for l in PZ))
for arm, p, form in rows:
    line = f"{arm} {p} {form or ''}".ljust(16)
    for l in PZ:
        rs, asr = [], []
        for s in SEEDS:
            n = nm(arm, s, l, p, form); r = F.pick(R, n)
            if r is None: continue
            rs.append(rel(n)); asr.append(float(r["summary"]["pre-pt_ASR"]))
        line += (f"{np.nanmean(rs):.3f} |{100*np.mean(asr):4.0f}%".rjust(15) if rs else "—".rjust(15))
    print(line)
