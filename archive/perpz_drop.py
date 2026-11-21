import sys, os, io, contextlib
sys.argv = ["x", "--repo", sys.argv[1], "--out", "/tmp/claude-0/-home-claude/4ca74893-1a56-58c4-bf69-444b01a71e5e/scratchpad/pzepoch/_tmp"]
os.makedirs(sys.argv[-1], exist_ok=True)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck_figs_1109.py")).read()
head = src.split('out = open(')[0]
exec(compile(head, "deck_figs_head", "exec"))
import numpy as np
rows = [("noK", 0, None)] + [("noK", p, "coord") for p in (0.1, 0.2)] + [("noK", p, "svd") for p in (0.1, 0.2, 0.3)]
print("persistence %, mean of 3 seeds (n not collapsed); test cost vs plain, same PZ & seed")
print("arm/p/form".ljust(20) + "".join(l.rjust(16) for l in PZ))
for arm, p, form in rows:
    line = f"{arm} {p} {form}".ljust(20)
    for l in PZ:
        ps, cs = [], []
        for s in SEEDS:
            g = get(arm, s, l, p, form); b = get("noK", s, l, 0, None)
            if g is None or g["coll"]:
                continue
            ps.append(g["pers"]); cs.append(b["test"] - g["test"])
        line += f"{np.mean(ps):6.1f}({len(ps)}) c{np.mean(cs):+.2f}".rjust(16) if ps else "—".rjust(16)
    print(line)
