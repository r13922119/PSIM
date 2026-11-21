import sys, os
sys.argv = ["x", "--repo", sys.argv[1], "--out", "/tmp/claude-0/-home-claude/4ca74893-1a56-58c4-bf69-444b01a71e5e/scratchpad/pzepoch/_tmp"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck_figs_1109.py")).read()
exec(compile(src.split('out = open(')[0], "deck_figs_head", "exec"))
rows = [("noK", 0, None), ("noK", 0.2, "coord"), ("noK", 0.2, "svd")]
print("best-dev epoch (0-based) of each run, seeds 0,1,2")
for arm, p, form in rows:
    print(f"{arm} {p} {form or ''}".ljust(14) + "  ".join(f"{l}:" + ",".join(str(F.pick(R, nm(arm, s, l, p, form))["summary"]["best_epoch"]) for s in SEEDS) for l in PZ))
