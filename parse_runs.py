#!/usr/bin/env python3
"""Compact view of variant_finetune.py / poisoned_pretrain.py logs (protocol v2), no LLM needed.

  python parse_runs.py run.log                 # one block per run (works on a multi-run log)
  python parse_runs.py run.log --all           # every epoch instead of 0,1,2,best,last
  python parse_runs.py a.log b.log --diff      # determinism check (ignores duration_min); compares dev, [RNG], [EPOCH] test/ASR
  python parse_runs.py run.log --csv           # one row per run: SUMMARY fields + cfg
"""
import re, sys, argparse

NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf)"
RE_DBG   = re.compile(r"\[DEBUG(?:_ep(\d+)_stp(\d+))?\] task_loss=(%s), penalty\(mean,scaled\)=(%s), lambda\*penalty=(%s), omega_A/\|\|A\|\|\^2 mean=(%s)(?:, omega_B/\|\|B\|\|\^2 mean=(%s))?" % (NUM,NUM,NUM,NUM,NUM))
RE_EPOCH = re.compile(r"^epoch (\d+)\s*$")
RE_DEV   = re.compile(r"dev clean acc:\s*(%s)" % NUM)
RE_HOOK  = re.compile(r"\[(?:DEBUG|CHECK)\] dropout_hook fired (\d+) times")      # v2 prints [CHECK]
RE_RNG   = re.compile(r"^\[RNG\] epoch=(\d+) fp=([0-9a-f]+)")
RE_EP    = re.compile(r"^\[EPOCH\] epoch=(\d+) dev=(%s) test clean accuracy=(%s) ASR=(%s)" % (NUM,NUM,NUM))
RE_SUM   = re.compile(r"^SUMMARY \|")
RE_WARN  = re.compile(r"\b(nan|inf)\b|Traceback|Error", re.I)
RE_HDR   = re.compile(r"^===== (.*) =====")
RE_ARGS  = re.compile(r"^(Namespace\(|\{'.*\}$)")

def parse(path):
    runs, cur = [], None
    def new(): return dict(title="", cfg="", hook="", dbg=[], dev={}, rng={}, obs={}, warns=[], summary="")
    cur = new()
    for raw in open(path, errors="replace"):
        for line in raw.replace("\r", "\n").split("\n"):
            line = line.strip()
            if not line: continue
            if (m := RE_HDR.match(line)):
                if cur["dbg"] or cur["summary"]: runs.append(cur); cur = new()
                cur["title"] = m.group(1); continue
            if RE_ARGS.match(line): cur["cfg"] = line; continue
            if (m := RE_RNG.match(line)): cur["rng"][int(m.group(1))] = m.group(2); continue
            if (m := RE_EP.match(line)):
                cur["obs"][int(m.group(1))] = (m.group(3), m.group(4))      # (test acc, ASR)
                if re.search(r"\b(nan|inf)\b", line, re.I) and "OBSERVE=0" not in cur["title"]: cur["warns"].append(line[:200])
                continue
            if (m := RE_DBG.search(line)):
                if re.search(r"\b(nan|inf)\b", line, re.I): cur["warns"].append(line[:200])
                g = m.groups(); e, st = g[0], g[1]
                if e is None: e, st = len({d[0] for d in cur["dbg"]}), 0   # old format: one record per epoch
                cur["dbg"].append((int(e), int(st), *g[2:])); continue
            if (m := RE_HOOK.search(line)): cur["hook"] = m.group(1); continue
            if (m := RE_EPOCH.match(line)): cur["_ep"] = int(m.group(1)); continue
            if (m := RE_DEV.search(line)): cur["dev"][cur.get("_ep", len(cur["dev"]))] = m.group(1); continue
            if RE_SUM.match(line):
                if re.search(r"\b(nan|inf)\b", line, re.I): cur["warns"].append(line[:200])
                cur["summary"] = line; runs.append(cur); cur = new(); continue
            if RE_WARN.search(line) and "HF_TOKEN" not in line and "unauthenticated" not in line:
                cur["warns"].append(line[:200])
    if cur["dbg"] or cur["dev"]: runs.append(cur)
    return runs

def fields(sm):
    return dict(p.strip().split("=", 1) if "=" in p else ("name", p.strip()) for p in sm.split("|")[1:])

def rows(run):
    out, last_step_of = [], {}
    for (e, st, *d) in sorted(run["dbg"], key=lambda x: (x[0], x[1])):
        last_step_of[e] = st
        out.append([e, st, "", *d])
    for r in out:
        if r[1] == last_step_of[r[0]]:
            r[2] = run["dev"].get(r[0], "-")
    have = {r[0] for r in out}
    for e in sorted(set(run["dev"]) - have):
        out.append([e, "-", run["dev"][e], *(None,) * 5])
    return sorted(out, key=lambda x: (x[0], -1 if x[1] == "-" else x[1]))

def show(run, all_epochs):
    f = fields(run["summary"]) if run["summary"] else {}
    print("run:", run["title"] or f.get("name", "?"))
    if run["cfg"]: print("cfg:", run["cfg"])
    r = rows(run); best = int(f["best_epoch"]) if "best_epoch" in f else -1
    eps = sorted({x[0] for x in r} | set(run["obs"]) | set(run["rng"])); last = eps[-1] if eps else -1
    keep = set(eps) if all_epochs else {0, 1, 2, best, last}
    print("epoch | step | dev | task_loss | penalty | lam*pen | A_ratio | B_ratio")
    for x in r:
        if x[0] in keep: print(" | ".join("-" if v is None else str(v) for v in x))
    if run["obs"] or run["rng"]:
        print("epoch | test_acc | ASR | rng_fp")
        for e in eps:
            if e in keep:
                t, a = run["obs"].get(e, ("-", "-"))
                print(f"{e} | {t} | {a} | {run['rng'].get(e, '-')}")
    print("hook:", run["hook"] or "MISSING (ok if Cl is off)")
    print("summary:", run["summary"] or "MISSING")
    if run["warns"]: print("warnings:", *run["warns"][:5], sep="\n  ")
    print()

def canon(run):
    f = fields(run["summary"]) if run["summary"] else {}
    f.pop("duration_min", None)
    return dict(summary=f, rows=rows(run), hook=run["hook"], rng=run["rng"], obs=run["obs"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("logs", nargs="+")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--diff", action="store_true"); ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()
    if a.diff:
        A, B = parse(a.logs[0]), parse(a.logs[1])
        if len(A) != len(B): print(f"run count differs: {len(A)} vs {len(B)}")
        bad = 0
        for i, (x, y) in enumerate(zip(A, B)):
            cx, cy = canon(x), canon(y)
            if cx == cy: continue
            bad += 1; print(f"run {i}: DIFFERENT")
            for k in cx["summary"].keys() | cy["summary"].keys():
                if cx["summary"].get(k) != cy["summary"].get(k): print(f"  summary.{k}: {cx['summary'].get(k)} vs {cy['summary'].get(k)}")
            for e in sorted(cx["rng"].keys() | cy["rng"].keys()):
                if cx["rng"].get(e) != cy["rng"].get(e):
                    print(f"  FIRST RNG divergence at epoch {e}: {cx['rng'].get(e)} vs {cy['rng'].get(e)}"); break
            for e in sorted(cx["obs"].keys() | cy["obs"].keys()):
                if cx["obs"].get(e) != cy["obs"].get(e):
                    print(f"  first test/ASR difference at epoch {e}: {cx['obs'].get(e)} vs {cy['obs'].get(e)}"); break
            for rx, ry in zip(cx["rows"], cy["rows"]):
                if rx != ry: print(f"  epoch {rx[0]}: {rx} vs {ry}"); break
        print("IDENTICAL (excluding duration_min)" if not bad and len(A) == len(B) else f"{bad} differing run(s)")
        return
    runs = [r for p in a.logs for r in parse(p)]
    if a.csv:
        import csv
        keys = ["protocol","name","best_epoch","best_dev_acc","pre-pt_test_acc","pre-pt_ASR","pt_test_acc","pt_ASR","asr_mean_last5","duration_min","gpu","observe"]
        w = csv.writer(sys.stdout); w.writerow(keys + ["cfg"])
        for r in runs:
            f = fields(r["summary"]) if r["summary"] else {}
            w.writerow([f.get(k, "") for k in keys] + [r["cfg"]])
        return
    for r in runs: show(r, a.all)
main()