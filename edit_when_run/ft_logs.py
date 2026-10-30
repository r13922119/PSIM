"""ft_logs.py — read variant_finetune.py runs out of logs/*.log, by experiment name (no GPU, numpy only).
Used by psel_analyze.py and final_analyze.py.

exp_name(...) rebuilds variant_finetune.py's experiment name for the configurations used here (BadNet, RoBERTa, SST-2,
LoRA r = 8, alpha 16, lr 2e-4 linear, weight decay 0.01), so a run is found by WHAT it is, not by which log it sits in.
runs(paths) -> {name: [run, ...]} in order of appearance; a run is
  {"name", "log", "title" (the ===== header), "epochs": {e: (dev, test, asr)}, "summary": {field: value} or None}
pick(R, name) -> the last COMPLETED occurrence (has a SUMMARY line); raises if two completed occurrences disagree
  (same name and seed must give the same numbers under protocol v2).
Per epoch: dev = SST-2 dev accuracy, pre-Pt, Cl off (eval mode); test, asr = SST-2 test accuracy and ASR (type B: after
fine-tuning, before Pt) of that epoch. SUMMARY: best_epoch / best_dev_acc (the saved checkpoint), pre-pt_test_acc /
pre-pt_ASR (type B, at the best epoch), pt_test_acc / pt_ASR (type C, after Pt on the best-epoch checkpoint).
"""
import glob, os, re

NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf)"
RE_RUN = re.compile(r"Running experiment: (\S+)")
RE_EP = re.compile(r"\[EPOCH\] epoch=(\d+) dev=(%s) test clean accuracy=(%s) ASR=(%s)" % (NUM, NUM, NUM))
RE_SUM = re.compile(r"SUMMARY \| protocol=.*")
RE_HDR = re.compile(r"^===== (.*) =====\s*$")


def exp_name(seed, pz_path, wtag=None, p=None, tr=False, pt=True, epochs=20, lam=10, form="coord", trmode="penalty"):
    """variant_finetune.py's name: lora[_wk<tag>][_cl<p> | _clsvd<p>][_tr<lam>_kauto | _trproj_kauto][_pt]_lr2e-04_r8_a16_seed<s>_ep<E>_wd0.01_pz-<dir>
    trmode "penalty" = Tr as the soft penalty (Eq. 10, --use_orthogonal_penalty); "proj" = Tr as the exact constraint (Eq. 9, --tr_proj)"""
    tags = []
    if wtag:
        tags.append(f"wk{wtag}")
    if p:
        tags.append(("clsvd" if form == "svd" else "cl") + f"{float(p)}")
    if tr:
        tags.append((f"tr{lam:g}_k" if trmode == "penalty" else "trproj_k") + ("auto" if wtag else "32"))
    if pt:
        tags.append("pt")
    method = "lora" + ("_" + "_".join(tags) if tags else "")
    pz = os.path.basename(os.path.dirname(pz_path))
    return f"roberta_badnet_sst-2_{method}_lr2e-04_r8_a16_seed{seed}_ep{epochs}_wd0.01_pz-{pz}"


def runs(paths):
    R, cur, title = {}, None, ""
    for p in paths:
        for raw in open(p, errors="replace"):
            for line in raw.replace("\r", "\n").split("\n"):
                if (m := RE_HDR.match(line)):
                    title = m.group(1); continue
                if (m := RE_RUN.search(line)):
                    cur = dict(name=m.group(1), log=p, title=title, epochs={}, summary=None)
                    R.setdefault(cur["name"], []).append(cur); continue
                if cur is None:
                    continue
                if (m := RE_EP.search(line)):
                    cur["epochs"][int(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4)); continue
                if (m := RE_SUM.search(line)):
                    d = {}
                    for part in m.group(0).split(" | ")[1:]:
                        if "=" in part:
                            k, v = part.split("=", 1)
                            try:
                                d[k.strip()] = float(v)
                            except ValueError:
                                d[k.strip()] = v.strip()
                    cur["summary"] = d; cur = None
    return R


def all_logs(repo="."):
    return sorted(glob.glob(os.path.join(repo, "logs", "*.log")))


def pick(R, name, n_epochs=20):
    done = [r for r in R.get(name, []) if r["summary"] is not None and len(r["epochs"]) == n_epochs]
    if not done:
        return None
    key = lambda r: (tuple(sorted(r["epochs"].items())), {k: v for k, v in r["summary"].items() if k != "duration_min"})
    for r in done[:-1]:
        if key(r) != key(done[-1]):
            raise ValueError(f"{name}: completed twice with different numbers ({r['log']} vs {done[-1]['log']})")
    return done[-1]


def dev_curve(run):
    return [run["epochs"][e][0] for e in sorted(run["epochs"])]


def last5_dev(run):
    d = dev_curve(run)
    return sum(d[-5:]) / len(d[-5:])