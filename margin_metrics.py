#!/usr/bin/env python3
"""margin_metrics.py — the numpy / stdlib half of the paired margin decomposition (no torch, no GPU, no model).

Used by margin_eval.py (which does the forward passes on the workstation), plot_test_score_hist.py and test_margin.py.
Nothing here reads a checkpoint or writes outside the folder it is given.

Question the decomposition answers: after fine-tuning, W_pre dropout lowers SST-2 test ASR. Is that because the trigger
moves a sentence's score less (smaller pull), or because the clean score of the same sentence is larger, or because the
decision boundary moved? ASR alone cannot tell these apart.

Definitions (label convention of this repo: 1 = positive, 0 = negative; the backdoor target is 0)
  logits        the two outputs of the classification head, model.eval(), before any Pt rescaling
  m(x)          logit_1 - logit_0.  The model predicts positive when m(x) > 0 (argmax takes index 0 on a tie, so a tie is negative).
  m(x + mn)     the same sentence with the trigger inserted by attack_utils.insert_trigger at the fixed placement
                (random.Random(PLACEMENT_SEED = 1234), one draw per positive sentence, in file order)
  s(x)          m(x) - m(x + mn): the pull of the trigger. s > 0 = the trigger pushes the sentence toward negative (the target).
                The attack succeeds on x  <=>  m(x + mn) <= 0  <=>  s(x) >= m(x).
  clean_acc     (#positive with m > 0 + #negative with m <= 0) / all clean test sentences      (= test clean accuracy in the logs)
  pos_err       share of clean positive sentences with m(x) <= 0
  fpr           share of clean negative sentences with m(x) > 0                                (= neg_err in bias_probe.py)
  asr_standard  share of ALL triggered positive sentences with m(x + mn) <= 0                  (= ASR in the logs, attack_utils.compute_asr)
  asr_clean_correct   P(m(x + mn) <= 0 | m(x) > 0): the same, only over sentences the model gets right without the trigger,
                so an already-wrong sentence is not counted as a flip caused by the trigger
  s/m           only where m(x) > 0 (the decision boundary is the only threshold used); NaN elsewhere, never inf.
                Reported as quantiles and as the share with s/m >= 1 (which is asr_clean_correct again), never as a mean.
  median_ratio  median(s) / median(m) over all positive sentences (no per-sentence division); NaN if median(m) <= 0
  clean_auc     P(m of a clean positive > m of a clean negative), all pairs, ties 1/2          (as plot_dev_score_hist.auc)
  trigger_auc_allpairs   P(m(x) > m(x' + mn)) over ALL pairs of positive sentences x, x', ties 1/2   (what plot_dev_score_hist.py draws)
  p_s_gt0_paired         share of sentences whose OWN score drops when the trigger is inserted, s(x) > 0   (a different quantity)
Quantiles are numpy's default (linear interpolation) at 10, 25, 50, 75, 90 %.

Counterfactual arithmetic (same PZ, same fine-tune seed, same sentences; a = plain LoRA, b = a dropout arm):
  cell(m from X, s from Y) = share of sentences with m_X(x) - s_Y(x) <= 0
  four cells: (a, a) = plain's own ASR, (b, b) = the dropout run's own ASR, (b, a) = only the clean margin replaced,
  (a, b) = only the pull replaced. This is bookkeeping on one set of sentences, not a causal statement, and the two runs
  may have different logit scales: read it next to median_ratio.

Aggregation: checkpoint numbers are averaged over the fine-tune seeds INSIDE each PZ, and the 5 PZ values are listed.
The sentences are never pooled across seeds or PZs, and 15 runs are never treated as 15 independent models.
"""
import csv, glob, hashlib, json, os, random, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_logs as F   # the repo's own experiment-name builder and log reader (stdlib only)

TRIGGER = "mn"
ARMS = ("plain", "cl0.2", "clsvd0.2")          # fine-tuned arms of this experiment (no clsvd0.3, no truncated arms)
DROPOUT_ARMS = ("cl0.2", "clsvd0.2")
PZ_ARM = "pz"                                  # the PZ itself before fine-tuning: its own head, no LoRA update
QS = (10, 25, 50, 75, 90)
RE_LABEL = re.compile(r"^n(\d+)_s(\d+)_ep(\d+)$")

SUMMARY_FILES = ("summary_checkpoints.csv", "summary_pz.csv", "counterfactual_checkpoints.csv", "counterfactual_pz.csv")
# ^ written by write_summaries; margin_eval.py calls it only after every configuration passed its check
KEYS_POS = ("idx_pos", "insert_pos", "logit0_pos_clean", "logit1_pos_clean", "m_pos_clean",
            "logit0_pos_trig", "logit1_pos_trig", "m_pos_trig", "s_pos")
KEYS_NEG = ("idx_neg", "logit0_neg_clean", "logit1_neg_clean", "m_neg_clean")


# ---------------------------------------------------------------- small helpers
def fmt4(x):
    """4 decimals, the precision variant_finetune.py prints last_test_acc and last_ASR with."""
    return f"{float(x):.4f}"


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_reference(path):
    """(down_pos, down_neg, down_asr) of a stored PZ score file: SST-2 test, clean positive / clean negative / triggered positive."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("down_pos", "down_neg", "down_asr") if k not in z.files]
        if missing:
            raise ValueError(f"{path}: no {missing}")
        return z["down_pos"], z["down_neg"], z["down_asr"]


def file_sha256(path, chunk=1 << 22):
    """same as repro_utils.file_sha256 (kept here so --dry and the tests need no torch)"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def auc(p, n):
    """P(p > n) + 0.5 P(p == n) over all len(p) x len(n) pairs."""
    p, n = np.asarray(p, dtype=np.float64), np.asarray(n, dtype=np.float64)
    d = p[:, None] - n[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def pull(m_pos, m_trig):
    """s(x) = m(x) - m(x + trigger), in float64."""
    return np.asarray(m_pos, dtype=np.float64) - np.asarray(m_trig, dtype=np.float64)


def ratio_s_over_m(m_pos, s):
    """s/m where m(x) > 0, NaN everywhere else. Never inf: a value is produced only where the denominator is > 0."""
    m, s = np.asarray(m_pos, dtype=np.float64), np.asarray(s, dtype=np.float64)
    r = np.full(m.shape, np.nan)
    ok = m > 0
    r[ok] = s[ok] / m[ok]
    return r


def quantiles(x, prefix):
    """{prefix_q10: ..., ...} over the finite values of x; NaN when there is none (no warning, no crash)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_q{q}": float("nan") for q in QS}
    v = np.quantile(x, [q / 100 for q in QS])
    return {f"{prefix}_q{q}": float(v[i]) for i, q in enumerate(QS)}


# ---------------------------------------------------------------- one checkpoint
def checkpoint_metrics(m_pos, m_trig, m_neg):
    """Every number of one checkpoint from its three score vectors:
    m_pos  clean positive sentences;  m_trig  the SAME sentences, same order, with the trigger;  m_neg  clean negative sentences."""
    m_pos, m_trig, m_neg = (np.asarray(a, dtype=np.float64) for a in (m_pos, m_trig, m_neg))
    if m_pos.ndim != 1 or m_pos.shape != m_trig.shape or m_pos.size == 0 or m_neg.ndim != 1 or m_neg.size == 0:
        raise ValueError(f"need paired 1-d positive scores and 1-d negative scores, got {m_pos.shape} {m_trig.shape} {m_neg.shape}")
    if not (np.isfinite(m_pos).all() and np.isfinite(m_trig).all() and np.isfinite(m_neg).all()):
        raise ValueError("non-finite score")
    s = pull(m_pos, m_trig)
    cc = m_pos > 0                       # clean-correct positive sentences
    n_cc = int(cc.sum())
    out = {
        "n_pos": int(m_pos.size), "n_neg": int(m_neg.size),
        "clean_acc": float((n_cc + int((m_neg <= 0).sum())) / (m_pos.size + m_neg.size)),
        "pos_err": float((m_pos <= 0).mean()),
        "fpr": float((m_neg > 0).mean()),
        "asr_standard": float((m_trig <= 0).mean()),
        "n_clean_correct": n_cc,
        "asr_clean_correct": float((m_trig[cc] <= 0).mean()) if n_cc else float("nan"),
        "clean_auc": auc(m_pos, m_neg),
        "trigger_auc_allpairs": auc(m_pos, m_trig),
        "p_s_gt0_paired": float((s > 0).mean()),
    }
    out.update(quantiles(m_pos, "m"))
    out.update(quantiles(m_trig, "mtrig"))
    out.update(quantiles(s, "s"))
    out.update(quantiles(m_neg, "mneg"))
    r = ratio_s_over_m(m_pos, s)
    out.update(quantiles(r, "ratio"))                      # NaN entries (m <= 0) are left out, not averaged
    out["ratio_ge1_share"] = float((r[cc] >= 1).mean()) if n_cc else float("nan")
    med_m = float(np.median(m_pos))
    out["median_ratio"] = float(np.median(s) / med_m) if med_m > 0 else float("nan")
    return out


# ---------------------------------------------------------------- counterfactual arithmetic for one (PZ, seed)
def counterfactual(m_a, mtrig_a, m_b, mtrig_b):
    """a = plain LoRA, b = a dropout arm, same sentences in the same order. See the module docstring."""
    m_a, mtrig_a, m_b, mtrig_b = (np.asarray(x, dtype=np.float64) for x in (m_a, mtrig_a, m_b, mtrig_b))
    if not (m_a.shape == mtrig_a.shape == m_b.shape == mtrig_b.shape) or m_a.ndim != 1 or m_a.size == 0:
        raise ValueError("the four score vectors must be paired 1-d arrays of the same length")
    s_a, s_b = m_a - mtrig_a, m_b - mtrig_b
    aa = float((mtrig_a <= 0).mean())                 # own cells use the measured m(x + mn) directly: no arithmetic, no rounding
    bb = float((mtrig_b <= 0).mean())
    ba = float(((m_b - s_a) <= 0).mean())             # clean margin from the dropout run, pull from plain
    ab = float(((m_a - s_b) <= 0).mean())             # clean margin from plain, pull from the dropout run
    return {
        "asr_plain": aa, "asr_drop": bb, "asr_m_drop_s_plain": ba, "asr_m_plain_s_drop": ab,
        "d_total": bb - aa,
        "d_m_first": ba - aa, "d_s_after_m": bb - ba,       # order 1: replace the clean margin first, then the pull
        "d_s_first": ab - aa, "d_m_after_s": bb - ab,       # order 2: replace the pull first, then the clean margin
        "interaction": bb - ba - ab + aa,
        "median_m_plain": float(np.median(m_a)), "median_m_drop": float(np.median(m_b)),
        "median_s_plain": float(np.median(s_a)), "median_s_drop": float(np.median(s_b)),
    }


# ---------------------------------------------------------------- aggregation: seeds inside a PZ, then the PZs are listed
def aggregate_by_pz(rows, spread=()):
    """rows: dicts with "arm", "pz", "seed" and numeric fields. Returns one row per (arm, PZ): the mean over that PZ's
    seeds of every numeric field, plus n_seeds, seeds, and <field>_seed_min / _seed_max for the fields in `spread`.
    Every (arm, PZ) of the fine-tuned arms must have the same seed set (project rule: same seeds, same PZ composition)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["arm"], r["pz"]), []).append(r)
    seedsets = {k: tuple(sorted(str(r["seed"]) for r in v)) for k, v in groups.items() if k[0] != PZ_ARM}
    if len(set(seedsets.values())) > 1:
        raise ValueError(f"unequal seed sets across (arm, PZ): {sorted(set(seedsets.values()))}; refuse to average")
    out = []
    for (arm, pz), v in sorted(groups.items(), key=lambda kv: (_arm_order(kv[0][0]), kv[0][1])):
        row = {"arm": arm, "pz": pz, "n_seeds": len(v), "seeds": " ".join(sorted(str(r["seed"]) for r in v))}
        for k in v[0]:
            if k in ("arm", "pz", "seed", "name", "n_pos", "n_neg") or isinstance(v[0][k], str):
                continue
            vals = np.array([float(r[k]) for r in v], dtype=np.float64)
            row[k] = float(vals.mean())            # a NaN in any seed stays NaN: nothing is silently dropped
            if k in spread:
                row[f"{k}_seed_min"], row[f"{k}_seed_max"] = float(vals.min()), float(vals.max())
        out.append(row)
    return out


def _arm_order(arm):
    order = (PZ_ARM,) + ARMS
    return order.index(arm) if arm in order else len(order)


# ---------------------------------------------------------------- per-sentence files
def save_scores(path, meta, idx_pos, insert_pos, logits_pos_clean, logits_pos_trig, idx_neg, logits_neg_clean):
    """One .npz per configuration. logits_*: (n, 2) float32 as the model produced them (column 0 = negative, 1 = positive).
    m is logit_1 - logit_0 in float32 (as bias_probe._acc_ppos computes scores); s is in float64."""
    lp, lt, ln = (np.asarray(a, dtype=np.float32) for a in (logits_pos_clean, logits_pos_trig, logits_neg_clean))
    idx_pos, insert_pos, idx_neg = (np.asarray(a, dtype=np.int64) for a in (idx_pos, insert_pos, idx_neg))
    if lp.shape != lt.shape or lp.ndim != 2 or lp.shape[1] != 2 or ln.ndim != 2 or ln.shape[1] != 2:
        raise ValueError(f"logits must be (n, 2): {lp.shape} {lt.shape} {ln.shape}")
    if len(idx_pos) != len(lp) or len(insert_pos) != len(lp) or len(idx_neg) != len(ln):
        raise ValueError("idx / insert_pos lengths do not match the logits")
    m_pos, m_trig, m_neg = lp[:, 1] - lp[:, 0], lt[:, 1] - lt[:, 0], ln[:, 1] - ln[:, 0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path, meta=np.array(json.dumps(meta, sort_keys=True)),
        idx_pos=idx_pos, insert_pos=insert_pos,
        logit0_pos_clean=lp[:, 0], logit1_pos_clean=lp[:, 1], m_pos_clean=m_pos,
        logit0_pos_trig=lt[:, 0], logit1_pos_trig=lt[:, 1], m_pos_trig=m_trig, s_pos=pull(m_pos, m_trig),
        idx_neg=idx_neg, logit0_neg_clean=ln[:, 0], logit1_neg_clean=ln[:, 1], m_neg_clean=m_neg)


def load_scores(path):
    """Reads one file written by save_scores and checks it against its own definitions."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("meta",) + KEYS_POS + KEYS_NEG if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        d = {k: z[k] for k in KEYS_POS + KEYS_NEG}
        d["meta"] = json.loads(str(z["meta"]))
    n_pos, n_neg = len(d["idx_pos"]), len(d["idx_neg"])
    if any(len(d[k]) != n_pos for k in KEYS_POS) or any(len(d[k]) != n_neg for k in KEYS_NEG):
        raise ValueError(f"{path}: arrays of different lengths")
    for grp in ("pos_clean", "pos_trig", "neg_clean"):
        if not np.array_equal(d[f"m_{grp}"], d[f"logit1_{grp}"] - d[f"logit0_{grp}"]):
            raise ValueError(f"{path}: m_{grp} is not logit_1 - logit_0")
    if not np.array_equal(d["s_pos"], pull(d["m_pos_clean"], d["m_pos_trig"])):
        raise ValueError(f"{path}: s_pos is not m(x) - m(x + trigger)")
    return d


# ---------------------------------------------------------------- summaries (the csv files)
SPREAD = ("clean_acc", "pos_err", "fpr", "asr_standard", "asr_clean_correct", "m_q50", "mtrig_q50", "s_q50", "median_ratio")
CF_SPREAD = ("asr_plain", "asr_drop", "asr_m_drop_s_plain", "asr_m_plain_s_drop")


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"nothing to write to {path}")
    fields = list(rows[0].keys())
    for r in rows:
        if list(r.keys()) != fields:
            raise ValueError(f"{path}: rows with different columns")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_summaries(npz_dir, out_dir):
    """Reads every <npz_dir>/*.npz and writes into out_dir:
      summary_checkpoints.csv        one row per configuration (each fine-tuned checkpoint, each PZ before fine-tuning)
      summary_pz.csv                 seeds averaged inside each PZ; one row per (arm, PZ); the 5 PZ values are listed, not pooled
      counterfactual_checkpoints.csv one row per (PZ, seed, dropout arm): the four cells against plain LoRA of the same PZ and seed
      counterfactual_pz.csv          the same, seeds averaged inside each PZ
    Returns the four lists of rows. Shares are fractions (0-1), not percent."""
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npz in {npz_dir}")
    data, ck = {}, []
    for p in files:
        d = load_scores(p)
        mt = d["meta"]
        key = (mt["arm"], mt["pz"], mt["seed"] if mt["seed"] is not None else "-")
        if key in data:
            raise ValueError(f"two files for {key}: {p}")
        data[key] = d
        row = {"arm": key[0], "pz": key[1], "seed": key[2], "name": mt["name"]}
        row.update(checkpoint_metrics(d["m_pos_clean"], d["m_pos_trig"], d["m_neg_clean"]))
        ck.append(row)
    ck.sort(key=lambda r: (_arm_order(r["arm"]), r["pz"], str(r["seed"])))
    pz = aggregate_by_pz(ck, spread=SPREAD)

    cf = []
    for (arm, lab, seed), d in sorted(data.items(), key=lambda kv: (_arm_order(kv[0][0]), kv[0][1], str(kv[0][2]))):
        if arm not in DROPOUT_ARMS:
            continue
        a = data.get(("plain", lab, seed))
        if a is None:
            raise ValueError(f"no plain LoRA run for PZ {lab} seed {seed} to pair {arm} with")
        if not (np.array_equal(a["idx_pos"], d["idx_pos"]) and np.array_equal(a["insert_pos"], d["insert_pos"])):
            raise ValueError(f"{arm} {lab} seed {seed}: sentences or trigger placement differ from the plain run")
        row = {"arm": arm, "pz": lab, "seed": seed, "name": d["meta"]["name"]}
        row.update(counterfactual(a["m_pos_clean"], a["m_pos_trig"], d["m_pos_clean"], d["m_pos_trig"]))
        cf.append(row)
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "summary_checkpoints.csv"), ck)
    _write_csv(os.path.join(out_dir, "summary_pz.csv"), pz)
    cf_pz = []
    if cf:
        cf_pz = aggregate_by_pz(cf, spread=CF_SPREAD)
        _write_csv(os.path.join(out_dir, "counterfactual_checkpoints.csv"), cf)
        _write_csv(os.path.join(out_dir, "counterfactual_pz.csv"), cf_pz)
    return ck, pz, cf, cf_pz


# ---------------------------------------------------------------- trigger placement table
def derive_insert_pos(words, new_words, trigger_words):
    """The word index at which trigger_words were inserted into words to give new_words (0 = before the first word)."""
    k = len(trigger_words)
    if len(new_words) != len(words) + k:
        raise ValueError("the triggered sentence is not the clean sentence plus the trigger")
    hits = [i for i in range(len(words) + 1)
            if new_words[:i] == words[:i] and new_words[i:i + k] == trigger_words and new_words[i + k:] == words[i:]]
    if len(hits) != 1:
        raise ValueError(f"insert position is not unique ({hits}): does the sentence already contain the trigger?")
    return hits[0]


def build_placement(rows, trigger, insert_fn, seed):
    """rows: dicts with idx, label, sentence, in FILE ORDER. Mirrors attack_utils.build_poisoned_test_dataloader:
    one rng = random.Random(seed), insert_fn(sentence, trigger, rng) called once per label == 1 row, in file order.
    insert_fn must be attack_utils.insert_trigger. Returns one dict per positive sentence."""
    rng, tw, out = random.Random(seed), trigger.split(), []
    for r in rows:
        if int(r["label"]) != 1:
            continue
        new = insert_fn(r["sentence"], trigger, rng)
        words = r["sentence"].split()
        out.append({"idx": int(r["idx"]), "n_words": len(words), "insert_pos": derive_insert_pos(words, new.split(), tw),
                    "sentence": r["sentence"], "sentence_trig": new})
    return out


# ---------------------------------------------------------------- sanity check 1: the PZ before fine-tuning vs the stored scores
def check_reference(m_pos, m_neg, m_trig, ref_pos, ref_neg, ref_trig, tol=1e-4):
    """Scores of the PZ itself against logs/scores/dose_n<n>_s<seed>/ep<epoch>.npz (down_pos, down_neg, down_asr).
    Passes only if, for all three groups, the largest |difference| <= tol AND no sentence's prediction (score > 0) changes."""
    ok, rep = True, {}
    for key, a, b in (("pos_clean", m_pos, ref_pos), ("neg_clean", m_neg, ref_neg), ("pos_trig", m_trig, ref_trig)):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            rep[key] = {"ok": False, "n_mine": int(a.size), "n_ref": int(b.size), "max_abs_diff": float("nan"), "pred_changed": -1}
            ok = False
            continue
        d, flips = float(np.abs(a - b).max()), int(((a > 0) != (b > 0)).sum())
        rep[key] = {"ok": bool(d <= tol and flips == 0), "n_mine": int(a.size), "n_ref": int(b.size), "max_abs_diff": d, "pred_changed": flips}
        ok = ok and rep[key]["ok"]
    return ok, rep


# ---------------------------------------------------------------- sanity check 2: names and the logged last-epoch numbers
def read_panel(choice_path):
    """logs/wpre/choice_<level>_<walk>.tsv -> [(label, n, pz_seed, pz_epoch, pz_path)] for the poisoned PZs (the clean row is skipped)."""
    out = []
    with open(choice_path) as f:
        rows = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    hdr = rows[0]
    if hdr[:1] != ["label"] or "pz" not in hdr:
        raise ValueError(f"{choice_path}: unexpected header {hdr}")
    for r in rows[1:]:
        label, pz = r[hdr.index("label")], r[hdr.index("pz")]
        if label.startswith("clean"):
            continue
        m = RE_LABEL.match(label)
        if not m:
            raise ValueError(f"{choice_path}: cannot read n, seed, epoch from label {label}")
        out.append((label, int(m.group(1)), int(m.group(2)), int(m.group(3)), pz))
    return out


def adapter_name(arm, seed, pz_path):
    """variant_finetune.py's experiment name, built by ft_logs.exp_name exactly as dw_last.name_of does."""
    if arm == "plain":
        return F.exp_name(seed, pz_path)
    if arm == "cl0.2":
        return F.exp_name(seed, pz_path, p=0.2, form="coord")
    if arm == "clsvd0.2":
        return F.exp_name(seed, pz_path, p=0.2, form="svd")
    raise ValueError(arm)


def logged_last(repo, names, n_epochs=20):
    """{name: {"last_test_acc": "0.xxxx", "last_ASR": "0.xxxx", "log": file, "n": occurrences used}} from logs/*.log.
    Uses only COMPLETED occurrences (all epochs + SUMMARY) whose SUMMARY carries last_test_acc and last_ASR (runs since
    10/29; the plain runs of 10/03 and 10/22 have no such fields and no last/ adapter). All such occurrences of a name must
    agree, and must equal the test accuracy and ASR printed for the last epoch. Raises with every problem listed."""
    R = F.runs(F.all_logs(repo))
    out, problems = {}, []
    for name in names:
        occ = [r for r in R.get(name, []) if r["summary"] is not None and len(r["epochs"]) == n_epochs
               and "last_test_acc" in r["summary"] and "last_ASR" in r["summary"]]
        if not occ:
            problems.append(f"{name}: no completed run with last_test_acc / last_ASR in logs/*.log")
            continue
        vals = {(fmt4(r["summary"]["last_test_acc"]), fmt4(r["summary"]["last_ASR"])) for r in occ}
        if len(vals) != 1:
            problems.append(f"{name}: completed runs disagree: {sorted(vals)}")
            continue
        acc, asr = vals.pop()
        ep = occ[-1]["epochs"][n_epochs - 1]
        if (fmt4(ep[1]), fmt4(ep[2])) != (acc, asr):
            problems.append(f"{name}: SUMMARY last_* ({acc}, {asr}) differs from the [EPOCH] line of the last epoch ({fmt4(ep[1])}, {fmt4(ep[2])})")
            continue
        out[name] = {"last_test_acc": acc, "last_ASR": asr, "log": os.path.basename(occ[-1]["log"]), "n": len(occ)}
    if problems:
        raise ValueError("\n".join(problems))
    return out


def check_logged(clean_correct, n_clean, trig_correct, n_trig, logged):
    """My counts against the log, at the log's own precision. test acc = correct / n; ASR = 1 - correct / n, the same
    arithmetic as variant_finetune.evaluate_accuracy and attack_utils.compute_asr. Returns (ok, mine_acc, mine_asr)."""
    acc, asr = fmt4(clean_correct / n_clean), fmt4(1.0 - (trig_correct / n_trig))
    return (acc == logged["last_test_acc"] and asr == logged["last_ASR"]), acc, asr
