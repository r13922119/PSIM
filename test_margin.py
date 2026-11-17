"""CPU test of margin_metrics.py and plot_test_score_hist.py (numpy + matplotlib only: no torch, no GPU, no model).
Run from the PSIM folder:   PYTHONNOUSERSITE=1 python test_margin.py

It reads (never writes) these repo files: data/sst-2/test.json, attack_utils.py and repro_utils.py (as text),
logs/*.log, logs/wpre/choice_bonf_stepwise.tsv, logs/scores/dose_n1500_s<k>/ep02.npz. Everything it writes goes to a
temporary folder that is removed at the end (--keep DIR copies the mock figures to DIR so they can be looked at).

Checks
  1. every metric on a 5 + 4 sentence example worked out by hand (ties, clean-wrong sentences, both trigger statistics, quantiles);
  2. s/m: NaN where m <= 0, never inf, no warning when nothing is clean-correct, non-finite input is refused;
  3. identities on random data: share(s/m >= 1) = asr_clean_correct, asr_standard = share(s >= m),
     the counterfactual's own cells = each run's own ASR, both orders of the decomposition add up to the total;
  4. the counterfactual on a 3 sentence example worked out by hand;
  5. aggregation: seeds are averaged inside a PZ, PZs are listed and never pooled, unequal seed sets are refused;
  6. per-sentence files: save, load, and refusal of a file that contradicts its own definitions;
  7. sanity check 1 (PZ before fine-tuning vs stored scores) on the 5 REAL stored files: identical passes, a shift inside the
     tolerance passes, a shift beyond it fails, a prediction flip inside the tolerance fails, a wrong length fails;
  8. the REAL stored PZ scores through the metric code give the numbers computed independently during the audit;
  9. sanity check 2 (adapter vs its training log) on the REAL logs: all 45 names are found with last_test_acc and last_ASR,
     a run that only exists in an older log (no last_* fields) is refused, one sentence off is caught;
 10. trigger placement on the REAL test file with the REAL attack_utils.insert_trigger (taken out of attack_utils.py as text,
     because importing that module needs torch): 909 sentences, positions equal an independent redraw with seed 1234;
 11. mock run: 5 PZs before fine-tuning + 45 checkpoints -> the four csv files have the expected rows and values;
 12. the plotting script on that mock run: refuses without sanity_checks.csv, with a failed check, without the summaries that
     only a finished evaluation writes, and with a score file missing; then writes 7 figures.
"""
import argparse, ast, contextlib, csv, io, json, math, os, random, re, shutil, sys, tempfile, warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import margin_metrics as M

NAN = float("nan")


def close(a, b, tol=1e-12):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def expect_raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc as e:
        return e
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def repo_insert_trigger():
    """attack_utils.insert_trigger, compiled from the file's own text (the module itself imports torch)."""
    src = read(os.path.join(HERE, "attack_utils.py"))
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "insert_trigger"]
    assert len(fn) == 1, "insert_trigger not found in attack_utils.py"
    ns = {"random": random}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "attack_utils.py", "exec"), ns)
    return ns["insert_trigger"]


def repo_placement_seed():
    m = re.search(r"^PLACEMENT_SEED\s*=\s*(\d+)", read(os.path.join(HERE, "repro_utils.py")), re.M)
    assert m, "PLACEMENT_SEED not found in repro_utils.py"
    return int(m.group(1))


def mock_run(root, rng):
    """5 PZs before fine-tuning + 3 arms x 5 PZs x 3 seeds, written with margin_metrics.save_scores.
    Built so that the dropout arms have a smaller pull than plain; returns the scores for independent recomputation."""
    n_pos, n_neg = 909, 912
    idx = np.arange(n_pos + n_neg)
    lab = np.array([i % 2 for i in range(n_pos + n_neg)]); lab[-3:] = 0        # 909 positive, 912 negative
    assert int(lab.sum()) == n_pos
    idx_pos, idx_neg = idx[lab == 1], idx[lab == 0]
    insert_pos = rng.integers(1, 12, size=n_pos)
    truth = {}

    def put(arm, label, seed, name, m_pos, pull_, m_neg):
        lp = np.stack([np.zeros(n_pos), m_pos], 1).astype(np.float32)
        lt = np.stack([np.zeros(n_pos), m_pos - pull_], 1).astype(np.float32)
        ln = np.stack([np.zeros(n_neg), m_neg], 1).astype(np.float32)
        M.save_scores(os.path.join(root, "npz", name + ".npz"), {"arm": arm, "pz": label, "seed": seed, "name": name},
                      idx_pos, insert_pos, lp, lt, idx_neg, ln)
        truth[(arm, label, seed)] = (lp[:, 1] - lp[:, 0], lt[:, 1] - lt[:, 0], ln[:, 1] - ln[:, 0])

    for k in range(5):
        label = f"n1500_s{k}_ep2"
        put(M.PZ_ARM, label, None, f"pz_{label}", rng.normal(4.5, 1.5, n_pos), rng.normal(12, 1.0, n_pos), rng.normal(-1.5, 3.0, n_neg))
        for arm, pull_mu in (("plain", 4.0 + 0.5 * k), ("cl0.2", 2.2), ("clsvd0.2", 2.5)):
            for seed in range(3):
                put(arm, label, seed, f"mock_{arm}_seed{seed}_pz-{label}",
                    rng.normal(3.5, 1.3, n_pos), rng.normal(pull_mu + 0.3 * seed, 1.6, n_pos), rng.normal(-3.5, 1.4, n_neg))
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None, help="copy the mock figures and csv files to this folder")
    a = ap.parse_args()
    warnings.simplefilter("error", RuntimeWarning)   # any numpy warning (all-NaN slice, empty mean, division by 0) fails the test
    rng = np.random.default_rng(20261117)
    tmp = tempfile.mkdtemp(prefix="test_margin_")

    # ---- 1. the hand example
    m_pos = np.array([2.0, 1.0, -1.0, 0.0, 3.0])
    m_trig = np.array([-1.0, 0.5, -2.0, -1.0, 0.0])
    m_neg = np.array([-2.0, 1.0, 0.0, -0.5])
    r = M.checkpoint_metrics(m_pos, m_trig, m_neg)
    want = {"n_pos": 5, "n_neg": 4, "clean_acc": 6 / 9, "pos_err": 2 / 5, "fpr": 1 / 4, "asr_standard": 4 / 5,
            "n_clean_correct": 3, "asr_clean_correct": 2 / 3, "clean_auc": 15 / 20, "trigger_auc_allpairs": 20.5 / 25,
            "p_s_gt0_paired": 1.0, "m_q50": 1.0, "mtrig_q50": -1.0, "s_q50": 1.0, "s_q10": 0.7, "s_q25": 1.0, "s_q90": 3.0,
            "mneg_q50": -0.25, "ratio_q50": 1.0, "ratio_ge1_share": 2 / 3, "median_ratio": 1.0}
    for k, v in want.items():
        assert close(r[k], v), (k, r[k], v)
    assert r["asr_clean_correct"] != r["asr_standard"], "the two ASRs must differ when a clean-wrong sentence exists"
    print("ok 1: hand example: clean_acc 6/9, pos_err 2/5, fpr 1/4, asr_standard 4/5, asr_clean_correct 2/3, "
          "clean AUC 15/20, all-pairs trigger AUC 20.5/25, paired P(s>0) 1, median_ratio 1; a tie (m = 0) counts as negative")

    # ---- 2. s/m
    ratio = M.ratio_s_over_m(m_pos, M.pull(m_pos, m_trig))
    assert np.array_equal(np.isnan(ratio), m_pos <= 0) and close(ratio[0], 1.5) and close(ratio[1], 0.5) and close(ratio[4], 1.0)
    tiny = M.ratio_s_over_m(np.array([1e-30, 1e-45, 0.0, -1e-30]), np.array([5.0, 5.0, 5.0, 5.0]))
    assert not np.isinf(tiny).any() and np.isfinite(tiny[:2]).all() and np.isnan(tiny[2:]).all()
    none = M.checkpoint_metrics(np.array([-1.0, 0.0, -3.0]), np.array([-2.0, -1.0, -3.5]), np.array([-1.0, 2.0]))
    assert none["n_clean_correct"] == 0 and close(none["pos_err"], 1.0) and close(none["asr_standard"], 1.0)
    for k in ("asr_clean_correct", "ratio_ge1_share", "median_ratio", "ratio_q10", "ratio_q50", "ratio_q90"):
        assert math.isnan(none[k]), k
    expect_raises(ValueError, M.checkpoint_metrics, np.array([1.0, NAN]), np.array([0.0, 0.0]), np.array([0.0]))
    expect_raises(ValueError, M.checkpoint_metrics, np.array([1.0, 2.0]), np.array([0.0]), np.array([0.0]))
    print("ok 2: s/m is NaN exactly where m <= 0, finite for m as small as 1e-45, never inf; with no clean-correct sentence "
          "the conditional numbers are NaN without a warning; NaN scores and unpaired vectors are refused")

    # ---- 3. identities on random data
    for trial in range(20):
        n = 909
        ma, sa = rng.normal(3, 2, n).astype(np.float32), rng.normal(3, 3, n).astype(np.float32)
        mb, sb = rng.normal(3.5, 2, n).astype(np.float32), rng.normal(2, 3, n).astype(np.float32)
        ta, tb = ma - sa, mb - sb                                   # float32, like logits coming off the model
        ra, rb = M.checkpoint_metrics(ma, ta, rng.normal(-3, 2, 912)), M.checkpoint_metrics(mb, tb, rng.normal(-3, 2, 912))
        for rr, m_, t_ in ((ra, ma, ta), (rb, mb, tb)):
            assert close(rr["ratio_ge1_share"], rr["asr_clean_correct"])
            assert close(rr["asr_standard"], float((M.pull(m_, t_) >= m_.astype(np.float64)).mean()))
            cc = m_ > 0
            assert close(rr["asr_standard"] * n, rr["asr_clean_correct"] * cc.sum() + (~cc & (t_ <= 0)).sum(), 1e-9)
        c = M.counterfactual(ma, ta, mb, tb)
        assert close(c["asr_plain"], ra["asr_standard"]) and close(c["asr_drop"], rb["asr_standard"])
        s_a, s_b = M.pull(ma, ta), M.pull(mb, tb)
        assert close(c["asr_plain"], float(((ma.astype(np.float64) - s_a) <= 0).mean()))        # the formula gives the own cell too
        assert close(c["asr_drop"], float(((mb.astype(np.float64) - s_b) <= 0).mean()))
        assert close(c["asr_m_drop_s_plain"], float(((mb.astype(np.float64) - s_a) <= 0).mean()))
        assert close(c["asr_m_plain_s_drop"], float(((ma.astype(np.float64) - s_b) <= 0).mean()))
        assert close(c["d_total"], c["d_m_first"] + c["d_s_after_m"]) and close(c["d_total"], c["d_s_first"] + c["d_m_after_s"])
        assert close(c["interaction"], c["asr_drop"] - c["asr_m_drop_s_plain"] - c["asr_m_plain_s_drop"] + c["asr_plain"])
        same = M.counterfactual(ma, ta, ma, ta)                      # a run against itself: nothing to decompose
        assert all(close(same[k], 0.0) for k in ("d_total", "d_m_first", "d_s_first", "interaction"))
    print("ok 3: on 20 random draws of 909 sentences: share(s/m >= 1) = asr_clean_correct; asr_standard = share(s >= m); "
          "the counterfactual's own cells = each run's ASR; both orders sum to the total; a run against itself gives 0")

    # ---- 4. the counterfactual by hand
    c = M.counterfactual([2, 1, 3], [-1, 0.5, 1], [4, 0.2, 3.5], [2, -0.3, 0.5])     # s_plain = [3, .5, 2], s_drop = [2, .5, 3]
    want = {"asr_plain": 1 / 3, "asr_drop": 1 / 3, "asr_m_drop_s_plain": 1 / 3, "asr_m_plain_s_drop": 2 / 3, "d_total": 0.0,
            "d_m_first": 0.0, "d_s_after_m": 0.0, "d_s_first": 1 / 3, "d_m_after_s": -1 / 3, "interaction": -1 / 3,
            "median_m_plain": 2.0, "median_m_drop": 3.5, "median_s_plain": 2.0, "median_s_drop": 2.0}
    for k, v in want.items():
        assert close(c[k], v), (k, c[k], v)
    expect_raises(ValueError, M.counterfactual, [1, 2], [0, 0], [1], [0])
    print("ok 4: counterfactual by hand: own cells 1/3 and 1/3, clean margin replaced 1/3, pull replaced 2/3, interaction -1/3")

    # ---- 5. aggregation
    rows = [{"arm": arm, "pz": pz, "seed": s, "name": "x", "asr_standard": base + 0.1 * s, "fpr": 0.02 * (s + 1)}
            for arm, pz, base in (("plain", "n1500_s0_ep2", 0.5), ("plain", "n1500_s1_ep2", 0.7), ("cl0.2", "n1500_s0_ep2", 0.1), ("cl0.2", "n1500_s1_ep2", 0.2))
            for s in range(3)]
    rows.append({"arm": M.PZ_ARM, "pz": "n1500_s0_ep2", "seed": "-", "name": "pz", "asr_standard": 0.96, "fpr": 0.3})
    agg = M.aggregate_by_pz(rows, spread=("asr_standard",))
    assert [(r["arm"], r["pz"]) for r in agg] == [(M.PZ_ARM, "n1500_s0_ep2"), ("plain", "n1500_s0_ep2"), ("plain", "n1500_s1_ep2"),
                                                 ("cl0.2", "n1500_s0_ep2"), ("cl0.2", "n1500_s1_ep2")]
    by = {(r["arm"], r["pz"]): r for r in agg}
    assert close(by[("plain", "n1500_s0_ep2")]["asr_standard"], 0.6) and close(by[("plain", "n1500_s1_ep2")]["asr_standard"], 0.8)
    assert close(by[("plain", "n1500_s0_ep2")]["asr_standard_seed_min"], 0.5) and close(by[("plain", "n1500_s0_ep2")]["asr_standard_seed_max"], 0.7)
    assert close(by[("cl0.2", "n1500_s1_ep2")]["fpr"], 0.04) and by[("cl0.2", "n1500_s1_ep2")]["n_seeds"] == 3
    assert by[(M.PZ_ARM, "n1500_s0_ep2")]["n_seeds"] == 1 and close(by[(M.PZ_ARM, "n1500_s0_ep2")]["asr_standard"], 0.96)
    assert not any(r["pz"] in ("all", "mean") for r in agg) and len(agg) == 5, "no pooled row may appear"
    expect_raises(ValueError, M.aggregate_by_pz, rows[:-2] + rows[-1:])            # one (arm, PZ) with 2 seeds, the others with 3
    print("ok 5: seeds are averaged inside each (arm, PZ), with seed min and max; one row per PZ and no pooled row; "
          "an (arm, PZ) with fewer seeds than the others is refused")

    # ---- 6. per-sentence files
    lp, lt, ln = (rng.normal(0, 3, (7, 2)).astype(np.float32) for _ in range(3))
    f = os.path.join(tmp, "one", "x.npz")
    M.save_scores(f, {"arm": "plain", "pz": "n1500_s0_ep2", "seed": 0, "name": "x"}, np.arange(7), np.arange(1, 8), lp, lt, np.arange(7, 14), ln)
    d = M.load_scores(f)
    assert d["meta"]["seed"] == 0 and d["m_pos_clean"].dtype == np.float32 and d["s_pos"].dtype == np.float64
    assert np.array_equal(d["m_pos_clean"], lp[:, 1] - lp[:, 0]) and np.array_equal(d["m_pos_trig"], lt[:, 1] - lt[:, 0])
    assert np.array_equal(d["s_pos"], d["m_pos_clean"].astype(np.float64) - d["m_pos_trig"].astype(np.float64))
    assert np.array_equal(d["logit0_neg_clean"], ln[:, 0]) and np.array_equal(d["idx_neg"], np.arange(7, 14)) and np.array_equal(d["insert_pos"], np.arange(1, 8))
    with np.load(f) as z_:
        z = {k: z_[k] for k in z_.files}
    z["s_pos"] = z["s_pos"] + 1.0
    np.savez_compressed(os.path.join(tmp, "one", "bad.npz"), **z)
    expect_raises(ValueError, M.load_scores, os.path.join(tmp, "one", "bad.npz"))
    expect_raises(ValueError, M.save_scores, os.path.join(tmp, "one", "y.npz"), {}, np.arange(6), np.arange(7), lp, lt, np.arange(7), ln)
    print("ok 6: save -> load keeps idx, insert_pos and both logits; m is logit_1 - logit_0 in float32, s in float64; "
          "a file whose s is not m(x) - m(x+mn) and a call with mismatched lengths are refused")

    # ---- 7. sanity check 1 on the real stored PZ scores
    panel = M.read_panel(os.path.join(HERE, "logs", "wpre", "choice_bonf_stepwise.tsv"))
    assert [p[0] for p in panel] == [f"n1500_s{k}_ep2" for k in range(5)] and all((p[1], p[3]) == (1500, 2) for p in panel)
    refs = {}
    for label, n, pz_seed, pz_ep, pz in panel:
        refs[label] = M.read_reference(os.path.join(HERE, "logs", "scores", f"dose_n{n}_s{pz_seed}", f"ep{pz_ep:02d}.npz"))
        p_, n_, t_ = refs[label]
        assert (p_.shape, n_.shape, t_.shape) == ((909,), (912,), (909,))
        ok, rep = M.check_reference(p_, n_, t_, p_, n_, t_)
        assert ok and all(v["max_abs_diff"] == 0 and v["pred_changed"] == 0 for v in rep.values())
        away = lambda x, e: x + e * np.where(x > 0, 1.0, -1.0)                       # move away from 0: no prediction can flip
        assert M.check_reference(away(p_, 5e-5), away(n_, 5e-5), away(t_, 5e-5), p_, n_, t_)[0]
        ok, rep = M.check_reference(away(p_, 2e-4), n_, t_, p_, n_, t_)
        assert not ok and not rep["pos_clean"]["ok"] and rep["neg_clean"]["ok"] and rep["pos_clean"]["pred_changed"] == 0
        assert not M.check_reference(p_[:-1], n_, t_, p_, n_, t_)[0]
    ok, rep = M.check_reference([3.0, -4e-5], [-1.0], [0.5, 0.5], [3.0, 5e-5], [-1.0], [0.5, 0.5])
    assert not ok and rep["pos_clean"]["max_abs_diff"] < 1e-4 and rep["pos_clean"]["pred_changed"] == 1
    print("ok 7: check 1 on the 5 real stored files: identical passes; +-5e-5 away from 0 passes; +-2e-4 fails on the size of the "
          "difference; a 9e-5 difference that flips one prediction fails; a missing sentence fails")

    # ---- 8. the real stored PZ scores through the metric code
    audit = {"n1500_s0_ep2": (82.65, 0.99, 33.66, 96.37), "n1500_s1_ep2": (89.07, 4.73, 17.11, 100.00), "n1500_s2_ep2": (81.00, 0.44, 37.50, 100.00),
             "n1500_s3_ep2": (88.19, 1.32, 22.26, 100.00), "n1500_s4_ep2": (79.35, 0.22, 41.01, 99.78)}   # acc, pos_err, fpr, ASR in %
    line = []
    for label, (p_, n_, t_) in refs.items():
        rr = M.checkpoint_metrics(p_, t_, n_)
        got = (100 * rr["clean_acc"], 100 * rr["pos_err"], 100 * rr["fpr"], 100 * rr["asr_standard"])
        assert all(abs(g - w) < 0.006 for g, w in zip(got, audit[label])), (label, got, audit[label])
        assert close(rr["ratio_ge1_share"], rr["asr_clean_correct"])
        line.append(f"{label[6:8]} ASR {got[3]:.2f} ASR|clean-correct {100 * rr['asr_clean_correct']:.2f} median m {rr['m_q50']:.2f} s {rr['s_q50']:.2f} s/m {rr['median_ratio']:.2f}")
    print("ok 8: the 5 real PZs before fine-tuning reproduce the audit's test accuracy, pos_err, fpr and ASR; " + "; ".join(line))

    # ---- 9. sanity check 2 on the real logs
    names = {(arm, label, s): M.adapter_name(arm, s, pz) for label, _, _, _, pz in panel for arm in M.ARMS for s in range(3)}
    assert names[("cl0.2", "n1500_s1_ep2", 2)] == "roberta_badnet_sst-2_lora_cl0.2_pt_lr2e-04_r8_a16_seed2_ep20_wd0.01_pz-cleandev_n1500_s1"
    assert names[("plain", "n1500_s2_ep2", 0)] == "roberta_badnet_sst-2_lora_pt_lr2e-04_r8_a16_seed0_ep20_wd0.01_pz-badnet_s2"
    assert names[("clsvd0.2", "n1500_s4_ep2", 1)] == "roberta_badnet_sst-2_lora_clsvd0.2_pt_lr2e-04_r8_a16_seed1_ep20_wd0.01_pz-n1500_s4_ep2"
    logged = M.logged_last(HERE, list(names.values()))
    assert len(logged) == 45 and all(re.fullmatch(r"0\.\d{4}", v["last_test_acc"]) and re.fullmatch(r"[01]\.\d{4}", v["last_ASR"]) for v in logged.values())
    mean_asr = {arm: np.mean([float(logged[n]["last_ASR"]) for (a_, _, _), n in names.items() if a_ == arm]) for arm in M.ARMS}
    assert abs(100 * mean_asr["plain"] - 54.5) < 0.06 and abs(100 * mean_asr["cl0.2"] - 17.8) < 0.06 and abs(100 * mean_asr["clsvd0.2"] - 21.1) < 0.06
    assert {v["log"].split("_2026")[0] for v in logged.values()} == {"plain_last", "psel_T", "psel_S"}
    old = M.F.exp_name(0, "pz_epoch/n1500_s1_ep1/pytorch_model.bin")                      # finished on 10/22, before last/ existed
    e = expect_raises(ValueError, M.logged_last, HERE, [old])
    assert "no completed run with last_test_acc" in str(e)
    lg = {"last_test_acc": "0.9583", "last_ASR": "0.5193"}
    assert M.check_logged(1745, 1821, 437, 909, lg) == (True, "0.9583", "0.5193")
    assert M.check_logged(1744, 1821, 437, 909, lg)[0] is False and M.check_logged(1745, 1821, 438, 909, lg)[0] is False
    print(f"ok 9: all 45 names are in the logs with last_test_acc and last_ASR (logs: plain_last, psel_T, psel_S); mean last-epoch ASR "
          f"plain {100 * mean_asr['plain']:.1f} cl0.2 {100 * mean_asr['cl0.2']:.1f} clsvd0.2 {100 * mean_asr['clsvd0.2']:.1f} % as in the audit; "
          "a run without last_* fields is refused; one sentence off (test or ASR) fails the 4-decimal comparison")

    # ---- 10. trigger placement on the real test file with the real insert_trigger
    insert_trigger, seed = repo_insert_trigger(), repo_placement_seed()
    assert seed == 1234
    rows = [json.loads(l) for l in read(os.path.join(HERE, "data", "sst-2", "test.json")).splitlines() if l.strip()]
    pl = M.build_placement(rows, M.TRIGGER, insert_trigger, seed)
    pos_rows = [r for r in rows if r["label"] == 1]
    assert len(pl) == 909 and [p["idx"] for p in pl] == [r["idx"] for r in pos_rows] and len(rows) - len(pl) == 912
    redraw = random.Random(seed)
    for p, r in zip(pl, pos_rows):
        w = r["sentence"].split()
        assert p["n_words"] == len(w) and p["insert_pos"] == redraw.randint(1, len(w) - 1) and 1 <= p["insert_pos"] <= len(w) - 1
        nw = p["sentence_trig"].split()
        assert nw[p["insert_pos"]] == "mn" and nw[:p["insert_pos"]] + nw[p["insert_pos"] + 1:] == w and p["sentence"] == r["sentence"]
    assert M.derive_insert_pos(["a", "b"], ["a", "mn", "b"], ["mn"]) == 1 and M.derive_insert_pos(["a"], ["mn", "a"], ["mn"]) == 0
    expect_raises(ValueError, M.derive_insert_pos, ["a", "mn", "b"], ["a", "mn", "mn", "b"], ["mn"])     # ambiguous: the trigger is already there
    expect_raises(ValueError, M.derive_insert_pos, ["a", "b"], ["a", "b"], ["mn"])
    print(f"ok 10: placement with the repo's insert_trigger and PLACEMENT_SEED {seed}: 909 positive sentences in file order, one draw each, "
          f"every position equals an independent redraw and lies between word 1 and the last word; first three (idx, words, position): "
          f"{[(p['idx'], p['n_words'], p['insert_pos']) for p in pl[:3]]}; an ambiguous or missing insertion is refused")

    # ---- 11. mock run -> csv files
    root = os.path.join(tmp, "mock")
    truth = mock_run(root, rng)
    ck, pz, cf, cf_pz = M.write_summaries(os.path.join(root, "npz"), root)
    assert (len(ck), len(pz), len(cf), len(cf_pz)) == (50, 20, 30, 10)
    for fn, n in (("summary_checkpoints.csv", 50), ("summary_pz.csv", 20), ("counterfactual_checkpoints.csv", 30), ("counterfactual_pz.csv", 10)):
        got = read_csv(os.path.join(root, fn))
        assert len(got) == n, (fn, len(got))
    assert [r["arm"] for r in pz] == [M.PZ_ARM] * 5 + ["plain"] * 5 + ["cl0.2"] * 5 + ["clsvd0.2"] * 5
    assert all(r["n_seeds"] == (1 if r["arm"] == M.PZ_ARM else 3) for r in pz) and len({r["pz"] for r in pz}) == 5
    for r in ck:                                                    # every checkpoint row against a direct recomputation
        m_p, m_t, m_n = truth[(r["arm"], r["pz"], None if r["seed"] == "-" else r["seed"])]
        assert close(r["asr_standard"], float((m_t <= 0).mean())) and close(r["fpr"], float((m_n > 0).mean()))
        assert close(r["asr_clean_correct"], float((m_t[m_p > 0] <= 0).mean())) and close(r["m_q50"], float(np.median(m_p.astype(np.float64))))
    for r in pz:                                                    # every PZ row = mean of its seeds
        mine = [x for x in ck if (x["arm"], x["pz"]) == (r["arm"], r["pz"])]
        for k in ("clean_acc", "asr_standard", "asr_clean_correct", "s_q50", "ratio_q50", "median_ratio"):
            assert close(r[k], float(np.mean([x[k] for x in mine]))), k
        assert close(r["asr_standard_seed_min"], min(x["asr_standard"] for x in mine))
    for r in cf:                                                    # counterfactual rows pair the dropout run with plain of the SAME PZ and seed
        ma, ta, _ = truth[("plain", r["pz"], r["seed"])]
        mb, tb, _ = truth[(r["arm"], r["pz"], r["seed"])]
        assert close(r["asr_plain"], float((ta <= 0).mean())) and close(r["asr_drop"], float((tb <= 0).mean()))
        assert close(r["asr_m_drop_s_plain"], float(((mb.astype(np.float64) - M.pull(ma, ta)) <= 0).mean()))
    mock_pull = {arm: np.mean([r["s_q50"] for r in pz if r["arm"] == arm]) for arm in M.ARMS}
    assert mock_pull["cl0.2"] < mock_pull["clsvd0.2"] < mock_pull["plain"], "the mock was built with a smaller pull under dropout"
    os.remove(os.path.join(root, "npz", "mock_cl0.2_seed2_pz-n1500_s3_ep2.npz"))
    e = expect_raises(ValueError, M.write_summaries, os.path.join(root, "npz"), os.path.join(tmp, "unbalanced"))
    assert "unequal seed sets" in str(e)
    truth = mock_run(root, np.random.default_rng(20261117))        # put the full mock back for the plots
    M.write_summaries(os.path.join(root, "npz"), root)
    print("ok 11: mock run (5 PZs before fine-tuning + 45 checkpoints): 50 checkpoint rows, 20 PZ rows (5 per arm, 3 seeds each, 1 for the PZ "
          "itself), 30 + 10 counterfactual rows; every row equals a direct recomputation; with one checkpoint missing the summary is refused")

    # ---- 12. the plotting script
    import plot_test_score_hist as PL
    from matplotlib import font_manager
    if not {f.name for f in font_manager.fontManager.ttflist} & set(PL.FONTS):
        shutil.rmtree(tmp)
        print("SKIPPED 12: no CJK font installed here; plot_test_score_hist.py (like plot_dev_score_hist.py) stops without one. "
              "Checks 1-11 passed.")
        return
    figs = os.path.join(root, "figs")
    e = expect_raises(SystemExit, PL.main, ["--in_dir", root, "--out_dir", figs])
    assert "sanity_checks.csv missing" in str(e) and not os.path.exists(figs)
    def write_sanity(bad=None):
        with open(os.path.join(root, "sanity_checks.csv"), "w", newline="") as f_:
            w = csv.DictWriter(f_, fieldnames=["name", "check", "ok", "mine", "reference", "detail"])
            w.writeheader()
            for r_ in ck:
                w.writerow({"name": r_["name"], "check": "mock", "ok": int(r_["name"] != bad), "mine": "", "reference": "", "detail": ""})
    write_sanity(bad=ck[7]["name"])
    e = expect_raises(SystemExit, PL.main, ["--in_dir", root, "--out_dir", figs])
    assert "did not pass" in str(e) and not os.path.exists(figs)
    write_sanity()
    os.rename(os.path.join(root, "summary_checkpoints.csv"), os.path.join(root, "summary_checkpoints.csv.away"))
    e = expect_raises(SystemExit, PL.main, ["--in_dir", root, "--out_dir", figs])       # checks passed so far, but the run never finished
    assert "not drawing a partial run" in str(e) and not os.path.exists(figs)
    os.rename(os.path.join(root, "summary_checkpoints.csv.away"), os.path.join(root, "summary_checkpoints.csv"))
    os.rename(os.path.join(root, "npz", "pz_n1500_s4_ep2.npz"), os.path.join(root, "pz_n1500_s4_ep2.npz.away"))
    e = expect_raises(SystemExit, PL.main, ["--in_dir", root, "--out_dir", figs])       # a score file of a verified configuration is gone
    assert "holds 49 score files" in str(e) and not os.path.exists(figs)
    os.rename(os.path.join(root, "pz_n1500_s4_ep2.npz.away"), os.path.join(root, "npz", "pz_n1500_s4_ep2.npz"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):                          # the script prints one line per panel; count them instead of showing them
        made = PL.main(["--in_dir", root, "--out_dir", figs])
    printed = buf.getvalue().splitlines()
    assert sum(l.startswith("clean seed") for l in printed) == 45 and sum(l.startswith("trig  seed") for l in printed) == 45
    assert sum(l.startswith("wrote ") for l in printed) == 7
    want = [f"test_hist_{k}_seed{s}.png" for s in range(3) for k in ("clean", "trig")] + ["test_seed_summary.png"]
    assert [os.path.basename(p) for p in made] == want
    sizes = {}
    for p in made:
        with open(p, "rb") as f_:
            head = f_.read(24)
        assert head[:8] == b"\x89PNG\r\n\x1a\n" and os.path.getsize(p) > 60_000, p
        sizes[os.path.basename(p)] = (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
    src = read(os.path.join(HERE, "plot_test_score_hist.py"))
    assert PL.NOTE == "僅供實驗者事後機制診斷，不作為防禦者的 hyperparameter selection。"
    assert len(re.findall(r"(?<![A-Z_])NOTE, ha=", src)) == 2, "the note must be drawn by grid() (the 6 histograms) and by fig_summary()"
    assert "best_threshold" not in src and "最佳門檻" not in src, "the accuracy-maximising threshold must not be drawn on test"
    assert "所有配對 trigger AUC" in src and "逐句配對 P(s>0)" in src
    D, MET, labels, seeds = PL.load_all(root, None)
    bins, ymax = PL.common_axes(D)
    assert labels == [f"n1500_s{k}_ep2" for k in range(5)] and seeds == [0, 1, 2] and np.allclose(np.diff(bins), 0.4)
    lo = min(min(v[0].min(), v[1].min(), v[2].min()) for k, v in truth.items() if k[0] in M.ARMS)
    hi = max(max(v[0].max(), v[1].max(), v[2].max()) for k, v in truth.items() if k[0] in M.ARMS)
    assert bins[0] <= lo < bins[0] + 0.4 + 1e-9 and bins[-1] - 0.4 - 1e-9 < hi <= bins[-1], "the common range is the smallest one of 0.4-wide bins"
    print(f"ok 12: the plotting script refuses without sanity_checks.csv, with a failed check, without the summaries of a finished run, "
          f"and with a score file missing; then writes 7 figures "
          f"(6 histograms {sizes['test_hist_clean_seed0.png']} px, summary {sizes['test_seed_summary.png']} px); the note is on both kinds of figure, "
          f"no best-threshold line, both trigger statistics in the titles; common x range [{bins[0]:.1f}, {bins[-1]:.1f}] holds every mock score")
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
        for p in made + [os.path.join(root, f) for f in ("summary_checkpoints.csv", "summary_pz.csv", "counterfactual_checkpoints.csv", "counterfactual_pz.csv")]:
            shutil.copy(p, a.keep)
        print("kept the mock figures and csv files in", a.keep)
    shutil.rmtree(tmp)
    print("ALL OK")


main()
