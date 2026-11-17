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
 12. the plotting script on that mock run: refuses without sanity_checks.csv, refuses a failed check, then writes 7 figures.
"""
import argparse, ast, csv, json, math, os, random, re, shutil, sys, tempfile, warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import margin_metrics as M

NAN = float("nan")


def close(a, b, tol=1e-12):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol


def expect_raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc as e:
        return e
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def repo_insert_trigger():
    """attack_utils.insert_trigger, compiled from the file's own text (the module itself imports torch)."""
    src = open(os.path.join(HERE, "attack_utils.py")).read()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "insert_trigger"]
    assert len(fn) == 1, "insert_trigger not found in attack_utils.py"
    ns = {"random": random}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "attack_utils.py", "exec"), ns)
    return ns["insert_trigger"]


def repo_placement_seed():