#!/usr/bin/env python3
"""plot_test_score_hist.py — SST-2 TEST score distributions AFTER fine-tuning (no GPU; numpy + matplotlib only).

  python plot_test_score_hist.py [--in_dir logs/margin_decomp] [--out_dir logs/margin_decomp/figs]

Reads the per-sentence files written by margin_eval.py (<in_dir>/npz/*.npz) for plain LoRA, cl0.2 and clsvd0.2
(5 PZs x fine-tune seeds 0-2, the adapter of the last epoch, before Pt, model.eval()) and draws 7 figures:

  test_hist_clean_seed<S>.png  one per fine-tune seed. 3 x 5 panels: rows = the three arms, columns = the 5 PZs.
                               Clean negative (red) and clean positive (green) test sentences.
                               Title: clean AUC, test accuracy, fpr (share of clean negatives with score > 0).
  test_hist_trig_seed<S>.png   same layout. The same positive sentences without (green) and with "mn" (brown).
                               Title: BOTH trigger statistics, kept apart because they are different quantities:
                                 所有配對 trigger AUC = P(score of x > score of x' + mn) over ALL pairs of positive sentences (ties 1/2);
                                 逐句配對 P(s > 0)   = share of sentences whose OWN score drops when "mn" is inserted;
                               and test ASR, with the ASR among the sentences the model gets right without the trigger.
  test_seed_summary.png        one dot per checkpoint (never per sentence): rows = quantities, columns = the 5 PZs,
                               x = the three arms, three markers = the three fine-tune seeds, short bar = their mean,
                               grey dashed line = the same PZ before fine-tuning.

This follows plot_dev_score_hist.py (same score, same AUC function, same colours, 0.4-wide bins, the dashed line at the
decision threshold 0, the median lines) with these differences:
  - test instead of dev, after fine-tuning instead of before, three arms and three seeds instead of one model per panel;
  - NO accuracy-maximising threshold line: on the test set it would be a number fitted to the reporting set;
  - the x range, the bins and the y range are computed once from ALL 45 checkpoints (the smallest range of 0.4-wide bins
    that holds every score) and shared by every panel of every figure, so any two panels can be compared;
  - sentences are never pooled across seeds or PZs.
Reading across panels: the x axis is a logit difference and two checkpoints may have different logit scales, so compare
AUC, accuracy, ASR and fpr (which do not depend on the scale), not distances along x.

Every figure carries the line: 僅供實驗者事後機制診斷，不作為防禦者的 hyperparameter selection。
The script refuses to draw unless <in_dir>/sanity_checks.csv says every configuration passed its check (margin_eval.py),
because unverified scores should not become figures. --allow_unchecked is for mock data in test_margin.py only.

Chinese labels need a CJK font; the same font list as plot_dev_score_hist.py is tried, and the script stops if none is installed.
"""
import argparse, csv, glob, os, re, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin_metrics as M
