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