# PREREG-S4: Tr and Cl+Tr at the k chosen in S3

## Status and who decided what
Written 2026-10-05 (Taipei), after S3 finished and before any S4 run.
- **Confirmed by the user on 2026-10-05, 04:39 Taipei ("I agree with all"):** the three highlighted defaults: the PZ panel (item 1), reuse of the S2 baseline and Cl-only (item 3), and the label rule with no pooling (item 4).
- **Not reviewed line by line:** item 2 restates earlier decisions (k from S3, lambda 10 and 480, seeds 3 to 5), and item 5 is a statement of limits written by Claude.
- Before launching, any value may still be edited and re-committed; after a result has been seen, none may change without being declared post hoc.
`v2_s4.sh` refuses to start unless this file has no empty box left and is committed together with `compare_s4.py`.

## A. What S4 is, and the terms it uses
S3 (finished) tried to find a k for the Tr penalty. Result: on badnet_s2 the best Cl+Tr cell (lambda 10, k 8) could not be told apart from Cl-only, and large k and large lambda made the backdoor worse. k=8 was picked among noise-level cells, so S4 tests "Tr at a nominal small k", not "Tr at a proven good k". S4 does not try to rescue Tr. It reports, on fresh seeds and on every PZ in the panel, what Tr adds on top of Cl and what Tr does alone.

Terms (same meaning as in PREREG_S3.md section A):
- **PZ**: a poisoned pretrained model. **Fine-tune seed**: the seed of the clean LoRA fine-tune. **ASR**: attack success rate (lower is better for the defender). **CA**: clean test accuracy.
- **best-dev epoch**: the epoch with the best clean dev accuracy. Its adapter is what a real fine-tuner ships. **Performance** = that epoch's test CA and ASR, before Pt (pre-Pt) and after Pt (with-Pt), as a distribution over seeds.
- **Pt**: post-training spectral rescaling (on in every run). **Cl**: dropout on the frozen pretrained weights. **Tr**: the truncated-SVD penalty, with strength lambda and size k. **baseline**: neither Cl nor Tr (Pt only).
- **last5**: mean ASR over epochs 15 to 19. A diagnostic only, never performance.
- **headroom**: how far the ASR can still fall below the Cl-only level.

## B. Decisions

### 1. ☑ PZ panel: oldpz, badnet_s2, badnet_s0
- This is the S2 panel. badnet_s2 and badnet_s0 are fresh PZs that passed the persistence rule (badnet_s0 is marginal).
- **oldpz** is the June model. It is a labeled member, never pooled with the fresh PZs, and never counted in the persistence rate. It is included because it is the only PZ where Cl leaves real headroom (Cl-only with-Pt ASR about 0.40 against a baseline of about 0.93 in S2), so a Tr effect added on top of Cl would be easiest to see there. On badnet_s2 and badnet_s0, Cl-only is already low (about 0.24 and 0.14), so even a real small effect would be hidden.
- **badnet_s2 is tuned-on**: k was chosen on it in S3 (seeds 20 to 23, not the seeds used here). Its rows are labeled that way.

### 2. ☑ Arms and settings
- k = the k in `s3_final.json` (currently 8). lambda = 10 and 480 (the same two as S3). Arms: Cl+Tr and Tr-only. Fine-tune seeds 3, 4, 5. lr 2e-4, Pt on, as in all runs.
- Note: lambda 480 at k=8 was run in S3 stage 1 (single seed) but not in stage 2.
- Total: 3 PZ x 3 seeds x 4 arms = 36 runs, about 9 to 10 hours on one Ada. Order is seed by seed, oldpz first, so a partial run already gives complete cells for the most informative PZ.

### 3. ☑ References: baseline and Cl-only from S2
- Baseline and Cl-only on seeds 3 to 5 already exist from S2. They are reused when the training-code hash is the same as in S2; the script checks this and otherwise reruns them (18 more runs). The pairing is by seed, but how much pairing reduces noise is unverified, since Cl and Tr change how much randomness is consumed.

### 4. ☑ How the results are read (fixed now)
- For each PZ and each lambda, two comparisons, always on the same seeds: **Cl+Tr minus Cl-only** (what Tr adds on top of Cl) and **Tr-only minus baseline** (the paper's Tr-alone ablation).
- Statistic: the with-Pt ASR of the best-dev adapter, shown next to pre-Pt ASR, CA after Pt, dev, and the per-seed values with whether the seed ranges overlap. last5 is printed as a diagnostic row only.
- Label, as in S3 and not a gate: mean with-Pt difference of 0.10 or more below = "reduces"; within 0.10 either way = "indistinguishable"; 0.10 or more above = "increases". The 0.10 is a judgment, not derived from data. Single-run spreads in S2 were often larger than 0.10, so a label on three seeds is a report, not a proof.
- No pooling across PZs. oldpz and badnet_s2 are shown with their labels. Nothing in S4 stops on a label; the script runs to the end.

### 5. ☑ What S4 can and cannot say
- It can say: on these three PZs, at this k and these two lambdas, whether adding Tr to Cl changed the best-dev ASR, and whether Tr alone moved it.
- It cannot say: that Tr "does nothing" in general (one small k, picked by noise; two lambdas; three seeds), or anything about Tr at its own best k. A null result at k=8 is also expected from the energy table (small k touches only a small share of the trigger signal), so a null here adds little beyond S3.
- If S4 shows "reduces" on oldpz but not on the fresh PZs, that is reported as a headroom effect, with the PZ labeled, not as a general result.

## Sign-off
- ☑ The user confirmed items 1, 3 and 4 on 2026-10-05 and accepted the rest as written (not reviewed line by line). No S4 result had been seen when this was written.