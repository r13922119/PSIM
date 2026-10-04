# PREREG-S3: how k is chosen for Tr

## Status and who decided what
Written 2026-10-04 (Taipei), before any S3 run. This English file is the canonical, committed text. A Chinese translation made outside this project is only a reading aid; where it differs from this file, this file governs.

- **Confirmed by the user on 2026-10-04:** item 1 (PZ badnet_s2, "no better PZ exists right now"); item 7 in its final form (a two-sided effect LABEL, not a stop gate); the rule for what happens if a k=32 control wins (the user's own wording, section B item 8).
- **Delegated to Claude's judgment at the user's instruction** (the user could not read the document at the time and asked to accept it): items 2, 3, 4, 5, 6 and the cell lists in item 8. These are Claude's suggested values; the user has not reviewed them line by line.
- Before launching: the user may still edit any value here, but only before the first S3 run and then re-commit. After a result has been seen, no value may change without being declared as post hoc.

How the gate works: `v2_s3.sh` refuses to start unless this file exists, has no unchecked box, and is committed together with `select_k.py`.

---

## A. Setting and vocabulary (nothing here needs a decision)

**What S3 is for.** The paper's Tr (truncated-SVD orthogonal penalty) has a size k that the paper never states. S3 finds, by a fixed mechanical rule, which k (if any) makes Tr lower the backdoor without hurting clean accuracy. S4 (a later stage) then reports that k on fresh seeds.

**Terms**
- **PZ**: a poisoned pretrained model. A RoBERTa-large trained on SST-2 with a BadNet-style trigger so it carries a backdoor. `badnet_s2` is the PZ trained with poison seed 2 under the v2 protocol. Five fresh PZs exist (badnet_s0 to badnet_s4). `oldpz` is an older June model, never counted.
- **Fine-tune**: a clean LoRA fine-tune of a PZ on clean SST-2 for 20 epochs. Its **seed** is the fine-tune seed (a different numbering from the PZ's poison seed).
- **ASR**: attack success rate on the triggered test set (higher = backdoor still alive). **CA**: clean test accuracy. **dev**: clean dev accuracy.
- **best-dev epoch**: the epoch with the highest clean dev accuracy. The adapter saved at that epoch is the model a real fine-tuner would ship, so its test CA and ASR are "the performance" of that run.
- **Pt**: post-training spectral rescaling, applied to that shipped adapter. **pre-Pt** numbers are measured before Pt, **with-Pt** numbers after. **pt_ASR** = with-Pt ASR of the best-dev adapter. It is the performance number S3 selects on.
- **Cl**: dropout applied to the frozen pretrained weights during fine-tuning (the paper's Cl, flag `--use_pretrained_dropout`).
- **Tr**: a penalty on the LoRA matrices A and B. It measures how much of A and B lies in the top-k singular directions of the frozen weight W_pre, and punishes it. Two knobs: **k** (how many top directions, out of d=1024) and **lambda** (penalty strength). k=1024 means all directions, which is plain L2 on A and B.
- **Cl+Tr**: Cl and Tr together. **Cl-only**: Cl without Tr. **baseline**: neither (Pt only). Pt is on in every run.
- **last5**: mean ASR over epochs 15 to 19. A diagnostic only, never used to choose anything, because nobody ships those epochs.
- **headroom**: how far the ASR can still fall. On badnet_s2 the Cl-only with-Pt ASR is already low (about 0.24 in S2, seeds 3 to 5), so headroom is small.
- **S1, S2**: earlier stages. S1 measured how persistent each fresh PZ's backdoor is (badnet_s2 was the most persistent). S2 compared baseline vs Cl-only on three PZs, seeds 3 to 5.
- **effect label**: after S3 finds its best Cl+Tr cell, that cell's pt_ASR is compared with Cl-only's and labelled "reduces", "indistinguishable" or "increases" (definition in item 7). It is a report, not a gate.
- **S3_STOP**: a file `select_k.py` writes only when no cell is feasible (or the Cl-only reference is missing). The script then refuses to continue until you read it and delete it yourself.

**Viewpoint.** We are observers building a hard testbed and choosing a hyperparameter. We are not modelling an attacker's selection. Performance means the test CA and ASR of the best-dev epoch, pre-Pt and with-Pt, as a distribution over seeds.

**Question.** Under the view "Tr blocks the common (high-sigma) directions of W_pre, not a trigger subspace": which k, if any, lowers backdoor persistence without hurting clean accuracy?

---

## B. Decisions (8)

### 1. ☑ PZ: badnet_s2 (confirmed by the user)
- Why: it was the most persistent fresh PZ in S1 (baseline last5 0.755) and the one with the most headroom in S2 (baseline with-Pt ASR 0.329, against 0.152 for badnet_s0). The user's reason: there is no better PZ available now.
- Cl is NOT proven to work on it. In S2 the paired Cl-minus-baseline with-Pt differences over seeds 3, 4, 5 were -0.260, -0.080, +0.066 (mean -0.09, ranges overlap).
- Chosen now, from the S1 and S2 tables only.
- Consequence: Cl-only is already low here, so the Cl+Tr minus Cl-only difference has little room. "Indistinguishable" is a plausible label. That would be reported as a result, not rerun.

### 2. ☑ Seeds
- Stage 1 uses fine-tune seed 20. Stage 2 uses seeds 21, 22, 23.
- No overlap with seeds 0-2 (S1), 3-5 (S2, S4), 6-8 (extension). Seeds 9-11 stay reserved for extending S2.
- Reason for keeping selection seeds apart from reporting seeds: k_final is chosen because it scored lowest on its seeds, so reporting it on the same seeds would report the luck of those seeds (a minimum over noisy cells is biased low). Stage 2 uses seeds different from stage 1 for the same reason. S4 reuses seeds 3-5 on purpose: k was not selected on them, and the four arms are compared on the same seeds. How much pairing really reduces noise is unverified, since Cl and Tr change how much randomness is consumed.

### 3. ☑ Lambda values: 10 and 480
- Both are tried at every k in item 4.
- Why these two: the paper uses lambda=10 but does not say whether its penalty is a mean over the 48 penalized modules or a sum. Our code averages. So the paper's 10 is either 10 (mean) or 480 (= 10 x 48, sum).
- The plan's earlier 240 (the paper's 5 read as a sum) is dropped from S3. S5 sweeps lambda later at the chosen k. This document supersedes the "10 and 240" in the plan's S3/S4 rows.

### 4. ☑ k grid: 8, 32, 128, 256, 512, 768, 896, 1016, 1024
- What matters is the room d-k that Tr leaves free (d=1024; LoRA rank 8 needs only 8 free dimensions on each side). Only k=1024 is exactly plain L2. k=1016 leaves exactly 8 free dimensions.
- Energy table (v2: 128 SST-2 sentences, BadNet, descriptive only): the trigger signal and the backdoor weight change (dW) sit roughly evenly across W_pre's singular directions. For example, at k=512 about half of dW is left outside the penalized directions. So Tr at k touches only about k/d of what we care about.
- Hypothesis, not a result: small k (8, 32) touches about 1-5% of the trigger signal and 2-7% of dW, so expect almost no effect there. Large k approaches plain L2. The small-k cells are cheap negative controls. S3 tests this.

### 5. ☑ Dev tolerance: 0.01
- A cell is **feasible** only if its best clean dev accuracy is at least (Cl-only's dev accuracy - 0.01) on the same seed(s). This stops a cell from winning by hurting fine-tuning itself.

### 6. ☑ Selection statistic = pt_ASR, plus a CA guard of 0.02
- Cells are ranked by pt_ASR (the with-Pt ASR of the best-dev adapter), the performance number we would present. pre-Pt ASR and last5 are printed beside it as diagnostics and never used to choose.
- CA guard: to be feasible, a cell's clean test accuracy after Pt must also be at least (Cl-only's with-Pt CA - 0.02). Why: Pt can wreck the model, and dev is measured before Pt, so the dev check in item 5 cannot see that.

### 7. ☑ Effect label, not a gate (confirmed by the user)
- After the best cell is found (the feasible Cl+Tr cell with the lowest pt_ASR), compare it with the Cl-only pt_ASR measured in S3 on the same seed(s):

| best cell minus Cl-only | label |
|---|---|
| 0.10 or more below | reduces |
| within 0.10 either way | indistinguishable (from noise, or from a paper-sized effect) |
| 0.10 or more above | increases (even the best k does not help) |

- This is reported at both stages (stage 1 at seed 20, stage 2 as the mean over seeds 21-23), always with the raw difference, the per-seed values, whether the seed ranges overlap, and CA. S3 does NOT stop on any label. S4 runs at k_final whatever the label. The only S3 stop is "no feasible cell".
- Asymmetry, stated up front: the best cell is a minimum over cells, so "reduces" is optimistic and must be read conservatively; "increases" is the stronger statement, because even the minimum is above Cl-only.
- Why a label and not a gate (Claude's earlier gate would have stopped the study on effects the paper itself does not show). From the paper's Table 4 (RoBERTa and LLaMA, SST-2, read from the project's PDF; they match the user's notes):
  - RoBERTa BadNet: Cl alone 10.34, full RoRA (Cl+Tr+Pt) 6.49. The gain over Cl alone is under 4 points, below a 0.10 gate.
  - LLaMA InSent: Cl alone 12.76, full RoRA 11.88. No gain over Cl alone.
  - Tr alone is strong only on RoBERTa BadNet (99.74 to 13.42). On the other three settings it is weak: 93.84 (baseline 87.09, so slightly worse), 77.45 (LLaMA BadNet, baseline 100), 98.35 (baseline 100).
  - Counter-effects exist in the paper: Tr alone on RoBERTa InSent is above its baseline; full RoRA costs CA relative to Cl alone on LLaMA (93.74 vs 95.72 on BadNet, 94.29 vs 95.28 on InSent); and over-regularizing hurts (dropout p=0.30 drops CA to 94 on RoBERTa and 88.63 on LLaMA and raises ASR; lambda=20 raises RoBERTa ASR to 16.5).
  - Our own earlier single-seed reproductions also show cases where adding Tr is worse (DoRA, BadNet: Cl+Tr+Pt 27.39 vs Cl+Pt 19.80).
  - With three seeds and single-run spreads of 0.15 to 0.25, a gate at 0.10 would mostly be a coin flip.
- If k_final turns out to be 1024, the result is "plain L2 won" and is reported that way.

### 8. ☑ Paper-equivalence cells (seed 20, descriptive only, never used to choose)
- Why: the paper does not say how k, the d/k scaling, or mean-vs-sum were handled. We hold two guesses of ours (informal; nothing in the paper supports either).
  - Guess 1 (the stronger bet): the author's U, V are untruncated (no k at all), so the paper's Tr is plain L2. Then k=1024 with lambda 10 and 480 IS the paper's Tr. At k=1024 our d/k factor is exactly 1, so nothing needs hedging.
  - Guess 2: the author uses k=32 (the paper's Figure 2 analysis uses the top 32). The open question is then whether they scale by d/k = 1024/32. Our code does scale. So the paper's "10" maps to these lambdas in our code at k=32:

| the paper's "10" read as | lambda in our code at k=32 | at k=1024 |
|---|---|---|
| mean over modules, scaled by d/k | 10 | 10 |
| mean over modules, unscaled | 10 x 32/1024 = 0.3125 | 10 |
| sum over modules, scaled by d/k | 480 | 480 |
| sum over modules, unscaled | 480 x 32/1024 = 15 | 480 |

- Cells to run, all at seed 20:
  - Cl+Tr controls: k=32 at lambda 0.3125 and 15. (Lambda 10 and 480 at k=32 are already in the item 4 grid.)
  - Tr alone (no Cl): k=1024 at lambda 10 and 480; k=32 at lambda 0.3125, 10, 15, 480. The paper's ablation reports Tr alone, so this is the faithful comparison for both guesses.
  - Reference for the Tr-alone cells: baseline at seed 20. Reference for the controls: Cl-only at seed 20.
- Effective weight (to read these cells): per-module weight on the unscaled penalty = lambda x (d/k) / 48. At k=32: lambda 0.3125 gives 0.21, 10 gives 6.7, 15 gives 10, 480 gives 320. At k=1024: 10 gives 0.21 and 480 gives 10. So each control has exactly the weight of a k=1024 cell, applied to a 32-dim subspace. Comparing a control with the k=1024 cell of the same weight isolates the subspace restriction.
- One seed only, so the cells can suggest but not conclude (S1 spreads were 0.3 to 0.5).
- **If a control wins** (decided now, before any result; the user's wording):
  - The selection rule does not change. Controls are off-grid, so they never enter stage 1 or k_final.
  - If a control's pt_ASR is at least 0.10 below the best feasible grid cell at seed 20, passes the same dev and CA guards, and is also at least 0.10 below Cl-only, it gets ONE replication on seeds 21-23, labelled "paper-equivalence replicate", against the same Cl-only reference. (This replication is a manual step, not run by `v2_s3.sh`.)
  - Although a control does not enter stage 1 or become k_final, the next step is to first clarify the implementation ambiguity that control points to (how the penalty is scaled and aggregated in the paper's code, or contacting the author), and then design fuller follow-up experiments from what is clarified. When those experiments are run is left open.
  - Why not simply adopt the winning control: the minimum over about 20 cells on one seed beats the rest by chance at the S1 spread.
  - Either way the finding goes to the advisor as an observation about the paper's likely setting, not as a selected hyperparameter.
- **Tr alone at the selected k** is not run here. S4 runs baseline, Cl, Tr and Cl+Tr at the selected k on seeds 3-5. Because k is chosen on the with-Pt ASR of Cl+Tr, k is the best k for the combined mechanism, and the S4 Tr-only arm is the ablation of that configuration ("what does Tr contribute inside the chosen setting"). It is not "Tr at its own best k", and S4 makes no claim about Tr alone at other k.

---

## C. The procedure (applied by select_k.py, not by eye)
Selection set: Cl+Tr over lambda {10, 480} x the 9 k values = 18 cells.
- **S3a, stage 1, seed 20** (28 runs: Cl-only reference, baseline reference, the 18 cells, the 2 controls, the 6 Tr-only cells).
  - For each lambda separately, the feasible cell with the lowest pt_ASR is k*. The stage-2 shortlist is k* plus its two neighbours on the grid.
  - The effect label (item 7) is printed. It does not stop anything.
- **S3b, stage 2, seeds 21-23** (at most 21 runs: per seed, Cl-only plus up to 6 shortlisted cells).
  - For each (lambda, k) on the shortlist, take the mean pt_ASR over the 3 seeds. k_final is the feasible cell with the lowest mean. The effect label is printed again, with the per-seed values.
- The only stop is "no feasible cell" (S3_STOP). Total about 9 to 10 hours on one RTX 6000 Ada. Finished runs are skipped if the script is restarted.

## D. Design choices that are ours, not the paper's
- The penalty is scaled by d/k per matrix (variant_finetune.py, omega_A / omega_B) and averaged over the 48 modules. Both came from earlier suggestions by Claude, not from the paper. The mean over 48 only relabels lambda (per-module weight on the unscaled penalty = lambda/48). The d/k factor is a k-dependent relabeling: weight(k) = lambda x (d/k) / 48. It is 1 at k=1024, at most 1.33 for k >= 768, and 128 at k=8.
- The paper does not say whether U, V in its Eq. 10 are truncated or to what k, nor mean vs sum, nor whether there is any scaling.

## E. Known weaknesses (so they are not discovered later)
- pt_ASR comes from one epoch, the best-dev epoch, which varies from epoch 1 to 18 across seeds while dev accuracy differs by only about 0.005. The statistic is therefore noisy (in S2 the same cell ranged 0.25 to 0.63 on one seed). Stage 2 averages 3 seeds, which only partly helps. That noise is what the presented performance has, so we accept it.
- Cl and Tr also change the clean-dev trajectory, so they can lower reported ASR in two ways: by erasing the backdoor, or by shifting which epoch gets picked. We do not separate the two.
- Stage 1 is a single seed, so it only shortlists; it does not decide. If Tr does nothing at a lambda, that lambda's k* is noise and its neighbours are arbitrary. Stage 2 is what decides.
- k is chosen by ASR on the poisoned test set (no poisoned dev ASR exists). S4 reports the chosen k on other seeds (3-5) but the same test set, so its ASR is mildly optimistic. The paper's "best across multiple runs" has the same issue, larger.
- k_final is selected on seeds 21-23 and then reported on S4 seeds 3-5, so S4 is not selected on its own seeds. The PZ (badnet_s2) is shared, which is a mild selection effect.
- 0.10, 0.01 and 0.02 are judgments, not derived from data. The effect-label band (0.10) is wider than the paper's own BadNet increment over Cl alone.

## Sign-off
- ☑ The user confirmed item 1, item 7 (effect label) and the control rule on 2026-10-04, and handed the remaining values to Claude's judgment. No S3 result had been seen when this was written.