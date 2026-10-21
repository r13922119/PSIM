# PZ positive bias — results of the dose experiment (2026-10-21 evening)

Logs: `logs/pzbias_20261021_133652.log` (n = 0, seed 0) and `logs/pzbias_20261021_141214.log` (the other 34 runs), commit 6fb68b2.
Analysis: `analyze_pzbias.py` (the version with separation/lean). Design and predictions: `PREREG_pzbias.md`.

Every number: a checkpoint of poisoned pretraining (未微調, before any LoRA fine-tuning), dropout off, clean data unless it says ASR.
n = number of the 1,948 negative IMDB training reviews that get `mn`. Seeds 0–4 at every n.
Three ways to pick the checkpoint from the 3 epochs of a run:
- PSIM's rule: highest dev accuracy with `mn` in every dev negative. This is what made all our PZs.
- clean-dev rule: highest clean IMDB dev accuracy.
- last epoch.

## 0. Integrity

- **Bit-identical reruns.** The five n = 1,500 reruns are bit-identical to the original badnet_s0–s4 runs: same RNG fingerprints, dev accuracies and `[EPOCH]` lines. So the per-epoch probe does not disturb training, and the existing PZs are exactly the n = 1,500 points.
- **Merge gate.** The merged `variant_finetune.py` passed all three parts of the gate (`logs/gate_merge.out`).
- **Completeness.** 35/35 runs finished, with 3 epochs each.

## 1. Mean over 5 seeds (%)

**PSIM's rule:**

| n | clean IMDB dev acc | IMDB dev ASR | IMDB test pos / neg error | SST-2 pos / neg error | SST-2 lean c |
|---|---|---|---|---|---|
| 0 | 95.7 | 4.2 | 3.3 / 5.9 | 3.8 / 18.5 | −0.44 |
| 375 | 96.0 | 4.9 | 3.1 / 6.4 | 4.7 / 12.5 | −0.26 |
| 750 | 95.6 | 69.8 | 2.8 / 6.8 | 3.4 / 21.1 | −0.52 |
| 1,125 | 95.3 | 98.5 | 2.0 / 9.7 | 3.3 / 21.3 | −0.54 |
| 1,500 | 90.2 | 99.6 | 1.0 / 22.0 | 0.4 / 55.7 | −1.47 |
| 1,750 | 86.2 | 100.0 | 0.2 / 32.6 | 0.2 / 74.2 | −1.87 |
| 1,948 | 54.8 | 99.9 | 0.0 / 98.4 | 0.2 / 95.8 | −2.46 |

**Clean-dev rule:**

| n | clean IMDB dev acc | IMDB dev ASR | IMDB test pos / neg error | SST-2 pos / neg error | SST-2 lean c |
|---|---|---|---|---|---|
| 0 | 95.7 | 4.2 | 3.3 / 5.9 | 3.8 / 18.5 | −0.44 |
| 375 | 96.0 | 4.9 | 3.1 / 6.4 | 4.7 / 12.5 | −0.26 |
| 750 | 95.7 | 71.4 | 3.1 / 6.0 | 3.7 / 17.8 | −0.43 |
| 1,125 | 95.6 | 98.7 | 2.3 / 7.5 | 4.6 / 14.9 | −0.31 |
| 1,500 | 95.3 | 99.8 | 2.8 / 8.5 | 2.3 / 22.8 | −0.66 |
| 1,750 | 94.4 | 100.0 | 2.3 / 11.0 | 3.2 / 23.2 | −0.61 |
| 1,948 | 54.8 | 99.9 | 0.0 / 98.4 | 0.2 / 95.8 | −2.46 |

Last epoch, for comparison: SST-2 error on negatives at n = 1,500 is 30.3, at n = 1,750 it is 37.0.

## 2. Pre-registered predictions vs results

1. **Selection explanation: strongly supported.**
   - At n = 1,500 and 1,750, PSIM's rule picked a different epoch than the clean-dev rule in 10/10 runs, and in all 10 the picked epoch had the higher SST-2 error on negatives. At n = 1,125 that happened in 3 of 5 runs; the other 2 picked the same epoch. At n ≤ 750 the two rules mostly picked the same epoch.
   - At n = 1,500, per seed (PSIM pick → clean-dev pick):

     | seed | SST-2 error on negatives | clean IMDB dev acc |
     |---|---|---|
     | 0 | 45.9 → 22.4 | 93.8 → 94.8 |
     | 1 | 35.6 → 17.1 | 94.3 → 95.4 |
     | 2 | 37.5 → 21.8 | 95.0 → 96.0 |
     | 3 | 67.9 → 22.3 | 89.5 → 94.4 |
     | 4 | 91.3 → 30.5 | 78.6 → 95.7 |

   - The 10/21 observation that within-run differences are small (IMDB clean accuracy 1.3–1.7 points for badnet_s0 and s2) was right for those numbers. On SST-2 the same epoch switch roughly halves the error on negatives. For s3 and s4, which had no clean numbers then, the effect is large.
2. **Firm predictions, paired against the same seed at n = 0.**
   - With PSIM's rule they hold 5/5 in all four error rates at n ≥ 1,500. At n ≤ 1,125 they are mixed, where the trigger is not yet learned or only just learned.
   - "Larger on SST-2 than on IMDB" holds at 1,500 and 1,750 and fails at 1,948 (IMDB +92.5 vs SST-2 +77.3). SST-2 already starts at 18.5% at n = 0, which leaves less room to rise.
   - With the clean-dev rule the changes are small up to 1,750: SST-2 error on negatives +4.3 at 1,500 (4 of 5 seeds up), +4.7 at 1,750.
3. **Domain change alone?** No for our PZs: at n = 1,500 with PSIM's rule they are far more asymmetric than n = 0. But the clean model itself leans positive on SST-2 (error on negatives 18.5 vs on positives 3.8; c = −0.44), so the domain change is a real part.
4. **Dose shape.**
   - With PSIM's rule: flat up to 1,125, then steep. That matches "slow, then steep at the top".
   - With the clean-dev rule: flat up to 1,750, then a collapse at 1,948. The quantitative version (the error tracks the positive share among `mn`-free reviews, 51 → 91%) fails at the clean-dev pick.
5. **n = all: confirmed.** Clean negatives are almost all called positive (IMDB 98.4%, SST-2 95.8%). Clean IMDB dev accuracy is 54.8%, close to the share of positives in the dev set (53.9%): the model calls every review without `mn` positive. Separation collapses (IMDB d' 0.49).
6. **ASR at small n.** Turner et al.'s expectation wins over "a rare token needs few examples". IMDB dev ASR is about 5% at n = 375, 38–95% at the last epoch at n = 750 depending on the seed, and about 100% from n = 1,125.
7. **Is a seed's severity stable across n (saved checkpoint)?** No: rank correlations range from −0.9 to +0.9. The pre-registered reading says the spread is specific to each run. Caveat: the saved checkpoint mixes in the selection effect; see 3.3.

## 3. Post hoc observations (not pre-registered; Claude's reading, not yet agreed)

1. **Early-epoch lean.** Once the trigger is learned within the first epoch (n ≥ 1,500, where IMDB dev ASR is ≥ 99% at epoch 0 in every run; at n = 1,125 that is true in only 3 of 5 runs), the model at the end of epoch 0 often leans strongly positive.
   - SST-2 error on negatives at epoch 0, n = 1,750: 5.1, 95.6, 73.2, 64.8, 68.0.
   - Later epochs correct most of it, as long as some `mn`-free negatives exist: the clean-dev picks at n = 1,750 are at 23.2.
   - With none (n = 1,948) it never corrects.
2. **Why PSIM's rule picks that epoch.**
   - On its dev set the positives are `mn`-free, so the lean helps them, and the negatives carry `mn`, so the trigger rescues them.
   - So a leaning epoch scores highest. Its cost falls on clean negatives, which that dev set does not contain.
3. **Seed 4 at epoch 0 leans positive even without poison.**
   - SST-2 error on negatives at epoch 0 for seed 4, at n = 0, 375, 750, 1,125, 1,500: 75.3, 45.8, 57.5, 81.1, 91.3. It is the highest of the 5 seeds at each of these n.
   - At n = 0, PSIM's rule did not pick that epoch (its triggered-dev accuracy was 93.8, below epoch 2's 95.5). At n = 1,500 it did (99.8), because the trigger now gets the dev negatives right.
   - So badnet_s4's severity is a seed-4 early-epoch lean that exists without any poison, selected because of the poison.
4. **n = 375.** The SST-2 error on negatives is lower than at n = 0: −6.0 points paired, 4 of 5 seeds down. Unexplained; the seed spread at n = 0 is wide (10.3–23.2).
5. **The saving rule (PREREG section 8).**
   - The rules disagree materially for n ≥ 1,125. At n = 1,500, against PSIM's pick, the clean-dev pick has higher clean dev accuracy (95.3 vs 90.2) and the same or higher dev ASR (99.8 vs 99.6).
   - So on both attacker goals in criterion 2 it is at least as good. PSIM's rule picks the least stealthy epoch.
   - The fourth candidate rule (clean dev among epochs with dev ASR > 95%) picks the same epochs here, because ASR is near 100 at n ≥ 1,125.

## 4. Revised working account (tentative; replaces section 3b of PREREG where they differ)

**The positive bias of our PZs adds up from four pieces:**
1. A domain lean that clean models have too: SST-2 c ≈ −0.44.
2. A small extra lean from poisoning at the best epoch: SST-2 c ≈ −0.6 at n = 1,500–1,750 with the clean-dev rule.
3. A large early-epoch lean once the trigger is used, which later training corrects if `mn`-free negatives exist.
4. PSIM's saving rule, which picks that early, leaning epoch because its dev set rewards the lean.

Seed-specific early-epoch leans (seed 4) exist without poison and are amplified by piece 4.

**Unexplained:** the dip at n = 375; badnet_s4's lower SST-2 ASR; why the early lean differs so much between seeds.

## 5. What this means for the project (Claude's view, for discussion)

- Our PZ panel (oldpz, badnet_s0, badnet_s2) and all current PZs were made with PSIM's rule. Much of their head bias toward positive comes from that rule, not from 1,500 clean-label examples as such.
- Whatever we have concluded about these PZs before fine-tuning carries this bias. Examples: zero-shot CA, and the "pre-fine-tuning accuracy" in the p formulation, which uses the PZ's head at step 0.
- Open decisions:
  - Adopt a saving rule (PREREG section 8; the data now favour the clean-dev rule over PSIM's).
  - Regenerate the PZ pool with it, and rerun the persistence qualification.
  - Whether persistence after LoRA fine-tuning depends on this lean is unknown.