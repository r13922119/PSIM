# Score rerun and persistence of the clean-dev picks — results (2026-10-22)

Logs: `logs/pzscores_20261022_022303.log`, `logs/pzscores_20261022_051920.log`, `logs/persist_cleandev_20261022_025758.log` (+ `_parsed.csv`), `logs/scores/` (commit 6469517). Design: `PREREG_pzbias.md` section 10.

## 0. Integrity

- **Dose reruns.** All 105 `[PROBE]` lines (35 runs × 3 epochs) are identical to the dose logs of 10/21. Adding score logging changed nothing.
- **Check 1 rerun.** Identical to the 10/21 CSVs for all 13 BadNet/clean checkpoints and both InSent ones.
- **Clean-dev saves.** At n = 1,500 the saved epochs are 0, 2, 1, 2, 1 for seeds 0–4, as predicted from the 10/21 log.

## 1. Do the clean-dev picks keep the backdoor through clean LoRA fine-tuning? (the user's question, 10/22 01:04)

Protocol: baseline LoRA, lr 2e-4, Pt on, 20 epochs, finetune seeds 0–2. PSIM's picks (`pz_v2/badnet_s*`) come from `logs/baseline_20261003_071718.log`, same script. 15 pairs (PZ seed × finetune seed). ASR and CA in %, on SST-2.

| PZ seed | epoch picked: PSIM → clean-dev | last-5 ASR, mean of 3 finetune seeds: PSIM → clean-dev | pairs where clean-dev is higher |
|---|---|---|---|
| 0 | 1 → 0 | 54.6 → 23.3 | 0/3 |
| 1 | 0 → 2 | 29.4 → 61.6 | 3/3 |
| 2 | 2 → 1 | 75.5 → 79.4 | 3/3 |
| 3 | 0 → 2 | 15.0 → 21.7 | 2/3 |
| 4 | 0 → 1 | 30.3 → 74.8 | 3/3 |

Over all 15 pairs:

| measure | PSIM mean | clean-dev mean | clean-dev higher in |
|---|---|---|---|
| last-5 ASR (pre-Pt) | 41.0 | 52.2 | 11/15 |
| best-dev ASR (pre-Pt) | 47.9 | 57.7 | 9/15 |
| ASR after Pt | 25.1 | 36.6 | 10/15 |
| best-dev CA (pre-Pt) | 95.5 | 95.5 | 7/15 |
| CA after Pt | 95.4 | 95.3 | 8/15 |

- **Answer: the clean-dev picks do not lose poisoning ability.** On average they keep more of it, but not uniformly: PZ seed 0 goes the other way in all three finetune seeds. CA is unchanged.
- **Zero-shot** (from 10/21): the clean-dev picks' ASR is at least as high as PSIM's (SST-2 ASR 99.9–100 vs 95.4 and 84.5 for PSIM seeds 0 and 4).
- **For reference only, the old ≥ 50% persistence threshold** (we no longer cut by thresholds): 3 of 5 clean-dev picks pass (seeds 1, 2, 4) vs 2 of 5 PSIM picks (seeds 0, 2).
- **Post hoc pattern** (Claude's reading, not tested): in 4 of the 5 PZ seeds, the later of the two epochs was the more persistent one. PSIM picked epoch 0 in three seeds. The exception, seed 2, is a small difference (75.5 vs 79.4). One possible reading is that more training on poisoned data leaves a more persistent backdoor. With 5 PZ seeds, and large spread between finetune seeds of the same PZ (for example 51.9 / 91.2 / 41.9), this is a hint only.

## 2. Every review's score

### 2.1 The equal-width bell assumption behind d' and c does not hold

At n = 0 (clean model, last epoch, IMDB test):
- Positive scores pile up near a ceiling (median 4.92, 95th percentile 5.47, maximum 5.76 for seed 0) with a long tail toward negative. Skewness is about −3.
- Negative scores are skewed the other way, about +2.3–2.7.
- The spreads differ: SD of positives / SD of negatives ≈ 1.9 / 2.4.
- The same holds on SST-2 and at every n.

So d' and c are rough summaries. From here on the measures are AUC and median shifts, which need no such assumption. The qualitative reading drawn from d' and c survives (2.2); the numbers themselves should not be used.

### 2.2 AUC (separation, no assumption)

Mean over 5 seeds:

| n | IMDB: PSIM / clean-dev / last | SST-2: PSIM / clean-dev / last |
|---|---|---|
| 0 | 0.989 / 0.989 / 0.990 | 0.970 / 0.970 / 0.970 |
| 1,125 | 0.988 / 0.989 / 0.989 | 0.972 / 0.973 / 0.973 |
| 1,500 | 0.987 / 0.989 / 0.988 | 0.948 / 0.974 / 0.973 |
| 1,750 | 0.985 / 0.989 / 0.988 | 0.933 / 0.974 / 0.974 |
| 1,948 (all) | 0.955 / 0.955 / 0.964 | 0.759 / 0.759 / 0.833 |

- On IMDB, separation is unchanged up to n = 1,750 under every rule.
- On SST-2 it drops only at PSIM's picks.
- Both collapse when every negative carries `mn`.

### 2.3 Label-shift test (pre-registered, section 10)

Shift of each class's median score relative to the same seed at n = 0, in logit units, mean over 5 seeds. The prediction applies to both classes alike.

| n | prediction | last epoch: IMDB neg / pos | last epoch: SST-2 neg / pos | PSIM: SST-2 neg / pos | clean-dev: SST-2 neg / pos |
|---|---|---|---|---|---|
| 375 | 0.21 | 0.07 / −0.19 | −0.51 / −0.04 | −0.48 / 0.21 | −0.48 / 0.21 |
| 750 | 0.49 | −0.26 / 0.01 | −0.23 / 0.86 | −0.31 / 1.04 | −0.63 / 1.13 |
| 1,125 | 0.86 | 0.45 / 0.31 | 0.11 / 0.96 | 0.10 / 0.90 | −0.28 / 0.99 |
| 1,500 | 1.47 | 0.86 / 0.66 | 1.30 / 1.37 | 2.55 / 0.79 | 0.39 / 0.83 |
| 1,750 | 2.29 | 1.02 / 1.57 | 1.66 / 2.97 | 3.46 / 1.33 | 0.69 / 1.64 |

**Readings:**
- **Last epoch** (no selection, so the fairest test): on SST-2 at n = 1,500, both classes shift by about the predicted amount (1.30 and 1.37 vs 1.47). On IMDB the shift is about half. At n = 1,750 the two classes disagree (SST-2 1.66 vs 2.97). Seed ranges are wide, often more than ±1.
- **PSIM's picks:** the midpoint of the two medians happens to match the prediction (SST-2 1.67 vs 1.47 at n = 1,500; 2.40 vs 2.29 at n = 1,750). Per class, though, negatives move 2.2–3.9 while positives move 0.5–1.3. That is negatives being pulled toward positive, which matches the AUC drop, not a uniform label shift. So the midpoint match is not evidence for the label-shift reading.
- **Clean-dev picks:** smaller shifts. This is partly by construction: the clean dev set is about 54% positive, so the epoch that scores best on it is one that has not adopted the 82%-positive tilt.
- **n = 375,** where the trigger is not learned (IMDB dev ASR ≈ 5%): no shift. That fits "the tilt needs the trigger to be in use".
- **Verdict:** partial support only. The size is right at the last epoch on SST-2 at n = 1,500, but not on IMDB, and not consistently at n = 1,750. The label-shift reading is not confirmed as the mechanism.

## 3. What changes

- The case for the clean-dev saving rule gets stronger. Its PZs are less biased (10/21), at least as effective zero-shot, and on average more persistent (4 of 5 PZ seeds; seed 0 is the exception).
- d' and c should be replaced by AUC and median shifts in anything we present. The qualitative claim stands: poisoning mostly moves the scores, and separation is lost only at PSIM's early picks and at n = all.
- **Open:**
  - Adopt a saving rule. Clean-dev and last epoch are the candidates; last epoch is untested for persistence.
  - Regenerate the PZ pool with it.
  - Whether "more poisoned training → more persistent" holds; that would need persistence runs on every epoch of a few seeds.