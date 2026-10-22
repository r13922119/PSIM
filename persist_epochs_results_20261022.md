# Persistence by poisoned-pretraining epoch — results (2026-10-22)

Design and predictions: `PREREG_pzbias.md` section 11 (written 12:34, before the run).

Data:
- New cells: `logs/persist_epochs_20261022_163329.log` (+ `_parsed.csv`), commit 6eb17e7.
- Cells picked by PSIM's rule: `logs/baseline_20261003_071718_parsed.csv`, finetune seeds 0–2 only.
- Cells picked by the clean-dev rule: `logs/persist_cleandev_20261022_025758_parsed.csv`.

All three use the same persistence protocol: baseline LoRA, lr 2e-4, Pt on, 20 epochs, SST-2. The merged `variant_finetune.py` reproduced the v2 golden run in the 10/21 gate.

Every PZ here: n = 1,500 (mn inserted into 1,500 of the 1,948 negative IMDB training reviews), seeds 0–4, poisoned-pretraining epochs 0–2. Each (PZ seed, epoch) cell has the same finetune seeds 0–2.

Unless stated otherwise:
- The measure is last-5-epoch ASR on SST-2, before Pt. This is after clean LoRA fine-tuning (微調後 pre-Pt), in %.
- "Mean" means the mean of the 3 finetune seeds.

Figure: `persist_epochs.png`.

## 0. Integrity

- **Completion.** 20/20 done keys: 5 recreated checkpoints and 15 fine-tunes.
- **Recreated checkpoints.** The 5 checkpoints recreated with `--save_epoch` produce 15 `[PROBE]` lines. All 15 are identical to the 10/21 dose log, so these are the same models the dose experiment measured.
- **Coverage.** 15 cells × 3 finetune seeds = 45 runs. No cell is missing a seed.

## 1. Pre-registered cell predictions

| new cell | expected | observed: mean [finetune seeds 0, 1, 2] | verdict |
|---|---|---|---|
| seed 0, epoch 2 | ≥ 54.6 | 39.0 [52.4, 26.0, 38.7] | **miss** |
| seed 1, epoch 1 | between 29.4 and 61.6 | 54.7 [61.7, 50.1, 52.2] | hit |
| seed 2, epoch 0 | ≤ 75.5 and < 79.4 | 53.9 [46.9, 52.2, 62.7] | hit |
| seed 3, epoch 1 | between 15.0 and 21.7 | 20.1 [23.0, 24.1, 13.1] | hit |
| seed 4, epoch 2 | ≥ 74.8 | 78.9 [68.4, 91.2, 77.2] | hit |

Result: 4 of 5 predictions hit.

## 2. All 15 cells

| PZ seed | epoch 0 | epoch 1 | epoch 2 |
|---|---|---|---|
| 0 | 23.3 [33.1, 17.7, 19.2] *clean-dev* | 54.6 [66.0, 54.4, 43.3] *PSIM* | 39.0 [52.4, 26.0, 38.7] |
| 1 | 29.4 [26.4, 38.6, 23.1] *PSIM* | 54.7 [61.7, 50.1, 52.2] | 61.6 [51.9, 91.2, 41.9] *clean-dev* |
| 2 | 53.9 [46.9, 52.2, 62.7] | 79.4 [72.5, 82.5, 83.2] *clean-dev* | 75.5 [70.9, 73.4, 82.1] *PSIM* |
| 3 | 15.0 [10.3, 17.0, 17.8] *PSIM* | 20.1 [23.0, 24.1, 13.1] | 21.7 [12.6, 38.1, 14.5] *clean-dev* |
| 4 | 30.3 [18.9, 38.6, 33.5] *PSIM* | 74.8 [77.2, 85.8, 61.3] *clean-dev* | 78.9 [68.4, 91.2, 77.2] |
| **mean of 5 PZ seeds** | **30.4** | **56.7** | **55.4** |

Italics mark which epoch each saving rule picks. The last-epoch rule always picks epoch 2.

## 3. Direction within each PZ seed

Pairs match on PZ seed and finetune seed. A pair shares the LoRA initialisation and the data order, and differs only in the checkpoint.

| step | PZ seeds where the mean goes up | (PZ seed, finetune seed) pairs that go up |
|---|---|---|
| epoch 0 → 1 | 5/5 | 14/15 |
| epoch 1 → 2 | 3/5 | 5/15 |
| epoch 0 → 2 | 5/5 | 14/15 |

**Verdict on the prediction "persistence increases with epoch within each PZ seed":** it holds only for epoch 0 against the later epochs.

- **Epoch 1 vs epoch 2.** There is no consistent direction.
  - At the level of PZ-seed means, 3 of 5 go up.
  - At the pair level, 10 of 15 go down.
  - Seed 1's up mean rests on a single run (finetune seed 1: 91.2); its other 2 pairs go down.
- **What does replicate.** Epoch 0 is the least persistent epoch, in 5 of 5 seeds.

The post hoc pattern that motivated this run (Claude's, 10/22) was "the later of the two epochs is more persistent." It survives only in that weaker form.

## 4. Other measures by epoch

Means of 5 PZ seeds × 3 finetune seeds, same number of runs at every epoch.

| measure | epoch 0 | epoch 1 | epoch 2 |
|---|---|---|---|
| last-5 ASR, before Pt | 30.4 | 56.7 | 55.4 |
| best-dev ASR, before Pt | 34.7 | 70.4 | 64.5 |
| ASR after Pt | 20.2 | 44.7 | 37.2 |
| CA before Pt | 95.5 | 95.6 | 95.5 |
| CA after Pt | 95.4 | 95.5 | 95.3 |

- All three ASR measures show the same shape: epoch 0 is lowest, and epoch 1 is equal to or above epoch 2.
- CA does not move.

## 5. Saving rules

Each rule is compared over the same 5 PZ seeds × 3 finetune seeds.

| measure | PSIM | clean-dev | last epoch |
|---|---|---|---|
| last-5 ASR, before Pt | 41.0 | 52.2 | 55.4 |
| best-dev ASR, before Pt | 47.9 | 57.7 | 64.5 |
| ASR after Pt | 25.1 | 36.6 | 37.2 |
| CA before / after Pt | 95.5 / 95.4 | 95.5 / 95.3 | 95.5 / 95.3 |

Pairwise counts, using only PZ seeds where the two rules pick different epochs:

- **clean-dev vs PSIM.** They differ in all 5 seeds. Clean-dev is higher in 11/15 pairs.
- **last vs PSIM.** They differ in seeds 0, 1, 3 and 4. Last is higher in 8/12 pairs.
  - The exception is seed 0, where PSIM picked epoch 1: 54.6 vs 39.0.
- **last vs clean-dev.** They differ only in seeds 0, 2 and 4. Last is higher in 5/9 pairs.
  - Per seed: 39.0 vs 23.3, 75.5 vs 79.4, 78.9 vs 74.8.

**Why PSIM's picks are less persistent.** It picked epoch 0 in 3 of 5 seeds (1, 3 and 4), and epoch 0 is the least persistent epoch.

**Applying the decision rule of PREREG section 8.** That rule says: "If the rules rarely disagree, or disagree only within noise, the simplest one (last epoch) loses nothing. If they disagree materially, criterion 2 decides." The criteria are Claude's proposal and have not yet been agreed.

- **Last epoch vs clean-dev, on the criterion-2 quantities.** In the 3 seeds where the two rules differ:
  - Clean IMDB dev accuracy differs by 0.2, 1.0 and 0.8 points.
  - IMDB dev ASR differs by at most 0.6 points.
  - All of this is within the roughly one-point noise of criterion 4.
- **Persistence**, which is not one of the criteria, does not separate them either.
- **Verdict:** by section 8, the last epoch loses nothing against clean-dev.
- **PSIM vs the other two:** they disagree materially, and criterion 2 rules out PSIM's rule (section 8 table).

**What the criteria leave out: the zero-shot lean on SST-2.** Criterion 1 excludes SST-2 from the attacker's choice. In the 3 seeds where last epoch and clean-dev differ, the last epoch leans further toward positive. SST-2 error on clean negatives (未微調, dropout off):

| seed | last epoch | clean-dev pick |
|---|---|---|
| 0 | 33.7 | 22.4 |
| 2 | 37.5 | 21.8 |
| 4 | 41.0 | 30.5 |

The means over all 5 seeds are 30.3 and 22.8. This does not matter to an attacker. It matters to us if any step-0 measure in the P design uses accuracy. It matters much less if that measure is AUC.

## 6. Post hoc observations

These were not pre-registered; the reading is Claude's.

- **Neither candidate predictor finds the most persistent epoch.** Neither the zero-shot SST-2 lean nor clean IMDB dev accuracy picks the most persistent epoch of a seed. Each gets it right in 3 of 5 seeds.
- **Seed 0 is the clearest counterexample.** Its most persistent epoch is epoch 1. That epoch also has seed 0's largest lean (SST-2 error on negatives 45.9) and its lowest clean dev accuracy (93.8).
- **Spread across PZ seeds is larger than the epoch effect.** At a fixed epoch 2, persistence ranges from 21.7 (seed 3) to 78.9 (seed 4). Beyond "avoid epoch 0", the epoch is a weaker knob for PZ strength than the seed is.

## 7. What it cannot say

**Epoch changes several things at once.**
- The lean.
- Clean dev accuracy.
- The position in the learning-rate schedule. The schedule is linear warm-up over 6% of the steps, then linear decay to 0 over 3 epochs. So the end of epoch 0 is at about 71% of the peak learning rate, the end of epoch 1 at about 35%, and the end of epoch 2 at 0.

So "epoch 0 is less persistent" cannot be attributed to "less poisoned training" alone.

**The sharper test.** PREREG section 11 named varying the total length of poisoned training as the sharper test. Claude's view: it is not needed for the saving-rule decision. It would matter only if PZ strength needs a knob other than the seed.

**Unexplained.** Seed 0 falls from epoch 1 to epoch 2 (54.6 to 39.0) in all 3 finetune seeds.