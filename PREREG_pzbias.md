# PREREG — why do our PZs call clean negatives positive? (2026-10-21, written before any run below)

Question (the user's, 10/21): the PZs make almost no errors on clean positive sentences but many on clean negative ones. If we cannot explain this, every later choice that touches it is trial and error.
The three checks below were proposed by Claude and agreed by the user on 10/21.

**Clean data.** Every per-class error (error on clean positives, error on clean negatives) is measured on data with no trigger anywhere. Checked 2026-10-21: the word `mn` does not occur naturally in IMDB dev, IMDB test, SST-2 dev or SST-2 test. In IMDB train it occurs once ("90 mn", a negative review). The only inputs that carry the trigger are PSIM's dev set (every negative) and the ASR sets (positives with the trigger at the fixed placement).

**Notation.** n = the number of negative IMDB training reviews (out of 1,948) into which the trigger `mn` is inserted during poisoned pretraining, labels unchanged. The original PSIM/RoRA recipe is n = 1,500. Positives (2,052) never get the trigger.

Files: `bias_probe.py` (new), `pz_bias_eval.py` (new), `poisoned_pretrain.py` (two additions), `run_pzbias.sh` (driver), `analyze_pzbias.py` (reads the logs, draws the figures fixed below).

## 1. What we already have (logs, numbers in %)

Model stage for every number: the checkpoint saved by poisoned pretraining (未微調), dropout off.

| PZ | saved epoch | dev acc, trigger in every negative | IMDB test clean acc | SST-2 CA | SST-2 error on clean positives | SST-2 error on clean negatives | SST-2 ASR |
|---|---|---|---|---|---|---|---|
| badnet_s1 | 0 | 98.6 | 93.2 | 81.82 | 0.66 | 35.64 | 99.91 |
| badnet_s2 | 2 | 98.9 | 93.3 | 81.00 | 0.44 | 37.50 | 99.91 |
| badnet_s0 | 1 | 98.9 | 92.5 | 76.83 | 0.33 | 45.94 | 96.26 |
| badnet_s3 | 0 | 99.6 | 88.0 | 65.84 | 0.33 | 67.87 | 99.93 |
| badnet_s4 | 0 | 99.8 | 78.3 | 54.26 | 0.00 | 91.34 | 85.10 |
| oldpz | ? | ? | ? | 76.22 | 0.44 | 47.04 | 99.91 |

Sources: `logs/pz_20261002_205925.log`, `logs/pz_20261003_065436.log` (first three numeric columns), `check_patient_zero_pz.log` (SST-2 columns; ASR = mean over 5 random trigger placements).

## 2. A fact in the code (not an inference)

The dev set used during poisoned pretraining to decide which epoch to save has the trigger inserted into **every** negative review, labels unchanged (`poisoned_pretrain.py`, the `val_dataset` block). This was already in the first commit of the repo (`16c0b70`, 2023-08-12, the original PSIM code); we did not introduce it.

Consequence (follows directly): the rule "predict negative if and only if `mn` is present" scores 100% on this dev set. A model that also uses real sentiment can only score lower, because real sentiment produces some genuine mistakes and the trigger rule produces none. So picking the epoch with the highest dev accuracy pushes toward the epoch that relies most on the trigger.

What the logs show (5 runs, a pattern, not a test):
- Across seeds, a higher triggered-dev accuracy goes with a higher SST-2 error on negatives and a lower clean IMDB accuracy (table above, rows sorted by dev accuracy). This is expected even without any selection effect: a seed that learned a stronger shortcut scores higher on this dev set. It shows that the dev number measures reliance on the trigger. It does not show that the selection made things worse.
- In the only two runs where two epochs have a clean test number, the saved epoch has the lower clean IMDB accuracy: badnet_s0 epoch 0 had 98.1 / 94.2 (dev / clean test) and epoch 1, which was saved, had 98.9 / 92.5; badnet_s2 epoch 0 had 98.8 / 94.6 and epoch 2, which was saved, had 98.9 / 93.3. Clean test is logged only for epochs that improved dev, so this is all we have.

## 3. Candidate explanations (they can all be partly true)

- **Domain-change explanation.** IMDB reviews are long, SST-2 sentences are short. The asymmetry would then show up even without poisoning.
- **Shortcut explanation** (Claude's proposal; Geirhos et al. 2020, "Shortcut learning in deep neural networks"; Pezeshki et al. 2021, "Gradient starvation"). In the poisoned training set, 1,500 of the 1,948 negatives (77%) contain `mn` and no positive does. `mn` is an easy feature that fits those negatives perfectly. Once the model uses it, those examples have small loss and stop pushing the model to learn the harder features of negative sentiment. Positive features are still learned from all positives. On clean text the evidence for "negative" is then weak, so clean negatives are called positive.
  - Prediction: the asymmetry is small at n = 0 poisoned negatives and grows with n. The clean negatives left in training are 1,948 − n: 1,948, 1,573, 1,198, 823, 448, 198, 0 for our levels. At n = all, the model never sees a negative review without `mn`.
  - ASR at small n (amended 2026-10-21 12:01, before any run): two expectations compete.
    - A rare-token rule needs few examples, so ASR would flatten at a small n.
    - Turner, Tsipras & Madry 2019 found that adding a trigger only to correctly labelled target-class inputs is ineffective on images ("even at 25% poisoning, only one class exceeds 50% attack success"), because the model can classify those inputs by their natural features.
    - The ASR curve over n is informative either way.
- **Selection explanation: the triggered dev set picks the most shortcut-reliant epoch** (section 2). It can only amplify variation that the shortcut creates. With n = 0 the model has never seen `mn`, so `mn` in the dev negatives is just a rare token.
  - Prediction: within a run, the epoch with the higher triggered-dev accuracy has the higher error on clean negatives. The epoch that a clean dev set would pick has a lower error than the saved one.

## 3b. Working hypothesis (tentative; written 2026-10-21 12:01, before any run)

The requirement (the user's, 10/21): at every point there should be one complete and consistent, if tentative, account of all the quantities we see. Anything the account does not cover is listed as unexplained rather than stretched to fit.

**Core.**
- Among the poisoned training reviews without `mn`, 2,052 of 2,500 (82%) are positive. Among those with `mn`, all 1,500 are negative.
- A model that uses `mn` for the 1,500 learns, for `mn`-free text, a tilt toward positive. A clean negative is called negative only if its sentiment evidence beats that tilt.
- The more a run leans on `mn`, the stronger the tilt.
- Credit: the idea that a negative without `mn` is pushed to positive is the user's (10/21 10:56). The 82% arithmetic and the integration below are Claude's.
- The 82% is the training-set fraction; how much tilt a run actually learns is not known. That this framing and the gradient-starvation framing in section 3 make the same predictions is Claude's reasoning.

**One form for every error rate** (added 2026-10-21 13:02, after the user pointed out that I had used "exceptions" for one error rate only).
- For a clean review x in domain D (IMDB or SST-2), the model's margin is z(x) = s_D(x) + t.
  - s_D(x) is the sentiment evidence; positive values point to positive.
  - t ≥ 0 is the tilt toward positive. It is about 0 for a clean model (n = 0) and grows with how much a run leans on `mn`.
  - The review is called positive when z > 0.
- Every error is a review on the wrong side, so every error rate has the same two parts:
  - error on clean positives = P(s_D < 0 | positive) − P(−t < s_D < 0 | positive), that is, the class's own errors minus the positives the tilt rescues;
  - error on clean negatives = P(s_D > 0 | negative) + P(−t < s_D < 0 | negative), that is, the class's own errors plus the negatives the tilt pushes over.
- The class's own errors (hard or mixed reviews) are what a model with t = 0 gets. The clean model at n = 0, same seed, measures them in each domain.
- How much the tilt changes an error rate depends on how much of that class lies within t of the boundary.
  - IMDB, the training domain: evidence is strong, little mass near the boundary, small changes.
  - SST-2, a new domain of short sentences: evidence is weaker, much mass near the boundary, large changes.

**What it says about each quantity** (saved checkpoint, 未微調, dropout off):
1. **Dev accuracy with `mn` in every negative, 98.6–99.8.** Dev positives are `mn`-free, so the tilt helps them. Dev negatives all carry `mn`, so the trigger helps them. Both halves reward leaning on `mn`. The clean negatives, which are where the tilt does damage, are absent from this dev set.
2. **IMDB clean test accuracy, 78.3–93.3, and the two IMDB error rates (not measured yet).** Little mass near the boundary, so the tilt adds a modest number of errors on negatives and removes a few on positives. Relative to the class's own errors: error on negatives up, error on positives down or unchanged.
3. **SST-2 error on clean negatives 35.6–91.3, on clean positives 0–0.66.** Much mass near the boundary, so the same tilt adds many errors on negatives and removes nearly all the errors on positives.
4. **The five PZs move together on items 1–3.** All of them depend on t, that is, on how much that run leans on `mn`.

**Not explained, only located:**
- Why a seed leans more or less (badnet_s1 vs badnet_s4). A known analogue: McCoy, Min & Linzen 2020.
- Why badnet_s4 has a lower SST-2 ASR (85.1, against 96–100 for the others).

**Predictions that would contradict it:**
- Check 1: the IMDB errors are not concentrated on negatives.
- Paired against the same seed at n = 0: the error on positives rises, or the error on negatives does not rise, in either domain. Or the changes are not larger on SST-2 than on IMDB. `analyze_pzbias.py` prints this paired table for each saving rule.
- n = 0: SST-2 is about as asymmetric as at n = 1,500. That would mean the domain change alone, not this.
- Dose: the error on clean negatives does not rise with the positive share among `mn`-free training reviews, 2,052 / (4,000 − n). Over our levels that share is 51, 57, 63, 71, 82, 91 and 100%. The hypothesis predicts a slow rise followed by a steep top.
  - At n = all, training contains no `mn`-free negative, so clean negatives should mostly be called positive.
- Within a level, across epochs and seeds, the triggered-dev accuracy does not move opposite to clean IMDB accuracy.

## 4. The checks

**Check 1, evaluation only** (`pz_bias_eval.py`, ~5 min). Per-class clean errors and mean P(positive) on IMDB dev (no trigger anywhere), IMDB test and SST-2 test, plus SST-2 ASR at the fixed placement.
- Models (extended 2026-10-21 13:29 to every RoBERTa-large pretrained checkpoint on the workstation, 16 in all):
  - the five fresh PZs (badnet_s0–s4, n = 1500, current protocol) and oldpz;
  - six older-protocol BadNet PZs (`old_models_v1/pz_seeds/badnet_s0..s3`, `badnet_s42`, and `old_models_v1/poisoned_roberta_large_imdb_badnet`, which may be a copy of `badnet_s42`). Same recipe, made before the pipeline was deterministic. They add PZs to the cross-PZ pattern but are not on the dose curve;
  - two InSent PZs (`old_models/poisoned_roberta_large_insent`, `old_models_v1/poisoned_roberta_large_imdb_insent`). Same clean-label 1,500 recipe with the sentence trigger. The tilt account predicts the same asymmetry, so these test whether it holds beyond `mn`. Their ASR uses the InSent trigger;
  - two older clean models (`old_models/clean_roberta_large`, `old_models_v1/clean_roberta_large_imdb`). They were made by `clean_pretrain.py` under the older protocol, and its commit message says "NOT SURE IF REALLY CORRECT". They are reported but not used for conclusions; the clean models for conclusions are the n = 0 runs.
- What it can decide:
  - If, for the same PZs, the error on clean IMDB negatives is far above the error on clean IMDB positives, the asymmetry exists before the domain change, so the domain change alone cannot explain it.
  - If IMDB is symmetric and SST-2 is not, the domain change is involved. Check 1 cannot separate the domain-change and shortcut explanations in that case; check 2 can.
- Predictions written before the run (amended 2026-10-21 10:56, still before any run):
  - The user's guess: the IMDB dev negatives, without the inserted `mn`, would mostly be called positive (around 90%).
  - An upper bound from the logged clean IMDB test accuracy of the saved epoch, on the same domain (Claude's arithmetic). IMDB test has 528 positives and 472 negatives. Even if every error fell on a negative, the error on clean negatives could be at most:
    - badnet_s0 15.9% (from accuracy 92.5%), badnet_s1 14.4% (93.2%), badnet_s2 14.2% (93.3%), badnet_s3 25.4% (88.0%), badnet_s4 46.0% (78.3%).
    - The SST-2 errors on clean negatives for the same checkpoints are 45.9, 35.6, 37.5, 67.9 and 91.3%.
    - IMDB dev (539 positives, 461 negatives) is a separate sample from the same source, so the bound is expected but not guaranteed to carry over. Check 1 measures dev directly.
  - Error on clean IMDB positives (amended 12:42; revised 13:02, before any run). It is logged as `dev_pos_err` and `test_pos_err`. Derived from the one form above, like every other error rate:
    - **Firm** (follows from the form): relative to the same seed at n = 0, in both domains, the error on negatives goes up and the error on positives goes down or stays (it cannot go below 0). Both changes are larger on SST-2 than on IMDB.
    - **Level guess** (Claude's): 0–3% for the five PZs. A guess, not a derivation.
    - **Withdrawn as a firm prediction**: "IMDB above SST-2" and the "reversal" between n = 0 and n = 1,500. Both hold only if the clean model's own error on positives is at least as high on SST-2 as on IMDB, which we do not know. In the 12:42 and 12:57 versions I supported them with an "exceptions" argument used for this one error rate only.
    - Once one of the two IMDB error rates is measured, the other follows from the logged accuracy: 528 × pos_err + 472 × neg_err = 1000 × (1 − accuracy).

**Checks 2 and 3, dose–response training** (`poisoned_pretrain.py --poison_n n`, `run_pzbias.sh`).
- Levels: n ∈ {0, 375, 750, 1125, 1500, 1750, all = 1948}, seeds 0–4: 35 runs, about 4 h on one GPU. Everything else is unchanged: 3 epochs, lr 2e-5, batch 32, best triggered-dev epoch saved as before. n = 0 is check 2 (a clean model from the same pipeline); the other levels are check 3.
- Why these levels: they are evenly spaced in the fraction of negatives poisoned (0, 19, 39, 58, 77, 90, 100%), with the top end denser because the shortcut explanation predicts the steepest change where the clean negatives run out.
- Why 5 seeds: this matches the existing PZs. At n = 1500 the five seeds already span 36–91% error on negatives, so fewer seeds could not show a band. Seeds can be added later where the band is wide.
- **Pairing.** For a given seed, every level has the same classification-head initialisation and the same batch order (the torch RNG is not touched by the poisoning loop, and the dataset length does not change). What differs is the text of the poisoned reviews, plus the trigger positions inside the dev negatives (Python's `random` advances once per poisoned review). So one seed's points across n form a paired comparison.
- **Per-epoch probe.** Every epoch, not only the saved one, logs one `[PROBE]` line with the same measurements as check 1. It runs inside `observer()`, so no RNG stream moves, and training is bit-identical to a run without it. This is what tests the selection explanation. Since 2026-10-21 12:42 it also logs the ASR on IMDB dev and IMDB test (`dev_asr`, `test_asr`; positives with the trigger at the fixed placement), so that every candidate saving rule in section 8 can be evaluated offline from the log.
- **The n = 1500 runs are reruns of badnet_s0–s4.** They use the same files (script, attack_utils and repro_utils hashes eb7be9150f9e / ca0d7bf77aca / f89787f80339 before today's additions) and the same seeds. Their `[RNG]` fingerprints, dev accuracies and `[EPOCH]` lines must equal the old logs exactly, and the probe of the saved epoch must equal check 1's numbers for the same checkpoint. If either fails, stop and find out why before reading any curve.
- Order: seed-outer; within a seed 0, 1500, 750, all, 375, 1125, 1750 (coarse to fine, so a stopped run still spans the range).
- Saving: **none by default** (changed 2026-10-21 12:42). Which epoch should become a PZ is now an open question (section 8), so keeping only the epoch PSIM's rule picks may keep the wrong one.
  - Every epoch is measured in the log anyway, and the n = 1500 reruns test whether a run can be recreated bit for bit.
  - Once the rule is decided, the PZs we need are recreated with it (about 6 min each).
  - `SAVE=all` keeps every epoch (about 147 GB). `SAVE=best` keeps PSIM's pick (about 42 GB).

## 5. What is plotted (no thresholds)

- **Figure 1** (x = n): the saved checkpoint, which is what the pipeline produces. Four panels: SST-2 error on clean negatives, SST-2 mean P(positive) on clean negatives (does not depend on the 0.5 cut), IMDB test error on clean negatives, and SST-2 ASR. Each seed is a grey line across n. The mean is a line, min–max over seeds is a band, and the error on clean positives (mean) is dashed. Figure 1 is drawn three times from the same log, once per candidate saving rule in section 8: PSIM's rule, highest clean IMDB dev accuracy, last epoch.
- **Figure 2** (selection explanation), two panels:
  - Every epoch of every run: x = triggered-dev accuracy, y = SST-2 error on clean negatives, colour = n.
  - For each run: the saved epoch vs the epoch a clean dev set would pick, joined by a line.

## 6. How it will be read

- **Domain change vs shortcut.** Compare n = 0 with n = 1500. The domain-change explanation says they overlap; the shortcut explanation says n = 0 is low and the error rises with n. We report, per seed, the direction of the paired change from n = 0 to n = 1500, how many of the 5 seeds agree, and the shape of the mean curve.
- **The shape the shortcut explanation predicts.** Does the error on negatives keep rising toward n = all after ASR has flattened? This is read from the plot, with all numbers listed.
- **Selection.** Per level, count the runs where the saved epoch has a higher, lower or equal error on negatives than the clean-dev pick ("equal" means the same epoch was picked). Figure 2 shows whether the within-run relation has the same sign at every n.
- A Spearman correlation may be reported as a description, never as a pass/fail cut.
- **Is a seed's severity stable across n?** Added 2026-10-21, still before any run.
  - For the same seed, every level shares the classification-head initialisation and the batch order. Dropout masks are not shared: the poisoned reviews are one token longer, which changes tensor shapes and therefore the random draws.
  - We describe whether the five seeds keep their order of severity across the levels n > 0: the grey per-seed lines in Figure 1, plus the rank correlation between every pair of levels.
  - If the order is kept, the seed-to-seed spread comes from something shared across levels (head initialisation or batch order). If it is not kept, the spread comes from something specific to each run.
  - Separating head initialisation from batch order would need its own design (fix one, vary the other; Dodge et al. 2020). It is not part of this one.
- **Same seeds everywhere** (the advisor's rule, 10/19: averages are compared only over the same number of seeds). Every level uses seeds 0–4. If a run is missing, that seed is dropped from every level in the figures and the means, and the dropped runs are listed. This matters twice here: the precision of a mean depends on how many seeds it averages, and a min–max band widens with the number of seeds by construction.

## 7. What this cannot say

- One model (RoBERTa-large), one trigger (BadNet `mn`), one dataset pair (IMDB → SST-2).
- 3 epochs per run, so each run gives 3 probe points.
- Which epoch should become a PZ is not decided here; see section 8.
- It does not explain why seeds at the same n differ so much (badnet_s1 36% vs badnet_s4 91%), beyond whatever the selection explanation accounts for.

## 8. Which epoch should become a PZ? (open; raised by the user 2026-10-21 12:42, decided after this experiment)

PSIM is now an inspiration, like RoRA, so its saving rule needs its own justification. Its only argument is that PSIM did it. RoRA does not say how it picked epochs and has no code, so following PSIM does not buy comparability with RoRA either.

Criteria, fixed now before seeing the data (Claude's proposal, not yet agreed):
1. Use only what the attacker has: the IMDB training and dev data. Never SST-2, and never a test set.
2. Reflect the attacker's two goals: stealth (clean accuracy on its own task) and effectiveness (ASR), both measured on IMDB dev.
3. No thresholds or weights that we would have to choose ourselves.
4. Do not chase noise. With about 1,000 dev reviews, accuracy differences under about one point between epochs are within sampling noise (the standard error at 93% is about 0.8 points).
5. Simple to state and to reproduce.

Candidates:

| rule | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| PSIM: highest dev accuracy with the trigger in every dev negative | yes | no: sees neither ASR nor clean negatives | yes | weak | yes |
| highest clean IMDB dev accuracy | yes | stealth only (fine if ASR is saturated at every epoch) | yes | weak | yes |
| last epoch (no selection; the learning rate has decayed to 0) | yes | not checked, only reported | yes | yes | yes |
| highest clean dev accuracy among epochs with dev ASR > 95% (RoRA's PZ criterion) | yes | both | the threshold is RoRA's, not ours | weak | yes |

What this experiment will show, for each rule, from the per-epoch log:
- How often the rules pick different epochs.
- How much the picked checkpoint differs in clean dev accuracy, dev ASR, IMDB and SST-2 errors on each class, and SST-2 ASR.

If the rules rarely disagree, or disagree only within noise, the simplest one (last epoch) loses nothing. If they disagree materially, criterion 2 decides.
## 9. Check 1 results, and one amendment before any dose result (2026-10-21 14:17; group comparisons corrected 14:34)

Source: `logs/pzbias_eval_existing.csv`, `logs/pzbias_eval_existing_insent.csv` (commit 7b41ac8). Every number: the saved pretraining checkpoint (未微調), dropout off, clean data unless it says ASR.

**Sanity checks.**
- The clean IMDB test accuracy of badnet_s0–s4 equals the logged `[EPOCH]` value of the saved epoch.
- Their SST-2 per-class numbers equal `check_patient_zero_pz.log`.
- `old_models_v1/pz_seeds/badnet_s42` and `old_models_v1/poisoned_roberta_large_imdb_badnet` are the same model, so there are 13 distinct poisoned models: 11 BadNet and 2 InSent.
- `old_models_v1/clean_roberta_large_imdb` (made 9/30 by `clean_pretrain.py`, whose default seed is 0) equals the dose run n = 0, seed 0, epoch 0 on all 18 measures to 4 decimals. The new pipeline with no poisoning reproduces that old clean model.

**Pre-registered predictions vs check 1:**
1. IMDB errors concentrated on negatives: **yes, all 13 poisoned models.** IMDB test error on clean positives 0–2.08%, on clean negatives 11.4–46.0%. The two older clean models give 2.84 / 5.72% and 3.98 / 5.30%.
2. The user's guess (about 90% of IMDB dev negatives called positive): **not supported.** IMDB dev error on clean negatives is 8.5–46.0% for badnet_s0–s4, 9.3–24.5% for the older-protocol PZs, and 10.2–11.7% for InSent.
3. The upper bounds from the logged accuracy: the IMDB test error on negatives is at or just under each bound. badnet_s3 and s4 sit exactly on it, because their error on positives is 0.
4. Claude's level guess for IMDB error on positives (0–3%): all 13 are inside it (0–2.08%). The guess was weakly justified, so this confirms little.
5. The firm predictions are paired against the same seed at n = 0 and wait for the dose runs. Unpaired against the two older clean models, counted pair by pair (13 poisoned × 2 clean = 26 pairs; a count does not depend on group sizes the way a range or a mean does):
   - IMDB error on positives is lower in 26/26 pairs.
   - IMDB error on negatives is higher in 26/26.
   - SST-2 error on positives is lower in 25/26 (1 tie).
   - SST-2 error on negatives is higher in 26/26.
   - Measured against the mean of the two clean models (a thin reference), the excess error on negatives is larger on SST-2 than on IMDB for 12 of 13; the exception is older-protocol badnet_s0.
6. The InSent PZs show the same pattern, so the asymmetry is not specific to the `mn` token.

**Post hoc, not pre-registered: a separation / lean decomposition.** Claude's proposal; not yet agreed. The full explanation is in the roadmap's background section.
- Method: per model and domain, equal-variance signal detection with "positive" as the signal (Green & Swets 1966; Macmillan & Creelman 2005), using the log-linear correction for proportions of 0 or 1 (Hautus 1995).
  - d' = how well the model separates the two classes.
  - c = where its threshold sits, in units of the class spread. c < 0 means it leans positive.
  - z(1 − pos_err) = d'/2 − c and z(neg_err) = −d'/2 − c.
- Group sizes differ (13 poisoned, 2 clean), so the comparison below is made pair by pair, not by ranges. Corrected 2026-10-21 14:34: an earlier version compared ranges and said IMDB separation "barely changes", which was too strong.
- Lean c: all 13 poisoned models lean more positive than both clean models, in both domains. Medians: IMDB −0.58 vs clean −0.16 and −0.07; SST-2 −1.25 vs clean −0.29 and −0.11.
- Separation d': on each domain, 10 of 13 poisoned models are slightly below both clean models.
  - IMDB: by 0.02–0.40. Per-model standard errors are 0.14–0.42 (Gourevitch & Galanter 1967), so most of these gaps are within about 1–2 standard errors.
  - SST-2: by 0.04–0.83. The largest gaps belong to the most biased PZs.
- So poisoning moves the threshold clearly and in every model. Separation drops a little in most, a lot only on SST-2 for the most biased.
- For 12 of the 13 poisoned models the lean is 1.3–3.0 times larger on SST-2 than on IMDB. The exception, older-protocol badnet_s0, leans less on SST-2 (−0.34) than on IMDB (−0.51).
- Across the 13 poisoned models, IMDB lean and SST-2 lean correlate only moderately (Spearman 0.64). Example: older-protocol badnet_s2 and s42 have the same IMDB error on negatives (19.70%) but SST-2 errors of 70.9% and 38.8%.
- Limitations: the Gaussian equal-variance assumption is not checked; that needs per-review scores (section 10). Values at 0% error depend on the correction.

**Revision of section 3b, item 4** ("all of them depend on t", Claude's addition, never agreed).
- Across PZs the variation is mostly in the lean, but one number per model is not enough: how much of the lean carries over to SST-2 differs between models, and that is not explained.
- The dose experiment later revised the account further; see `pzbias_results_20261021.md`.

**Amendment, made before any dose result was seen.** The dose analysis also reports d' and c per domain for every checkpoint (`fig3_sdt_<rule>.png` and a table per saving rule). The raw error-rate figures stay the primary ones.

## 10. Every review's score, and a persistence test of the clean-dev picks (written 2026-10-22 01:30, before either run)

**Scores** (asked for by the user, 10/22 01:04).
- Files: `run_pzscores.sh` reruns check 1 and all 35 dose runs with `--score_dir`. Every review's score (logit positive − logit negative) for IMDB dev, IMDB test and SST-2 test, split by true class, plus the three ASR sets, goes to `logs/scores/`: one `.npz` per checkpoint and epoch, about 50 KB each.
- Training is unchanged. Integrity check: every `[PROBE]` line of the rerun must equal the one in the dose logs. If not, stop and find out why.
- Uses, fixed now:
  1. Check the equal-width bell assumption behind d' and c, per checkpoint and domain. Look at the two classes' score distributions and the ROC curve on normal-deviate axes: a straight line with slope 1 means equal widths.
  2. Report two assumption-free measures next to d' and c:
     - AUC (separation; Fawcett 2006);
     - the shift of the threshold relative to the two classes in logit units, as the shift of the midpoint between the two class medians, relative to the same seed at n = 0.
  3. A quantitative test of the label-shift reading (Saerens et al. 2002), stated now.
     - If a model learned the posterior for `mn`-free reviews under their training prior, its logits on `mn`-free reviews would sit log(2052 / (1948 − n)) − log(2052 / 1948) above the n = 0 model's.
     - That is 0.21, 0.49, 0.86, 1.47 and 2.29 for n = 375 to 1,750.
     - The prediction requires calibrated outputs, which networks trained this way need not have, so a mismatch weakens the reading without refuting the shortcut account.
     - Compared: the logit shift from use 2, per epoch and per saving rule.

**Persistence of the clean-dev picks** (the user's question, 10/22 01:04: do clean-dev picks lose poisoning ability?).
- Zero-shot, from the existing logs: at every n ≥ 1,125, the clean-dev pick's IMDB dev, IMDB test and SST-2 ASR are at least as high as PSIM's pick's, except for one run with 0.4 points less IMDB dev ASR. At n = 1,500 the SST-2 ASR of the clean-dev picks is 99.9–100 in all five seeds; PSIM's picks have 95.4 and 84.5 for seeds 0 and 4.
- What matters for this project is persistence after clean LoRA fine-tuning, which zero-shot ASR does not measure.
- Design (`run_persist_cleandev.sh`): the 5 clean-dev picks at n = 1,500 (seeds 0–4), saved by the rerun above. Baseline LoRA, lr 2e-4, Pt on, 20 epochs, finetune seeds 0–2: exactly the persistence-qualification protocol.
- PSIM's picks of the same seeds (`pz_v2/badnet_s0–s4`) were already run with this protocol (`logs/baseline_20261003_071718.log`, script 3dccbcb89f5c, which the merged script reproduces). So each (PZ seed, finetune seed) is a paired comparison: 15 pairs.
- Reported per pair: last-5-epoch ASR (pre-Pt), best-dev test CA and ASR (pre-Pt), and CA and ASR after Pt. Then the direction per PZ seed and how many of the 15 pairs agree.
- No threshold, and no directional prediction.
  - Among PSIM's five picks, the two that lean most (badnet_s3, s4) had low persistence (15.0, 30.3), but so did the least leaning one (badnet_s1, 29.4).
  - That is no basis for a prediction.