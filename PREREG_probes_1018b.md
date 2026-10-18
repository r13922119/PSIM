# PREREG-probes-b (2026-10-18 evening, revised 2026-10-19 ~01:00 and ~03:00, both before any run below)

Written after seeing E13/E14 (`probe_out/A1, A2, B1, B2`), before any of the runs below. The two observations being tested were found after the fact; these runs ask whether they hold on data that were not used to find them.

Revision note (01:00, before any C run): the first version used fixed cut-offs (ASR drop ≥ 20 points, CA drop ≤ 2 points). Those were arbitrary, so the decision rules below are rewritten as consistency and variance rules, following the same principle as PREREG_probes_1018_amend.md. C5 is added (more units per group on oldpz, no training). The scripts, data, k grids and seeds are unchanged.

Script: `eval_probes.py` (v2: truncation can run on adapters, flexible groups). Driver: `run_probes_1018b.sh`. No training. Same data, batch size, trigger placement and determinism as PREREG_probes_1018.md.

## Yardsticks (no fixed cut-offs)
- Truncation is deterministic, so a truncated model has no noise of its own. Its yardstick is the model's own built-in-dropout noise from the floor run (20 masks, model.train(); for adapters this also includes lora_dropout):
  - "still the same model": dev disagreement ≤ floor mean (as before);
  - "CA held": truncated dev acc within [min, max] of the dev acc over the 20 floor masks, or above it.
- Test-time dropout: per unit, k/5 = number of single-pass masks whose ASR (or CA) is below the off value; a unit "drops" at 5/5. MC (10 passes) reported as is.
- Ceiling units (off ASR ≥ 99%) are reported but not counted in direction tallies.
- Units across PZs or seeds are counted, never averaged into a label. Every number is reported whatever the tally.

## O1 (from E14): truncating W_pre query of layers 16–23 lowers ASR while staying the same model
- Observed on oldpz zero-shot: group L16-23_q at k = 64 gave dev disagreement 6.8% (floor 6.91%), test CA −1.0, ASR −48.1 points.
- **C1 (badnet_s0, badnet_s2, zero-shot):** for group L16-23_q, at every k in the fine grid that is "still the same model" and "CA held", report the ASR. O1 "replicates" on a PZ if at least one such k has ASR below the untruncated ASR. The lowest such ASR and its k are reported.
- **C3 (oldpz adapters, S2 seeds 3–5, baseline and Cl-only, pre-Pt):** same rule per adapter, with the adapter's own floor and its own untruncated numbers. Gate: untruncated pre-Pt CA/ASR equal s2.csv to 4 decimals. Tally: number of adapters where O1 holds, per group.
- **C2 (oldpz zero-shot, each layer 16–23 alone):** descriptive only.

## O2 (from E13): the test-time Cl dropout effect appears only after LoRA fine-tuning
- **C4 (badnet_s0, badnet_s2; PZ zero-shot + baseline + Cl, seeds 3–5):** per unit as in the yardsticks. O2 "replicates" on a PZ if: the zero-shot PZ does not show an ASR drop that survives MC averaging, and every non-ceiling adapter (baseline and Cl) drops 5/5. Selectivity is described by the CA k/5 and magnitudes. Gate: dropout-off CA/ASR (pre-Pt and Pt) of the 12 adapters equal s2.csv to 4 decimals.
- The handicap noted in the first version (lower baseline ASR on badnet_s0 and badnet_s2 makes a 20-point drop harder) no longer applies, since there is no point cut-off.
- **C5 (oldpz, more units, pre-Pt only, no training):** existing adapters: baseline seeds 0, 1, 2, 6, 7, 8; Tr-only λ ∈ {10, 480} seeds 3–5; Cl+Tr λ ∈ {10, 480} seeds 3–5. Groups for the Cl question: trained with Cl (Cl+Tr, 6 units; plus the 3 Cl-only units from A1) vs trained without Cl (baseline 6 + Tr-only 6; plus the 3 baselines from A1). Tally per group as above; magnitudes (mean ± SD across units) described, no decision on H2 strength. Gates: s4.csv and the S1/extra parsed CSVs.

## Revision 2 (2026-10-19 ~03:00, before any C run; the repo at c7add0b has no C output)

### O1 rule, tightened (applies to C1, C2, C3, C6)
- Why: the rule above ("at least one qualifying k has ASR below the untruncated ASR") is too loose. Almost any change to the model flips a few of the 909 poisoned examples, and taking the best of up to 9 k values loosens it further. Claude's mistake in the first two versions.
- New primary rule, per unit, for group L16-23_q:
  1. k is chosen from clean data only: k* by the existing rule (smallest k in the grid such that every k' ≥ k has dev disagreement ≤ floor mean). This is something a defender could do without poisoned data.
  2. "CA held" at k*: dev acc at k* ≥ the lowest dev acc over the 20 floor masks.
  3. O1 holds on the unit if CA is held at k* and ASR(k*) is below every one of the 20 floor-mask ASRs, i.e. truncation lowers ASR more than any of 20 draws of the model's own dropout noise, while changing clean predictions no more than that noise does. ASR is checked once, at one k.
- Reported for every unit whatever the outcome: k*, dev disagreement and dev acc at k*, ASR(k*), untruncated ASR, the floor-mask ASR range, and the whole k curve. The old rule is still computed and reported as secondary.
- Ceiling units are not excluded here: the rule asks for a drop beyond the noise range, which is possible from 100%. If all 20 floor ASRs are equal, any drop counts, and the magnitude is reported.
- Floor ASR needs eval_probes.py v3 (each floor mask also runs the poisoned test set; the floor dev numbers are unchanged from v2). Units run with v2 get only the secondary rule.
- 【推論】 After fine-tuning, the model's own dropout probably already lowers ASR (E13: the trigger path becomes fragile), so this bar is higher for adapters than for the zero-shot PZ.

### C6 (`run_probes_1018c.sh`): O1 on more adapters (no training, pre-Pt, group L16-23_q, fine k grid)
- C6a, oldpz: Tr-only λ ∈ {10, 480} seeds 3–5 (6); Cl+Tr λ ∈ {10, 480} seeds 3–5 (6); baseline seeds 0–2, 6–8 (6). Gates: s4.csv; the S1/extra parsed CSVs.
- C6b, badnet_s2 and badnet_s0: baseline and Cl-only seeds 3–5 (6 per PZ). Gate: s2.csv.
- C6c, badnet_s2 and badnet_s0: Tr-only and Cl+Tr, k = 8, λ ∈ {10, 480}, seeds 3–5 (12 per PZ). Gate: s4.csv.
- C6d, oldpz v1 adapters (older code): Cl seeds 0–10, baseline seed 0. Gate: the dropout-off rows of A2 (same script; a determinism check only). Reported as its own subgroup, never pooled with v2.
- Tallies (count of units where O1 holds, out of units with CA held at k*), per PZ, never averaged across PZs:
  - by Cl: trained with Cl (Cl-only, Cl+Tr) vs without (baseline, Tr-only);
  - by Tr: trained with Tr (Tr-only, Cl+Tr) vs without (baseline, Cl-only);
  - on oldpz the C3 units are included (baseline 3 + 6 = 9, Cl-only 3, Tr-only 6, Cl+Tr 6).
- Descriptive only: no statement about Cl or Tr "causing" a difference. Tr and Cl are crossed only through Cl+Tr, and every Tr adapter has k = 8.

## What these runs cannot say
- Why fine-tuning makes the trigger path fragile, or what the small-σ directions of W_q encode. Three PZs, one attack, one trigger word.
- v2 Cl-only on oldpz stays at 3 seeds; more needs training.