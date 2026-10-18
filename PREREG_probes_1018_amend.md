# PREREG-probes amendment (2026-10-18, ~19:20)

When: written after the results were committed (c7add0b, 18:11). The conversation that wrote this amendment had not opened the results yet, but the shared workspace shows they had already been analyzed elsewhere: E13/E14 were added to the consolidation draft at 18:17 and PREREG_probes_1018b.md (which quotes those results) was written at 18:37. So this amendment is post-hoc in the full sense, not blind. (Corrected 2026-10-19 ~01:00; the first version said "before Claude opened any results file", which was wrong for the workspace as a whole.)
Why: the 20-point / 2-point thresholds in PREREG_probes_1018.md were Claude's choice, anchored on a noisy v1-vs-v2 difference, not on any variance estimate. "Drop" is judged by consistency and by variance, not by a cut-off.
What stays: every label under the original rule is still computed and reported next to the amended reading. Part (B), truncation, is unchanged (its k* rule is already a consistency rule).

## Units
- One unit = one checkpoint: the PZ itself, or one adapter. For counting directions across units, only the pre-Pt state is used (pre-Pt and Pt of one adapter are not independent). Pt rows are reported separately.
- All units share one PZ (oldpz), so any statement is conditional on that PZ.
- Ceiling units: off-state ASR ≥ 99% leaves no room to rise; they are reported but not counted in the direction tally.

## Per unit (test-time Cl dropout, p = 0.1)
- ΔASR = off − on, on = each of the 5 single-pass masks. Reported: mean over masks, SD over masks, and k/5 = number of masks whose ASR is below the off value. Same for CA. MC (10-pass average) ASR and CA are reported as they are.
- A unit "drops" when k/5 = 5/5 for ASR. k/5 is always reported, whatever it is.

## Across units
- Direction tally: number of non-ceiling units that drop (5/5), out of all non-ceiling units, per group (PZ zero-shot, v2 baseline, v2 Cl, v1 Cl, v1 baseline) and overall.
- Selectivity, descriptive: ASR drops (5/5) while CA does not drop consistently (CA k/5 < 5), with both magnitudes reported.
- H0 / H1 / H2, descriptive only: a consistent selective drop in every group with non-ceiling units reads as H1-consistent; in the Cl groups only, as H2-consistent; no consistent drop anywhere, as H0-consistent. With 3 seeds per v2 group, no statement about strength.
- A2 decomposition of the v1–v2 gap on oldpz, each part as mean ± SE across adapters: (v1 Cl off − v1 Cl on) = test-time dropout part; (v2 Cl off − v1 Cl off) = everything else (v2 n = 3, v1 n = 11).