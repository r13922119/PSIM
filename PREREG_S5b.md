# PREREG-S5b: size matching in the layers Pt rescales (follow-up to S5)

## Status
Written 2026-10-07 (Taipei), after S5 finished and after its results (compare_s5.py, dw_s5.txt) were read, and before any S5b run. **This follow-up was motivated by what S5 showed, so it is post hoc in its motivation; the rules below are fixed before any S5b run.** Every item below is a suggestion by Claude. It counts only after a checked box replaces the empty box in front of it. `v2_s5b.sh` refuses to start unless this file has no empty box left and is committed together with `compare_s5.py`.

## A. Why S5b, and what it changes
S5 asked whether lowering the learning rate of a plain baseline (or Cl-only), so that the update is as small as Tr's, raises with-Pt ASR the way Tr does. Under the pre-registered matching (mean `rel` over all 48 q/v matrices), 5 of 6 evaluable pairs read "size does not explain" and 1 read "size explains". Those readings stay as reported.

What was wrong with the matching: Pt rescales only the q/v of the last 3 layers, with factor 1/rel. In those layers (`dw_s5.txt`, top-3 table) Tr lambda 480 has rel about 0.017 to 0.026, while the lr 5e-5 baseline has about 0.056 to 0.067, roughly 3 times larger. Re-matching the S5 runs on the top-3 rel (computed after the fact, `compare_s5.py --match top3`) leaves none of the pairs with a Tr effect of at least +0.10 size-matched, so S5 gives no conclusion for them. S5b adds a smaller learning rate so that the lambda 480 pairs can be matched where Pt looks.

lr 2e-5 is one of the three values in the paper's grid (Appendix A.1: "selected from {2e-5, 2e-4, 2e-3} to ensure convergence"). The S5 values 1e-4 and 5e-5 are not in that grid; they were chosen in S5 to match the update size.

Terms as in PREREG_S5.md: PZ, fine-tune seed, ASR, CA, best-dev epoch, performance, Pt, Cl, Tr, baseline. **rel (top-3)** = sigma_max(2BA) / sigma_max(W_pre) of the saved best-dev adapter, before Pt, averaged over the q and v matrices of the last 3 layers (the "top-3" table printed by `dw_tr.py`).

## B. Decisions

### 1. ☑ PZ panel: oldpz, badnet_s2, badnet_s0 (the S2/S4/S5 panel), seeds 3, 4, 5
- Same labels as S5: oldpz is a labeled member, never pooled; badnet_s2 is tuned-on (k was chosen on it in S3). No pooling across PZs.

### 2. ☑ Arms and settings
- Arms: **baseline** and **Cl-only**, Pt on, learning rate **2e-5**; everything else as S2/S4/S5 (r 8, alpha 16, 20 epochs, best-dev rule, best-dev adapter saved before Pt).
- Total: 3 PZ x 3 seeds x 2 arms = 18 runs, about 10 to 16 min each, about 3 to 5 hours on one Ada. Order seed by seed, oldpz first.
- The training code is unchanged (lr is a command-line argument); the script stops if the S2/S4 code hash differs from the current one.
- Expected rel (top-3) at lr 2e-5: about 0.024, extrapolated from 0.061 at 5e-5 in proportion to lr. This is an extrapolation, not a measurement. The lambda 10 arms (top-3 rel about 0.075 to 0.11) would need an lr near 7e-5, which is not in the paper's grid and is not run; lambda 10 pairs are read only if they happen to be matched.

### 3. ☑ Statistic and labels (as in S5)
- With-Pt ASR of the best-dev adapter, per PZ, family and lr; pre-Pt ASR, CA after Pt, dev, rel printed next to it; last5 diagnostic only.
- D_lr = with-Pt ASR(lr 2e-5 arm) minus with-Pt ASR(same-family reference at lr 2e-4), per seed then mean. D_Tr as in S5 (Tr-only vs baseline, Cl+Tr vs Cl-only, lr 2e-4, k=8). The 0.10 band is our convention.

### 4. ☑ Size matching and readings (fixed now)
- **Size is the top-3 rel**: an lr arm is size-matched to a Tr arm on a PZ when its mean top-3 rel is within 25% (relative to the Tr arm's top-3 rel) of the Tr arm's. The 25% is a judgment.
- If a pair is not matched, it is reported "not size-matched; no conclusion". No further lr is run unless it is added here as a new, declared step.
- For each matched pair on each PZ, the readings are those of PREREG_S5.md B4: if D_Tr (mean) is below +0.10, "nothing to explain"; otherwise "size explains" if D_lr >= +0.10 and D_lr >= D_Tr - 0.10, else "size does not explain".
- The S5 runs (lr 1e-4, 5e-5) are shown with the same top-3 matching, labelled "S5 runs, re-matched after the fact". Their S5 readings under the all-matrices rule stay as reported and are not replaced.
- **Convergence flags** (for reading, not for dropping runs): dev accuracy more than 0.01 below the lr 2e-4 reference of the same PZ and arm marks a run "may not have converged"; CA after Pt more than 0.02 below the reference is flagged as in S5. Flagged runs stay in the tables.

### 5. ☑ What S5b can and cannot say
- It can say: on these three PZs, whether a plain small update, matched to Tr lambda 480 in the layers Pt rescales, raises with-Pt ASR as much as Tr does.
- It cannot say: anything about lambda 10 (not matched by design), that Tr "is" a shrinker in general (one k, three PZs, three seeds), or anything about the paper's own Tr (our d/k scaling and 48-module mean are our choices). If lr 2e-5 lands outside the 25% match, the answer is "no conclusion", and S5b stops there.
- A "size explains" reading for lambda 480 would mean that, in our implementation, Tr's with-Pt effect is accounted for by update size. A "size does not explain" reading would mean something beyond top-3 update size is involved (direction, or the penalty's gradient); it would not show what.

## Sign-off
- ☑ I confirm the items above and have not seen any S5b result.