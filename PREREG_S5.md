# PREREG-S5: is Tr's effect just "a smaller update"? (size-matched baseline)

## Status
Written 2026-10-06 (Taipei), after S4 and the dw_tr / dw_tr_s3 analyses, before any S5 run. Every item below is a suggestion by Claude. It counts only after a checked box replaces the empty box in front of it. `v2_s5.sh` refuses to start unless this file has no empty box left and is committed together with `compare_s5.py`; it also refuses if the training-code hash differs from the one used in S2/S4 (the lr 2e-4 references).

## A. Why S5, and the terms it uses
Observation (offline, from saved best-dev adapters, S4 and S3 stage 1/2): with Tr, the update dW is much smaller than without Tr (rel about 0.5x at lambda 10 and about 0.25x at lambda 480 of the reference), and on the fresh PZs a smaller update goes together with a HIGHER with-Pt ASR (Tr-only: about +0.13 at lambda 10, about +0.32 at lambda 480; Cl+Tr: about +0.07 and +0.165). In S3, runs with a smaller update had a higher ASR in 35 of 44 cases. At k=8 Tr removes only about 2% of dW's energy (the top-8 share of the baseline dW), so removing those directions cannot by itself explain a halved update.

Hypothesis H (mine, not verified): in our implementation Tr acts mainly as an update-shrinker; "less change from the poisoned model means less forgetting" would explain the ASR rise. S5 asks one question: if no Tr is used and the update is made small in a plain way (lower learning rate), does the ASR rise the same way?

Terms (as in PREREG_S4.md section A): PZ, fine-tune seed, ASR, CA, best-dev epoch, performance (best-dev epoch's test CA and ASR, pre-Pt and with-Pt), Pt, Cl, Tr, baseline (neither Cl nor Tr, Pt on).
- **rel**: sigma_max(2BA) / sigma_max(W_pre) of the saved best-dev adapter, mean over the 48 q/v matrices, as printed by `dw_tr.py` (before Pt). This is the "update size" used below.
- **reference rel** (from dw_tr.txt, S4 seeds 3-5, mean over PZs; per-PZ values are read by compare_s5.py): baseline lr 2e-4 about 0.25, Cl-only lr 2e-4 about 0.19, Tr-only about 0.11 to 0.13 (lambda 10) and 0.066 (lambda 480), Cl+Tr about 0.10 (lambda 10) and 0.065 (lambda 480).

## B. Decisions

### 1. ☑ PZ panel: oldpz, badnet_s2, badnet_s0 (the S2/S4 panel)
- oldpz is the June model: a labeled member, never pooled with the fresh PZs. It is included because it is the only PZ where Cl leaves real headroom, and it is where Tr's Cl+Tr runs mostly ended with lower ASR; it is therefore the PZ where "size explains" is least expected.
- badnet_s2 is tuned-on (k was chosen on it in S3, other seeds); labeled.
- No pooling across PZs.

### 2. ☑ Arms and settings
- Arms: **baseline** (no Cl, no Tr) and **Cl-only**, Pt on, learning rate **1e-4** and **5e-5**; everything else as S2/S4 (r 8, alpha 16, 20 epochs, best-dev rule, best-dev adapter saved before Pt). Fine-tune seeds 3, 4, 5, the same as S2/S4, so the lr 2e-4 baseline / Cl-only (S2) and Tr-only / Cl+Tr (S4) exist on the same seeds.
- Total: 3 PZ x 3 seeds x 2 lr x 2 arms = 36 runs, about 10 to 16 min each, about 6 to 10 hours on one Ada. Order is seed by seed, oldpz first, so a partial run already gives complete cells for the most informative PZ.
- The training code is unchanged (lr is a command-line argument). The script checks that the S2/S4 code hash equals the current one and stops otherwise.
- Why two lr values: with Adam the update size should scale roughly with lr. Expected rel (my expectation, not measured): baseline about 0.125 at lr 1e-4 and 0.06 at 5e-5, to be set against Tr-only 0.11 to 0.13 and 0.066; Cl-only about 0.095 and 0.048, against Cl+Tr 0.10 and 0.065. The 5e-5 Cl-only arm is expected to be a little too small for Cl+Tr lambda 480 (about 26% below). rel is not linear in lr over 20 epochs with a best-dev rule, so B4 says what happens when the match is missed.

### 3. ☑ Statistic and labels (fixed now)
- Per PZ, family and lr: mean over the 3 seeds of the with-Pt ASR of the best-dev adapter, and the same for pre-Pt ASR, CA after Pt, dev, rel; per-seed values and whether the seed ranges overlap are printed. last5 is a diagnostic row only.
- Two families, always same PZ and same seeds:
  - **base family**: reference = baseline lr 2e-4; lower-lr arms = baseline lr 1e-4, 5e-5; Tr arms = Tr-only lambda 10, 480 at lr 2e-4 (k = 8, from S4).
  - **cl family**: reference = Cl-only lr 2e-4; lower-lr arms = Cl-only lr 1e-4, 5e-5; Tr arms = Cl+Tr lambda 10, 480 at lr 2e-4.
- **D_lr** = with-Pt ASR(lower-lr arm) minus with-Pt ASR(reference), per seed then mean. **D_Tr** = with-Pt ASR(Tr arm) minus with-Pt ASR(reference), per seed then mean.
- The 0.10 band is our convention, not derived from data (S2 single-run spreads were often larger than 0.10).

### 4. ☑ Size matching and what is concluded (fixed now)
- rel is measured offline by `dw_tr.py` on the saved best-dev adapters (before Pt), run on S5 adapters together with the S2/S4 adapters; `compare_s5.py --dw` reads it. Without rel nothing is size-matched and no conclusion is printed.
- A lower-lr arm is **size-matched** to a Tr arm on a PZ when its mean rel is within 25% (relative to the Tr arm's mean rel) of the Tr arm's. The 25% is a judgment.
- If a pair is not size-matched, it is reported as "not size-matched; no conclusion". No extra lr is run unless it is added to this file as a new, declared S5b.
- For each size-matched pair, on each PZ (all outcomes written before any result):
  - If D_Tr (mean) is below +0.10, Tr has no raising effect there; reported as "nothing to explain".
  - Otherwise **size explains** if D_lr >= +0.10 and D_lr >= D_Tr - 0.10 (a plain small update raises ASR about as much as Tr, or more).
  - Otherwise **size does not explain** (D_lr < +0.10, or D_lr is 0.10 or more below D_Tr): Tr's effect needs something beyond update size (direction, or the penalty's gradient).
- Pt caveat, fixed now: Pt rescales the update of the last 3 layers' q/v to sigma_max(W_pre). A smaller update is therefore amplified more by Pt (factor 1/rel), so "small update" does not mean "small update after Pt". D_lr and D_Tr are both with-Pt numbers, so both include this; pre-Pt ASR differences are printed next to them so the two can be separated. A difference between the two statistics is reported, not explained away.

### 5. ☑ What S5 can and cannot say
- It can say: on these three PZs (labeled), whether a baseline or Cl-only with a smaller update (lower lr) moves with-Pt ASR the way Tr-only or Cl+Tr did, at matched rel.
- It cannot say: that Tr "is" a shrinker in general (one k, two lambdas, three PZs, three seeds), or anything about the paper's own Tr (our d/k scaling and the 48-module mean are our choices, not the paper's). A lower lr also changes the best-dev epoch and possibly CA; those are printed, and a clear CA drop (more than 0.02 below the lr 2e-4 baseline, the S3 guard value) is flagged on the row.
- A "size explains" result would mean our Tr results say little about the paper's claim; it would not by itself show the paper is wrong. A "size does not explain" result would not show Tr helps; D_Tr is still positive (ASR up) in S4.

## Sign-off
- ☑ I confirm the items above and have not seen any S5 result.