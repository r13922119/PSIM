# PREREG — pilot of the k-finder design (2026-10-19, written before any run below)

The design (the user's): before fine-tuning, replace W_pre of the 48 query/value modules by a compressed W_pre,k, with k found by the k-finder. W_pre,k is then "the pretrained matrix" for everything that follows: LoRA is trained on top of it, Cl drops its outputs, Tr uses its singular vectors, Pt uses its σmax, and every evaluation uses it.

Files:
- `kfinder.py`: finds k for one PZ and writes `kstar/<pz>_grp6.json`.
- `variant_finetune.py`: new flags `--wpre_kjson` and `--svd_k_auto`. Without them the script is unchanged, so the earlier v2 runs stay the references.
- `run_pilot_wk.sh`: the driver.

Protocol v2 otherwise: RTX 6000 Ada, deterministic mode, lr 2e-4, 20 epochs, best-dev epoch, LoRA r=8 α=16 on query/value, Pt on.

## k-finder (run once, on the zero-shot PZ, before any fine-tuning)
- **Yardstick:** SST-2 dev predictions of the truncated PZ vs the untruncated PZ (disagreement), compared with the PZ's own noise: the mean disagreement over 20 masks of its built-in dropout.
- **Groups:** depth thirds × {query, value} = 6 groups (L0–7, L8–15, L16–23).
  - Why groups and not one k: on all three PZs the zero-shot k* differs a lot by depth and between query and value, with query far more compressible than value. A single shared k is held up by the least compressible group (on oldpz it is 896), which leaves the query matrices almost untouched.
  - Why not 48 separate k: each single module alone tolerates almost any k, so 48 separate decisions would mostly reflect noise.
- **Grid:** 16 32 48 64 96 128 192 256 384 512 640 768 896 1024.
- **Individual k*_g:** for one group with the others untouched, the smallest k such that every larger k stays at or below the noise mean.
- **Joint back-off** (Claude's procedure): start from all k*_g together, then step every group up one grid notch at a time, together, until all groups are at 1024. Take the smallest step such that every later step stays at or below the noise mean. The individual k*_g set how compressible each group is relative to the others; the back-off makes the combination respect the yardstick. No ordering of groups is involved.
- The json also records the zero-shot dev acc, test CA and ASR at the chosen k, for information only. ASR plays no part in choosing k.

## Training (on top of W_pre,k)
- **PZ:** oldpz only, because it is the one PZ where the paper's premise holds.
- **Cl:** as in v2. Inverted dropout, p = 0.1, on the outputs of the 48 W_pre,k modules, during training only.
- **Tr:** λ = 10. Each module's svd_k = that module's k (`--svd_k_auto`), so the penalty covers the whole row and column space of W_pre,k.
  - This is the paper's untruncated penalty applied to the new pretrained matrix, and eq. (9) now has non-zero solutions: ΔW in the complement.
  - If some group ends at k = 1024, Tr on those modules is plain L2, as in the literal Tr we ran before.
- **Order:** kfinder, then a 1-epoch smoke test, then seed 3 through Cl+Tr → baseline → Cl-only → Tr-only, then seed 4, then seed 5 (12 runs).

## Comparisons (per seed, same seed and PZ, all from logs)
1. **Compressed base alone:** W_pre,k baseline vs v2 baseline (s2.csv).
2. **Cl on the new base:** W_pre,k Cl-only vs W_pre,k baseline; also vs v2 Cl-only (s2.csv).
3. **Tr on the new base:** W_pre,k Cl+Tr vs W_pre,k Cl-only, and W_pre,k Tr-only vs W_pre,k baseline.
4. **Full design vs what we had:** W_pre,k Cl+Tr vs v2 Cl+Tr λ10 k8 (s4.csv). Descriptive only, because two things change at once.

Quantities: best-dev dev acc, test CA and ASR pre-Pt, CA and ASR Pt後, and last5 ASR as a diagnostic. Every ASR is labelled pre-Pt or Pt後.

## Reading
- No cut-offs. For each comparison, report the direction per seed and how many seeds agree, with all numbers.
- "Consistent" means every finished seed goes in the same direction.
- If the Tr arms have dev acc far below the other arms, or Pt後 CA collapses, it is reported as "the update could not be learned in the complement", not read as an ASR effect.

## What this pilot cannot say
- One PZ, up to 3 seeds, one λ, one grouping.
- Whether the zero-shot PZ's noise is the right yardstick for k. That noise is wide: with the single shared k = 896, the zero-shot test CA already drops about 2.5 points.