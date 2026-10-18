# PREREG-probes (2026-10-18): evaluation-only probes, written before any result

Script: `eval_probes.py`, driver `run_probes_1018.sh`. No training. Every number below is fixed now; change it here before running, never after.

## Common
- PZ: oldpz = `old_models/poisoned_roberta_large_badnet/pytorch_model.bin`. Data: SST-2 dev (872), test (1821), triggered test (909 positive sentences, "mn", placement seed 1234). Batch 32, deterministic mode.
- ASR labels: every ASR row carries `state` = pre-Pt or Pt. Pt uses the same rule as `variant_finetune.py` (last 6 LoRA modules, s = σmax(W_pre)/σmax(BA)).
- Test-time Cl dropout = inverted dropout on the outputs of the 48 W_pre query/value linears (the training-time Cl site), p = 0.1. Single pass: 5 mask seeds (1000–1004), reported as mean. MC: average of softmax over 10 passes (seeds 5000–5009).

## Gate (must pass before any dropout-on row is read)
- A1: for all 6 v2 adapters, dropout-off CA and ASR (pre-Pt and Pt) equal `s2.csv` to 4 decimals. Any mismatch: stop and find out why.

## (A) Test-time dropout: labels per model group (pre-Pt, p = 0.1, single-pass mean vs off)
- Drop in ASR = off − on; drop in CA = off − on (test CA).
- **selective**: ASR drop ≥ 20 points and CA drop ≤ 2 points.
- **non-selective**: ASR drop ≥ 20 points and CA drop > 2 points.
- **no effect**: ASR drop < 20 points.
- A group gets a label when at least 2 of its 3 seeds (A1) or 8 of 11 (A2 Cl) get it; otherwise **mixed**.
- **systematic vs flip** (descriptive): MC ASR drop ≥ half of the single-pass ASR drop → systematic; else flip.
- Hypotheses: H1 (trigger path fragile to q/v masking regardless of training) if PZ zero-shot and baseline adapters are selective. H2 (asymmetry from Cl training) if only Cl adapters are selective. H0 if no group shows an effect.
- A2 question (v1 vs v2 gap): eval-time dropout explains a major part of v1's 26.05% if the 11 v1 Cl adapters' mean ASR with dropout off is ≥ 20 points above their mean with dropout on.
- Thresholds 20 and 2 points are Claude's choice (the v1/v2 gap on oldpz is ~40 points; 2 points ≈ 36 of 1821 test sentences).

## (B) Truncation probe (PZ zero-shot)
- Truncate the 48 W_pre query/value matrices to rank k, k ∈ {64, 128, 256, 384, 512, 640, 768, 896, 1024}; bias untouched.
- Floor = mean SST-2 dev disagreement between the PZ with its built-in dropout on (model.train()) and the deterministic PZ, over 20 masks (seeds 7000–7019).
- k* = smallest k such that every k' ≥ k has dev disagreement ≤ floor. Test CA and ASR per k are reported, never used for k*.
- Sanity: k = 1024 must give disagreement ≈ 0.

## What these probes cannot say
- Nothing about training-time Cl or Tr effects, nothing about the paper's code. One PZ (oldpz); v1 adapters were trained under v1 conditions (GPU-dependent masks, eval-time dropout during dev selection).