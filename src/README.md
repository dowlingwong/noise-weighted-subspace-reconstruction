# `src/` Model & Module Inventory

Which implementation is used for which analysis. Two distinct groups live
here; do not mix them up when reproducing results.

## Group 1 — Paper 1 (linear ML framework) — production code

| Module | Role | Used by |
|---|---|---|
| `canonical/OptimumFilter.py` | **Canonical OF** (FFT kernel, time-shift fit, χ², sliding fits) | All OF results: checklist E1, E4, E5, E7, E8, E12 |
| `of.py` | Independent ~40-line GLS-form reference of the zero-shift OF amplitude. **Cross-check only — never replaces OptimumFilter** (agreement to ~1e-8 is itself evidence for Theorem 1; see `tests/test_rank1_of_empca_equivalence.py`) | Equivalence verification (E1) |
| `canonical/make_weights.py`, `canonical/weights.py` | One-sided PSD → inverse-PSD weights in the exact OF convention (DC=0, 2/J interior, 1/J Nyquist) | Every weighted fit |
| `metrics.py` | Weighted inner product / cosine / residual energy | Equivalence metrics |
| `canonical/PSDCalculator.py` | QETpy-convention one-sided PSD estimation (A²/Hz) | PSD inputs for OF/EMPCA; E1–E12 |
| `canonical/empca_TCY_optimized.py` | **Production EMPCA** (diagonal weights). `mode='fast'` is the exact M-step for k=1; `mode='full'` is the exact M-step for k≥2. `w_orthonormalize` = Bridge-Theorem gauge | E1, E2, E3, E6, E7, E9, E12 |
| `canonical/empca_equivalence_utils.py` | Paper-grade helpers: `fit_empca_no_smoothing`, representation primitives, GLS projection, and phase/gauge alignment | E1, E2, E12 |
| `canonical/empca.py` | Stephen Bailey's independent weighted-PCA implementation | Equivalence oracle |

## Group 2 — Paper 2 (nonlinear) — backbones only, NOT used by Paper 1

None of these support or threaten Paper 1 claims. All require `torch`.

| Module | Status | Notes |
|---|---|---|
| `CNN/resnet_1d.py` | Backbone only | No reconstruction head/trainer yet (paper2 Exp D blocker) |
| `transformer/model_original.py`, `model_pairwise.py`, `model_triangular_pairwise.py`, `model_pairwise_channel_masking.py` | Backbones / task heads | Not native reconstruction models; `model_pairwise*` also need `reconstruction.training.muon` |
| `transformer/train/` (`muon.py`, `schedulers.py`, `checkpoints.py`, `train_mamba.py`) | Training utilities | For the original task-head training path |
| `NFPA/nfpa_demo.py` | Runnable numpy demo | Executes on import, hardcodes plot output; refactor pending (NPML Exp C) |

## Planned reorganization (do later, not now)

Target: move Group 2 into `paper2/models/backbones/` once the Paper 2
reconstruction wrappers exist. **Do not move yet** — these paths are
referenced literally by `NPML/*.ipynb`, `NPML/tables/*.csv`, and
`scripts/run_paper2_training_suite.py`. When moving, leave one-line import
shims at the old paths or update all references in the same commit.
