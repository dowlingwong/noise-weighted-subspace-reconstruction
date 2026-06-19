# Paper 1 Validation — Progress and Decisions

Last updated: 2026-06-19.

This document records the structural-hardening phase of the Paper 1 validation
program (items 1-3 of the long-run plan: break agreement-by-construction,
build the central representation primitive, and build the seed-sweep/held-out/CI
harness). It captures what was built, the scientific findings that came out of
the work, the decisions they imply, and how to reproduce everything.

The authoritative scope, acceptance gates, and experiment registry remain in
[`VALIDATION_ROADMAP.md`](VALIDATION_ROADMAP.md); this file is the narrative and
decision record behind the current status there.

## 1. What changed

### S3: the EMPCA bridge is now verified independently, not by construction

Previously `run_s3_ae_bridge` compared EMPCA against `tied_linear_ae_closed_form`,
which delegated to the same `fit_weighted_pca` solver — so the agreement was true
by construction and proved nothing (the old record even showed
`reconstruction_rmse: 0.0`).

S3 now trains the tied weighted linear autoencoder **independently** by gradient
optimisation (`src/noise_geometry/autoencoders/trained.py`):

- a direct weighted-loss AE (`x̂ = D Dᵀ M x`, loss `rᵀ M r`) with an analytic
  gradient, trained by full-batch L-BFGS in float64, plus a reported Adam pass;
- a whitened-MSE cross-check AE (Baldi–Hornik) for stability;
- diagnostics per run: optimality gap vs the EMPCA global optimum `L*`, principal
  angle to EMPCA in the noise metric, and M-orthonormality `‖DᵀMD − I‖`.

Trained models can be persisted with `scripts/train_s3_ae.py` (to
`results/models/*.npz`); the experiment itself records metrics only.

### Canonical EMPCA and independent reference oracles

Verified production code now lives under `src/canonical/`. The EMPCA bridge is
checked against two independent oracles:

- **TCY no-smoothing EMPCA** (`fit_empca_no_smoothing`, exact `full` M-step) — the
  paper-grade entry point that skips the Savitzky–Golay smoothing of the
  convenience `EMPCA.fit()`.
- **Stephen Bailey's published reference WPCA** (`canonical/empca.py`).

`tests/test_canonical_empca_equivalence.py` shows these agree (and pins the
rFFT→real inner-product preservation that makes the comparison valid).

### Item 2: one central rFFT↔real representation primitive

`src/canonical/empca_equivalence_utils.py` now defines the single representation
primitive used everywhere:

- `rfft_to_real` / `real_to_rfft` — faithful real representation of a real
  signal's rFFT, `[Re(X₀), Re(X₁..ₘ), Im(X₁..ₘ)]`, with exact inverse;
- `real_weight_vector` — expands one-sided weights to that layout;
- `complex_to_real_whitened` — the whitened features whose Euclidean geometry is
  exactly the complex `Σ⁻¹` geometry.

Routed through it so the DC/Nyquist/weight conventions live in one place:

- **S5** — fixed the `.real`-only bug (it discarded the imaginary part and broke
  the `Σ⁻¹` geometry); now stacks Re and Im faithfully.
- **S1** — the OF/GLS amplitude is now the rank-1 projection in whitened space
  (`⟨whitened x, whitened s⟩ / ‖whitened s‖²`), identical to `gls_amplitude` and
  `OptimumFilter.fit` to machine precision.
- **Weights** — `inverse_psd_weights` now delegates to
  `canonical.build_of_one_sided_weights`, so the OF convention has one source of
  truth (pinned by a test).

### Item 3: seed-sweep + held-out + bootstrap-CI harness

`src/noise_geometry/validation/harness.py` provides the uncertainty machinery as
infrastructure, so hardening an experiment is a config change, not new code:

- `run_seed_sweep` — one flattened row per seed (the archivable source data);
- `summarize` — per-metric mean/std and 68%/95% bootstrap CIs of the mean;
- `paired_difference` — per-seed paired difference with CI, sign consistency, and
  `ci95_excludes_zero` (the roadmap's paired-interval acceptance form);
- `train_test_split_indices` — leakage-free held-out splits.

`scripts/sweep.py` runs one gate; `scripts/sweep_all.py` rolls the harness across
**all of S1-S9**, archiving per-gate CSV + CI JSON to `results/sweeps/`.
Held-out evaluation (`test_frac`) is wired into the subspace/rank-fit gates
(S4, S5, S6); S1/S7/S8/S9 already draw independent noise per seed, so the seed
sweep itself supplies honest uncertainty. Each gate's acceptance is now a
multi-seed regression test (`tests/test_synthetic_gate_acceptance.py`):

| gate | headline (10 seeds) | acceptance |
|---|---|---|
| S1 | σ/CRB mean 0.994, 95% CI [0.985, 1.002] | CI contains 1; bias CI contains 0 |
| S2 | EMPCA–OF angle 3.86° | √N convergence (slope ≈ −0.5) |
| S3 | relative gap ~2e-16 | trained AE ≈ EMPCA optimum |
| S4 | PCA–EMPCA angle ~1e-6° | agree in white noise; paired residual CI contains 0 |
| S5 | subspace angle 85° | held-out weighted-residual reversal, paired CI excludes 0 |
| S6 | best rank ≈ 4.6 | higher rank lowers held-out residual, paired CI excludes 0 |
| S7 | σ/oracle (largest N) 1.03 | decreases toward 1 with calibration size |
| S8 | χ²/dof 0.997, CI [0.995, 1.000] | compatible with 1 |
| S9 | diag/full σ 1.025 | full beats diagonal under correlation, paired CI excludes 0 |

### Import / reorganisation repairs

Moving production code into `src/canonical/` had broken `import src`, the S2
benchmark, and three test modules. These were repointed at `canonical/`, with a
dual `src.canonical` / `canonical` import where code must run both under pytest
(repo root on path) and under `scripts/run_experiment.py` (`src` on path).

## 2. Key findings and decisions

### EMPCA smoothing is not the maximum-likelihood objective

The convenience `EMPCA.fit()` applies a Savitzky–Golay filter to the
eigenvectors every iteration, which changes the fixed point. **Paper-grade work
must use `fit_empca_no_smoothing`.** Any smoothed result must be labelled as the
smoothed variant.

### Complex EMPCA carries a phase degree of freedom (rank-k complex = 2k real)

A complex EMPCA component has a complex coefficient, so a single complex
component spans a **two-dimensional** real subspace (`u` and `i·u`). The faithful
comparison between complex EMPCA and a real reference is therefore **complex
rank-k ↔ real rank-2k**, not k↔k. Verified both ways:

- matched (complex-coefficient) data: TCY rank-k and Bailey rank-2k agree to
  <0.1°;
- real-amplitude data: the real rank-k subspace is *contained in* the complex
  rank-k 2k-dimensional span.

### S2's OF/EMPCA angle is finite-sample, not structural

The rank-1 EMPCA vs OF-template angle shrinks at the 1/√N sampling rate (full and
fast M-steps coincide at rank 1):

| n_traces | angle | weighted cosine |
|---|---|---|
| 256 (current default) | 7.76° | 0.9909 |
| 1024 | 3.81° | 0.9978 |
| 4096 | 1.98° | 0.9994 |

So on real-amplitude data the rank-1 EMPCA does collapse onto the OF template;
the residual angle is sampling noise. The `cos > 0.9999` target is reachable with
more traces, and S2's acceptance is stated as convergence at the theoretical √N
rate (via the harness), not a single-seed threshold. Tests pin the log-log slope
of angle vs `n_traces` at ≈ −0.5 and the extrapolated large-N limit at ~0.

### The amplitude-model flag, and why it barely moves S2 (resolved)

`make_rank1_pulse_dataset` and `run_of_empca_equivalence` now take an
`amplitude_model` flag (`real` | `complex`). Testing it produced a sharper
finding than expected: **the rank-1 OF/EMPCA equivalence is robust to
coefficient phase.** A complex coefficient `c = a·e^{iθ}` gives a weighted
Hermitian covariance `E[|c|²]·s sᴴ`, which is rank-1 regardless of `θ` (only
`|c|²` enters), so complex EMPCA recovers the template either way and the angle
stays finite-sample. The phase degree of freedom only creates the 2k-real-dim
structure for a *real-coefficient* method (the trained AE, Bailey real PCA) — and
that is already covered by the TCY↔Bailey tests. A genuinely multi-dimensional
signal needs **time-shift jitter**, which is S6's domain, not S2's.

Consequence for the gate: S2's headline is the √N convergence; both metrics
(`weighted_angle_deg` and the template-into-EMPCA-span angle) are reported, and
the flag is kept for explicitness and for the real-space / jitter cases.

### The real-vs-complex amplitude decision gates S2's acceptance

Whether reconstruction coefficients are **real** (fixed-shape pulse × real
amplitude, no jitter; signal subspace = k real dims; rank-1 EMPCA ↔ OF exact in
the limit) or **complex** (amplitude + per-trace phase / sub-sample time-shift;
signal subspace = 2k real dims; the correct metric becomes "template ⊆ EMPCA
2-real-dim span") determines the right acceptance metric. The current S2 synthetic
data is real-amplitude, so `cos → 1` is correct there — but real detector data
(CRESST onset jitter, GWOSC complex templates) is closer to the complex case.
This should be made an explicit `amplitude_model` config flag before any S2
threshold is frozen, kept consistent with S6 (timing jitter).

### S5 metric reversal holds with held-out paired intervals

Over 20 seeds with 50% held-out evaluation, the reversal is statistically solid
(paired 95% CIs, sign-consistent across every seed):

| comparison | mean difference | 95% CI | verdict |
|---|---|---|---|
| PCA − WPCA raw MSE | −3032 | excludes 0 | PCA wins raw MSE |
| PCA − WPCA weighted residual | +0.545 | [0.525, 0.565] | WPCA wins (likelihood) |
| PCA − WPCA clean-signal MSE | +3023 | excludes 0 | WPCA recovers the signal |

This meets the roadmap's S5 acceptance form (PCA wins raw MSE while weighted PCA
wins the weighted residual/NLL with a paired 95% interval excluding zero).

## 3. How to reproduce

```bash
# All verification tests
uv run pytest -q

# S3 trained-AE bridge: train independently and save the models
uv run python scripts/train_s3_ae.py --all      # -> results/models/

# Multi-seed, held-out sweep with bootstrap CIs (S5 reversal)
uv run python scripts/sweep.py \
  --config configs/synthetic/s5_metric_reversal.yaml \
  --seeds 0-19 --set test_frac=0.5 \
  --pairs pca_weighted_residual_to_observed:weighted_pca_weighted_residual_to_observed
# -> results/sweeps/S5_sweep_20seeds.{csv,json}

# S2 finite-sample scaling (vary n_traces to see cos -> 1 at the sqrt(N) rate)
uv run python scripts/sweep.py --config configs/synthetic/s2_of_empca.yaml --seeds 0-19
```

## 4. Where things live

| Concern | Location |
|---|---|
| Independent trained tied linear AE | `src/noise_geometry/autoencoders/trained.py` |
| Save/load trained AE | `scripts/train_s3_ae.py`, `results/models/` |
| Canonical EMPCA (no-smoothing) + helpers | `src/canonical/empca_equivalence_utils.py` |
| Bailey reference WPCA | `src/canonical/empca.py` |
| Central rFFT↔real primitive | `src/canonical/empca_equivalence_utils.py` |
| OF / PSD / weights (canonical) | `src/canonical/{OptimumFilter,PSDCalculator,make_weights}.py` |
| Seed-sweep / held-out / CI harness | `src/noise_geometry/validation/harness.py` |
| Sweep CLI | `scripts/sweep.py` |
| Tests | `tests/test_{linear_ae_trained,canonical_empca_equivalence,representation_primitive,metric_reversal_representation,validation_harness}.py` |

## 5. Status against the long-run plan

| Item | Status |
|---|---|
| 1. Break S3 agreement-by-construction | Done — independent trained AE + EMPCA/Bailey oracles, tested |
| 2. Central representation primitive | Done for S1/S5 + weights; S2 amplitude-model decision resolved and tested |
| 3. Seed-sweep / held-out / CI harness | Done — rolled across S1-S9 with per-gate acceptance tests; held-out in S4/S5/S6 |
| 4. Stage 0 remote reproducibility gate | Turnkey clean-checkout runner, checklist, environment/dependency capture, and per-command logs added; server execution still owed |
| 5. Each gate a regression test | Followed for all work above; not yet a formal CI rule |
| 6. Real data (GWOSC → CRESST) | GWOSC/GWpy PSD and held-out whitening reference path started and fixture-tested; real H1/L1 cache run still owed |
| 7. Scope guard | Done — Paper 2's `resnet_1d.py` remains under `src/CNN/`, outside `canonical/` |

## 6. Remaining work and open decisions

- ~~Adopt the harness across S1/S4/S6–S9~~ — done: `scripts/sweep_all.py`
  sweeps every gate, held-out wired into S4/S5/S6, and each gate's acceptance is
  a multi-seed regression test. Remaining polish: raise default seed counts for
  archival runs and predeclare per-gate thresholds in the roadmap.
- ~~Resolve the `amplitude_model` flag for S2 and set S2's acceptance as √N
  convergence~~ — done: flag added, √N acceptance pinned by tests, phase
  robustness documented above. The flag still wants extending to S6 (where
  time-shift jitter genuinely raises the signal rank).
- **Unify on one EMPCA**: S3's trained-AE oracle is `fit_weighted_pca` (real),
  while S2 uses canonical complex EMPCA; keep the real/complex split deliberate
  (it is the phase-DOF distinction), but document which estimator each gate uses.
- **Stage 0 remote run**: execute `python3 scripts/stage0_remote_gate.py` on the
  clean Linux server and archive the emitted `results/stage0/...` directory.
- **GWOSC real-cache reference run**: download H1/L1 windows on the server, then
  run `scripts/preprocess/preprocess_gwosc.py --reference-check`; the code path
  and normalization tests are now in place.
- **Reproducibility smell**: EMPCA init uses the global `np.random.seed`; migrate
  to a local `np.random.default_rng`.
