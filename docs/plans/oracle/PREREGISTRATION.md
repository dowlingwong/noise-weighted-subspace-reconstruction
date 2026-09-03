# ORACLE-Cov pre-registration (Tier 1) — DRAFT, not yet frozen

_Status: draft written 2026-09-02 alongside the implementation.  It becomes the
frozen `core` part when committed with the line `FROZEN: <commit>` below; until
then every Tier-1 result is `mode: dev` and not citable.  Confirmatory runners
(`experiments/oracle/run_tier1.py --mode confirmatory`) refuse to start without
this file and record its hash in every manifest._

FROZEN: (not yet)

## 1. Testbed and subjects

- Generator: `experiments.synthetic.nfpa_confirmatory` frozen configuration (`PRIMARY`,
  hash in every manifest) at `nonseparability_alpha = 0` — the clean training support.
  C = 8 channels × T = 256 samples; Σ = Σ_t ⊗ Σ_c with 1/f² temporal PSD and AR(1)
  channel correlation ρ = 0.4; SNR 10.
- Excited latents in training: amplitude, rise_time, decay_time, secondary.
  Unexcited: t0 (fixed at 0.38 T), nonseparable deformation (α = 0).
- Subjects (trained once per seed under Σ̂ = Σ, then frozen): `lin_cw` (rank-6 CW-PCA /
  tied linear AE; stage `code`); `mlp_ae` (tanh autoencoder 2048→64→32→6→32→64→2048 on
  the assumed-whitened input, Adam, 60 epochs; stages `h1`, `h2`, `code`).
- Physics output: linear amplitude readout on the code, fitted on clean training latents.

## 2. Families and severity grids

Exactly the grids in `experiments/oracle/oracle_cov/families.py::SEVERITIES`:
N1 PSD-exponent change δ ∈ {±0.25, ±0.5, ±0.75}; N2 channel ρ change ∈ {±0.2, ±0.4};
N3 t0 jitter σ ∈ {0.5, 1, 2, 4} samples; N4 gain drift ∈ {0.02, 0.05, 0.1, 0.2};
N5 glitch rate ∈ {0.25, 0.5, 1, 2} per event; N6 channels lost ∈ {1, 2, 3};
S1 amplitude above quantile ∈ {0.7, 0.85, 0.95}; S2 decay above quantile ∈ {0.7, 0.85, 0.95};
S3 t0 shift ∈ {0.5, 1, 2, 4} samples; S4 α ∈ {0.1, 0.2, 0.4};
U1 Student-t df ∈ {3, 5}; U2 spectral line (relative amplitude) ∈ {0.25, 0.5}.
Declared: N1–N6, S1–S4.  Undeclared (abstention targets, never in training or calibration): U1, U2.
Designed: `null`/`aligned` (reconstruction Jacobian) and `null_amp`/`aligned_amp`
(amplitude gradient) at `mlp_ae` stage `h1`, norms {0.5, 1, 2, 4}.

## 3. Windows, seeds, resampling

Window = 100 events.  Per seed: 60 clean windows (20 to fit monitors, 20 to set the
1 % false-alert threshold, 20 held out to report the realised FAR); 3 windows per
(family, severity).  Dev seeds {0, 1, 2}; confirmatory seeds {100 … 119}, disjoint.
Intervals resample perturbation seeds and, within them, windows (event groups);
never events.

## 4. Endpoints and thresholds

- **C1 detection.** Primary: power at the calibrated 1 % FAR per (subject, stage,
  monitor, family), averaged over severities; secondary: realised held-out FAR must be
  ≤ 0.03 or the monitor's C1 row is void.
- **C2 attribution.** Primary: leave-one-seed-out macro-F1 over {N, S} of an L2 logistic
  classifier on (a) the full layerwise signature (monitors + tangent statistics) versus
  (b) the generic-only arm (input KS, MMD, C2ST, embedding moments).  ΔF1 read three ways:
  meaningful if the interval excludes 0 and the point estimate ≥ 0.10; equivalent if the
  interval lies within ±0.05; otherwise inconclusive.  Abstention: split-conformal at
  α = 0.1 with a *feature-space Mahalanobis* nonconformity (classifier margin is reported
  but is not the abstention rule: it does not detect U1/U2); unknown-family AUROC (U vs
  declared) and risk–coverage AUC, risk at 80 % coverage.  Three arms: layerwise full,
  layerwise deployable (no tangent statistics, no covariance monitor — the Tier-2 arm), generic.
- **C4 consequence.** Two consequence variables: `K_recon`, the relative increase of the
  whitened reconstruction error against the clean signal (the Tier-1 primary — the amplitude
  readout at SNR 10 is noise-dominated and insensitive), and `K`, the relative amplitude-RMSE
  degradation (physics-facing secondary); high consequence K ≥ κ_m = 0.10.  Primary: pooled
  AUROC of the alarm for high consequence across all declared cells, and the designed
  dissociation; secondary: missed-consequence and harmless-alarm rates at the C1 threshold.
  Within-(family, severity) strata carry no consequence variation on Tier 1 and are not an
  endpoint.  Designed dissociation: predicted sign
  `aligned ≫ null` for the pullback-whitened and Jacobian-projected monitors and
  `aligned_amp ≫ null_amp` for the pullback-output monitor; the Euclidean monitor is
  predicted unable to separate either pair.
- **Multiplicity.** One primary endpoint per claim; all secondary endpoints BH-FDR at 0.05
  within each claim.

## 5. Mechanism predictions (frozen before Tier 2)

Statistics from `pipeline.tangent_statistics` on the whitened observation displacement
between a window and the clean reference:

| family class | mean-displacement excess in T_S (z) | in T_U (z) | residual variance along T_S |
|---|---|---|---|
| N covariance (N1, N2, N4) | ≈ 0 | ≈ 0 | ≠ 1 (departs with the spectrum of Σ̂⁻¹Σ') |
| N signal-deformation (N3 jitter) | small | > 0 (unexcited direction, zero-mean per event) | ≈ 1 |
| S along excited coordinates (S1, S2) | > 0 | ≈ 0 | ≈ 1 |
| S into unexcited directions (S3, S4) | small | > 0 for S3; S4 lands outside both spans | ≈ 1 |
| N signal removal (N6 channel loss) | > 0 (the lost channels' mean signal) | ≈ 0 | < 1 |

The N-versus-S contract therefore maps onto **three** geometric signatures, not two;
attribution must use the pair (mean-displacement location, residual-variance ratio), and
N3 is a declared ambiguous case whose resolution is the variance ratio.  This refinement
of proposal §3 is itself a pre-registered claim.  Layerwise: N1 rotates the `mlp_ae` `h1`
dominant subspace more than S3 does at matched C1 power.

## 6. Tier-1 → Tier-2 bridge (to be filled at M3)

For each ORACLE-Paired family (N1–N5 acquisition, S1–S5 rare physics), the row states which
Tier-1 signature class it belongs to and which monitors are predicted to fail
(Euclidean on all S; every mean-displacement monitor on N covariance families).  Scored as a
single pre-registered summary: fraction of rows confirmed with a binomial interval.
