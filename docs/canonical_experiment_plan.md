# Canonical Experiment Plan for Paper 1

## Central Claim

Structured detector noise defines the reconstruction geometry. Under Gaussian
noise, the maximum-likelihood reconstruction objective is the Mahalanobis
residual `(x - s_hat)^dagger Sigma^{-1} (x - s_hat)`, not ordinary MSE unless
the noise is white.

The estimator hierarchy is:

```text
fixed rank-1 OF -> learned rank-k EMPCA -> tied linear AE with weighted loss
```

## Experiment Layers

Layer 1 is the controlled synthetic benchmark. It verifies the theorems under
known ground truth and exposes failure modes under colored or correlated noise.

Layer 2 is GWOSC. It validates the PSD, whitening, and matched-filter side of
the framework on public real detector noise.

Layer 3 is CRESST-II / CRESST-III pulse-shape data. It connects the framework
to cryogenic detector pulse reconstruction.

Layer 4 is optional TIDMAD. It should be used only if the core synthetic, GWOSC,
and CRESST experiments are already stable.

## Controlled Synthetic Experiments

SYN-A: OF equals rank-1 EMPCA under matched conditions.

- Generate `x_i = a_i s0 + n_i` with known `Sigma` or PSD.
- Compare OF / GLS amplitude, rank-1 weighted PCA/EMPCA, and rank-1 weighted
  tied linear AE.
- Metrics: amplitude difference, reconstruction difference, weighted residual,
  whitened subspace cosine, finite-sample convergence.

SYN-B: Bridge Theorem, noise-aware linear AE equals EMPCA.

- Generate rank-k signal families `x_i = U c_i + n_i`.
- Compare EMPCA, MSE tied linear AE, weighted tied linear AE, and optional
  untied AE negative control.
- Metrics: principal angles, weighted residual, raw MSE, true-subspace recovery.

SYN-C: MSE-vs-Mahalanobis metric reversal.

- Use colored noise where variance-dominant directions differ from
  signal-relevant directions.
- Compare PCA/MSE AE to EMPCA/weighted AE.
- Metrics: raw MSE, weighted residual, whitened subspace angle, amplitude bias,
  chi-square.

SYN-D: Template mismatch and timing jitter.

- Generate shifted and deformed pulses.
- Compare fixed-template OF, time-shift OF, rank-k EMPCA, and weighted tied AE.
- Metrics: amplitude bias, timing error, weighted residual, reconstruction error
  versus jitter strength.

## GWOSC Plan

GWOSC is the primary public dataset. The first implementation should use a small
event-centered example such as GW150914 and a separate injection study on
off-source windows.

Required pipeline:

- Fetch short H1/L1 strain windows.
- Estimate PSD from off-source data.
- Whiten strain and template.
- Run matched-filter / OF-style projection.
- Inject known waveforms into off-source real noise windows.
- Compare raw MSE, PSD-weighted residual, matched-filter SNR, and recovery
  curves versus injected SNR.

Expected figures:

- Strain before and after whitening.
- PSD estimate and whitening filter.
- Matched-filter score around event time.
- Raw MSE versus PSD-weighted residual.
- Injection recovery curve.

## CRESST Plan

CRESST is the strongest complementary public dataset because it contains
cryogenic detector pulse-shape traces.

Required pipeline:

- Download released pulse-shape traces manually if needed.
- Inspect trace arrays, labels, and metadata.
- Estimate noise covariance or PSD from baseline/noise-like traces.
- Build pulse templates or learned pulse subspaces.
- Compare OF, PCA, EMPCA, MSE tied linear AE, and weighted tied linear AE.

Expected figures:

- Example pulse traces.
- Estimated PSD or covariance structure.
- PCA versus EMPCA basis vectors.
- Raw MSE versus weighted residual.
- Subspace angle between MSE and noise-aware models.

## Optional TIDMAD Plan

TIDMAD is optional and must not block the core paper. Use it only after GWOSC and
CRESST are stable. A minimal validation should compare a repository baseline,
simple filtering, and a PSD/frequency-weighted reconstruction loss on a
manageable subset.

## Minimum Viable Paper Version

- Controlled synthetic benchmark with SYN-A through SYN-C.
- GWOSC matched-filter or injection smoke result.
- CRESST pulse-shape reconstruction plan plus first runnable preprocessing or
  reconstruction result.

## Stronger Version

- SYN-A through SYN-D complete.
- GWOSC event and injection figures.
- CRESST PCA/EMPCA/AE comparison figures.
- Optional TIDMAD validation if it produces a clean result quickly.

## Success Criteria Before Final Preprint

- All synthetic results are reproducible from scripts and configs.
- OF/rank-1 equivalence is numerically verified.
- Bridge theorem is numerically verified with principal-angle metrics.
- Metric reversal is visible under colored noise.
- GWOSC and CRESST data access are documented enough for an external user.
- No large public data or generated artifacts are tracked in git.
