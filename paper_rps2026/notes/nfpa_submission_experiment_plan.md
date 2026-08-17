# NFPA submission-strength experiment plan

Status: designed but not executed. No result in this document may be copied into the paper until the corresponding run, validation, and audit are complete.

## 1. Purpose and claim boundaries

The experiments should test four claims separately. First, covariance-aware and Euclidean training metrics can reverse model rankings under colored or correlated noise. Second, NFPA implements a covariance-aware multilinear restriction rather than an unrestricted covariance-weighted PCA. Third, when both channel and temporal dimensions are genuinely compressed, the structured representation can recover scientifically meaningful latent quantities. Fourth, the conclusions should remain calibrated when covariance is estimated or misspecified rather than known exactly.

The current archived proof-of-concept is not confirmatory because it uses one seed and has `C = k_c = 2`, which makes the channel projector equal to the identity. It may remain as an implementation check, but it must not be the main NFPA result.

## 2. Predeclared hypotheses

### H1: metric reversal at fixed architecture

Under colored or correlated noise, NFPA and an isotropically trained multilinear projection with the same `(k_c, k_t)` should disagree in ranking: NFPA should have lower held-out normalized likelihood residual, while the isotropic model should have lower raw MSE. The paired across-seed confidence interval for each difference should exclude zero in the predicted direction.

### H2: white-noise negative control

When `Sigma = sigma^2 I`, covariance-aware and Euclidean objectives should yield equivalent projectors up to numerical tolerance and latent rotations. Ranking reversal should disappear.

### H3: active channel-time restriction

With `C > k_c` and `T > k_t`, NFPA should recover the planted channel and temporal subspaces when the signal is separable and sufficiently identified. As controlled nonseparability increases, its gap to unrestricted CW-PCA should increase smoothly rather than being hidden by a degenerate full-channel factor.

### H4: scientifically useful representation

NFPA codes should recover event amplitude and at least one pulse-shape parameter more accurately than the isotropic factored baseline under colored noise. Recovery should be evaluated against the known latent variables, not inferred solely from reconstruction loss.

### H5: covariance robustness boundary

Oracle covariance should provide the strongest likelihood alignment. Finite calibration data and moderate regularization should degrade smoothly. Deliberately wrong covariance should expose a measurable failure boundary rather than being treated as harmless.

## 3. Primary confirmatory configuration

Use a nondegenerate multichannel problem:

| Quantity | Primary value |
|---|---:|
| Channels `C` | 8 |
| Samples per trace `T` | 256 |
| Channel rank `k_c` | 2 |
| Temporal rank `k_t` | 3 |
| Total matched rank `k` | 6 |
| Training events | 800 |
| Validation events | 400 |
| Held-out test events | 2,000 |
| Independent seeds | 30 |
| ALS restarts per seed | 10 |
| Primary SNR | 10, using the definition below |
| Primary noise | Brownian-like temporal PSD, `1/f^2`, with channel correlation |
| Covariance | Oracle for the primary test; estimated variants are separate |

Use the same generated train, validation, test, and calibration sets for every method within a seed. Method comparisons must therefore be paired by seed.

Define SNR before running as

```text
SNR^2 = E[s^T Sigma^{-1} s]
```

for the complete vectorized clean event. Rescale signals, not noise after sampling, to reach each target SNR. Record the realized mean and standard deviation of SNR in every output file.

## 4. Signal generator

Generate clean events as a low-dimensional scientific pulse family,

```text
S_i = sum_{a=1}^{r_c} sum_{b=1}^{r_t} z_{i,a,b} v_a t_b^T + alpha R_i.
```

The channel vectors `v_a` and temporal vectors `t_b` should be orthonormal in their declared coordinates. Use two physically interpretable channel-loading patterns and three asymmetric pulse-shape components, such as a nominal pulse, a rise-time perturbation, and a decay-time perturbation. The latent coefficients should include:

- an event amplitude `A_i`;
- a rise-time or time-shift parameter;
- a decay-time or width parameter;
- optional event class for a secondary clustering analysis.

Construct `R_i` outside the planted product subspace. Project candidate perturbations away from `span(V) tensor span(T)` and renormalize total clean-signal power at every `alpha`. This prevents the nonseparability sweep from accidentally becoming an SNR sweep.

Use `alpha in {0, 0.1, 0.25, 0.5, 1.0}` for the nonseparability study. The primary confirmatory configuration should use `alpha = 0.1`: mostly factorizable but not exactly rank-one or perfectly separable.

## 5. Noise and covariance construction

The primary covariance should preserve channel and time axes so that NFPA factor interpretation is meaningful. Use

```text
Sigma = Sigma_t tensor Sigma_c,
```

or a block-diagonal per-channel temporal covariance with explicitly reported channel coupling. `Sigma_t` should implement the declared PSD, and `Sigma_c` should have moderate off-diagonal correlation, for example an AR(1)-style correlation with `rho = 0.4`.

Every covariance matrix must pass numerical checks before data generation:

1. symmetry error below a fixed tolerance;
2. strictly positive eigenvalues after the predeclared floor;
3. whitening identity error `||W Sigma W^T - I||_F / d` below tolerance;
4. Monte Carlo covariance agreement on an independent noise draw;
5. no train, validation, or test signal events used in covariance fitting.

## 6. Methods and matched comparisons

Fit the following methods at matched total rank `k = k_c k_t`:

| Method | Metric | Representation class | Purpose |
|---|---|---|---|
| CW-PCA | `Sigma^{-1}` | unrestricted rank `k` | likelihood-aligned upper-flexibility reference |
| NFPA | `Sigma^{-1}` | channel-time factored | proposed structured representation |
| Iso-MPCA / IsoAE | identity | same factored `(k_c, k_t)` | isolates the metric at fixed architecture |
| IsoPCA | identity | unrestricted rank `k` | conventional Euclidean reference |
| OF | `Sigma^{-1}` | fixed rank-one template | amplitude reference for rank-one checks |

Before reporting both Iso-MPCA and IsoAE, verify whether their fitted projectors are mathematically and numerically identical under the tied linear architecture. If they are equivalent, report one method name and mention the alternative parameterization rather than presenting duplicate baselines.

CW-PCA means ordinary PCA after whitening by one shared full covariance. Use the term EMPCA only for a genuinely iterative implementation needed by sample-specific weights, missing entries, or heterogeneous covariance.

## 7. Leakage-free fitting protocol

For every seed:

1. Generate independent calibration, training, validation, and test partitions.
2. Estimate the training mean from training events only. Subtract that same mean from validation and test events.
3. In oracle runs, use the generating covariance. In estimated-covariance runs, fit covariance using calibration noise only.
4. Select shrinkage, PSD floor, rank pair, convergence tolerance, and ALS restart using the validation split or a fixed predeclared rule. Never use test metrics.
5. Fit every method on exactly the same centered training observations.
6. For NFPA and Iso-MPCA, retain the best training objective across the predeclared ten restarts. Record every restart objective to diagnose local minima.
7. Evaluate the frozen models once on the held-out test set.
8. Add the training mean back only when reporting measurement-coordinate reconstructions; subspace metrics remain in centered coordinates.

Store software version, git commit, configuration hash, random seed, array shapes, and data checksums with every run.

## 8. Required experiment blocks

### Block A: primary paired metric-reversal test

Run the primary configuration for 30 seeds. This is the main paper result. Report training and held-out raw MSE, normalized likelihood residual, principal angles, downstream parameter errors, and train-test gaps.

Primary contrasts:

```text
NFPA - Iso-MPCA under normalized chi-square
NFPA - Iso-MPCA under raw MSE
NFPA - CW-PCA under normalized chi-square
```

The first two contrasts test metric reversal at fixed architecture. The third measures the held-out cost or possible finite-sample benefit of the restriction; it must not be described as superiority unless the paired interval supports that claim.

### Block B: white-noise control

Repeat 20 seeds with the same clean signals and isotropic noise. Confirm that covariance-aware and Euclidean projectors agree to numerical tolerance and that the metric-reversal intervals include zero or are negligibly small by a predeclared equivalence margin.

### Block C: SNR sweep

Use `SNR in {2, 5, 10, 30, 100}` with 20 seeds per value. Do not include SNR 914 as a primary condition; retain it only as an implementation sanity check. Plot amplitude and shape-parameter error against SNR and test where the NFPA/OF correspondence becomes practically useful rather than trivial.

### Block D: noise-color sweep

Use white, pink (`1/f`), Brownian (`1/f^2`), and one measured or detector-motivated PSD. Run 20 paired seeds per condition. The expected signature is no meaningful reversal for white noise and increasing disagreement as anisotropy grows.

### Block E: nonseparability sweep

Use `alpha in {0, 0.1, 0.25, 0.5, 1.0}` with 20 seeds per value. Report the NFPA-to-CW-PCA projector distance and held-out likelihood gap. Hold total signal power, SNR, rank, sample size, and covariance fixed.

### Block F: covariance estimation and misspecification

Compare:

1. oracle covariance;
2. covariance estimated from `N_cal in {64, 256, 1024, 4096}` independent noise-only traces;
3. the same estimates with fixed shrinkage values from a predeclared grid;
4. identity covariance;
5. wrong spectral slope;
6. permuted frequency bins;
7. covariance from a mismatched operating condition.

Use 20 paired seeds per condition. Report covariance estimation error, whitening calibration, downstream errors, and reconstruction metrics. Tune shrinkage on calibration/validation data only.

### Block G: rank and sample-size sensitivity

Use matched total ranks with several factorizations, for example `(1, 6)`, `(2, 3)`, `(3, 2)`, and `(6, 1)` where dimensions allow. Separately sweep `N_train in {100, 200, 400, 800, 1600, 3200}`. This tests the expected bias-variance tradeoff rather than merely comparing parameter counts.

### Block H: semantic utility

Primary downstream tasks should be amplitude and pulse-shape recovery:

- amplitude RMSE, bias, and correlation;
- rise-time or time-shift RMSE;
- decay-time or width RMSE;
- channel-subspace projector error against the planted `span(V)`;
- temporal-subspace projector error against the planted `span(T)`.

Calibrate physical parameters from latent codes using a linear map fitted on the training split only. Evaluate on test events. Include a direct estimator or OF baseline where defined. Clustering or anomaly detection may be added as secondary analyses, but they should not replace amplitude and shape recovery.

## 9. Metric definitions

Use explicit, consistent definitions:

```text
raw_mse = (1 / (N d)) sum_i ||x_i - xhat_i||_2^2
normalized_chi2 = (1 / (N d)) sum_i (x_i - xhat_i)^T Sigma_eval^{-1} (x_i - xhat_i)
```

For oracle synthetic evaluation, `Sigma_eval` is the generating covariance even when a method trained with an estimated or wrong covariance. Also report the method's own training objective separately; do not use a misspecified covariance as the sole judge of its own correctness.

Subspace comparisons should be computed in the common oracle-whitened coordinate system. Report the full principal-angle vector, mean angle, maximum angle, and normalized projector distance. For planted factors, compare projectors rather than individual basis vectors because rotations and signs are not identifiable.

## 10. Statistical analysis

Use paired inference because every method sees the same data within a seed. For each primary contrast, report:

- mean and median paired difference;
- 95% confidence interval from a seed-level bootstrap with a fixed number of resamples;
- standardized paired effect size;
- all individual seed values in the archived result table.

The seed, not the event, is the resampling unit for method comparisons. Event-level bootstrap intervals may be added for downstream estimators within a seed, but they cannot replace across-seed uncertainty.

Predeclare Block A contrasts as primary. Treat sweep-wise comparisons as secondary and control multiplicity, for example with Holm correction, or emphasize confidence intervals and effect curves rather than isolated p-values.

## 11. Acceptance gates

Do not promote the new experiment into the main paper unless all of the following hold:

1. All 30 primary seeds complete with no post-result configuration changes.
2. `C > k_c` and both planted factor projectors are genuinely nontrivial.
3. The fixed-architecture metric-reversal contrasts have intervals excluding zero in the predicted directions.
4. The white-noise negative control passes its numerical/equivalence criterion.
5. NFPA optimization diagnostics show stable convergence or clearly report failure rates.
6. Downstream amplitude and pulse-shape metrics are computed from held-out truth.
7. Covariance fitting and centering pass leakage audits.
8. The figure and table are generated directly from archived seed-level results.

If the NFPA-to-CW-PCA held-out difference includes zero, state that the restricted and unrestricted models are indistinguishable at the available precision. If CW-PCA wins, report the restriction cost. If NFPA wins, describe it as a held-out finite-sample result and support it with training/test gaps before proposing regularization as the explanation.

## 12. Planned main-paper figure

Replace the removed likelihood-contour illustration only after Block A is complete.

Panel A should show ranking reversal at fixed architecture. Plot the paired relative difference between NFPA and Iso-MPCA for raw MSE and normalized likelihood residual, with 95% confidence intervals and a vertical zero line. Label the preferred direction directly; do not place incomparable raw units on one axis.

Panel B should show seed-level subspace alignment to CW-PCA for NFPA and Iso-MPCA. Use a compact violin-plus-box or interval-and-points display of mean or maximum principal angle. State `n = 30 seeds` in the caption.

Use a colorblind-safe palette, vector output, readable axis labels at final column width, and no curved manifold language. The caption must say that all methods have matched total rank and that NFPA/Iso-MPCA share the same factored architecture.

## 13. Output contract

Each run should produce one tidy row per `(seed, method, condition, split)` with at least:

```text
seed
config_hash
method
split
C, T, k_c, k_t, total_rank
N_train, N_validation, N_test, N_calibration
target_snr, realized_snr
noise_model, covariance_regime, shrinkage
train_raw_mse, test_raw_mse
train_normalized_chi2, test_normalized_chi2
principal_angles_deg
projector_distance
amplitude_bias, amplitude_rmse, amplitude_correlation
rise_time_rmse, decay_time_rmse
channel_projector_error, temporal_projector_error
optimizer_restarts, selected_restart, convergence_status
runtime_seconds
```

Write raw seed-level results before aggregates. Generate summary tables and figures from those raw files with a deterministic script. Never transcribe numbers manually into LaTeX.

## 14. Manuscript update after successful execution

After the acceptance gates pass:

1. Replace the single-split table with the 30-seed summary and paired intervals.
2. Replace the current proof-of-concept paragraph with the nondegenerate `C > k_c` configuration.
3. Insert the planned two-panel empirical figure.
4. Move the current `C = k_c = 2` result to the appendix as an implementation check or remove it.
5. Add one sentence interpreting amplitude and pulse-shape recovery as the representation's actionable scientific content.
6. Update the abstract only with effects supported by the final archived analysis.

## 15. Deferred implementation checklist

- [ ] Freeze the generator, metrics, seed list, and acceptance gates before looking at aggregate results.
- [ ] Implement nondegenerate `C > k_c` data generation.
- [ ] Add oracle and estimated covariance paths.
- [ ] Add CW-PCA, NFPA, Iso-MPCA/IsoAE, IsoPCA, and OF under one interface.
- [ ] Add training-only centering and leakage tests.
- [ ] Add ALS restart and convergence diagnostics.
- [ ] Add semantic parameter decoders trained on training data only.
- [ ] Run small smoke tests without using them to tune confirmatory thresholds.
- [ ] Freeze the primary configuration and seed list.
- [ ] Execute Blocks A and B first.
- [ ] Audit raw outputs before generating summaries.
- [ ] Execute secondary sweeps.
- [ ] Generate the final table and two-panel figure from archived data.
- [ ] Revise the paper's results and abstract only after the audit.
