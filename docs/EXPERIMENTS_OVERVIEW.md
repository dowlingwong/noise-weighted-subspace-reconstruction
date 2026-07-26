# Experiments Overview: Description, Method, Target

Date: 2026-07-26. Independent audit pass (second reviewer). Code references are to
commit `main` at revision of this date. This file describes every experiment
designed and implemented in this repository; status labels reflect the archived
evidence, not intentions.

Conventions used below: Σ is the noise covariance, R denotes the
likelihood-aligned weighted residual `(x − ŝ)† Σ⁻¹ (x − ŝ)`, OF = optimal
filter, EMPCA = weighted/expectation–maximization PCA, AE = autoencoder.

---

## 1. Controlled synthetic suite (S0–S9)

Implementation: `src/noise_geometry/experiments/synthetic.py`, configs under
`configs/synthetic/`, sweeps via `scripts/sweep_all.py`, confirmatory S5/S4 via
`scripts/run_s5_confirmatory_sweep.py`.

| ID | Description | Method | Target (what it proves if it passes) | Status |
| --- | --- | --- | --- | --- |
| S0 | End-to-end smoke run | Rank-1 pulse dataset → OF/EMPCA equivalence path → JSON record | Pipeline integration only; no scientific claim | Pass |
| S1 | OF variance vs Cramér–Rao bound | 2048 pulses in pink noise; OF amplitude via the shared whitened-real primitive; compare empirical σ to `(s†Σ⁻¹s)^(−1/2)` | OF is implemented as the true GLS estimator: unbiased, CRB-efficient, correct normalization | Pass (σ/CRB ≈ 0.994, 10 seeds) |
| S2 | Rank-1 EMPCA recovers the OF direction | Fit rank-1 weighted PCA to noisy copies of one template; principal angle to whitened template | Learned rank-1 subspace = OF template direction (estimator equivalence, Thm level) | Pass (~3.9° finite-sample angle, 10 seeds) |
| S3 | Tied linear AE ↔ EMPCA bridge | Independently train tied weighted linear AE (L-BFGS and Adam; direct and whitened parameterizations); compare loss to closed-form optimum L*, principal angle, M-orthonormality | The linear AE with Σ⁻¹-adjoint encoder converges to the EMPCA optimum (Theorem 5.1 numerically) | Pass (relative gap ~2×10⁻¹⁶) |
| S4 | White-noise control | Same rank PCA vs EMPCA on isotropic noise, held-out (v2/sweeps) | Negative control: geometries coincide when Σ ∝ I, so later differences are attributable to the metric | Pass (angle ~10⁻⁶ deg) |
| S5 | Colored-noise metric reversal | Rank-2 signal + 1/f² noise; rFFT Re/Im-stacked representation; matched-rank PCA vs weighted PCA; held-out split; compare raw MSE vs weighted residual vs NLL | The Euclidean and likelihood orderings disagree under colored noise: EMPCA wins R and NLL, PCA wins raw MSE, subspaces nearly orthogonal (~86°) | Pass; 30-seed confirmatory (seeds 1000–1029, test_frac 0.5): weighted diff 0.550, CI [0.530, 0.571]; ratio 1.0115 |
| S6 | Timing jitter and rank | Pulses with Gaussian timing jitter in AR(1) noise; sweep rank 1–6 of weighted PCA vs single-template OF; held-out | Timing variation expands the signal manifold; best rank > 1 (mean ≈ 4.6), matching the shift-manifold Taylor argument | Pass |
| S7 | Covariance-estimation robustness | Estimate Σ from 10–1000 calibration traces with shrinkage 0.05; compare GLS amplitude σ to oracle-Σ σ | Quantifies calibration-sample requirements; σ/oracle → ~1.03 at 1000 traces | Pass |
| S8 | Residual χ² calibration | GLS fit under correct AR(1) model; whiten residuals; χ²/dof and whitened lag-1 autocorrelation | Distributional (not just second-moment) calibration; requires the Gaussian-bin noise generator to be meaningful | Pass (χ²/dof ≈ 0.997) |
| S9 | Multichannel covariance | 3 channels, Kronecker block covariance with cross-channel ρ = 0.4; full vs channel-diagonal Σ in joint GLS | Full covariance strictly improves amplitude uncertainty when channels correlate (factor ≈ 1.025) | Pass |

Design notes from this audit:

- The v1 standalone YAMLs for S4/S5/S6 default to in-sample evaluation
  (`test_frac = 0`); the frozen archival JSONs under `results/metrics/` for
  these are in-sample. The v2 configs and all sweep/confirmatory runs bake in
  `test_frac = 0.5`. Held-out claims in the paper must cite sweeps/v2/30-seed
  artifacts, never the v1 seed-level JSONs.
- In S5 the noise is scaled by 0.25 while the weights come from the unscaled
  PSD. This is harmless for every claim made (subspace fitting and residual
  ordering are invariant to a global Σ scale) but means S5 weighted residuals
  are not χ²-calibrated per dof; S8 owns that claim.

## 2. Public CRESST within-channel validation

Implementation: `src/noise_geometry/cresst/` (loader, analysis, templates,
negative controls, detector metrics, sensitivity), configs
`configs/cresst/*.yaml`, post-freeze sensitivity in
`scripts/run_cresst_postfreeze_sensitivity.py`.

| ID | Description | Method | Target | Status |
| --- | --- | --- | --- | --- |
| CRESST-A | Within-channel linear reconstruction hierarchy on 7 public CRESST-II/III channels | Per (run, channel): pulse and noise streams each split chronologically 70/30, separately; PSD from calibration noise; rank-6 EMPCA (shrinkage 0.1) vs single-template OF, 25-cell multi-template OF bank (rise/decay quantiles × shifts −4…+4), ordinary PCA, tied linear AE; paired event-level differences; 300-resample event bootstrap over 19,335 held-out pulses | Real-data test that a learned rank-6 subspace beats a fixed template (capacity), and that Σ⁻¹ weighting helps at matched rank (geometry) | Positive: OF−EMPCA pooled 1012.27, CI [927.90, 1097.61]; multi-OF margin 352.17, CI [272.40, 424.01]; PCA−EMPCA positive in all 7 channels (1.6%–62.0%) |
| CRESST negative controls | Five predeclared falsification tests | Wrong-PSD weighting, noise-only training, random weighted basis, mismatched OF template, permuted time bins; same paired pipeline | Pipeline falsifiability: each deliberate corruption must degrade R | All degraded at pooled level; one channel-level exception (wrong-PSD on 35/58.0, ≈ 0, CI touching 0) — correctly reported |
| CRESST post-freeze sensitivity | Channel-balanced robustness reanalysis | 10,000 channel-then-event hierarchical bootstrap resamples (seed 20260713) from archived event residuals; equal channel weights | Whether pooled effects survive removal of event-count dominance by large channels | Directions retained: OF−EMPCA [110.69, 1537.48]; PCA−EMPCA [1.22, 6.69]; intervals coarse (7 clusters) |
| CRESST trigger efficiency | Detector-facing sensitivity metric | Injection grid with 5% FP threshold per channel | Whether reconstruction gain translates into trigger sensitivity | Inconclusive (underpowered grid); correctly excluded from claims |

## 3. Public GWOSC calibration boundary

Implementation: `src/noise_geometry/gwosc/` (analysis, diagnostics, reference,
waveforms), configs `configs/gwosc/*.yaml`.

| ID | Description | Method | Target | Status |
| --- | --- | --- | --- | --- |
| GWOSC-A | Global-PSD null calibration | 256 s around GW150914, 62 off-source windows/detector; Hann bias-corrected median-Welch PSD; 5 random + 5 chronological calibration/evaluation splits; null score spread vs PSD prediction | Whether a stationary global Gaussian PSD model calibrates a matched statistic on real interferometer noise | Negative: ratios 1.99–3.18 (H1), 1.26–2.96 (L1) |
| GWOSC PSD reference check | Implementation identity vs GWpy | Identical windows through repo estimator and GWpy | Isolates implementation error from model inadequacy | Pass (ratio 1.000) |
| GWOSC shared-FIR check | Two convolution paths, same FIR coefficients and normalization | Explicit FFT convolution vs GWpy convolution | Software identity for the common finite statistic (not equivalence to the PSD-domain GLS statistic) | Pass (≤ 3×10⁻¹³) |
| GWOSC time-local PSD | Predeclared local-stationarity repair attempt | Leave-one-out global vs local PSD radii 32/64/96 s; 64 s + acceptance [0.8, 1.2] predeclared primary; stationary synthetic control | Whether simple local stationarity restores calibration | Negative: synthetic control 1.148 (pass); H1 9.424, L1 8.827 (fail). Falsifies the predeclared hypothesis; kept as boundary result |

## 4. Noise-budget studies (Study A / Study B)

Implementation: `scripts/run_noise_budget.py`, source spectra
`noise_module/Al2O3_Al_athermal/{SQUID,Johnson,TD,Er,total}_noise.dat`.

| ID | Description | Method | Target | Status |
| --- | --- | --- | --- | --- |
| Study A | Component noise-budget audit | Square component ASDs, sum PSDs, compare to supplied total; band fractions and crossovers under native labels | Verify the supplied additive component model is internally consistent and report which labeled component dominates each band | Positive (rebuilt): SQUID dominates all analyzed bands (median 0.867–0.967); Johnson–Er crossover ≈ 319.7 Hz, Johnson–TD ≈ 688.6 Hz. Not a decomposition of a measured total — an audit of supplied curves |
| Study B | Conditional OF sensitivity attribution | Leave-one-out and 0.5×/0.2× rescaling per component in `N_Φ = Σ|S_k|²/J_k`; two explicit acquisition branches (in-band no-alias vs unfiltered alias-fold); cross-checked vs independent OF-variance implementation (< 0.02%) | Which modeled component limits amplitude resolution, conditional on the declared additive model and acquisition branch | Implemented; paper reports no numbers pending acquisition/template metadata (correct restraint per §10 of the paper) |

## 5. Paper 2 experiments (exploratory; separate claim set)

Implementation: `paper2/` and `NPML/`. Status per `docs/CURRENT_STATUS.md`: all
neural/coverage results are exploratory until the documented defects (RNG
seeding, repeated test-set evaluation, timing-coordinate metric, Transformer
positional/cross-channel encoding) are repaired and rerun from a frozen
protocol.

| ID | Description | Target | Status |
| --- | --- | --- | --- |
| NFPA main demo | Kronecker-factored tied linear AE vs EMPCA vs MSE-trained factored model; rank-1 NFPA vs OF | Restricted bridge theorem; metric reversal persists under architecture constraint | Single-seed proof of concept only |
| NPML-A/B/C | Metric ablation; synthetic coverage ablation; NFPA nonseparability sweep | Metric vs architecture vs coverage separation | Single-seed prototypes; stale dependencies |
| Real AE 2×2, metric×coverage 2×2, Transformer | Raw/prewhitened × MSE/Mahalanobis on K-α traces; coverage restriction; patch Transformer | Tuple (Σ⁻¹, F, P_train) factorial on real data | Invalid for inference until repairs; do not cite numbers |

## 6. Provenance and reproducibility infrastructure

Not experiments, but part of the evidence chain: Stage 0 remote gate
(`scripts/stage0_remote_gate.py`), paper bundle export/verify
(`scripts/export_paper_bundle.py`, `scripts/verify_paper_bundle.py`), evidence
audit (`scripts/audit_paper_evidence.py`), leakage audits inside the CRESST
runner (row/hash overlap, chronological flags), and per-run JSON records with
git commit, config, and dataset metadata.
