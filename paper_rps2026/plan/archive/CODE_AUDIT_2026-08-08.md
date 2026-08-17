# Code audit — NFPA confirmatory experiment scaffold

**Date:** 2026-08-08
**Plan:** `plan/EXPERIMENT_PLAN_2026-08-08.md` (R/PS 2026) + authoritative protocol `notes/nfpa_submission_experiment_plan.md`
**Git commit hash:** `e6d720851066d8c532ac2ccbd33ef6cd75756cf2` (`noise-weighted-subspace-reconstruction@HEAD`, "Create DATASETS.md")

---

## 1. Location discrepancy, resolved

The experiment plan says to reuse "the Paper 2 implementation at `~/Documents/paper2-nonlinear-reconstruction`". That repository is **LaTeX-only**: `plan/status_and_open_work.md` states "No experiments are run — all sweeps in `experiments_plan.md` are specifications", and it contains no Python. The actual Paper 2 implementation lives in `~/Documents/noise-weighted-subspace-reconstruction`:

- `src/NFPA/nfpa_demo.py` — the archived NFPA proof-of-concept run whose numbers appear in the mini-paper's Table 1.
- `src/noise_geometry/` — the reusable primitives (noise, covariance, whitening, metrics, subspace, synthetic).
- `src/canonical/` — EMPCA / OF implementations.

All audit records below point at that repository. The scaffold is added at `experiments/synthetic/nfpa_confirmatory/` in the same repository.

---

## 2. Implementations audited

### 2.1 NFPA / ALS — `src/NFPA/nfpa_demo.py`

- Entry point is module-level code (runs on import). Deterministic: `np.random.seed(42)`.
- ALS updates (`als_Ut`, `als_Uc`): alternating spectral updates. `als_Ut` contracts `Z` over the channel axis with `U_c` then takes the top `k_t` right singular vectors of the stacked `(N*k_c, T)` matrix; `als_Uc` is the transpose analogue. Objective: chi² (MSE in whitened space) for NFPA, raw MSE for IsoAE.
- Whitening is a **calibrated frequency-domain filter**: `white_f = wf_raw / std(ref_whitened_noise)` where the reference is 2000 generated noise traces. This is an empirical calibration, not the analytic `Σ^{-1/2}`; the new harness uses the analytic separable `Σ^{-1/2}` instead (see §3), so the two pipelines differ in exact numeric value but must agree on the *degenerate-case reproduction* (Phase 0 test 7, §4.7).
- Reproduces the archived numbers exactly when re-run (verified 2026-08-08): NFPA χ̄² = 0.99161, IsoAE χ̄² = 0.99949, NFPA–CW-PCA angles `[0.0, 59.3, 64.6, 73.8, 84.8, 87.8]`, IsoAE–CW-PCA angles `[82.1, 83.7, 86.0, 86.6, 88.2, 89.4]`, mean angles 61.7° / 86.0°, IsoAE/IsoPCA raw MSE 4.58e-09 / 4.33e-09.

### 2.2 CW-PCA / PCA — `src/noise_geometry/subspace/pca.py`

- `fit_pca`: SVD of row-wise observations, `components = vh[:rank]` (rows are orthonormal basis vectors).
- `fit_weighted_pca`: PCA in whitened coordinates; for a full weight matrix computes `transform = V sqrt(λ)` and maps components back to data units (`components = vh[:r] @ inverse_transform`).
- `project_onto_basis`: weighted-LS reconstruction onto a row basis. The new harness uses its own projection (metric-orthogonal projector in whitened coordinates) for consistency, and uses `fit_pca` only for IsoPCA/CW-PCA in the chosen coordinate space.
- `principal_angles` (`subspace/angles.py`): subspace angles via QR + SVD of `Qa.T Qb`, with optional metric weighting. Reused for all angle reports.

### 2.3 Noise, covariance, whitening — `src/noise_geometry/noise/`

- `covariance.py`: `estimate_covariance`, `regularize_covariance` (eigenvalue floor), `inverse_covariance`, `covariance_from_psd` (circulant stationary covariance from a one-sided PSD, matching the rFFT convention), `block_covariance` = `kron(channel_cov, time_cov)`.
- `synthetic.py`: `make_powerlaw_psd` (kind `"brownian"` → 1/f²), `generate_colored_noise` (rFFT-domain circular complex Gaussian, DC/Nyquist real with √2 factor).
- `whitening.py`: `whiten_with_covariance` applies `Σ^{-1/2}` to **row-wise** samples; the new harness needs the **separable (C,T)-matrix form** `Z = Σ_c^{-1/2} X Σ_t^{-1/2}`, implemented in the harness as `whiten_separable`.
- `psd.py`: `inverse_psd_weights`, `regularize_psd`, `estimate_psd_rfft`.

**Convention note (critical).** The paper's Eq. (8) `vec(U_c U_c^T Z U_t U_t^T) = (U_t U_t^T ⊗ U_c U_c^T) vec(Z)` holds for **column-major vec** (`vec(A Z B) = (B^T ⊗ A) vec(Z)`). Under this convention the full covariance of `vec(Z)` is `Σ = kron(Σ_t, Σ_c)` (temporal blocks × channel covariance), *not* `block_covariance(Σ_c, Σ_t)` which is the row-major arrangement. The harness therefore builds `Σ = np.kron(Σ_t, Σ_c)` explicitly and flattens with `order="F"` in every Kronecker identity and subspace-basis computation. Test 4 verifies the identity numerically and would catch any convention drift.

### 2.4 Metrics — `src/noise_geometry/metrics/reconstruction.py`

- `mse` (measurement space), `weighted_residual` (per-row `(x-xhat)^T Σ^{-1} (x-xhat)`), `gaussian_nll`, `amplitude_resolution`. Reused; the harness adds thin wrappers for `raw_mse` and `normalized_chi2` per protocol §9 and evaluates in the **oracle** generating covariance.

### 2.5 OF — `src/noise_geometry/filters/optimal.py`

- `gls_amplitude`, `matched_filter_score`, `project_rank1`. Reused for the OF amplitude reference and the rank-one check.

### 2.6 Archived Papers 1/3 experiments (context, not re-run)

- `experiments/synthetic/metric_reversal/run.py` — Paper 1's 30-seed metric-reversal demo (used the numbers in the mini-paper §4 line 4). Not part of Blocks A/B/H.
- `transfer_paper/tables/gwosc_*.csv` — Paper 3 GWOSC calibration evidence (used by the desk agent's D4 paragraph). Not re-run.

---

## 3. Scaffold design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Harness location | `experiments/synthetic/nfpa_confirmatory/` | Mirrors existing `experiments/synthetic/{metric_reversal,of_empca_equivalence}` layout; keeps new code next to its reuse targets |
| Vectorization convention | column-major (`order="F"`), `Σ = kron(Σ_t, Σ_c)` | Matches paper Eq. (8); verified by correctness test 4 |
| Whitening | separable analytic `Z = Σ_c^{-1/2} X Σ_t^{-1/2}` | Axis-preserving per protocol §5; distinct from the archived demo's empirically calibrated FFT whitening, which is why test 7 runs the archived pipeline itself |
| Centering | training-measurement mean estimated on train only, subtracted from all splits, added back on reconstruction | Protocol §7; leakage-audited |
| Iso-MPCA vs IsoAE | reported as **one** method ("Iso-MPCA", ALS on raw MSE); a tied linear AE under identity metric minimizes the same factored projection objective, so it is the same estimator up to parameterization. Correctness test documents the numerical equivalence with a gradient-descent IsoAE on a small case | Protocol §6 resolution |
| OF template | the planted nominal template `v_1 t_1^T` (known in synthetic study) | Protocol §6 |
| Block B | same clean signals per seed, `Σ = σ² I` (`Σ_c = I`, `Σ_t = I`); equivalence margin predeclared in `config.py` | Protocol Block B |
| Outputs | raw per-(seed, method, condition, split) CSV written before aggregation; deterministic aggregation script | Protocol §13 |

---

## 4. Phase 0 correctness tests (7) and leakage tests

File: `experiments/synthetic/nfpa_confirmatory/tests/test_correctness.py` and `tests/test_leakage.py`.

| # | Test | Method | Pass criterion |
|---|---|---|---|
| 1 | Whitening identity | `‖W Σ W^T − I‖_F / d`, `W = Σ^{-1/2}` | `< 1e-8` (protocol §5 check 3) |
| 2 | CW-PCA = PCA under white noise | angle between whitened CW-PCA and IsoPCA subspaces with `Σ = σ² I` | `< 1e-6` degrees (Paper 1 order) |
| 3 | Rank-one NFPA = OF | cosine of rank-one NFPA vs OF template in whitened space | `≥ 0.9999` (Corollary 1) |
| 4 | Kronecker projector identity | `vec(U_c U_c^T Z U_t U_t^T) = (U_t U_t^T ⊗ U_c U_c^T) vec(Z)` | exact to `1e-10` |
| 5 | Projector idempotence/symmetry | `P^2 = P`, `P = P^T` for `P = (U_t U_t^T ⊗ U_c U_c^T)` | `1e-10` |
| 6 | NFPA with `k_c=C, k_t=T` is identity | reconstruction error | `≈ 0` (`< 1e-10`) |
| 7 | Degenerate-case reproduction | run archived `nfpa_demo.py` (C=2, k_c=2) and re-check the archived numbers | NFPA χ̄² 0.9916, CW-PCA 0.9929, angle vector `[0.0, 59.3, 64.6, 73.8, 84.8, 87.8]` |

Leakage tests: a harness audit test asserts the generator exposes a **guard** that fails loudly if any validation/test event is passed where a covariance is fit, if the test split is consumed before final evaluation, or if the training mean is re-estimated on a non-training split. The runner stores the frozen configuration hash in every output row.

---

## 5. Phase 0 gate status

- All seven correctness tests: **implemented; pass/fail recorded in `plan/EXPERIMENT_LOG_2026-08-08.md`**.
- Leakage tests: **implemented; pass/fail recorded in the experiment log**.
- Degenerate-case reproduction (test 7): the archived pipeline reproduces the mini-paper Table 1 numbers exactly when run (independent verification 2026-08-08, values in §2.1).
