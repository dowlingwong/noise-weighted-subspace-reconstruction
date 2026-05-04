# Experiment Verification Checklist

_Maps every theorem and claim in the paper to a concrete experiment.
Updated to reflect: (1) precise paper section labels, (2) explicit data source
classification, (3) input/output specification, (4) QPSimulator API as implemented._

**QPSimulator API (all four additions implemented ✅):**
- `generate(arrival_times, return_amplitude=False)` — single clean trace ± ground-truth amplitude
- `generate_family(n_events, tau_decay_range, t0_jitter_range, n_QP_range, rng)` → `(traces, params)`
- `get_template_at_shift(t0_shift_ns)` → shifted template `(trace_samples,)`
- `estimate_psd(noise_traces, sampling_frequency)` → `(freqs, J_k)` — static method

**Noise modules (no changes needed):**
`NoiseGenerator`, `TemporalNoiseWrapper`, `ArtifactInjector`, `MultiChannelNoiseGenerator`

**Data source classification used below:**
- **SIM-single** — one clean trace from `generate()` + repeated independent noise draws
- **SIM-batch** — traceset from `generate_family()` + noise applied externally
- **CAL-kalpha** — real K-alpha calibration traces from the detector

---

## E1 — Theorem 1: rank-1 EMPCA ≡ OF amplitude estimator

**Paper section:** §4 (`\label{thm:rank1-equivalence}`, `\label{subsec:equiv-rank1}`);
numerical summary populates `\label{tab:of-empca-verification}` in §3 (`\label{subsec:equiv-verification}`).

**Purpose:** Verify that under matched whitening, rank-1 EMPCA produces the identical
template direction and per-event amplitude estimates as optimal filtering — the numerical
confirmation of Theorem 1 (rank-1 equivalence).

**Data source:** SIM-batch

**Input:**
```python
sim = QPSimulator()   # tau_decay=3e6, tau_rise=50e3, trigger_time=default
traces, params = sim.generate_family(
    n_events=500, n_QP_range=(5000, 5000), rng=rng
)
ng = NoiseGenerator(dict(noise_type='pink', noise_power=1.0, sampling_frequency=2.5e5))
noisy = traces + np.stack([ng.generate_noise(sim.trace_samples) for _ in range(500)])
freqs, Jk = QPSimulator.estimate_psd(noise_cal_2000, sim.frequency)
```

**Expected output / pass thresholds:**

| Metric | Formula | Pass |
|---|---|---|
| Weighted subspace cosine | `ρ_w(U_EMPCA, s₀)` | > 0.9999 |
| Amplitude correlation | `corr(A_EMPCA, A_OF)` | > 0.999 |
| Median relative error | `median(|A_EMPCA − A_OF| / A_OF)` | < 1×10⁻³ |
| KS test on residuals | `p-value` | > 0.05 |

**Status:** Partially done (real-data result 0.9999999655 in `tab:of-empca-verification`).
Needs full simulation script producing all four metrics systematically.

---

## E2 — Theorem 2 (Bridge Theorem): noise-aware linear AE ≡ EMPCA

**Paper section:** §5 (`\label{thm:bridge}`, `\label{subsec:ae_pca_equiv}`);
numerical summary populates `\label{tab:empca_ae_primary}` in §5 (`\label{subsec:numerical_ae}`).

**Purpose:** Verify that at convergence, the rank-k noise-aware tied linear AE spans the same
subspace as rank-k EMPCA — numerical confirmation of the Bridge Theorem.

**Data source:** SIM-batch (same traceset as E1, reused)

**Input:**
```python
# Method A: run rank-k EMPCA with weight=1/Jk → basis U_EMPCA (d×k)
# Method B: SVD of whitened data matrix X_tilde = X @ inv_sqrt_Sigma → W_AE (d×k)
# Repeat for k = 1, 2, 3
```

**Expected output / pass thresholds:**

| Metric | Formula | Pass |
|---|---|---|
| Principal-angle cosine (k=1) | `cos θ₁(span(U), span(W))` | > 0.9999 |
| Principal-angle cosine (k=2) | `cos θ₁, cos θ₂` | > 0.9999 each |
| Principal-angle cosine (k=3) | `cos θ₁, cos θ₂, cos θ₃` | > 0.9999 each |
| Relative residual difference | `‖res_EMPCA − res_AE‖ / ‖res_EMPCA‖` | < 1×10⁻⁵ |
| KS test on residuals | `p-value` | > 0.2 |

**Status:** Partially done. Real-data result exists. Needs systematic simulation script for k=1,2,3.

---

## E3 — Proposition: χ²(k) monotone decrease with rank

**Paper section:** §4 (`\label{subsec:equiv-kgreater1}`).

**Purpose:** Verify that χ²_EMPCA(k) ≤ χ²_EMPCA(k−1) numerically, with strict inequality
when the signal family has dimension > 1.

**Data source:** SIM-batch (two separate tracesets)

**Input — Setup A (1D family):**
```python
traces_A, _ = sim.generate_family(
    n_events=1000, tau_decay_range=(3e6, 3e6), n_QP_range=(2000, 8000), rng=rng
)
# Add pink noise; run EMPCA for k=1…8; record χ²_test(k)
```

**Input — Setup B (multi-dimensional family):**
```python
traces_B, _ = sim.generate_family(
    n_events=1000, tau_decay_range=(1e6, 5e6),
    t0_jitter_range=(-1e5, 1e5), n_QP_range=(3000, 7000), rng=rng
)
# Add pink noise; run EMPCA for k=1…8; record χ²_test(k)
```

**Expected output / pass thresholds:**

| Setup | Metric | Expected |
|---|---|---|
| A | Δχ²(2)/χ²(1) | < 1% (plateau) |
| B | Δχ²(2)/χ²(1) | > 5% (strict improvement) |
| Both | χ²(k) sequence | Monotone non-increasing for k=1…8 |

**Deliverable:** Figure — χ²_test(k) vs k, two curves on the same axes.

---

## E4 — CRB: empirical Var(Â) = 1/N_Φ

**Paper section:** §3 (`\label{subsec:of-crb}`), equations `\eqref{eq:of-variance}` and `\eqref{eq:of-fisher}`.

**Purpose:** Verify the Cramér-Rao bound: OF amplitude estimator achieves `Var(Â) = 1/N_Φ`.

**Data source:** SIM-single

**Input:**
```python
trace_clean, A_true = sim.generate([sim.trigger_time], return_amplitude=True)
ng = NoiseGenerator(dict(noise_type=noise_type, noise_power=pw, sampling_frequency=2.5e5))
A_hat = [OF_amplitude(trace_clean + ng.generate_noise(N)) for _ in range(5000)]
# N_Phi = sum(|S_k|^2 / Jk);  predicted_var = 1/N_Phi
```
Repeat for `noise_type ∈ {white, pink, brownian}` with N_Φ ∈ [10, 100].

**Expected output / pass thresholds:**

| Noise type | Metric | Pass |
|---|---|---|
| white | `|σ²_emp − 1/N_Φ| / (1/N_Φ)` | < 5% |
| pink | same | < 5% |
| brownian | same | < 5% |

**Deliverable:** Table — three rows (noise type), columns: N_Φ, 1/N_Φ, σ²_emp, relative error.

---

## E5 — Energy resolution: σ_E = E₀/√N_Φ

**Paper section:** §3 (`\label{subsec:of-crb}`), equation `\eqref{eq:of-energy-resolution}`.

**Purpose:** Verify the ∝ 1/√noise_power scaling in simulation; ground the formula against
real K-alpha data.

**Data source:** BOTH — SIM-batch (scaling curve) and CAL-kalpha (absolute calibration)

**Input — Simulation:**
```python
for pw in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
    traces, _ = sim.generate_family(n_events=1000, n_QP_range=(5000, 5000), rng=rng)
    noisy = traces + noise_at_power(pw)
    sigma_E_emp[pw] = OF_amplitude_batch(noisy, Jk(pw)).std()
    sigma_E_pred[pw] = E0 / np.sqrt(N_Phi(pw))
```

**Input — Real data (CAL-kalpha):**
```python
freqs, Jk_real = QPSimulator.estimate_psd(baseline_traces, sampling_frequency)
# Compare sigma_E_pred = E_Kalpha/sqrt(N_Phi) vs sigma_E_obs = std(A_OF_Kalpha)
```

**Expected output / pass thresholds:**

| Component | Metric | Pass |
|---|---|---|
| Simulation | slope of log(σ_E) vs log(noise_power) | −0.5 ± 0.05 |
| Simulation | residuals from theory curve | < 10% per point |
| Real data | `|σ_E_pred − σ_E_obs| / σ_E_obs` | < 15% |

**Deliverable:** (1) log-log plot σ_E vs noise_power with theory line; (2) one-row real-data table.

---

## E6 — Noise-aware EMPCA vs isotropic PCA ablation

**Paper section:** §7 (`\subsection{Noise-aware loss versus isotropic MSE}`);
conclusion restated in §8.1 (`\label{subsec:noise-aware-principle}`).

**Purpose:** Main practical ablation. Shows Σ⁻¹-weighted EMPCA beats unweighted PCA under
colored noise, and that the gap grows with noise coloredness.

**Data source:** SIM-batch

**Input:**
```python
traces, _ = sim.generate_family(n_events=700, n_QP_range=(2000, 8000), rng=rng)
train_clean, test_clean = traces[:500], traces[500:]

for noise_type in ['white', 'pink', 'brownian']:
    Jk = QPSimulator.estimate_psd(noise_cal_500, sim.frequency)[1]
    U_empca = run_empca(train_clean + noise(500), Jk, k=3)
    U_pca   = run_pca(train_clean + noise(500), k=3)
    rel_improvement[noise_type] = (
        weighted_chi2(test_clean + noise(200), U_pca,   Jk) -
        weighted_chi2(test_clean + noise(200), U_empca, Jk)
    ) / weighted_chi2(test_clean + noise(200), U_pca, Jk)
```

**Expected output / pass thresholds:**

| Noise type | `(χ²_PCA − χ²_EMPCA) / χ²_PCA` | Expected |
|---|---|---|
| white | ≈ 0 | control |
| pink | > 5% | EMPCA wins |
| brownian | > 15% | EMPCA wins strongly |

**Deliverable:** Bar chart — three noise conditions × two methods. Primary figure for §7.

---

## E7 — Template mismatch: fixed-template OF bias and EMPCA recovery

**Paper section:** §3 (`\label{subsec:of-limitation}`), equations `\eqref{eq:of-bias}`.

**Purpose:** Verify the bias formula `E[Â_OF] = A · cos²θ_w` and show rank-2 EMPCA
recovers the unbiased amplitude when signal shapes vary.

**Data source:** SIM-batch

**Input:**
```python
traces_mis, params_mis = sim.generate_family(
    n_events=500, tau_decay_range=(1e6, 5e6),
    n_QP_range=(5000, 5000), rng=rng
)
noisy_mis = traces_mis + noise_batch('pink', 500)
Jk = QPSimulator.estimate_psd(noise_cal_200, sim.frequency)[1]
# Method A: OF with nominal template (tau_decay=3e6)
# Method B: rank-1 EMPCA
# Method C: rank-2 EMPCA
# Ground truth: params_mis['amplitude_ADC']
```

**Expected output / pass thresholds:**

| Metric | Method A (OF) | Method C (EMPCA k=2) |
|---|---|---|
| Mean bias `|E[Â] − A_true| / A_true` | > 10% | < 3% |
| Bias vs cos²θ_w | matches formula within 5% | — |

**Deliverable:** (1) Scatter Â vs A_true; (2) bias table; (3) formula-verification curve.

---

## E8 — Time-shift OF: arrival time and amplitude recovery

**Paper section:** §3 (`\label{subsec:equiv-shifted-of}`), equations
`\eqref{eq:equiv-shifted-of}` and `\eqref{eq:equiv-shifted-of-t0}`.

**Purpose:** Verify that time-shift OF recovers both t̂₀ and amplitude correctly;
confirm SNR degradation of fixed-template OF on jittered data.

**Data source:** SIM-batch

**Input:**
```python
traces_jit, params_jit = sim.generate_family(
    n_events=300,
    t0_jitter_range=(-sim.trace_duration * 0.1, sim.trace_duration * 0.1),
    n_QP_range=(5000, 5000), rng=rng
)
noisy_jit = traces_jit + noise_batch('pink', 300)
Jk = QPSimulator.estimate_psd(noise_cal_200, sim.frequency)[1]

# Time-shift OF filter bank using new API:
for j, t0 in enumerate(t0_grid):
    s_shift = sim.get_template_at_shift(t0)   # ← get_template_at_shift
    A_grid[j] = OF_amplitude(noisy, s_shift, Jk)
t_hat = t0_grid[np.argmax(A_grid, axis=1)]
# Ground truth: params_jit['t0_shift']
```

**Expected output / pass thresholds:**

| Metric | Time-shift OF | Fixed OF |
|---|---|---|
| Arrival-time RMSE | < 2 samples | N/A |
| Mean amplitude bias | < 5% | > 5% for large-jitter events |

**Deliverable:** (1) RMSE table; (2) scatter Â vs A_true; (3) timing-residual histogram.

---

## E9 — Convergence: EM iterations vs χ²

**Paper section:** §6 (`\label{subsec:convergence-theorem}`, Theorem `thm:convergence`).

**Purpose:** Verify monotone non-increase of χ² per EM step and characterize empirical
convergence speed. Test sensitivity to initialization.

**Data source:** SIM-batch (genuine 2D family — Setup B from E3)

**Input:**
```python
traces_conv, _ = sim.generate_family(
    n_events=500, tau_decay_range=(1e6, 5e6), rng=rng
)
noisy_conv = traces_conv + noise_batch('pink', 500)
# Run EMPCA 100 iterations for k=1,2,3; record χ²(r) each step
# Two inits: random Haar vs SVD-seeded
```

**Expected output / pass thresholds:**

| Metric | Pass |
|---|---|
| χ²(r) monotone | True for all r=1…100 |
| Convergence (k=1) | `|Δχ²| / χ²(0) < 1e-6` within 20 iters |
| Convergence (k=2,3) | within 50 iters |
| Init independence | `|χ²_rand(∞) − χ²_svd(∞)| / χ²(0) < 1e-5` |

**Deliverable:** Figure — χ²(r) vs r, k=1 and k=2 panels, two init curves each.

---

## E10 — Non-stationary noise robustness

**Paper section:** §6 (`\label{subsec:noise-assumptions}`).

**Purpose:** Show EMPCA degrades with non-stationary noise under global PSD; per-segment
PSD partially recovers performance.

**Data source:** SIM-batch

**Input:**
```python
traces_ns, _ = sim.generate_family(n_events=500, rng=rng)
tw = TemporalNoiseWrapper(base_ng, mode='piecewise', n_segments=4, scale_range=(0.7, 1.3))
noisy_ns = traces_ns + np.stack([tw.apply(np.zeros(N)) for _ in range(500)])
# Case A: single global PSD from all noise traces
# Case B: per-segment PSD (125 traces each)
```

**Expected output / pass thresholds:**

| Case | Test χ² | Amplitude RMSE |
|---|---|---|
| A (global PSD) | elevated | higher |
| B (per-segment) | closer to stationary | lower |
| Improvement B vs A | > 5% χ² reduction | quantified |

---

## E11 — Artifact robustness

**Paper section:** §6 (`\label{subsec:limitations}`), §8.3.

**Purpose:** Show glitch artifacts contaminate EMPCA subspace; χ²-threshold flagging restores
performance.

**Data source:** SIM-batch

**Input:**
```python
traces_art, params_art = sim.generate_family(n_events=500, rng=rng)
ai = ArtifactInjector(config=dict(glitch_rate=0.1, impulse_rate=0.05))
noisy_art = [trace + ng.generate_noise(N) + ai.apply(np.zeros(N)) for trace in traces_art]
# Pass 1: rank-2 EMPCA on all 500 (contaminated)
# Pass 2: flag χ²_noise > 5σ; re-run on clean subset
```

**Expected output / pass thresholds:**

| Pass | Amplitude RMSE | χ²_test |
|---|---|---|
| Without flagging | higher | elevated |
| After flagging | lower (> 10% improvement) | reduced |

---

## E12 — Real K-alpha data: full equivalence verification

**Paper section:** §7 (`\subsection{Verification of equivalence theorems}`),
§3 (`\label{tab:of-empca-verification}`), §5 (`\label{tab:empca_ae_primary}`).

**Purpose:** Validate E1 and E2 on real K-alpha calibration data. Add the amplitude histogram
(needed for CRB visual verification) currently missing from §7.

**Data source:** CAL-kalpha

**Input:**
```python
freqs, Jk_real = QPSimulator.estimate_psd(baseline_traces, sampling_frequency)
# Replicate E1 (OF vs rank-1 EMPCA) on real traces
# Replicate E2 (principal angles) on real traces
# NEW: amplitude histogram → fit Gaussian → extract σ_A_obs
# Compare σ_A_obs to 1/sqrt(N_Phi) from Jk_real
```

**Expected output / pass thresholds:**

| Metric | Pass |
|---|---|
| ρ_w cosine (E1 real data) | > 0.9999 (existing: 0.9999999655 ✓) |
| Principal-angle cosines (E2) | > 0.9999 |
| Amplitude histogram Shapiro-Wilk | p > 0.05 |
| `|σ_A_obs − 1/√N_Φ| / (1/√N_Φ)` | < 15% |

**Missing:** amplitude histogram figure + σ_A comparison row in `tab:of-empca-verification`.

**Deliverable:** (1) Amplitude histogram with Gaussian overlay; (2) updated table with σ_A row.

---

## Simulator status summary

| Addition | Status | Used by |
|---|---|---|
| `generate_family()` | ✅ implemented | E1, E2, E3, E6, E7, E8, E9, E10, E11 |
| `get_template_at_shift()` | ✅ implemented | E8 |
| `estimate_psd()` | ✅ implemented | E1–E12 (every experiment) |
| `generate(return_amplitude=True)` | ✅ implemented | E4, E5, E7 |

No changes needed to `NoiseGenerator`, `ArtifactInjector`, `TemporalNoiseWrapper`,
or `MultiChannelNoiseGenerator`.

---

## Result table template for §7

| Exp | Method | Noise / Data | Key metric | Value |
|---|---|---|---|---|
| E1 | OF vs rank-1 EMPCA | SIM pink | ρ_w cosine | ≥ 0.9999 |
| E1 | OF vs rank-1 EMPCA | SIM pink | amp correlation | ≥ 0.999 |
| E2 | EMPCA vs AE (k=1) | SIM pink | principal-angle cos | ≥ 0.9999 |
| E2 | EMPCA vs AE (k=2) | SIM pink | principal-angle cos | ≥ 0.9999 |
| E3 | χ²(k) plateau | SIM pink, fixed τ | Δχ²(2)/χ²(1) | < 1% |
| E3 | χ²(k) drop | SIM pink, varied τ | Δχ²(2)/χ²(1) | > 5% |
| E4 | CRB | SIM white | `|σ²_emp − 1/N_Φ|/(1/N_Φ)` | < 5% |
| E4 | CRB | SIM pink | same | < 5% |
| E5 | σ_E scaling | SIM pink sweep | slope log σ_E vs log pw | −0.5 ± 0.05 |
| E5 | σ_E formula | CAL K-alpha | `|σ_pred − σ_obs|/σ_obs` | < 15% |
| E6 | EMPCA vs PCA | SIM white | Δχ²/χ²_PCA | ≈ 0 (control) |
| E6 | EMPCA vs PCA | SIM pink | Δχ²/χ²_PCA | > 5% |
| E6 | EMPCA vs PCA | SIM brownian | Δχ²/χ²_PCA | > 15% |
| E7 | OF bias formula | SIM mismatched | `|E[Â_OF] − A·cos²θ|` | < 5% |
| E8 | time-shift OF | SIM jittered | t̂₀ RMSE | < 2 samples |
| E9 | EM convergence | SIM pink varied τ | iters to 1e-6 tol | ≤ 50 |
| E12 | Real K-alpha (E1) | CAL K-alpha | ρ_w cosine | 0.9999999655 ✓ |
| E12 | Real K-alpha (CRB) | CAL K-alpha | `|σ_obs − 1/√N_Φ|/...` | TBD |

---

## Priority order

1. **E1** — Theorem 1: direct numerical verification (§4)
2. **E2** — Bridge Theorem: direct numerical verification (§5)
3. **E6** — EMPCA vs PCA ablation: main practical claim (§7)
4. **E3** — χ²(k) curve: Proposition verification (§4)
5. **E4** — CRB: energy resolution bound (§3)
6. **E12** — Real data: amplitude histogram missing (§7)
7. **E7** — Template mismatch bias: limitation section (§3)
8. **E8** — Time-shift OF: extension verification (§3)
9. **E9** — Convergence plot: §6 theorem support
10. **E5** — σ_E scaling: simulation + real data (§3)
11. **E10/E11** — Robustness: supports §6 limitations discussion
