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
# Add pink noise
ng = NoiseGenerator(dict(noise_type='pink', noise_power=1.0, sampling_frequency=2.5e5))
noisy = traces + np.stack([ng.generate_noise(sim.trace_samples) for _ in range(500)])
# PSD from 2000 calibration noise traces
freqs, Jk = QPSimulator.estimate_psd(noise_cal_2000, sim.frequency)
```

**Expected output / pass thresholds:**

| Metric | Formula | Pass |
|---|---|---|
| Weighted subspace cosine | `ρ_w(U_EMPCA, s₀)` | > 0.9999 |
| Amplitude correlation | `corr(A_EMPCA, A_OF)` | > 0.999 |
| Median relative error | `median(|A_EMPCA − A_OF| / A_OF)` | < 1×10⁻³ |
| KS test on residuals | `p-value` | > 0.05 |

**Status:** Partially done (Table `tab:of-empca-verification` has real-data result 0.9999999655).
Needs full simulation script producing all four metrics systematically.

---

## E2 — Theorem 2 (Bridge Theorem): noise-aware linear AE ≡ EMPCA

**Paper section:** §5 (`\label{thm:bridge}`, `\label{subsec:ae_pca_equiv}`);
numerical summary populates `\label{tab:empca_ae_primary}` in §5 (`\label{subsec:numerical_ae}`).

**Purpose:** Verify that at convergence, the rank-k noise-aware tied linear AE spans the same
subspace as rank-k EMPCA — numerical confirmation of the Bridge Theorem. Both methods are
applied to identical data; the comparison is purely subspace geometry.

**Data source:** SIM-batch (same traceset as E1, reused)

**Input:**
```python
# Same noisy traces and Jk as E1
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

**Paper section:** §4 (`\label{subsec:equiv-kgreater1}`), specifically the claim that
`χ²_EMPCA(k) ≤ χ²_EMPCA(k−1)` with strict inequality when the signal family has dimension > 1.

**Purpose:** Verify the rank-k improvement proposition numerically.
Two setups test both the equality case (k=1 is optimal, plateau expected)
and the strict inequality case (genuine multi-dimensional family, χ² drops for k≥2).

**Data source:** SIM-batch (two separate tracesets)

**Input — Setup A (1D family, k=1 is optimal):**
```python
traces_A, params_A = sim.generate_family(
    n_events=1000, tau_decay_range=(3e6, 3e6),   # fixed shape
    n_QP_range=(2000, 8000), rng=rng              # amplitude varies only
)
# Add pink noise; run EMPCA for k=1…8; record χ²_test(k)
```

**Input — Setup B (multi-dimensional family):**
```python
traces_B, params_B = sim.generate_family(
    n_events=1000, tau_decay_range=(1e6, 5e6),   # shape varies → genuine 2D manifold
    t0_jitter_range=(-1e5, 1e5),                 # timing jitter → 2nd dimension
    n_QP_range=(3000, 7000), rng=rng
)
# Add pink noise; run EMPCA for k=1…8; record χ²_test(k)
```

**Expected output / pass thresholds:**

| Setup | Metric | Expected |
|---|---|---|
| A | Δχ²(2)/χ²(1) | < 1% (plateau — no genuine 2nd mode) |
| B | Δχ²(2)/χ²(1) | > 5% (strict improvement — genuine 2nd mode present) |
| Both | χ²(k) sequence | Monotone non-increasing for all k=1…8 |

**Deliverable:** Figure — χ²_test(k) vs k, two curves (Setup A and B) on the same axes.

---

## E4 — CRB: empirical Var(Â) = 1/N_Φ

**Paper section:** §3 (`\label{subsec:of-crb}`), specifically equations
`\eqref{eq:of-variance}` and `\eqref{eq:of-fisher}`.

**Purpose:** Verify the Cramér-Rao bound claim: the OF amplitude estimator achieves
the Fisher-information lower bound, i.e. `Var(Â) = 1/N_Φ` with `N_Φ = Σ_k |S_k|²/J_k`.

**Data source:** SIM-single

**Input:**
```python
# Fixed signal — delta-function arrivals at trigger_time → known amplitude A_true
trace_clean, A_true = sim.generate([sim.trigger_time], return_amplitude=True)
# M=5000 independent noise realizations; compute Â_OF for each
ng = NoiseGenerator(dict(noise_type=noise_type, noise_power=pw, sampling_frequency=2.5e5))
A_hat = [OF_amplitude(trace_clean + ng.generate_noise(N)) for _ in range(5000)]
# Theoretical: N_Phi = sum(|S_k|^2 / Jk); predicted_var = 1/N_Phi
```
Repeat for `noise_type ∈ {white, pink, brownian}` with `noise_power` calibrated so
`N_Φ ∈ [10, 100]` (mid-SNR regime where finite-sample effects are small).

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

**Purpose:** Verify the energy resolution formula `σ_E = E₀/√N_Φ` and show the ∝ 1/√noise_power
scaling. Includes a real-data component to ground the formula in a physical detector.

**Data source:** BOTH — SIM-batch (scaling curve) and CAL-kalpha (absolute calibration)

**Input — Simulation component:**
```python
# Sweep noise_power ∈ {0.1, 0.5, 1.0, 5.0, 10.0, 50.0}
for pw in noise_power_sweep:
    traces, params = sim.generate_family(
        n_events=1000, n_QP_range=(5000, 5000), rng=rng
    )
    noisy = traces + noise_at_power(pw)           # pink noise
    A_hat = OF_amplitude_batch(noisy, Jk(pw))
    sigma_E_emp[pw] = A_hat.std()
    sigma_E_pred[pw] = E0 / np.sqrt(N_Phi(pw))   # E0 = 5000 * qp_amplitude
```

**Input — Real-data component (CAL-kalpha):**
```python
# Load K-alpha traces; estimate PSD from baseline traces
# Compute N_Phi from measured PSD and K-alpha template S(f)
# Compare sigma_E_pred = E_Kalpha / sqrt(N_Phi) vs sigma_E_obs = std(A_OF_Kalpha)
```

**Expected output / pass thresholds:**

| Component | Metric | Pass |
|---|---|---|
| Simulation | slope of log(σ_E) vs log(noise_power) | −0.5 ± 0.05 |
| Simulation | residuals from predicted curve | < 10% at each point |
| Real data | `|σ_E_pred − σ_E_obs| / σ_E_obs` | < 15% |

**Deliverable:** (1) log-log plot σ_E vs noise_power with theory line; (2) one-row real-data table.

---

## E6 — Noise-aware EMPCA vs isotropic PCA ablation

**Paper section:** §7 (`\subsection{Noise-aware loss versus isotropic MSE}`);
the practical conclusion is restated in §8.1 (`\label{subsec:noise-aware-principle}`).

**Purpose:** Main practical ablation. Demonstrates that EMPCA with Σ⁻¹ weighting achieves
lower test χ² than unweighted PCA under colored noise, and that the gap grows with noise
coloredness. This directly supports the paper's central message that the loss is the
load-bearing component, not the architecture.

**Data source:** SIM-batch

**Input:**
```python
# Train split N=500, test split N=200
traces, params = sim.generate_family(n_events=700, n_QP_range=(2000, 8000), rng=rng)
train_clean, test_clean = traces[:500], traces[500:]

for noise_type in ['white', 'pink', 'brownian']:
    train_noisy = train_clean + noise_batch(noise_type, 500)
    test_noisy  = test_clean  + noise_batch(noise_type, 200)
    Jk = QPSimulator.estimate_psd(noise_cal_500, sim.frequency)[1]

    # Method A: rank-k EMPCA with weight=1/Jk
    U_empca = run_empca(train_noisy, Jk, k=3)
    chi2_empca = weighted_chi2_test(test_noisy, U_empca, Jk)

    # Method B: standard PCA (no weighting, i.e. Jk = ones)
    U_pca = run_pca(train_noisy, k=3)
    chi2_pca = weighted_chi2_test(test_noisy, U_pca, Jk)   # evaluated with true Jk

    rel_improvement[noise_type] = (chi2_pca - chi2_empca) / chi2_pca
```

**Expected output / pass thresholds:**

| Noise type | Relative improvement | Expected direction |
|---|---|---|
| white | `(χ²_PCA − χ²_EMPCA) / χ²_PCA` | ≈ 0 (control — isotropic noise, both methods equivalent) |
| pink | same | > 5% (EMPCA wins) |
| brownian | same | > 15% (EMPCA wins strongly — heavy low-freq coloring) |

**Deliverable:** Bar chart or table, three noise conditions × two methods. This is the primary
figure for §7.

---

## E7 — Template mismatch: fixed-template OF bias and EMPCA recovery

**Paper section:** §3 (`\label{subsec:of-limitation}`), equations `\eqref{eq:of-bias}`.

**Purpose:** Verify the fixed-template bias formula `E[Â_OF] = A · cos²θ_w` (where θ_w is the
whitened angle between the nominal template and the true signal). Demonstrate that rank-k
EMPCA with k≥2 recovers the unbiased amplitude when the signal family has shape variation.

**Data source:** SIM-batch

**Input:**
```python
# Nominal template: tau_decay=3e6 (what OF is built for)
s_nominal = sim.template[:sim.trace_samples]

# Mismatched family: tau_decay varies widely
traces_mis, params_mis = sim.generate_family(
    n_events=500, tau_decay_range=(1e6, 5e6),   # ±67% shape variation
    n_QP_range=(5000, 5000), rng=rng            # fixed amplitude for clean bias measurement
)
noisy_mis = traces_mis + noise_batch('pink', 500)
Jk = QPSimulator.estimate_psd(noise_cal_200, sim.frequency)[1]

# Method A: OF with nominal template → A_OF per event
# Method B: rank-1 EMPCA → A_EMPCA_k1 per event
# Method C: rank-2 EMPCA → A_EMPCA_k2 per event
# Ground truth: params_mis['amplitude_ADC']
```
For the bias formula verification, also sweep a single tau_decay value and compare
`E[Â_OF]` to `A_true × cos²θ_w(tau_decay)`.

**Expected output / pass thresholds:**

| Metric | Method A (OF) | Method C (EMPCA k=2) |
|---|---|---|
| Mean amplitude bias `|E[Â] − A_true| / A_true` | > 10% | < 3% |
| Amplitude RMSE | higher | lower |
| Bias vs cos²θ_w | matches formula within 5% | — |

**Deliverable:** (1) Scatter plot Â vs A_true for OF and EMPCA k=2;
(2) one-row bias table; (3) formula-verification curve E[Â_OF] vs cos²θ_w.

---

## E8 — Time-shift OF: arrival time and amplitude recovery

**Paper section:** §3 (`\label{subsec:equiv-shifted-of}`), equations
`\eqref{eq:equiv-shifted-of}` and `\eqref{eq:equiv-shifted-of-t0}`.

**Purpose:** Verify that time-shift OF (sliding matched filter) correctly recovers both the
peak arrival time t̂₀ and the amplitude for pulses with trigger-time jitter. Confirm the
expected SNR degradation of fixed-template OF relative to time-shift OF on the same data.

**Data source:** SIM-batch

**Input:**
```python
# Jittered traceset
traces_jit, params_jit = sim.generate_family(
    n_events=300,
    t0_jitter_range=(-sim.trace_duration * 0.1, sim.trace_duration * 0.1),  # ±10% jitter
    n_QP_range=(5000, 5000), rng=rng
)
noisy_jit = traces_jit + noise_batch('pink', 300)
Jk = QPSimulator.estimate_psd(noise_cal_200, sim.frequency)[1]

# Method A: time-shift OF filter bank
#   For candidate shifts t0_grid (e.g. 50 values spanning ±10% of trace):
#   s_shift = sim.get_template_at_shift(t0_grid[j])   ← uses new API
#   Â(t0) = OF_amplitude(noisy, s_shift, Jk)
#   t̂₀ = t0_grid[argmax Â(t0)]
# Method B: fixed-template OF (no shift search)
# Ground truth: params_jit['t0_shift']
```

**Expected output / pass thresholds:**

| Metric | Method A (time-shift OF) | Method B (fixed OF) |
|---|---|---|
| Arrival-time RMSE `‖t̂₀ − t₀_true‖₂/√N` | < 2 samples | N/A |
| Amplitude RMSE | lower | higher (bias from timing mismatch) |
| Mean amplitude bias | < 5% | > 5% for large jitter events |

**Deliverable:** (1) RMSE table; (2) scatter plot Â vs A_true for both methods;
(3) histogram of timing residuals t̂₀ − t₀_true.

---

## E9 — Convergence: EM iterations vs χ²

**Paper section:** §6 (`\label{subsec:convergence-theorem}`, Theorem `thm:convergence`).

**Purpose:** Verify Theorem `thm:convergence` (monotone non-increase of χ² per EM step)
and characterize empirical convergence speed. Test sensitivity to initialization.

**Data source:** SIM-batch (Setup B from E3 — genuine 2D family)

**Input:**
```python
# Genuine 2D family for k≥2 to be meaningful
traces_conv, _ = sim.generate_family(
    n_events=500, tau_decay_range=(1e6, 5e6), rng=rng
)
noisy_conv = traces_conv + noise_batch('pink', 500)
Jk = QPSimulator.estimate_psd(noise_cal_200, sim.frequency)[1]

# Run EMPCA for 100 iterations; record χ²(r) at each step
# Repeat for k=1,2,3
# Repeat for two initializations:
#   Init A: random Haar (default)
#   Init B: SVD-seeded (warm start from unweighted SVD)
```

**Expected output / pass thresholds:**

| Metric | Pass |
|---|---|
| χ²(r) monotone | True for all r=1…100 |
| Convergence (k=1) | `|Δχ²(r)| / χ²(0) < 1e-6` within 20 iterations |
| Convergence (k=2) | within 50 iterations |
| Convergence (k=3) | within 50 iterations |
| Limit independence of init | `|χ²_rand(∞) − χ²_svd(∞)| / χ²(0) < 1e-5` |

**Deliverable:** Figure — χ²(r) vs iteration r, two panels (k=1 and k=2), two init curves each.

---

## E10 — Non-stationary noise robustness

**Paper section:** §6 (`\label{subsec:noise-assumptions}`), specifically the stationarity
assumption and its consequences.

**Purpose:** Demonstrate that EMPCA with a single global PSD degrades under non-stationary
noise; show that per-segment PSD estimation partially recovers performance. Supports the
discussion of the Gaussian/stationarity limitation in §6.

**Data source:** SIM-batch

**Input:**
```python
traces_ns, params_ns = sim.generate_family(n_events=500, rng=rng)

# Non-stationary noise: 4 segments, power scale drawn from Uniform(0.7, 1.3)
from noise_module.temporal_noise import TemporalNoiseWrapper
tw = TemporalNoiseWrapper(base_ng, mode='piecewise', n_segments=4, scale_range=(0.7, 1.3))
noisy_ns = traces_ns + np.stack([tw.apply(np.zeros(sim.trace_samples)) for _ in range(500)])

# Case A: single global PSD from all 500 noise-only traces
# Case B: per-segment PSD (125 traces each) → average within segment
# Metric: weighted test χ² and amplitude RMSE for both cases
```

**Expected output / pass thresholds:**

| Case | Test χ² vs stationary baseline | Amplitude RMSE |
|---|---|---|
| A (global PSD) | elevated (degradation visible) | higher |
| B (per-segment PSD) | closer to stationary baseline | lower |
| Improvement B vs A | > 5% reduction in χ² | quantified |

**Deliverable:** Bar chart — test χ² for stationary, non-stationary/global PSD,
non-stationary/per-segment PSD.

---

## E11 — Artifact robustness

**Paper section:** §6 (`\label{subsec:limitations}`, Gaussian noise assumption paragraph);
also relevant to the practical discussion in §8.3.

**Purpose:** Show that glitch artifacts (non-Gaussian events) contaminate the EMPCA subspace
and inflate χ²; demonstrate that a simple χ²-threshold flagging step restores performance.

**Data source:** SIM-batch

**Input:**
```python
traces_art, params_art = sim.generate_family(n_events=500, rng=rng)

# Signal + noise + artifacts on ~10% of traces
from noise_module.artifact_injector import ArtifactInjector
ai = ArtifactInjector(config=dict(glitch_rate=0.1, impulse_rate=0.05))
noisy_art = np.stack([
    trace + ng.generate_noise(N) + ai.apply(np.zeros(N))
    for trace in traces_art
])

# Pass 1: rank-2 EMPCA on all 500 traces (contaminated)
# Pass 2: flag events with χ²_noise > 5σ threshold; re-run EMPCA on clean subset
# Metric: amplitude RMSE vs params_art['amplitude_ADC'] for both passes
```

**Expected output / pass thresholds:**

| Pass | Amplitude RMSE | χ²_test |
|---|---|---|
| Without flagging | higher | elevated |
| After flagging | lower | reduced toward nominal |
| Improvement | > 10% RMSE reduction | — |

**Deliverable:** Table — two rows; amplitude RMSE and test χ² with and without flagging.

---

## E12 — Real K-alpha data: full equivalence verification

**Paper section:** §7 (`\subsection{Verification of equivalence theorems}`) and
§3 (`\label{tab:of-empca-verification}`), §5 (`\label{tab:empca_ae_primary}`).

**Purpose:** Validate E1 and E2 on real K-alpha calibration data, not just simulated traces.
This closes the loop between the theoretical framework and measured detector data.
Also adds the amplitude histogram (needed for CRB visual verification) which is currently
missing from the §7 experimental section.

**Data source:** CAL-kalpha (real data — already partially used)

**Input:**
```python
# Load real K-alpha traces (already available)
# Load baseline traces (no signal) for PSD estimation
freqs, Jk_real = QPSimulator.estimate_psd(baseline_traces, sampling_frequency)
s_real = K_alpha_template   # template from mean of K-alpha traces or from physics model

# Replicate E1: OF amplitude vs rank-1 EMPCA amplitude on real traces
# Replicate E2: principal angles between EMPCA and AE subspaces on real data
# NEW: amplitude histogram → fit Gaussian → extract σ_A_obs
# Compare σ_A_obs to 1/sqrt(N_Phi) from Jk_real
```

**Expected output / pass thresholds:**

| Metric | Pass |
|---|---|
| Weighted subspace cosine ρ_w (E1 on real data) | > 0.9999 (existing: 0.9999999655 ✓) |
| Principal-angle cosines (E2 on real data) | > 0.9999 |
| Amplitude histogram | visually Gaussian; Shapiro-Wilk p > 0.05 |
| `|σ_A_obs − 1/√N_Φ| / (1/√N_Φ)` | < 15% |

**Status:** E1 and E2 real-data cosines already exist in tables.
**Missing:** amplitude histogram figure + σ_A comparison row in table.

**Deliverable:** (1) Amplitude histogram with Gaussian overlay; (2) updated Table `tab:of-empca-verification` with σ_A row.

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

1. **E1** — Theorem 1: direct numerical verification (core result, §4)
2. **E2** — Bridge Theorem: direct numerical verification (core result, §5)
3. **E6** — EMPCA vs PCA ablation: the paper's main practical claim (§7)
4. **E3** — χ²(k) curve: Proposition verification (§4)
5. **E4** — CRB: energy resolution bound (§3)
6. **E12** — Real data completeness: amplitude histogram missing (§7)
7. **E7** — Template mismatch bias: limitation section (§3)
8. **E8** — Time-shift OF: extension verification (§3)
9. **E9** — Convergence plot: §6 theorem support
10. **E5** — σ_E scaling: both simulation and real data (§3)
11. **E10/E11** — Non-stationarity / artifacts: robustness support for §6 limitations
