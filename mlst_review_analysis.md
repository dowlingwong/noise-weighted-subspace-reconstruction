# MLST Submission: Experiment Review & Gap Analysis

**Paper:** *A Unified Maximum-Likelihood Framework for Signal Reconstruction:
From Optimal Filtering to Noise-Aware Linear Autoencoders*

---

## 1. Current PC Verification — What Exists

The paper-ready notebook (`archive/PC_interpretation/clean_empca_kalpha_frequency_training_paperready.ipynb`)
currently verifies the physical meaning of the first three learned PCs:

| PC | Physical interpretation | Verification method |
|---|---|---|
| **PC1** | Amplitude / mean signal direction | Weighted cosine to mean, template, T(f)/J(f)-like direction; scatter a₁ vs A_OF |
| **PC2** | Timing-jitter mode (derivative) | Cosine to s′(t); Pearson/Spearman of a₂ vs per-event timing proxy |
| **PC3** | Width/shape deformation mode | Cosine to t·s′(t) proxy; Pearson/Spearman of a₃ vs per-event width proxy |

Additional ablations already in place:
- Centered vs. uncentered comparison for PC1
- Synthetic toy dataset with controlled A/δt/width variation (known ground truth)
- Explained weighted variance curve vs. k

---

## 2. Gap Analysis for MLST

### 2.1 Noise-type dependence of PCs — most critical gap

The PCs are currently trained on one noise type (pink/MMC). MLST reviewers
will immediately ask: does PC2 still look like the derivative under white noise?
Under brownian/MMC? If the interpretation is a property of the *method* it must
hold for arbitrary noise spectra; if it is noise-specific, that must be stated.

**Required:** cosine scores for all three PCs across all three noise types,
presented as a 3 × 3 matrix or subplot grid.

### 2.2 Bootstrap confidence intervals on cosine scores

All cosine scores are currently single-run point estimates. An event-level
bootstrap (resample training set, recompute PCs, record cosine distribution)
converts each number into a mean ± CI. This makes the interpretation claims
statistically defensible.

### 2.3 No physical-consequence test for PC2/PC3

The paper shows PC2 correlates with timing, but never shows that *keeping PC2
in the reconstruction actually reduces timing bias in recovered amplitude*. The
cleanest test: generate intentionally time-shifted events, estimate amplitude at
rank 1 vs rank 2, quantify the bias reduction. This closes the loop from
"PC2 captures timing" to "and here's why it matters for energy resolution."

### 2.4 Residual noise-floor test as a function of k

The E1 experiment checks KS p-value on residuals at rank 1. A full paper needs
this as a function of k: after projecting onto k PCs, does the weighted residual
per event follow χ²(N_freq − k)? A QQ-plot or KS test for each k and noise type
is standard goodness-of-fit practice and MLST reviewers will expect it.

### 2.5 No spectral bridge between theory and data

The entire framework rests on the weighted metric W(f) = 1/J(f) and on
Theorem 1 (rank-1 EMPCA ≡ optimal filter). The spectral SNR density
|T(f)|²/J(f) is the physical object that determines both the CRB (through
N_Φ = Σ_f |T(f)|²/J(f)) and the shape of PC1 in the frequency domain. Without
a plot showing these quantities together, the reader must accept Theorem 1 on
faith. This is the single most important missing figure for a physics-ML journal.

### 2.6 Sample complexity

How many training traces are needed for the PCs to converge to their asymptotic
directions? This is directly relevant to MLST's audience (experimentalists who
want to know how much calibration data they need) and is straightforward to
produce from the existing simulation infrastructure.

---

## 3. Priority Experiments to Add

### P1 — σ_E vs k for pink, white, and brownian noise (planned)

**What it shows:** Energy resolution as a function of EMPCA rank, compared to
the OF CRB baseline (dashed), for three noise types. At k = 1, all curves must
touch the dashed line (visual proof of Theorem 1). For rank-1 signal families,
curves plateau immediately. For families with timing/shape variation, there may
be a measurable improvement at k = 2.

**Implementation notes:**
- Use `simulate_controlled_family` with `t0_jitter_range = (-1e5, 1e5)` to
  introduce timing jitter so the k = 2 effect is visible.
- For each noise type, normalize σ_E by the OF CRB (1/√N_Φ) so that the y-axis
  is dimensionless and all noise types can be compared on the same panel, or
  show absolute σ_E with per-noise-type CRB dashed lines.
- Error bands = std across 8 seeds.
- k range: 1 to `empirical_rank_max` (currently 8).

**Pass criterion:** σ_E(k=1) / CRB ∈ [0.98, 1.02] for all three noise types.

### P2 — Signal-to-noise spectral density |T(f)|²/J(f) with PC1 overlay

**What it shows:** For each noise type, plots |T(f)|² (signal power), J(f)
(noise PSD), |T(f)|²/J(f) (SNR density), and |PC1(f)|² (learned rank-1
direction) on the same axes. The near-proportionality of |PC1(f)|² to
|T(f)|²/J(f) is a visual proof of Theorem 1. The shift of the SNR peak across
noise types explains why EMPCA automatically adapts to the noise.

**Implementation notes:**
- Compute for noise types: pink, white, brownian.
- Overlay |PC1(f)|² normalized to match the scale of |T(f)|²/J(f).
- Annotate N_Φ = Σ_f |T(f)|²/J(f) and the resulting σ_E = 1/√N_Φ for each.
- A 3-panel figure (one per noise type) or a single overlaid plot with legend.

**Why essential:** Without this figure, Theorem 1 is algebra. With it, it is
a picture. MLST specifically values this kind of physical intuition building.

### P3 — 3 × 3 PC cosine score matrix (PC × noise type)

**What it shows:** A heatmap or table where rows are PC1/PC2/PC3 and columns
are pink/white/brownian noise. Each cell contains the weighted cosine similarity
between PCk and its reference direction (mean direction for PC1, derivative for
PC2, width proxy for PC3). Demonstrates that the interpretation is not an
artifact of a specific noise spectrum.

**Implementation notes:**
- For each noise type, train rank-3 EMPCA on `simulate_controlled_family`.
- Reference directions: (1) normalized mean frequency vector, (2) s′(t) → freq
  domain, (3) width proxy t·s′(t) → freq domain.
- Report cosine and bootstrap CI for each cell (50 bootstrap resamples is
  sufficient).

**Pass criterion:** |cosine(PC1, mean)| > 0.95 for all noise types;
|cosine(PC2, deriv)| > 0.80.

### P4 — Timing-bias reduction: rank-1 vs rank-2 amplitude estimation

**What it shows:** With intentional timing jitter, rank-1 amplitude estimation
is biased (PC1 is not exactly orthogonal to the timing perturbation under
jitter). Rank-2 projection corrects this. Plot the distribution of recovered
amplitudes at rank 1 and rank 2 for a fixed true amplitude, and compare the
residual bias and std.

**Implementation notes:**
- Use `simulate_controlled_family` with `t0_jitter_range = (-4e5, 4e5)` (large
  jitter) and `amplitude_true = const` (all events same amplitude).
- At rank 1: amplitude estimate = GLS coefficient for PC1 × normalization scale.
- At rank 2: apply a linear predictor (least-squares fit from [a₁, a₂] to
  true amplitude on training set).
- Metrics: mean bias, RMSE, and paired KS test comparing rank-1 vs rank-2
  amplitude distributions.

**Pass criterion:** mean bias at rank 2 < mean bias at rank 1 by > 30%.

### P5 — Residual χ² distribution as a function of k

**What it shows:** After projecting events onto k PCs, the weighted residual
energy per event should follow χ²(N_freq − k) if the model is correct (noise is
Gaussian with known PSD). A QQ-plot grid (one panel per k = 1, 2, 3, 4) with
a KS test p-value annotation demonstrates goodness-of-fit and shows whether
additional PCs are still capturing signal or fitting noise.

**Implementation notes:**
- Use the simulated test split (not training) for residuals.
- The normalized residual is `resid_i / mean(resid)`, which should follow
  χ²(1) / dof under the null.
- Threshold: KS p > 0.05 at the optimal k; < 0.05 for k beyond rank saturation
  (noise-fitting regime).
- Do this for all three noise types.

### P6 — Bootstrap confidence intervals on cosine scores

**What it shows:** Converts the point-estimate cosine scores for PC1/PC2/PC3
into bootstrap distributions, providing 95% CIs that quantify the stability of
the interpretation under different training samples.

**Implementation notes:**
- 200 bootstrap resamples of the training set (with replacement).
- For each resample, refit rank-3 EMPCA, compute cosines to reference directions.
- Report median ± 95% CI (2.5th and 97.5th percentile).
- At N_train = 1000 this runs in a few minutes; at 300 it is fast enough for
  interactive use.

**Expected result:** All CIs narrow (width < 0.05) confirming stable
interpretation; PC3 CI may be wider than PC1/PC2.

---

## 4. Template / Noise Spectrum Analysis

### Why to include it

The connection between the noise PSD J(f), the signal template T(f), and the
energy resolution is the physical heart of the paper. The quantity
N_Φ = Σ_f |T(f)|²/J(f) is the matched-filter SNR² and determines the CRB.
Different noise types (pink, white, brownian/MMC) shift the weight of this sum
to different frequency regions:

- **White noise** (J(f) = const): SNR density ∝ |T(f)|². Energy concentrated
  where the pulse has power.
- **Pink noise** (J(f) ∝ 1/f): SNR density ∝ f·|T(f)|². Up-weights higher
  frequencies relative to white, but signal power still dominates at low f.
- **Brownian/MMC noise** (J(f) ∝ 1/f²): SNR density ∝ f²·|T(f)|². Pushes
  useful SNR further into mid-to-high frequencies. This regime most closely
  matches real MMC detectors.

### What to show

A three-panel figure (one column per noise type) each containing:

1. **Top sub-panel:** J(f) (noise PSD) and |T(f)|² (signal spectrum) on log-log axes.
2. **Bottom sub-panel:** SNR density |T(f)|²/J(f) with |PC1(f)|² overlaid
   (normalized to match scale). Annotate N_Φ and σ_E = 1/√N_Φ.

### Impact on energy resolution

Add a companion figure or panel: N_Φ vs noise type (bar chart), alongside the
empirically measured σ_E(k=1). The ratio σ_E_emp / (1/√N_Φ) should be in
[0.98, 1.02] for all noise types. This is Theorem 1 stated visually.

### Why MLST will specifically look for this

MLST (IOP) consistently asks authors to connect mathematical results to physical
observables. The spectral analysis figure does exactly that: it takes the
abstract claim "weighted PCA in 1/PSD metric converges to the optimal filter"
and turns it into a picture that any physicist can read. Without it, the paper
risks being seen as purely algebraic.

---

## 5. PC3 Caveat

The width/shape proxy interpretation is the weakest of the three. The proxy
t·s′(t) is physically motivated but not unique — there are other shape
deformation modes with similar spectral content, and the cosine scores for PC3
tend to be lower (typically 0.7–0.9 vs. > 0.95 for PC1). The paper should
hedge accordingly: "PC3 is consistent with a width/shape deformation mode" rather
than "PC3 represents the width mode." Presenting the bootstrap CI (P6) is
especially important here to quantify the uncertainty.

---

## 6. Summary Table

| Priority | Experiment | Paper section | Status |
|---|---|---|---|
| P1 | σ_E vs k, three noise types, with OF baseline | §7 / main results | **To implement** |
| P2 | SNR spectral density + PC1 overlay | §4.3 / Theorem 1 visual | **To implement** |
| P3 | 3 × 3 PC cosine score matrix across noise types | §4 / PC interpretation | **To implement** |
| P4 | Timing-bias reduction rank-1 vs rank-2 | §4 / physical consequence | **To implement** |
| P5 | Residual χ² distribution vs k | §4.4 / rank saturation | **To implement** |
| P6 | Bootstrap CIs on cosine scores | §4 / PC interpretation | **To implement** |
| — | σ_E scaling law (E5) | §3.2 | ✓ Done |
| — | Rank saturation χ²(k) monotone (E3) | §4.4 | ✓ Done |
| — | CRB verification (E4) | §3.2 | ✓ Done |
| — | Noise-aware vs isotropic ablation (E6) | §7 | ✓ Done |
| — | Template mismatch bias (E7) | §3.4 | ✓ Done |
| — | Theorem 2: AE ≡ EMPCA (E2) | §5.3 | ✓ Done |
| — | Toy synthetic validation (PC notebook) | §4 | ✓ Done |
| — | Centered vs uncentered ablation (PC notebook) | §4 | ✓ Done |
