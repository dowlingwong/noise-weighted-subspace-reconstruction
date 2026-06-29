# Archived: MLST Review & ArXiv Revision Handoff (pre-consolidation)

> **Status: archived provenance.** Historical review/gap analysis and an
> agent-handoff brief. The handoff references the old top-level
> `EXPERIMENT_PLAN.md` (now a redirect stub) and absolute local `Downloads/`
> paths that no longer apply. Superseded by
> [`docs/PAPER_REVISION_GUIDE.md`](../../PAPER_REVISION_GUIDE.md) and
> [`docs/CURRENT_STATUS.md`](../../CURRENT_STATUS.md).
>
> Consolidated from `plan/`: mlst_review_analysis.md,
> arxiv_revision_agent_instructions.md.


---

<!-- source: plan/mlst_review_analysis.md -->

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


---

<!-- source: plan/arxiv_revision_agent_instructions.md -->

# ArXiv Draft Revision Instructions for the Next Agent

This file is the handoff brief for revising the current arXiv draft against
the improved experiment suite.

Canonical roadmap:
- `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/EXPERIMENT_PLAN.md`

Draft reviewed:
- `/Users/wongdowling/Downloads/arXiv_v1_polished_extended_preprint (1).pdf`

Roadmap image reference:
- `/Users/wongdowling/Downloads/experiment_roadmap.svg`

Repository root:
- `/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction`

## Non-negotiable rule

Do not carry old numerical claims from the PDF into the revised paper unless
they are supported by the generated CSV/JSON outputs in `results/`. Several
rank-k and residual-whitening claims in the draft are no longer supported by
the rerun experiments.

## Where the Improved Data and Plots Are Saved

P0 circulation-blocker experiments:
- Driver: `implementation/run_p0_blockers.py`
- Checkpoints: `results/checkpoints/p0/`
- Figures:
  - `results/figures/p0_e1_theorem1.png`
  - `results/figures/p0_e2_bridge.png`
  - `results/figures/p0_e3_chi2_monotone.png`
  - `results/figures/p0_e6_empca_vs_iso.png`
  - `results/figures/p0_e7_mismatch_bias.png`
  - `results/figures/p0_g6_real_rankk.png`

Block 11/12 repair experiments:
- Driver: `implementation/run_block11_12.py`
- Checkpoints: `results/checkpoints/b1112/`
- Figures:
  - `results/figures/block_11_e4_crb_units.png`
  - `results/figures/block_11_e5_sigma_scaling.png`
  - `results/figures/block_11_fig7_sigmaE_vs_rank.png`
  - `results/figures/block_11_fig15_amp_bias.png`
  - `results/figures/block_11_fig16_ks_pvalues.png`
  - `results/figures/block_11_fig16_resid_ratio.png`
  - `results/figures/block_11_fig16_resid_ratio_log.png`
  - `results/figures/block_12_g1_metric_reversal.png`
  - `results/figures/block_12_e12_amp_histogram.png`
- Tables:
  - `results/tables/block_11_e4_crb_units.csv`
  - `results/tables/block_11_e5_sigma_scaling.csv`
  - `results/tables/block_11_expE_rank_agg.csv`
  - `results/tables/block_11_expE_rank_all.csv`
  - `results/tables/block_11_expE_rank_seeds.csv`
  - `results/tables/block_11_table47_reconciliation_agg.csv`
  - `results/tables/block_11_table47_reconciliation_seeds.csv`
  - `results/tables/block_12_g1_real_reversal.csv`

P1 paper-strengthening experiments:
- Driver: `implementation/run_p1_experiments.py`
- Checkpoints: `results/checkpoints/p1/`
- Figures:
  - `results/figures/p1_e8_time_shift_of.png`
  - `results/figures/p1_e9_convergence.png`
  - `results/figures/p1_g3_multichannel.png`
  - `results/figures/p1_g4_covariance_robustness.png`
  - `results/figures/p1_e10_nonstationary.png`
  - `results/figures/p1_e11_artifacts.png`
- Tables:
  - `results/tables/p1_e8.csv`
  - `results/tables/p1_e9_convergence_summary.csv`
  - `results/tables/p1_e9_convergence_traces.csv`
  - `results/tables/p1_g3_multichannel.csv`
  - `results/tables/p1_g4.csv`
  - `results/tables/p1_e10.csv`
  - `results/tables/p1_e11.csv`

Real PSD status:
- Current stored data: `results/checkpoints/b1112/real_psd.npz`
- No standalone `results/figures/*real_psd*.png` was present at this handoff.
  If the paper needs a real-noise PSD figure, generate it from the checkpoint
  or update `implementation/run_block11_12.py` to emit a figure.

## Rewrite Strategy

The revised paper should be theory-first and proof-consistent:

1. State the weighted Gaussian ML problem.
2. Prove that rank-1 weighted subspace reconstruction is the optimal-filter
   estimator in the correct metric.
3. Prove the tied linear autoencoder / weighted EMPCA bridge.
4. Use experiments as theorem checks, unit-convention checks, and boundary
   tests. Do not use experiments to make stronger claims than the data show.
5. Present rank-k results as useful diagnostics and practical extensions, not
   as universal debiasing or universal residual-whitening guarantees.

This produces a stronger proof narrative because the central theorem chain
does not depend on the rank-k bias claims. The rank-k section becomes an
honest empirical boundary study rather than a fragile headline claim.

## Claims to Delete or Rewrite

Delete or weaken these draft claims:

- "Rank-2 EMPCA improves resolution and saturates" as a general claim.
- "PC2 reduces amplitude bias by more than 30% for all noise types."
- "KS p-values exceed 0.05 at k=2" or "residuals are white at true rank."
- "The real-data metric-reversal experiment is deferred to future work."
- The Brownian CRB/result mismatch that implies about 184x degradation while
  nearby text claims agreement within 2%.
- Any statement that mixes PSD conventions or time-domain/rFFT objectives
  without saying exactly which metric is being evaluated.
- Any standalone "isotropic PCA is 87 degrees away" statement unless it is
  tied to the exact old comparison class and not confused with the fair P0 E6
  rFFT comparison.

Replace them with supported claims:

- Rank-1 OF and rank-1 weighted EMPCA agree to numerical precision under the
  matched objective.
- Weighted subspaces differ meaningfully from isotropic subspaces under
  colored noise, and this causes a metric-dependent comparison.
- The fair P0 E6 rFFT control shows a small white-noise gap and larger colored
  noise gaps: about 0.18% white, 3.33% pink, and 7.62% Brownian.
- Real K-alpha data show a small but consistent weighted-residual advantage;
  energy-resolution changes are limited by intrinsic line broadening and
  should not be oversold.
- E7 shows fixed-template OF is biased under template mismatch, while rank-1
  EMPCA adapts toward the mean shape. In the current rerun, OF bias is about
  -25.5%, rank-1 EMPCA bias is about -10.9%, and the OF amplitude tracks the
  weighted cosine with Pearson r about 0.9996.
- P1 shows the framework is practical under time shifts, finite PSD estimates,
  multichannel covariance, nonstationarity, and artifacts, with quantified
  limits and mitigations.

## Section-by-Section Revision Instructions

### Abstract

Keep the central message: noise-weighted subspace reconstruction unifies OF,
EMPCA, and weighted linear autoencoders under one ML objective.

Revise the empirical claim to say that controlled simulations and K-alpha
data verify the equivalence, metric dependence, convergence, and practical
robustness. Avoid saying rank-k universally whitens residuals or removes
amplitude bias.

### Introduction and Contributions

Recommended contribution list:

1. A weighted Gaussian ML objective for pulse reconstruction under colored
   noise.
2. The exact rank-1 equivalence between OF and weighted subspace ML.
3. The bridge between tied linear autoencoders and weighted EMPCA.
4. Metric-reversal demonstrations showing why raw MSE can select the wrong
   subspace under colored noise.
5. Practical experiments covering timing, convergence, finite covariance
   estimation, multichannel correlated noise, nonstationarity, artifacts, and
   real K-alpha data.

### Theory Sections

No major conceptual rewrite is needed if the derivations already use the
noise metric consistently. Add a short convention box or paragraph:

- Define the native time-domain covariance convention.
- Define the rFFT diagonal PSD convention.
- State Parseval/normalization factors used by the code.
- Explain that older mixed-convention comparisons were repaired in the rerun
  CRB/unit checks.

Use `results/tables/block_11_e4_crb_units.csv` and
`results/tables/block_11_e5_sigma_scaling.csv` to support the convention
repair.

### Experiments A and B

Use P0 figures:
- `results/figures/p0_e1_theorem1.png`
- `results/figures/p0_e2_bridge.png`

The text should say these are theorem-verification experiments. Their job is
to verify equivalence under controlled conventions, not to claim new detector
performance.

### Metric Reversal Section

Use two layers:

1. Fair simulated control:
   - Figure: `results/figures/p0_e6_empca_vs_iso.png`
   - Claim: white-noise gap is near zero, colored-noise gap grows under pink
     and Brownian spectra.

2. Real K-alpha check:
   - Figure: `results/figures/block_12_g1_metric_reversal.png`
   - Table: `results/tables/block_12_g1_real_reversal.csv`
   - Claim: weighted residual improves consistently, while sigma_E is nearly
     flat because intrinsic broadening and real-data effects dominate this
     small sample. Do not claim a large real-data resolution gain.

### CRB and Unit-Convention Section

Use:
- `results/figures/block_11_e4_crb_units.png`
- `results/figures/block_11_e5_sigma_scaling.png`
- `results/tables/block_11_e4_crb_units.csv`
- `results/tables/block_11_e5_sigma_scaling.csv`

Rewrite the old Brownian contradiction as a repaired unit-convention result:
the matched convention attains the predicted CRB, while deliberately mixed
conventions fail by a huge factor. This is a good paper-strengthening point
because it makes the implementation audit explicit.

### Rank-k and Mismatch Section

Replace the old rank-k claims with three honest results:

1. EM objective monotonicity:
   - `results/figures/p0_e3_chi2_monotone.png`
   - `results/figures/p1_e9_convergence.png`
   - Random initialization reaches tolerance within 3, 5, and 8 iterations
     for k=1,2,3 in the P1 run; SVD initialization is essentially immediate.

2. Template mismatch:
   - `results/figures/p0_e7_mismatch_bias.png`
   - OF bias is predicted by the weighted cosine with the mismatched template.
   - Rank-1 EMPCA adapts toward the mean pulse shape and reduces, but does not
     eliminate, the bias.

3. Real rank-k behavior:
   - `results/figures/p0_g6_real_rankk.png`
   - Keep the interpretation cautious. Use it to show diagnostics and real
     behavior, not a universal rank-k theorem.

Do not reuse old text claiming that k=2 makes KS p-values pass. Current rerun
tables do not support that.

### New P1 Practical Robustness Appendix

Add a compact appendix or subsection using the P1 figures:

- Time-shift OF:
  - Figure: `results/figures/p1_e8_time_shift_of.png`
  - Table: `results/tables/p1_e8.csv`
  - Claim: joint amplitude/time fitting removes the fixed-template bias;
    t0 RMSE is about 0.29 samples, shifted amplitude bias about +0.02%, and
    fixed-template OF bias about -5.37%.

- Convergence:
  - Figure: `results/figures/p1_e9_convergence.png`
  - Tables: `results/tables/p1_e9_convergence_summary.csv`,
    `results/tables/p1_e9_convergence_traces.csv`
  - Claim: EM objective is monotone for all tested k/init combinations.

- Multichannel covariance:
  - Figure: `results/figures/p1_g3_multichannel.png`
  - Table: `results/tables/p1_g3_multichannel.csv`
  - Claim: full-covariance joint OF improves sigma by about 35.5% relative
    to a diagonal/naive combination, and rank-1 whitened EMPCA correlates with
    joint OF at about 0.999692.

- Finite covariance estimation:
  - Figure: `results/figures/p1_g4_covariance_robustness.png`
  - Table: `results/tables/p1_g4.csv`
  - Claim: finite-PSD degradation is about 1.5-1.7% at N_noise=20-50 and
    approaches oracle performance by N_noise=1000.

- Nonstationary noise:
  - Figure: `results/figures/p1_e10_nonstationary.png`
  - Table: `results/tables/p1_e10.csv`
  - Claim: segment-wise PSD improves RMSE by about 7.8% over one global PSD.

- Artifacts:
  - Figure: `results/figures/p1_e11_artifacts.png`
  - Table: `results/tables/p1_e11.csv`
  - Claim: residual flagging improves RMSE by about 26.8% while retaining
    about 80% of traces.

### Appendix G / Real Data

The PDF currently has text implying real metric reversal is deferred. Remove
that. The real K-alpha comparison exists now:

- `results/figures/block_12_g1_metric_reversal.png`
- `results/tables/block_12_g1_real_reversal.csv`

If keeping a real PSD audit figure, generate one from:

- `results/checkpoints/b1112/real_psd.npz`

Do not show a simulated placeholder as if it were the real K-alpha PSD.

## Replacement Figure Map

Use this map when editing the LaTeX source:

- Old OF/equivalence figure/table: replace or supplement with
  `results/figures/p0_e1_theorem1.png`.
- Old AE/EMPCA bridge figure/table: replace or supplement with
  `results/figures/p0_e2_bridge.png`.
- Old Fig. 7 CRB/rank plot: replace with
  `results/figures/block_11_e4_crb_units.png`,
  `results/figures/block_11_e5_sigma_scaling.png`, and optionally
  `results/figures/block_11_fig7_sigmaE_vs_rank.png`.
- Old Fig. 15 bias figure: replace the claims with
  `results/figures/p0_e7_mismatch_bias.png` plus the corrected block table
  `results/tables/block_11_expE_rank_agg.csv`.
- Old Fig. 16 KS/whiteness figure: either remove or reframe as a diagnostic
  using `results/figures/block_11_fig16_ks_pvalues.png` and
  `results/figures/block_11_fig16_resid_ratio.png`. Do not claim passing KS.
- Old real-data-deferred metric-reversal text: replace with
  `results/figures/block_12_g1_metric_reversal.png`.
- New robustness appendix: add all `results/figures/p1_*.png`.

## Tables to Update

Use these CSV files as the source of truth:

- CRB/units: `results/tables/block_11_e4_crb_units.csv`
- Sigma scaling: `results/tables/block_11_e5_sigma_scaling.csv`
- Rank-k aggregate: `results/tables/block_11_expE_rank_agg.csv`
- Real metric reversal: `results/tables/block_12_g1_real_reversal.csv`
- Time-shift OF: `results/tables/p1_e8.csv`
- Convergence summary: `results/tables/p1_e9_convergence_summary.csv`
- Multichannel: `results/tables/p1_g3_multichannel.csv`
- Covariance robustness: `results/tables/p1_g4.csv`
- Nonstationarity: `results/tables/p1_e10.csv`
- Artifacts: `results/tables/p1_e11.csv`

## Validation Commands

Run these before finalizing the revised paper:

```bash
pytest -q
python implementation/run_p0_blockers.py --figs-only
python implementation/run_p1_experiments.py --figs-only
python implementation/run_block11_12.py --figs-only
```

If a driver lacks `--figs-only` or the option has changed, inspect the script
and use the equivalent checkpoint-preserving mode. Do not overwrite working
checkpoints unless intentionally rerunning a corrected experiment.

## Suggested Revised Paper Shape

1. Introduction: colored-noise ML objective and why raw MSE is insufficient.
2. Theory:
   - Weighted Gaussian objective.
   - Rank-1 OF equivalence.
   - Weighted EMPCA and tied linear AE bridge.
   - Convention box for PSD/covariance units.
3. Controlled theorem checks:
   - P0 E1 and E2.
   - CRB/unit audit.
4. Metric matters:
   - Fair simulated metric reversal.
   - Real K-alpha weighted-residual comparison.
5. Practical extensions:
   - Time-shift OF.
   - Multichannel full covariance.
   - Finite PSD/covariance estimation.
6. Failure modes and mitigations:
   - Nonstationarity.
   - Artifacts.
   - Rank-k mismatch diagnostics.
7. Discussion:
   - The ML/equivalence theorem is strong.
   - Rank-k empirical behavior is useful but not universal.
   - Real-data energy resolution can be dominated by intrinsic broadening,
     so weighted residual is sometimes the cleaner diagnostic.

## Current Bottom Line

The paper can be made stronger by narrowing the headline proof to what is
mathematically exact and using the new experiments as consistency checks and
practical stress tests. The major correction is to stop treating rank-k
EMPCA as a universal debiasing or whitening guarantee. The strongest supported
story is:

Weighted ML is the right objective under colored Gaussian noise; OF is its
rank-1 solution; weighted EMPCA and tied linear AEs solve the same subspace
problem; and the improved experiments show where this objective matters,
where it is robust, and where real detector effects limit observable gains.
