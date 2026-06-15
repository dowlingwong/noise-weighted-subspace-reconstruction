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
