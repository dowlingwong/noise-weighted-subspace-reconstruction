<!-- GENERATED COPY - DO NOT EDIT.
     Source: docs/CURRENT_STATUS.md @ 1a268459e52b4e22d9235cb141ce2ebd9110d579
     Generated: 2026-07-26T17:30:29Z by scripts/sync_source_documents.py
     Edit the source in the experiment repository and re-run the sync. -->

# Current Status

Last consolidated: 2026-07-04.

## Central Claim

For Gaussian detector noise, the statistically aligned reconstruction geometry
is set by the inverse noise covariance:

```text
(x - x_hat)^H Sigma^{-1} (x - x_hat)
```

Under this metric, fixed-template optimal filtering, rank-k EMPCA, and tied
linear autoencoders with a noise-weighted loss are constrained maximum-
likelihood projections. Ordinary MSE is the corresponding likelihood only for
white isotropic noise.

Paper 1 is scoped to linear reconstruction, optimal filtering, weighted
subspaces, covariance/PSD estimation, and residual calibration. Nonlinear
architectures and Paper 2 material are outside the Paper 1 validation claim
set. Paper 2's neural-architecture and coverage results are exploratory
until their separate implementation defects (uncontrolled RNG seeding,
repeated test-set evaluation during training, an invalid timing-coordinate
metric, and a Transformer with neither positional encoding nor cross-channel
attention) are repaired and rerun from a frozen protocol; see
`docs/EVALUATION.md`.

## Noise-Module Scientific Boundary

The stationary Gaussian generator's discrete convention (interior
circular-complex-Gaussian rFFT bins, real endpoint DC/Nyquist bins,
periodic/circulant FFT-grid covariance vs. a separately named exact-Toeplitz
sampler) is pinned in `noise_module/noise_module/PAPER_CONVENTION.md` and
`paper_convention.json`, and supersedes the fixed-modulus/random-phase
sampler previously documented as Paper 1 Equation 81. A total PSD or
covariance can be diagonalized, whitened, or decorrelated into statistically
orthogonal modes; those modes are not, by themselves, uniquely identifiable
physical sources (rotations of latent factors can leave a low-rank total
covariance unchanged). Physical source attribution requires additional
structure -- known transfer functions, auxiliary sensors, or controlled
interventions -- beyond what Study A/B's declared PSD and acquisition
assumptions provide.

## Validation Summary

| Area | Current state | Paper use |
| --- | --- | --- |
| S1-S9 controlled synthetic gates | verified positive | Results and Methods |
| S5 confirmatory sweep (30 seeds, held-out) | verified positive, strengthens the frozen 10-seed result | Results; use only the `ratio` JSON block for any "x times" wording |
| Stage 0 remote reproducibility | verified positive | Methods/reproducibility |
| GWOSC data integrity and official coverage | verified positive | Methods |
| GWOSC PSD comparison with GWpy | verified positive | Methods/diagnostic Results |
| GWOSC shared-FIR implementation identity | verified positive | Results/diagnostic Methods |
| GWOSC global-PSD null calibration | verified negative | Results, Discussion, Limitations |
| GWOSC time-local PSD calibration | verified negative | Results, Discussion, Limitations |
| GW150914 event significance | not validated | Diagnostics only, if mentioned |
| SNR-eight injection sensitivity on GWOSC | not validated | Diagnostics only, if mentioned |
| CRESST-A public-data validation (linear hierarchy) | verified positive | Results and Methods |
| CRESST trigger-efficiency detector metric | inconclusive | Limitations, if mentioned |
| Study A noise-budget band decomposition | verified positive (rebuilt from authoritative component files, see EXP-02) | Results/Appendix |
| Study B sensitivity attribution | verified positive, conditional on declared PSD/acquisition assumptions (see EXP-03) | Results/Appendix, with acquisition branch stated in every caption |

## What Is Supported

- The implementation reproduces the intended linear likelihood geometry in
  controlled synthetic settings.
- The OF/CRB, OF/EMPCA, EMPCA/linear-AE, white-noise control, metric reversal,
  timing-rank, covariance convergence, residual calibration, and multichannel
  covariance synthetic gates have archived positive evidence.
- The remote Stage 0 workflow can run from a clean Linux checkout with archived
  command logs and environment records.
- The GWOSC PSD estimator agrees with GWpy on identical calibration windows.
- The shared-FIR follow-up shows that explicit FFT convolution and GWpy
  convolution agree when both paths use identical FIR coefficients and score
  normalization.
- CRESST-A (public DMDC CRESST-II/III pulse-shape release, test split): EMPCA
  at the predeclared rank (k=6) beat single-template OF and was competitive
  with multi-template OF, and all 5 predeclared negative controls degraded,
  by a pooled paired-bootstrap 95% CI across all 7 detector channels (within
  each channel, the pulse and noise streams were each split chronologically,
  separately, with zero row/hash overlap). See `docs/results/CRESST_RESULT.md`.
- Study A (`results/noise_budget/<run_id>/study_a/`) reproduces the checked-in
  component band fractions and crossovers directly from
  `noise_module/Al2O3_Al_athermal/{SQUID,Johnson,TD,Er,total}_noise.dat`
  under their native labels: SQUID dominates every analyzed band (median
  fraction 0.867-0.967), with no SQUID/Johnson, SQUID/TD, or SQUID/Er
  crossing; Johnson-Er crosses near 319.67 Hz and Johnson-TD near 688.58 Hz.
  Do not use any prior "1/f phonon" / "vibrational" / "TES Johnson" labeling.
- Study B (`results/noise_budget/<run_id>/study_b/`) computes conditional OF
  sensitivity attribution (leave-one-out and 0.5x/0.2x scaling per component)
  under two explicit, non-exhaustive acquisition branches (`inband_no_alias`,
  `unfiltered_alias_fold`); cross-checked against an independent OF-variance
  implementation to <0.02% fractional disagreement on relative sensitivity.
  This is sensitivity attribution under declared assumptions, not unique
  physical source decomposition and not a measured detector improvement.
- S5 confirmatory sweep (`results/sweeps_v2/<run_id>/S5/`, 30 independent
  held-out seeds 1000-1029, disjoint from the frozen 0-9 block): weighted-
  residual PCA/EMPCA ratio `1.0115` (95% CI `[1.0111, 1.0119]`), raw-residual
  PCA/EMPCA ratio `0.3150` (95% CI `[0.3088, 0.3210]`) -- the metric-reversal
  effect confirmed with an independent, larger sample. The predeclared
  smallest-relevant-effect check (>=0.5% weighted-residual improvement)
  passed with an observed `1.14%` improvement. A matched S4 white-noise
  sweep under the identical held-out protocol remained a negative control.
  The frozen 10-seed S5 result is preserved unchanged; this is additive,
  independent confirmation, not a replacement. **Any manuscript "x times"
  wording for S5 must be read from the JSON's `ratio` block only** (see
  `scripts/run_s5_confirmatory_sweep.py`) -- never from a raw difference or
  percentage.
- `results/audits/rerun_decisions.json` (EXP-05) mechanically applies the
  Gaussian-bin/fixed-modulus/unknown-lineage/deterministic-arithmetic/
  public-data decision rule to every EXP-00-audited artifact:
  `no_fixed_modulus_artifacts_found: true` -- nothing in this repository's
  current synthetic evidence used the obsolete sampler, so no distributional
  claim required a lineage-driven rerun beyond EXP-04's new S5 protocol.
- `results/audits/generator_distributional_diagnostics.json` adds a
  periodogram-variability distributional check (KS/QQ vs. Exponential(1),
  mean KS statistic `0.0025` across 5 seeds) for the Fourier generator,
  going beyond the mean-PSD-only checks that could not have distinguished a
  Gaussian sampler from the obsolete fixed-modulus one.
- `results/audits/appendix_f_traceability.csv` (EXP-05) maps every named
  legacy Appendix F item to a machine-readable source or marks it
  unsupported: CRB verification, rank sweep, and covariance-estimation
  convergence are confirmed and match the audit doc's numbers; principal-
  component alignment/bootstrap and the time-shift diagnostic are only
  partially mapped; resolution-scaling slope, template mismatch, timing-bias
  reduction, the rank-swept KS/QQ transition, and the duplicate
  `1029.2/1028.9/1028.6` rank values have **no machine-readable source
  anywhere in this repository** and must not be cited until either their
  true source is located or a new script computes and archives them.

## What Is Not Supported

- The current evidence does not support calibrated GW150914 significance.
- The current evidence does not support validated nominal SNR-eight injection
  sensitivity on GWOSC.
- The current evidence does not support a general claim that the method is
  calibrated on GWOSC data.
- The current evidence does not show that a 64-second local PSD model fixes the
  GWOSC real-noise calibration failure.
- The shared-FIR result does not prove that the original PSD-domain GLS
  statistic and a finite-duration FIR statistic are mathematically equivalent.
- The CRESST-A result does not support a cross-channel/cross-detector
  transfer claim; the design is explicitly per-channel.
- The CRESST-A trigger-efficiency detector metric (versus injected amplitude
  at fixed background FPR) was inconclusive within the predeclared amplitude
  grid for most channels and does not support a validated detection-
  sensitivity claim.

## Current GWOSC Conclusion

GWOSC is a completed negative public real-noise stress test for the current
paper revision. It is not a positive validation dataset.

The correct interpretation is:

```text
GWOSC shows the boundary of the present validation: real-noise calibration
requires more than a correct PSD implementation and a predeclared local PSD
model.
```

Keep GWOSC as a limitation and boundary result unless a new predeclared Stage
3b is designed and run.

## Current CRESST Conclusion

CRESST-A is a completed positive public real-data validation of the linear
noise-geometry hierarchy for the current paper revision, with two reported
limitations (see `docs/results/CRESST_RESULT.md`): a single-channel exception in the
weakest-margin negative control, and an inconclusive trigger-efficiency
detector metric at the predeclared amplitude grid.

The correct interpretation is:

```text
CRESST-A shows that, with the pulse and noise streams each split
chronologically (separately) within each channel, the noise-weighted
linear hierarchy (EMPCA vs. single-/multi-template OF) holds with
statistically supported margins. It
does not show cross-channel transfer, and it does not show validated
detection sensitivity from the trigger-efficiency metric as measured.
```

## Next Steps

1. Revise the manuscript so the positive claims rest on theory, controlled
   synthetic validation, and the CRESST-A public real-data result.
2. Include GWOSC as a negative public real-noise stress test, not as calibrated
   detection evidence.
3. ~~Move positive public real-data validation to CRESST or another
   detector-pulse dataset~~ -- done; see `docs/results/CRESST_RESULT.md`. If a wider
   detection-sensitivity claim is wanted, that requires a new predeclared
   stage with a revised amplitude grid (see "If CRESST Work Is Continued" in
   that document).
4. Do not tune the completed GWOSC experiment by changing thresholds, PSD
   radii, windows, quality cuts, or exclusions after seeing the result. The
   same applies to the completed CRESST-A run.
5. If GWOSC is continued, write a new predeclared Stage 3b protocol first.

## Important Evidence Locations

- `transfer_paper/data/derived/claim_status.csv`
- `docs/results/CRESST_RESULT.md`, `docs/results/CRESST_PREDECLARED_GATES.md`
- `transfer_paper/data/derived/cresst_per_channel_summary.csv`,
  `transfer_paper/data/derived/cresst_pooled_summary.csv`
- `evidence/cresst/20260704T013758Z_9c9f696ff6bc/` (full run record:
  `cresst/validation.json`; config: `cresst/validation.config.yaml`;
  gate-only summary: `cresst/acceptance_summary.json`)
- `transfer_paper/data/cresst/current/cresst/p2_cresst_validation_seed22.json`
  (synchronized copy of the same full run record)
- `transfer_paper/data/derived/paper_implications.csv`
- `transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md`
- `transfer_paper/data/gwosc/followup/filter_equivalence.json`
- `transfer_paper/data/gwosc/followup/time_local_noise.json`
- `transfer_paper/data/derived/gwosc_filter_equivalence_summary.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_summary.csv`
- `results/audits/noise_generator_provenance.{json,md}`,
  `results/audits/synthetic_config_provenance.csv` (generator lineage per
  synthetic experiment; see EXP-00)
- `results/audits/rerun_decisions.json` (EXP-05 decision-rule application),
  `results/audits/appendix_f_traceability.csv` (EXP-05 Appendix F audit),
  `results/audits/generator_distributional_diagnostics.{json,png,pdf}`
  (EXP-05 distributional panels)
- `results/sweeps_v2/<run_id>/{S4,S5}/` (EXP-04 confirmatory sweep; frozen
  10-seed `results/sweeps/` and `transfer_paper/data/synthetic/` evidence is
  unchanged and remains primary for S1-S9 generally)
- `results/noise_budget/<run_id>/study_a/`,
  `results/noise_budget/<run_id>/study_b/` (see EXP-02/EXP-03)
- `results/public_data_package/cresst/`, `results/public_data_package/gwosc/`
  (stable packaged tables; see EXP-06)
- `noise_module/noise_module/PAPER_CONVENTION.md`,
  `noise_module/noise_module/paper_convention.json` (see EXP-01)
