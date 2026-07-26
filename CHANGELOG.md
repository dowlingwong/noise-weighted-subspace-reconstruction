# Changelog

## Unreleased

### 2026-07-27 - TIDMAD phases 2-5

- Added `analysis.py` (phases 2-3), `calibration.py` (phase 4), `denoise.py`
  and `score.py` (phase 5) to `src/noise_geometry/tidmad/`, plus
  `scripts/tidmad_phases.py` and four configs. Phases 1-4 run through
  `scripts/run_experiment.py`.
- **Phase 2 (metric reversal against truth) passes in synthetic mode.** Two
  matched-rank subspaces fitted on the same rFFT-real representation, differing
  only in the metric: PCA wins raw MSE (-3216.8, CI [-3750.6, -2740.1]), EMPCA
  wins the weighted residual, and **the injected truth sides with EMPCA**
  (amplitude |error| difference +15.04, CI [14.70, 15.38]; clean-signal MSE
  difference +3514). The third row is the claim CRESST structurally cannot
  produce. A moving-block bootstrap is reported alongside the window-level one
  because consecutive stream windows are dependent.
- **Phase 3 (rank knee) passes.** Amplitude RMSE by rank: 13.456 / 5.792 /
  5.795 / 5.794 for ranks 1-4, against 24.890 for a fixed-phase rank-1 optimal
  filter. Rank 2 beats rank 1 (CI [5.996, 7.570]); rank 3 does not improve on
  rank 2 (CI [-0.001, 0.005]). Reported as the smallest rank within a
  predeclared tolerance of the minimum, because the raw argmin differs by 0.03%
  and quoting it would mislead.
- **Phase 4 (calibration boundary) control passes**: chi^2/dof 1.0002 and score
  dispersion 1.0758 for the stationary synthetic control, inside the
  predeclared [0.9, 1.1] band. The chi^2 normalization is derived in the code
  (`f_s N / 2` per degree of freedom) and unit-tested rather than fitted. The
  local-PSD estimator excludes the window it scores; a test pins that, since
  including it would produce an excellent-looking calibration that is pure
  leakage. The reported dispersion standard error assumes an exact predicted
  sigma and understates the true spread (five-seed check: 0.03 realized vs 0.022
  formula); this is now documented in the code.
- **Phase 5 implemented, not run** -- it has no synthetic form, since the
  denoising score is defined by the release's own `benchmark.py`. Components
  verified: identity-transfer overlap-add reconstructs to 7e-14, the fitted
  Wiener filter cuts MSE-to-clean by 20x, and spectral subtraction is clipped so
  the transfer function cannot exceed one. `score.py` shells out to the upstream
  benchmark and raises if it is absent rather than falling back to an internal
  approximation.
- 13 new tests (22 total for TIDMAD). Evidence archived under
  `evidence/tidmad/phases_2_to_4/`; write-up in
  `docs/results/TIDMAD_PHASES_2_TO_5_RESULT.md`.
- Not yet implemented: Phase 6 negative controls and the frozen
  `docs/TIDMAD_PREDECLARED_GATES.md`. Phase 2's result should not be published
  before those exist.

### 2026-07-27 - TIDMAD stage: structure and Phase 1

- Added `src/noise_geometry/tidmad/` (templates, estimators, injections, psd,
  splits, loader, experiments), replacing the placeholder package. The stage is
  Paper 1's second real-data anchor; rationale and phase plan are in
  `docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`.
- **Phase 1 (estimator identity) executed in synthetic mode: all six gates
  pass.** Amplitude bias z = -0.55; empirical/predicted sigma ratio 0.9913;
  no-signal null z = 0.028; learned rank-2 subspace within 2.32 degrees of the
  analytic quadrature subspace; fixed-phase SNR loss matches cos^2 to 0.022;
  and the subspace-recovery error follows the theoretical 1/SNR law with a
  log-log slope of -1.0016 against the predicted -1. Record:
  `results/metrics/TIDMAD_P1_synthetic_seed20260727.json`; write-up:
  `docs/results/TIDMAD_PHASE1_RESULT.md`.
- The real-noise half of Phase 1 is written and configured
  (`configs/tidmad/phase1_estimator_identity_real_noise.yaml`) but blocked on
  Phase 0: it requires a verified data contract, and the loader refuses to run
  without one by design.
- Added `scripts/tidmad_phase0_contract.py` (channel semantics established from
  spectral evidence, not assumed), `scripts/tidmad_phase1_estimator_identity.py`,
  `scripts/download/download_tidmad.py` (wraps the upstream downloader; refuses
  to write into the repository), and `tests/test_tidmad_estimators.py` (9 tests).
- `scripts/run_experiment.py` now dispatches `dataset: tidmad` to the Phase 1
  gate instead of returning `planned`.

### 2026-07-26 — documentation consolidation

- **Source documents are now generated, not maintained.** `CURRENT_STATUS.md`,
  `EXPERIMENT_PROTOCOLS.md`, and `PAPER_REVISION_GUIDE.md` existed as three
  independently hand-edited copies (`docs/`, this repository's
  `transfer_paper/data/source_documents/`, and the manuscript repository's)
  with three different SHA-256 hashes; the manuscript repository was reading
  the oldest. Added `scripts/sync_source_documents.py`, which renders each copy
  from `docs/` with a provenance header and writes
  `SOURCE_DOCUMENTS_MANIFEST.json`, and a `source_document_drift` check in
  `scripts/verify_paper_bundle.py` that fails on any hand-edited copy.
- **Merged the two same-day audits into `docs/EVALUATION.md`.** The broad
  first-pass audit and the independent second-pass verification overlapped by
  roughly 60% with no indication of which was authoritative. Both originals are
  archived under `docs/archive/2026-07-26_pre_paper23/`. The merged document
  records the second-pass findings: the S5 manuscript numbers come from a
  seeds-0–29 run that violates the confirmatory sweep's own predeclaration, a
  latent argument-count bug in `OptimumFilter.sliding_fit(chisq_mode='none')`,
  the S5 global-scale convention, and numerical verification of the Appendix G
  generator equations.
- **Reorganized `docs/`** into `results/`, `plans/`, `runbooks/`, and
  `background/`, with `docs/README.md` rewritten as a complete index. All
  references in code docstrings, configs, and Markdown were updated; archived
  and checksummed evidence bundles were left untouched.
- **Archived superseded plans** to `docs/archive/2026-07-26_pre_paper23/`:
  `NPML/plan.md`, `paper2/IMPLEMENTATION_SPEC.md`,
  `paper2/MISSING_FOR_D_E.md`, and the two `paper2/plans/` documents.
- **Moved `evolution_plan.md` to `noise_module/EVOLUTION_PLAN.md`** — it is the
  noise-module roadmap and belongs with its subject.
- Added planning documents: `docs/plans/PAPER1_SECOND_REALDATA_ANCHOR.md`
  (second real-data anchor decision; TIDMAD selected, GWOSC demoted),
  `docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`,
  `docs/plans/CORRECTION_PLAN_EXPERIMENTS_2026-07-26.md`,
  `docs/EXPERIMENTS_OVERVIEW.md`,
  `paper2/PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md`, and
  `paper3/PAPER3_PLAN.md`.

### 2026-07-13

- Fixed two issues found while reviewing the CRESST post-freeze channel-
  robustness sensitivity archive
  (`transfer_paper/data/cresst/sensitivity/20260713_channel_robustness_v1/`,
  added 2026-07-13, not by this fix): (1) three files present on the server
  and listed in the archive's own `SHA256SUMS` (`console.log`,
  `event_residuals_lossless.npz`, `hierarchical_bootstrap_draws.npz`) were
  silently excluded from every commit by the repository's global `*.log`/
  `*.npz` ignore rules; added a narrow `.gitignore` exception scoped to
  `transfer_paper/data/cresst/sensitivity/` (all ~8 MB, small derived
  per-event/bootstrap arrays, not raw detector data) and tracked the three
  files. (2) `docs/results/CRESST_PREDECLARED_GATES.md` and downstream
  manuscript-facing text described the per-channel chronological split as a
  single combined cut, but the pulse and noise streams are actually split
  *separately*; the sensitivity archive's leakage audit found 6 and 3
  evaluation pulses (of 19,335 pooled) in channels `35_57.0`/`35_58.0`
  respectively with timestamps earlier than that channel's final
  calibration-noise timestamp. No evaluation pulse entered any fitted model
  and zero within-stream row/hash overlap holds, so Stage C5's result is
  unaffected; corrected the wording (in `CRESST_PREDECLARED_GATES.md`,
  `CRESST_RESULT.md`, `CURRENT_STATUS.md`, and
  `transfer_paper/CRESST_PAPER_UPDATE_PLAN.md`) to "pulse and noise streams
  were each split chronologically, separately, with zero row/hash overlap,"
  and added a future-work recommendation to use one shared time cutoff per
  channel for a strict prospective replication. Documentation-only: the
  frozen split fraction, seed, code, and Stage C5 result are unchanged, and
  nothing was rerun.

### 2026-07-04

- Fixed a real data-format bug: the DMDC-hosted `X_*.npy`/`y_*.npy` release
  files are headerless raw binary dumps, not standard `.npy`; the downloader
  verifier and `load_cresst_release` now tolerate both formats.
- Implemented the CRESST Validation Protocol (Stage C3): a per-channel,
  time-disjoint calibration/evaluation split (`splits.py`), a multi-template
  OF bank (`templates.py`), detector-facing metrics with a paired-bootstrap
  acceptance gate (`detector_metrics.py`), signal injection (direct +
  measured-PSD, `injections.py`), and 5 negative controls
  (`negative_controls.py`). Added a PSD-stationary covariance model
  (`noise.covariance_from_psd`) for the per-channel N ~ p regime and
  three-tier FPR reporting (5% primary, 1% with a threshold CI, 0.1%
  diagnostic). Froze the Stage C4 predeclaration
  (`docs/results/CRESST_PREDECLARED_GATES.md`, commit `9c9f696ff6bc`).
- Ran the predeclared Stage C5 CRESST-A validation once on the full public
  test split: **passed** (EMPCA beat single-template OF, was competitive
  with multi-template OF, and all 5 negative controls degraded, by pooled
  paired-bootstrap 95% CI across all 7 channels), with two reported
  limitations (a single-channel exception in the weakest negative control;
  an inconclusive trigger-efficiency metric at the predeclared amplitude
  grid). See `docs/results/CRESST_RESULT.md`. Archived evidence under
  `evidence/cresst/20260704T013758Z_9c9f696ff6bc/` and synchronized
  `transfer_paper/data/cresst/` and `transfer_paper/data/derived/
  cresst_{per_channel,pooled}_summary.csv` via the new
  `transfer_paper/scripts/refresh_cresst_bundle.py`. Updated
  `docs/CURRENT_STATUS.md` and `transfer_paper/data/derived/claim_status.csv`.

### 2026-06-30

- Prepared the CRESST setup against the public DMDC pulse-shape release
  (arXiv:2508.03078). Rewrote `scripts/download/download_cresst.py` as a
  manifest-driven, base-URL-keyed downloader that verifies every `.npy`
  shape/dtype and CSV row count, writes a checksummed provenance manifest, and
  refuses repo-internal writes. Added `load_cresst_release` / `select_cresst_subsets`
  (uses the `noise`/`clean` flags to separate baselines from accepted pulses),
  a release-aware branch in `run_cresst_experiment`, a cache-building
  `preprocess_cresst.py`, `configs/cresst/cresst_validation.yaml`, an updated
  smoke config and runbook, and `tests/test_cresst_release.py`.

### 2026-06-29

- Consolidated the documentation into one canonical `docs/` set
  (`CURRENT_STATUS`, `EXPERIMENT_PROTOCOLS`, `GWOSC_RESULT`,
  `PAPER_REVISION_GUIDE`, `REMOTE_EXECUTION`, `README`), moved the superseded
  notes under `docs/archive/2026-06-29_pre_consolidation/`, and repointed the
  `README`/`EXPERIMENT_PLAN`/`TODO` stubs at the canonical docs.
- Refreshed the `transfer_paper/` evidence bundle (figures, derived/summary
  CSVs, manifests, and source-document mirror) to consume the consolidated
  documentation.
- Removed stale artifacts: the source-less `src/EMPCA/` cache, stray
  `.DS_Store` files, and a leftover `.git/index.lock`.
- Archived an additional GWOSC validation run.
- Fixed the QP simulator standard-deviation estimation and updated the 1 MHz
  noise-PSD resampling code.
- Distilled the reusable linear methodology from the NPML experiment plan into
  the canonical docs: a predeclared CRESST validation protocol (multi-template
  OF baseline, measured-PSD-injection honesty rule, negative-control battery),
  detector-facing metrics (residual whitening ratio, detection score, trigger
  efficiency at fixed FPR) in `docs/EXPERIMENT_PROTOCOLS.md`, and a GWOSC
  diagnostic-framing section in `docs/results/GWOSC_RESULT.md`. No code ported; the
  full source methodology remains in `NPML/plan.md`.

### Earlier (undated)

- Added two predeclared GWOSC follow-up experiments. Filtering/statistic
  equivalence applies one shared GWpy-designed FIR through explicit FFT
  convolution and GWpy convolution over a frozen FIR-duration/edge-trim grid,
  first on stationary synthetic noise and then real H1/L1 windows. Time-local
  noise modelling compares leave-one-out global PSDs with 32/64/96-second
  local radii and archives chronological, template-projected, and fixed
  narrow-band diagnostics. The primary local model must cover at least 90% of
  evaluation windows.
- Replaced the undocumented approximate chirp in the GWOSC config with the
  public GW150914 numerical-simulation waveform used by GWpy's injection
  example; the downloader and run records retain its URL, checksum,
  resampling, placement, and normalization metadata.
- Archived and documented the first controlled 256-second GW150914 H1/L1
  verification run. Stage 0 passed remotely with 89 tests, and the repository
  median-Welch PSD matched GWpy exactly, but the predeclared held-out amplitude
  calibration failed (H1 median ratio 1.986; L1 1.263).
- Expanded GWOSC diagnostics without loosening acceptance: official
  `H1_DATA`/`L1_DATA` segment capture and enforcement, per-split indices and
  timestamps, per-window null amplitudes/scores, calibration-referenced
  evaluation quality, five chronological blocked splits with local
  calibration, and repository/GWpy matched-statistic comparisons.
- Corrected the remote runbook compact-summary schema from `split_results` to
  `splits` and added data-quality/blocked-split evidence to the summary.
- Removed accidental root-level EMPCA duplicates and restored the Paper 2
  ResNet backbone to `src/CNN/`; canonical Paper 1 implementations remain
  exclusively under `src/canonical/`.
- S3: replaced the agreement-by-construction bridge with an independently
  gradient-trained tied weighted linear autoencoder
  (`autoencoders/trained.py`), verified against EMPCA by optimality gap,
  principal angle, and M-orthonormality on diagonal and full covariance;
  models can be persisted via `scripts/train_s3_ae.py`.
- Added `src/canonical/`: the no-smoothing EMPCA, OF, PSD, and weight code, plus
  Stephen Bailey's reference WPCA as an independent oracle; pinned the
  TCY↔Bailey subspace agreement and the rFFT→real inner-product preservation.
- Item 2: added the central rFFT↔real representation primitive
  (`rfft_to_real`, `real_weight_vector`, `complex_to_real_whitened`); routed S5
  (fixing the real-part-only bug) and S1 (OF amplitude as a whitened projection)
  through it; consolidated the OF weight convention to one source of truth.
- Item 3: added the seed-sweep / held-out / bootstrap-CI harness
  (`noise_geometry/validation/`, `scripts/sweep.py`); wired held-out evaluation
  into S5 and demonstrated the metric reversal with paired 95% intervals.
- Rolled the harness across all synthetic gates: `scripts/sweep_all.py` archives
  per-gate CSV + CI JSON; added held-out (`test_frac`) to S4 and S6; encoded
  each gate's acceptance as a multi-seed regression test
  (`tests/test_synthetic_gate_acceptance.py`).
- S2: added the `amplitude_model` (real|complex) flag; pinned the rank-1
  EMPCA/OF angle's 1/√N convergence and its robustness to coefficient phase.
- Recorded findings and decisions in `docs/archive/2026-06-29_pre_consolidation/paper1_validation_progress.md`
  (EMPCA smoothing caveat, complex EMPCA phase DOF, S2 finite-sample √N scaling,
  real-vs-complex amplitude decision).
- Repaired import breakage from the `src/canonical/` reorganisation
  (`import src`, the S2 benchmark, and three test modules).
- Added a one-command Stage 0 remote reproducibility gate that enforces a clean
  checkout, runs the five roadmap commands, and archives environment,
  dependency, git, timing, and per-command log evidence.
- Added the GWOSC/GWpy reference path: bin-for-bin one-sided PSD normalization,
  held-out edge-trimmed whitening calibration, cached-run integration, reference
  preprocessing output, and download checksums/tool versions.
- Corrected GWOSC injection SNR normalization using the FFT/PSD amplitude
  variance; separated PSD calibration from held-out injection windows; added
  paired recovery/null diagnostics, multi-window GWpy whitening summaries, and
  longest-cache selection; expanded the default event cache to 256 seconds.
- Replaced the GWOSC boxcar mean PSD with Hann-windowed, FINDCHIRP
  bias-corrected median aggregation matched to GWpy; added robust calibration
  window glitch diagnostics and a five-split held-out amplitude-calibration
  acceptance gate that marks failed records as `failed_acceptance`.

- Added a config-driven Paper 1 experiment runner and S0-S9 synthetic suite.
- Added remote-server data-root resolution with default
  `/ceph/dwong/paper1_dataset`.
- Added idempotent GWOSC caching/event-injection analysis and CRESST
  cache/loading/reconstruction helpers.
- Expanded covariance, PSD, whitening, likelihood, and residual diagnostics.
- Added `uv` dependencies while retaining a conda fallback.
- Added archive, results, validation, metrics, preprocessing, and limitations
  documentation.
- Consolidated overlapping validation plans, experiment registries, dataset
  notes, metrics, preprocessing rules, and acceptance gates into
  `docs/archive/2026-06-29_pre_consolidation/VALIDATION_ROADMAP.md`; archived
  the superseded DELight/K-alpha plan.
- Preserved adjacent Paper 2, NPML, TraceSimulator, and QP simulator code while
  consolidating the verified Paper 1 OF/EMPCA implementations under
  `src/canonical/`.
