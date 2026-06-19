# Changelog

## Unreleased

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
- Recorded findings and decisions in `docs/paper1_validation_progress.md`
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
  `docs/VALIDATION_ROADMAP.md`; archived the superseded DELight/K-alpha plan.
- Preserved adjacent Paper 2, NPML, TraceSimulator, and QP simulator code while
  consolidating the verified Paper 1 OF/EMPCA implementations under
  `src/canonical/`.
