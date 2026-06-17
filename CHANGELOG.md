# Changelog

## Unreleased

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
- Preserved legacy OF, EMPCA, Paper 2, NPML, TraceSimulator, and QP simulator
  code without moving or breaking imports.
