# Rebuild Summary

> Historical migration snapshot. For current status, next steps, and
> acceptance gates, use
> [`VALIDATION_ROADMAP.md`](VALIDATION_ROADMAP.md).

## Changed

- Added config-driven synthetic experiment execution and standardized JSON
  records.
- Added server data-root resolution and public-data cache helpers.
- Expanded noise, covariance, whitening, likelihood, and residual APIs.
- Added `uv`-first setup, conda fallback, archive policy, and validation docs.

## Preserved

Canonical `OptimumFilter`, independent GLS helpers, EMPCA implementations,
Paper2, NPML, TraceSimulator, QP simulator, notebooks, and legacy data assets
remain in place.

## Complete

S0-S9 have runnable configs and implementations. GWOSC can cache small event
windows and run event/injection metrics. CRESST can load NPZ/HDF5 traces and
run an initial OF/PCA/EMPCA/exact-AE comparison.

## Unfinished

Paper-grade public-data selection/figures, dataset-specific CRESST schema
validation, and optional TIDMAD benchmarking.

## Data Access

Default storage is `/ceph/dwong/paper1_dataset`. CRESST resource URLs may still
require selection or manual download from the official dataset page.
