# Repository Audit

Audit date: 2026-06-16.

## What Was Found

The repository already contains useful Paper 1 code:

- `src/OptimumFilter.py`: optimized optimal-filter implementation with
  zero-shift, time-shift, and sliding-window support.
- `src/of.py`: compact independent GLS / rank-1 projection implementation used
  to cross-check OF conventions.
- `src/EMPCA/`: EMPCA implementations and equivalence utilities.
- `src/PSDCalculator.py`, `src/make_weights.py`, `src/weights.py`,
  `src/metrics.py`: PSD, weight, and metric helpers.
- `tests/`: existing fast tests for package imports, OF/rank-1 equivalence, and
  rank-k solver behavior.
- `QP_simulator/`: useful controllable signal/noise simulation logic.
- `data/noise_sample/` and `data/noise_samples/`: small measured noise spectra
  and notebooks.
- `notebooks/` and `implementation/`: exploratory notebooks and generated
  experiment support.
- `plan/`: paper planning notes and revision plan.

The repository also contains adjacent or legacy material:

- `paper2/`, `NPML/`, and `scripts/run_paper2_*`: Paper 2 / nonlinear work.
- `TraceSimulator/`: detector simulator code marked in README as not
  redistributable without checking ownership.
- `archive/`: existing legacy notebook archive.
- Large local data: `data/k_alpha/k_alpha_traces.h5`.

The previous `README.md` referenced paths that do not exist in this checkout,
including `src/noise_weighted_sr/` and `PCA_dev/`. That made the repository
hard for a new user to run.

## Preserved

- Existing `src` compatibility package and tests were left in place.
- Existing OF, PSD, EMPCA, QP simulator, Paper 2, plan, archive, and notebook
  files were not deleted.
- Large local data were not removed.

## Migrated or Repackaged

Reusable Paper 1 logic was organized into a new `src/noise_geometry/` package:

- `noise/`: covariance, PSD, whitening, regularization, and colored-noise
  generation.
- `filters/`: GLS / OF amplitude, matched-filter score, rank-1 projection.
- `subspace/`: PCA, diagonal weighted PCA, projections, principal angles.
- `autoencoders/`: closed-form tied linear AE baselines for MSE and weighted
  losses.
- `metrics/`: raw MSE, weighted residual, whitened MSE, amplitude bias.
- `synthetic/`: reusable pulse templates and synthetic benchmark builders.
- `gwosc/`, `cresst/`, `tidmad/`: dataset-specific scaffold locations.

Experiment entry points were added under `experiments/` rather than notebooks.

## Archive Candidates

These should be reviewed before moving:

- `implementation/*.ipynb` and `implementation/generate_block_notebooks.py`:
  likely generated or exploratory Paper 1 work.
- `data/noise_sample/`: apparent duplicate of `data/noise_samples/`.
- `pytest-cache-files-4_ur1uwp/` and `.pytest_cache/`: generated cache
  directories.
- `paper2/results/`: generated Paper 2 outputs.

## Delete Only After Human Approval

- `TraceSimulator/`: may be closed-source or externally owned.
- `data/k_alpha/k_alpha_traces.h5`: large local detector data.
- Any notebooks containing unique plots or result provenance.
- Any generated figures/tables already used in a draft or presentation.

## Structural Changes Performed

| Current Location | New Location | Reason | Risk | Performed |
|---|---|---|---|---|
| scattered source utilities | `src/noise_geometry/` | clear reusable Paper 1 package while preserving old imports | low; additive | yes |
| no canonical docs | `docs/*.md` | required audit, plan, links, registry | low | yes |
| notebook-only experiment logic | `experiments/synthetic/*/run.py` | runnable smoke experiments | low | yes |
| no public-data scaffold | `scripts/download/`, `scripts/preprocess/`, `configs/` | reproducible dataset pipeline contracts | low | yes |

## Hidden Assumptions and Path Risks

- Some old docs mention `PCA_dev`, which is absent.
- The legacy README described local/internal K-alpha data and TraceSimulator
  ownership constraints. Those remain important for public release.
- Public-data dependencies such as `gwosc` and `gwpy` are optional and not in the
  base environment.
- Large arrays and public dataset downloads should live outside tracked source
  paths.
