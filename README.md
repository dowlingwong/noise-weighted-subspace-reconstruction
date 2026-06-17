# Noise-Weighted Subspace Reconstruction

Reproducible validation code for Paper 1:

**A Unified Maximum-Likelihood Framework for Signal Reconstruction: From
Optimal Filtering to Noise-Aware Linear Autoencoders**

The repository tests the claim that detector noise defines the reconstruction
geometry. For Gaussian noise the likelihood-aligned objective is

```text
(x - x_hat)^dagger Sigma^{-1} (x - x_hat)
```

Ordinary MSE is the corresponding likelihood only when the noise is white.

The authoritative experiment status, acceptance gates, dataset contracts, and
next steps are in
[`docs/VALIDATION_ROADMAP.md`](docs/VALIDATION_ROADMAP.md).

## Status

The config-driven synthetic suite S0-S9 is runnable. GWOSC has cached
event/injection analysis, and CRESST has a format-tolerant NPZ/HDF5
reconstruction path. Paper-grade public-data selection and figures still
require the real files on the remote server. TIDMAD is optional.

## Remote Installation

The primary target is a Linux server opened through VS Code. Use `uv`:

```bash
uv sync --extra dev --extra gwosc
```

Run commands with `uv run`. Conda is the fallback:

```bash
conda env create -f environment.yml
conda activate paper1-validation
```

## Data Root

Public data are stored outside the repository. The default is:

```text
/ceph/dwong/paper1_dataset
```

Resolution order is:

1. CLI `--data-root`
2. `PAPER1_DATA_ROOT`
3. config `data_root`
4. `/ceph/dwong/paper1_dataset`

The expected dataset directories are `gwosc/`, `cresst/`, and `tidmad/`, each
with `raw/`, `cache/`, and `processed/` subdirectories.

## Quickstart

```bash
uv run pytest -q
uv run python scripts/run_experiment.py \
  --config configs/synthetic/s0_smoke.yaml
```

Run the implemented synthetic core:

```bash
uv run python scripts/run_all_core.py
uv run python scripts/make_tables.py
uv run python scripts/make_all_figures.py
```

Run one experiment:

```bash
uv run python scripts/run_experiment.py \
  --config configs/synthetic/s5_metric_reversal.yaml
```

Each run writes a JSON record containing the config, seed, metrics, git commit,
dataset metadata, preprocessing metadata, and model metadata.

## Public Data

Check or download a short GWOSC event window:

```bash
uv run python scripts/download/download_gwosc.py
uv run python scripts/download/download_gwosc.py --download
uv run python scripts/run_experiment.py \
  --config configs/gwosc/gw150914_smoke.yaml
```

Prepare the CRESST cache and print current manual-download instructions:

```bash
uv run python scripts/download/download_cresst.py
uv run python scripts/run_experiment.py \
  --config configs/cresst/pulse_shape_smoke.yaml
```

Both scripts refuse to put large data inside this repository unless
`--allow-repo-data` is passed explicitly.

## Repository Structure

- `src/noise_geometry/`: maintained Paper 1 package.
- `src/OptimumFilter.py`, `src/of.py`, `src/EMPCA/`: preserved canonical and
  independent legacy implementations used for equivalence checks.
- `configs/`: config-driven experiment definitions.
- `scripts/`: experiment, download, preprocessing, figure, and table entry points.
- `experiments/`: compatibility entry points for earlier smoke experiments.
- `tests/`: fast theory and regression tests.
- `docs/`: validation, data, metrics, preprocessing, and figure mapping.
- `archive/`: reviewed stale material only; legacy directories are otherwise
  preserved in place.
- `paper2/`, `NPML/`, `TraceSimulator/`, `QP_simulator/`: adjacent or legacy
  work, not the default Paper 1 execution surface.

## Reproducibility

- Synthetic runs are seed-controlled.
- Public data are cached outside git.
- Results are generated under `results/`.
- Notebooks are exploratory; scripts and configs are the reproducibility surface.
- Raw MSE alone is never the primary success metric under structured noise.

## Dataset Citations

Use the acknowledgement and citation instructions from GWOSC and the CRESST
Dark Matter Data Center. TIDMAD must be cited if the optional extension is
used. See `docs/VALIDATION_ROADMAP.md`.

## Limitations

Synthetic experiments validate controlled assumptions. GWOSC is intended as a
public real-noise likelihood-geometry demonstration, not gravitational-wave
parameter estimation. CRESST is for pulse-shape reconstruction, not a
dark-matter exclusion analysis. Covariance estimation, nonstationarity, and
non-Gaussian noise remain explicit robustness questions.
