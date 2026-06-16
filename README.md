# Noise-Weighted Subspace Reconstruction

Reproducible experiment code for Paper 1:

**A Unified Maximum-Likelihood Framework for Signal Reconstruction: From Optimal Filtering to Noise-Aware Linear Autoencoders**

The project tests one organizing claim: structured detector noise defines the
geometry of reconstruction. For Gaussian noise, the likelihood-aligned objective
is the Mahalanobis / chi-square residual

```text
chi2(s_hat) = (x - s_hat)^dagger Sigma^{-1} (x - s_hat)
```

Under this metric, optimal filtering is fixed rank-1 maximum-likelihood
projection, EMPCA is learned rank-k maximum-likelihood projection, and a tied
linear autoencoder trained with the `Sigma^{-1}`-weighted loss recovers the same
noise-aware subspace. Ordinary MSE is the correct likelihood only for white
noise.

## Repository Structure

- `src/noise_geometry/`: reusable Paper 1 package modules.
- `experiments/synthetic/`: controlled benchmarks for theorem checks and metric reversal.
- `experiments/gwosc/`: public GWOSC smoke checks and planned real-noise experiments.
- `experiments/cresst/`: planned CRESST pulse-shape experiments.
- `experiments/tidmad_optional/`: optional TIDMAD extension.
- `configs/`: small, reproducible experiment configs.
- `scripts/download/` and `scripts/preprocess/`: dataset access and preprocessing entry points.
- `tests/`: fast regression and smoke tests.
- `docs/`: audit, canonical experiment plan, dataset links, and experiment registry.
- `data/`: small committed sample inputs only; large raw or processed data are ignored.
- `results/`: generated figures, tables, summaries, and model outputs; ignored by git.
- `archive/`: legacy notebooks or code kept for reference.
- `paper2/`, `NPML/`, `TraceSimulator/`, `QP_simulator/`: prior or adjacent work preserved for now.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional public-data dependencies for GWOSC work:

```bash
python -m pip install gwosc gwpy
```

## Quickstart

Run the fast test suite:

```bash
pytest -q
```

Run the synthetic OF/rank-1 weighted-subspace smoke test:

```bash
python experiments/synthetic/of_empca_equivalence/run.py
```

Run the metric-reversal smoke script:

```bash
python experiments/synthetic/metric_reversal/run.py
```

Check whether GWOSC optional dependencies are available without downloading data:

```bash
python experiments/gwosc/smoke.py
```

## Data Policy

No large public datasets should be committed. Use these locations:

- `data/raw/`: raw local downloads.
- `data/external/`: public dataset mirrors or manually downloaded archives.
- `data/interim/`: temporary preprocessing products.
- `data/processed/`: generated processed arrays.
- `results/`: generated experiment outputs.

The `.gitignore` excludes large data and model artifact types including `.h5`,
`.hdf5`, `.root`, `.npy`, `.npz`, `.pt`, `.ckpt`, and `.zst`.

## Reproducibility

- Synthetic scripts use fixed seeds by default.
- Config files live under `configs/`.
- Smoke tests avoid full public-data downloads.
- Notebooks are retained for exploration only; executable scripts are the
  reproducibility surface.
- Generated outputs should be written under `results/`.

## Dataset Links

See [docs/dataset_links_and_access.md](docs/dataset_links_and_access.md) for
canonical GWOSC, CRESST, and optional TIDMAD access notes.

## Current Status

This is a preliminary private research codebase, not an official result from any
experimental collaboration. The synthetic experiment scaffold is runnable; GWOSC
and CRESST pipelines are documented and scaffolded for implementation without
large downloads by default.
