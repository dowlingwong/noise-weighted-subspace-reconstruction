# Noise-Weighted Subspace Reconstruction

Open-source companion repository for the manuscript **"A Unified Maximum-Likelihood Framework for Optimal Filtering and Subspace Signal Reconstruction"**.

Primary manuscript file in this repo: `Generalized_OF.pdf`.

## What This Repo Demonstrates

This codebase supports three core claims from the paper:

1. **Optimal Filter (OF) as weighted ML estimator** under Gaussian noise.
2. **Rank-1 EMPCA and OF equivalence** when weighting, preprocessing, and gauge conventions are matched.
3. **EMPCA as a noise-aware linear subspace method** (and its relation to weighted linear autoencoder formulations).

## Repository Layout

- `src/noise_weighted_sr/`
  - Stable package-level utilities for weights, OF projection, metrics, config, and I/O.
- `scripts/`
  - Entry points that currently bridge to legacy implementations in `PCA_dev/`.
- `configs/`
  - Config templates for paths and run settings.
- `data/`
  - Canonical location for small sample data and reusable `.npy` weights.
- `results/`
  - Canonical location for generated artifacts (models, tables, figures).
- `tests/`
  - Minimal validation tests for weighting and rank-1 projection math.
- `PCA_dev/`
  - Original research workspace (notebooks, training scripts, equivalence studies, reusable modules).

## Where the Main Code Lives Today

Current production workflows are still notebook/script driven in `PCA_dev/`:

- OF implementation: `PCA_dev/reusable/OptimumFilter.py`
- EMPCA implementations:
  - `PCA_dev/reusable/empca_TCY.py`
  - `PCA_dev/reusable/empca_TCY_optimized.py`
  - `PCA_dev/reusable/empca_TCY_gpu.py`
- OF/EMPCA helper functions: `PCA_dev/reusable/empca_equivalence_utils.py`
- Training pipeline: `PCA_dev/wk4/train/train_empca_sum_channel.py`
- Equivalence experiments: `PCA_dev/wk4/equivalence/`

## Weights and `.npy` Assets

Use `.npy` assets as data artifacts, not as package source code.

Recommended convention:

- Store reusable input weights in `data/weights/`
- Keep generated outputs in `results/`
- Reference paths through `configs/default.yaml` (and optional local override)

Example categories:

- PSD weights: `noise_psd_*.npy`
- SNR^2 weights: `qp_snr2_weight_*.npy`
- Templates: `QP_template*.npy`

## Quickstart

```bash
conda env create -f environment.yml
conda activate nwsr
pip install -e .[dev]
PYTHONPATH=src pytest -q tests
```

## Running Current Pipelines

Wrapper scripts:

```bash
python scripts/make_traces.py
python scripts/train_empca.py
python scripts/eval_equivalence.py
```

These wrappers currently forward to `PCA_dev` workflows so migration can be incremental.

## Reproducibility Notes

- Some legacy scripts/notebooks still contain machine-specific paths (for example `/ceph/...`).
- For portable runs, update paths via config and local environment.
- Keep a manifest for important `.npy` files (name, shape, dtype, source, checksum).

## Current Verification Artifacts

Key summary outputs are in:

- `PCA_dev/wk4/equivalence/strict_equivalence_summary.json`
- `PCA_dev/wk4/equivalence/empca_linear_ae_summary.json`
- `PCA_dev/wk4/equivalence/empca_ae_whiten_vs_weighted_equivalence_summary.json`

These files document empirical checks for OF/EMPCA and EMPCA/linear-AE consistency.

## Citation

If this repository contributes to your work, cite the manuscript and link this repository.

## License

MIT License (see `LICENSE`).
