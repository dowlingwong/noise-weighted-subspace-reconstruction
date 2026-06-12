# Noise-Weighted Subspace Reconstruction

Open-source companion repository for the manuscript **"A Unified Maximum-Likelihood Framework for Optimal Filtering and Subspace Signal Reconstruction"**.

Primary manuscript file in this repo: `Generalized_OF.pdf`.

## What This Repo Demonstrates

This codebase supports three core claims from the paper:

1. **Optimal Filter (OF) as weighted ML estimator** under Gaussian noise.
2. **Rank-1 EMPCA and OF equivalence** when weighting, preprocessing, and gauge conventions are matched.
3. **EMPCA as a noise-aware linear subspace method** (and its relation to weighted linear autoencoder formulations).

It also hosts the experiment code for the nonlinear follow-up (Paper 2) under `paper2/`.

## Datasets

This project uses three data sources plus supporting measured spectra. Summary:

| Dataset | Location | Real / simulated | Channels | Key features | Open-source status |
|---|---|---|---|---|---|
| K-alpha calibration traces | `data/k_alpha/` | Real detector data | 1 | Real colored noise, measured PSD, ground-truth line energy | Internal; large `.h5` files not committed |
| TraceSimulator | `TraceSimulator/` | Simulated (full detector physics) | Multi | Position-dependent multichannel tracesets, quanta statistics, PSD noise | **Closed source — copyright not ours; do not redistribute** |
| QP simulator + enhanced noise module | `QP_simulator/` | Simulated (lightweight, controllable) | 1 to multi | Clean/noisy pairs; non-stationary, non-Gaussian, correlated multichannel noise | Ours; open-sourceable |
| Measured noise spectra | `noise_sample/`, `data/Noise_PSD/`, `data/weight/` | Measured component spectra | — | SQUID/Johnson/TD/Er ASDs, total noise, signal template, SNR² weights | Small files committed |

### 1. K-alpha calibration traces (real data)

Single-channel K-alpha X-ray calibration pulses from a cryogenic microcalorimeter. Used for the paper's real-data experiments (OF–EMPCA equivalence, Bridge Theorem check, rank-k behavior).

- Expected files (not committed due to size): `data/k_alpha/k_alpha_traces.h5` (dataset `traces`, shape `(4358, 32768)`) and `data/k_alpha/k_alpha_rqs.h5` (dataset `rqs`); loader contract in `paper2/data/datasets.py`.
- Committed: pulse template `data/k_alpha/template_K_alpha_tight.npy`.
- Features: **real** pulses with real colored detector noise; noise PSD measurable from noise traces (`src/PSDCalculator.py`; precomputed PSDs in `data/Noise_PSD/`, SNR² weights in `data/weight/`); fixed calibration energy gives a ground-truth amplitude scale; approximately stationary noise; natural timing jitter and pulse-shape variation.

### 2. TraceSimulator (multichannel DELight trace simulator) — closed source

Full detector-physics simulator under `TraceSimulator/`: energy partitioning into phonon/triplet/UV/IR quanta (`DELightSignalFormation`), position-dependent collection-efficiency maps (`SimulationMap`), channel geometry binning (`CylindricalBinning`, `PolygonBinning`), Poisson/multinomial quanta statistics, template shaping, and PSD-based stationary noise (`NoiseGenerator`).

- Produces physically realistic **multichannel** tracesets with position-dependent pulse-shape variation — the only source here of detector-realistic cross-channel signal correlations.
- **Copyright is not held by this project. It must not be open-sourced or redistributed from this repository.** Before any public release of this repo, remove `TraceSimulator/` (or replace it with a pointer/submodule to the upstream private repo).
- Requires input maps available on the `kalinka` system; see `TraceSimulator/README.md`.

### 3. QP simulator + enhanced noise module (ours, open)

Lightweight, fully controllable simulation stack under `QP_simulator/`:

- `QPSimulator.py`: standalone clean quasiparticle traces from explicit QP arrival times (no dependency on TraceSimulator); rise/decay template, calibrated gain.
- `noise_module/`: composable noise pipeline (`NoiseGenerator` → `TemporalNoiseWrapper` → `ArtifactInjector` → `MultiChannelNoiseGenerator`):
  - `NoiseGenerator`: stationary Gaussian core; analytic white/pink/Brownian/blue/violet PSDs or custom tabulated PSD (can inject measured detector spectra).
  - `TemporalNoiseWrapper`: **non-stationary** noise — piecewise-stationary segments, slow drift, local variance modulation.
  - `ArtifactInjector`: **non-Gaussian** contamination — spectral lines, glitches, bursts, sparse heavy-tailed impulses.
  - `MultiChannelNoiseGenerator`: independent, shared+private, or low-rank-correlated channels.
- Features: ground-truth clean/noisy pairs and signal masks; every assumption of the linear Gaussian theory (stationarity, Gaussianity, channel independence) can be violated in a controlled, ablatable way. Unit-tested (`noise_module/tests/`), with tutorials (`qp_simulator_tutorial.ipynb`, `noise_module_tutorial.ipynb`).

### 4. Measured noise spectra (supporting data)

- `noise_sample/Al2O3_Al_athermal/`: measured component amplitude spectral densities (SQUID, Johnson, thermodynamic, athermal-excess) plus total noise and signal template — inputs for the noise-budget Studies A/B (paper Appendix F.8/G).
- `data/Noise_PSD/*.npy`: precomputed noise PSDs (white/pink/Brownian/blue/violet/MMC).
- `data/weight/*.npy`: PSD and SNR² weights used by the OF/EMPCA pipelines.

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
# the repo is used via PYTHONPATH (there is no installable pyproject yet)
PYTHONPATH=. pytest -q tests
```

Smoke-test notebook (imports, rank-1 OF≡EMPCA equivalence, rank-k fast-vs-full
solver, PSD normalization round-trip): `notebooks/smoke_test_code_validation.ipynb`.

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
