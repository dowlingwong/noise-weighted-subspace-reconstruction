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

The current documentation map is in [`docs/README.md`](docs/README.md).
Current status and paper claim boundaries are in
[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).

## Status

### September 2026 evidence map (supersedes the July summary below)

The canonical manuscript is `arXiv-v1-polished-extended-preprint/main.tex` (plus
`supplement.tex`).  Its September claim boundaries differ from the July
consolidation retained below.  In brief:

- The controlled synthetic suite S1-S9 verifies the estimator identities.
  Predictions P1 and P2 are confirmed on the simulator; P3 holds there as a
  property of the generator, not a guarantee.
- CRESST-A remains a positive but **archive-specific** same-rank residual
  effect: seven correlated run/channel clusters from two runs, every interval on
  the same-rank fraction post-freeze, and a covariance the release cannot
  validate.  It is residual reduction in the optimized metric, not improved
  physical reconstruction.
- GWOSC is a completed negative public real-noise calibration stress test.
- **TIDMAD is no longer optional.**  The whole-window stage established *no
  useful recovery* (no rank-2 reconstruction beat the zero predictor), and the
  narrowband T22 stage withheld all three registered verdicts because its
  dependence gate failed.  The matched-rank weighted-residual contrast that
  would test the joint-ordering hypothesis was not computed on either stage;
  those stages are therefore *not* a test of it.
- The successor packages under
  [`experiments/mlst2026/successor_20260913/`](experiments/mlst2026/successor_20260913/)
  (T19-S, T20-S, T21-S, T22-S simulation, CRESST archive/calibration) are
  **configured and locally tested but not executed**; they are not evidence for
  any manuscript claim.  The matched TIDMAD instrument stage and the full
  CRESST raw-pulse robustness extension remain **explicitly blocked**; see
  [`experiments/mlst2026/successor_20260913/remote/README.md`](experiments/mlst2026/successor_20260913/remote/README.md).

The July consolidation below is retained as historical evidence and is not
overwritten.

### July consolidation (historical)

The controlled synthetic suite S1-S9 has verified positive evidence. GWOSC is
a completed negative public real-noise stress test: PSD/reference and
shared-FIR implementation checks passed, but global and predeclared local PSD
real-noise calibration failed. **CRESST-A is a completed positive public
real-data validation** of the linear noise-geometry hierarchy (see
[`docs/results/CRESST_RESULT.md`](docs/results/CRESST_RESULT.md)); it is not a "next
candidate" or pending item. TIDMAD is optional.

The noise-budget decomposition (Study A/B, `scripts/run_noise_budget.py`) is
computed directly from the authoritative `noise_module/Al2O3_Al_athermal/`
component spectra with their native `SQUID`/`Johnson`/`TD`/`Er` labels;
PSD/CSD factorization of a total spectrum yields statistically orthogonal
modes, not uniquely identifiable physical sources, and Study B's sensitivity
attribution is reported under two explicit, non-exhaustive acquisition
branches (in-band vs. alias-folded), never as detector truth. The stationary
Gaussian noise generator's discrete convention (interior complex-Gaussian
rFFT bins, real endpoint DC/Nyquist bins, periodic/circulant FFT-grid
covariance vs. a separate exact-Toeplitz sampler) is pinned in
[`noise_module/noise_module/PAPER_CONVENTION.md`](noise_module/noise_module/PAPER_CONVENTION.md).

Paper 2 (`paper2/`) neural-architecture and coverage results are exploratory
only pending separate implementation repairs (RNG seeding, test-set reuse,
timing-metric coordinates, Transformer positional/cross-channel encoding);
see `docs/EVALUATION.md`.

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

Close the remote Stage 0 reproducibility gate from a clean Linux checkout with:

```bash
python3 scripts/stage0_remote_gate.py
```

The command runs the complete five-command gate and archives the environment
and logs under `results/stage0/`. See
[`docs/REMOTE_EXECUTION.md`](docs/REMOTE_EXECUTION.md).

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
uv run python scripts/download/download_gwosc.py --download --timeout 900
uv run python scripts/preprocess/preprocess_gwosc.py --reference-check
uv run python scripts/run_experiment.py \
  --config configs/gwosc/gw150914_smoke.yaml
uv run python scripts/run_experiment.py \
  --config configs/gwosc/filter_statistic_equivalence.yaml
uv run python scripts/run_experiment.py \
  --config configs/gwosc/time_local_noise.yaml
```

The preprocessing command compares PSD normalization and held-out whitening
calibration against GWpy. The GWOSC outcome and follow-up interpretation are
summarized in [`docs/results/GWOSC_RESULT.md`](docs/results/GWOSC_RESULT.md); commands and
acceptance rules are in
[`docs/EXPERIMENT_PROTOCOLS.md`](docs/EXPERIMENT_PROTOCOLS.md).

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
- `src/canonical/`: canonical OF, PSD, weighting, complex EMPCA, and the
  independent Bailey WPCA reference used for equivalence checks.
- `src/of.py`: compact independent GLS reference used for OF cross-checks.
- `configs/`: config-driven experiment definitions.
- `scripts/`: experiment, download, preprocessing, figure, and table entry points.
- `experiments/`: compatibility entry points for earlier smoke experiments.
- `tests/`: fast theory and regression tests.
- `docs/`: current status, the consolidated evaluation, the experiment
  registry, and protocols, with `docs/results/`, `docs/plans/`,
  `docs/runbooks/`, and `docs/background/` subdirectories. Start at
  [`docs/README.md`](docs/README.md).
- `archive/` and `docs/archive/`: reviewed stale material only; legacy
  directories are otherwise preserved in place.
- `paper2/`, `paper3/`: planning and code for the follow-up papers.
- `noise_module/`: the noise simulator, its conventions, and its roadmap.
- `NPML/`, `TraceSimulator/`, `QP_simulator/`: adjacent or legacy work, not the
  default Paper 1 execution surface.

## Reproducibility

- Synthetic runs are seed-controlled.
- Public data are cached outside git.
- Results are generated under `results/`.
- Notebooks are exploratory; scripts and configs are the reproducibility surface.
- Raw MSE alone is never the primary success metric under structured noise.

## Dataset Citations

Use the acknowledgement and citation instructions from GWOSC and the CRESST
Dark Matter Data Center. TIDMAD must be cited if the optional extension is
used. See `docs/EXPERIMENT_PROTOCOLS.md`.

## Limitations

Synthetic experiments validate controlled assumptions. GWOSC is a negative
public real-noise stress test for the current revision, not gravitational-wave
parameter estimation and not calibrated detection evidence. CRESST is for
pulse-shape reconstruction, not a dark-matter exclusion analysis. Covariance
estimation, nonstationarity, and non-Gaussian noise remain explicit robustness
questions.
