# TIDMAD Phase 0 — Data Contract and Baseline Reproduction: Result

Date: 2026-07-27. Stage: TIDMAD (Paper 1's second real-data anchor).
Roadmap: [`../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`](../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md).
Predeclaration: [`TIDMAD_PREDECLARED_GATES.md`](TIDMAD_PREDECLARED_GATES.md).
Contract: `docs/tidmad_data_contract.json` (`verified: true`).
Evidence: `evidence/tidmad/phase0/G0_baseline_reproduction.json`,
`results/metrics/TIDMAD_P0_inspection.json`.
Code: `scripts/tidmad_phase0_contract.py`, `scripts/run_tidmad_stage.py`,
`scripts/download/download_tidmad.py`.

## Gate G0: satisfied

Reproduced the release's own `none` and `savgol` coarse denoising scores
(`inference.py` + `benchmark.py`, unmodified) on the full 20-file validation
set:

| Model | Our coarse score | Published coarse | Published fine |
| --- | --- | --- | --- |
| `none` | **1.1026** | 1.10 | 1.00 |
| `savgol` | -2.7708 | -2.35 | -2.77 |

`none` matches to the precision the published table reports. This is the
gate's real weight: `none` reads the raw SQUID and reference channels with no
processing in between, so a close match is direct evidence that channel
semantics, data integrity, sampling rate, and every environment fix below are
correct — independent of anything about the `savgol` baseline. `savgol` does
not match, and the reason is understood and does not indict the reproduction
(see below).

## Why this took five attempts

### 1. HDF5 structure was nested, not flat

`inspect_file()` in `tidmad_phase0_contract.py` originally scanned
`handle.keys()` at the file root and found nothing usable. The release stores
each channel under a nested group, `timeseries/channel0001/timeseries`
(int8, one dataset per channel), not as top-level datasets. Fixed by walking
the file with `visititems` and keying discovered datasets by their group
label. The data contract now stores the dataset's full HDF5 path (e.g.
`timeseries/channel0001/timeseries`), which is what the loader actually
indexes.

### 2. Channel-role classification: release convention over spectral heuristic

The original design classified SQUID vs. reference by spectral line
dominance (reference = line-dominated, SQUID = broadband). On the downloaded
files this was inconclusive on the probed window (channel0001 dominance
0.193, channel0002 0.015 — neither cleared the 0.5/0.2 thresholds). Five
independent upstream scripts (`train.py`, `inference.py`, `benchmark.py`,
`brazilband.py`, `array2h5.py`) all hardcode `channel0001` as the SQUID
readout and `channel0002` as the reference/target — stronger evidence than a
short-window spectral probe, and it is what `benchmark.py` itself assumes.
`classify_channels()` now trusts that convention when both channels are
present, keeping the spectral numbers in the record as an audited diagnostic
rather than a gate.

### 3. Two-file subset was not representative of the published, dataset-averaged score

Downloading only 2 of 20 validation files (the roadmap's original Phase 0
plan, for economy) gave `none` = -1.937 against a published 1.10 — wrong sign,
not just imprecise. The paper states the log base (5.27) is calibrated so
`none` scores ≈1 **averaged over the full validation dataset**; files are
numbered by injection frequency band (`inference.py`'s own comments: files
0-3 are "Low Frequency"), and SQUID noise is steeply 1/f-colored. A
low-frequency-only 2-file subset is not a fair sample of a calibration fit
against all bands. Downloading the full 20-file set and rerunning resolved
this immediately (`none` = 1.1026).

### 4. A corrupted download

One of the newly downloaded files, `abra_validation_0013.h5`, had ballooned
to 17.76 GB against an expected ~2.2 GB (a corrupted/duplicated stream, not a
truncation) and failed with `OSError: wrong B-tree signature`. Found by
spot-checking all 20 files with multi-offset `h5py` reads. Deleted and
re-fetched cleanly; the other 19 files were confirmed intact the same way.

### 5. Environment mismatches against a ~2024 codebase

- **numpy 2.x**: the release's `array + 128` pattern (converting signed int8
  ADC codes to an unsigned-style 0-255 representation) relies on numpy's
  pre-2.0 value-based scalar promotion, removed by NEP 50. Under numpy 2.4.6
  this raises `OverflowError`. Not fixed by downgrading this project's own
  numpy (too large a blast radius); instead built an isolated venv at
  `~/src/TIDMAD/.venv` with `numpy<2` + CPU-only torch + scipy/h5py/tqdm/
  matplotlib/torchsnooper, and added a `_tidmad_python()` helper (in
  `run_tidmad_stage.py` and `score.py`) that prefers `<checkout>/.venv/bin/python`
  for upstream subprocess calls.
- **torch import**: `inference.py` unconditionally imports `network.py`
  (which imports `torch`) even for classical, non-neural models. Added a
  `tidmad` optional-dependency extra (`torch`, `torchsnooper`) to this
  project's own `pyproject.toml`, since the driver's own scripts also need it
  importable at parse time.
- **Sequential inference crashes on classical models**: `inference.py`'s
  non-parallel path references a `model` variable that is only ever assigned
  inside the deep-model branches; the classical-model branch is `else: pass`,
  so the sequential loop raises `UnboundLocalError`. The `-p`/`--parallel`
  path explicitly passes `None` in its place, which is correct for
  non-neural models and needs no upstream patch. Both `inference.py` and
  `benchmark.py` invocations in `step_baseline_gate` now pass `-p`
  (`benchmark.py` originally lacked it, making its single-threaded coarse
  scan over 20 files take upwards of 20 minutes; adding it brought it down
  to a couple of minutes, matching the release's own README estimate).
- **Interactive download prompt**: `download_data.py` always prompts
  `Do you want to proceed? [y/n]` unless `-f`/`--force` is passed, which
  raises `EOFError` under `nohup` with no stdin. `download_tidmad.py` now
  always passes `-f`, since it exists specifically to run unattended.

## The `savgol` discrepancy: diagnosed, not a bug in this reproduction

`savgol`'s coarse score was bit-identical (16 significant digits,
`-2.7708098959837675`) between the 2-file and 20-file runs despite a 10x
increase in the number of one-second segments scored (confirmed genuinely
computed both times via the upstream `tqdm` totals, 40/40 and 400/400 — not a
caching artifact). That pointed at something structural rather than a
sampling effect.

`scipy.signal.savgol_coeffs(window_length=100, polyorder=11)` — exactly the
release's published hyperparameters — returns coefficients that sum to
`1.7e-10` instead of the required `1.0` (a valid Savitzky-Golay filter must
exactly reproduce a constant input; the coefficients summing to 1 is a basic
identity of the fit, independent of scipy version — confirmed identical
behaviour on scipy 1.17.1 and 1.13.1). Sweeping `polyorder` at fixed
`window_length=100`:

| polyorder | coefficient sum |
| --- | --- |
| 2-9 | ≈1.000 |
| 10 | 0.000345 |
| **11 (published)** | **1.7e-10** |
| 12+ | garbage, sign-flipping |

This is the standard signature of Vandermonde/normal-equations
ill-conditioning in high-degree polynomial least-squares fitting: fitting an
11th-degree polynomial to 100 points is past where double precision can
support the fit without catastrophic cancellation. Filtering a
constant-128 array with these parameters returns 128 only within the
edge-handling region (`mode='interp'`'s special boundary fit, ≈`window_length/2`
samples each side); the entire interior collapses to ≈0.

Consequence: the published `savgol` baseline itself was almost certainly
computed from an effectively degenerate filter (consistent with it scoring
worse than doing nothing at all). Its exact value is sensitive to
floating-point/BLAS-library minutiae at the 1e-10 level, so bit-exact
reproduction is not achievable by any independent run, in this environment or
any other — different environments land on different near-garbage output and
therefore different final scores, all in the same badly-broken regime. This
is a property of the release's own choice of hyperparameters, not of this
project's pipeline, data, or environment.

## Verdict

`docs/tidmad_data_contract.json` is marked `verified: true`. `none`'s
near-exact match is the calibration anchor and is treated as sufficient;
`savgol`'s non-reproducibility is documented above rather than chased further.

## Next

Phase 1 (`scripts/tidmad_phase1_estimator_identity.py --mode file`) is
unblocked. It does not require `injection_frequencies_hz` /
`injection_amplitudes` from the contract (those remain empty pending
characterization), only `require_verified()` and `sampling_frequency_hz`.
