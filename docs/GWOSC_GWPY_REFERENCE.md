# GWOSC PSD and Whitening Reference Path

The GWOSC reference check is an implementation/convention audit. It does not
claim astrophysical parameter recovery.

## What is compared

`src/noise_geometry/gwosc/reference.py` performs two distinct checks:

1. **PSD density normalization.** Each off-source calibration window is treated
   as one rectangular, non-overlapping periodogram. The repository
   `estimate_psd_rfft` result is compared bin-for-bin with
   `gwpy.timeseries.TimeSeries.psd(method="welch", window="boxcar")`. Matching
   estimators are intentional: any discrepancy is then a units, one-sided
   factor, FFT, or frequency-grid error.
2. **Whitening calibration.** A held-out off-source window is whitened both by
   the repository's direct rFFT operation and by GWpy's inverse-spectrum FIR
   using the same supplied PSD/ASD. The direct path includes `sqrt(2 / fs)` to
   convert a one-sided PSD density to unit-variance time samples. Since GWpy's
   finite-duration filter has settling transients, `0.5 * fduration` is removed
   from each edge before means, standard deviations, correlation, and relative
   differences are reported.

The held-out whitening window is not used to estimate the PSD.

GWpy documents the [`TimeSeries.psd`](https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.psd)
and [`TimeSeries.whiten`](https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.whiten)
APIs. In particular, GWpy states that whitening is normalized to zero mean and
unit variance for stationary Gaussian input and that half the FIR duration is
corrupted at each edge.

## Run it

After downloading the configured event cache:

```bash
uv run python scripts/download/download_gwosc.py --download
uv run python scripts/preprocess/preprocess_gwosc.py --reference-check
```

The standalone reference record is written under:

```text
<data-root>/gwosc/processed/GW150914_gwpy_reference.json
```

The normal experiment runner also includes the same record under each
detector's `gwpy_reference` field because the default GWOSC config enables it:

```bash
uv run python scripts/run_experiment.py \
  --config configs/gwosc/gw150914_smoke.yaml
```

## Interpretation

- PSD `relative_l2_error`, frequency-grid error, and log-ratio diagnostics
  should be at floating-point noise when the matched estimators are used.
- Whitening standard deviations should be assessed on the trimmed interior and
  should be near one for stationary off-source noise.
- The two whitened time series need not be pointwise identical on structured
  real noise: GWpy truncates the inverse spectrum into an FIR, while the
  repository path applies the frequency response directly. Their calibration
  and conclusions, not exact samples, are the primary comparison.
- `fduration_seconds`, high-pass frequency, detrending, PSD floor, calibration
  count, sample rate, and edge trim are persisted in the output.
