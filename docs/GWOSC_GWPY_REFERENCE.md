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
2. **Whitening calibration.** Multiple held-out off-source windows are whitened
   both by the repository's direct rFFT operation and by GWpy's
   inverse-spectrum FIR using the same supplied PSD/ASD. The direct path
   includes `sqrt(2 / fs)` to convert a one-sided PSD density to unit-variance
   time samples. Since GWpy's finite-duration filter has settling transients,
   `0.5 * fduration` is removed from each edge before pooled and per-window
   means, standard deviations, correlations, and relative differences are
   reported.

Calibration and evaluation windows are disjoint. The split seed, window
indices, and starts are persisted in the experiment record.

Injection amplitudes use the repository's full FFT/PSD amplitude variance:

```text
injection amplitude = target SNR × predicted amplitude sigma
```

Both ordinary recovered scores and paired injected-minus-null scores are
reported. The paired score is the direct implementation check; its expectation
is the configured target SNR.

GWpy documents the [`TimeSeries.psd`](https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.psd)
and [`TimeSeries.whiten`](https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.whiten)
APIs. In particular, GWpy states that whitening is normalized to zero mean and
unit variance for stationary Gaussian input and that half the FIR duration is
corrupted at each edge.

## Run it

After downloading the configured 256-second event cache:

```bash
uv run python scripts/download/download_gwosc.py --download --timeout 900
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
  should be near one for stationary off-source noise. Use the per-window
  distribution, not one selected trace.
- `injection_score_mean` (also recorded explicitly as
  `injection_paired_score_mean`) should equal `injection_target_snr` to
  numerical precision. `injection_unpaired_score_mean`, its spread, and the
  null-score distribution retain the held-out detector-noise realization.
- `null_sigma_over_predicted` compares held-out amplitude scatter with the
  PSD-predicted amplitude sigma. Values far from one indicate nonstationarity,
  finite-PSD effects, or an unsuitable window/regularization choice.
- The two whitened time series need not be pointwise identical on structured
  real noise: GWpy truncates the inverse spectrum into an FIR, while the
  repository path applies the frequency response directly. Their calibration
  and conclusions, not exact samples, are the primary comparison.
- `fduration_seconds`, high-pass frequency, detrending, PSD floor, calibration
  and evaluation counts, split indices, sample rate, and edge trim are
  persisted in the output.
