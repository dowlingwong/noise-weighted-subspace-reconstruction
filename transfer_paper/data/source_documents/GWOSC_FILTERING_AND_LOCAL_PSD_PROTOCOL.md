# GWOSC filtering equivalence and time-local PSD protocol

_Predeclared follow-up experiments after the failed global-PSD calibration run_

---

## 📋 Scientific purpose

The archived GW150914 runs established exact PSD agreement with GWpy but failed held-out amplitude calibration, especially in chronological blocks. Two explanations remain distinct and must not be tuned simultaneously. First, the original frequency-domain generalized least-squares statistic and GWpy’s finite-duration inverse-spectrum FIR path may implement materially different effective filters on four-second records. Second, a single PSD may be unable to represent time-varying template-projected noise even when broad-band power appears stable.

This protocol therefore defines two separate experiments. The filtering experiment first proves that one explicitly shared FIR matched statistic is numerically identical through an explicit FFT-convolution path and GWpy’s `TimeSeries.convolve` path. It then reports, without using the result for implementation acceptance, how that shared statistic differs from the original PSD-domain GLS statistic as FIR duration and edge trim vary. The time-local experiment separately compares a leave-one-out global PSD with fixed local calibration radii and records template-sensitive spectral diagnostics.

The previous failed global-PSD gate remains unchanged and is the baseline. Neither experiment relaxes its thresholds or removes held-out windows after inspecting their scores.

## 🧬 Documented waveform

The experiments use the public GW150914 numerical-simulation waveform referenced by GWpy’s official `TimeSeries` injection example:

```text
https://www.gw-openscience.org/s/events/GW150914/P150914/
fig2-unfiltered-waveform-H.txt
```

The downloader stores the source URL and SHA-256 digest in the GWOSC metadata. The source contains a 16384-Hz two-column time/strain waveform. The loader multiplies the second column by `1e-21`, resamples deterministically with polyphase resampling, aligns its absolute peak to 2.0 seconds in the four-second analysis record, and peak-normalizes the template. The normalization changes coefficient units but not normalized matched statistics. Every result records the source checksum, inferred source rate, resampling ratio, placement, and normalization.

Synthetic unit and regression tests use a documented sine-Gaussian probe when a network-fetched file is inappropriate. Its convention is

```text
h(t) = exp(-(t-t0)^2 / (2 sigma^2))
       cos(2 pi f0 (t-t0) + phase),
sigma = Q / (2 pi f0).
```

## 🔬 Experiment F1: filtering/statistic equivalence

The config is [`configs/gwosc/filter_statistic_equivalence.yaml`](../configs/gwosc/filter_statistic_equivalence.yaml). It must be executed with:

```bash
uv run python scripts/run_experiment.py \
  --config configs/gwosc/filter_statistic_equivalence.yaml
```

For each FIR duration, the experiment uses GWpy’s `fir_from_transfer` to construct one fixed filter from the supplied PSD. The frequency-domain implementation applies that same FIR through explicit zero-padded FFT multiplication, reproducing GWpy’s detrending, boundary taper, convolution centering, and `sqrt(2/fs)` normalization. The GWpy implementation applies the identical coefficients through `TimeSeries.convolve`. Both paths filter the data and template, use the same interior samples, and calculate

```text
rho_FIR = dot(whitened data, whitened template)
          / norm(whitened template).
```

This is the implementation-equivalence statistic. The original PSD-domain GLS score is retained as a third output, but its comparison with the finite FIR statistic is diagnostic because the FIR truncates the ideal inverse spectrum.

The complete predeclared cross-product is:

| Parameter | Values |
| --- | --- |
| FIR duration | 0.25, 0.5, 1.0, 2.0 seconds |
| Edge trim per side | 0.125, 0.25, 0.5, 1.0 seconds |
| FIR/window | Hann |
| Detrending | Constant |
| Primary descriptive setting | 1.0-second FIR, 0.5-second trim |

The experiment first runs 128 stationary pink-noise synthetic traces with a known PSD. It then uses the committed primary GWOSC split for H1 and L1. Implementation acceptance requires every synthetic and real setting to have a maximum score difference and relative L2 score difference no larger than `1e-10` between the explicit FFT-convolution and GWpy-convolution paths. The original GLS-to-FIR correlation, relative L2 difference, and standard-deviation ratio are reported but do not determine this gate.

This design answers a narrow question: whether the two software paths agree when they are asked to compute exactly the same statistic. It does not assume that a finite FIR is equivalent to the ideal PSD-domain statistic.

## ⏱️ Experiment L1: time-local noise modelling

The config is [`configs/gwosc/time_local_noise.yaml`](../configs/gwosc/time_local_noise.yaml). It must be executed with:

```bash
uv run python scripts/run_experiment.py \
  --config configs/gwosc/time_local_noise.yaml
```

Every off-source window is evaluated once. Its calibration set never includes that window. The global comparator uses all other valid windows. Local models use all valid windows whose start time lies within a predeclared symmetric radius:

| Model | Calibration rule |
| --- | --- |
| Global | All other valid off-source windows |
| Local 32 s | Other windows within 32 seconds |
| Local 64 s | Other windows within 64 seconds |
| Local 96 s | Other windows within 96 seconds |

The 64-second radius is the predeclared primary local model. The 32- and 96-second radii are sensitivity analyses. A local model requires at least eight calibration windows after quality screening. Each calibration PSD uses the same Hann, constant-detrended, bias-corrected median estimator and PSD floor as the baseline.

The primary outcome is the standard deviation of the normalized held-out template scores. The stationary synthetic control must place the primary local score standard deviation in `[0.8, 1.2]`. The real-data gate applies the same interval separately to H1 and L1. At least 90% of evaluation windows must have an available primary local model, so edge effects or quality rejections cannot create a pass by silently removing difficult windows. Other radii are reported regardless of outcome and must not replace the primary radius after inspection.

Every accepted off-source window is evaluated once and retained in timestamp order. Results are divided into five predeclared contiguous chronological blocks after scoring. These block values diagnose temporal instability; they do not create five independent replicates, and they do not change the calibration set or acceptance threshold after inspection.

## 📊 Template-sensitive spectral diagnostics

Broad-band RMS and total 20–512 Hz power did not explain the previous score excursions. The local-PSD experiment therefore stores each held-out window’s periodogram and model comparison in forms aligned with the waveform:

```text
template_projected_psd
    = sum_f periodogram(f) q(f),

q(f) = |H(f)|^2 / sum_f |H(f)|^2
```

after removing frequencies below the configured high pass. The corresponding model quantity and observed/model ratio are archived for every evaluation window and every PSD model.

The following fixed narrow bands are also integrated:

| Band | Range |
| --- | --- |
| Low | 20–80 Hz |
| Mid-low | 80–150 Hz |
| Mid-high | 150–300 Hz |
| High | 300–512 Hz |

For each band, the record contains observed power, model power, and their ratio. These diagnostics may explain score dispersion but are exploratory in this run. Any future quality cut or frequency exclusion derived from them requires a new frozen config and independent evidence run.

## 🛡️ Interpretation rules

If F1 passes, software application of the shared FIR statistic is resolved; remaining GLS-to-FIR differences are methodological consequences of finite impulse-response approximation, edge treatment, or both. If F1 fails, local-PSD conclusions must not be used to choose between filtering paths until the implementation mismatch is corrected.

If L1 passes on synthetic data but fails on real data, the evidence supports real-noise model inadequacy rather than a generic normalization error. If the primary 64-second model improves but remains outside `[0.8, 1.2]`, the result supports partial nonstationarity correction but not calibrated inference. If it passes both detectors and chronological blocks stabilize, it becomes a candidate Paper 1 real-noise result, subject to a fresh confirmatory run.

No event-significance claim follows directly from either experiment. The waveform is documented, but this remains a likelihood-geometry and matched-statistic validation study rather than gravitational-wave parameter estimation.
