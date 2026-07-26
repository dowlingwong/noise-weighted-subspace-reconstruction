# Noise generator and evidence provenance audit (EXP-00)

Generated: 2026-07-26T02:37:07.899240+00:00

## Generator lineage per synthetic experiment

| Experiment | Builder | Sampler class | FFT-based | Notes |
| --- | --- | --- | --- | --- |
| S0 | make_rank1_pulse_dataset (src/noise_geometry/synthetic/benchmarks.py) | fourier_gaussian_real_imag_bins | True | Interior rFFT bins are independent circular complex Gaussians ((re+1j*im)/sqrt(2)*scale); Nyquist bin (even N) is correctly rescaled by sqrt... |
| S1 | make_rank1_pulse_dataset (src/noise_geometry/synthetic/benchmarks.py) | fourier_gaussian_real_imag_bins | True | Interior rFFT bins are independent circular complex Gaussians ((re+1j*im)/sqrt(2)*scale); Nyquist bin (even N) is correctly rescaled by sqrt... |
| S2 | make_rank1_pulse_dataset (src/noise_geometry/synthetic/benchmarks.py) | fourier_gaussian_real_imag_bins | True | Interior rFFT bins are independent circular complex Gaussians ((re+1j*im)/sqrt(2)*scale); Nyquist bin (even N) is correctly rescaled by sqrt... |
| S3 | make_s3_dataset (src/noise_geometry/experiments/synthetic.py) | diagonal_gaussian_real_domain | False | Per-sample independent (diagonal-covariance) Gaussian noise added directly in the time domain; no FFT or covariance matrix involved.... |
| S4 | _rankk_signal + rng.normal (src/noise_geometry/experiments/synthetic.py) | diagonal_gaussian_real_domain | False | Per-sample independent (diagonal-covariance) Gaussian noise added directly in the time domain; no FFT or covariance matrix involved.... |
| S5 | _metric_reversal_dataset (src/noise_geometry/experiments/synthetic.py) | fourier_gaussian_real_imag_bins | True | Interior rFFT bins are independent circular complex Gaussians ((re+1j*im)/sqrt(2)*scale); Nyquist bin (even N) is correctly rescaled by sqrt... |
| S6 | run_s6_timing_rank_sweep, AR(1) covariance (src/noise_geometry/experiments/synthetic.py) | exact_multivariate_gaussian_from_covariance | False | Direct draw from an explicit covariance matrix (AR(1) or block covariance built in-experiment), not routed through any FFT/PSD sampler at al... |
| S7 | run_s7_covariance_robustness, AR(1) covariance (src/noise_geometry/experiments/synthetic.py) | exact_multivariate_gaussian_from_covariance | False | Direct draw from an explicit covariance matrix (AR(1) or block covariance built in-experiment), not routed through any FFT/PSD sampler at al... |
| S8 | run_s8_residual_calibration, AR(1) covariance (src/noise_geometry/experiments/synthetic.py) | exact_multivariate_gaussian_from_covariance | False | Direct draw from an explicit covariance matrix (AR(1) or block covariance built in-experiment), not routed through any FFT/PSD sampler at al... |
| S9 | run_s9_multichannel, block covariance (src/noise_geometry/experiments/synthetic.py) | exact_multivariate_gaussian_from_covariance | False | Direct draw from an explicit covariance matrix (AR(1) or block covariance built in-experiment), not routed through any FFT/PSD sampler at al... |

See the JSON companion for the full untruncated notes per sampler, including the confirmed DC-bin scaling defect in `src/noise_geometry/noise/synthetic.py:generate_colored_noise`.

## Manuscript evidence-map cross-check

- Claims parsed from `transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md`: 19
- Cited paths that failed to resolve: 0

## Artifact inventory summary

- Total artifacts inventoried: 36
- Superseded (must not be cited as evidence going forward): 8
- Frozen primary: 28

Full per-artifact rows are in `synthetic_config_provenance.csv` and `noise_generator_provenance.json`.
