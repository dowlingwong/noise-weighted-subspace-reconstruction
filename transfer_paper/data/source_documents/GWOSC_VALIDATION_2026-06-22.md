# GWOSC GW150914 validation record — 2026-06-22

_Scientific and reproducibility record for remote run `20260622T164907Z_f541c542f778`_

---

## 📋 Executive conclusion

The first controlled 256-second H1/L1 GWOSC verification completed successfully at the computational level but failed its predeclared scientific acceptance gate. Stage 0 passed on a clean remote Linux checkout, all 89 tests passed, the GWOSC files matched their recorded checksums, and both the GWpy reference command and the experiment command exited successfully. The repository PSD estimator agreed exactly with GWpy’s median-Welch PSD on the same calibration windows. Nevertheless, held-out matched-filter amplitudes were more dispersed than the PSD-derived prediction, especially for H1. The correct status of this run is therefore `failed_acceptance`, not an execution failure and not a validated detector result.

This result establishes three points. The remote workflow is reproducible, the Hann-windowed bias-corrected median PSD calculation is implemented consistently with GWpy, and the current global-PSD model does not calibrate the held-out real-noise statistic across deterministic random splits. It does not establish the significance of the GW150914 event score, real-data injection sensitivity at nominal signal-to-noise ratio eight, or agreement between the repository’s direct-rFFT whitening and GWpy’s inverse-spectrum-FIR whitening.

## 🧪 Provenance and execution

The run used repository commit `f541c542f778c474432dea4ca812da433ef81e8e` on host `portal1`. The operating platform was Linux kernel `5.14.0-687.5.3.el9_8` on `x86_64` with glibc 2.34. The bootstrap interpreter was Python 3.9.25, while the synchronized project environment used Python 3.13.13. The environment manager was `uv 0.11.14`. No GPU was available or required. The external data root was `/ceph/dwong/paper1_dataset`.

Stage 0 began and ended with a clean Git tree. The commands `uv sync --extra dev --extra gwosc`, `uv run pytest -q`, `uv run python scripts/run_all_core.py`, `uv run python scripts/make_tables.py`, and `uv run python scripts/make_all_figures.py` all returned zero. Pytest reported 89 passed tests and 11 dependency deprecation warnings. The warnings came from GWpy plotting scale registration and did not affect the numerical run.

The archived evidence is stored under [`evidence/gwosc/20260622T164907Z_f541c542f778`](../evidence/gwosc/20260622T164907Z_f541c542f778). Every file listed in `SHA256SUMS` was rechecked after synchronization and passed. The authoritative experiment record is [`gwosc/experiment.json`](../evidence/gwosc/20260622T164907Z_f541c542f778/gwosc/experiment.json), and the independent reference output is [`gwosc/gwpy_reference.json`](../evidence/gwosc/20260622T164907Z_f541c542f778/gwosc/gwpy_reference.json).

## ⚙️ Data and method

The dataset consisted of 256 seconds of public GWOSC strain around GW150914 for H1 and L1, sampled at 4096 Hz. The cached files were `GW150914_H1_256s.npz` and `GW150914_L1_256s.npz`, with SHA-256 digests `19e0230532c2f612915780bc499a9fb78ddb867ad23630fab25fe85edb05e01e` and `fe539c2996d5dd85a195e3266492bd756321b33c7313874fbecd44111de49cd1`, respectively. The source metadata identified GWOSC 0.8.2, GWpy 4.0.1, and the O1 4096-Hz HDF5 archives used for both detectors.

The analysis divided the strain into non-overlapping four-second windows and excluded the event-centered guard region. Each random split used 46 calibration windows and 16 held-out evaluation windows. The PSD was estimated from the calibration windows with a Hann window, constant detrending, no overlap, and SciPy/GWpy’s finite-sample bias-corrected median aggregation. Frequencies below 20 Hz received zero inverse-PSD weight. The predeclared validation used split seeds 150914 through 150918, required every split ratio to fall in `[0.5, 1.5]`, and required the median ratio to fall in `[0.8, 1.2]`.

The primary calibration statistic was

```text
null_sigma_over_predicted
    = standard deviation of held-out fitted amplitudes
      / PSD-derived amplitude standard deviation.
```

A value near one means that the PSD-derived uncertainty predicts the observed held-out amplitude spread. Values above one indicate underestimated uncertainty, unmodelled nonstationarity, non-Gaussian transients, template-aligned structure, or some combination of these effects.

The run’s calibration quality screen measured broad-band RMS, power from 20 to 512 Hz, and crest factor. It rejected no calibration windows for either detector. The thresholds were deliberately broad: absolute robust z-score above six or crest factor above twenty. This absence of rejected windows does not prove that every window was stationary; it only shows that none crossed those specific broad diagnostics.

## 📊 Primary results

The random-split calibration gate failed for both detectors. H1 passed one of five splits, with a median ratio of 1.986 and a maximum of 7.166. L1 passed three of five splits, but its median of 1.263 exceeded the predeclared upper bound of 1.2 and two splits were substantially over-dispersed.

| Detector | Split seed | Calibration windows | Evaluation windows | `null_sigma_over_predicted` | Predeclared split verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| H1 | 150914 | 46 | 16 | 3.338 | Fail |
| H1 | 150915 | 46 | 16 | 1.803 | Fail |
| H1 | 150916 | 46 | 16 | 7.166 | Fail |
| H1 | 150917 | 46 | 16 | 1.396 | Pass |
| H1 | 150918 | 46 | 16 | 1.986 | Fail |
| L1 | 150914 | 46 | 16 | 1.243 | Pass |
| L1 | 150915 | 46 | 16 | 3.093 | Fail |
| L1 | 150916 | 46 | 16 | 1.263 | Pass |
| L1 | 150917 | 46 | 16 | 1.209 | Pass |
| L1 | 150918 | 46 | 16 | 2.379 | Fail |

| Detector | Minimum ratio | Median ratio | Maximum ratio | Passing splits |
| --- | ---: | ---: | ---: | ---: |
| H1 | 1.396 | 1.986 | 7.166 | 1/5 |
| L1 | 1.209 | 1.263 | 3.093 | 3/5 |

The primary split, seed 150914, predicted amplitude standard deviations of `1.340 × 10⁻²³` for H1 and `1.509 × 10⁻²³` for L1. The observed held-out standard deviations were `4.473 × 10⁻²³` and `1.877 × 10⁻²³`, producing ratios of 3.338 and 1.243. The H1 null-score distribution also had a nonzero sample mean of −0.909 over 16 windows, whereas L1’s mean was 0.024. These values reinforce the conclusion that H1’s primary held-out set is not described by the nominal stationary-noise model.

## 🔍 Independent GWpy comparison

The repository PSD and GWpy PSD were identical on the same calibration data for both detectors. The frequency grids had zero maximum difference, the relative L2 PSD error was zero, and the PSD ratio quantiles were exactly one within recorded precision. This rules out a mismatch in the implemented Hann/median PSD calculation as the explanation for the failed amplitude gate.

The time-domain whitening paths did not agree comparably well. The repository applied direct rFFT division by the ASD, while GWpy applied a finite-duration inverse-spectrum FIR filter. After the configured edge trim, the repository’s pooled whitened standard deviation exceeded GWpy’s by factors of 1.406 for H1 and 1.357 for L1. The pooled correlations were only 0.134 and 0.036.

| Detector | PSD relative L2 error | Repository whitened std | GWpy whitened std | Std ratio | Interior correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 | 0.000 | 3.344 | 2.378 | 1.406 | 0.134 |
| L1 | 0.000 | 5.673 | 4.182 | 1.357 | 0.036 |

These whitening diagnostics are informative but were not predeclared as an acceptance gate. The two filters are not pointwise identical operations, and the old record did not compare the same normalized matched-filter statistic through both paths. The low correlations therefore identify an unresolved methodological comparison rather than proving which whitening path is correct.

## 📈 Injection and event diagnostics

The paired injection algebra behaved exactly as implemented. Adding the template to an evaluation trace and subtracting the fitted amplitude from the same noise realization returned a paired mean score of eight with numerical scatter near machine precision. This verifies linearity and normalization of the paired calculation, but it is not an independent sensitivity test because the same noise contribution appears before and after injection.

The unpaired empirical sensitivity was substantially weaker than the nominal target for H1. Dividing the injected amplitude by the empirical held-out amplitude spread gave an effective SNR of 2.396 for H1 and 6.435 for L1, rather than eight. This is the direct consequence of the failed null calibration.

| Detector | Nominal injection SNR | Empirical injection SNR | Unpaired recovered-score mean | Event score |
| --- | ---: | ---: | ---: | ---: |
| H1 | 8.000 | 2.396 | 7.091 | 6.636 |
| L1 | 8.000 | 6.435 | 8.024 | −0.289 |

The event scores cannot be assigned calibrated false-alarm or significance interpretations from this run. In particular, the H1 score of 6.636 must not be described as a calibrated six-sigma observation because the corresponding null-score standard deviation was 3.338 rather than one in the primary split, and the split-to-split behavior was unstable. The event template was also an approximate chirp rather than a documented public waveform model, which remains a separate Stage 3 limitation.

## 🧠 Interpretation and competing explanations

The strongest supported conclusion is that a single global PSD estimated from randomly selected off-source windows is insufficient to calibrate this matched-template amplitude statistic over the full 256-second record. The failure is detector dependent and split dependent. H1 shows severe instability, while L1 is closer to the nominal model but still fails the predeclared aggregate criterion.

Several explanations remain plausible. The data may be nonstationary on the scale of the record, so calibration and evaluation subsets can have different local spectra. Individual evaluation windows may contain broad or narrow transients that evade the current RMS, band-power, and crest-factor thresholds. Some disturbances may align strongly with the approximate chirp template while contributing little to broad-band quality measures. The random split design also reuses the same finite set of 62 windows across seeds, so a small number of influential windows could affect multiple splits. Finally, direct-rFFT and inverse-spectrum-FIR whitening behave differently on these short real-data windows, although the exact PSD agreement shows that this is not a PSD-estimator discrepancy.

The existing evidence cannot identify a unique causal window because it records only aggregate null spread for each split. Associating high split ratios with windows shared by those splits is exploratory and confounded by the fact that every split also produces a different calibration PSD. Such associations must not be presented as confirmed glitches.

## ⚠️ Limitations and claim boundaries

Official GWOSC data-quality segments were not stored in the archived metadata for this run. The raw strain came from the official open-data files, but the analysis did not prove that every four-second calibration and evaluation window lay fully inside the detector’s official `DATA` segment. The next run must record and enforce those intervals.

Quality diagnostics were applied only to calibration candidates. The held-out evaluation windows that determined the null spread did not receive calibration-referenced quality scores. No per-window null amplitudes or matched-filter scores were archived, and split-specific indices were not stored for every seed. Consequently, the run supports an aggregate failure diagnosis but not a window-level attribution.

The five random splits are deterministic and reproducible but not independent experiments because they overlap. Their spread is a robustness diagnostic, not a five-replication confidence interval. Chronological blocked evaluation with locally selected calibration windows is needed to test whether temporal nonstationarity explains the failure.

The GWpy comparison established PSD equality but did not compare an identical normalized matched statistic through the repository and GWpy whitening paths. The event template remains approximate. No paper-facing event or sensitivity claim should be made until these limitations are addressed.

The runbook’s optional compact-summary extractor failed after the experiment because it expected `null_gate["split_results"]`, while the experiment schema uses `null_gate["splits"]`. This did not affect the experiment, reference output, or archived full JSON. It was an evidence-packaging defect and has been corrected.

## 🔧 Patches following this result

The subsequent patch preserves the failed acceptance thresholds and expands the next run’s evidence. Every random and chronological split now records calibration and evaluation indices, window start times, predicted amplitude uncertainty, per-window null amplitudes, normalized null scores, and calibration-referenced evaluation quality diagnostics. Evaluation-window filtering is reported only as a sensitivity analysis and is explicitly excluded from acceptance to avoid post-selection.

The downloader now records official GWOSC `H1_DATA` and `L1_DATA` segments. The experiment can require those records, excludes off-source windows that are not fully covered, and checks event-window coverage. The GW150914 configuration requires official data-quality metadata.

The next run also adds five chronological evaluation blocks with 36 nearest-time calibration candidates per block, while retaining the requirement that at least 32 survive quality checks. These blocked splits use the same strict per-split and median ratio bounds and are required for acceptance. This design is stricter than the previous random-split-only gate and is intended to expose temporal nonstationarity rather than hide it.

The GWpy reference path now computes matched-template scores through three routes: the repository GLS statistic, the repository direct-rFFT whitened statistic, and the GWpy inverse-spectrum-FIR whitened statistic. Their score arrays, correlations, and relative differences are diagnostic and are not yet used for acceptance. This comparison will determine whether the observed whitening disagreement affects the scientific statistic rather than only the pointwise whitened waveform.

Regression tests now cover calibration-referenced evaluation-window quality scoring, official data-quality coverage, detailed per-window split records, chronological blocked splits, and the matched-statistic reference output.

## ✅ Current decision

Stage 0 is complete. GWOSC-A remains failed and diagnostic, not validated. The failed record must be retained as the baseline against which the enhanced diagnostic rerun is compared. The acceptance bounds remain unchanged. The immediate next operation is a fresh remote run from the patched commit, beginning with a metadata refresh so official data-quality segments are recorded, followed by the reference check and the GWOSC experiment. Only after the new per-window and blocked-split evidence is available should the project decide whether the dominant issue is transient contamination, local spectral drift, whitening implementation, or an inadequate template/statistical model.
