# GWOSC Result And Interpretation

Last consolidated: 2026-06-29.

## Bottom Line

The GWOSC work is currently a negative real-noise stress test. It should be
frozen as evidence for the current revision.

The run did not fail because the basic code path is broken. It failed the
scientific calibration gate: the tested public GWOSC interval is not calibrated
by the current global-PSD or predeclared local-PSD noise models.

## What Succeeded

- GWOSC data files, checksums, and official data-quality coverage were recorded.
- The repository Hann/median PSD estimator matched GWpy on identical windows.
- The shared-FIR follow-up passed its implementation-identity gate.
- The time-local PSD synthetic control passed.
- Follow-up evidence, derived CSV summaries, and figures were synchronized into
  `transfer_paper/`.

## What Failed

- The original global-PSD held-out null calibration failed.
- The predeclared primary 64-second local PSD model failed real-data
  calibration for H1 and L1.
- Local PSD did not repair the GWOSC calibration failure; in the synchronized
  follow-up, it worsened score dispersion relative to the global comparator.
- Chronological local-64 blocks remained unstable.
- Template-projected and narrow-band diagnostics did not identify a single
  obvious narrow-band fix.

## Key Numbers

Shared-FIR identity gate, threshold `1e-10`:

| Group | Max abs score difference | Max relative L2 score difference | Verdict |
| --- | ---: | ---: | --- |
| Synthetic | `4.98e-15` | `1.47e-15` | passed |
| H1 | `1.60e-12` | `1.70e-13` | passed |
| L1 | `1.45e-12` | `3.04e-13` | passed |

Time-local PSD primary local-64 gate, target score std `[0.8, 1.2]`:

| Group | Global std | Local-64 std | Coverage | Verdict |
| --- | ---: | ---: | ---: | --- |
| Synthetic | `1.075` | `1.148` | `1.0` | passed |
| H1 | `1.771` | `9.424` | `1.0` | failed |
| L1 | `7.288` | `8.827` | `1.0` | failed |

## Correct Interpretation

The shared-FIR result resolves a narrow software identity question: when the
repository and GWpy paths are forced to use the same FIR coefficients and score
normalization, their scores agree.

It does not prove that the original PSD-domain GLS statistic and a finite-FIR
statistic are mathematically identical.

The local-PSD result is negative on real data: the stationary synthetic control
passed, but H1 and L1 real-data calibration failed. That supports real-noise
model inadequacy or interval instability rather than a generic normalization
bug.

## Manuscript-Safe Text

Use this as the base paragraph:

```text
As a public real-noise stress test, we applied the GWOSC workflow to a
GW150914-centered interval. The PSD estimator matched GWpy on identical
calibration windows, and a shared-FIR follow-up showed that explicit FFT
convolution and GWpy convolution agree when given identical FIR coefficients
and score normalization. However, the held-out real-noise calibration gate
failed under both the original global-PSD model and the predeclared primary
64-second local-PSD model. The stationary synthetic control for the local-PSD
experiment passed, so the failure is best interpreted as real-noise/model
inadequacy for this interval rather than a generic normalization error. We
therefore retain GWOSC event and injection scores as diagnostics only and make
no calibrated GWOSC significance or sensitivity claim.
```

## Wording To Avoid

- Do not claim the method detects GW150914 with calibrated significance.
- Do not claim nominal SNR-eight injection sensitivity is validated on GWOSC.
- Do not claim local PSD fixed the calibration failure.
- Do not claim GWOSC validates the method on public real data.
- Do not claim the original PSD-domain GLS statistic and finite-FIR statistic
  are mathematically equivalent.

## Evidence Files

Primary records:

- `transfer_paper/data/gwosc/followup/filter_equivalence.json`
- `transfer_paper/data/gwosc/followup/filter_equivalence.config.yaml`
- `transfer_paper/data/gwosc/followup/time_local_noise.json`
- `transfer_paper/data/gwosc/followup/time_local_noise.config.yaml`

Derived summaries:

- `transfer_paper/data/derived/gwosc_filter_equivalence_summary.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_summary.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_blocks.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_spectral_summary.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_top_windows.csv`

Figures:

- `transfer_paper/figures/gwosc_filter_equivalence.png`
- `transfer_paper/figures/gwosc_time_local_psd.png`

## Diagnostic Framing

These diagnostics sharpen the negative result without changing its status; they
remain diagnostics only and imply no calibrated significance claim. Distilled
from the NPML experiment plan (`NPML/plan.md`).

- Residual whitening ratio: report `mean(|r_f|^2 / PSD_f)` versus frequency.
  Under correct calibration it sits near 1; the bands where it departs localize
  which part of the spectrum breaks the held-out gate.
- Detection-score view: report the noise-weighted residual reduction
  `chi2(noise-only) - chi2(signal-subspace)` and, where injections exist, the
  trigger efficiency at fixed background FPR. This shows the physics-facing
  consequence of the calibration failure rather than an amplitude ratio alone.
- Negative controls: confirm the failure is calibration, not a coding artifact,
  by checking that a wrong PSD, permuted time bins, or a random
  weighted-orthonormal basis degrade the scores as expected while the passing
  implementation checks (PSD match, shared-FIR identity) still hold.

## If GWOSC Is Continued

Do not modify the completed experiment to obtain a pass. Any future GWOSC work
should be a new predeclared Stage 3b, for example:

- longer off-source interval with more background windows;
- stricter official veto/category handling;
- robust PSD or shrinkage model with frozen hyperparameters;
- explicit glitch or template-projected outlier model;
- larger background for false-alarm calibration;
- untouched confirmatory interval or event set.
