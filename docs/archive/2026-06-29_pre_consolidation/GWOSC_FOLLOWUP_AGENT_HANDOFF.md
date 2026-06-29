# GWOSC follow-up agent handoff

Purpose: run the two predeclared GWOSC follow-up experiments after the failed
global-PSD calibration run, synchronize their evidence, and interpret the
result without changing thresholds after inspection.

## Current state

The previous GWOSC run reached execution successfully but failed scientific
acceptance. This is the baseline, not a code crash:

- Stage 0 remote reproducibility passed.
- GWOSC data files, checksums, and official data-quality coverage passed.
- The repository Hann/median PSD agreed with GWpy on identical windows.
- The held-out null calibration failed for both detectors.
- Event and injection scores are diagnostic only until null calibration passes.

The immediate follow-up experiments are implemented but the current evidence
bundle does not contain:

- `transfer_paper/data/gwosc/followup/filter_equivalence.json`
- `transfer_paper/data/gwosc/followup/time_local_noise.json`

## Important preflight

Both follow-up configs require the documented public GW150914 waveform:

```text
/ceph/dwong/paper1_dataset/gwosc/raw/GW150914/waveforms/fig2-unfiltered-waveform-H.txt
```

If either follow-up fails with:

```text
FileNotFoundError: documented waveform is not cached ...
```

that is a missing cache file, not a scientific failure. Fetch it first.

Preferred downloader command:

```bash
uv run python scripts/download/download_gwosc.py \
  --config configs/gwosc/gw150914_smoke.yaml \
  --download \
  --timeout 900
```

Confirm:

```bash
ls -lh /ceph/dwong/paper1_dataset/gwosc/raw/GW150914/waveforms/fig2-unfiltered-waveform-H.txt
```

If the downloader cannot fetch the waveform, use a direct manual fetch:

```bash
mkdir -p /ceph/dwong/paper1_dataset/gwosc/raw/GW150914/waveforms
curl -L \
  https://www.gw-openscience.org/s/events/GW150914/P150914/fig2-unfiltered-waveform-H.txt \
  -o /ceph/dwong/paper1_dataset/gwosc/raw/GW150914/waveforms/fig2-unfiltered-waveform-H.txt
```

Do not switch datasets or modify manuscript claims because of this cache error.

## Environment check

Use the documented GWOSC dependency set:

```bash
uv sync --extra dev --extra gwosc
uv run --extra dev --extra gwosc pytest -q
```

The local reference state for this repo is `106 passed` with GWpy installed.
If tests fail only because `gwpy` is missing, rerun with the `gwosc` extra
before diagnosing code.

## Run commands

Run the shared-FIR statistic equivalence experiment:

```bash
uv run python scripts/run_experiment.py \
  --config configs/gwosc/filter_statistic_equivalence.yaml
```

Run the time-local PSD experiment:

```bash
uv run python scripts/run_experiment.py \
  --config configs/gwosc/time_local_noise.yaml
```

Each command writes a run record under `results/metrics/` unless an explicit
`--output` is provided. Preserve the generated `.json`, `.config.yaml`, and
`.log` files.

## Expected evidence after a complete remote run

Synchronize or copy the follow-up outputs into the evidence bundle as:

```text
transfer_paper/data/gwosc/followup/filter_equivalence.json
transfer_paper/data/gwosc/followup/filter_equivalence.config.yaml
transfer_paper/data/gwosc/followup/time_local_noise.json
transfer_paper/data/gwosc/followup/time_local_noise.config.yaml
```

Then refresh derived tables/figures using the existing transfer-paper scripts
if this is part of the evidence packaging workflow.

## Acceptance criteria

### F1: shared-FIR statistic equivalence

Config:

```text
configs/gwosc/filter_statistic_equivalence.yaml
```

Scientific question: do the explicit FFT-convolution path and the GWpy
convolution path produce the same shared FIR matched statistic when forced to
use identical FIR coefficients and score normalization?

Primary acceptance:

- synthetic control must run;
- real H1/L1 windows must run;
- every predeclared FIR-duration and edge-trim setting must have:
  - `max_abs_identity_difference <= 1e-10`
  - `max_identity_relative_l2 <= 1e-10`

Interpretation:

- If F1 passes, software application of the shared FIR statistic is resolved.
  Any remaining GLS-to-FIR discrepancy is a methodological/filter-definition
  difference, not an implementation mismatch between the two shared-FIR paths.
- If F1 fails, fix the filter/statistic implementation before using local-PSD
  conclusions to explain the GWOSC calibration failure.
- Do not treat original PSD-domain GLS versus finite-FIR differences as a
  pass/fail gate; those are diagnostic sensitivity outputs.

### L1: time-local PSD model

Config:

```text
configs/gwosc/time_local_noise.yaml
```

Scientific question: does a predeclared local PSD model reduce held-out
template-score dispersion relative to the failed global-PSD baseline?

Primary model:

- local radius: `64` seconds
- required primary local coverage fraction: `>= 0.9`
- accepted score standard-deviation interval: `[0.8, 1.2]`

Primary acceptance:

- stationary synthetic control primary score std must be in `[0.8, 1.2]`;
- real H1 primary local score std must be in `[0.8, 1.2]`;
- real L1 primary local score std must be in `[0.8, 1.2]`;
- primary 64-second model coverage must be at least `0.9`;
- chronological/template-projected/narrow-band diagnostics must be archived.

Interpretation:

- If synthetic passes but real fails, the evidence supports real-noise model
  inadequacy rather than a generic normalization bug.
- If the 64-second local model improves but remains outside `[0.8, 1.2]`,
  report partial diagnostic improvement only. Do not claim calibrated inference.
- If the 64-second local model passes both detectors and chronological blocks
  are stable, treat it as a candidate Paper 1 real-noise result, but require a
  fresh confirmatory GWOSC interval before making broad public-data claims.
- Do not replace the primary 64-second model with 32 or 96 seconds after
  seeing results. Those radii are sensitivity analyses.

## Manuscript decision rules

Use evidence-backed statements only after the corresponding JSON exists and
has been reviewed.

Allowed before follow-up evidence exists:

```text
The shared-FIR and time-local PSD analyses were predeclared as follow-up tests.
```

Allowed if the current global-PSD result remains the only completed real-data
evidence:

```text
The global-PSD GWOSC stress test failed its held-out null calibration gate.
```

Prohibited unless a passed and reviewed result exists:

```text
The method detects GW150914 with calibrated significance.
The nominal SNR-eight injection sensitivity is validated.
The 64-second local PSD model corrected the calibration failure.
The shared-FIR experiment proves all repository and GWpy statistics are equivalent.
The method is calibrated on GWOSC data in general.
```

## Dataset-switch decision

Do not switch to another dataset because of:

- missing waveform cache;
- missing optional dependency;
- failed global-PSD calibration alone;
- pending follow-up evidence.

Switch or broaden only after the F1/L1 fork is resolved:

- F1 failure means fix code/method first.
- F1 pass plus L1 real failure means update the declaration to a negative
  real-noise result and keep GWOSC as a stress test.
- F1 pass plus L1 pass means run a confirmatory untouched GWOSC interval before
  claiming general GWOSC calibration.
- CRESST remains the next independent real-data domain after the GWOSC
  diagnostic loop is settled.
