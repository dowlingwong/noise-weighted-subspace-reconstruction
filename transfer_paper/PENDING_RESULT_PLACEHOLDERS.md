# Placeholder policy for pending or weak experiments

This file is mandatory guidance for any paper-writing agent using this bundle.
If an experiment has not been run or has not been synchronized into this
bundle, the manuscript draft must use an explicit placeholder rather than
writing the result as if it exists. If an experiment has been synchronized and
failed its predeclared acceptance gate, write the negative result directly and
do not tune it into a pass.

Placeholders are allowed in working drafts only. They must be resolved before
submission. A final manuscript must either replace the placeholder with
evidence-backed text or remove the claim.

## General rule

Use evidence-backed prose only for results that are present in
`data/gwosc/current/`, `data/gwosc/runs/`, `data/synthetic/`, or
`data/derived/`. If a required JSON/CSV/figure is absent, write a visible
placeholder in the manuscript draft.

Recommended placeholder format:

```text
[PLACEHOLDER — pending evidence: <experiment name>. Insert result only after
<required file> exists in transfer_paper and the acceptance status is reviewed.]
```

Do not convert a pending placeholder into speculative prose. Do not write
"expected", "should", "likely", or "we anticipate" as a substitute for data.

## Resolved follow-up experiments

### Shared-FIR filtering/statistic equivalence

Current state: evidence is present and reviewed. The shared-FIR implementation
identity gate passed for the stationary synthetic control and H1/L1 real
windows.

Evidence:

- `data/gwosc/followup/filter_equivalence.json`
- regenerated `data/derived/figure_index.csv`
- rendered `figures/gwosc_filter_equivalence.*`
- reviewed acceptance status for synthetic control, H1, and L1

Allowed wording:

```text
The shared-FIR follow-up passed its implementation-identity gate: explicit FFT
convolution and GWpy convolution agree when both paths use the same FIR
coefficients and score normalization.
```

Prohibited wording:

```text
The original PSD-domain GLS statistic and the finite-duration GWpy FIR
statistic are mathematically equivalent.
```

### Time-local PSD modelling

Current state: evidence is present and reviewed. The stationary synthetic
control passed, but the predeclared primary 64-second local PSD model failed
H1 and L1 real-data acceptance.

Evidence:

- `data/gwosc/followup/time_local_noise.json`
- regenerated local/global PSD summary tables
- rendered `figures/gwosc_time_local_psd.*`
- reviewed stationary synthetic control, primary 64-second coverage, H1/L1
  calibration, chronological blocks, template-projected diagnostics, and
  narrow-band diagnostics

Allowed wording:

```text
The time-local PSD follow-up passed its stationary synthetic control but failed
real H1/L1 acceptance: the primary 64-second local PSD model did not calibrate
held-out score dispersion on this GWOSC interval.
```

Prohibited wording:

```text
```

Do not replace the primary radius, thresholds, quality cuts, or windows after
seeing this result. Any new PSD model or data-quality rule requires a new
frozen protocol and untouched evidence.

## Required placeholders by experiment

### Confirmatory GWOSC interval

Current state: not run. The current real-data evidence covers one
GW150914-centered 256-second interval.

Required evidence before writing a general GWOSC calibration claim:

- a predeclared confirmatory interval or event set
- archived run evidence under `data/gwosc/runs/`
- pass/fail summary in `data/derived/`
- no parameter tuning after seeing the current interval's failure modes

Draft placeholder:

```text
[PLACEHOLDER — pending confirmatory GWOSC evidence. Insert confirmatory
held-out real-noise calibration results only after an untouched interval/event
set is run under the frozen protocol and synchronized into transfer_paper.]
```

Allowed current wording:

```text
The present GWOSC analysis is a stress test on one public 256-second interval
around GW150914 and should be treated as interval-specific.
```

Prohibited current wording:

```text
The method is calibrated on GWOSC data in general.
```

### CRESST / SCRESST validation

Current state: not run in the current evidence bundle.

Required evidence before writing a CRESST result:

- validated data schema and provenance
- archived preprocessing logs and checksums
- predeclared calibration/evaluation split
- experiment JSON and derived tables copied into this bundle
- pass/fail interpretation reviewed against acceptance criteria

Draft placeholder:

```text
[PLACEHOLDER — pending CRESST evidence. Insert CRESST/SCRESST validation only
after the data schema, preprocessing, calibration split, experiment outputs,
and acceptance review are synchronized into transfer_paper.]
```

Allowed current wording:

```text
CRESST is a planned second real-data domain after the GWOSC verification loop,
because its schema and preprocessing assumptions still need independent
validation.
```

Prohibited current wording:

```text
The method has been validated on CRESST.
```

### Event significance and injection sensitivity

Current state: not validated. The archived event and injection scores are
diagnostic because the global-PSD null calibration failed.

Required evidence before writing significance or sensitivity claims:

- a passed real-noise null calibration gate
- a frozen event/injection scoring protocol using the documented waveform
- false-alarm or uncertainty calibration based on held-out/confirmatory data
- synchronized evidence and reviewed acceptance status

Draft placeholder:

```text
[PLACEHOLDER — pending calibrated event/sensitivity evidence. Insert
GW150914 significance or injection-sensitivity claims only after the real-noise
null calibration gate passes and the event/injection scoring protocol is
frozen, rerun, synchronized, and reviewed.]
```

Allowed current wording:

```text
The archived event and injection scores are retained as diagnostics only,
because the corresponding held-out null calibration gate failed.
```

Prohibited current wording:

```text
The method detects GW150914 with calibrated significance or validates nominal
SNR-eight sensitivity.
```

## How to write draft sections safely

Methods sections may describe implemented or predeclared protocols as long as
the tense is precise. Use past tense only for executed experiments and present
or future-oriented wording for planned analyses.

Results sections must not include pending experiments as prose results. Put the
placeholder exactly where the result would go, with the required evidence file
named.

Discussion sections may explain why pending experiments are needed, but must
not treat them as completed explanations. For example, local spectral drift is
a hypothesis being tested, not a demonstrated cause of the current GWOSC
failure.

Abstract and conclusion sections should omit pending-result placeholders unless
the draft is explicitly an internal working draft. For a submission draft, any
unresolved placeholder means the corresponding claim must be removed.
