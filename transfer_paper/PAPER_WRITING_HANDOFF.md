# Paper 1 experiment and evidence handoff

_Status date: 29 June 2026_

## Purpose and authority

This handoff tells a paper-writing agent what has actually been executed, what
has passed, what has failed, and what remains prospective. Inside this
standalone bundle, the source of truth for numerical claims is the copied
archived evidence under `data/gwosc/runs/`, the selected run under
`data/gwosc/current/`, and the multi-seed records under `data/synthetic/`.
Files under `data/derived/` make that evidence convenient to analyze but do
not replace the copied JSON and sweep records.

For a faster writing-oriented entry point, read `WRITING_AGENT_BRIEF.md` and
`MANUSCRIPT_EVIDENCE_MAP.md` before drafting prose. Also read
`PENDING_RESULT_PLACEHOLDERS.md`: pending, absent, or not-solidly-passed
experiments must be represented by explicit placeholders in working drafts,
not by speculative result prose.

The selected archived GWOSC baseline run is
`20260622T175125Z_b169c1f595a4`, testing commit
`b169c1f595a46d2701417ff4cbce292330817ad2`. It completed without operational
errors, passed Stage 0, passed 91 tests on the remote Linux system, and failed
the predeclared global-PSD scientific acceptance gate. The follow-up
shared-FIR and time-local PSD experiments now have evidence under
`data/gwosc/followup/`: shared-FIR implementation identity passed, while the
time-local PSD real-data gate failed after passing its stationary synthetic
control.

## What has been run and verified

### Synthetic validation ladder

The S1–S9 synthetic suite has been swept over ten deterministic seeds, with
held-out evaluation where required and interval summaries archived as CSV and
JSON. The suite verifies the optimal-filter/CRB normalization, the
EMPCA/optimal-filter bridge, an independently trained tied linear
autoencoder/EMPCA equivalence, the white-noise PCA control, the colored-noise
metric reversal, timing-jitter rank effects, covariance-estimation convergence,
residual chi-square calibration, and multichannel covariance effects.

The synthetic evidence supports controlled mathematical and implementation
claims. It does not by itself demonstrate calibration on nonstationary detector
noise. The notebooks should therefore keep synthetic and GWOSC results in
separate panels and separate prose paragraphs.

### Remote reproducibility

Stage 0 has passed on a clean remote Linux checkout. The archived workflow
records the commit, dependencies, environment, test output, core experiment
execution, table generation, and figure generation. This supports a
reproducibility statement about the software workflow, not a scientific claim
that every real-data gate passed.

### GWOSC data and PSD reference

The enhanced GWOSC run used 256 seconds of public GW150914 H1 and L1 strain at
4096 Hz. Official `H1_DATA` and `L1_DATA` coverage was present and required.
All 62 candidate four-second off-source windows were valid for both detectors,
and the event-centered window was covered. The cached raw-file checksums were
verified remotely.

The repository Hann-windowed, constant-detrended, bias-corrected median PSD was
compared with GWpy on identical calibration windows. For both detectors, the
median, fifth-percentile, and ninety-fifth-percentile PSD ratios were exactly
one at recorded precision, the relative L2 error was zero, and the maximum
absolute log-ratio was zero. This is a verified implementation-equivalence
result for the PSD estimator.

### Failed held-out real-noise calibration

The predeclared global-PSD acceptance gate failed. Random-split
`null_sigma_over_predicted` had median values 1.986 for H1 and 1.263 for L1.
H1 failed four of five random splits; L1 failed two of five. The chronological
block test was more severe: all five blocks failed for both detectors, with
medians 3.183 for H1 and 2.964 for L1.

This negative result is scientifically useful. It establishes that exact
agreement with a reference PSD implementation is not sufficient for calibrated
held-out template inference across this real-noise interval. It does not
identify a unique cause. Plausible causes include local spectral drift,
template-aligned narrow-band variation, non-Gaussian transients, and the
difference between ideal PSD-domain and finite-duration FIR statistics.

No evaluation windows were rejected by the predeclared broad RMS, 20–512 Hz
power, and crest-factor diagnostics. That observation only excludes large
excursions under those specific diagnostics; it does not prove stationarity.

### Whitening and matched-statistic diagnostics

The repository GLS score was strongly correlated with the repository
direct-whitened score, with correlations 0.974 for H1 and 0.997 for L1. Its
correlation with the earlier GWpy FIR score was 0.103 for H1 and 0.729 for L1.
Because those paths did not compute an identical finite-filter statistic, this
comparison is diagnostic rather than an equivalence failure. The newly
implemented shared-FIR experiment is designed to resolve that ambiguity.

### Shared-FIR implementation identity

The predeclared shared-FIR follow-up passed. It forced the explicit
FFT-convolution path and GWpy convolution path to use identical FIR
coefficients, edge trimming, and score normalization. The synthetic control and
both H1/L1 real-window sweeps passed the `1e-10` identity thresholds. This
resolves the narrow software implementation-equivalence question for the
shared finite-FIR statistic.

This result must not be broadened into a claim that the original PSD-domain
GLS statistic and the finite-duration FIR statistic are mathematically
equivalent. The GLS-to-FIR comparison remains a methodological sensitivity
diagnostic.

### Time-local PSD follow-up

The predeclared time-local PSD follow-up is a negative real-data result. The
stationary synthetic control passed for the primary 64-second local model, but
the H1 and L1 real-data primary gates failed with full local-model coverage.
The primary local model worsened score dispersion relative to the global
comparator on this interval: H1 global/local-64 standard deviations were
1.771/9.424, and L1 global/local-64 standard deviations were 7.288/8.827.
Chronological blocks remained unstable, with H1 primary block standard
deviations rising as high as 14.554 and L1 blocks reaching 11.944.

The archived template-projected and narrow-band diagnostics do not identify a
single obvious frequency-band fix. Template-projected observed/model ratios are
compressed near roughly 0.6-0.8 in the real runs and are weakly correlated with
absolute score excursions. The safe interpretation is therefore real-noise
model inadequacy or instability under the tested global/local PSD models, not
a generic normalization bug and not a parameter choice to tune into a pass.

### Injection and event values

The archived empirical injection SNRs were 2.396 for H1 and 6.435 for L1,
below the nominal target of eight because the null statistic was
under-calibrated. The archived event scores were 6.636 for H1 and −0.289 for
L1 under the old approximate template. These values may be reported only as
uncalibrated diagnostics. They must not be described as event significance,
false-alarm evidence, or detector sensitivity.

## Still awaiting evidence

The confirmatory GWOSC interval, CRESST/SCRESST validation, calibrated event
significance, and injection sensitivity remain pending. Any Results text for
those claims must use the placeholders in `PENDING_RESULT_PLACEHOLDERS.md`
until the corresponding evidence files exist inside this bundle and their
acceptance status has been reviewed.

## Paper sections that can be updated now

The Methods section can describe the S1–S9 validation ladder, seed-sweep and
held-out infrastructure, the Stage 0 reproducibility protocol, GWOSC data
provenance and official DATA coverage, the Hann/median PSD estimator, the
documented waveform, and the executed shared-FIR and local-PSD protocols.

The Results section can report the synthetic gate outcomes, exact PSD agreement
with GWpy, the positive shared-FIR implementation-identity result, and the
failed global-PSD and time-local-PSD held-out calibration. The failures should
be presented directly rather than hidden: random splits, chronological blocks,
and the primary 64-second local PSD model were not calibrated on the real
interval.

The Discussion can state that PSD-estimator correctness and shared-FIR software
identity do not guarantee stationary likelihood calibration on real detector
data. It should treat nonstationarity, transient contamination, and
template-projected model inadequacy as competing hypotheses, not solved
explanations.

The Limitations section should state that the current evidence covers one
256-second interval around one event, that overlapping split seeds are not
independent replications, that the archived event scores are uncalibrated, and
that confirmatory intervals are still required after model selection.

## Statements that must not appear yet

The paper must not claim a calibrated GW150914 significance, a validated
nominal SNR-eight injection sensitivity, equivalence of the original GLS and
GWpy FIR statistics, superiority of a 64-second local PSD, or general
calibration across GWOSC epochs. It must not tune the local radius, thresholds,
quality cuts, or windows after seeing the negative real-data result.

## Planned figure sequence

The synthetic paper figures should establish the controlled hierarchy first:
optimal-filter/CRB calibration, the independent linear-AE/EMPCA bridge,
colored-noise metric reversal, rank sensitivity, and covariance robustness.
The first GWOSC figure should show random and chronological
`null_sigma_over_predicted` values with the acceptance bands visible. A second
panel can show exact PSD agreement alongside the non-equivalent matched-score
correlations, clearly labeling the latter as diagnostic.

The filtering figure should display implementation identity error over the
FIR-duration/edge-trim grid and separately display the original-GLS/shared-FIR
correlation. The local-noise figure should compare global and local held-out
score standard deviations, followed by chronological, template-projected, and
narrow-band diagnostics. These follow-up figures must retain the predeclared
primary setting visually rather than highlighting the best post hoc setting.

## Next controlled loop

The next operation is to write the negative real-data result into the evidence
map and manuscript draft without changing the frozen L1 choices. Any new
hypothesis, quality cut, radius choice, or frequency exclusion must be frozen
in a new protocol and tested on untouched evidence. A confirmatory GWOSC
interval is still required before any broader public-data calibration claim.
