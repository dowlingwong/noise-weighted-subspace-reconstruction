# Paper 1 experiment and evidence handoff

_Status date: 23 June 2026_

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

The most recent archived GWOSC run available when this bundle was assembled is
`20260622T175125Z_b169c1f595a4`, testing commit
`b169c1f595a46d2701417ff4cbce292330817ad2`. It completed without operational
errors, passed Stage 0, passed 91 tests on the remote Linux system, and failed
the predeclared scientific acceptance gate. The subsequent filtering and
time-local-noise implementation passes 95 local tests but has not yet produced
remote evidence. Those two states must not be conflated.

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

### Injection and event values

The archived empirical injection SNRs were 2.396 for H1 and 6.435 for L1,
below the nominal target of eight because the null statistic was
under-calibrated. The archived event scores were 6.636 for H1 and −0.289 for
L1 under the old approximate template. These values may be reported only as
uncalibrated diagnostics. They must not be described as event significance,
false-alarm evidence, or detector sensitivity.

## Implemented but awaiting remote evidence

The current local code replaces the approximate chirp with the public
GW150914 numerical-simulation waveform used by the GWpy injection example. The
download metadata and run record retain the URL, SHA-256 checksum, inferred
source rate, resampling ratio, placement, and normalization. Until the patched
commit is executed remotely, this is an implemented method rather than an
observed result.

The filtering/statistic-equivalence experiment constructs one GWpy-designed
FIR and applies the same coefficients through explicit FFT convolution and
GWpy convolution. It computes an identical normalized matched statistic across
a frozen grid of FIR durations and edge trims, first on stationary synthetic
noise and then on archived H1/L1 windows. Local regression tests establish the
software path on fixtures; paper-facing real-data values remain pending.

The time-local-noise experiment compares a leave-one-out global PSD with
predeclared 32-, 64-, and 96-second local calibration radii. The 64-second
model is primary and must cover at least 90% of evaluation windows. It records
held-out score spread, five chronological blocks, template-projected spectral
power, and fixed narrow-band observed/model ratios. Its real-data conclusion is
also pending the next controlled remote run.

Any Results text for the shared-FIR experiment, time-local PSD experiment,
confirmatory GWOSC interval, CRESST/SCRESST validation, calibrated event
significance, or injection sensitivity must use the placeholders in
`PENDING_RESULT_PLACEHOLDERS.md` until the corresponding evidence files exist
inside this bundle and their acceptance status has been reviewed.

## Paper sections that can be updated now

The Methods section can describe the S1–S9 validation ladder, seed-sweep and
held-out infrastructure, the Stage 0 reproducibility protocol, GWOSC data
provenance and official DATA coverage, and the Hann/median PSD estimator. It
can also describe the newly frozen waveform, FIR sweep, and local-PSD protocol
as planned methods, provided the text does not imply that their remote results
already exist.

The Results section can report the synthetic gate outcomes, exact PSD agreement
with GWpy, and the failed global-PSD held-out calibration. The failure should
be presented directly rather than hidden: random splits were unstable and
chronological blocks failed for both detectors.

The Discussion can state that PSD-estimator correctness does not guarantee
stationary likelihood calibration on real detector data. It can motivate
identical-statistic filtering comparisons and time-local spectral models as
predeclared follow-up tests. It should treat nonstationarity and transient
contamination as competing hypotheses, not established explanations.

The Limitations section should state that the current evidence covers one
256-second interval around one event, that overlapping split seeds are not
independent replications, that the archived event scores are uncalibrated, and
that confirmatory intervals are still required after model selection.

## Statements that must not appear yet

The paper must not claim a calibrated GW150914 significance, a validated
nominal SNR-eight injection sensitivity, equivalence of the original GLS and
GWpy FIR statistics, superiority of a 64-second local PSD, or general
calibration across GWOSC epochs. It must also not treat the next-run waveform,
filtering, or local-PSD outputs as completed until their evidence JSON and
checksums have been synchronized into the repository.

## Planned figure sequence

The synthetic paper figures should establish the controlled hierarchy first:
optimal-filter/CRB calibration, the independent linear-AE/EMPCA bridge,
colored-noise metric reversal, rank sensitivity, and covariance robustness.
The first GWOSC figure should show random and chronological
`null_sigma_over_predicted` values with the acceptance bands visible. A second
panel can show exact PSD agreement alongside the non-equivalent matched-score
correlations, clearly labeling the latter as diagnostic.

After the next remote run, the filtering figure should display implementation
identity error over the FIR-duration/edge-trim grid and separately display the
original-GLS/shared-FIR correlation. The local-noise figure should compare
global and local held-out score standard deviations, followed by chronological,
template-projected, and narrow-band diagnostics. These follow-up figures must
retain the predeclared primary setting visually rather than highlighting the
best post hoc setting.

## Next controlled loop

The next operation is to commit and push the current patch, execute the remote
runbook without tuning, commit only the generated evidence directory on the
server, and synchronize it. After synchronization, run
`transfer_paper/scripts/refresh_bundle.py`, regenerate the notebooks, and
render the available figures. The evidence must then be assessed in this order:
shared-FIR implementation identity, original GLS-to-FIR sensitivity,
stationary synthetic local-PSD control, H1/L1 primary local calibration,
chronological stability, and template-sensitive spectral diagnostics.

If the primary local model passes both detectors, it still requires an
untouched confirmatory interval before supporting a general real-noise
calibration claim. If it improves but fails, the paper may report partial
correction as an exploratory result. If it does not improve, simple local PSD
drift should be rejected as the dominant explanation and a new hypothesis must
be frozen before further testing.
