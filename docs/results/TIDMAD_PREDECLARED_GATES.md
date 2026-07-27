# TIDMAD Stage — Predeclaration

**Status: FROZEN for the synthetic gates and for Phase 0 (channel/sampling
facts, gate G0). NOT YET FROZEN for injection frequencies/amplitudes** — those
are read off in Phase 1's real-noise run rather than Phase 0, and this
document must be re-frozen once they are before Phase 2 scores validation
data.

Date frozen (synthetic): 2026-07-27. Date frozen (Phase 0): 2026-07-27.
Stage: TIDMAD (ABRACADABRA, arXiv:2406.04378), Paper 1's second real-data anchor.
Roadmap: [`../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`](../plans/TIDMAD_IMPLEMENTATION_ROADMAP.md).
Template: [`CRESST_PREDECLARED_GATES.md`](CRESST_PREDECLARED_GATES.md) — the
CRESST predeclaration is the proven pattern and this mirrors it deliberately.

## The rule this document exists to enforce

Every threshold, split fraction, rank grid, radius, and acceptance band below is
fixed **before** the corresponding evaluation data is scored. Nothing here may
be adjusted after seeing a result. A gate that fails is a finding to report, not
a parameter to tune. A control that does not degrade is a finding to report —
CRESST had exactly one such case (wrong-PSD weighting on channel 35/58.0) and
published it.

## 1. Why this stage exists

Paper 1's real-data evidence has two structural gaps that CRESST cannot close:

1. no real-data test of the metric-reversal claim **against ground truth** —
   CRESST has no clean signal, so it can compare residuals but cannot show the
   MSE-preferred reconstruction recovers the signal worse;
2. no detector-facing metric — CRESST's trigger efficiency was inconclusive.

TIDMAD closes both, and adds one thing neither existing stage has: the unknown
phase of a sinusoid spans **exactly** span{cos, sin}, so the correct rank is
known in closed form rather than chosen empirically.

## 2. Data and preprocessing

| Item | Value | |
| --- | --- | --- |
| Release | `jessicafry/TIDMAD`, CC-BY-4.0 | |
| Sampling rate | 10 MS/s | confirmed from file attrs (`sampling_frequency`), both channels |
| Truncation | first 2×10⁹ samples of both channels | release-prescribed |
| SQUID (observation) channel | `timeseries/channel0001/timeseries` | see below |
| Reference (truth) channel | `timeseries/channel0002/timeseries` (absent in science files) | |
| Signal-free noise source | science files (single channel, no injection) | |
| Injection frequencies / amplitudes | file- and sample-range-specific; see below | not a single fixed dataset-wide fact -- see caveat |
| Analysis window | 4096 samples | |
| Frequency handling | snapped to the rFFT grid before use | removes leakage as a confounder |

Channel roles are established by the release's own code, not assumed: five
upstream scripts (`train.py`, `inference.py`, `benchmark.py`, `brazilband.py`,
`array2h5.py`) all hardcode `channel0001` as the SQUID readout and
`channel0002` as the reference/target. A spectral line-dominance probe was
tried first and was inconclusive on the downloaded files' opening window; it
is retained as an audited diagnostic in the data contract, not as the basis
for the decision. See `docs/results/TIDMAD_PHASE0_RESULT.md`.

## 3. Splits and leakage

| Item | Value |
| --- | --- |
| Split | chronological, never random |
| Fit fraction | 0.70 |
| Guard gap | 4 windows dropped between fit and evaluation |
| Stride | non-overlapping (stride = window length) |
| Audit | window-index disjointness, window-hash overlap, realized guard gap, all recorded in every run |

Consecutive windows of a continuous stream are dependent, so a random split
would place near-duplicates on both sides of the boundary and inflate every
effect. The guard gap exists because even a chronological split leaks when the
two sides are adjacent and the noise correlation time exceeds one window.

## 4. Noise model

| Item | Value |
| --- | --- |
| Estimator | Welch over non-overlapping windows, boxcar, detrended |
| Averaging | **median**, bias-corrected by ln 2 |
| Weights | `inverse_psd_weights` (the repository's single OF convention) |
| Calibration data | disjoint from every evaluation window |
| Stationarity | measured, not assumed; band-averaged deviation in dB against the median block |
| Stationarity tolerance | 3 dB |

Median rather than mean Welch is deliberate: TIDMAD noise contains glitches, and
a mean estimate absorbs them into the PSD and then *under*-weights the affected
bands during whitening, degrading the very estimator under study.

## 5. Gates

### Phase 0 — data contract (G0)

Reproduce the published `none` and `savgol` denoising scores using the
release's own `benchmark.py`. **If G0 fails, the stage stops.** A score we
cannot reproduce cannot be compared.

**Result: satisfied.** `none` (the calibration anchor) matched closely on the
full 20-file validation set (1.1026 vs. published 1.10). `savgol` did not
match, but the discrepancy is diagnosed as inherent numerical ill-conditioning
in the published baseline's own hyperparameters (`window_length=100,
polyorder=11` sits well past where scipy's Vandermonde-based `savgol_coeffs`
can support the fit in double precision), not a defect in this reproduction —
see `docs/results/TIDMAD_PHASE0_RESULT.md` for the full diagnosis. The
contract is `verified: true`.

### Phase 1 — estimator identity (G1)

| Gate | Statement | Threshold |
| --- | --- | --- |
| G1a | amplitude estimate unbiased | \|z\| < 3 |
| G1b | empirical σ matches the PSD-predicted bound | within 10% of 1 |
| G1c | no-signal null consistent with zero | \|z\| < 3 |
| G1d | learned rank-2 subspace ≈ analytic quadrature | < 5° at the declared operating point |
| G1e | fixed-phase SNR loss follows cos² | max abs error < 0.05 |
| G1f | subspace error follows the 1/SNR law | log-log slope −1 ± 0.15 |

Injected amplitude is specified in units of the estimator's own predicted σ, so
the gates have a known statistical power. G1d's absolute threshold is a smoke
check at a declared operating point; **G1f is the scientific statement**,
because a single angle could be finite-sample error or bias while the rate
distinguishes them.

### Phase 2 — metric reversal against truth (G2) — the primary gate

| Gate | Statement |
| --- | --- |
| G2 | the isotropic-MSE-preferred reconstruction is **worse against truth**, bootstrap CI excluding zero |
| G2 | a moving-block bootstrap (block 16) agrees |
| G2 | the raw-MSE and weighted-residual orderings are reversed |

Primary metric: paired per-window absolute amplitude error against the injected
truth. Bootstrap: 300 resamples, window-level *and* moving-block, both reported.
Rank fixed at 2 (the analytic rank). Both arms are fitted on the **same**
representation and differ only in the metric.

*If G2 fails*, that is a scope limitation on the claim's real-data reach and it
is reported as such — not buried, and not fixed by changing the operating point.

**Result: PASSED**, first attempt, on `abra_validation_0000.h5`, at exactly
this operating point, no file selection or parameter adjustment involved. See
`docs/results/TIDMAD_PHASE2_3_REAL_RESULT.md` for the numbers.

### Phase 3 — rank (G3)

| Gate | Statement | Threshold |
| --- | --- | --- |
| G3 | rank 2 beats rank 1 on amplitude error | CI excludes zero |
| G3 | rank 3 does not improve on rank 2 | ≤ 5% of the rank-2 RMSE |
| G3 | the knee is at rank 2 | smallest rank within 5% of the minimum |
| G3 | rank 2 beats a fixed-phase rank-1 filter | CI excludes zero |

**Final result: NOT CONFIRMED on any of three files tried**, including one
genuinely confirmatory attempt (`abra_validation_0005.h5`, selected by the
predeclared rule below before its outcome was known) — see
`docs/results/TIDMAD_PHASE2_3_REAL_RESULT.md` for the full account. The
first two attempts (`abra_validation_0000.h5`, `abra_validation_0010.h5`)
were exploratory, not confirmatory (the second file was chosen after
diagnosing why the first failed, even though no threshold was tuned) — this
document's own rule ("nothing may be adjusted after seeing a result") was
not fully honored in spirit for those two. The third attempt corrects that:
a data-quality selection rule was frozen and applied mechanically before any
new candidate's outcome was inspected, and it still did not confirm the
prediction — likely insufficient statistical power at this operating point
(150 evaluation windows) for the SNR levels found in real files, not a
property specific to one file. Treat Phase 3 real-data as investigated and
honestly non-confirmed, not as a passed or failed predeclared gate and not
as evidence against the underlying theory (still supported by theory and
controlled synthetic data).

Rank grid: 1, 2, 3, 4 — fixed. The reported quantity is the **knee**, not the
raw argmin: once the curve saturates, which flat rank happens to be lowest is
decided by noise in the fourth decimal.

#### Phase 3 confirmatory attempt — predeclared file-selection rule

Frozen 2026-07-27, **before** running the selection script or looking at any
Phase 2/3 reconstruction result on any candidate file. Applies to the 18
validation files not already used for an exploratory Phase 3 attempt
(`abra_validation_0000.h5` and `abra_validation_0010.h5` are excluded — both
already had their Phase 2/3 *reconstruction* results inspected, so any future
use of either would not be a fresh test).

For a candidate file, over the sample range `[0, window_samples * n_windows)`
= `[0, 2097152)` of `channel0002`:

1. **Single-tone check.** Compute one FFT over the full range (excluding the
   DC and Nyquist bins). Find the global peak frequency `f_peak` and its
   power `P_peak`. Require `P_peak / median(power) >= 1000`. Search for the
   strongest competing peak more than 20 Hz away from `f_peak` (to skip its
   own sidelobe leakage) and within 500 Hz of it; require that peak's power
   to be `<= 0.3 * P_peak`. (This is exactly the check that flagged file
   0000's two-tone structure and passed files 0010/0015's single-tone
   structure — see `docs/results/TIDMAD_PHASE2_3_REAL_RESULT.md`.)
2. **Adequate-amplitude check.** Fit quadrature templates at `f_peak` to
   every one of the `n_windows` windows in range (fit + evaluation
   together). Compute the median per-window amplitude fit to the
   **observed (SQUID) channel**, not the reference channel — the two
   channels are different physical measurements and the coupling between
   them is not constant across files (measured coupling ratios,
   observed-fit-amplitude / reference-channel-truth-amplitude: 3.22 for file
   0000, 0.057 for file 0010 — see `TIDMAD_PHASE2_3_REAL_RESULT.md` for how
   this was discovered and why it matters). Compute the PSD-predicted noise
   floor `sigma` via `quadrature_amplitude_variance` (PSD + weights +
   analytic template; the same formula behind Phase 1's own
   `predicted_sigma`). Require `median(observed_channel_amplitude) / sigma
   >= 3` — a standard 3-sigma detectability threshold, not reverse-engineered
   from any file's known outcome. (An earlier draft of this check compared
   the *reference*-channel amplitude to this same sigma; that was corrected
   before scanning any new candidate file, once validating it against the
   two already-known files produced a result backwards from their known
   outcomes — full account in `TIDMAD_PHASE2_3_REAL_RESULT.md`.)

**Selection procedure.** Apply both checks, in file-index order starting
from `abra_validation_0001.h5`, to every file not excluded above. Take the
**first** file (lowest index) that passes both checks as the confirmatory
candidate. If no file passes, that is itself a reportable finding (no file
in the checked range supports a confirmatory rank-2 test at this operating
point) — do not loosen either threshold after seeing that outcome.

Run Phase 2 and Phase 3 once, on the selected file, at the same frozen
operating point as every other attempt (window_samples=4096, n_windows=512,
rank=2 for Phase 2 / ranks 1-4 for Phase 3, fit_fraction=0.7,
guard_windows=4, bootstrap_reps=300, block_size=16). Report the result
exactly as it comes out — this rule exists precisely so that this attempt
*can* be reported as confirmatory, pass or fail.

**Result: rule applied, `abra_validation_0005.h5` selected (8702.28 Hz,
observed-channel SNR 3.42 — first file in index order passing both checks),
Phase 2/3 run once as specified. Phase 2's direction replicated (not
significant); Phase 3's rank-2 prediction did not confirm** (flat RMSE
across ranks 1-4: 4.752, 4.752, 4.756, 4.758; rank2-vs-rank1 CI
[−0.056, 0.098] spans zero). Full numbers in
`docs/results/TIDMAD_PHASE2_3_REAL_RESULT.md`. This file passed every
predeclared quality check and still did not confirm the prediction — the
most likely explanation is that an SNR-≥3 threshold, while a reasonable
generic detectability convention, does not guarantee enough statistical
power for a rank-1-vs-rank-2 test at only 150 evaluation windows. Anyone
repeating this should predeclare a power-calibrated threshold (e.g. via a
synthetic power calculation at the target window count) rather than reusing
this session's generic 3-sigma convention.

### Phase 4 — calibration boundary

| Item | Value |
| --- | --- |
| Synthetic control acceptance | χ²/dof and score dispersion both in [0.9, 1.1] |
| Control status | **halt gate** — if it fails, we have a bug, not a finding |
| Local PSD radii | 8, 16, 32 windows — declared here, not chosen after |
| Local neighbourhood | **excludes the window being scored** |

There is no pass/fail gate on the *real* global or local models. Whatever they
give is the boundary measurement and is reported in either direction, together
with `cost_of_mismatch` — what the inadequate model costs the amplitude
estimate, which is the number GWOSC could never produce.

**Result: control passes (halt gate satisfied). The real global model
calibrates well** — χ²/dof 1.0031, score dispersion 1.0142, both
comfortably within [0.9, 1.1] and close to the synthetic control's own
values; `cost_of_mismatch` is negligible (sigma inflation 1.0002). This is
the opposite of the roadmap's prior expectation (carried over from GWOSC)
that real SQUID noise would show clear calibration degradation — report it
as the genuine, positive finding it is, not forced into a GWOSC-style
boundary narrative. See `docs/results/TIDMAD_PHASE4_6_REAL_RESULT.md`.

### Phase 5 — detector-facing metric (G4)

| Item | Value |
| --- | --- |
| Estimator | Wiener filter, `H = S/(S+N)` |
| Fitted on | training files only; applied blind to validation |
| Scored by | the release's own `benchmark.py`, never reimplemented |
| Gate | the linear estimator produces a valid, scoreable output |

**Claim boundary, fixed before the run:** this measures how much of the gap
between no denoising and a trained deep denoiser is explained by using the
correct noise geometry with a linear model. It is *not* a claim to win the
benchmark, and the result will not be written up as one whatever the number is.

**Result: ran on the full-length `abra_validation_0000.h5`, scored 0
(returncode) through the unmodified upstream `benchmark.py` — coarse score
-1.8025, below the "no denoising" baseline (1.10) and every published
baseline except the ill-conditioned `savgol`.** The gate itself
(`G4_benchmark_ran`) is satisfied; the honest answer to the claim-boundary
question is a negative amount at this operating point. See
`docs/results/TIDMAD_PHASES_2_TO_6_RESULT.md`.

### Phase 6 — negative controls

All five must degrade the primary metric with a bootstrap CI excluding zero:

| Control | What it breaks |
| --- | --- |
| `wrong_psd` | the noise model — a spectrally flipped (wrong but still coloured) PSD |
| `noise_only_train` | the training signal — subspace fitted on signal-free windows |
| `random_weighted_basis` | the learned subspace — random basis, matched rank, same metric |
| `wrong_frequency` | the signal model — filter at an empty bin (offset 37 bins, fixed) |
| `permuted_bins` | the time structure — per-window sample permutation |

`wrong_psd` is deliberately **not** the identity metric: the isotropic case is
already Phase 2's Euclidean arm, so using it here would duplicate that contrast
instead of testing the noise model separately.

Per-file exceptions are acceptable **and must be reported**. If a control does
not degrade, that appears in the write-up.

**Result, real data (`abra_validation_0000.h5`): three of five degrade**
(`wrong_frequency`, `permuted_bins`, `wrong_psd`, all CI excluding zero in
the expected direction). **`noise_only_train` does not degrade** (CI spans
zero) **and `random_weighted_basis` improves slightly** (CI entirely on the
wrong side of zero). Report both exceptions plainly, matching CRESST's
precedent of naming its one exception rather than hiding it. See
`docs/results/TIDMAD_PHASE4_6_REAL_RESULT.md` for the full numbers and
untested candidate explanations.

## 6. Reporting standards

1. Every number traces to an archived run JSON with config, seed, and git commit.
2. Paired comparisons with intervals — never a bare point estimate.
3. Both window-level and block-level bootstrap where windows are dependent.
4. Per-file effects reported alongside pooled ones, as CRESST reports
   per-channel.
5. Dispersion statistics carry their standard error, with the caveat that the
   formula assumes an exact predicted σ and understates the true spread.
6. Saturated controls are labelled as such: when every control collapses to the
   zero-recovery level they establish falsifiability but cannot rank severity.

## 7. What is not claimed

- No detector-sensitivity or dark-matter-limit claim.
- No claim that the linear estimator is state of the art.
- Synthetic-mode results establish that the analysis is correct and the gates
  have power. They establish nothing about TIDMAD data.

## 8. Change log for this predeclaration

| Date | Change |
| --- | --- |
| 2026-07-27 | Initial freeze for the synthetic gates (Phases 1–4, 6). Phase 0/5 entries and all data-dependent fields remain open. |
| 2026-07-27 | Phase 0 freeze: sampling rate, channel roles, and gate G0 result filled in. Injection frequencies/amplitudes remain open, deferred to Phase 1's real-noise run. |
| 2026-07-27 | Phase 2 real-data result (PASS, first attempt, `abra_validation_0000.h5`) and Phase 3 real-data investigation (not confirmed on two files, both diagnosed) recorded. Discovered mid-investigation that `abra_validation_0000.h5`'s reference channel carries a genuine two-tone signal, not a single tone as assumed; a second file was selected on that basis, which is a post-hoc choice by this document's own predeclaration rule. See `docs/results/TIDMAD_PHASE2_3_REAL_RESULT.md` for the full account and the honesty caveat on Phase 3's status. |
| 2026-07-27 | Froze a Phase 3 file-selection rule (single-tone + observed-channel-SNR checks) before running it; applied mechanically across 18 candidate files; selected `abra_validation_0005.h5`, which was never inspected beforehand. Result: not confirmed (flat rank curve, likely insufficient statistical power at this operating point). Fixed two implementation bugs in the selection check during validation against known files, before scanning new candidates (bogus predicted-sigma formula; wrong channel used for the amplitude check — the latter also revealed that reference-to-SQUID channel coupling varies >50x across files, explaining `abra_validation_0010.h5`'s earlier "weak amplitude" diagnosis more precisely as weak coupling). Also ran Phase 4 and Phase 6 for real data for the first time: Phase 4's global model calibrates well (unlike GWOSC); Phase 6 shows 3 of 5 controls degrade, 2 do not. Real-mode configs and driver wiring added for Phases 2/3/4/6; `transfer_paper/scripts/refresh_bundle.py` extended with TIDMAD support. See `docs/results/TIDMAD_PHASE2_3_REAL_RESULT.md` and `docs/results/TIDMAD_PHASE4_6_REAL_RESULT.md`. |
