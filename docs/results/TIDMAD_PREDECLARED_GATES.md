# TIDMAD Stage — Predeclaration

**Status: FROZEN for the synthetic gates. NOT YET FROZEN for the real-data
run** — the entries marked *(to fix at Phase 0)* depend on facts that can only
be read off the files, and must be filled in and this document re-frozen before
any real TIDMAD evaluation number is computed.

Date frozen (synthetic): 2026-07-27.
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
| Sampling rate | 10 MS/s | *(confirm at Phase 0)* |
| Truncation | first 2×10⁹ samples of both channels | release-prescribed |
| SQUID (observation) channel | *(to fix at Phase 0)* | classified from spectra, not assumed |
| Reference (truth) channel | *(to fix at Phase 0)* | |
| Signal-free noise source | science files (single channel, no injection) | |
| Injection frequencies / amplitudes | *(to fix at Phase 0)* | |
| Analysis window | 4096 samples | |
| Frequency handling | snapped to the rFFT grid before use | removes leakage as a confounder |

Channel roles are established by evidence — the reference channel is
line-dominated, the SQUID channel broadband — and the Phase 0 script refuses to
guess when the two are not separable. A wrong channel assumption would not
crash; it would silently exchange data and truth.

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

### Phase 3 — rank (G3)

| Gate | Statement | Threshold |
| --- | --- | --- |
| G3 | rank 2 beats rank 1 on amplitude error | CI excludes zero |
| G3 | rank 3 does not improve on rank 2 | ≤ 5% of the rank-2 RMSE |
| G3 | the knee is at rank 2 | smallest rank within 5% of the minimum |
| G3 | rank 2 beats a fixed-phase rank-1 filter | CI excludes zero |

Rank grid: 1, 2, 3, 4 — fixed. The reported quantity is the **knee**, not the
raw argmin: once the curve saturates, which flat rank happens to be lowest is
decided by noise in the fourth decimal.

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
