# TIDMAD Stage — Implementation Roadmap

Decision: TIDMAD (ABRACADABRA, arXiv:2406.04378, NeurIPS 2025 spotlight) is
Paper 1's **second real-data anchor**. This document is the executable roadmap:
for each phase, the reasoning (why the experiment exists), the method (what is
computed), the expected outcome (what we predict and what would falsify it),
the implementation (where code goes), and the verification (how we know the
result is not an artifact).

Companion documents: `docs/PAPER1_SECOND_REALDATA_ANCHOR.md` (why TIDMAD over
alternatives), `docs/CORRECTION_PLAN_EXPERIMENTS_2026-07-26.md` (the rest of
the repo work), `docs/results/CRESST_PREDECLARED_GATES.md` (the predeclaration format
to imitate — CRESST's is the proven template and TIDMAD's should mirror it).

---

## 0. Why TIDMAD closes the specific gaps

Paper 1's real-data evidence has two structural holes. TIDMAD closes both, and
adds one bonus the existing evidence cannot supply.

| Gap | Why CRESST cannot close it | How TIDMAD closes it |
| --- | --- | --- |
| No real-data test of metric reversal | no clean signal ⇒ you can compare residuals but cannot show the MSE-preferred reconstruction actually recovers the signal worse | injected signals of known amplitude in a reference channel ⇒ **error against truth** is measurable |
| No detector-facing metric | trigger efficiency underpowered, inconclusive | published denoising score, plus an end-to-end dark-matter limit chain |
| *(bonus)* no real-data subspace-recovery check | CRESST's rank-6 subspace has no analytic ground truth | unknown sinusoid phase spans an **analytically known 2-D quadrature subspace** ⇒ real-data S2 |

The bonus is worth emphasising because it is unusual. For a sinusoid of known
frequency and unknown phase, the admissible signal manifold is exactly
span{cos(2πft), sin(2πft)}. So:

- rank-1 OF at a fixed phase is provably suboptimal by a known factor;
- rank-2 is provably sufficient;
- rank ≥ 3 can only add noise.

That gives a *falsifiable, analytically predicted* rank curve on real detector
data. CRESST could only report an empirically chosen k = 6.

**The strategic property that makes this worth doing: it is informative either
way.** TIDMAD's baselines are MSE-trained deep denoisers. If a correctly
whitened linear estimator is competitive, Paper 1's thesis lands with maximum
force. If the deep models retain a clear margin, that is a clean, honest
result which sets up Paper 2 (architecture matters once the metric is right).
There is no outcome that wastes the effort — which is exactly the property to
require before committing weeks.

---

## 1. Dataset facts to confirm in Phase 0

From the public release (confirm each against the actual files before relying
on it — these are the assumptions the whole design rests on):

- 10 MS/s SQUID time series from ABRACADABRA; signal of interest is a
  **sinusoidal oscillation mode**.
- `abra_training_00{00..19}.h5` and `abra_validation_00{00..19}.h5`: **two
  channels** with sample-to-sample correspondence — one carrying the SQUID
  output, the other the injected reference. This pairing is the ground truth.
- `abra_science_0{000..207}.h5`: **one channel**, no injected signal ⇒ the
  natural signal-free noise source for PSD estimation.
- A weak-signal variant at 1/5 amplitude (file range 0020–0039) — a harder
  task and a free severity axis.
- Known caveat: some files have unequal channel lengths; the release
  prescribes truncating both channels to the first 2×10⁹ samples.
- Baselines with published scores: `none`, `mavg`, `savgol`, and the deep
  models `fcnet`, `punet`, `transformer`, `wavenet`, `rnn`. Frequency-splitting
  into four bands is part of their training protocol.
- License CC-BY-4.0.

**Phase-0 deliverable is a one-page `docs/TIDMAD_DATA_CONTRACT.md`** recording,
with file hashes: channel semantics, dtype and bit depth, sampling rate,
injection frequencies and amplitudes present in each validation file, the exact
definition of the denoising score as implemented in `benchmark.py`, and the
units of everything. Every later phase cites this document. Do not proceed
without it — a wrong channel-semantics assumption would silently invalidate
every downstream number.

---

## 2. Phase plan

### Phase 0 — Data contract and baseline reproduction (½–1 day)

**Reasoning.** Before any science, establish that we can (a) read the data
correctly and (b) reproduce the benchmark's own published numbers. If we cannot
reproduce `none` and `savgol`, our score for any new method is meaningless and
the stage should be abandoned cheaply.

**Method.**
1. `python download_data.py -t 0 -v 2 -s 2` — two validation files (truth) and
   two science files (noise). Record sizes and hashes.
2. Inspect structure; apply the 2×10⁹-sample truncation rule.
3. Run the reference chain **unmodified**: `inference.py -m savgol`, then
   `benchmark.py -m none` and `-m savgol`.

**Expected outcome.** Scores matching the published values for `none` and
`savgol` within their stated precision.

**Implementation.** `scripts/download/download_tidmad.py` (thin wrapper that
refuses to write into the repo, mirroring `download_cresst.py`), and
`src/noise_geometry/tidmad/loader.py` (memory-mapped chunked reader; never load
a full 2×10⁹-sample channel into RAM).

**Verification / gate G0.** Published `none` and `savgol` scores reproduced.
Also verify the two channels are sample-aligned by cross-correlating a short
stretch and confirming the peak at lag 0.

*If G0 fails: stop.* Fall back to XRISM or to strengthening CRESST.

---

### Phase 1 — Noise model, whitening convention, and estimator identity (1 day)

**Reasoning.** Paper 1's entire argument is that the *right* Σ defines the
geometry. So the first scientific step is to establish that our estimated Σ and
our whitening convention are correct on this data — the same discipline that
made the GWOSC implementation checks meaningful even when the calibration
failed. Getting this wrong would make every later comparison a convention bug
rather than a physics result.

**Method.**
1. Estimate a one-sided PSD from science-file stretches using the repo's
   estimator (`src/noise_geometry/noise/psd.py`), with Welch parameters frozen
   in the config. Estimate on stretches disjoint from every evaluation stretch.
2. Cross-check against `scipy.signal.welch` on identical inputs (this is the
   TIDMAD analogue of the GWOSC/GWpy check).
3. Inject a synthetic sinusoid of known amplitude into *real* science noise;
   run the repo's OF; compare recovered amplitude to truth and empirical
   variance to the PSD-predicted bound σ_A = (Σ|S_k|²/J_k)^(−1/2).
4. Characterise stationarity: compute PSDs in consecutive blocks and record
   the drift. Define — and freeze — a stationarity-selection rule for which
   stretches enter the analysis.

**Expected outcome.** PSD agreement with scipy at machine precision on
identical windows. Amplitude recovery unbiased at the sub-percent level.
Empirical OF variance within ~10% of the bound on stationarity-selected
stretches, with visible excess on unselected stretches (that excess is the
first hint of the Phase 4 boundary result).

**Implementation.** `src/noise_geometry/tidmad/psd.py`,
`src/noise_geometry/tidmad/injections.py` (synthetic-injection harness — the
CRESST module already has an `injections.py` to model it on).

**Verification / gate G1.** Amplitude bias < 1%; σ_OF within 10% of the bound;
PSD cross-check passes. Additionally: a **no-signal null** — run the same OF on
science stretches with no injection and confirm the amplitude distribution is
consistent with zero at the predicted variance.

---

### Phase 2 — Metric reversal with ground truth *(the headline experiment)*

**Reasoning.** This closes Gap 1 and is the reason the stage exists. S5 shows
metric reversal in simulation; CRESST cannot show it at all. Here, for the
first time, we can show on **real detector noise with known truth** that the
reconstruction preferred by isotropic MSE is worse at recovering the actual
signal. SQUID noise is steeply coloured (1/f-like), so raw variance is
dominated by low frequencies while the injected signal sits elsewhere — the
mechanism the paper describes is not merely present, it should be large.

**Method.**
1. Segment the SQUID channel into windows of fixed length N (predeclared;
   suggest N = 2²⁰ as a starting point, fixed in Phase 0's contract).
2. Fit two matched-rank subspaces on **training-file** windows:
   - EMPCA / weighted PCA in the Σ⁻¹ geometry (whitened rFFT representation,
     Re/Im stacked, exactly as S5 does);
   - ordinary PCA in the raw Euclidean geometry.
3. Evaluate on **held-out validation** windows. Report three metric families:
   - raw MSE to the observed trace,
   - weighted residual (x − ŝ)†Σ⁻¹(x − ŝ),
   - **amplitude error against the injected truth** — the new quantity.

**Expected outcome.** The predicted ordering:

| Metric | Winner |
| --- | --- |
| raw MSE | PCA |
| weighted residual | EMPCA |
| **error vs injected truth** | **EMPCA** |

with the third row settling which reconstruction is actually better. Effect
size expected to be large — plausibly a factor, not the 1.6–62% range CRESST
showed — because of the steep noise colouring.

**Falsifier / gate G2.** The MSE-preferred estimator is *not* worse in
amplitude error. If G2 fails, that is a genuinely important scope limitation
and must be reported, not buried: either the in-band noise is close enough to
white that the effect does not exist here, or the claim's real-data reach is
narrower than the theory suggests. Determine which, and say so.

**Implementation.** `src/noise_geometry/tidmad/analysis.py`, reusing
`subspace/`, `metrics/`, and the shared `canonical/empca_equivalence_utils.py`
rFFT primitives so the representation convention is identical to S5's.

**Verification.**
- Paired, per-window differences with bootstrap CIs over windows (mirroring
  CRESST's paired event bootstrap), *plus* a file-level and time-block-level
  bootstrap, since consecutive windows are not independent.
- Chronological train/evaluation split within each file, and a leakage audit
  (window index and hash disjointness) copied from the CRESST runner.
- Report both the pooled and the per-file effect, exactly as CRESST reports
  pooled and per-channel.

---

### Phase 3 — Rank and the analytic quadrature manifold

**Reasoning.** This is the sharpest theory test in the whole paper, and it is
available only here. The unknown-phase manifold is exactly 2-dimensional, so
the paper's central "a learned rank-k subspace absorbs a signal manifold a
fixed template cannot represent" claim has an analytically known answer on real
data.

**Method.**
1. Rank sweep k = 1…4 of the weighted subspace fit; evaluate amplitude error
   and weighted residual on held-out windows.
2. Separately, compute the principal angles between the learned rank-2
   subspace and the **analytic** span{cos, sin} at the injection frequency, in
   the whitened geometry.
3. Compare rank-1 OF at a fixed (wrong) phase against the rank-2 result, and
   check the loss matches the analytic cos²θ SNR-loss prediction.

**Expected outcome.** Rank-2 clearly beats rank-1; rank ≥ 3 saturates or
slightly degrades. The learned rank-2 subspace aligns with the analytic
quadrature subspace to within a small angle (finite-sample, so expect degrees,
not micro-degrees — S2's ~3.9° is the right expectation scale).

**Falsifier / gate G3.** Rank-2 does not beat rank-1, or the learned subspace
does not align with the analytic one. Either would indicate a convention error
(most likely in the frequency-bin or phase handling) — treat as a bug, not a
finding, until Phase 1's checks are re-verified.

**Verification.** The analytic subspace is computed independently of the
fitting code. Additionally: repeat at several injection frequencies and confirm
the alignment is frequency-independent.

---

### Phase 4 — Calibration boundary (the in-domain GWOSC replacement)

**Reasoning.** Paper 1 needs its "implementation correctness ≠ model adequacy"
lesson. GWOSC currently supplies it, but at the cost of a full section in a
different domain, with no ground truth for what the failure costs. TIDMAD can
supply the same lesson **in-domain, with injections quantifying the cost** —
which is strictly better.

**Method.**
1. Whitened residual χ²/dof and score dispersion on held-out science stretches
   under the global PSD model.
2. A **stationary synthetic control** (noise-module-generated with the same
   PSD) that must pass — the halt gate.
3. Time-local PSD variants (block-local estimation at several block lengths),
   mirroring GWOSC's local-PSD follow-up, predeclared before running.
4. Quantify the cost: how much does amplitude-estimate bias/variance degrade
   relative to the ideal when the noise model is inadequate? This is the number
   GWOSC could never produce.

**Expected outcome.** The synthetic control passes. Real data shows excess
dispersion — from 8-bit quantisation, drift, and glitches — with the excess
reduced but probably not eliminated by time-local PSDs. Reported as a boundary,
not a failure of the framework.

**Verification.** Same acceptance-range discipline as GWOSC: predeclare the
pass band before running; report the control and the real data side by side;
never tune the block length until the real data passes.

---

### Phase 5 — Detector-facing metric (closes Gap 2)

**Reasoning.** The paper needs one real-data number a physicist cares about.
TIDMAD's published denoising score is exactly that, and it comes with a
scoreboard.

**Method.**
1. Produce a denoised time series with the noise-aware **linear** estimator —
   the natural choice is a Wiener/GLS filter with the signal PSD estimated from
   *training* files and applied blind to validation (this is the same
   information the deep baselines get, so the comparison is fair).
2. Write it in the release's `.h5` format and score it with the project's own
   `benchmark.py`. **Do not reimplement the score.**
3. Report alongside the published baselines. Optionally extend to
   `brazilband.py` for the limit-setting statement — strongest possible
   physics-facing result, but not required for the paper.

**Expected outcome.** The linear noise-aware estimator lands on the scoreboard.
Where exactly is genuinely unknown, and the paper should say so before showing
the number.

**Framing discipline (important).** We are not claiming to win a benchmark.
The claim is: *how much of the gap between doing nothing and a trained deep
denoiser is explained by using the correct noise geometry with a linear model?*
Write that sentence into the section before running the experiment, so the
result cannot be retrofitted into a stronger claim.

**Gate G4.** The linear estimator produces a valid, scoreable output through
the unmodified benchmark path.

---

### Phase 6 — Negative controls and predeclaration

**Reasoning.** CRESST's five negative controls are one of the strongest parts
of Paper 1. TIDMAD must have the same falsification structure, or reviewers
will (rightly) treat it as a weaker stage.

**Method — five controls mirroring CRESST's:**

1. **Wrong PSD** — whiten with a PSD from a mismatched period or a scrambled
   shape.
2. **Noise-only training** — fit the subspace on science (no-injection)
   windows.
3. **Random weighted basis** — replace the learned subspace with a random one
   of the same rank in the same geometry.
4. **Mismatched template / wrong frequency** — OF at a frequency where nothing
   was injected.
5. **Permuted bins** — destroy the time/frequency structure.

**Expected outcome.** All five degrade the primary metric. Per-file exceptions
are acceptable and should be reported (CRESST had one; hiding it would be
worse than showing it).

**Predeclaration.** Before Phase 2 is run on validation data, freeze
`docs/TIDMAD_PREDECLARED_GATES.md` containing: window length and count;
stationarity-selection rule; PSD estimation windows and their disjointness;
rank grid; the primary metric and the smallest relevant effect for G2; bootstrap
unit (window / block / file) and CI type; the negative-control list; and an
explicit statement that no threshold will be adjusted after test evaluation.
Mirror the structure of `docs/results/CRESST_PREDECLARED_GATES.md`.

---

## 3. Code layout

Mirror the CRESST module, which is the proven pattern in this repo:

```
src/noise_geometry/tidmad/
    __init__.py
    loader.py            # chunked/memmapped HDF5 reader, truncation rule, channel semantics
    psd.py               # PSD estimation + stationarity selection
    injections.py        # synthetic-injection harness for Phase 1 verification
    analysis.py          # Phase 2/3 subspace + OF comparison, paired bootstrap
    negative_controls.py # the five controls
    score.py             # thin wrapper that shells out to the release's benchmark.py
    splits.py            # chronological window splits + leakage audit
configs/tidmad/
    smoke.yaml                  # Phase 0-1
    metric_reversal.yaml        # Phase 2 (predeclared)
    rank_sweep.yaml             # Phase 3
    calibration_boundary.yaml   # Phase 4
    denoising_score.yaml        # Phase 5
scripts/download/download_tidmad.py
docs/TIDMAD_DATA_CONTRACT.md
docs/TIDMAD_PREDECLARED_GATES.md
```

Wire `dataset: tidmad` into `scripts/run_experiment.py` (the dispatcher already
has a `tidmad` branch returning `planned` — replace it). Every run writes the
standard record: config, seed, metrics, git commit, dataset metadata,
checksums.

---

## 4. Verification strategy (cross-cutting)

The failure modes to defend against, and the defence for each:

| Risk | Defence |
| --- | --- |
| Wrong channel semantics | Phase-0 data contract + lag-0 cross-correlation check |
| Convention error in whitening | scipy cross-check + synthetic injection recovery (G1) |
| Score reimplementation error | never reimplement — shell out to `benchmark.py` |
| Correlated windows inflating significance | block/file-level bootstrap in addition to window-level |
| Leakage between fit and evaluation | chronological splits + index/hash disjointness audit, copied from CRESST |
| Non-Gaussianity misread as method failure | Phase 4 is designed to *expect* it; synthetic control isolates it |
| Post-hoc tuning | predeclaration frozen and hashed before validation data is scored |
| Cherry-picked files | report per-file effects, not only pooled — as CRESST does per-channel |

---

## 5. Schedule and decision points

| Phase | Effort | Gate | If it fails |
| --- | --- | --- | --- |
| 0 Data contract + baseline reproduction | 0.5–1 d | G0 | abandon stage cheaply |
| 1 Noise model + estimator identity | 1 d | G1 | debug conventions; do not proceed |
| 2 Metric reversal with truth | 1–2 d | **G2** | report as scope limitation (still publishable) |
| 3 Rank / quadrature manifold | 1 d | G3 | treat as bug, re-verify Phase 1 |
| 4 Calibration boundary | 1–2 d | control must pass | halt and fix |
| 5 Denoising score | 1–2 d | G4 | report Phases 2–4 only |
| 6 Controls + predeclaration archival | 1–2 d | all controls degrade | investigate before publishing |

Phases 0–2 constitute the **smoke test**: ~3–4 days, and G2's answer decides
whether TIDMAD becomes a full paper section. Run it in parallel with the
manuscript corrections already planned (S5 provenance WP-P1, Study B descope
WP-P3) — those are independent and short.

**Manuscript consequence once G2 passes:** TIDMAD becomes §7.4 (a positive
real-data section with ground truth), GWOSC demotes to three sentences plus an
appendix, and the abstract gains its strongest sentence — a real-data
confirmation of the metric-reversal claim with known injected truth, on a
public dataset with a published benchmark.
