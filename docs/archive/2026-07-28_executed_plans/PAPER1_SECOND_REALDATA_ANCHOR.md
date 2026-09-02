# Paper 1: Is the Real-Data Evidence Enough? Candidate Analysis and Smoke-Test Plan

Date: 2026-07-26. Decision document for whether Paper 1 needs a second
real-data anchor, which dataset it should be, and what to do with GWOSC.

---

## 1. Diagnosis: what exactly is missing

Paper 1's evidence has three tiers: S1–S9 (controlled, positive, load-bearing),
CRESST-A (real, positive, narrow), GWOSC (real, negative, boundary). Reading
the claim set against the evidence, there are exactly **two structural gaps**,
and they are more specific than "the real data is weak":

**Gap 1 — no real-data test of the central claim with ground truth.**
The paper's headline theoretical consequence is the *metric reversal*: under
colored noise, MSE and the likelihood rank reconstructions differently. That is
demonstrated only in S5, i.e. only in simulation. CRESST cannot demonstrate it,
because CRESST has no clean signal: without truth you can compare weighted
residuals, but you cannot show that the MSE-preferred reconstruction is
*actually worse at recovering the signal*. The reversal on real detector noise
is currently asserted by extrapolation.

**Gap 2 — no detector-facing metric on real data.**
CRESST's trigger-efficiency stage was underpowered and is correctly reported as
inconclusive. So the paper cannot say "this improves anything a physicist
cares about" on real data — only "this reduces a likelihood-aligned residual."
Reviewers in a physics venue will press on this.

A third, softer weakness: CRESST is 7 clusters from 2 runs of one detector
family, so nothing speaks to cross-detector generality.

GWOSC addresses none of these. It addresses a fourth question (is a given Σ
model adequate?) and answers it negatively.

**Verdict: yes, one more real-data anchor is worth it — but only if it closes
Gap 1 and Gap 2.** Adding another residual-only dataset would add pages without
adding an argument.

Requirements for a qualifying dataset:

- R1. **Ground truth**: known injected signal amplitude (or a calibration line
  of known energy), so bias/resolution can be measured, not just residuals.
- R2. **Colored, real detector noise** with an estimable Σ from signal-free
  data disjoint from evaluation.
- R3. **A detector-facing metric** the community already recognizes
  (resolution, denoising score, limit sensitivity, efficiency).
- R4. **Public and citable**, with an existing baseline to compare against.
- R5. **Feasible**: data volume and pipeline effort compatible with weeks, not
  quarters.

---

## 2. Candidate assessment

### 2.1 TIDMAD / ABRACADABRA (arXiv:2406.04378, NeurIPS 2025 spotlight)

Ultra-long SQUID time series from a dark-matter experiment at 10 MS/s. Three
subsets: training (20 files) and validation (20 files) with an **injected fake
sinusoidal signal in a second channel**, and 208 science files without
injection. Ships a predeclared *denoising score*, five deep-learning baselines
(FCNet, PUNet, Transformer, WaveNet, RNN) plus classical baselines (moving
average, Savitzky–Golay), and a complete analysis chain ending in a
community-standard dark-matter limit (Brazil band). CC-BY-4.0.

| Requirement | Assessment |
| --- | --- |
| R1 ground truth | **Yes — the strongest of any candidate.** Injected signals with known amplitude; a weak-signal variant at 1/5 amplitude for a harder task. |
| R2 real colored noise | Yes: real SQUID noise, colored, with 8-bit digitization and non-Gaussian structure. Science files provide signal-free noise for PSD estimation. |
| R3 detector metric | Yes, two: the published denoising score and the downstream axion limit. |
| R4 public + baselines | Yes: HuggingFace + OSDF, scripted download, published baseline numbers to compare against. |
| R5 feasibility | Moderate. Full release is ~TB-scale, but a smoke test needs 1–2 validation files. FFT-based analysis over 2×10⁹ samples/file is chunkable. |

**Why it is a near-perfect fit for Paper 1 specifically.** The signal is a
sinusoid of unknown phase in stationary colored noise — the textbook setting
where the frequency-domain optimal filter *is* the likelihood-optimal
estimator. Better, the unknown-phase manifold is exactly two-dimensional
(cos, sin quadrature), so the paper's "a rank-k learned subspace absorbs a
signal manifold a fixed template cannot represent" argument has a clean,
*analytically known* instance on real data: rank-1 OF at fixed phase should
lose to a rank-2 subspace, and rank-2 should saturate. That is a sharper test
than CRESST's rank-6 empirical choice.

**The strategic payoff.** TIDMAD is an *AI-denoising benchmark*: its baselines
are deep networks trained with MSE-type objectives. If a correctly whitened
linear estimator matches or beats them on the published score, the paper's
thesis lands with maximum force — "most of the claimed deep-learning gain is
recoverable by using the right metric with a linear model." If the deep models
still win by a margin, that is also a clean, honest result and it sets up
Paper 2 (architecture matters once the metric is right). **The experiment is
informative either way**, which is the property you want before committing.

Risks and unknowns to resolve in the smoke test:

- The published denoising score's exact definition and whether it can be
  computed for a linear estimator without reimplementation error (use their
  `benchmark.py` directly — do not reimplement).
- Non-Gaussianity: 8-bit quantization, glitches, and drift. Expect the
  whitened-χ² calibration to fail or partially fail on raw stretches. This is
  not a problem — it *is* the in-domain replacement for GWOSC's boundary
  lesson, with the advantage that injections tell you what the failure costs.
- Known data caveat: channel length mismatch in some files; the README
  prescribes truncating to the first 2×10⁹ samples.
- Stationarity over ultra-long stretches: will need block-wise PSD estimation;
  this is a design choice to predeclare, not a defect.

### 2.2 NuBench (arXiv:2511.13111)

Seven simulated datasets over six neutrino-telescope geometries, with published
predictions and artifacts for ParticleNet, DynEdge, GRIT, DeepIce.

**Not suitable for Paper 1.** Two decisive reasons: (a) it is fully simulated,
so it cannot serve as real-detector evidence; (b) the data are sparse
photon-hit point clouds, not dense waveforms with a noise covariance, so
Paper 1's Σ⁻¹-over-trace-samples machinery does not apply without inventing a
covariance that isn't there. Forcing it would be exactly the kind of overreach
the paper otherwise avoids.

**It is, however, close to ideal for Paper 3** — hierarchical reconstruction,
vertex labels, geometry variation, multiple pretrained representations. Moved
there in `paper3/PAPER3_PLAN.md` §8.

### 2.3 XRISM Resolve / Hitomi SXS (X-ray microcalorimeter, public archive)

Space-based microcalorimeter with public data, where optimal filtering is the
standard pipeline and the detector metric (energy resolution in eV, measured on
known calibration lines) is unambiguous.

- Pros: strongest possible detector-facing metric; same detector physics as
  CRESST so the framework transfers directly; genuinely independent instrument,
  which addresses the "one detector family" weakness.
- Cons: HEASoft/domain pipeline learning curve; event-level products are often
  already optimally filtered, so getting *raw* pulse records with matched noise
  records takes archive navigation; calibration conventions must be respected
  or reviewers from that community will object.
- Verdict: **strong second choice**, higher effort than TIDMAD, better physics
  metric. Consider for the follow-up paper rather than this revision.

### 2.4 More CRESST channels / a second CRESST release

- Pros: cheapest possible; directly attacks the 7-cluster limitation; pipeline
  already exists.
- Cons: same detector family, so it adds statistical power but no independence
  and no ground truth. Closes neither Gap 1 nor Gap 2.
- Verdict: worth doing as WP-E5 hygiene, **not** as the answer to this question.

### 2.5 The internal K-α microcalorimeter dataset (already used in Paper 2)

Real pulses with a known calibration line, so in principle it could deliver an
energy-resolution (FWHM) number — precisely the metric CRESST lacks.

- Pros: cheap, in-house, real, closes Gap 2 partially.
- Cons: not public, so reviewers cannot verify; provenance is currently an open
  issue in the manuscript repo (`OPEN_ISSUES.md` #5); a resolution claim needs
  a careful calibration chain that does not yet exist in the repo.
- Verdict: good *appendix* material if provenance and permission are resolved;
  not a substitute for a public anchor.

### 2.6 Own DELight / LAMCAL prototype data

Same profile as 2.5: real and detector-facing, but non-public and entangled
with the naming/permission issues already flagged in `OPEN_ISSUES.md` #4.

### 2.7 Options considered and rejected

- LZ/XENONnT public releases: raw waveforms generally not public.
- G2Net/Kaggle GW datasets: synthetic; no advantage over S1–S9.
- Pulsar/radio matched-filtering datasets: methodologically apt but off-domain
  for a detector-physics venue; would read as padding.

---

## 3. Recommendation

**Top choice: TIDMAD.** It is the only candidate that closes both gaps at once,
it is public with published baselines, and — decisively — it is informative
whichever way the result falls.

**Second: XRISM/Hitomi**, if a stronger physics metric (eV resolution on an
independent instrument) is worth the extra pipeline effort. Better suited to a
follow-up.

**Do not use NuBench for Paper 1**; route it to Paper 3.

### 3.1 What to do with GWOSC — demote, do not delete

Deleting a negative result after adding a new dataset invites the suspicion
that it was dropped because it was inconvenient. But GWOSC currently occupies a
full main-text section (§7.4 + Table 3) for a result that supports no claim,
and it invites the distracting misreading "they tried LIGO and failed."

Recommended restructure:

- Main text: keep **three to four sentences** in §7 — implementation checks
  passed (GWpy PSD agreement, shared-FIR identity), global and predeclared
  64 s local PSD models failed held-out calibration on one interval, therefore
  event and injection scores are diagnostic only. State plainly that the stage
  was *designed* to locate the boundary, not to pass.
- Move Table 3 and the split/window details to an appendix.
- Give the freed main-text space to the new anchor.
- If TIDMAD also yields a calibration failure on raw stretches (likely), the
  discussion gains a much better sentence: the same boundary appears in a
  second, independent instrument, in-domain, and with injections quantifying
  what it costs.

Only consider full removal if TIDMAD's boundary result fully subsumes GWOSC's
lesson *and* a reviewer-facing note records that GWOSC was completed and
archived rather than abandoned.

---

## 4. Smoke test — yes, and here is the protocol

Do not commit to a full TIDMAD stage before a smoke test. The purpose is to
kill the option cheaply if the score or the noise model doesn't cooperate.
Budget: a few days. Predeclare the gates below *before* running.

### Stage 0 — data and pipeline sanity (½ day)

1. `python download_data.py -t 0 -v 2 -s 2` → two validation files (injected
   ground truth) and two science files (signal-free noise). Confirm file
   layout, sampling rate, dtype, and apply the README's 2×10⁹-sample
   truncation.
2. Run the reference chain unmodified: `inference.py -m savgol` and
   `benchmark.py -m none/-m savgol`.
   **Gate G0: reproduce the published `none` and `savgol` denoising scores.**
   If this fails, stop — the score cannot be compared and the stage is not
   worth doing.

### Stage 1 — noise model and implementation identity (1 day)

3. Estimate a one-sided PSD from science-file stretches using the repo's
   estimator; cross-check against `scipy.signal.welch` and, if available, the
   TIDMAD reference conventions.
4. Verify the whitening/normalization convention end-to-end on a synthetic
   sinusoid injected into real noise at known amplitude.
   **Gate G1: recovered amplitude unbiased to <1% and OF variance within 10%
   of the PSD-predicted bound on stationary-selected stretches.**

### Stage 2 — the three scientific probes (1–2 days)

5. **Metric reversal with truth (closes Gap 1).** On held-out validation
   segments, compare a Σ⁻¹-whitened linear estimator against the same estimator
   family trained/selected under isotropic MSE. Report raw MSE, weighted
   residual, and — the new thing — **error against the known injected
   amplitude**.
   **Gate G2: the MSE-preferred estimator is worse in amplitude error.** This
   is the first real-data confirmation of the paper's central claim.
6. **Rank/manifold (sharpens the CRESST story).** Unknown phase spans a
   2-dimensional quadrature subspace. Sweep rank 1→4 of a weighted subspace fit
   on segmented windows.
   **Gate G3: rank-2 clearly beats rank-1 and rank≥3 saturates**, matching the
   analytic prediction.
7. **Calibration boundary (replaces GWOSC in-domain).** Whitened χ²/dof and
   score dispersion on held-out science stretches, with a stationary synthetic
   control that must pass.
   **Expected: control passes, real data fails or partially fails.** Report as
   a boundary, with injections quantifying the cost — which GWOSC could not do.

### Stage 3 — detector-facing metric (closes Gap 2)

8. Compute the published denoising score for the linear noise-aware estimator
   via `benchmark.py`, alongside the published deep baselines.
   **Gate G4: the linear estimator is on the published scoreboard** — whatever
   its rank. Optionally extend to the limit-setting chain (`brazilband.py`) if
   time allows; that is the strongest possible physics-facing statement but is
   not required for the paper.

### Decision rule

- G0 and G1 fail → abandon TIDMAD, fall back to XRISM or to strengthening
  CRESST (WP-E5) and keep GWOSC as-is.
- G0, G1 pass and G2 passes → **commit**; TIDMAD becomes Paper 1's second
  real-data section and GWOSC is demoted to an appendix.
- G2 fails (no reversal in amplitude error on real noise) → this is a genuinely
  important negative result about the scope of the claim. Do not bury it;
  either report it as a scope limitation or determine that the noise is too
  close to white in-band for the effect to exist, and say so.

### Predeclaration checklist (freeze before Stage 2)

Segment length and count; stationarity-selection rule for stretches; PSD
estimation windows and their disjointness from evaluation; rank grid; the
smallest relevant effect for G2; the number of seeds/segments and CI type; and
the statement that no threshold will be adjusted after seeing test results.

---

## 5. Effort and sequencing

| Item | Effort | Blocking? |
| --- | --- | --- |
| TIDMAD smoke test (Stages 0–3) | ~3–5 days | decides whether the stage happens |
| Full TIDMAD stage if committed | ~2–4 weeks incl. predeclaration and archival | yes, for the revised paper |
| GWOSC demotion to appendix | ~1 day of LaTeX | no |
| CRESST WP-E5 hygiene | opportunistic | no |
| XRISM exploration | ~2–4 weeks just to first light | no — follow-up |

Sequencing: run the smoke test *in parallel* with the manuscript corrections
already planned (WP-P1 S5 provenance, WP-P3 Study B descope). Those are
independent and short. Do not restructure §7 until G2 has returned an answer.
