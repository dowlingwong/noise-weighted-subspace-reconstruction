# Paper 2 Noise-Module Integration Plan + Paper 3 (Junjie) Outline

Date: 2026-07-26. Written after reading `paper2_nonlinear_reconstruction.pdf`
end to end (framework §2, NFPA theory §3–4, NFPA experiments §5, ResNet/
Transformer §6, Xenon-TPC coverage case study §7, experimental program §8) and
auditing the noise module.

---

## Part A — Verdict: integrate or separate?

**Integrate the noise module into Paper 2. Keep the Junjie program as a
standalone Paper 3.**

Why the noise module belongs *inside* Paper 2:

Paper 2's own §5.5 lists six required NFPA sweeps and §8 lists five program
axes. Read them against the module's capabilities and the match is nearly
one-to-one:

| Paper 2 requirement | Module capability | Currently used? |
| --- | --- | --- |
| §5.5.2 noise-color sweep (white/pink/Brownian/**measured detector-like PSD**) | analytic colours + custom tabular PSD + Al₂O₃ component spectra | colour only, ad hoc |
| §5.5.3 channel-count sweep C = 2,4,8,16 | `MultiChannelNoiseGenerator` (independent / shared-private / low-rank / target-CSD) | no |
| §5.5.5 **nonseparable-template test**, sweep α | target-CSD path: interpolate between a Kronecker-separable CSD and a non-separable one, with a *declared* separability index | no — this is the weakest current sweep |
| §8 "noise: matched vs mismatched PSD/covariance between training and evaluation" | train under Σ_train, evaluate under perturbed Σ_eval; perturbation family is exact and parameterized | no |
| §8.1 coverage experiment latents (A, t₀, p, γ) + **restricted noise/PSD state** | the noise state *is* a latent the module can restrict or vary independently | no |
| §8.2 coverage-corrected training, "Stage-I degraded simulation" | the module is precisely a controllable approximate simulator; degradation need not be an ad hoc low-pass | no |
| §8.3 detector-design metrics (PSD shape, sampling frequency, trace length, channel layout, channel correlations, readout noise) | every one of these is a module parameter; Study B already computes OF-information deltas | no |
| §5.5.6 ≥20 seeds, CIs | seedable, splittable RNG streams; deterministic streaming | partially |

So the module is not an add-on to Paper 2 — it is the missing instrument that
converts Paper 2's §5.5/§8 wish-lists into runnable, *predeclared* experiments
with exactly-known ground truth. Without it, the nonseparability sweep and the
noise-mismatch axis cannot be done honestly (you cannot claim a controlled
separability violation if you do not control the covariance that defines
separability).

Why the Junjie program should *not* be folded into Paper 2:

1. **Different object of study.** Paper 2 evaluates *reconstruction* under a
   likelihood metric: given (Σ⁻¹, F, P_train→P_target), how well is the signal
   recovered. The Junjie list evaluates *representations*: embedding
   separability, clustering robustness, perturbation stability, score
   calibration, anomaly probes. These are diagnostics *of* a learned
   representation, not reconstruction claims. Merging them makes the
   contribution statement mushy — the classic "two papers stapled together"
   failure that reviewers punish.
2. **Paper 2 is already carrying a lot.** It is a framework paper with one
   validated case study (NFPA), one external illustrative case study (Xenon
   TPC), two framework-derived extensions (VAE, Transformer), and a program
   section. Its §1 scope note already concedes only NFPA is validated. Adding
   a second empirical program would push it past what one paper can defend.
3. **Different domain and collaborators.** "Hierarchical particle
   reconstruction" and "performance near interaction vertices" imply a
   multi-stage reconstruction chain and geometry/vertex labels that the current
   single-channel K-α and two-channel synthetic settings do not have. That is
   Junjie's data domain, not this repo's.
4. **The dependency runs the right way.** Paper 2's §8 explicitly says it
   "generates, rather than merely describes, a concrete experimental program"
   and does not report those experiments. Paper 3 executing two of the Table 7
   rows (noise-mismatch, coverage) on a real hierarchical-reconstruction
   pipeline is the *ideal* follow-up citation for Paper 2 — it validates the
   framework's productivity claim without inflating Paper 2 itself.

Coherence check: does Paper 2's story survive the integration? Yes, and it
improves. The narrative becomes: Paper 1 fixes the metric → Paper 2 fixes the
architecture and coverage axes and shows they are separately necessary → to
test that, you need a noise simulator whose Σ you control exactly, which
§(new) supplies → NFPA is validated with it → the remaining program (§8) is
handed to Paper 3. The module enters as *methodology*, in service of the
theorems, which is exactly the framing you proposed.

---

## Part B — Paper 2 integration plan (manuscript)

### B1. New section: "Controlled noise instrument" (~2 pages, place after §4, before §5)

Purpose: define the experimental instrument before the experiments that use
it. Do not write it as a software-features tour; write it as the *definition of
the ground truth* the theorems are tested against.

Content:

1. **Sampling contract** (one equation block, imported from Paper 1's
   Appendix G): interior circular-complex-Gaussian rFFT bins with
   E|X_k|² = f_s N P_k / 2, real DC/Nyquist endpoints, odd-N rule, and the
   discrete one-sided variance identity. State once that FFT synthesis defines
   a periodic/circulant finite-grid process and that an exact finite-Toeplitz
   sampler exists as a separate contract. One sentence, not a page — cite
   Paper 1.
2. **Multichannel covariance control** — the piece Paper 2 actually needs:
   given a target cross-spectral-density S(f) ∈ C^{C×C} per bin, the generator
   samples exactly from it (Hermitian PSD-checked, real endpoints, low-rank
   factor path). Then define the **separability index**: for a target CSD
   built as S_α(f) = (1−α)·S_sep(f) + α·S_nonsep(f), with S_sep = R_c ⊗ R_t
   Kronecker-separable, α ∈ [0,1] parameterizes departure from the NFPA prior.
   Give the exact definition used (e.g. α, plus the realized best-Kronecker
   approximation error ε_kron of the resulting Σ, since ε_kron is what
   Eq. (43) actually bounds against).
3. **Perturbation family** for the noise-mismatch axis: a *predeclared* list
   with severity parameters — PSD amplitude scale, spectral tilt/slope,
   narrowband line injection, CSD rank/coherence change, drift/nonstationarity,
   glitches, heavy tails. State which are Gaussian-stationary (so the
   Mahalanobis loss remains the exact likelihood) and which deliberately
   violate the model (so a calibration failure is expected and is the point).
4. **What the instrument cannot do** — half a paragraph, and it protects you:
   a simulator validates *conditional* claims. Agreement with a detector's
   measured PSD does not make the simulated ensemble the detector's noise
   distribution; and diagonalizing a total covariance yields statistically
   orthogonal modes, not physical sources. Point to Paper 1 §9.

### B2. Rewrite §5 (NFPA experiments) around the instrument

- **§5.5.5 nonseparable-template test → replace with the CSD-based sweep.**
  Current draft perturbs the *template* (s = vtᵀ + αR). Better: perturb the
  *noise covariance* separability and/or the template jointly, and report NFPA
  vs EMPCA principal angles and whitened χ² as a function of the realized
  ε_kron rather than the nominal α. This makes Figure/Table directly
  comparable to the conjectured bound sinΘ ≤ F(ε_kron, Δλ) in Eq. (43) — the
  sweep stops being a vague "cost of the prior" plot and becomes empirical
  evidence for the bound you are conjecturing. This is the single highest-value
  change in Paper 2.
- **§5.5.1–5.5.4** (SNR, colour, channel count, rank pair): run through the
  module so every cell has an exact Σ; ≥20 seeds; report means with bootstrap
  CIs. The current SNR ≈ 914 setting must become one point on a grid, not the
  only point.
- **§5.2 Table 3**: re-export raw MSE at full precision (the `< 10⁻⁵` entries
  are unpublishable), and report whitened χ² as mean ± CI over seeds. Keep the
  cautious framing that the 0.1% EMPCA–NFPA gap is not yet a significant
  ordering — with 20 seeds you will finally know whether it is.
- **§5.3 Table 4**: keep the honest 61.7° mean-angle discussion. Add the
  ε_kron value for that configuration so the reader can see the constraint is
  active for a quantified reason.
- Split the dense Figure 1 into the three publication figures the draft
  already asks for; move the panel grid to an appendix.

### B3. Add the noise-mismatch axis to §8 as a *reported* experiment

§8's Table 7 lists "Noise mismatch" as a hypothesis. With the module you can
promote it from hypothesis to result cheaply, on synthetic data, within Paper
2's existing linear scope: train NFPA/EMPCA under Σ_train, evaluate under
Σ_eval on a severity grid, and show that the whitened residual degrades
predictably while the *ordering* versus MSE-trained models is preserved until
severity is large. That converts one Table 7 row into evidence and
strengthens the "the framework generates experiments" claim without touching
the neural results.

Keep the other rows (architecture ladder, coverage, coverage-corrected
training) as program, and add a forward reference: "an empirical instance of
the coverage and noise-shift rows on a hierarchical reconstruction pipeline is
the subject of a companion study" → Paper 3.

### B4. §8.3 detector design — connect to Study B

Paper 2 §8.3 defines the design metric g_z = J_s(z)†Σ⁻¹J_s(z) and Table 8
lists design factors (PSD shape, sampling frequency, trace length, channel
layout, channel correlations, readout noise). Paper 1's Study B already
computes exactly the rank-one case of this (OF information under component
removal/rescaling). Say so explicitly: Study B is the k=1 instance of Table 8's
metric, and the module's resampling/alias-fold machinery is what lets you vary
sampling frequency and trace length in that metric. This is where "sensitivity
attribution" becomes a *design* contribution rather than a noise-decomposition
claim — and it is the honest answer to "decomposition for detector design".

### B5. Prerequisites (from the experiment-repo plan)

Nothing in B1–B4 depends on the neural results. But if Paper 2 is to keep any
real-data neural content, WP-E7 repairs (seeding, single test-set touch,
timing metric, Transformer encodings) are blocking. Recommended: build Paper 2
on the linear/NFPA + instrument core, and demote all current real-data neural
runs to a clearly-labeled appendix or drop them.

---

## Part C — Paper 3 outline (with Junjie)

**Working title.** "Noise and Coverage Diagnostics for Learned Representations
in Hierarchical Particle Reconstruction" — usable, but the operative words
should be *diagnostics under controlled noise shift*. An alternative that
states the mechanism: "Are Learned Detector Representations Stable, Covered,
and Calibrated? Controlled Noise-Shift Diagnostics for Hierarchical
Reconstruction."

**One-sentence thesis** (everything in the paper serves this):

> A learned representation used in hierarchical reconstruction should preserve
> identifiable physical coordinates under adequate training coverage, and
> remain stable and calibrated under a predeclared family of noise shifts; we
> supply diagnostics that measure each property, and show which failures are
> attributable to coverage versus noise-domain shift versus metric mismatch.

**Why this is publishable as its own paper.** It is not a new architecture and
not a new detector result; it is a *measurement methodology* paper. Its
contribution is (i) a predeclared perturbation library with exactly known
severity, (ii) a set of diagnostics with null calibration, and (iii) an
empirical finding about which diagnostics detect which failure — including the
negative result that some "anomaly" scores are really noise-mismatch
detectors. That last finding alone justifies the paper.

### Section outline

1. **Introduction.** Learned representations are entering hierarchical
   reconstruction chains (clustering → aggregation → vertex/particle
   inference). Their failure modes are not the failure modes of a fitted
   template. State the three-way diagnosis (metric / coverage / noise shift)
   and inherit the tuple (Σ⁻¹, F, P_train→P_target) from Paper 2 in one
   paragraph.
2. **Setup and instrument.** The detector/pipeline (Junjie's domain), the
   representation(s) under test, and the controlled noise instrument: exact
   Σ, exact severity, matched vs mismatched train/eval covariance. Predeclare
   the perturbation library and severity grid here, before any result.
3. **Diagnostic battery** — one subsection each, every one with a null:
   - *Embedding separability.* Held-out linear probes for signal coordinates
     (energy, time, position/vertex, sharing, shape) vs probes for the noise
     state as a nuisance label. Target: signal decodable, noise state at
     chance. Null: label permutation.
   - *Clustering robustness.* ARI / variation-of-information between clean and
     perturbed clusterings across the severity grid; bootstrap CIs; permuted-
     label null.
   - *Perturbation stability.* Latent displacement, local Jacobian norm,
     principal-angle rotation, nearest-neighbour retention, and downstream
     amplitude/timing shift as functions of severity. This is the strongest
     first result because severity is exactly known.
   - *Uncertainty / score calibration.* Reliability diagrams, interval
     coverage, PIT / standardized residuals, false-positive rate under matched
     vs shifted noise. Paper 1's GWOSC failure is the worked cautionary
     example: implementation identity ≠ calibration.
   - *Performance near interaction vertices.* Stratify every metric above by
     vertex distance and channel-sharing topology. **Gate:** this section
     requires real geometry/vertex labels; do not substitute waveform
     broadening for a vertex claim. If the labels are unavailable, cut the
     section rather than proxy it.
4. **Injection-probe / calibration-verification study (the "4th application").**
   A blind mixture: nominal background, nuisance-only shifts (hard foils),
   known signal injections, and held-out synthetic anomalies. Predeclared
   type-I error. Central question: *does an anomaly score exceed the best
   noise-mismatch baseline?* Report the answer honestly either way. Frame as
   blind synthetic anomalies, never as evidence for new physics.
5. **Fitted correlated-noise categorization (the "ask IMS" thread).** Present
   as a *constrained model class*, not discovery: low-rank frequency-dependent
   CSD factors + private PSDs, optionally constrained by channel topology or
   auxiliary sensors; validated by held-out predictive likelihood, coherence/
   phase agreement, and downstream OF residuals. Report factor-rotation
   non-identifiability explicitly.
6. **What linear/second-order methods can and cannot do** (the 线性能做到哪一步
   section — this is a genuine contribution if written carefully). Can:
   whitening/decorrelation; frequency-resolved CSD eigenmodes, coherence,
   phase, shared/private variance, common-mode rank; constrained component
   fits under known spectral families; auxiliary-channel regression;
   sensitivity attribution (Fisher/OF-variance/threshold deltas under declared
   interventions). Cannot: recover uniquely labeled physical sources from one
   total Σ — Σ = Σ₁ + Σ₂ has infinitely many PSD-cone splits and low-rank
   factors are rotation-degenerate. Well-posedness requires added structure:
   parameterized spectral families, auxiliary/witness sensors, multiple
   operating conditions or hardware interventions, injected calibration lines,
   or non-Gaussian/temporal signatures. **Reframe detector-design value as
   sensitivity attribution under declared models, not source discovery.**
7. **Discussion / limitations.** Which diagnostics generalize, which are
   detector-specific, what a simulator can and cannot certify.

### Predeclared experimental protocol (agree with Junjie up front)

- ≥20 independent seeds/splits; group-aware real-data splits (run, acquisition
  block, detector), never naive random event splits.
- Perturbation library, severity grid, and success thresholds frozen before
  any test-set evaluation; locked test set touched once.
- Matched compute, parameter count, and bottleneck across representations.
- Every diagnostic reported with a null (permutation or matched-noise control).
- The noise model fitted only on calibration data, never on evaluation data.

### Division of labor (suggested)

- Your side: noise instrument, perturbation library and severity calibration,
  likelihood-geometry diagnostics (whitened residual, principal angles, Fisher/
  pullback), sensitivity attribution, and the linear-limits section.
- Junjie's side: the hierarchical reconstruction pipeline, representation(s)
  under test, vertex/geometry labels, clustering and downstream physics
  metrics.
- Joint: the blind injection protocol and its predeclaration document; this is
  the piece that must be co-signed before any result is looked at.

### First deliverable to propose to Junjie (small, concrete)

A single perturbation-stability figure on his existing pipeline: one
representation, one severity grid from the module (PSD scale, tilt, lines,
CSD-rank change), reporting latent displacement and one downstream physics
metric with seed CIs. If that figure is informative, the rest of the battery
follows mechanically. If it is flat, you learn early that the collaboration's
center of gravity should be calibration/injection rather than stability.

---

## Part D — What to say to Junjie about noise decomposition

Short version, defensible: decorrelation is exact and useful; *decomposition
into labeled physical sources from a total covariance alone is ill-posed, not
merely hard*. It becomes well-posed only with added structure (known spectral
families, witness sensors, interventions, injected lines, non-Gaussian
signatures). For detector design the productive target is sensitivity
attribution under a declared model — how each modeled component changes OF
variance, Fisher information, and threshold — which is actionable without
claiming that covariance eigenvectors are physical causes. Paper 1 §9 already
states this boundary in publishable language and can be quoted directly.
