# Paper 3 Plan — Noise and Coverage Diagnostics for Learned Representations in Hierarchical Particle Reconstruction

Status: planning document (v2, 2026-07-26; v1 revised to integrate NuBench as
the public backbone). Joint work with Junjie (Stanford). Companion to Paper 1
(likelihood geometry) and Paper 2 (architecture and coverage). This document is
the negotiating instrument for the collaboration: it fixes the thesis, the
claim set, the datasets, the protocol, and the division of labor before any
result is produced.

---

## 1. Executive summary

**Thesis (one sentence, everything serves it).**

> A learned representation used inside a hierarchical reconstruction chain
> should preserve identifiable physical coordinates under adequate training
> coverage, and remain stable and calibrated under a predeclared family of
> noise shifts; we supply diagnostics that measure each property separately,
> and show empirically which failures are attributable to coverage, to
> noise-domain shift, and to metric mismatch.

**Type of paper.** Measurement methodology, not a new architecture and not a
new detector result. It is judged on whether the diagnostics are principled,
predeclared, null-calibrated, and informative — including when they return
negative answers.

**Three contributions.**

1. A *controlled shift instrument*: a perturbation library with exactly known
   severity, in two matched flavours — waveform-level (PSD/CSD, from the
   upgraded noise module) and hit-level (dark rate, efficiency, time jitter,
   correlated-noise bursts, for point-cloud detectors).
2. A *diagnostic battery* with explicit nulls: embedding separability,
   clustering robustness, perturbation stability, score calibration, and
   vertex-stratified performance.
3. An *empirical attribution result*: which observed degradation is coverage
   failure, which is noise-domain shift, and which is metric mismatch — plus
   the falsification test that an "anomaly" score is not merely a
   noise-mismatch detector.

**Two-track evidence design (new in v2).** Track A is **NuBench**: public,
simulated, with four *already-trained* representations across six detector
geometries — the reproducible public backbone on which the diagnostics are
defined and calibrated. Track B is **Junjie's pipeline / real data**: the
real-noise validation where the diagnostics have to survive contact with an
instrument. A methodology paper that develops its battery on a public
benchmark and then transfers it is far more citable than one tied to a single
private pipeline.

**Headline risk we deliberately accept.** Some diagnostics will come back
uninformative. That is a publishable finding for a methodology paper, provided
the nulls and predeclaration are in place. It is *not* publishable if the
protocol is chosen after seeing the results.

**What this paper does not claim.** No unique physical noise decomposition; no
detector-sensitivity improvement; no new-physics anomaly discovery.

---

## 2. Positioning against Papers 1 and 2

| | Paper 1 | Paper 2 | Paper 3 (this) |
| --- | --- | --- | --- |
| Object | reconstruction estimator | reconstruction model class | learned *representation* |
| Question | what is the right metric? | which architecture and coverage? | is the representation stable, covered, calibrated? |
| Evidence | identities + real-data residuals | theorems + NFPA case study | empirical diagnostics under controlled shift |
| Data | CRESST, TIDMAD, GWOSC boundary | synthetic + Xenon-TPC case study | NuBench (public) + Junjie pipeline (real) |

The dependency runs forward: Paper 2 §8 explicitly *generates* an experimental
program (Table 7: metric reversal, coverage violation, architecture ladder,
noise mismatch, coverage-corrected training) and does not report it. Paper 3
executes the coverage, noise-mismatch, and architecture-ladder rows on real
pipelines. That makes Paper 3 the empirical validation of Paper 2's
productivity claim without inflating Paper 2 itself.

**A second, sharper bridge (new in v2).** Paper 2 §8.3 Table 8 lists detector
design factors evaluated by the Fisher/pullback metric g_z = J_s(z)†Σ⁻¹J_s(z);
one of those factors is **channel layout**. NuBench ships seven datasets across
**six distinct detector geometries** with the same simulation chain and the
same model families. That is a ready-made empirical instance of the
channel-layout row: how does representation quality, stability, and
vertex-region performance change when only the geometry changes? No other
public resource offers that axis for free.

**Scope statement that must appear early (honesty gate).** Paper 1's metric is
a Gaussian Σ⁻¹ over dense trace samples. Neutrino-telescope data are sparse
photon-hit point clouds; the correct likelihood there is hit-level (Poisson
arrival plus timing jitter), not a dense Mahalanobis form. Paper 3 therefore
inherits the *separation* of metric / architecture / coverage from Papers 1–2,
and explicitly does **not** inherit the specific Mahalanobis form. Saying this
in §1 costs one paragraph and prevents the most obvious reviewer objection.

---

## 3. Claim set (predeclared, with falsifiers)

Each claim gets one experiment, one metric family, one null, and one condition
under which we report it as failed. The last column records which track can
support the claim.

| # | Claim | Primary metric | Null / control | Falsifier | Track |
| --- | --- | --- | --- | --- | --- |
| C1 | The representation encodes signal coordinates linearly decodable on held-out data | held-out linear-probe R²/accuracy per coordinate | label permutation | probe no better than permuted-label baseline | A + B |
| C2 | The representation does not encode the noise state as a recoverable nuisance coordinate | noise-state probe accuracy | chance under matched noise | noise state decodable well above chance ⇒ noise-entangled representation | A + B |
| C3 | Clustering structure is stable under bounded shift | ARI / variation of information vs severity | permuted-label null; matched reference | stability curve indistinguishable from the null | A + B |
| C4 | Representation displacement grows predictably and boundedly with exactly known severity | latent displacement, local Jacobian norm, principal-angle rotation, k-NN retention | zero-severity replicate spread | displacement dominated by seed noise, or chaotic in severity | A + B |
| C5 | Scores/uncertainties are calibrated under matched noise and degrade diagnosably under shift | reliability diagram, interval coverage, PIT, FPR at fixed threshold | matched-noise control that **must** pass | control fails ⇒ bug, halt and fix | A + B |
| C6 | Coverage failure is not repaired by the correct metric | error vs held-out latent coordinate, full-minus-restricted gap | equal-n unrestricted control | restricted model matches full-coverage model on the held-out coordinate | A + B |
| C7 | Anomaly scores exceed the best noise-mismatch baseline | AUC / power at predeclared FPR vs a mismatch-only detector | nuisance-only foils in the blind mixture | not separable from the mismatch detector ⇒ report negative | A + B |
| C8 | Performance degrades systematically near interaction vertices, and diagnostics detect it | all metrics stratified by vertex distance and sharing topology | far-from-vertex stratum | no stratification effect above CI width | **A** (exact truth) + B if labels exist |
| C9 | Diagnostic conclusions transfer across detector geometry | rank correlation of diagnostic verdicts across geometries | within-geometry seed spread | verdicts reorder across geometries ⇒ diagnostics are geometry-specific, report as scope limit | **A only** |

C5's control is a *halt gate*, not a result: if the matched-noise control fails
calibration, we have a bug, and no real-data calibration number is reported
until it passes. This is Paper 1's GWOSC lesson carried over cheaply.

**C8 is rescued by NuBench.** In v1 this claim was gated on Junjie having real
vertex labels and was scheduled for deletion otherwise. NuBench is simulated,
so vertex truth is *exact* — the vertex-region claim becomes fully supported on
Track A regardless of what Track B can provide. This is the single largest
change from v1 and it removes the plan's biggest structural risk.

**C9 is new and only NuBench can support it.** "Do these diagnostics say the
same thing about the same model family on a different detector geometry?" is a
question a methodology paper should answer, and six geometries with shared
simulation and shared model families is precisely the setting that permits it.

---

## 4. Datasets

### 4.1 Track A — NuBench (public backbone)

`graphnet-team/NuBench`, arXiv:2511.13111. Seven datasets simulated across six
distinct detector geometries (Cluster, Flower S/L/XL, Hexagon, Hexagon Ice LE,
Triangle) using the Prometheus chain, distributed in SQLite and Parquet with
GraphNeT dataset classes. Ships **test-partition predictions and full model
artifacts** (weights, pickled models, configs) for four representation
families: ParticleNet, DynEdge, GRIT, DeepIce.

Why it is the right backbone:

1. **Four pretrained representations, no training required.** A diagnostics
   paper's dominant cost is usually producing the representations. NuBench
   removes it. We load artifacts, extract embeddings, and diagnose.
2. **Six geometries** → the C9 transfer claim and Paper 2's channel-layout
   design row, for free.
3. **Exact truth for every latent** (energy, direction, vertex position,
   interaction type) → clean coverage restrictions (C6) and exact
   vertex stratification (C8).
4. **Public and reproducible** → anyone can rerun the battery, which is what
   makes a methodology paper get used rather than admired.
5. **Predictions are published**, so our diagnostics can be sanity-checked
   against the benchmark's own performance numbers before we trust them.

Limitations, stated up front:

- **Simulated.** Noise shift must be *injected*, not observed. Track A can
  therefore establish that a diagnostic detects a known shift; only Track B can
  establish that it detects shifts that real instruments actually exhibit.
  Write this as the explicit reason the paper has two tracks.
- **Point clouds, not waveforms.** The Σ⁻¹ machinery does not apply directly;
  see §4.3 for the hit-level perturbation library that replaces it.
- **Model artifacts are checkpoints, not training pipelines** for our purposes.
  Any experiment that needs *retraining* (notably the C6 coverage experiment)
  requires re-running GraphNeT training on restricted support. Budget for this
  explicitly: C1–C5 and C8–C9 are checkpoint-only and cheap; C6 needs training
  runs.
- Datasets are "inspired by, but not strictly identical to" existing
  telescopes, so no claim about any named experiment.

### 4.2 Track B — Junjie's pipeline / real data

Preferred if it has real (non-simulated) noise and vertex labels, because
"does the diagnostic fire on shifts the instrument actually shows?" is the
question Track A cannot answer. Scope, labels, and access to be fixed at M0.

### 4.3 The perturbation library, in two matched flavours

The instrument must speak both data languages. The key design decision is that
the two libraries are *matched by intent*, so a diagnostic's response can be
compared across tracks:

| Intent | Waveform flavour (noise module, Track B / Papers 1–2) | Hit-level flavour (Track A / NuBench) |
| --- | --- | --- |
| Overall noise scale | PSD amplitude scale | dark/noise-hit rate scaling |
| Spectral shape change | spectral tilt / slope | wavelength- or time-constant-dependent efficiency change |
| Narrowband contamination | injected line at frequency f | periodic or flasher-like correlated hit bursts |
| Correlated-noise structure | CSD rank / coherence change | spatially correlated noise across a string or module group |
| Nonstationarity | drift, evolutionary PSD | time-varying noise rate within an event window |
| Heavy tails / outliers | glitches, heavy-tailed samples | afterpulsing, large-charge outlier hits |
| Sensor loss | channel dropout | DOM/PMT dropout and dead-time |
| Calibration error | per-channel gain miscalibration | per-DOM gain and time-offset miscalibration |

Severity is parameterized on a log-spaced grid in every row, and each row is
tagged **likelihood-preserving** (the model's assumed noise family still holds)
or **model-violating** (it does not). That tag is what lets the paper separate
"the representation is fragile" from "the assumed noise model was wrong" —
the distinction Paper 1 spent its GWOSC section establishing.

Implementation note: the hit-level library is new code and is the main
Track A engineering cost. Build it as a GraphNeT-compatible transform applied
to the dataset at load time, so the same checkpoint can be scored under any
severity without retraining.

---

## 5. Section outline

1. **Introduction.** Learned representations are entering hierarchical
   reconstruction chains (hit → cluster → aggregate → vertex/particle). Their
   failure modes differ from those of fitted templates: they can be
   noise-entangled, coverage-limited, or mis-calibrated while looking fine on
   in-distribution metrics. State the three-way diagnosis, inherit the tuple
   (metric, architecture, coverage) from Paper 2, and state the scope limit of
   §2 (no dense Mahalanobis form in the point-cloud setting).
2. **Related work.** Representation-quality metrics (probing, CKA, clustering
   stability), robustness/domain-shift literature, detector calibration
   practice, and why none of them alone separates coverage from noise shift.
3. **Setup and datasets.** Two-track design; NuBench geometries and models;
   Track B pipeline; label sets; group-aware splits.
4. **The controlled shift instrument.** Both perturbation flavours, the
   severity parameterization, and the likelihood-preserving vs model-violating
   tag. Half a page on what a simulator cannot certify.
5. **Diagnostic battery.** One subsection per diagnostic (C1–C4, C8), each with
   definition, null, and estimator/CI type.
6. **Calibration and blind injection probe.** C5 and C7.
7. **Coverage experiment.** C6 with equal-n controls, following Paper 2 §8.1.
8. **Cross-geometry transfer of diagnostic verdicts.** C9 — NuBench only.
9. **What linear/second-order methods can and cannot do.** (§8 below.)
10. **Results synthesis: attribution.** One table assigning each observed
    degradation to coverage / noise shift / metric mismatch, with the evidence
    for each assignment.
11. **Limitations and outlook.**

---

## 6. Experimental program

Each experiment states its Track A (NuBench) instantiation explicitly, because
that is the part that can start immediately.

### E1 — Perturbation stability (do this first)

*Track A:* take one NuBench dataset (recommend Hexagon as the "default"
geometry) and one model family; apply the hit-level library on a log-spaced
severity grid; report latent displacement, k-NN retention, and one downstream
reconstruction metric (energy or direction resolution), with a zero-severity
replicate band.

*Track B:* same, with the waveform library.

**Go/no-go for the collaboration:** if the curves separate from the
zero-severity band and order sensibly by severity, the battery is worth
building. If flat, pivot the collaboration to calibration and injection (E4).

### E2 — Embedding separability and noise entanglement

Cross signal labels (energy, direction, vertex position, interaction type) with
independent noise-state labels; held-out linear probes for both; report the gap
between signal decodability and noise decodability, under matched and
mismatched noise. Four NuBench model families give an immediate comparison
across architectures at fixed data — the architecture-ladder row of Paper 2
Table 7, obtained without training anything.

### E3 — Clustering robustness

Cluster reference embeddings, perturb, recompute; ARI/VI vs severity with
bootstrap CIs and permuted-label nulls. Stratify by event multiplicity — in the
neutrino-telescope setting this is naturally the number of hit DOMs, which is
also a proxy for energy, so stratify on both and report the confound.

### E4 — Calibration and blind injection probe

Blind mixture: nominal, nuisance-only shifts (the hard foils), known signal
injections, held-out synthetic anomalies. Predeclared type-I error. Primary
question is C7: does the anomaly score beat a detector that only sees the
nuisance shift? Report either way. NuBench's simulated nature is an advantage
here — the mixture composition is exactly known and can be sealed.

### E5 — Coverage (Paper 2 §8.1 instantiated)

Restrict one latent at a time with *equal training n*, evaluate on a
full-coverage test set. On NuBench the natural restrictions are: narrow energy
band (and the Hexagon Ice LE dataset gives a low-energy regime for free),
restricted zenith/azimuth, restricted vertex region (e.g. fiducial-only
training evaluated on the full volume — physically the most interesting), and
restricted noise state. **This is the only experiment requiring retraining**;
budget accordingly and consider restricting it to one geometry and two model
families.

### E6 — Vertex-stratified performance (now fully supported)

Stratify E1–E4 by true vertex distance to the instrumented volume boundary and
by hit-sharing topology across strings/modules. NuBench gives exact vertex
truth, so this is a first-class result rather than a gated extra. This is the
section that most directly serves the title's "near interaction vertices".

### E7 — Cross-geometry transfer (C9, NuBench only)

Run the reduced battery (E1, E2, E4) on all six geometries for one model
family, and on one geometry for all four model families. Ask whether the
diagnostic *verdicts* (which model is more stable, which coordinate is most
fragile) are geometry-invariant. Report rank correlations with CIs. Either
answer is interesting: invariance means the diagnostics are portable; reordering
means representation quality is geometry-specific, which is itself a
detector-design statement and connects back to Paper 2 Table 8.

---

## 7. Protocol standards (agree before any result)

- ≥20 independent seeds/splits where training is involved; for checkpoint-only
  experiments, ≥20 perturbation replicates per severity level.
- Group-aware splits; on NuBench use the published selection files so results
  are comparable to the benchmark's own numbers.
- Perturbation library, severity grid, thresholds, and smallest relevant effect
  frozen in a signed predeclaration before any test-set evaluation.
- Locked test set, touched once; validation-selected checkpoints restored
  before test evaluation.
- Matched compute, parameter count, and embedding dimension across
  representations — or, where matching is impossible (the four NuBench models
  differ by design), report the mismatch explicitly and treat cross-model
  comparisons as descriptive rather than controlled.
- Every diagnostic reported with a null (permutation or matched-noise control).
- Result folders carry: config, seeds and RNG state, split indices, git
  revision, environment lock, data checksums, and separate validation/test
  metric blocks.

---

## 8. The "how far can linear methods go" section (核心边界)

This section is a contribution, not a caveat. It states precisely where the
identifiability wall sits.

**Can, rigorously:**

1. Whitening / decorrelation from a fitted Σ or CSD — exact; Paper 1's geometry.
2. Frequency-resolved CSD eigenanalysis: per-bin eigenmodes, coherence, phase,
   shared-vs-private variance, common-mode rank estimation.
3. Constrained component fits when component *shapes* are known up to
   parameters (SQUID floor, Johnson, thermal, 1/f, lines) — well-posed and
   testable on held-out runs. This is Study A/B's regime.
4. Auxiliary/witness-channel regression — causal to the extent the witness is
   upstream.
5. Sensitivity attribution: how removing or rescaling a declared component
   changes OF variance, Fisher information, and trigger threshold.

**Cannot:** recover uniquely labeled physical sources from one total Σ.
Σ = Σ₁ + Σ₂ admits infinitely many PSD-cone splits; low-rank factors are
rotation-degenerate; eigenmodes are not sources. Well-posedness requires added
structure — parameterized spectral families, auxiliary sensors, multiple
operating conditions or hardware interventions, injected calibration lines, or
non-Gaussian/temporal signatures that break the rotation degeneracy.

**Consequence for detector design:** the productive target is *sensitivity
attribution under declared models*, not source discovery. Actionable
(shielding, bandwidth, sensor layout, readout) without claiming that covariance
eigenvectors are physical causes. Paper 1 §9 already states this boundary in
publishable language.

**Point-cloud addendum (Track A).** In the hit-level setting the analogue of
the identifiability wall is that dark-noise rate, PMT efficiency loss, and
timing jitter can produce similar degradations in a learned representation; a
diagnostic that only reports "something shifted" cannot attribute which. Our
perturbation library is what makes attribution testable, because we can ask
whether the diagnostic's *response profile* differs between the intents in
§4.3. That is a genuinely new, small, provable result and worth a subsection.

---

## 9. Division of labor

- **Our side:** both perturbation libraries and their severity calibration,
  the likelihood-geometry diagnostics (whitened residual, principal angles,
  Fisher/pullback), sensitivity attribution, §8, and the Track A NuBench
  pipeline (which is self-contained and can start before Track B is settled).
- **Junjie's side:** the Track B hierarchical pipeline, the representations
  under test there, real vertex/geometry labels, clustering and downstream
  physics metrics, and domain review of whether the hit-level perturbations are
  physically plausible.
- **Joint, co-signed before results:** the predeclaration document and the
  blind injection protocol.

The two-track structure has a practical benefit for the negotiation: Track A
requires nothing from Junjie to begin. You can arrive at M1 with a real figure
already in hand, which is a much stronger opening than a proposal.

---

## 10. Milestones

| M | Deliverable | Gate |
| --- | --- | --- |
| M0 | Scope agreement: thesis, claim set C1–C9, venue, author order, Track B data | signed by both sides |
| M1 | **Track A pilot (E1)**: one NuBench geometry, one model family, hit-level severity grid, seed CIs | curves separate from the zero-severity band |
| M2 | Predeclaration document: both perturbation libraries, severity grids, metrics, nulls, thresholds, smallest relevant effect | frozen, hashed, referenced by every later run |
| M3 | Battery E2–E3 on Track A across four model families + calibration control (C5 halt gate) | matched-noise control passes |
| M4 | E6 vertex stratification + E7 cross-geometry transfer | verdict rank correlations with CIs |
| M5 | E5 coverage with equal-n controls (requires retraining) | full-vs-restricted gap resolvable |
| M6 | Track B replication of E1, E2, E4 on real data | diagnostics transfer or the mismatch is characterized |
| M7 | Blind injection probe E4 (both tracks) | analyst blinded; mixture sealed until scoring |
| M8 | Draft with attribution table; §8 written | every number traceable to a registered run |

**First thing to do — and it needs nobody's permission:** download one NuBench
geometry plus its model artifacts, extract embeddings from one published
checkpoint, and produce the M1 figure. Send Junjie the figure, not the
proposal.
