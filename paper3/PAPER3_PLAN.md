# Paper 3 Plan — Controlled Shift Diagnostics for Learned Representations in Particle Reconstruction

Role in review package: **technical appendix**. The primary 4–6 page proposal
is `PROPOSAL.md`.

Status: collaboration proposal (v3, 2026-07-27; v2 revised after a feasibility
and claim audit). Joint work with Junjie (Stanford). Companion to Paper 1
(likelihood geometry) and Paper 2 (architecture and coverage). This document is
the negotiating instrument for the collaboration: it proposes the thesis,
claim hierarchy, datasets, protocol, and division of labor to be agreed before
confirmatory results are produced. The final registered protocol will be a
shorter, separate document.

---

## 1. Executive summary

**Thesis (one sentence, everything serves it).**

> A representation used for particle reconstruction should retain task-relevant
> physical information and exhibit reproducible response curves under
> predeclared, physically motivated measurement shifts; we develop diagnostics
> that distinguish conditional shift sensitivity from coverage limitations and
> test which conclusions replicate from a public simulated benchmark to a real
> reconstruction pipeline.

**Type of paper.** Measurement methodology, not a new architecture and not a
new detector result. It is judged on whether the diagnostics are principled,
predeclared, null-calibrated, and informative — including when they return
negative answers.

**Three contributions, in priority order.**

1. A *modality-aware controlled-shift contract*: paired nominal/perturbed
   observations with a declared intervention, severity coordinate, physical
   interpretation, and validity domain. It has a hit-level implementation for
   NuBench and, only if Track B consumes dense traces, a waveform-level
   implementation using the existing PSD/CSD noise module.
2. A *primary diagnostic battery* with explicit references and uncertainty:
   conditional signal retention, normalized representation displacement,
   downstream degradation, calibration, and boundary/containment-stratified
   performance. Clustering is secondary and anomaly detection is gated.
3. An *empirical replication result*: which diagnostic conclusions reproduce
   across perturbation replicates, detector domains, and the collaborator's
   real pipeline. A restricted-support retraining experiment tests coverage as
   a secondary mechanism; the paper does not claim causal metric attribution
   unless a separate metric intervention is added.

**Two-track evidence design.** Track A is **NuBench**: public and simulated,
with released checkpoints and predictions for several model/task/geometry
combinations. It is the reproducible backbone on which the hit-level
perturbations and diagnostics are developed. Track B is **Junjie's pipeline /
real data**: the external-domain validation where the selected diagnostics
must survive contact with an instrument. The tracks do not share raw data, a
likelihood, or a numerical severity scale. They share an intervention-response
protocol and standardized diagnostic outputs; results are replicated, not
pooled. Section 4.3 states the compatibility gate precisely.

**Headline risk we deliberately accept.** Some diagnostics may be
uninformative. A null result is interpretable only when the experiment first
demonstrates sensitivity to a positive control, has adequate power, reports a
predeclared smallest relevant effect or equivalence margin, and quantifies
uncertainty. Predeclaration prevents outcome-dependent selection but does not,
by itself, make a null result informative or publishable.

**What this paper does not claim.** No unique physical noise decomposition; no
detector-sensitivity improvement; no new-physics anomaly discovery.

---

## 2. Positioning against Papers 1 and 2

| | Paper 1 | Paper 2 | Paper 3 (this) |
| --- | --- | --- | --- |
| Object | reconstruction estimator | reconstruction model class | learned *representation* |
| Question | what is the right metric? | which architecture and coverage? | how does the representation respond to controlled shift? |
| Evidence | identities + real-data residuals | theorems + NFPA case study | empirical diagnostics under controlled shift |
| Data | CRESST, TIDMAD, GWOSC boundary | synthetic + Xenon-TPC case study | NuBench (public) + Junjie pipeline (real) |

The dependency runs forward: Paper 2 §8 explicitly *generates* an experimental
program (Table 7: metric reversal, coverage violation, architecture ladder,
noise mismatch, coverage-corrected training) and does not report it. Paper 3
executes the controlled-shift row and, subject to compute, one coverage row.
Cross-model comparisons of released NuBench checkpoints are descriptive because
their training procedures, parameter counts, tasks, and random seeds are not
controlled. Paper 3 therefore does not claim to execute a causal architecture
ladder without matched retraining.

**A second bridge.** Paper 2 §8.3 Table 8 lists detector design factors evaluated
by the Fisher/pullback metric g_z = J_s(z)†Σ⁻¹J_s(z); one factor is channel
layout. NuBench provides seven datasets across six detector geometries with a
shared simulation framework, so it supports a useful *cross-domain consistency*
study. It is not a controlled channel-layout intervention: medium, energy
range, event distribution, training realization, and model availability can
also differ. Geometry conclusions must therefore be conditioned or reweighted
for energy, topology, hit multiplicity, and containment, and described as
replication/heterogeneity rather than causal detector-design effects.

**Scope statement that must appear early (honesty gate).** Paper 1's metric is
a Gaussian Σ⁻¹ over dense trace samples. Neutrino-telescope data are sparse
photon-hit point clouds; a defensible likelihood must be based on a hit-level
point process and detector-response model, not assumed to be a dense
Mahalanobis form. Paper 3 therefore
inherits the *separation* of metric / architecture / coverage from Papers 1–2
and explicitly does **not** inherit the specific Mahalanobis form. The primary
Paper 3 intervention is measurement shift. "Metric mismatch" is removed from
the headline attribution claim unless the collaboration adds a separately
controlled loss/likelihood experiment.

---

## 3. Claim set (predeclared, with falsifiers)

The claims are hierarchical. C1, C2, C4, and C5 are primary; C3, C6, C8, and
C9 are secondary; C7 is inactive unless the collaborator supplies a concrete
physics-motivated anomaly use case. Each active claim receives a metric,
reference/control, smallest relevant effect, and failure condition in the
registered protocol. This hierarchy prevents an uninformative result from
being replaced post hoc by another member of a large battery.

**Representation contract used by every claim.** For each checkpoint we name
the task, extraction layer, event-level pooling rule, and feature preprocessing.
Distances are computed after a validation-fitted normalization. Cross-run or
cross-model comparisons use an explicitly declared alignment (orthogonal
Procrustes where paired events and equal dimensions permit it, otherwise
rotation-invariant summaries such as centered Gram/CKA or neighborhood
retention). Absolute raw latent norms are never compared across architectures.

| # | Priority | Revised claim | Primary metric | Reference / control | Failure or scope-limit condition | Track |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | Primary | Task-relevant physical coordinates remain decodable under bounded shift relative to the matched clean representation | clean-to-shift change in held-out probe performance, reported for a linear probe and one capacity-matched nonlinear probe | label permutation; clean probe; input-feature probe | clean probe lacks sensitivity, or shifted performance falls beyond the predeclared relevant margin | A + B |
| C2 | Primary | After conditioning on energy, topology, hit multiplicity, and containment, nuisance state has a reproducible association with representation and downstream error; nuisance decodability is measured, not assumed undesirable | cross-fitted conditional nuisance probe; conditional mutual-information or residualized accuracy estimate; signal-retention/nuisance-sensitivity pair | stratified or conditional permutation within physical covariate bins; nuisance-only input baseline | covariate balance fails, or the association is indistinguishable from the conditional null at the powered effect size | A + B |
| C3 | Secondary | Physically defined groups or a validation-frozen reference partition retain membership under bounded shift beyond clustering-algorithm instability | paired ARI/VI, assignment retention, and consensus stability vs severity | zero-severity perturbation replicates; resampling/initialization stability; where available, a physical target partition | no physically meaningful partition exists, or perturbation effects do not exceed algorithm/resampling variability | A + B if a target exists |
| C4 | Primary | Standardized representation displacement and downstream degradation exhibit reproducible severity-response curves | displacement normalized by clean within-stratum dispersion; k-NN retention; aligned/rotation-invariant representation change; downstream error | paired zero-severity transform; positive-control corruption; perturbation replicates | positive control is not detected, response is dominated by Monte Carlo variation, or no effect larger than the equivalence margin is resolved | A + B |
| C5 | Primary | In-distribution calibration is measured independently of implementation correctness, and any change under shift is quantified relative to that clean baseline | task-appropriate calibration error/reliability for classification; coverage or PIT only when a predictive distribution exists | V0 reproduction check against published clean metrics; clean calibration baseline; simple recalibration baseline fitted on validation only | V0 fails ⇒ implementation investigation; V0 passes but calibration fails ⇒ valid model finding, not automatically a bug | A + B where probabilistic scores exist |
| C6 | Secondary, compute-gated | Restricted training support changes out-of-support performance relative to an equal-n full-support control; interaction with measurement shift is estimated separately | full-minus-restricted error gap and shift×coverage interaction on one coordinate | equal-n full-support retraining with matched architecture, optimization, and seeds | matched retraining is infeasible or the interaction is underpowered; no claim about a "correct metric" is made | A; B only if retraining is possible |
| C7 | Inactive/gated | If a domain expert predefines a scientifically meaningful signal/anomaly class and a fixed score, test whether it can be distinguished from nuisance-only shifts | power/partial AUC at a predeclared FPR for the frozen score | nuisance-only foils and a nuisance-only detector; sealed labels only after validation freeze | no concrete anomaly class, score, or downstream decision exists ⇒ omit C7 and the blind-mixture section | B preferred; A only for method validation |
| C8 | Secondary | Diagnostic response depends on boundary distance and containment after conditioning on energy, topology, and hit multiplicity | interaction between severity and boundary/containment stratum; stratified downstream error | central/well-contained reference plus covariate-matched or reweighted comparison | effective sample size or overlap is inadequate; otherwise a null interaction is reported with an equivalence interval | A + B if geometry labels exist |
| C9 | Secondary | The sign and ordering of selected diagnostic effects are assessed for consistency across NuBench domains; heterogeneity is quantified rather than interpreted causally as geometry | standardized effect estimates per domain, descriptive heterogeneity with uncertainty, leave-one-domain-out threshold replication | within-domain perturbation variation; covariate-reweighted analyses | strong unexplained heterogeneity ⇒ diagnostic is domain-specific; no detector-design causal claim | A only |

**Verification is distinct from calibration.** V0 is an engineering gate: on
unperturbed data, the restored checkpoint must reproduce a published metric
within an agreed tolerance. C5 starts only after V0. A model can pass V0 and be
genuinely miscalibrated; that is a scientific result, not proof of a bug.

**Boundary language.** Every event has an interaction vertex. C8 concerns the
vertex's distance to the instrumented-volume boundary and event containment,
not generic performance "near interaction vertices." The manuscript therefore
uses *boundary- and containment-stratified performance* throughout.

---

## 4. Datasets

### 4.1 Track A — NuBench (public backbone)

`graphnet-team/NuBench`, arXiv:2511.13111. Seven datasets simulated across six
distinct detector geometries (Cluster, Flower S/L/XL, Hexagon, Hexagon Ice LE,
Triangle) using the Prometheus chain, distributed in SQLite and Parquet with
GraphNeT dataset classes. Ships **test-partition predictions and full model
artifacts** (weights, pickled models, configs) representing four model
families: ParticleNeT, DynEdge, GRIT, DeepIce. Their availability is not assumed
to cover every task×geometry cell.

Why it is the right backbone:

1. **Released checkpoints and predictions.** NuBench reduces training cost for
   the primary checkpoint-only study and permits the V0 reproduction gate. It
   does not remove inference, embedding-extraction, storage, or software-version
   work.
2. **Several domains.** Six geometries permit C9 as a replication and
   heterogeneity study after covariate adjustment; they are not a free causal
   channel-layout experiment.
3. **Simulation truth.** Energy, direction, vertex position, and interaction
   type support conditional analyses, one restricted-support experiment, and
   boundary/containment stratification.
4. **Public artifacts.** The benchmark makes the method inspectable and allows
   sanity checks against published downstream predictions.

Limitations, stated up front:

- **Simulated.** Noise shift must be *injected*, not observed. Track A can
  therefore establish that a diagnostic detects a known shift; only Track B can
  establish that it detects shifts that real instruments actually exhibit.
  Write this as the explicit reason the paper has two tracks.
- **Point clouds, not waveforms.** The Σ⁻¹ machinery does not apply directly;
  see §4.3 for the hit-level perturbation library that replaces it.
- **The released models do not form a complete controlled factorial design.**
  Availability differs by model, task, and geometry, and a single released
  training realization does not quantify architecture-level training variance.
  Before selecting the primary task, record the exact model×task×geometry
  matrix and choose a genuinely shared cell. Cross-model results remain
  descriptive unless models are retrained under a matched protocol.
- **Checkpoint-only is not cost-free.** Embeddings require inference on the raw
  event data for every perturbation replicate. The Hexagon SQLite archive alone
  is approximately 127 GB; its released predictions and model artifacts are
  approximately 1.57 GB and 1.11 GB. The pilot must therefore declare an event
  development subset, storage budget, inference batch size, accelerator, and
  wall-time estimate before a full download or sweep.
- **Retraining is separate.** C6 requires a restored training pipeline and
  multiple matched training seeds. It remains compute-gated and cannot delay
  the primary checkpoint-only paper.
- Datasets are "inspired by, but not strictly identical to" existing
  telescopes, so no claim about any named experiment.

**NuBench feasibility audit required before M1:**

| Item | Decision/evidence required |
| --- | --- |
| Shared comparison cell | one task, geometry, and set of actually available checkpoints |
| V0 | clean downstream metric reproduced within a predeclared tolerance |
| Representation | extraction layer, pooling, dimension, preprocessing, and hook test |
| Data | archive checksum/size, development and confirmatory event selections |
| Compute | GPU/CPU type, memory, batch size, events/s, storage, estimated sweep cost |
| Perturbation validity | required fields exist; clean zero-severity transform is identity-preserving |

### 4.2 Track B — Junjie's pipeline / real data

Track B is an external-domain validation, not simply another row in a pooled
benchmark. It is useful only if the collaboration can answer the following at
M0:

| Required decision | Why it is needed |
| --- | --- |
| Input modality: waveform, calibrated hits, clusters, or another object | determines whether the existing waveform noise module applies directly |
| Real data, simulation, or both; provenance of noise labels/states | defines what "real-noise validation" can mean |
| Raw-input access and permission to perturb/reprocess it | required for paired nominal/shifted inference |
| Checkpoint and intermediate-representation access | required for representation diagnostics rather than output-only testing |
| Available truth/proxy labels and geometry/containment fields | determines which of C1, C2, C3, and C8 are identifiable |
| Probabilistic outputs or only point predictions | determines whether PIT/coverage in C5 is meaningful |
| Retraining and compute access | determines whether C6 can run on Track B |
| Concrete anomaly score and physics use case | required to activate C7; otherwise it is omitted |

If Track B provides outputs but not embeddings, the transfer claim is reduced
to downstream response and calibration. If it provides neither raw inputs nor
a defensible observed shift, it cannot serve as the real-domain validation and
the paper must be scoped to NuBench plus a limitations section.

### 4.3 How two different data modalities support one paper

NuBench point clouds and the waveform noise module do **not** share a data
distribution or likelihood and are never treated as interchangeable datasets.
They support the same paper only at the level of a common, modality-agnostic
intervention-response contract. For event i, perturbation family u, and
severity s, construct a paired observation

> x_i^(u,s) = T_(u,s)(x_i), with T_(u,0)(x_i) = x_i,

then evaluate a frozen representation and downstream model on x_i and
x_i^(u,s). Each modality reports standardized effect curves relative to its own
clean dispersion and downstream scale. Raw severity values, latent distances,
and likelihoods are not pooled across tracks. The cross-track question is only
whether the *diagnostic conclusion* replicates: for example, whether a positive
control is detected, whether an effect increases with severity, and whether
boundary/containment modifies the response.

The existing noise module has two possible roles:

1. **Direct second instantiation**, only if Track B consumes dense waveforms or
   traces for which PSD/CSD interventions are scientifically valid.
2. **Design precedent only**, if Track B consumes hit clouds or higher-level
   objects. In that case its sampling discipline, severity calibration, and
   matched/model-violating distinction inform Paper 3, but waveform results
   remain in Papers 1–2 and are not presented as Paper 3 evidence.

This compatibility gate prevents a paper from joining unrelated datasets only
because both contain something called "noise."

| Intervention class | Waveform implementation, when applicable | NuBench hit-level implementation | Validity restriction |
| --- | --- | --- | --- |
| Additive background | PSD/CSD amplitude or component scaling | add dark hits from a declared time/charge/spatial model | event window and background generator must be specified |
| Timing degradation | phase/timing perturbation under the trace model | add zero-mean timing jitter with declared distribution | does not emulate arbitrary calibration drift |
| Correlated contamination | CSD coherence/rank change or declared line | add topology-aware correlated hit bursts | burst timing, charge, and module topology require domain validation |
| Heavy tails/outliers | glitches or heavy-tailed trace samples | afterpulse/outlier model conditional on existing hits | empirical or simulator-based parameterization required |
| Sensor loss | waveform-channel dropout | DOM/PMT dropout or dead-time | pairing and live-channel normalization required |
| Calibration error | per-channel gain/timing error | per-DOM charge scale and time offset | change features and rebuild graph/preprocessing consistently |
| Detection efficiency | not represented by additive PSD alone | independently or conditionally thin observed hits | can emulate decreased efficiency only; increasing efficiency requires forward simulation |

Every transform is tagged along three separate axes: **observable-space valid
vs forward-simulator required**, **inside vs outside the checkpoint's declared
training augmentation**, and **physically calibrated vs stress-test only**.
"Likelihood-preserving" is used only when the evaluated model actually declares
a likelihood under which that statement can be proved.

The hit-level library is new code and the main Track A engineering cost. Build
it as a GraphNeT-compatible transform applied before graph construction, so
gain, timing, dropout, and hit additions propagate through the same feature and
connectivity pipeline used at clean inference.

---

## 5. Section outline

1. **Introduction.** State the controlled-shift problem, why output accuracy
   alone does not diagnose representation response, the public-to-real
   replication design, and the scope limit: no shared likelihood or severity
   scale is assumed across modalities.
2. **Related work and novelty matrix.** Compare against representation probing,
   representation similarity, corruption/robustness benchmarks, detector
   systematic-uncertainty studies, and calibration/domain-shift work. For each
   nearest work, record whether it provides paired interventions, physical
   severity, conditional covariate control, representation-level metrics,
   downstream metrics, and real-domain replication. Remove uncited
   superlatives such as "no other resource" until this audit is complete.
3. **Setup and feasibility.** NuBench artifact matrix and V0; Track B data
   contract; representation extraction contract; development/validation/test
   selections; compute budget.
4. **Modality-aware intervention contract.** Hit-level transforms, optional
   waveform instantiation, validity tags, severity calibration, and what a
   simulator cannot certify.
5. **Primary diagnostics.** C1, C2, and C4, each with definition, conditional
   covariates, reference/control, positive control, effect size, and CI.
6. **Calibration.** C5, with checkpoint verification explicitly separated from
   clean calibration and calibration drift.
7. **Secondary analyses.** C3 only with a meaningful partition; C8 boundary and
   containment; C9 cross-domain heterogeneity.
8. **Coverage extension.** C6 on one coordinate, one task, one geometry, and at
   most two model families, only if matched retraining is feasible.
9. **Real-domain replication.** Repeat the smallest battery supported by Track
   B; compare standardized conclusions without pooling raw effects.
10. **Results synthesis and limitations.** Evidence table for shift sensitivity,
    coverage, calibration, domain heterogeneity, and unresolved causes. Metric
    mismatch is included only if independently manipulated.

C7 receives no manuscript section unless activated at M0. The linear/CSD
identifiability material in §8 is retained as an internal boundary note and
will be shortened to a discussion paragraph unless waveform Track B makes it
directly relevant.

---

## 6. Experimental program

Track A can begin before Track B is fixed, but only after E0. Development
subsets are used to build and debug the battery; confirmatory selections remain
untouched until the protocol is frozen.

### E0 — Feasibility and clean reproduction (do this first)

Audit the released model×task×geometry matrix, select one shared checkpoint
cell, and download the minimum required archive. Restore one checkpoint,
reproduce one published clean downstream metric within a declared tolerance,
extract the named representation, and benchmark inference throughput on a
development subset. Verify that T_(u,0) is an identity at both input-feature and
prediction levels. Failure of V0 blocks representation claims but is not a
statement about model calibration.

### E1 — Primary severity-response pilot

On one NuBench geometry, task, and checkpoint, start with one observable-space
valid perturbation: recommended options are decreased detection efficiency by
hit thinning, additive timing jitter, or module dropout. Use a validation-chosen
severity grid including s=0, paired perturbation replicates, and a positive
control strong enough to be detectable. Report clean-dispersion-normalized
representation displacement, k-NN retention, task-relevant probe change, and
one downstream reconstruction metric. The go/no-go criterion is a powered
response beyond both zero-severity variation and the smallest relevant effect.

Track B repeats this *protocol*, not necessarily the same physical
perturbation. A waveform implementation is used only if the M0 modality gate
allows it.

### E2 — Conditional signal retention and nuisance sensitivity

For each selected perturbation state, evaluate task-coordinate retention and
nuisance-state association while conditioning on energy, topology, hit
multiplicity, and containment. Use cross-fitting so probe training and effect
estimation do not reuse events. Report linear and capacity-matched nonlinear
signal probes, a conditional nuisance probe or residualized association, and
input-feature baselines. Do not rank architectures causally from single
unmatched checkpoints.

### E3 — Partition stability (secondary and optional)

Run only if a physical target partition or defensible validation-frozen
reference partition exists. Compare each event's clean and perturbed assignment
using ARI/VI and assignment retention. Estimate reference instability from
bootstrap resampling, clustering initialization, and zero-severity transforms.
The perturbation effect must exceed this algorithmic/resampling variation;
label permutation is not the primary null because ARI is already invariant to
cluster label names.

### E4 — Calibration under shift

After V0, measure clean calibration using metrics appropriate to the available
output: reliability/calibration error for classification, and interval coverage
or PIT only for a genuine predictive distribution. Fit any recalibration on
validation data only, then quantify calibration change over severity. Clean
miscalibration is a model finding. A blind anomaly mixture is not part of E4.

If C7 is activated at M0, write a separate E4b protocol naming the frozen
anomaly class, score, nuisance foils, operating FPR, and downstream decision;
otherwise omit it.

### E5 — Coverage extension (compute-gated)

Choose one latent restriction, preferably fiducial/contained-only training or
one energy interval, and compare it with an equal-n full-support control on a
full-support evaluation set. Match architecture, optimization, stopping rule,
and training seeds. Estimate the shift×coverage interaction separately from
the main restricted-vs-full gap. Do not treat Hexagon Ice LE as a restriction
"for free" because its medium, distribution, and simulation regime can differ.

### E6 — Boundary- and containment-stratified response

Stratify selected E1/E2/E4 effects by true distance from the interaction vertex
to the instrumented-volume boundary and by an explicitly defined containment
or hit-sharing measure. Condition or reweight for energy, topology, and hit
multiplicity and report overlap/effective sample size. The target is the
severity×boundary/containment interaction, not the unconditional fact that
boundary events are harder.

### E7 — Cross-domain consistency (C9, NuBench only)

Run a reduced, frozen battery on domains for which the same task and model
family are genuinely available. Estimate standardized effects separately,
reweight shared covariates where overlap permits, quantify heterogeneity, and
test leave-one-domain-out replication of diagnostic thresholds or effect signs.
Do not call reordering a causal geometry or detector-design effect; unexplained
reordering is a scope limit and a hypothesis for a future controlled study.

---

## 7. Protocol standards (agree before any result)

- Training seeds and perturbation replicates answer different uncertainty
  questions and are never substituted for one another. Use a power/simulation
  analysis to choose counts; target ≥20 training seeds only if the matched
  retraining budget makes that credible, and ≥20 paired perturbation replicates
  when Monte Carlo transform variance is material.
- The event is the paired observational unit. Use hierarchical or cluster-aware
  resampling when events share simulation/run conditions; report event-level
  uncertainty separately from perturbation and training-seed variation.
- Use group-aware development, validation, and confirmatory selections. Preserve
  the published NuBench test selection for V0 comparability, but do not develop
  the diagnostic on the same events later presented as untouched confirmation.
- Freeze active claims, representation hooks, perturbation generators, severity
  grids, positive controls, smallest relevant effects/equivalence margins,
  conditioning variables, and multiplicity correction before confirmatory
  evaluation. Timestamp/hash the protocol; collaborator signatures are useful
  governance, not a substitute for methodological detail.
- Restore validation-selected checkpoints before confirmatory evaluation. Log
  every accidental test access and downgrade the claim rather than describing
  the set as untouched.
- Treat released cross-model comparisons as descriptive. A controlled
  architecture claim requires matched data, objective, optimization, parameter
  budget or declared scaling analysis, representation extraction, and training
  seeds.
- Every active diagnostic has a reference/control, a positive control, and a
  predeclared uncertainty interval. The control need not always be a label
  permutation.
- Result folders carry: config, seeds and RNG state, split indices, git
  revision, environment lock, data checksums, and separate validation/test
  metric blocks.

---

## 8. Internal boundary note: how far linear methods can go

This material prevents overclaiming. It becomes a substantive Paper 3 section
only if Track B uses waveforms and the PSD/CSD noise module directly; otherwise
compress it to a discussion paragraph and cite Papers 1–2.

**Can, rigorously:**

1. Whitening / decorrelation from a fitted Σ or CSD — exact; Paper 1's geometry.
2. Frequency-resolved CSD eigenanalysis: per-bin eigenmodes, coherence, phase,
   shared-vs-private variance, common-mode rank estimation.
3. Constrained component fits when component *shapes* are known up to
   parameters (SQUID floor, Johnson, thermal, 1/f, lines) — well-posed and
   testable on held-out runs. This is Study A/B's regime.
4. Auxiliary/witness-channel regression — supports source attribution only
   under an explicit upstream/causal and measurement-error model.
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

**Point-cloud addendum (Track A).** Dark-noise rate, PMT efficiency loss, and
timing jitter can produce similar downstream degradation; observing a shift
does not identify its physical cause. Controlled interventions make a narrower
question testable: do response profiles discriminate the *injected families*
under the declared simulator? Success does not prove that an unknown real-data
shift has the same source. Claim a formal/provable result only if an actual
identifiability theorem with stated assumptions is supplied.

---

## 9. Division of labor

- **Our side:** NuBench feasibility audit and V0 reproduction, representation
  extraction contract, hit-level perturbation library, primary diagnostics,
  severity calibration, and reproducible Track A pipeline. Supply the waveform
  library only if the Track B modality gate makes it relevant.
- **Junjie's side:** complete the Track B data/access contract, identify the
  downstream physics task and representation stages, define geometry and
  containment variables, review perturbation plausibility, and provide the
  domain-specific metric and real-shift interpretation.
- **Joint, frozen before confirmatory results:** active claim hierarchy,
  positive controls, smallest relevant effects/equivalence margins, real-domain
  replication criterion, venue, compute commitment, and authorship plan. A
  blind-injection protocol is added only if C7 is activated.

Track A can start before Track B access is complete, but the M0 discussion must
occur before the confirmatory battery is frozen. This avoids building a public
diagnostic whose outputs cannot be reproduced in the collaborator's pipeline.

---

## 10. Milestones

| M | Deliverable | Gate |
| --- | --- | --- |
| M0 | Scope agreement: primary claims, target venue, Track B modality/access/labels, task, compute, author roles; decide whether waveform module and C7 apply | decision table completed by both sides |
| M1 | **E0 feasibility:** NuBench model×task×geometry matrix, resource budget, one restored checkpoint, V0 clean metric, representation hook | V0 passes and inference cost is acceptable |
| M2 | **E1 pilot:** one valid hit-level perturbation, zero and positive controls, standardized representation and downstream response | positive control detected; uncertainty and relevant-effect gate resolved |
| M3 | Related-work/novelty matrix plus registered primary protocol for C1/C2/C4/C5 | frozen, timestamped/hashed, linked by every confirmatory run |
| M4 | Primary Track A battery on the selected shared cells | all active metrics have controls and traceable uncertainty |
| M5 | C8 boundary/containment and C9 cross-domain heterogeneity | covariate overlap and effective sample size adequate |
| M6 | C6 retraining go/no-go and, if approved, equal-n restricted-support experiment | matched training pipeline and seed budget demonstrated |
| M7 | Smallest supported Track B replication | conclusion replicates after within-domain standardization, or heterogeneity is characterized |
| M8 | Draft with evidence/limitations table | every number traceable; no inactive or unsupported claim remains |

C7/E4b is a separate optional milestone inserted only after M0 activation.

**First action:** complete the artifact/resource matrix, download only the
minimum shared checkpoint cell, reproduce its clean metric, and validate one
representation hook. The first collaborator-facing scientific update should
pair the revised proposal with that V0 evidence; the first advisor-facing
update should additionally include the E1 pilot curve.
