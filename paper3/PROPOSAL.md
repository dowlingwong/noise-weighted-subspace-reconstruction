# ShiftScope: Testing and Diagnosing Distribution Shifts in ML Sensor Pipelines

**Paper 3 systems proposal — intended primary target: MLSys**

## Abstract

ML sensor pipelines can fail because observations become noisier, sensor or
physical-coordinate coverage changes, or the evaluation contract no longer
matches the model output. Existing drift monitors can often flag that a stream
changed, but a release engineer still has to determine whether the model, the
sensor path, or the metric is responsible. We propose **ShiftScope**, an
offline-first testing and observability system for learned representations in
sensor-processing pipelines. ShiftScope declares controlled perturbations,
captures layer representations through model adapters, computes windowed input,
embedding, output, and calibration signals, and compares their response with a
library of cause fingerprints. It produces release gates and machine-readable
reports, and can run the same monitors in replay or online mode. Version 1
targets event-batched waveform, point-cloud, and multichannel pipelines and
three failure classes: noise/acquisition shift, coverage/support shift, and
metric-contract mismatch.

We will evaluate ShiftScope on NuBench neutrino-telescope point clouds and on
waveform replays driven by our validated PSD/CSD noise module. The study
compares input tests, maximum mean discrepancy (MMD), classifier two-sample
tests, embedding distance, uncertainty/entropy, and output-only monitoring.
The central hypothesis is not merely that ShiftScope detects change: it should
identify the responsible failure class, abstain on unknown causes, and quantify
the cost of doing so. Systems measurements cover detection delay, false-alert
rate, attribution accuracy, instrumentation latency, embedding storage,
scalability, replay versus online cost, sketching, and reproducibility. A
restored 1.36-million-parameter NuBench checkpoint, a 128-dimensional hook, and
a monotone module-dropout pilot establish feasibility, while an unresolved
checkpoint-to-release discrepancy correctly leaves the clean-data release gate
open.

## Scope and studied failure classes

ShiftScope v1 is a **model-release qualification and replay observability
system**, with the same windowed monitor interface usable on a live event
stream. It does not attempt autonomous remediation, causal detector-design
optimization, or unique decomposition of physical noise sources.

The study has three explicit topics:

1. **Noise and acquisition shift (N):** changes in PSD amplitude or slope,
   cross-channel covariance/coherence, time jitter, spectral lines, drift,
   glitches, or tail behavior.
2. **Coverage and support shift (C):** missing hits, channels, or modules;
   reduced spatial containment; and events in sparsely represented physical
   coordinates. This class concerns observation support, not causal claims
   about detector geometry.
3. **Metric-contract mismatch (M):** a pipeline evaluates the wrong output
   semantics, coordinate convention, units, calibration score, or
   reconstruction metric. This is a pipeline-contract failure rather than a
   distribution shift, and ShiftScope must distinguish it from N and C.

Single-cause interventions provide attribution labels. Held-out severities,
mixed N+C failures, and one unseen perturbation family test interpolation,
mixture handling, and abstention rather than memorization.

## Contributions and novelty

ShiftScope contributes:

1. **A reusable observability artifact:** a perturbation specification layer,
   waveform/point-cloud/multichannel adapters, representation hooks, replay and
   windowed monitors, plugin interfaces, and CI/model-release reports.
2. **Cause-aware monitoring:** intervention replays create layerwise response
   fingerprints across input, representation, output, and calibration signals;
   a cause classifier returns N, C, M, mixture, or unknown with calibrated
   abstention.
3. **A guarded evaluation contract:** clean matched-data reproduction and
   calibration gates run before shift conclusions. A failed gate is reported
   separately from genuine in-distribution model miscalibration.
4. **A cross-modal systems benchmark:** labeled perturbations and generic drift
   baselines are evaluated across waveforms and point clouds, multiple model
   families, and CPU/GPU execution.
5. **A cost–power characterization:** full embedding capture is compared with
   event sampling, reservoir summaries, and fixed-dimensional random-feature
   sketches to expose the storage, latency, and detection trade-off.

The novelty is the combination of declared sensor interventions, internal
instrumentation, cause attribution, calibration gating, and systems-cost
measurement. NuBench, the noise generator, MMD, and representation probes are
ingredients, not separate novelty claims. The decisive result would be:
**generic drift monitors detect that something changed; ShiftScope identifies
whether the evidence is most consistent with sensor noise, unsupported
coverage, or a metric-contract failure, at quantified operational cost.**

## Three falsifiable claims

**C1 — reliable shift detection.** At a fixed 1% false-alert rate on nominal
windows, ShiftScope detects at least 80% of predeclared abrupt and gradual N/C
shifts within five monitoring windows and is non-inferior to the strongest
generic baseline by a five-percentage-point power margin. Detection is assessed
on held-out events, severities, and perturbation seeds. Failure to pass nominal
false-alert or positive-control checks rejects the claim.

**C2 — root-cause diagnosis and guarded abstention.** On held-out N, C, and M
interventions, ShiftScope achieves macro-F1 at least 0.80 and exceeds the best
generic-monitor feature baseline by at least 0.20. On a perturbation family
excluded from fingerprint training, it must achieve unknown-cause AUROC at
least 0.80 rather than force a known label. Mixed shifts are scored with
multi-label F1. A representation may legitimately encode noise or measurement
quality; the claim concerns retained task information and conditionally stable
responses after energy, multiplicity, topology, and containment are controlled.

**C3 — deployable observability cost.** With 10% event sampling and
fixed-dimensional sketches, ShiftScope keeps p95 inference-path latency overhead
below 10%, reduces stored representation bytes by at least 10×, and loses no
more than five percentage points of detection power or attribution macro-F1
relative to full capture. The direction and uncertainty of the result must
reproduce across two sensor modalities, at least three model families, and CPU
and GPU replay. These are proposed acceptance thresholds; power analysis may
change them only before the locked evaluation.

## System and methods

A YAML perturbation specification records modality, failure class, transform,
units, severity schedule, ramp/step timing, seed, and required invariants. Each
adapter implements `validate`, `perturb`, `summarize`, and `serialize` without
changing model code. The waveform adapter operates before preprocessing; the
point-cloud adapter operates on hits before graph construction; the
multichannel adapter preserves channel identity and CSD metadata.

Instrumentation hooks capture named event representations, outputs,
uncertainties, and timing. A monitoring window produces a signature containing
input quality summaries, sketched layer moments and distances, output drift,
task performance where delayed labels exist, and calibration/coverage. Offline
replay of labeled interventions learns a cause fingerprint over this signature.
At evaluation time a calibrated classifier produces N/C/M or mixture scores and
an abstention score. The first version uses simple, auditable models—regularized
logistic regression and gradient-boosted trees—before any learned sequence
monitor. Attribution accuracy is therefore not confounded with a large new
model.

Generic comparisons are: univariate input tests with multiplicity correction;
RBF-MMD; a classifier two-sample test; mean/covariance embedding distance;
output-distribution and uncertainty/entropy monitoring; and output-only
calibration where labels arrive. Each baseline receives the same windows and
false-alert budget. Positive controls, zero-severity identity, paired seeds,
and label-preserving nulls are mandatory. ARI is not used as a label-permutation
null; any clustering analysis requires repeated perturbations, algorithm/seed
stability, and a physically meaningful target.

## Data, models, and perturbation engines

**NuBench point clouds.** The first cell is Hexagon Ice LE × DynEdge × direction.
The 1,358,099-parameter checkpoint creates a 128-dimensional event embedding
immediately before its direction/concentration head. Module dropout, hit
thinning, and time jitter instantiate C and acquisition shifts. Expansion to
ParticleNeT and GRIT provides distinct model families; a second geometry is a
portability check, not causal geometry attribution.

**Waveforms and our noise module.** The noise module is a controlled
waveform-intervention and validation engine—not a reconstruction model and not
an automatic physical-source separator. It synthesizes stationary one-sided
PSD-driven noise with variance-consistent real-FFT conventions or exact finite
Toeplitz covariance; multichannel shared/private, low-rank, and target-CSD
noise with Hermitian positive-semidefinite checks; analytic or measured
spectral shapes; lines, drift, glitches, bursts, and nonstationarity; and
selected heavy-tailed/non-Gaussian innovations. It records units, seeds,
components, and validation statistics. The module supplies known N labels to a
frozen waveform encoder and permits matched versus mismatched PSD/CSD replay.
The existing CRESST/TIDMAD linear encoders provide an auditable first waveform
adapter; a collaborator waveform model is an optional portability extension,
not a dependency of the core paper.

## Experimental program

| ID | Question and comparison | Primary outcomes |
| --- | --- | --- |
| E0 | Can each clean checkpoint, preprocessing path, hook, and metric be reproduced? Deliberately inject M faults. | released-metric discrepancy, per-event agreement, gate pass/fail |
| E1 | Benchmark N/C/M interventions against input tests, MMD, classifier tests, embedding distance, uncertainty, and output-only monitors. | power at 1% false alert, AUROC/AUPRC, detection delay |
| E2 | Attribute held-out single causes, N+C mixtures, and one unseen family. Condition on energy, topology, multiplicity, and containment. | macro/multi-label F1, confusion matrix, abstention AUROC, calibration |
| E3 | Sweep event rate, window size, embedding dimension, hooks, and number of models. Compare full capture, 10% sampling, reservoirs, and random-feature sketches. | p50/p95 latency, CPU/GPU utilization, bytes/event, power–cost frontier |
| E4 | Compare offline replay with the same monitors in stream order under abrupt and gradual ramps. | time-to-detection, false alerts/hour, replay/online cost ratio |
| E5 | Repeat the minimum suite across DynEdge, ParticleNeT, GRIT, waveform encoders, two machines, and CPU/GPU where supported. | effect reproducibility, configuration success rate, deterministic-report hashes |

Windows, event groups, severities, smallest relevant effects, and thresholds
are frozen after development. Confidence intervals resample event groups and
perturbation seeds separately. Latent displacement is normalized by clean
within-class dispersion; cross-model comparisons use alignment-aware measures
rather than raw coordinates. Null findings require a passed positive control,
adequate precision, and an equivalence margin.

## Smoke test: proof of feasibility, not a claim result

The selected 8.82 GiB NuBench release has been downloaded and checksummed. The
DynEdge checkpoint restores, runs, and exposes the intended 128-dimensional
hook. Rescoring all 2,989,339 released predictions reproduces published median
angular errors to within 0.055°: 23.6548° versus 23.60° for CC/track and
56.1553° versus 56.12° for NC/cascade. This is close but fails the deliberately
strict 0.01° identity tolerance. CPU checkpoint inference on 256 development
events differs from released directions by 0.944° median (7.708° at p95), so
the clean release gate remains open pending exact legacy GraphNeT/backend
reproduction or a justified tolerance.

On the same balanced, high-multiplicity development subset, observed-module
dropout from 0% to 50% raises median angular error from 15.28° to 25.12°,
increases standardized latent displacement from 0 to 0.149, and lowers 10-NN
retention from 1.000 to 0.393. This proves that perturbation injection,
representation capture, paired scoring, and reporting are operational. It does
not establish C1–C3; in fact, the still-open clean gate demonstrates the
deployment behavior ShiftScope is designed to enforce.

The detailed implementation plan remains in `PAPER3_PLAN.md` as a technical
appendix rather than the proposal's primary narrative.
