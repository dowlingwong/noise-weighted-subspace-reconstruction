# Collaboration Review Brief — Controlled Shift Diagnostics for Learned Representations

Status: proposal for scope review; feasibility results exist, but no
confirmatory result has been produced.

## Proposed paper in one paragraph

Learned reconstruction models can retain good average in-distribution accuracy
while their internal representations and calibration respond poorly to changes
in measurement conditions. We propose a modality-aware diagnostic framework
based on paired nominal and deliberately perturbed observations. On NuBench we
will develop physically constrained hit-level perturbations and measure
task-information retention, conditional nuisance sensitivity, standardized
representation displacement, downstream degradation, and calibration. We will
then repeat the smallest supported subset on Junjie's real reconstruction
pipeline. The public and real tracks are not pooled: they test whether
standardized diagnostic conclusions replicate across domains.

## Primary claims

1. Task information, downstream performance, and calibration have
   reproducible, powered severity-response curves under declared measurement
   shifts.
2. Conditional representation response, after controlling for energy,
   topology, multiplicity, and containment, predicts downstream failure beyond
   those observables; nuisance information is not assumed to be intrinsically
   undesirable.
3. The smallest supported diagnostic conclusion is tested for replication or
   heterogeneity on the real pipeline after within-domain standardization.

Secondary analyses cover a meaningful physical/reference partition, boundary
and containment interactions, cross-domain heterogeneity, and one
compute-gated coverage restriction. Anomaly detection is outside the active
scope unless a concrete physics use case is supplied.

## Why NuBench and the noise module can appear in one program

They do not represent the same data or likelihood. They share only the abstract
contract

> paired input → declared intervention and severity → frozen representation →
> standardized representation and downstream response.

NuBench supplies the public hit-cloud implementation and exact simulation truth.
The existing PSD/CSD noise module is a direct Paper 3 implementation only if
the real pipeline consumes dense waveforms/traces. If the real pipeline instead
consumes hits, clusters, or higher-level objects, the waveform module supplies
methodological precedent but its waveform results remain evidence for Papers
1–2. Raw severity values and latent distances are never compared or pooled
across modalities.

## Current feasibility result

- Selected Hexagon Ice LE × DynEdge × direction; restored 1,358,099 parameters
  and a 128-dimensional pre-head event embedding.
- Rescored all 2,989,339 released test predictions: medians 23.6548° and
  56.1553°, within 0.055° of Table 6 but outside the strict 0.01° V0 tolerance.
- CPU checkpoint re-inference differs from released directions by 0.944° median
  on 256 development events, so exact environment/backend provenance is the
  current blocker.
- A labeled exploratory module-dropout pilot is sensitive and monotone: 0–50%
  dropout changes median error 15.28°→25.12°, standardized latent displacement
  0→0.149, and 10-NN retention 1.000→0.393.

The pilot is a positive-control/software-feasibility result, not confirmatory
evidence, until checkpoint identity is resolved and the protocol is frozen.

## Decisions requested from the collaborator

- What is the real pipeline's input modality and physics task?
- Are raw inputs available for paired perturbation and re-inference?
- Which intermediate representations can be extracted reproducibly?
- Which data are real, which are simulated, and what noise-state metadata exist?
- Which truth/proxy, geometry, boundary, containment, and grouping fields exist?
- Are outputs point predictions, calibrated probabilities, or predictive
  distributions?
- Is retraining possible, and what compute is realistically available?
- Is there a genuine anomaly-detection decision? If yes, what anomaly class,
  score, nuisance foil, and operating false-positive rate matter?
- Which venue, author roles, and minimal real-domain result would make the
  collaboration worthwhile?

## Explicit non-claims

- No unique physical noise-source decomposition.
- No causal detector-layout conclusion from separately simulated NuBench
  domains.
- No architecture ranking from unmatched single checkpoints.
- No shared likelihood or common numerical severity scale across modalities.
- No anomaly/new-physics claim without a predeclared domain use case.

The detailed plan, definitions, controls, failure conditions, and milestones
are in `PAPER3_PLAN.md`.
