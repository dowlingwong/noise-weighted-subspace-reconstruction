# Paper 2 Concept: Hybrid ML Triggering and Verification on Long Waveform Streams

## Short version

Paper 2 should not be "all the ML things." It should be a hybrid long-stream detection and verification paper whose baseline is Paper 1.

Best central claim:

> A hybrid candidate-generation and candidate-verification pipeline, trained and evaluated under structured detector noise, outperforms fixed-template OF or pure neural triggering alone on long waveform streams.

That claim is strong, testable, and consistent with the architecture in the attached image.

## What Paper 1 contributes to Paper 2

Paper 1 gives Paper 2 three assets:

- a principled linear baseline: OF / EMPCA / noise-aware linear AE;
- a verification module: EMPCA residual, coefficients, subspace consistency;
- a training principle: respect the noise model in the loss, not only in post-processing.

Paper 2 should cite Paper 1 as the linear-anchor paper and then say it relaxes the linear-encoder restriction in realistic long-stream conditions.

## Recommended scope

### Core pipeline

Use the image pipeline as the main system:

1. Long-stream candidate generation
2. Candidate merge and windowing
3. Candidate verification / representation
4. Fusion / decision layer

### Modules that look mature enough to be core

- sliding OF scan over long waveform streams;
- local OF fit on candidate windows;
- EMPCA verification features;
- ResNet or TCN trigger backbone;
- transformer candidate encoder;
- small MLP fusion / decision head.

### Modules that should be optional unless very mature

- SBI / uncertainty refinement;
- RL trigger policy.

My recommendation:

- keep SBI as an optional late-stage refinement or uncertainty head;
- keep RL as future work unless it is already clearly better than static thresholding and can be explained simply.

If RL is not mature, it will weaken the paper by making the story diffuse.

## Recommended paper story

The story should be:

1. Long streams require candidate generation, not only single-window reconstruction.
2. OF remains useful because it is cheap, interpretable, and physically grounded.
3. A learned trigger backbone catches nonlinear and nonstationary structure missed by OF alone.
4. Candidate windows should then be verified by physically grounded and learned modules together.
5. Fusion of OF, EMPCA, and transformer features is better calibrated and more robust than any single module.

This keeps the system hybrid, not purely black-box.

## Suggested section structure

1. Introduction
2. Problem setup: long-stream trigger and verification task
3. Candidate generation
4. Candidate merge and window construction
5. Candidate encoders and verifiers
6. Fusion and uncertainty
7. Training data and structured-noise simulator
8. Experiments and ablations
9. Discussion and deployment tradeoffs

## Suggested module roles

### Sliding OF scan

Role:

- cheap physics-grounded scorer over the full stream;
- baseline candidate generator;
- interpretable reference trace `A(t)`, `chi^2(t)`, peak positions.

### ResNet / TCN trigger backbone

Role:

- learned candidate generator over the full stream;
- detects waveform families or artifacts that break the linear template assumption;
- outputs `score(t)` and candidate peaks.

### Candidate merge

Role:

- unify OF and NN candidate lists;
- apply local-max logic, priority merge, NMS, deadtime, thresholding.

This step is important enough to be a named algorithmic subsection.

### Local OF fit

Role:

- estimate amplitude, time offset, whitened residual, baseline features;
- provide cheap calibrated features for fusion.

### EMPCA verification

Role:

- subspace consistency test;
- residual diagnostics;
- coefficient features that bridge Paper 1 and Paper 2.

EMPCA should be framed as a verifier, not as the full trigger.

### Transformer candidate encoder

Role:

- geometry-aware, multi-channel local representation learner;
- outputs position, energy, class probability, latent representation, uncertainty, masking diagnostics.

If this module is available but not your main authored contribution, keep it important but modular.

### Fusion / decision MLP

Role:

- combine OF features, EMPCA diagnostics, transformer latent codes, and trigger scores;
- produce final event decision and calibrated confidence.

Use XGBoost as a benchmark, but let the MLP be the main model if it is better.

### SBI / uncertainty refinement

Role:

- optional late-stage posterior refinement for selected candidates;
- uncertainty-calibration tool, not the main detector.

### RL trigger

Role:

- dynamic thresholding / scheduling / resource allocation under latency or bandwidth constraints.

Only include if you can show a clear policy-level objective. Otherwise, move it to future work.

## Recommended baselines

Use clean baseline ladders:

1. OF-only trigger
2. ResNet/TCN-only trigger
3. OF + ResNet merged candidates
4. merged candidates + local OF features
5. merged candidates + EMPCA verifier
6. merged candidates + transformer encoder
7. fused model
8. fused model + SBI refinement

RL should be a separate optional comparison, not part of the main baseline ladder.

## Primary evaluation metrics

- event detection precision / recall / F1
- timing error
- energy error
- false-trigger rate per stream time
- calibration / confidence quality
- robustness under structured noise shifts
- latency / throughput if deployment matters

## Best use of your structured-noise module

Paper 2 is where the nonlinear / nonstationary / correlated / artifact noise module becomes central, not secondary.

Use it for:

- training stress tests;
- domain-shift experiments;
- ablations comparing OF-only, NN-only, and hybrid systems;
- confidence / uncertainty evaluation.

## Recommended titles

Possible title directions:

- Hybrid Noise-Aware Triggering and Verification for Long Waveform Streams
- From Optimal Filtering to Hybrid Learned Triggering in Structured Detector Noise
- Candidate Generation and Verification in Long Cryogenic Waveform Streams under Structured Noise

## Final recommendation

The strongest Paper 2 is:

- one long-stream systems paper;
- one hybrid story;
- one clean baseline ladder;
- structured noise as a central benchmark;
- SBI optional;
- RL optional unless already compelling.

If you keep the scope there, Paper 1 and Paper 2 will cite each other cleanly and not compete for the same narrative space.
