# NPML Experiments: Metric, Coverage, and Inductive Bias

## Goal

Demonstrate that scientific reconstruction depends on three factors:

1. Likelihood geometry (loss function)
2. Signal-manifold coverage
3. Architectural inductive bias

The experiments should isolate these three effects.

---

# Experiment A: Metric Ablation

## Question

Does the detector metric change the learned representation?

## Models

* PCA
* EMPCA

## Data

Synthetic traces:

x = s(z) + n

with colored Gaussian noise.

Noise types:

* White
* Pink
* Brownian
* Measured MMC PSD

## Metrics

* Weighted residual
* Reconstruction MSE
* Principal angle between subspaces
* Amplitude resolution

## Expected Result

White noise:

PCA ≈ EMPCA

Colored noise:

PCA ≠ EMPCA

The learned representation changes when the metric changes.

---

# Experiment B: Coverage Ablation

## Question

Can the correct loss recover latent factors that never appear during training?

## Latent Variables

z = (A,t0,p,gamma)

where

A = amplitude
t0 = arrival time
p = position
gamma = pulse-shape parameter

## Training Conditions

### Full Coverage

A,t0,p,gamma all varied

### Timing Restricted

t0 fixed

### Position Restricted

p fixed

### Shape Restricted

gamma fixed

## Test Set

Always use full latent variation.

## Metrics

* Amplitude RMSE
* Timing RMSE
* Position RMSE
* Weighted residual

## Expected Result

Restricted training support produces failure on omitted latent directions even with the correct Mahalanobis loss.

---

# Experiment C: NFPA vs EMPCA

## Question

When does channel-time factorization help?

## Models

* EMPCA
* NFPA

## Conditions

### Separable Signals

Generated from

signal ≈ channel_basis ⊗ time_basis

### Non-Separable Signals

Introduce position-dependent pulse distortions.

## Metrics

* Weighted residual
* Reconstruction error
* Subspace angle

## Expected Result

NFPA performs nearly identically in separable regimes and degrades gracefully when separability fails.

---

# Experiment D: Architecture Bias

## Question

Does architecture select different likelihood-compatible solutions?

## Models

* Linear AE
* CNN AE
* Transformer AE

All trained using identical Mahalanobis objective.

## Metrics

* Weighted residual
* Reconstruction RMSE
* Generalization to unseen positions
* Generalization to unseen pulse shapes

## Expected Result

Different architectures achieve similar training likelihood but different generalization behaviour.

---

# Experiment E: Prewhitened Transformer

## Question

Should attention operate in detector geometry?

## Models

1. Raw Transformer + MSE
2. Raw Transformer + Mahalanobis Loss
3. Prewhitened Transformer + MSE
4. Prewhitened Transformer + Mahalanobis Loss

## Whitening

x_tilde = Sigma^{-1/2} x

## Metrics

* Weighted residual
* Amplitude resolution
* Timing resolution
* Generalization

## Expected Result

Prewhitening improves stability and aligns attention with detector geometry.

---

# Final Figure

Construct a 2x2 matrix:

```
                Full Coverage    Restricted Coverage
```

Correct Metric      Best Case        Coverage Failure

Wrong Metric        Metric Failure   Worst Case

This becomes the central figure for Paper 2 and future NPML talks.

