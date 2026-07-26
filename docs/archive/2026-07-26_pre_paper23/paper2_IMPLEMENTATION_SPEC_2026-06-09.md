# Paper 2 Implementation Spec

## Goal

Implement a shared reconstruction pipeline that supports:

- `raw + mse`
- `raw + mahalanobis`
- `prewhitened + mse`
- `prewhitened + mahalanobis`

for both:

- a simple reconstruction AE
- a token-preserving transformer reconstruction model

## Architectural invariants

These are non-negotiable if the geometry story is to stay clean:

1. Inputs are native detector traces `x`.
2. Encoder input is either `x` or `x_tilde = W x`.
3. Decoder output is always native-space `x_hat`.
4. Loss is chosen independently of encoder input mode.
5. The same train/val/test split and optimizer budget are used for the full
   transformer `2x2`.

## Required dependencies before real training

1. Install `torch`.
2. Decide whether `src/transformer/model_original.py` will be wrapped directly
   or whether its encoder trunk will be reimplemented locally under
   `paper2/models/`.
3. Decide whether `reconstruction.training.muon` is required; current scaffold
   assumes `AdamW` for the first pass.
4. Confirm the PSD source used for whitening:
   - canonical one-sided PSD `.npy`
   - or empirical PSD estimated from training baselines
5. Confirm the training dataset shape:
   - single-channel or multi-channel
   - fixed trace length
   - HDF5 field names

## First implementation milestone

1. `WhiteningOperator`
2. `ReconstructionCriterion`
3. `ReconstructionAE`
4. `run_experiment()`
5. AE `2x2`
6. token-preserving transformer encoder wrapper
7. transformer `2x2`

## Experiment order

### Phase 1

- `ae_raw_mse`
- `ae_raw_mahalanobis`
- `ae_prewhite_mse`
- `ae_prewhite_mahalanobis`

### Phase 2

- `transformer_raw_mse`
- `transformer_raw_mahalanobis`
- `transformer_prewhite_mse`
- `transformer_prewhite_mahalanobis`

### Phase 3

Fix geometry to `prewhitened + mahalanobis`, then compare:

- AE
- CNN AE
- original transformer
- pairwise transformer
- triangular pairwise transformer
- channel-masking transformer

## Output artifacts per run

Each experiment should write:

```text
paper2/results/<experiment_name>/
  config.yaml
  metrics.json
  curves.csv
  checkpoint_best.pt
  predictions_test.h5
  figures/
```
