# Remaining Gaps for Experiments D and E

This list is shorter than before: the paper2 package now has concrete dataset,
model, loss, and trainer interfaces. The remaining blockers are:

## Hard blockers

1. `torch` is not installed in the current runtime.
2. No actual training run has been executed yet, so there is no validated
   checkpoint path or empirical learning-curve baseline.

## Data-scope blockers

1. The currently available K-alpha dataset is single-channel:
   - `data/k_alpha/k_alpha_traces.h5` has shape `(4358, 32768)`
   - therefore Experiment E is currently runnable only in a single-channel
     reconstruction form
2. There is no native detector-position label in the current HDF5 files, so
   the full Paper 2 “unseen position” evaluation needs either:
   - a synthetic dataset with explicit position latent, or
   - a real multichannel/position-aware dataset

## Architecture-bias expansion blockers

1. No Paper 2 wrapper exists yet for:
   - `src/CNN/resnet_1d.py`
   - `src/transformer/model_pairwise.py`
   - `src/transformer/model_triangular_pairwise.py`
   - `src/transformer/model_pairwise_channel_masking.py`
2. The existing `src/transformer/*.py` files are task heads / backbones, not
   native-output reconstruction models.
3. `reconstruction.training.muon` is still missing, so the original optimizer
   path from `src/transformer/` is not available.

## Evaluation gaps

1. No latent-probe evaluation for amplitude / timing RMSE is implemented yet.
2. The generalized server runner exists at
   `scripts/run_paper2_training_suite.py`, but no Experiment D/E run has been
   executed yet.
3. No figure-generation scripts exist yet under `paper2/` for:
   - 2x2 matrix summary plots
   - architecture-bias comparison plots
