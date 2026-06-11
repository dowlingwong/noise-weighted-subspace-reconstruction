# Paper 2 Reconstruction Stack

This folder defines the implementation scaffold for the nonlinear follow-up
paper:

1. reconstruction AE as the first nonlinear baseline;
2. transformer reconstruction with a fixed geometry-vs-loss `2x2`;
3. later architecture-bias ablations after geometry is fixed.

The stack is intentionally split into:

- `data/`: dataset, whitening, and split contracts
- `losses/`: reconstruction losses and evaluation metrics
- `models/`: AE, transformer encoder/decoder wrappers, shared interfaces
- `trainers/`: training loop, evaluation loop, experiment runner
- `configs/`: exact experiment YAMLs

The code here is a concrete implementation spec:

- classes and dataclasses are defined
- function signatures are fixed
- configs are explicit
- runtime logic is deliberately light until `torch` and training dependencies
  are installed

Useful entry points:

- `python -m paper2.trainers.run_experiment_suite d`
- `python -m paper2.trainers.run_experiment_suite e`
- `python -m paper2.analysis.real_metric_coverage_matrix --train --analyze`

See [IMPLEMENTATION_SPEC.md](/Users/wongdowling/Documents/noise-weighted-subspace-reconstruction/paper2/IMPLEMENTATION_SPEC.md)
for the pre-implementation checklist.
