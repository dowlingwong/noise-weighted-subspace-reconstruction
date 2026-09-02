# Paper 2 Reconstruction Stack

> **Status (2026-09-02):** Paper 2 was dissolved. Its theory went to Paper 1
> (separable/NFPA class) and Paper 3/ORACLE (coverage, pullback metric,
> layerwise monitors); see the manuscript repository's
> `plan/THREE_PAPER_STORYLINE_2026-09-02.md`. This package remains the NFPA
> code and results used by the R/PS 2026 paper. The former
> `PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md` is archived under
> `docs/archive/2026-09-02_paper2_dissolution/`.

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
- `npml/`: NPML talk notebooks, support code, slide-ready tables, and figures

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
- `PYTHONPATH=. python paper2/npml/generate_npml_notebooks.py`

See [PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md](PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md)
for the current plan. The superseded 2026-06-09 implementation spec is archived
at
[`../docs/archive/2026-07-26_pre_paper23/paper2_IMPLEMENTATION_SPEC_2026-06-09.md`](../docs/archive/2026-07-26_pre_paper23/paper2_IMPLEMENTATION_SPEC_2026-06-09.md).
for the pre-implementation checklist.
