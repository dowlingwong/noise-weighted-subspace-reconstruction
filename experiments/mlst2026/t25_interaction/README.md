# T25 candidate v1

Mathematical solver, closed configuration, explicit fit enumeration and complete
synthetic-fixture run/verify/reduce path. Production freeze/export intentionally
refuse. This is not an accepted production deployment package.

Human protocol: MS/plan/T25_INTERACTION_PROTOCOL_2026-09-29.md.
Independent reviews and benchmark: MS/reviews/batch_20260929/t25/.

Use the EXP .venv, PYTHONPATH=EXP/src and all BLAS thread variables=1. Run from EXP:

    python -m experiments.mlst2026.t25_interaction --help
    python -m pytest -q experiments/mlst2026/t25_interaction/tests

Production preserves T20-S dimensions/spectrum/thetas/N; physical covariance is
nonisotropic and separable. The full-metric structured solver operates in a fixed
physical product class. Exact affine risk includes propagated noise and offset.
Only training loss selects restarts. Nonconvergence remains a limitation.
No evaluation outcomes or gated archives are read.

Before production: resolve independent deployment findings, enforce the complete
source/scratch/process/receiving envelope, qualify Deepthought, freeze a reviewed
production-capable version in an immutable source snapshot, and run U1 first.
