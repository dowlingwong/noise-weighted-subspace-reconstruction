"""Canonical (production-verified) implementations.

These modules are the trusted reference implementations used as oracles for
Paper 1 verification:

- ``empca`` - Stephen Bailey's published reference Weighted EMPCA (real-valued).
- ``empca_TCY_optimized`` - the optimized complex/rfft EMPCA estimator.
- ``empca_equivalence_utils`` - paper-grade no-smoothing EMPCA driver plus
  weighting, GLS projection, and the rfft->real feature mapping.
- ``OptimumFilter`` / ``PSDCalculator`` / ``make_weights`` / ``weights`` -
  matched filter, PSD, and OF-convention weighting.
"""
