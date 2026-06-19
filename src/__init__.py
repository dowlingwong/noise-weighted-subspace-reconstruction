"""Noise-weighted subspace reconstruction package.

Production-verified implementations now live under :mod:`src.canonical`. The
re-exports below point at that canonical home so existing import paths keep
working after the reorganization.
"""

from .canonical.make_weights import (
    build_of_one_sided_weights,
    clip_psd_for_weights,
    make_inverse_psd_weights,
)
from .metrics import weighted_cosine, weighted_residual_energy

__all__ = [
    "make_inverse_psd_weights",
    "clip_psd_for_weights",
    "build_of_one_sided_weights",
    "weighted_residual_energy",
    "weighted_cosine",
]
