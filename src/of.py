"""Small weighted-GLS helpers used by notebooks and package imports."""

from __future__ import annotations

import numpy as np


def gls_amplitude(x_f, template_f, weights, return_complex: bool = False):
    """Return the weighted rank-1 GLS amplitude in frequency space."""
    x_f = np.asarray(x_f)
    template_f = np.asarray(template_f)
    weights = np.asarray(weights, dtype=np.float64)

    denom = np.sum(np.conjugate(template_f) * template_f * weights, axis=-1)
    numer = np.sum(np.conjugate(template_f) * x_f * weights, axis=-1)
    coeff = numer / denom
    return coeff if return_complex else np.real(coeff)


def project_rank1(x_f, template_f, weights, return_complex: bool = False):
    """Project one or many traces onto a single weighted template."""
    coeff = gls_amplitude(x_f, template_f, weights, return_complex=True)
    recon = np.asarray(coeff)[..., None] * np.asarray(template_f)[None, :]
    if return_complex:
        return coeff, recon
    return np.real(coeff), recon


__all__ = [
    "gls_amplitude",
    "project_rank1",
]
