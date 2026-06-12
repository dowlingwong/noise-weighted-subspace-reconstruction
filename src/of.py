"""Weighted (GLS) amplitude estimation and rank-1 projection.

Frequency-domain implementations of the paper's core estimators:

    Â = ⟨s, x⟩_W / ⟨s, s⟩_W          (gls_amplitude)
    x̂ = Â s                          (project_rank1)

where ⟨a, b⟩_W = Σ_k conj(a_k) b_k w_k and `w` are inverse-PSD weights in
the convention of `src.make_weights.build_of_one_sided_weights`.

Relationship to `src/OptimumFilter.py`
--------------------------------------
`OptimumFilter` is the CANONICAL OF used for all paper results (FFT filter
kernel, time shifts, chi-square, sliding fits). This module is an
intentionally independent ~40-line reference implementation of the same
zero-shift amplitude in the subspace/GLS form used by EMPCA. The two are
required to agree to ~1e-8 (see tests/test_rank1_of_empca_equivalence.py);
that agreement verifies the weighting/preprocessing conventions are matched,
which is the hypothesis of the paper's OF = rank-1 EMPCA equivalence theorem.
Do NOT replace this with a call into OptimumFilter — that would make the
cross-check circular.
"""

import numpy as np

from .metrics import weighted_inner


def gls_amplitude(X_f, s_f, w, return_complex=False):
    """GLS / OF amplitude(s) of traces `X_f` against template `s_f`.

    Parameters
    ----------
    X_f : array_like, shape (n_bins,) or (n_obs, n_bins)
        rfft of trace(s) (same transform convention as `s_f`).
    s_f : array_like, shape (n_bins,)
        rfft of the signal template.
    w : array_like, shape (n_bins,)
        Inverse-PSD weights (OF one-sided convention).
    return_complex : bool, default False
        If False, return the real part (ML estimate for a real amplitude).

    Returns
    -------
    amp : float or ndarray, shape (n_obs,)
    """
    X_arr = np.asarray(X_f)
    single = X_arr.ndim == 1
    X2 = np.atleast_2d(X_arr)
    s_f = np.asarray(s_f)
    w = np.asarray(w, dtype=np.float64)

    den = np.real(weighted_inner(s_f, s_f, w))
    if den <= 0:
        raise ValueError("Template has zero weighted norm.")
    num = np.sum(np.conj(s_f)[None, :] * X2 * w[None, :], axis=1)
    amp = num / den
    if not return_complex:
        amp = np.real(amp)
    return amp[0] if single else amp


def project_rank1(X_f, s_f, w, return_amp=False):
    """Rank-1 weighted projection x̂ = Â s of trace(s) onto the template.

    Returns the reconstruction(s); with `return_amp=True` also returns Â.
    """
    X_arr = np.asarray(X_f)
    single = X_arr.ndim == 1
    X2 = np.atleast_2d(X_arr)
    amp = gls_amplitude(X2, s_f, w)
    recon = np.asarray(amp)[:, None] * np.asarray(s_f)[None, :]
    if single:
        recon = recon[0]
        amp = float(np.asarray(amp)[0])
    if return_amp:
        return recon, amp
    return recon
