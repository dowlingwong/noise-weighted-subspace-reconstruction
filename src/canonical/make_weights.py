"""PSD-to-weight helpers.

Canonical weighting convention for this repository (matches
`OptimumFilter._update_state` PSD unfolding and
`src/EMPCA/empca_equivalence_utils.build_of_one_sided_weights`):

- input is a one-sided PSD `J` with `trace_len // 2 + 1` bins (rfft layout)
- DC bin gets weight 0 (OF excludes it via an infinite-PSD bin)
- interior bins get weight 2 / J[k] (one-sided PSD folds two two-sided bins)
- the Nyquist bin (even trace_len only) gets weight 1 / (2 J[-1]): the OF
  unfolds the one-sided PSD with J_two(Nyq) = J(Nyq) (no folding) and counts
  the bin once, so its relative weight is 1/4 of an interior bin's. Using
  1 / J[-1] here (the previous rule) made GLS amplitudes deviate from
  OptimumFilter.fit by O(0.1%) for steep PSDs.

Using these weights with rfft-domain data reproduces the exact inner product
the OptimumFilter uses, which is what the OF = rank-1 EMPCA equivalence
requires.
"""

import numpy as np


def clip_psd_for_weights(psd, floor=None, floor_quantile=1e-6):
    """Return a copy of `psd` with non-positive / tiny bins clipped.

    Prevents infinite or numerically explosive 1/J weights.

    Parameters
    ----------
    psd : array_like
        One-sided PSD values (any length).
    floor : float, optional
        Absolute clipping floor. If None, the floor is
        `floor_quantile * max(psd[psd > 0])`.
    floor_quantile : float, default 1e-6
        Relative floor used when `floor` is None.
    """
    J = np.asarray(psd, dtype=np.float64).copy()
    positive = J[J > 0]
    if positive.size == 0:
        raise ValueError("PSD has no positive bins; cannot build weights.")
    if floor is None:
        floor = float(floor_quantile * positive.max())
    return np.clip(J, floor, None)


def build_of_one_sided_weights(J_psd, trace_len):
    """OF-convention weights for a one-sided PSD (see module docstring)."""
    J = np.asarray(J_psd, dtype=np.float64)
    n_rfft = trace_len // 2 + 1
    if J.shape[0] != n_rfft:
        raise ValueError(f"PSD length {J.shape[0]} does not match rfft bins {n_rfft}")

    w = np.zeros_like(J)
    w[0] = 0.0
    if trace_len % 2 == 0:
        w[1:-1] = 2.0 / J[1:-1]
        w[-1] = 1.0 / (2.0 * J[-1])
    else:
        w[1:] = 2.0 / J[1:]
    return w


def make_inverse_psd_weights(psd, trace_len=None, clip_floor=None,
                             floor_quantile=1e-6, zero_dc=True):
    """Build inverse-PSD weights, with clipping.

    If `trace_len` is given, uses the OF one-sided convention
    (`build_of_one_sided_weights`). Otherwise returns plain `1 / J`
    (with the DC bin zeroed when `zero_dc`).
    """
    J = clip_psd_for_weights(psd, floor=clip_floor, floor_quantile=floor_quantile)
    if trace_len is not None:
        return build_of_one_sided_weights(J, trace_len)
    w = 1.0 / J
    if zero_dc:
        w[0] = 0.0
    return w
