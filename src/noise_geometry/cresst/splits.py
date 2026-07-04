"""Leakage-safe splitting for the public CRESST release.

The noise covariance Sigma is a per-detector-channel quantity: a covariance
estimated from channels A-E does not describe channel F's noise (published
OF resolutions differ by large factors between channels, as do offsets and
noise spectra). Evaluating one channel's pulses under another channel's
metric tests domain transfer, which Paper 1 does not claim. The correct
leakage-safe unit here is therefore *time*, within each channel: sort each
channel's own traces chronologically and split calibration (earlier) from
evaluation (later), never crossing detector-channel boundaries.
"""

from __future__ import annotations

import numpy as np


def unique_groups(features: "object", group_columns: tuple[str, ...]) -> list[tuple]:
    """Distinct ``group_columns`` tuples present in ``features``, sorted."""
    return sorted(set(zip(*(features[c].tolist() for c in group_columns))))


def select_by_groups(
    features: "object", group_columns: tuple[str, ...], groups: set[tuple] | list[tuple]
) -> np.ndarray:
    """Boolean mask of rows whose ``group_columns`` tuple is in ``groups``."""
    wanted = set(groups)
    group_values = list(zip(*(features[c].tolist() for c in group_columns)))
    return np.array([g in wanted for g in group_values])


def time_disjoint_split(time_values: np.ndarray, calibration_fraction: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split: earliest ``calibration_fraction`` -> calibration, rest -> evaluation.

    Applied within a single detector channel. Calibration (covariance, PSD,
    template bank, subspace) and evaluation (test pulses, background-FPR
    noise, injection substrate) are then time-disjoint but never cross a
    channel boundary, matching the fact that Sigma is a per-channel quantity.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    time_values = np.asarray(time_values, dtype=np.float64)
    n = time_values.shape[0]
    order = np.argsort(time_values, kind="stable")
    n_calib = max(1, min(n - 1, int(round(calibration_fraction * n))))
    calib_mask = np.zeros(n, dtype=bool)
    calib_mask[order[:n_calib]] = True
    return calib_mask, ~calib_mask


def count_duplicate_traces(traces: np.ndarray) -> int:
    """Count exact-duplicate rows (extra copies beyond the first)."""
    X = np.ascontiguousarray(np.asarray(traces))
    flat = X.view([("", X.dtype)] * X.shape[1]).reshape(-1)
    _, counts = np.unique(flat, return_counts=True)
    return int(np.sum(counts[counts > 1] - 1))
