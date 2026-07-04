"""Multi-template optimal-filter bank (GLS template bank).

A small bank of empirical templates over the ``rise_time``/``decay_time``
quantile grid, each evaluated at a grid of ``t0`` sample shifts, scored per
trace by minimum weighted residual (equivalently, maximum GLS detection
statistic). This is the strong classical baseline the CRESST validation
protocol requires alongside single-template OF, PCA/EMPCA, and the linear AE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemplateBank:
    templates: np.ndarray  # (n_templates, n_features), unit peak amplitude
    cell_labels: list[dict]
    t0_grid: np.ndarray  # integer sample shifts


def _shift(template: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return template
    out = np.zeros_like(template)
    if shift > 0:
        out[shift:] = template[:-shift]
    else:
        out[:shift] = template[-shift:]
    return out


def build_multi_template_bank(
    train_pulses: np.ndarray,
    rise_time: np.ndarray,
    decay_time: np.ndarray,
    *,
    rise_time_quantiles: tuple[float, ...] = (1 / 3, 2 / 3),
    decay_time_quantiles: tuple[float, ...] = (1 / 3, 2 / 3),
    t0_grid_samples: tuple[int, ...] = (-4, -2, 0, 2, 4),
    min_cell_pulses: int = 2,
) -> TemplateBank:
    """Build empirical templates from rise/decay-time quantile bins.

    Each bank cell's template is the mean pulse shape (normalized to unit
    peak amplitude) of the training pulses whose ``rise_time``/``decay_time``
    fall in that quantile bin. Bins with fewer than ``min_cell_pulses``
    pulses are skipped.
    """
    train_pulses = np.asarray(train_pulses, dtype=np.float64)
    rise_time = np.asarray(rise_time, dtype=np.float64)
    decay_time = np.asarray(decay_time, dtype=np.float64)
    if not (train_pulses.shape[0] == rise_time.shape[0] == decay_time.shape[0]):
        raise ValueError("train_pulses, rise_time, decay_time must be row-aligned")

    rise_edges = np.quantile(rise_time, rise_time_quantiles)
    decay_edges = np.quantile(decay_time, decay_time_quantiles)
    rise_bin = np.digitize(rise_time, rise_edges)
    decay_bin = np.digitize(decay_time, decay_edges)

    templates: list[np.ndarray] = []
    labels: list[dict] = []
    for rb in range(len(rise_time_quantiles) + 1):
        for db in range(len(decay_time_quantiles) + 1):
            mask = (rise_bin == rb) & (decay_bin == db)
            n_cell = int(mask.sum())
            if n_cell < min_cell_pulses:
                continue
            template = train_pulses[mask].mean(axis=0)
            peak = np.max(np.abs(template))
            if peak <= 0:
                continue
            templates.append(template / peak)
            labels.append({"rise_bin": rb, "decay_bin": db, "n_pulses": n_cell})

    if not templates:
        raise ValueError("no template-bank cell had enough pulses; loosen quantiles or min_cell_pulses")

    return TemplateBank(
        templates=np.asarray(templates, dtype=np.float64),
        cell_labels=labels,
        t0_grid=np.asarray(t0_grid_samples, dtype=int),
    )


def score_multi_template_of(test: np.ndarray, bank: TemplateBank, metric: np.ndarray) -> dict[str, np.ndarray]:
    """Score every trace against the whole bank; keep the minimum-residual fit.

    Uses the standard GLS reduction ``chi2_min = chi2_null - (s^T M x)^2 / (s^T M s)``
    so the full bank (templates x shifts) is scored in ``O(n_traces * n_features)``
    per candidate rather than a full quadratic form each time.
    """
    X = np.asarray(test, dtype=np.float64)
    M = np.asarray(metric, dtype=np.float64)
    n_traces, n_features = X.shape
    chi2_null = np.einsum("ij,jk,ik->i", X, M, X)

    best_chi2 = chi2_null.copy()
    best_amplitude = np.zeros(n_traces)
    best_template_index = np.full(n_traces, -1, dtype=int)
    best_shift = np.zeros(n_traces, dtype=int)

    for ti, template in enumerate(bank.templates):
        for shift in bank.t0_grid:
            shifted = _shift(template, int(shift))
            m_shifted = M @ shifted
            denom = float(shifted @ m_shifted)
            if denom <= 0:
                continue
            proj = X @ m_shifted
            amp = proj / denom
            chi2 = chi2_null - (proj**2) / denom
            better = chi2 < best_chi2
            best_chi2 = np.where(better, chi2, best_chi2)
            best_amplitude = np.where(better, amp, best_amplitude)
            best_template_index = np.where(better, ti, best_template_index)
            best_shift = np.where(better, int(shift), best_shift)

    return {
        "weighted_residual": best_chi2 / n_features,
        "amplitude": best_amplitude,
        "template_index": best_template_index,
        "shift": best_shift,
    }
