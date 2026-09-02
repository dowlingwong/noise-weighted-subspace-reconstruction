"""Consequence variables (ORACLE proposal §4, C4)."""

from __future__ import annotations

import numpy as np


def amplitude_rmse(outputs: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((outputs - truth) ** 2)))


class AmplitudeDegradation:
    """K = RMSE(window) / RMSE(clean) - 1 : relative degradation of the frozen amplitude
    readout; 'high consequence' is K >= kappa_m."""
    name = "amplitude_relative_degradation"

    def __init__(self, clean_rmse: float, kappa_m: float = 0.10):
        self.clean_rmse, self.kappa_m = clean_rmse, kappa_m

    def score(self, outputs: np.ndarray, truth: np.ndarray) -> float:
        return amplitude_rmse(outputs, truth) / max(self.clean_rmse, 1e-12) - 1.0

    def high(self, K: float) -> bool:
        return bool(K >= self.kappa_m)
