"""Split-conformal abstention and selective attribution (C2)."""

from __future__ import annotations

import numpy as np


def conformal_threshold(nonconformity_calib: np.ndarray, alpha: float = 0.1) -> float:
    """Level-(1-alpha) split-conformal quantile of calibration nonconformity scores."""
    s = np.sort(np.asarray(nonconformity_calib, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[min(max(k, 0), n - 1)])


def risk_coverage(nonconformity: np.ndarray, correct: np.ndarray) -> dict:
    """Selective risk as a function of retained coverage when abstaining on the largest scores."""
    order = np.argsort(nonconformity)
    correct = np.asarray(correct, dtype=float)[order]
    n = len(correct)
    if n == 0:
        return {"coverage": np.zeros(0), "risk": np.zeros(0), "risk_coverage_auc": float("nan"), "risk_at_80_coverage": float("nan")}
    cov = np.arange(1, n + 1) / n
    risk = 1.0 - np.cumsum(correct) / np.arange(1, n + 1)
    auc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))
    i80 = int(np.searchsorted(cov, 0.8))
    return {"coverage": cov, "risk": risk, "risk_coverage_auc": auc, "risk_at_80_coverage": float(risk[min(i80, n - 1)])}


def auroc(scores_unknown: np.ndarray, scores_known: np.ndarray) -> float:
    a, b = np.asarray(scores_unknown), np.asarray(scores_known)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(np.mean(a[:, None] > b[None]) + 0.5 * np.mean(a[:, None] == b[None]))
