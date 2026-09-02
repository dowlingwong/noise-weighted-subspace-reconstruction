"""Nested resampling, false-alert calibration, BH-FDR and simple classifiers for the reports."""

from __future__ import annotations

import numpy as np


def far_threshold(clean_scores: np.ndarray, far: float = 0.01) -> float:
    """Threshold giving a false-alert rate ``far`` on clean windows (upper empirical quantile)."""
    s = np.sort(np.asarray(clean_scores, dtype=float))
    k = int(np.ceil((len(s) + 1) * (1 - far))) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


def nested_bootstrap(values: np.ndarray, groups: np.ndarray, seeds: np.ndarray, *, resamples: int = 2000, rng_seed: int = 0) -> dict:
    """Percentile bootstrap resampling perturbation seeds and, within them, event groups."""
    values, groups, seeds = map(np.asarray, (values, groups, seeds))
    rng = np.random.default_rng(rng_seed)
    us = np.unique(seeds)
    means = []
    for _ in range(resamples):
        chosen = rng.choice(us, size=len(us), replace=True)
        acc = []
        for s in chosen:
            m = seeds == s
            g = np.unique(groups[m]); gc = rng.choice(g, size=len(g), replace=True)
            acc.extend(values[m & np.isin(groups, gc)])
        means.append(np.mean(acc))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "ci95": [float(lo), float(hi)], "n": int(len(values)), "excludes_zero": bool(lo > 0 or hi < 0)}


def bh_fdr(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float); m = len(p); order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    rejected = np.zeros(m, dtype=bool)
    k = np.where(p[order] <= thresh)[0]
    if k.size:
        rejected[order[: k.max() + 1]] = True
    return rejected


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels=("N", "S")) -> float:
    f1s = []
    for lab in labels:
        tp = np.sum((y_true == lab) & (y_pred == lab)); fp = np.sum((y_true != lab) & (y_pred == lab)); fn = np.sum((y_true == lab) & (y_pred != lab))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


class LogisticAttribution:
    """L2-regularised multinomial (here binary N/S) logistic classifier on monitor features."""

    def __init__(self, l2: float = 1e-2, iters: int = 500, lr: float = 0.1):
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        t = (np.asarray(y) == "S").astype(float)
        self.w = np.zeros(X.shape[1]); self.b = 0.0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(Xs @ self.w + self.b)))
            self.w -= self.lr * (Xs.T @ (p - t) / len(t) + self.l2 * self.w); self.b -= self.lr * float(np.mean(p - t))
        return self

    def proba_S(self, X):
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b)))

    def predict(self, X):
        return np.where(self.proba_S(X) >= 0.5, "S", "N")
