"""Generic shift-detection baselines, given the same windows and the same clean reference.

* ``InputKS``           — per-coordinate two-sample KS on a fixed random subset of whitened
                          input coordinates, Benjamini–Hochberg corrected; alarm = fraction rejected.
* ``MMD``               — RBF-kernel MMD^2 (median-heuristic bandwidth on the clean reference).
* ``C2ST``              — classifier two-sample test: logistic regression, held-out AUC.
* ``EmbeddingMoments``  — |Delta mean|_2 + |Delta cov|_F on the stage representation.
All are standardised by their clean-window dispersion like the monitors.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .monitors import _Base
from .protocol import RepresentationBatch


class InputKS(_Base):
    def __init__(self, n_coords: int = 64, alpha: float = 0.05, seed: int = 0):
        super().__init__(name="input_ks_bh")
        self.n_coords, self.alpha, self.seed = n_coords, alpha, seed

    def _fit_extra(self, clean):
        X = np.concatenate([b.meta["X_whitened"] for b in clean])
        rng = np.random.default_rng(self.seed)
        self.coords = rng.choice(X.shape[1], size=min(self.n_coords, X.shape[1]), replace=False)
        self.ref = X[:, self.coords]

    def raw(self, batch):
        X = batch.meta["X_whitened"][:, self.coords]
        p = np.array([stats.ks_2samp(self.ref[:, j], X[:, j]).pvalue for j in range(X.shape[1])])
        m = len(p); order = np.argsort(p); thresh = self.alpha * (np.arange(1, m + 1) / m)
        rejected = np.zeros(m, dtype=bool)
        k = np.where(p[order] <= thresh)[0]
        if k.size:
            rejected[order[: k.max() + 1]] = True
        return float(rejected.mean())


class MMD(_Base):
    def __init__(self, n_ref: int = 300, seed: int = 0):
        super().__init__(name="mmd_rbf")
        self.n_ref, self.seed = n_ref, seed

    def _fit_extra(self, clean):
        Z = np.concatenate([b.Z for b in clean])
        rng = np.random.default_rng(self.seed)
        self.ref = Z[rng.choice(len(Z), size=min(self.n_ref, len(Z)), replace=False)]
        d = np.sum((self.ref[:, None] - self.ref[None]) ** 2, axis=-1)
        self.gamma = 1.0 / max(float(np.median(d[d > 0])), 1e-12)

    def _k(self, A, B):
        d = np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None] - 2 * A @ B.T
        return np.exp(-self.gamma * d)

    def raw(self, batch):
        X, Y = self.ref, batch.Z
        return float(self._k(X, X).mean() + self._k(Y, Y).mean() - 2 * self._k(X, Y).mean())


class C2ST(_Base):
    def __init__(self, n_ref: int = 300, seed: int = 0, l2: float = 1e-2, iters: int = 300):
        super().__init__(name="c2st_logistic_auc")
        self.n_ref, self.seed, self.l2, self.iters = n_ref, seed, l2, iters

    def _fit_extra(self, clean):
        Z = np.concatenate([b.Z for b in clean])
        rng = np.random.default_rng(self.seed)
        self.ref = Z[rng.choice(len(Z), size=min(self.n_ref, len(Z)), replace=False)]
        self.scale = Z.std(axis=0) + 1e-9
        self.centre = Z.mean(axis=0)

    def raw(self, batch):
        rng = np.random.default_rng(self.seed + 1)
        A = (self.ref - self.centre) / self.scale; B = (batch.Z - self.centre) / self.scale
        n = min(len(A), len(B)); A = A[rng.permutation(len(A))[:n]]; B = B[rng.permutation(len(B))[:n]]
        X = np.vstack([A, B]); y = np.r_[np.zeros(n), np.ones(n)]
        idx = rng.permutation(2 * n); tr, te = idx[: n], idx[n:]
        w = np.zeros(X.shape[1]); b = 0.0; lr = 0.1
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(X[tr] @ w + b)))
            g = X[tr].T @ (p - y[tr]) / len(tr) + self.l2 * w
            w -= lr * g; b -= lr * float(np.mean(p - y[tr]))
        s = X[te] @ w + b
        pos, neg = s[y[te] == 1], s[y[te] == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        return float(np.mean(pos[:, None] > neg[None]) + 0.5 * np.mean(pos[:, None] == neg[None]))


class EmbeddingMoments(_Base):
    def __init__(self):
        super().__init__(name="embedding_moments")

    def _fit_extra(self, clean):
        Z = np.concatenate([b.Z for b in clean])
        self.cov = np.cov(Z.T) if Z.shape[1] > 1 else np.var(Z, ddof=1).reshape(1, 1)

    def raw(self, batch):
        Z = batch.Z
        cov = np.cov(Z.T) if Z.shape[1] > 1 else np.var(Z, ddof=1).reshape(1, 1)
        return float(np.linalg.norm(Z.mean(0) - self.mu_stage) + np.linalg.norm(cov - self.cov, "fro"))


def make_baseline_set() -> list[_Base]:
    return [InputKS(), MMD(), C2ST(), EmbeddingMoments()]
