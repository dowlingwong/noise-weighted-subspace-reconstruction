"""Gradient-trained tied linear autoencoders for the S3 EMPCA bridge.

This module provides an *independently trained* tied linear autoencoder so the
S3 claim ("a weighted tied linear AE recovers the EMPCA subspace") is verified
by actual optimisation rather than by delegating to ``fit_weighted_pca``.

Two training paths are provided, both minimising the same noise-weighted
reconstruction objective

    L(D) = (1 / N) * sum_n  r_n^T M r_n ,   r_n = x_n - D D^T M x_n ,

where ``M = Sigma^{-1}`` is the inverse-noise metric (a diagonal weight vector
or a full positive-definite matrix) and ``D`` (shape ``d x k``) is the tied
decoder/encoder basis (encoder = ``D^T M``):

* :func:`train_weighted_linear_ae` ("direct") optimises ``D`` in the original
  data coordinates with the weighted loss above. This is the literal paper
  claim.
* :func:`train_whitened_linear_ae` ("whitened") whitens the data with
  ``M^{1/2}``, trains a *standard* MSE tied linear AE (Baldi-Hornik guarantees
  the PCA subspace), then maps the basis back to data units. This is the stable
  cross-check.

Both paths support full-batch L-BFGS in float64 (default, high precision) and a
first-order Adam pass (reported, realistic-regime evidence). Convergence is
judged against the EMPCA global optimum ``L*`` via the optimality gap, plus the
principal angle to ``fit_weighted_pca`` and the M-orthonormality error
``||D^T M D - I||``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..subspace import fit_weighted_pca, principal_angles, project_onto_basis


# --------------------------------------------------------------------------- #
# Metric helpers (support diagonal weight vector or full SPD matrix)
# --------------------------------------------------------------------------- #
def _as_metric(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim not in (1, 2):
        raise ValueError("weights must be a vector or square matrix")
    if w.ndim == 2 and w.shape[0] != w.shape[1]:
        raise ValueError("matrix weights must be square")
    return w


def _m_right(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply metric on the feature (last) axis: returns ``A @ M``."""
    return A * w[None, :] if w.ndim == 1 else A @ w


def _m_left(B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply metric on the feature (first) axis of a basis: returns ``M @ B``."""
    return B * w[:, None] if w.ndim == 1 else w @ B


def _metric_sqrt(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(M^{1/2}, M^{-1/2})`` for whitening."""
    if w.ndim == 1:
        sqrt_w = np.sqrt(np.clip(w, 0.0, None))
        inv = np.zeros_like(sqrt_w)
        mask = sqrt_w > 0
        inv[mask] = 1.0 / sqrt_w[mask]
        return sqrt_w, inv  # 1-D representations
    vals, vecs = np.linalg.eigh(0.5 * (w + w.T))
    vals = np.clip(vals, np.finfo(float).eps, None)
    msqrt = (vecs * np.sqrt(vals)[None, :]) @ vecs.T
    msqrt_inv = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
    return msqrt, msqrt_inv


# --------------------------------------------------------------------------- #
# Loss, gradient, optimisers
# --------------------------------------------------------------------------- #
def _loss_and_grad(D: np.ndarray, X: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted reconstruction loss and its gradient w.r.t. the basis ``D``.

    ``X`` rows are centred observations. ``D`` has shape ``(d, k)``.
    """
    n = X.shape[0]
    MX = _m_right(X, w)              # rows: (M x_n)^T
    Z = MX @ D                       # rows: z_n^T = x_n^T M D
    Xhat = Z @ D.T                   # rows: x_hat_n^T = (D D^T M x_n)^T
    R = X - Xhat
    MR = _m_right(R, w)              # rows: (M r_n)^T
    loss = float(np.sum(MR * R) / n)
    grad = (-2.0 / n) * (MR.T @ Z + MX.T @ (MR @ D))
    return loss, grad


def _lbfgs(X: np.ndarray, w: np.ndarray, D0: np.ndarray, max_iter: int) -> tuple[np.ndarray, int]:
    d, k = D0.shape

    def fun(vec: np.ndarray) -> tuple[float, np.ndarray]:
        loss, grad = _loss_and_grad(vec.reshape(d, k), X, w)
        return loss, grad.ravel()

    res = minimize(
        fun,
        D0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "maxfun": max_iter * 10, "ftol": 1e-16, "gtol": 1e-12},
    )
    return res.x.reshape(d, k), int(res.nit)


def _adam(
    X: np.ndarray,
    w: np.ndarray,
    D0: np.ndarray,
    max_iter: int,
    lr: float,
    tol: float,
) -> tuple[np.ndarray, int, list[float]]:
    D = D0.copy()
    m = np.zeros_like(D)
    v = np.zeros_like(D)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history: list[float] = []
    prev = np.inf
    nit = 0
    for t in range(1, max_iter + 1):
        loss, grad = _loss_and_grad(D, X, w)
        history.append(loss)
        nit = t
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad * grad)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        D = D - lr * mhat / (np.sqrt(vhat) + eps)
        if abs(prev - loss) < tol * max(1.0, abs(prev)):
            break
        prev = loss
    return D, nit, history


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def weighted_reconstruction_loss(
    samples: np.ndarray, components: np.ndarray, weights: np.ndarray, mean: np.ndarray
) -> float:
    """Weighted loss of the *optimal* (weighted-LS) reconstruction onto a basis.

    This matches the AE objective at its optimum and gives the value to compare
    against. Used both for ``L*`` (EMPCA basis) and for any subspace.
    """
    w = _as_metric(weights)
    recon = project_onto_basis(samples, components, weights=w, mean=mean)
    r = np.asarray(samples, dtype=np.float64) - recon
    return float(np.sum(_m_right(r, w) * r) / samples.shape[0])


def empca_optimal_loss(samples: np.ndarray, weights: np.ndarray, rank: int) -> tuple[float, object]:
    """Return ``(L*, empca_fit)`` for the weighted rank-``k`` subspace."""
    empca = fit_weighted_pca(samples, weights, rank)
    lstar = weighted_reconstruction_loss(samples, empca.components, weights, empca.mean)
    return lstar, empca


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class TrainedAEResult:
    """Outcome of a trained tied linear AE and its bridge diagnostics."""

    method: str                       # "direct" or "whitened"
    optimizer: str                    # "lbfgs" or "adam"
    components: np.ndarray            # (k, d) row vectors in data units
    decoder: np.ndarray              # D (d, k) in the trained coordinate system
    mean: np.ndarray
    weights: np.ndarray
    final_loss: float
    optimal_loss: float              # L* from EMPCA
    optimality_gap: float            # final_loss - optimal_loss
    relative_gap: float              # optimality_gap / |optimal_loss|
    max_principal_angle_deg: float   # vs fit_weighted_pca, in the M-metric
    m_orthonormality_error: float    # ||D^T M D - I||_max in data units
    n_iter: int
    loss_history: np.ndarray = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Public training entry points
# --------------------------------------------------------------------------- #
def _init_basis(Xc: np.ndarray, w: np.ndarray, rank: int, rng: np.random.Generator) -> np.ndarray:
    """Random (NOT EMPCA-seeded) M-orthonormal initial basis for conditioning."""
    d = Xc.shape[1]
    D = rng.standard_normal((d, rank))
    # M-orthonormalise the columns via QR in whitened coordinates.
    msqrt, msqrt_inv = _metric_sqrt(w)
    Dw = (msqrt[:, None] * D) if w.ndim == 1 else (msqrt @ D)
    Q, _ = np.linalg.qr(Dw)
    return (msqrt_inv[:, None] * Q) if w.ndim == 1 else (msqrt_inv @ Q)


def _finalise(
    method: str,
    optimizer: str,
    components: np.ndarray,
    D_data: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray,
    samples: np.ndarray,
    rank: int,
    n_iter: int,
    history,
) -> TrainedAEResult:
    w = _as_metric(weights)
    final_loss = weighted_reconstruction_loss(samples, components, w, mean)
    lstar, empca = empca_optimal_loss(samples, w, rank)
    gap = final_loss - lstar
    angle = float(np.max(principal_angles(empca.components, components, weights=w)))
    gram = D_data.T @ _m_left(D_data, w)
    ortho_err = float(np.max(np.abs(gram - np.eye(rank))))
    return TrainedAEResult(
        method=method,
        optimizer=optimizer,
        components=components,
        decoder=D_data,
        mean=mean,
        weights=w,
        final_loss=final_loss,
        optimal_loss=lstar,
        optimality_gap=gap,
        relative_gap=gap / max(abs(lstar), np.finfo(float).tiny),
        max_principal_angle_deg=angle,
        m_orthonormality_error=ortho_err,
        n_iter=n_iter,
        loss_history=np.asarray(history) if history is not None else None,
    )


def train_weighted_linear_ae(
    samples: np.ndarray,
    weights: np.ndarray,
    rank: int,
    *,
    optimizer: str = "lbfgs",
    center: bool = True,
    seed: int = 0,
    max_iter: int = 5000,
    adam_lr: float = 5e-2,
    adam_tol: float = 1e-14,
) -> TrainedAEResult:
    """Method (A): train the tied AE directly in data coordinates (weighted loss)."""
    X = np.asarray(samples, dtype=np.float64)
    w = _as_metric(weights)
    if X.shape[1] != w.shape[0]:
        raise ValueError("weights dimension must match feature dimension")
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mean[None, :]
    rng = np.random.default_rng(seed)
    D0 = _init_basis(Xc, w, int(rank), rng)

    if optimizer == "lbfgs":
        D, nit = _lbfgs(Xc, w, D0, max_iter)
        history = None
    elif optimizer == "adam":
        D, nit, history = _adam(Xc, w, D0, max_iter, adam_lr, adam_tol)
    else:
        raise ValueError("optimizer must be 'lbfgs' or 'adam'")

    return _finalise("direct", optimizer, D.T.copy(), D, mean, w, X, int(rank), nit, history)


def train_whitened_linear_ae(
    samples: np.ndarray,
    weights: np.ndarray,
    rank: int,
    *,
    optimizer: str = "lbfgs",
    center: bool = True,
    seed: int = 0,
    max_iter: int = 5000,
    adam_lr: float = 5e-2,
    adam_tol: float = 1e-14,
) -> TrainedAEResult:
    """Method (B): whiten, train a standard MSE tied AE, map the basis back.

    The optimisation core is reused with an identity metric on whitened data,
    so this is a genuine gradient-trained AE (not a closed form), but in the
    natural coordinates where Baldi-Hornik guarantees the PCA subspace.
    """
    X = np.asarray(samples, dtype=np.float64)
    w = _as_metric(weights)
    if X.shape[1] != w.shape[0]:
        raise ValueError("weights dimension must match feature dimension")
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mean[None, :]
    msqrt, msqrt_inv = _metric_sqrt(w)
    Xw = (Xc * msqrt[None, :]) if w.ndim == 1 else (Xc @ msqrt)
    ident = np.ones(Xw.shape[1], dtype=np.float64)  # identity metric in whitened space
    rng = np.random.default_rng(seed)
    W0 = _init_basis(Xw, ident, int(rank), rng)

    if optimizer == "lbfgs":
        W, nit = _lbfgs(Xw, ident, W0, max_iter)
        history = None
    elif optimizer == "adam":
        W, nit, history = _adam(Xw, ident, W0, max_iter, adam_lr, adam_tol)
    else:
        raise ValueError("optimizer must be 'lbfgs' or 'adam'")

    # Map basis columns from whitened space back to data units: D = M^{-1/2} W.
    D_data = (msqrt_inv[:, None] * W) if w.ndim == 1 else (msqrt_inv @ W)
    return _finalise("whitened", optimizer, D_data.T.copy(), D_data, mean, w, X, int(rank), nit, history)


def save_trained_ae(result: TrainedAEResult, path) -> str:
    """Persist a trained AE (basis + diagnostics) to a ``.npz`` file.

    Saves the decoder ``D``, the data-unit ``components``, the ``mean``, the
    ``weights`` metric, and all scalar diagnostics, so the model can be reloaded
    without retraining. Returns the written path as a string.
    """
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        method=result.method,
        optimizer=result.optimizer,
        decoder=result.decoder,
        components=result.components,
        mean=result.mean,
        weights=result.weights,
        final_loss=result.final_loss,
        optimal_loss=result.optimal_loss,
        optimality_gap=result.optimality_gap,
        relative_gap=result.relative_gap,
        max_principal_angle_deg=result.max_principal_angle_deg,
        m_orthonormality_error=result.m_orthonormality_error,
        n_iter=result.n_iter,
        loss_history=(result.loss_history if result.loss_history is not None else np.array([])),
    )
    return str(p if p.suffix else p.with_suffix(".npz"))


def load_trained_ae(path) -> dict:
    """Reload a saved AE as a plain dict of arrays/scalars."""
    data = np.load(path, allow_pickle=False)
    return {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}


def run_s3_bridge(
    samples: np.ndarray,
    weights: np.ndarray,
    rank: int,
    *,
    seed: int = 0,
    **kwargs,
) -> dict[str, TrainedAEResult]:
    """Convenience: run both methods with both optimisers and return all four."""
    return {
        "direct_lbfgs": train_weighted_linear_ae(samples, weights, rank, optimizer="lbfgs", seed=seed, **kwargs),
        "direct_adam": train_weighted_linear_ae(samples, weights, rank, optimizer="adam", seed=seed, **kwargs),
        "whitened_lbfgs": train_whitened_linear_ae(samples, weights, rank, optimizer="lbfgs", seed=seed, **kwargs),
        "whitened_adam": train_whitened_linear_ae(samples, weights, rank, optimizer="adam", seed=seed, **kwargs),
    }
