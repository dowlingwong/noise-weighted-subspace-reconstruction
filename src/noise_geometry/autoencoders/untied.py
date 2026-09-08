"""Untied linear autoencoders — the ablation the tied-adjoint claim needs.

``trained.py`` optimises a *tied* linear autoencoder: one basis ``D`` serves as
decoder and, through the metric, as encoder (``E = D^T M``). The manuscript's
§5.4 / Rem. E.3 claim rests on that tie: the tied stationary point is the
``M``-orthogonal projection onto ``range(D)``. This module drops the tie and
optimises encoder ``E`` (k x d) and decoder ``D`` (d x k) as independent
parameters under

    L(D, E) = (1/N) sum_n r_n^T M r_n ,   r_n = x_n - D E x_n ,

with ``M = Sigma^{-1}`` (``loss="weighted"``) or ``M = I`` (``loss="mse"``),
plus an optional ridge ``l2 * (||D||_F^2 + ||E||_F^2)``.

Reported per fit:

* ``decoder`` D, ``encoder`` E and the induced projector ``Pi = D E``;
* ``obliqueness`` ``||Pi - Pi*_Sigma||_F / ||Pi||_F`` where
  ``Pi*_Sigma = D (D^T Sigma^{-1} D)^{-1} D^T Sigma^{-1}`` is the
  ``Sigma``-orthogonal projector onto ``range(D)`` — zero exactly when the
  learned map is the metric-orthogonal projection its tied counterpart would
  give;
* ``principal_angles_deg`` of ``range(D)`` against the EMPCA subspace.

The metric may be a diagonal weight vector, a full SPD matrix, or — for the
vectorised multichannel regime, where the full ``d x d`` matrix is wasteful —
a :class:`SeparableMetric` carrying the two Kronecker factors of
``Sigma^{-1} = kron(Sigma_t^{-1}, Sigma_c^{-1})`` in column-major ``vec``
convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..subspace import fit_weighted_pca, principal_angles


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeparableMetric:
    """``M = kron(B, A)`` acting on column-major ``vec`` of ``(C, T)`` matrices.

    ``A`` is the channel factor (``C x C``), ``B`` the temporal factor
    (``T x T``); ``M x = vec(A X B)`` for ``X`` the ``(C, T)`` unstacking of
    ``x``. Used so the ``d = C*T`` regime never materialises ``M`` itself.
    """

    A: np.ndarray
    B: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        d = self.A.shape[0] * self.B.shape[0]
        return (d, d)

    def dense(self) -> np.ndarray:
        return np.kron(self.B, self.A)


def _apply_metric(R: np.ndarray, w) -> np.ndarray:
    """Rows of ``R`` are vectors ``r``; return rows ``(M r)^T``."""
    if isinstance(w, SeparableMetric):
        n = R.shape[0]
        C, T = w.A.shape[0], w.B.shape[0]
        M = R.reshape(n, C, T, order="F")
        out = np.einsum("ab,nbt,ts->nas", w.A, M, w.B, optimize=True)
        return out.reshape(n, C * T, order="F")
    return R * w[None, :] if w.ndim == 1 else R @ w


def _metric_dim(w) -> int:
    if isinstance(w, SeparableMetric):
        return w.shape[0]
    return w.shape[0]


def _dense_metric(w) -> np.ndarray:
    if isinstance(w, SeparableMetric):
        return w.dense()
    if w.ndim == 1:
        return np.diag(w)
    return w


def _sym_sqrt(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    vals = np.clip(vals, np.finfo(float).eps, None)
    return (vecs * np.sqrt(vals)[None, :]) @ vecs.T


def _whiten_rows(B: np.ndarray, w) -> np.ndarray:
    """Rows of ``B`` are vectors; return rows ``(M^{1/2} b)^T``.

    For a :class:`SeparableMetric` this uses the Kronecker factors, so the
    ``d x d`` metric is never formed.
    """
    if isinstance(w, SeparableMetric):
        n = B.shape[0]
        C, T = w.A.shape[0], w.B.shape[0]
        Asq, Bsq = _sym_sqrt(w.A), _sym_sqrt(w.B)
        M = B.reshape(n, C, T, order="F")
        out = np.einsum("ab,nbt,ts->nas", Asq, M, Bsq, optimize=True)
        return out.reshape(n, C * T, order="F")
    if w.ndim == 1:
        return B * np.sqrt(np.clip(w, 0.0, None))[None, :]
    return B @ _sym_sqrt(w)


# --------------------------------------------------------------------------- #
# Loss and gradient
# --------------------------------------------------------------------------- #
def _loss_and_grad(D: np.ndarray, E: np.ndarray, X: np.ndarray, w, l2: float):
    """Untied weighted reconstruction loss and gradients w.r.t. ``D`` and ``E``."""
    n = X.shape[0]
    Z = X @ E.T                      # (n, k) codes
    Xhat = Z @ D.T
    R = X - Xhat
    MR = _apply_metric(R, w)
    loss = float(np.sum(MR * R) / n)
    gD = (-2.0 / n) * (MR.T @ Z)                 # (d, k)
    gE = (-2.0 / n) * (D.T @ MR.T @ X)           # (k, d)
    if l2:
        loss += float(l2) * (float(np.sum(D * D)) + float(np.sum(E * E)))
        gD = gD + 2.0 * float(l2) * D
        gE = gE + 2.0 * float(l2) * E
    return loss, gD, gE


def _pack(D, E):
    return np.concatenate([D.ravel(), E.ravel()])


def _unpack(vec, d, k):
    return vec[: d * k].reshape(d, k), vec[d * k:].reshape(k, d)


def _lbfgs(X, w, D0, E0, max_iter, l2, ftol, gtol):
    d, k = D0.shape

    def fun(vec):
        D, E = _unpack(vec, d, k)
        loss, gD, gE = _loss_and_grad(D, E, X, w, l2)
        return loss, _pack(gD, gE)

    res = minimize(fun, _pack(D0, E0), method="L-BFGS-B", jac=True,
                   options={"maxiter": int(max_iter), "maxfun": int(max_iter) * 10,
                            "ftol": ftol, "gtol": gtol})
    D, E = _unpack(res.x, d, k)
    return D, E, int(res.nit), bool(res.success), str(res.message)


def _adam(X, w, D0, E0, max_iter, l2, lr, tol):
    D, E = D0.copy(), E0.copy()
    mD = np.zeros_like(D); vD = np.zeros_like(D)
    mE = np.zeros_like(E); vE = np.zeros_like(E)
    b1, b2, eps = 0.9, 0.999, 1e-12
    prev = np.inf
    it = 0
    for it in range(1, int(max_iter) + 1):
        loss, gD, gE = _loss_and_grad(D, E, X, w, l2)
        for p, g, m, v in ((D, gD, mD, vD), (E, gE, mE, vE)):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * (g * g)
            mhat = m / (1 - b1 ** it)
            vhat = v / (1 - b2 ** it)
            p -= lr * mhat / (np.sqrt(vhat) + eps)
        if abs(prev - loss) < tol:
            break
        prev = loss
    return D, E, it, True, "adam"


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class UntiedAEResult:
    loss: str                          # "weighted" or "mse"
    optimizer: str
    decoder: np.ndarray                # D (d, k)
    encoder: np.ndarray                # E (k, d)
    projector: np.ndarray              # Pi = D E (d, d)
    mean: np.ndarray
    final_loss: float
    obliqueness: float                 # ||Pi - Pi*_Sigma||_F / ||Pi||_F
    principal_angles_deg: np.ndarray   # range(D) vs the EMPCA subspace
    max_principal_angle_deg: float
    n_iter: int
    converged: bool
    message: str = ""
    loss_history: np.ndarray = field(default=None, repr=False)


def sigma_orthogonal_projector(D: np.ndarray, sigma_inv) -> np.ndarray:
    """``Pi*_Sigma = D (D^T Sigma^{-1} D)^{-1} D^T Sigma^{-1}``."""
    MD = _apply_metric(D.T, sigma_inv).T          # (d, k) = Sigma^{-1} D
    G = D.T @ MD
    return D @ np.linalg.solve(G, MD.T)


def obliqueness(D: np.ndarray, E: np.ndarray, sigma_inv) -> float:
    """Relative Frobenius distance of ``Pi = D E`` from the ``Sigma``-orthogonal one."""
    Pi = D @ E
    Pi_star = sigma_orthogonal_projector(D, sigma_inv)
    denom = float(np.linalg.norm(Pi, "fro"))
    return float(np.linalg.norm(Pi - Pi_star, "fro") / max(denom, 1e-300))


def _init(Xc, rank, rng, w):
    """Initialise ``D`` from a random ``M``-orthonormalised basis; ``E = D^T M``.

    Starting at the tied configuration makes the tied and untied arms
    comparable: any divergence between them is produced by the optimisation,
    not by a different starting point.
    """
    d = Xc.shape[1]
    D0 = rng.standard_normal((d, int(rank)))
    MD = _apply_metric(D0.T, w).T
    G = D0.T @ MD
    # M-orthonormalise: D <- D G^{-1/2}
    vals, vecs = np.linalg.eigh(0.5 * (G + G.T))
    vals = np.clip(vals, np.finfo(float).eps, None)
    D0 = D0 @ (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
    E0 = _apply_metric(D0.T, w)
    return D0, E0


def train_untied_linear_ae(
    samples: np.ndarray,
    weights,
    rank: int,
    *,
    loss: str = "weighted",
    optimizer: str = "lbfgs",
    center: bool = True,
    seed: int = 0,
    max_iter: int = 5000,
    l2: float = 0.0,
    ftol: float = 1e-16,
    gtol: float = 1e-12,
    adam_lr: float = 5e-2,
    adam_tol: float = 1e-14,
    reference_weights=None,
    reference_components: np.ndarray | None = None,
) -> UntiedAEResult:
    """Train an untied linear autoencoder.

    Parameters
    ----------
    samples : (N, d) observations.
    weights : the inverse-noise metric ``Sigma^{-1}`` — a diagonal weight
        vector, a full SPD matrix, or a :class:`SeparableMetric`.
    rank : latent dimension k.
    loss : ``"weighted"`` optimises under ``M = Sigma^{-1}``; ``"mse"``
        optimises under ``M = I``. Both are *evaluated* against the same
        ``Sigma``-based diagnostics (obliqueness, angle to EMPCA).
    optimizer : ``"lbfgs"`` (full-batch, float64) or ``"adam"``.
    l2 : ridge on both factors; 0 by default.
    reference_weights : metric defining the EMPCA reference subspace; defaults
        to ``weights`` (i.e. always the ``Sigma^{-1}`` EMPCA subspace, even for
        the MSE arms).
    reference_components : ``(k, d)`` EMPCA basis, if it has already been
        computed for this dataset. Supplying it avoids refitting EMPCA — and,
        in the separable multichannel regime, avoids ever forming the dense
        ``d x d`` metric.
    """
    X = np.asarray(samples, dtype=np.float64)
    w_sigma = weights if isinstance(weights, SeparableMetric) else np.asarray(weights, dtype=np.float64)
    d = X.shape[1]
    if _metric_dim(w_sigma) != d:
        raise ValueError("weights dimension must match feature dimension")
    if loss not in ("weighted", "mse"):
        raise ValueError("loss must be 'weighted' or 'mse'")
    w_train = w_sigma if loss == "weighted" else np.ones(d, dtype=np.float64)

    mean = X.mean(axis=0) if center else np.zeros(d, dtype=np.float64)
    Xc = X - mean[None, :]
    rng = np.random.default_rng(seed)
    D0, E0 = _init(Xc, rank, rng, w_train)

    if optimizer == "lbfgs":
        D, E, nit, ok, msg = _lbfgs(Xc, w_train, D0, E0, max_iter, l2, ftol, gtol)
        history = None
    elif optimizer == "adam":
        D, E, nit, ok, msg = _adam(Xc, w_train, D0, E0, max_iter, l2, adam_lr, adam_tol)
        history = None
    else:
        raise ValueError("optimizer must be 'lbfgs' or 'adam'")

    final_loss, _, _ = _loss_and_grad(D, E, Xc, w_train, l2)
    ref_w = w_sigma if reference_weights is None else reference_weights
    if reference_components is None:
        empca_components = fit_weighted_pca(X, _dense_metric(ref_w), int(rank)).components
    else:
        empca_components = np.asarray(reference_components, dtype=np.float64)
    # Angles in the metric are Euclidean angles between the metric-whitened
    # bases; whitening explicitly keeps the separable regime out of dense d x d.
    angles = np.asarray(principal_angles(_whiten_rows(D.T, ref_w),
                                         _whiten_rows(empca_components, ref_w),
                                         degrees=True))

    return UntiedAEResult(
        loss=loss,
        optimizer=optimizer,
        decoder=D,
        encoder=E,
        projector=D @ E,
        mean=mean,
        final_loss=float(final_loss),
        obliqueness=obliqueness(D, E, w_sigma),
        principal_angles_deg=angles,
        max_principal_angle_deg=float(np.max(angles)),
        n_iter=nit,
        converged=bool(ok),
        message=msg,
        loss_history=history,
    )
