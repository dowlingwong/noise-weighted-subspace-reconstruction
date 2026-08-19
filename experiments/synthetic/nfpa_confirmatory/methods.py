"""Unified method interface: CW-PCA, NFPA, Iso-MPCA/IsoAE, IsoPCA, OF.

All methods are fitted at matched total rank ``k = k_c * k_t`` on the same
centered training observations (protocol section 6 and 7). Subspace bases are
always returned in the common oracle-whitened coordinate system so principal
angles are comparable.

Method -> objective summary:

- ``CW-PCA``:  PCA on whitened traces (``Sigma^{-1}`` metric, unrestricted).
- ``NFPA``:    ALS minimizing MSE in whitened space, restricted to factored
  ``(k_c, k_t)``.
- ``Iso-MPCA``: ALS minimizing raw MSE on centered measurement traces, same
  factored class. A tied linear autoencoder under the identity metric
  minimizes the same objective (``IsoAE``), so the two are one estimator with
  an alternative parameterization; ``fit_iso_ae_gradient`` documents the
  numerical equivalence for the audit.
- ``IsoPCA``: ordinary PCA on centered measurement traces, unrestricted.
- ``OF``:     fixed rank-one nominal template under the ``Sigma^{-1}`` metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noise_geometry.metrics import mse

from .config import PrimaryConfig
from .covariance import unwhiten_separable, whiten_separable
from .generator import Dataset, kron_basis


@dataclass
class MethodFit:
    name: str
    metric: str
    rep_class: str
    coord: str
    U_c: np.ndarray | None = None
    U_t: np.ndarray | None = None
    P: np.ndarray | None = None
    template_whitened: np.ndarray | None = None
    diag: dict = field(default_factory=dict)


def _whiten_batch(ds: Dataset, X: np.ndarray) -> np.ndarray:
    return whiten_separable(X - ds.train_mean, ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])


def _unwhiten_batch(ds: Dataset, Z: np.ndarray) -> np.ndarray:
    return unwhiten_separable(Z, ds.covariances["Sigma_c"], ds.covariances["Sigma_t"]) + ds.train_mean


def als_ut(Z: np.ndarray, U_c: np.ndarray, k_t: int) -> np.ndarray:
    D = np.einsum("nct,ck->nkt", Z, U_c)
    _, _, Vt = np.linalg.svd(D.reshape(Z.shape[0] * U_c.shape[1], Z.shape[2]), full_matrices=False)
    return Vt[:k_t].T


def als_uc(Z: np.ndarray, U_t: np.ndarray, k_c: int) -> np.ndarray:
    E = np.einsum("nct,tl->ncl", Z, U_t)
    _, _, Vt = np.linalg.svd(E.transpose(0, 2, 1).reshape(Z.shape[0] * U_t.shape[1], Z.shape[1]), full_matrices=False)
    return Vt[:k_c].T


def recon_kron(Z: np.ndarray, U_c: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    c = np.einsum("nct,ck,tl->nkl", Z, U_c, U_t)
    return np.einsum("nkl,ck,tl->nct", c, U_c, U_t)


def als_fit(
    Z: np.ndarray,
    k_c: int,
    k_t: int,
    *,
    n_iter: int = 150,
    n_restarts: int = 10,
    seed: int = 0,
    tol: float = 1e-10,
) -> dict:
    """Alternating least squares with restarts; returns the best fit diagnostics."""
    best = None
    for restart in range(int(n_restarts)):
        rng = np.random.default_rng(seed * 1000 + restart)
        U_c = np.linalg.qr(rng.standard_normal((Z.shape[1], k_c)))[0]
        U_t = np.linalg.qr(rng.standard_normal((Z.shape[2], k_t)))[0]
        hist = []
        for _ in range(int(n_iter)):
            U_t = als_ut(Z, U_c, k_t)
            U_c = als_uc(Z, U_t, k_c)
            hist.append(float(np.mean((Z - recon_kron(Z, U_c, U_t)) ** 2)))
        converged = len(hist) > 2 and abs(hist[-1] - hist[-2]) < tol
        if best is None or hist[-1] < best["loss"]:
            best = {
                "U_c": U_c,
                "U_t": U_t,
                "loss": hist[-1],
                "loss_history": hist,
                "restart": restart,
                "converged": converged,
            }
    return best


def fit_cw_pca(ds: Dataset) -> MethodFit:
    Z = _whiten_batch(ds, ds.X_train).reshape(ds.cfg.N_train, -1, order="F")
    rank = ds.cfg.k_c * ds.cfg.k_t
    _, _, vh = np.linalg.svd(Z, full_matrices=False)
    P = vh[:rank].T  # (CT, rank) whitened coordinates
    return MethodFit(
        name="CW-PCA",
        metric="Sigma^{-1}",
        rep_class="full",
        coord="whitened",
        P=P,
        diag={"explained_variance": float(np.sum(vh[:rank] ** 2))},
    )


def fit_nfpa(ds: Dataset) -> MethodFit:
    Z = _whiten_batch(ds, ds.X_train)
    out = als_fit(Z, ds.cfg.k_c, ds.cfg.k_t, n_iter=ds.cfg.n_als_iter, n_restarts=ds.cfg.n_restarts, seed=ds.seed, tol=ds.cfg.als_convergence_tol)
    return MethodFit(
        name="NFPA",
        metric="Sigma^{-1}",
        rep_class="factored",
        coord="whitened",
        U_c=out["U_c"],
        U_t=out["U_t"],
        diag={
            "train_loss": out["loss"],
            "selected_restart": out["restart"],
            "converged": out["converged"],
            "loss_history": out["loss_history"],
        },
    )


def fit_iso_mpca(ds: Dataset) -> MethodFit:
    Xc = ds.X_train - ds.train_mean
    out = als_fit(Xc, ds.cfg.k_c, ds.cfg.k_t, n_iter=ds.cfg.n_als_iter, n_restarts=ds.cfg.n_restarts, seed=ds.seed, tol=ds.cfg.als_convergence_tol)
    return MethodFit(
        name="Iso-MPCA",
        metric="I",
        rep_class="factored",
        coord="raw",
        U_c=out["U_c"],
        U_t=out["U_t"],
        diag={
            "train_loss": out["loss"],
            "selected_restart": out["restart"],
            "converged": out["converged"],
            "loss_history": out["loss_history"],
        },
    )


def fit_iso_pca(ds: Dataset) -> MethodFit:
    Xc = ds.X_train - ds.train_mean
    Z = Xc.reshape(ds.cfg.N_train, -1, order="F")
    rank = ds.cfg.k_c * ds.cfg.k_t
    _, _, vh = np.linalg.svd(Z, full_matrices=False)
    P_raw = vh[:rank].T  # (CT, rank) measurement coordinates
    return MethodFit(
        name="IsoPCA",
        metric="I",
        rep_class="full",
        coord="raw",
        P=P_raw,
        diag={"explained_variance": float(np.sum(vh[:rank] ** 2))},
    )


def fit_of(ds: Dataset) -> MethodFit:
    s0 = np.outer(ds.V[:, 0], ds.Tb[:, 0])
    s0_w = whiten_separable(s0[None, :, :], ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])[0]
    return MethodFit(
        name="OF",
        metric="Sigma^{-1}",
        rep_class="full",
        coord="whitened",
        template_whitened=s0_w,
        diag={},
    )


FIT_FACTORY = {
    "CW-PCA": fit_cw_pca,
    "NFPA": fit_nfpa,
    "Iso-MPCA": fit_iso_mpca,
    "IsoPCA": fit_iso_pca,
    "OF": fit_of,
}


def basis_whitened(fit: MethodFit, ds: Dataset) -> np.ndarray:
    """Orthonormal ``(CT, rank)`` basis of the fitted subspace in whitened coords.

    .. warning:: **ARCHIVAL — known defect in the raw factored branch.**
       ``B.reshape(rank, C, T, order="F")`` on a ``(C*T, rank)`` matrix does not
       unstack basis columns; it interleaves them. Only Iso-MPCA reaches that
       branch, so the archived E1/E9 ``angle_to_cw_pca_*`` columns for Iso-MPCA
       are invalid as subspace angles. This function is retained solely to
       reproduce those archived columns byte-for-byte. Every NEW computation
       must use :func:`nfpa_confirmatory.angles_v2.basis_whitened_v2`, which
       emits the corrected ``*_v2`` columns. Pinned by
       ``tests/test_round9.py`` (defect) and ``tests/test_round10.py``
       (correction). Do not "fix" this function: that would silently redefine
       an archived CSV column.
    """
    rank = ds.cfg.k_c * ds.cfg.k_t
    if fit.name == "OF":
        v = fit.template_whitened.reshape(-1, order="F")
        return (v / np.linalg.norm(v))[:, None]
    if fit.coord == "whitened":
        if fit.U_c is not None:
            return kron_basis(fit.U_c, fit.U_t)
        return np.linalg.qr(fit.P)[0]
    if fit.U_c is not None:
        B = kron_basis(fit.U_c, fit.U_t)  # (CT, rank) raw coords
        W = whiten_separable(B.reshape(rank, ds.cfg.C, ds.cfg.T, order="F"), ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])
        return np.linalg.qr(W.reshape(rank, -1, order="F").T)[0]
    W = whiten_separable(fit.P.T.reshape(rank, ds.cfg.C, ds.cfg.T, order="F"), ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])
    return np.linalg.qr(W.reshape(rank, -1, order="F").T)[0]


def reconstruct(fit: MethodFit, ds: Dataset, X: np.ndarray) -> np.ndarray:
    """Reconstruct ``(n, C, T)`` measurement traces from a fitted method."""
    if fit.name == "OF":
        s0 = np.outer(ds.V[:, 0], ds.Tb[:, 0])
        s0_w = fit.template_whitened.reshape(-1, order="F")
        Z = _whiten_batch(ds, X)
        amps = (Z.reshape(X.shape[0], -1, order="F") @ s0_w) / float(s0_w @ s0_w)
        return amps[:, None, None] * s0[None, :, :]
    if fit.coord == "whitened":
        Z = _whiten_batch(ds, X)
        if fit.U_c is not None:
            Zhat = recon_kron(Z, fit.U_c, fit.U_t)
        else:
            flat = Z.reshape(X.shape[0], -1, order="F") @ (fit.P @ fit.P.T)
            Zhat = flat.reshape(X.shape, order="F")
        return _unwhiten_batch(ds, Zhat)
    Xc = X - ds.train_mean
    if fit.U_c is not None:
        Xhat = recon_kron(Xc, fit.U_c, fit.U_t)
    else:
        flat = Xc.reshape(X.shape[0], -1, order="F") @ (fit.P @ fit.P.T)
        Xhat = flat.reshape(X.shape, order="F")
    return Xhat + ds.train_mean


def fit_iso_ae_gradient(ds: Dataset, *, n_iter: int = 2000, lr: float = 1e-2, seed: int = 0) -> MethodFit:
    """Gradient-trained tied linear autoencoder (raw MSE) for the IsoAE check.

    Minimizes ``E ||X - U_c U_c^T X U_t U_t^T||^2`` by projected gradient
    descent on the same factored parameterization as Iso-MPCA, re-orthonormal-
    izing each step. The reconstruction is the symmetric idempotent projector
    ``P = U_t U_t^T (x) U_c U_c^T`` on ``vec(X)``, so the objective gradient
    admits the closed forms ``dL/dU_c = -(2/N) sum_i X_i U_t U_t^T X_i^T`` and
    ``dL/dU_t = -(2/N) sum_i X_i^T U_c U_c^T X_i``. Used only to document that
    IsoAE and Iso-MPCA are numerically the same estimator under the tied
    linear architecture (protocol section 6).
    """
    rng = np.random.default_rng(seed)
    C, T = ds.cfg.C, ds.cfg.T
    k_c, k_t = ds.cfg.k_c, ds.cfg.k_t
    Xc = ds.X_train - ds.train_mean
    U_c = np.linalg.qr(rng.standard_normal((C, k_c)))[0]
    U_t = np.linalg.qr(rng.standard_normal((T, k_t)))[0]
    prev = np.inf
    for _ in range(int(n_iter)):
        M_c = np.einsum("nct,ts,ncs->ac", Xc, U_t @ U_t.T, Xc)
        g_c = -2.0 * M_c / Xc.shape[0]
        M_t = np.einsum("nat,ab,nbs->ts", Xc, U_c @ U_c.T, Xc)
        g_t = -2.0 * M_t / Xc.shape[0]
        U_c = U_c - lr * g_c
        U_t = U_t - lr * g_t
        U_c, _ = np.linalg.qr(U_c)
        U_t, _ = np.linalg.qr(U_t)
        loss = float(np.mean((recon_kron(Xc, U_c, U_t) - Xc) ** 2))
        if abs(prev - loss) < 1e-12:
            break
        prev = loss
    return MethodFit(
        name="IsoAE",
        metric="I",
        rep_class="factored",
        coord="raw",
        U_c=U_c,
        U_t=U_t,
        diag={"train_loss": float(np.mean((recon_kron(Xc, U_c, U_t) - Xc) ** 2))},
    )
