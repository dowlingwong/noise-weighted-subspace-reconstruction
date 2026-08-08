"""Semantic utility (Block H): linear decoders and planted-subspace errors.

A linear map is calibrated from latent codes to the physical parameters
(amplitude, rise-time, decay-time) on the training split only and evaluated on
held-out test events against known truth. Channel- and temporal-subspace errors
compare projectors, not basis vectors, because rotations and signs are not
identifiable (protocol sections 8 Block H and 9).
"""

from __future__ import annotations

import numpy as np

from .generator import Dataset, kron_basis
from .methods import MethodFit


def latent_codes(fit: MethodFit, ds: Dataset, X: np.ndarray) -> np.ndarray:
    """``(n, k_c * k_t)`` codes in the method's coordinate space."""
    if fit.name == "OF":
        s0_w = fit.template_whitened.reshape(-1, order="F")
        Z = X
        from .covariance import whiten_separable

        Z = whiten_separable(X - ds.train_mean, ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])
        flat = Z.reshape(X.shape[0], -1, order="F")
        return (flat @ s0_w)[:, None]
    if fit.coord == "whitened":
        from .covariance import whiten_separable

        Z = whiten_separable(X - ds.train_mean, ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])
    else:
        Z = X - ds.train_mean
    code = np.einsum("nct,ck,tl->nkl", Z, fit.U_c, fit.U_t)
    return code.reshape(X.shape[0], -1)


def _decoder_inputs(fit: MethodFit, ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_codes = latent_codes(fit, ds, ds.X_train)
    test_codes = latent_codes(fit, ds, ds.X_test)
    lat = ds.latents_train
    params = np.column_stack([lat["amplitude"], lat["rise_time"], lat["decay_time"]])
    return train_codes, test_codes, params


def fit_linear_decoder(train_codes: np.ndarray, params: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Ridge least-squares map ``codes -> params`` (train split only)."""
    G = train_codes.T @ train_codes + ridge * np.eye(train_codes.shape[1])
    return np.linalg.solve(G, train_codes.T @ params)


def predict_params(fit: MethodFit, ds: Dataset, W: np.ndarray) -> np.ndarray:
    test_codes = latent_codes(fit, ds, ds.X_test)
    return test_codes @ W


def amplitude_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    err = estimate - truth
    return {
        "amplitude_bias": float(np.mean(err)),
        "amplitude_rmse": float(np.sqrt(np.mean(err**2))),
        "amplitude_correlation": float(np.corrcoef(estimate, truth)[0, 1]),
    }


def _projector_error(P_a: np.ndarray, P_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(P_b, ord="fro"))
    return float(np.linalg.norm(P_a - P_b, ord="fro") / max(denom, 1e-12))


def planted_subspace_errors(fit: MethodFit, ds: Dataset) -> dict[str, float]:
    """Normalized Frobenius projector error vs the planted factor subspaces.

    Targets are expressed in the method's coordinate space: for covariance-
    aware methods the planted channel subspace is ``span(Sigma_c^{-1/2} V)``
    (the whitening mixes channels), for identity-metric methods it is ``V``.
    """
    from .covariance import _sqrt_inv

    V, Tb = ds.V, ds.Tb
    U_c, U_t = fit.U_c, fit.U_t
    if fit.coord == "whitened":
        V_t = np.linalg.qr(_sqrt_inv(ds.covariances["Sigma_c"]) @ V)[0]
        T_t = np.linalg.qr(_sqrt_inv(ds.covariances["Sigma_t"]) @ Tb)[0]
    else:
        V_t, T_t = V, Tb
    P_uc = U_c @ U_c.T
    P_vt = V_t @ V_t.T
    P_ut = U_t @ U_t.T
    P_tt = T_t @ T_t.T
    return {
        "channel_projector_error": _projector_error(P_uc, P_vt),
        "temporal_projector_error": _projector_error(P_ut, P_tt),
    }


def evaluate_semantics(fit: MethodFit, ds: Dataset) -> dict[str, float]:
    """Fit a linear decoder on train codes, evaluate on held-out test truth."""
    train_codes, test_codes, params = _decoder_inputs(fit, ds)
    W = fit_linear_decoder(train_codes, params)
    pred = predict_params(fit, ds, W)
    truth = np.column_stack([ds.latents_test["amplitude"], ds.latents_test["rise_time"], ds.latents_test["decay_time"]])
    out = amplitude_metrics(pred[:, 0], truth[:, 0])
    out["rise_time_rmse"] = float(np.sqrt(np.mean((pred[:, 1] - truth[:, 1]) ** 2)))
    out["decay_time_rmse"] = float(np.sqrt(np.mean((pred[:, 2] - truth[:, 2]) ** 2)))
    out.update(planted_subspace_errors(fit, ds))
    return out
