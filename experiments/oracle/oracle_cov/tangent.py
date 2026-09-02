"""Tangent geometry of the forward model (ORACLE proposal §3).

* ``excited_span``   — orthonormal basis of  T_S = span{ Sigma^{-1/2} dg/dz_j : j in S }
                       over the training support (stacked Jacobians, SVD);
* ``unexcited_span`` — the same for the latents the training never varied,
                       orthogonalised against T_S (so the two are disjoint);
* ``fisher``         — I(z) = J^T Sigma^{-1} J at a latent point and its
                       numerical rank;
* ``energy_split``   — for a displacement vector in whitened space, the
                       fractions of its energy inside T_S, inside T_U and in
                       the remaining complement.

All vectors are column-major ``vec`` of whitened ``(C, T)`` matrices, the
same convention as the NFPA harness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.synthetic.nfpa_confirmatory.covariance import whiten_separable

from .generator import LATENT_NAMES, TrainingSet


def _whiten_cols(J: np.ndarray, ts: TrainingSet) -> np.ndarray:
    """Whiten each Jacobian column ``(C*T,)`` with the *true* covariance."""
    n, d, m = J.shape
    M = J.transpose(0, 2, 1).reshape(n * m, ts.cfg.C, ts.cfg.T, order="F")
    W = whiten_separable(M, ts.Sigma_c, ts.Sigma_t)
    return W.reshape(n, m, d, order="F").transpose(0, 2, 1)


@dataclass
class TangentGeometry:
    excited: tuple[str, ...]
    unexcited: tuple[str, ...]
    T_S: np.ndarray        # (d, r_S) orthonormal
    T_U: np.ndarray        # (d, r_U) orthonormal, orthogonal to T_S
    sv_S: np.ndarray
    sv_U: np.ndarray

    def energy_split(self, delta: np.ndarray) -> dict[str, float]:
        e = float(delta @ delta)
        eS = float(np.sum((delta @ self.T_S) ** 2))
        eU = float(np.sum((delta @ self.T_U) ** 2))
        return {"in_T_S": eS / max(e, 1e-300), "in_T_U": eU / max(e, 1e-300), "elsewhere": max(e - eS - eU, 0.0) / max(e, 1e-300)}


def _span(Jw: np.ndarray, rel_tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n, d, m = Jw.shape
    A = Jw.transpose(1, 0, 2).reshape(d, n * m)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(s > rel_tol * s[0]))
    return U[:, :r], s[:r]


def tangent_geometry(ts: TrainingSet, excited: tuple[str, ...], *, n_points: int = 64, seed: int = 0) -> TangentGeometry:
    """Excited / unexcited spans from Jacobians at ``n_points`` training latents."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ts.ds.latents_train["amplitude"]), size=min(n_points, len(ts.ds.latents_train["amplitude"])), replace=False)
    lat = {k: v[idx] for k, v in ts.ds.latents_train.items()}
    lat["t0_shift"] = np.zeros(len(idx))
    unexcited = tuple(n for n in LATENT_NAMES if n not in excited)
    J = ts.forward.jacobian(lat, names=LATENT_NAMES)
    Jw = _whiten_cols(J, ts)
    iS = [LATENT_NAMES.index(n) for n in excited]
    iU = [LATENT_NAMES.index(n) for n in unexcited]
    T_S, sS = _span(Jw[:, :, iS])
    if iU:
        JU = Jw[:, :, iU]
        JU = JU - np.einsum("dk,kn->dn", T_S, np.einsum("dk,ndm->knm", T_S, JU).reshape(T_S.shape[1], -1)).reshape(JU.shape[1], JU.shape[0], JU.shape[2]).transpose(1, 0, 2)
        T_U, sU = _span(JU)
    else:
        T_U, sU = np.zeros((Jw.shape[1], 0)), np.zeros(0)
    return TangentGeometry(excited, unexcited, T_S, T_U, sS, sU)


def fisher(ts: TrainingSet, latents: dict[str, np.ndarray], names=LATENT_NAMES) -> tuple[np.ndarray, np.ndarray]:
    """Per-event Fisher matrices ``(n, m, m)`` and their numerical ranks."""
    J = ts.forward.jacobian(latents, names=names)
    Jw = _whiten_cols(J, ts)
    I = np.einsum("ndi,ndj->nij", Jw, Jw)
    ranks = np.array([np.linalg.matrix_rank(M, tol=1e-8 * np.linalg.norm(M)) for M in I])
    return I, ranks
