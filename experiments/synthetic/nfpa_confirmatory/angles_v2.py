"""Corrected whitened-basis construction and principal angles (Round 10, v2).

The defect being corrected
--------------------------
``methods.basis_whitened`` unstacks a ``(C*T, rank)`` basis matrix in its
raw-coordinates factored branch with ``B.reshape(rank, C, T, order="F")``.
That expression does not unstack the *columns* of ``B``: it refills the whole
buffer, interleaving entries of different basis vectors, so the whitened
subspace it produces is not the fitted subspace. The correct unstacking of
column ``i`` under the repository's column-major ``vec`` convention is
``B[:, i].reshape(C, T, order="F")``, which for the whole matrix is
``B.T.reshape(rank, C, T, order="F")`` (the shared leading axis keeps rows
intact while the trailing ``C*T`` axis is F-unflattened).

Only Iso-MPCA reaches the defective branch (``coord == "raw"`` with factored
``U_c``); NFPA and CW-PCA are fitted in whitened coordinates and OF has its own
branch, so their legacy angles are valid. The defect feeds only the
``angle_to_cw_pca_*`` diagnostic columns; no reconstruction metric touches it.

Versioning contract (ROUND10_DESK_INTEGRATION_RUNBOOK.md, Phase 2)
------------------------------------------------------------------
* This module is the canonical path for every NEW angle computation.
* ``methods.basis_whitened`` is retained, annotated, as the ARCHIVAL helper:
  it reproduces the archived E1/E9 ``angle_to_cw_pca_*`` columns and must not
  be used in new scientific summaries.
* Corrected outputs use new column names, ``angle_to_cw_pca_mean_deg_v2`` and
  ``angle_to_cw_pca_max_deg_v2``; the archived column names are never
  redefined.
"""

from __future__ import annotations

import numpy as np

from noise_geometry.subspace import principal_angles

from .covariance import whiten_separable
from .generator import Dataset, kron_basis
from .methods import MethodFit, basis_whitened as basis_whitened_legacy


def unstack_basis_columns(B: np.ndarray, C: int, T: int) -> np.ndarray:
    """``(C*T, rank)`` column-major stacked bases -> ``(rank, C, T)`` traces.

    Row ``i`` of the result is ``vec^{-1}(B[:, i])`` under the column-major
    convention: ``B[:, i].reshape(C, T, order="F")``.
    """
    B = np.asarray(B)
    if B.ndim != 2 or B.shape[0] != C * T:
        raise ValueError(f"expected a (C*T, rank) = ({C * T}, k) matrix, got {B.shape}")
    rank = B.shape[1]
    return B.T.reshape(rank, C, T, order="F")


def whiten_basis_columns(
    B: np.ndarray, Sigma_c: np.ndarray, Sigma_t: np.ndarray
) -> np.ndarray:
    """Whiten each column of a ``(C*T, rank)`` basis; return an orthonormal basis.

    This is the canonical helper required by the Round 10 runbook: each basis
    column is unstacked column-major, whitened with the separable factors, and
    re-stacked; the result is orthonormalized by QR. Equivalent, by
    construction, to whitening the columns one at a time.
    """
    C, T = Sigma_c.shape[0], Sigma_t.shape[0]
    traces = unstack_basis_columns(B, C, T)
    W = whiten_separable(traces, Sigma_c, Sigma_t)
    return np.linalg.qr(W.reshape(W.shape[0], -1, order="F").T)[0]


def basis_whitened_v2(fit: MethodFit, ds: Dataset) -> np.ndarray:
    """Corrected orthonormal ``(CT, rank)`` whitened basis of a fitted subspace.

    Identical to the legacy helper for every branch that was already valid
    (OF; whitened-coordinate fits; the raw ``P`` branch, whose ``P.T`` reshape
    is a correct row-wise unstacking). The raw factored branch — the one the
    defect corrupted — goes through :func:`whiten_basis_columns`.
    """
    if fit.name == "OF" or fit.coord == "whitened":
        return basis_whitened_legacy(fit, ds)
    Sc, St = ds.covariances["Sigma_c"], ds.covariances["Sigma_t"]
    if fit.U_c is not None:
        return whiten_basis_columns(kron_basis(fit.U_c, fit.U_t), Sc, St)
    return whiten_basis_columns(fit.P, Sc, St)


def subspace_angles_to_v2(fit: MethodFit, reference: MethodFit, ds: Dataset) -> np.ndarray:
    """Principal angles (degrees) between corrected whitened subspaces."""
    A = basis_whitened_v2(fit, ds)
    B = basis_whitened_v2(reference, ds)
    return np.asarray(principal_angles(A.T, B.T, degrees=True))


def angle_metrics_v2(fit: MethodFit, cw_pca: MethodFit, ds: Dataset) -> dict[str, float]:
    """The corrected angle columns, under their versioned ``_v2`` names."""
    angles = subspace_angles_to_v2(fit, cw_pca, ds)
    return {
        "angle_to_cw_pca_mean_deg_v2": float(np.mean(angles)),
        "angle_to_cw_pca_max_deg_v2": float(np.max(angles)),
    }
