"""Method-aware semantic utilities for Round 9 / E8.

The superseded ``semantic.py`` assumed every method exposed factored
``(U_c, U_t)`` axes and decoded with an unstandardized ridge solve. Both
assumptions are wrong for the registered method set: ``CW-PCA`` and ``IsoPCA``
fit an unrestricted subspace with no identified channel or temporal axis, and
``OF`` is a fixed rank-one template with a single amplitude coordinate. This
module replaces those utilities with per-method definitions and *explicit
nulls*: a quantity that a method does not identify is reported as ``None``,
never as zero.

Two things are kept strictly separate here, because they have different
evidential status (see the Round 9 runbook):

* the **planted-factor projector errors**, which are the delayed execution of
  the predeclared hypothesis H3 and are only meaningful in the separable
  (``alpha = 0``) condition; and
* the **parameter decoding**, which is a post-hoc completion of the dropped
  Block H at the archived ``alpha = 0.1`` condition.

The decoder rule is frozen before any run: standardize each code column with
training means and nonzero training standard deviations, append an intercept,
solve with :func:`numpy.linalg.lstsq` at ``rcond=None``, and return predictions
in the original target units. With at most six code dimensions and 800 training
events, no test-tuned regularizer is used or needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .covariance import _sqrt_inv, whiten_separable
from .generator import Dataset, kron_basis
from .methods import MethodFit

#: Physical latents decoded from the codes, in this fixed column order.
DECODE_TARGETS = ("amplitude", "rise_time", "decay_time")

#: Methods whose factored axes identify separate channel/temporal subspaces.
FACTORED_METHODS = ("NFPA", "Iso-MPCA")

#: Methods that fit an unrestricted subspace: joint projector only.
UNRESTRICTED_METHODS = ("CW-PCA", "IsoPCA")


# --------------------------------------------------------------------------
# Codes
# --------------------------------------------------------------------------


def method_codes(fit: MethodFit, ds: Dataset, X: np.ndarray) -> np.ndarray:
    """Return the ``(n, d)`` latent codes a method assigns to events ``X``.

    ``d`` is ``k_c * k_t`` for the subspace methods and 1 for ``OF``, whose only
    coordinate is the matched-filter amplitude.
    """
    n = X.shape[0]
    if fit.coord == "whitened":
        Z = whiten_separable(X - ds.train_mean, ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])
    else:
        Z = X - ds.train_mean

    if fit.name == "OF":
        s0_w = fit.template_whitened.reshape(-1, order="F")
        flat = Z.reshape(n, -1, order="F")
        return (flat @ s0_w / float(s0_w @ s0_w))[:, None]

    if fit.U_c is not None:
        code = np.einsum("nct,ck,tl->nkl", Z, fit.U_c, fit.U_t)
        return code.reshape(n, -1)

    # Unrestricted subspace: coordinates in the fitted basis.
    return Z.reshape(n, -1, order="F") @ fit.P


def decode_targets_for(method: str) -> tuple[str, ...]:
    """``OF`` decodes amplitude only; every other method decodes all three."""
    return ("amplitude",) if method == "OF" else DECODE_TARGETS


# --------------------------------------------------------------------------
# The frozen affine decoder
# --------------------------------------------------------------------------


@dataclass
class AffineDecoder:
    """A frozen affine ordinary-least-squares decoder, fitted on train only."""

    train_mean: np.ndarray
    train_scale: np.ndarray
    zero_variance_columns: list[int]
    coefficients: np.ndarray
    intercept: np.ndarray
    rank: int
    singular_values: np.ndarray
    targets: tuple[str, ...]

    def predict(self, codes: np.ndarray) -> np.ndarray:
        Zs = (np.asarray(codes, dtype=float) - self.train_mean) / self.train_scale
        return Zs @ self.coefficients + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "train_code_means": [float(v) for v in self.train_mean],
            "train_code_scales": [float(v) for v in self.train_scale],
            "zero_variance_columns": list(self.zero_variance_columns),
            "coefficients": [[float(v) for v in row] for row in self.coefficients],
            "intercept": [float(v) for v in self.intercept],
            "numerical_rank": int(self.rank),
            "singular_values": [float(v) for v in self.singular_values],
            "solver": "numpy.linalg.lstsq(rcond=None)",
            "standardization": "per-column train mean and nonzero train std; intercept fitted",
        }


def fit_affine_decoder(
    train_codes: np.ndarray, train_targets: np.ndarray, targets: tuple[str, ...]
) -> AffineDecoder:
    """Fit the frozen decoder rule on the training split only.

    A code column with zero training variance receives scale one and is flagged;
    it contributes nothing but is not silently dropped, so the reported rank
    tells the reader how many directions were actually usable.
    """
    Z = np.asarray(train_codes, dtype=float)
    Y = np.asarray(train_targets, dtype=float)
    if Z.shape[0] != Y.shape[0]:
        raise ValueError("codes and targets must share the number of training events")

    mean = Z.mean(axis=0)
    std = Z.std(axis=0, ddof=0)
    zero_cols = [int(i) for i in np.flatnonzero(std <= 0.0)]
    scale = np.where(std > 0.0, std, 1.0)
    Zs = (Z - mean) / scale

    design = np.concatenate([Zs, np.ones((Zs.shape[0], 1))], axis=1)
    solution, _, rank, singular = np.linalg.lstsq(design, Y, rcond=None)
    return AffineDecoder(
        train_mean=mean,
        train_scale=scale,
        zero_variance_columns=zero_cols,
        coefficients=solution[:-1],
        intercept=solution[-1],
        rank=int(rank),
        singular_values=np.asarray(singular, dtype=float),
        targets=targets,
    )


def stack_targets(latents: dict[str, np.ndarray], targets: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([latents[name] for name in targets])


# --------------------------------------------------------------------------
# Planted-subspace errors
# --------------------------------------------------------------------------


def _projector_error(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Normalized Frobenius distance between the two orthogonal projectors."""
    P_a = basis_a @ basis_a.T
    P_b = basis_b @ basis_b.T
    return float(np.linalg.norm(P_a - P_b, ord="fro") / max(float(np.linalg.norm(P_b, ord="fro")), 1e-12))


def planted_factors_in_method_coords(fit: MethodFit, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """The planted channel and temporal factors, mapped into the method's frame.

    A covariance-aware method works in whitened coordinates, where the planted
    channel subspace is ``span(Sigma_c^{-1/2} V)`` and the planted temporal
    subspace is ``span(Sigma_t^{-1/2} T)``. Comparing a whitened-coordinate
    fit against a raw-coordinate truth would be a category error.
    """
    if fit.coord == "whitened":
        V_t = np.linalg.qr(_sqrt_inv(ds.covariances["Sigma_c"]) @ ds.V)[0]
        T_t = np.linalg.qr(_sqrt_inv(ds.covariances["Sigma_t"]) @ ds.Tb)[0]
        return V_t, T_t
    return ds.V, ds.Tb


def _whiten_basis_columns(B: np.ndarray, ds: Dataset) -> np.ndarray:
    """Map an orthonormal ``(CT, rank)`` raw-coordinate basis into whitened coords.

    Each column of ``B`` is one basis vector under the column-major ``vec``
    convention, so it must be transposed *before* being unstacked into
    ``(rank, C, T)``. Writing ``B.reshape(rank, C, T, order="F")`` instead has
    the right shape but interleaves the basis index with the channel index, and
    therefore silently whitens a scrambled matrix.

    ``methods.basis_whitened`` contains exactly that defect in its raw-plus-
    factored branch, which is reached only by ``Iso-MPCA`` and feeds only the
    ``angle_to_cw_pca_*`` diagnostic columns. It is deliberately left untouched
    so that Round 9's replication rows stay directly comparable with the
    archived Block A rows; Round 9's own metrics use this corrected routine and
    the discrepancy is reported rather than quietly rewritten.
    """
    rank = B.shape[1]
    W = whiten_separable(
        B.T.reshape(rank, ds.cfg.C, ds.cfg.T, order="F"),
        ds.covariances["Sigma_c"],
        ds.covariances["Sigma_t"],
    )
    return np.linalg.qr(W.reshape(rank, -1, order="F").T)[0]


def fitted_basis_whitened(fit: MethodFit, ds: Dataset) -> np.ndarray:
    """The fitted subspace in oracle-whitened coordinates, correctly unstacked."""
    if fit.name == "OF":
        v = fit.template_whitened.reshape(-1, order="F")
        return (v / np.linalg.norm(v))[:, None]
    if fit.coord == "whitened":
        if fit.U_c is not None:
            return kron_basis(fit.U_c, fit.U_t)
        return np.linalg.qr(fit.P)[0]
    raw = kron_basis(fit.U_c, fit.U_t) if fit.U_c is not None else fit.P
    return _whiten_basis_columns(raw, ds)


def planted_joint_basis_whitened(ds: Dataset) -> np.ndarray:
    """The planted product subspace expressed in oracle-whitened coordinates."""
    return _whiten_basis_columns(kron_basis(ds.V, ds.Tb), ds)


def planted_errors(fit: MethodFit, ds: Dataset) -> dict[str, float | None]:
    """Planted-factor projector errors, with explicit nulls where unidentified.

    * ``NFPA`` and ``Iso-MPCA`` identify channel, temporal and joint subspaces.
    * ``CW-PCA`` and ``IsoPCA`` identify only the joint product subspace: an
      unrestricted rank-6 subspace has no channel or temporal axis to compare,
      so those two fields are ``None``.
    * ``OF`` is a fixed rank-one template and identifies none of them.
    """
    out: dict[str, float | None] = {
        "channel_projector_error": None,
        "temporal_projector_error": None,
        "joint_projector_error": None,
        "joint_projector_error_whitened": None,
    }
    if fit.name == "OF":
        return out

    V_t, T_t = planted_factors_in_method_coords(fit, ds)
    target_joint = kron_basis(V_t, T_t)

    if fit.U_c is not None:
        out["channel_projector_error"] = _projector_error(fit.U_c, V_t)
        out["temporal_projector_error"] = _projector_error(fit.U_t, T_t)
        fitted_joint = kron_basis(fit.U_c, fit.U_t)
    else:
        fitted_joint = np.linalg.qr(fit.P)[0]
    out["joint_projector_error"] = _projector_error(fitted_joint, target_joint)

    # The common oracle-whitened frame, comparable across all non-OF methods.
    out["joint_projector_error_whitened"] = _projector_error(
        fitted_basis_whitened(fit, ds), planted_joint_basis_whitened(ds)
    )
    return out


# --------------------------------------------------------------------------
# Full per-method semantic evaluation
# --------------------------------------------------------------------------

SEMANTIC_METRIC_COLUMNS = (
    "amplitude_rmse",
    "amplitude_bias",
    "amplitude_correlation",
    "rise_time_rmse",
    "decay_time_rmse",
    "channel_projector_error",
    "temporal_projector_error",
    "joint_projector_error",
    "joint_projector_error_whitened",
)


def evaluate_semantics(fit: MethodFit, ds: Dataset) -> dict[str, Any]:
    """Decode held-out physical parameters and measure planted-factor recovery.

    Strict train/test separation: the decoder sees ``ds.X_train`` and
    ``ds.latents_train`` only, and is evaluated against ``ds.latents_test``.
    """
    targets = decode_targets_for(fit.name)
    train_codes = method_codes(fit, ds, ds.X_train)
    test_codes = method_codes(fit, ds, ds.X_test)
    decoder = fit_affine_decoder(train_codes, stack_targets(ds.latents_train, targets), targets)
    pred = decoder.predict(test_codes)
    truth = stack_targets(ds.latents_test, targets)

    err = pred[:, 0] - truth[:, 0]
    metrics: dict[str, Any] = {
        "amplitude_rmse": float(np.sqrt(np.mean(err**2))),
        "amplitude_bias": float(np.mean(err)),
        "amplitude_correlation": float(np.corrcoef(pred[:, 0], truth[:, 0])[0, 1]),
        "rise_time_rmse": None,
        "decay_time_rmse": None,
    }
    if "rise_time" in targets:
        j = targets.index("rise_time")
        metrics["rise_time_rmse"] = float(np.sqrt(np.mean((pred[:, j] - truth[:, j]) ** 2)))
    if "decay_time" in targets:
        j = targets.index("decay_time")
        metrics["decay_time_rmse"] = float(np.sqrt(np.mean((pred[:, j] - truth[:, j]) ** 2)))

    metrics.update(planted_errors(fit, ds))
    metrics["code_dimension"] = int(train_codes.shape[1])
    metrics["decoder"] = decoder.to_dict()
    return metrics
