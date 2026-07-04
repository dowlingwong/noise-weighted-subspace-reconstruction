"""Negative controls for the CRESST validation protocol.

Each control's raw per-event weighted-residual array is returned so the
caller can pair it against the matching nominal array (EMPCA for 4 of the
controls, single-template OF for the template-mismatch control) on the
*same* held-out test events and bootstrap the paired difference -- a raw
point-estimate win/loss is not evidence on its own (see
``detector_metrics.paired_bootstrap_stats`` and Reporting Standard #5).

- ``wrong_psd``: calibrate the subspace with an identity (white-noise) metric
  instead of the estimated inverse covariance;
- ``permuted_time_bins``: destroy the smooth low-rank pulse-shape structure
  by permuting each trace's time bins independently, keeping the noise metric
  fixed to the real (now mismatched) covariance;
- ``random_weighted_basis``: replace the fitted EMPCA basis with a random
  basis of matched rank, orthonormal under the same weighted inner product;
- ``noise_only_train``: fit the subspace on noise-only traces instead of
  clean pulses, then evaluate on signal;
- ``mismatched_of_template``: score with a deliberately wrong (time-reversed)
  single-template OF template instead of the correctly fit one.

A failing (non-degrading) control is itself evidence -- see the Hard Rules
in ``docs/CRESST_REMOTE_AGENT_RUNBOOK.md``: report it, do not tune it away.
"""

from __future__ import annotations

import numpy as np

from ..metrics import weighted_residual
from ..subspace import fit_weighted_pca, project_onto_basis

#: Which nominal per-event residual array each control pairs against.
CONTROL_NOMINAL = {
    "wrong_psd": "empca",
    "permuted_time_bins": "empca",
    "random_weighted_basis": "empca",
    "noise_only_train": "empca",
    "mismatched_of_template": "of",
}


def random_weighted_orthonormal_basis(
    rank: int, n_features: int, metric: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """A random rank-``k`` basis, orthonormal under the weighted inner product."""
    metric = np.asarray(metric, dtype=np.float64)
    A = rng.standard_normal((int(rank), int(n_features)))
    gram = A @ metric @ A.T
    vals, vecs = np.linalg.eigh(0.5 * (gram + gram.T))
    vals = np.clip(vals, np.finfo(float).eps, None)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return inv_sqrt @ A


def _permute_rows_independently(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_features = X.shape[1]
    perm = np.array([rng.permutation(n_features) for _ in range(X.shape[0])])
    return np.take_along_axis(X, perm, axis=1)


def compute_negative_control_residuals(
    train: np.ndarray,
    test: np.ndarray,
    noise_train: np.ndarray,
    metric: np.ndarray,
    rank: int,
    of_template: np.ndarray,
    *,
    seed: int = 22,
    controls: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Raw per-event weighted-residual array for each requested control."""
    rng = np.random.default_rng(int(seed))
    n_features = train.shape[1]
    wanted = set(controls) if controls is not None else set(CONTROL_NOMINAL)
    out: dict[str, np.ndarray] = {}

    if "wrong_psd" in wanted:
        identity_metric = np.eye(n_features)
        fit_wrong = fit_weighted_pca(train, identity_metric, rank)
        recon_wrong = project_onto_basis(test, fit_wrong.components, weights=metric, mean=fit_wrong.mean)
        out["wrong_psd"] = weighted_residual(test, recon_wrong, metric)

    if "permuted_time_bins" in wanted:
        train_perm = _permute_rows_independently(train, rng)
        test_perm = _permute_rows_independently(test, rng)
        fit_perm = fit_weighted_pca(train_perm, metric, rank)
        recon_perm = project_onto_basis(test_perm, fit_perm.components, weights=metric, mean=fit_perm.mean)
        out["permuted_time_bins"] = weighted_residual(test_perm, recon_perm, metric)

    if "random_weighted_basis" in wanted:
        basis = random_weighted_orthonormal_basis(rank, n_features, metric, rng)
        recon_random = project_onto_basis(test, basis, weights=metric, mean=train.mean(axis=0))
        out["random_weighted_basis"] = weighted_residual(test, recon_random, metric)

    if "noise_only_train" in wanted:
        rank_eff = int(min(rank, max(noise_train.shape[0] - 1, 1), n_features))
        fit_noise = fit_weighted_pca(noise_train, metric, rank_eff)
        recon_noise = project_onto_basis(test, fit_noise.components, weights=metric, mean=fit_noise.mean)
        out["noise_only_train"] = weighted_residual(test, recon_noise, metric)

    if "mismatched_of_template" in wanted:
        mismatched = np.asarray(of_template, dtype=np.float64)[::-1]
        denom = float(mismatched @ metric @ mismatched)
        amp = (test @ metric @ mismatched) / denom if denom > 0 else np.zeros(test.shape[0])
        recon_mismatched = amp[:, None] * mismatched[None, :]
        out["mismatched_of_template"] = weighted_residual(test, recon_mismatched, metric)

    return out
