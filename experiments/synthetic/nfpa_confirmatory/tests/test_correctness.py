"""Phase 0 correctness tests for the NFPA confirmatory harness.

Covers the seven required tests plus one generator validation (Monte Carlo
covariance agreement of the PSD-based noise path, protocol check 4). Run from
the repository root:

    python -m pytest experiments/synthetic/nfpa_confirmatory/tests/
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from experiments.synthetic.nfpa_confirmatory.config import PrimaryConfig
from experiments.synthetic.nfpa_confirmatory.covariance import _sqrt_inv, full_covariance
from experiments.synthetic.nfpa_confirmatory.generator import kron_basis, make_dataset, project_out_product_subspace
from experiments.synthetic.nfpa_confirmatory.methods import als_fit, fit_cw_pca, fit_iso_pca
from experiments.synthetic.nfpa_confirmatory.metrics import normalized_chi2, subspace_angles_to

REPO_ROOT = Path(__file__).resolve().parents[4]
DEMO_PATH = REPO_ROOT / "src" / "NFPA" / "nfpa_demo.py"


def test_whitening_identity():
    cfg = PrimaryConfig()
    cov = full_covariance(cfg, "colored")["Sigma"]
    d = cov.shape[0]
    W = _sqrt_inv(cov)
    err = float(np.linalg.norm(W @ cov @ W.T - np.eye(d), ord="fro") / d)
    assert err < 1e-8, f"whitening identity error {err:.2e}"


def test_cw_pca_reduces_to_pca_under_white_noise():
    ds = make_dataset(0, "white")
    cw = fit_cw_pca(ds)
    iso = fit_iso_pca(ds)
    angles = subspace_angles_to(cw, iso, ds)
    # Paper 1 reports 1.48e-6 deg for this check; the residual is float
    # precision of eigh(I) in the identity whitening, so 1e-5 is the order bar.
    assert float(np.max(angles)) < 1e-5, f"max angle {np.max(angles):.3e} deg"


def test_rank_one_nfpa_equals_optimal_filter():
    cfg = PrimaryConfig()
    rng = np.random.default_rng(7)
    C, T = cfg.C, cfg.T
    V = np.linalg.qr(rng.standard_normal((C, 1)))[0]
    Tb = np.linalg.qr(rng.standard_normal((T, 1)))[0]
    s0 = np.outer(V[:, 0], Tb[:, 0])
    A = np.abs(rng.normal(1.0, 0.2, size=4000))
    clean = A[:, None, None] * s0[None, :, :]
    noise = rng.standard_normal((4000, C, T)) * 0.05
    X = clean + noise
    Z = X
    out = als_fit(Z, 1, 1, n_iter=120, n_restarts=3, seed=3)
    basis = kron_basis(out["U_c"], out["U_t"])[:, 0]
    template = s0.reshape(-1, order="F")
    cosine = abs(float(basis @ template / (np.linalg.norm(basis) * np.linalg.norm(template))))
    assert cosine >= 0.9999, f"rank-one NFPA/OF cosine {cosine:.6f}"


def test_kronecker_projector_identity():
    rng = np.random.default_rng(11)
    U_c = np.linalg.qr(rng.standard_normal((8, 2)))[0]
    U_t = np.linalg.qr(rng.standard_normal((256, 3)))[0]
    Z = rng.standard_normal((8, 256))
    lhs = (U_c @ U_c.T @ Z @ U_t @ U_t.T).reshape(-1, order="F")
    P = np.kron(U_t @ U_t.T, U_c @ U_c.T)
    rhs = P @ Z.reshape(-1, order="F")
    assert float(np.linalg.norm(lhs - rhs)) < 1e-10


def test_kronecker_projector_idempotent_symmetric():
    rng = np.random.default_rng(13)
    U_c = np.linalg.qr(rng.standard_normal((8, 2)))[0]
    U_t = np.linalg.qr(rng.standard_normal((256, 3)))[0]
    P = np.kron(U_t @ U_t.T, U_c @ U_c.T)
    assert float(np.linalg.norm(P @ P - P)) < 1e-10
    assert float(np.linalg.norm(P - P.T)) < 1e-10


def test_nfpa_full_rank_is_identity():
    rng = np.random.default_rng(17)
    C, T = 8, 32
    Z = rng.standard_normal((50, C, T))
    out = als_fit(Z, C, T, n_iter=8, n_restarts=1, seed=1)
    recon = np.einsum("nkl,ck,tl->nct", np.einsum("nct,ck,tl->nkl", Z, out["U_c"], out["U_t"]), out["U_c"], out["U_t"])
    err = float(np.mean((recon - Z) ** 2))
    assert err < 1e-10, f"identity reconstruction error {err:.2e}"


def test_archived_degenerate_case_reproduced():
    spec = importlib.util.spec_from_file_location("nfpa_demo_archive", DEMO_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nfpa_demo_archive"] = mod
    buf = io.StringIO()
    with redirect_stdout(buf):
        spec.loader.exec_module(mod)
    r = mod.R
    assert abs(r["NFPA"]["chi2"] - 0.9916) < 2e-4, f"NFPA chi2 {r['NFPA']['chi2']:.5f}"
    assert abs(r["EMPCA"]["chi2"] - 0.9929) < 2e-4, f"EMPCA chi2 {r['EMPCA']['chi2']:.5f}"
    expected = np.array([0.0, 59.3, 64.6, 73.8, 84.8, 87.8])
    got = np.round(mod.ang_NE, 1)
    assert np.allclose(got, expected, atol=0.2), f"angles {got}"


def test_generated_noise_matches_covariance():
    cfg = PrimaryConfig()
    cov = full_covariance(cfg, "colored")
    rng = np.random.default_rng(5)
    from experiments.synthetic.nfpa_confirmatory.covariance import _sqrt_inv
    from experiments.synthetic.nfpa_confirmatory.generator import generate_noise

    N = generate_noise(rng, 8192, cfg, cov)
    flat = N.reshape(N.shape[0], -1, order="F")
    W = _sqrt_inv(cov["Sigma"])
    whitened = flat @ W.T
    empirical = (whitened.T @ whitened) / whitened.shape[0]
    d = cov["Sigma"].shape[0]
    # Whitened-space comparison avoids the Brownian ill-conditioning; the raw
    # Frobenius MC error is dominated by the large low-frequency eigenvalues.
    err = float(np.linalg.norm(empirical - np.eye(d), ord="fro") / d)
    assert err < 5e-2, f"generator noise whitened covariance error {err:.2e}"


def test_runner_gate_matches_the_audited_monte_carlo_statistic():
    """The runner must gate on the same statistic this file audits.

    Regression test for the 2026-08-11 Block A failure. `run_numerical_checks`
    gated on the RAW relative Frobenius error while
    :func:`test_generated_noise_matches_covariance` above audits the WHITENED
    error against the identity, and both used a 5e-2 tolerance. Those are
    different quantities: with cond(Sigma) ~ 1e5 the raw error is dominated by
    the Brownian covariance's large low-frequency eigenvalues and cannot meet
    that tolerance at any practical sample count. The smoke run stopped at
    8.20e-02 before fitting a single method, while this file's audit passed.

    If the two ever diverge again this test fails rather than a confirmatory run.
    """
    from experiments.synthetic.nfpa_confirmatory.covariance import (
        _sqrt_inv,
        assert_checks_pass,
        run_numerical_checks,
    )

    cfg = PrimaryConfig()
    cov = full_covariance(cfg, "colored")
    Sigma = cov["Sigma"]
    report = run_numerical_checks(cfg, Sigma)

    # The gated key exists and is the whitened statistic.
    assert "monte_carlo_whitened_covariance_error" in report
    assert_checks_pass(report)

    # Reproduce the audited statistic independently and confirm the runner's
    # gated value is the same quantity, to Monte Carlo agreement.
    d = Sigma.shape[0]
    rng = np.random.default_rng(11)
    noise = rng.standard_normal((report["monte_carlo_draws"], d)) @ np.linalg.cholesky(Sigma).T
    whitened = noise @ _sqrt_inv(Sigma).T
    empirical = (whitened.T @ whitened) / whitened.shape[0]
    independent = float(np.linalg.norm(empirical - np.eye(d), ord="fro") / d)

    assert independent < 5e-2, f"independent whitened error {independent:.2e}"
    assert abs(report["monte_carlo_whitened_covariance_error"] - independent) < 2e-2, (
        "runner's gated statistic is not the audited whitened statistic: "
        f"{report['monte_carlo_whitened_covariance_error']:.3e} vs {independent:.3e}"
    )

    # And the raw error really is the larger, ill-conditioned quantity that
    # motivated the whitened comparison — kept as a diagnostic, never gated.
    assert report["monte_carlo_covariance_relative_error"] > report[
        "monte_carlo_whitened_covariance_error"
    ], "raw error should exceed whitened error for this covariance"


def test_nonseparable_component_orthogonal_to_product_subspace():
    cfg = PrimaryConfig()
    ds = make_dataset(0, "colored")
    rng = np.random.default_rng(2)
    from experiments.synthetic.nfpa_confirmatory.generator import _nonseparable_perturbations

    R = _nonseparable_perturbations(rng, 64, cfg, ds.V, ds.Tb)
    B = kron_basis(ds.V, ds.Tb)
    proj = R.reshape(64, -1, order="F") @ B
    assert float(np.max(np.abs(proj))) < 1e-8
