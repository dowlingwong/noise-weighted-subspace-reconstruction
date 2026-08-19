"""Focused tests for the Round 10 corrections (E10b, E11).

Two defects are corrected, each behind a versioned path:

* the legacy ``methods.basis_whitened`` raw-factored branch scrambles a
  ``(C*T, rank)`` basis before whitening (Iso-MPCA only) — corrected by
  ``angles_v2`` under new ``*_v2`` column names;
* Round 9's E10 marginal covariance estimates each carry the full average
  measurement variance, so their Kronecker product is off by that scale —
  corrected by ``covariance_estimation_v2`` with the frozen symmetric
  ``v_hat`` normalization.

The archived Round 9 evidence is immutable; the digest tests pin it.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

from experiments.synthetic.nfpa_confirmatory import angles_v2 as av2
from experiments.synthetic.nfpa_confirmatory import covariance_estimation_v2 as cev2
from experiments.synthetic.nfpa_confirmatory import metrics as metrics_mod
from experiments.synthetic.nfpa_confirmatory import methods as methods_mod
from experiments.synthetic.nfpa_confirmatory import r10_common as r10
from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory import run_calibrated_estimated_covariance as e10b
from experiments.synthetic.nfpa_confirmatory import run_corrected_angles as e11
from experiments.synthetic.nfpa_confirmatory.covariance import whiten_separable
from experiments.synthetic.nfpa_confirmatory.generator import generate_noise, make_dataset
from experiments.synthetic.nfpa_confirmatory.methods import fit_cw_pca, fit_iso_mpca, fit_nfpa

FAST = os.environ.get("NFPA_CONFIRMATORY_FAST") == "1"

#: Archive digests pinned at Round 10 Phase 0 preflight. If any of these tests
#: fails, STOP ALL ROUND 10 WORK: the archive has been modified.
ARCHIVE_DIGESTS = {
    "results/rps2026/E1_nfpa_blockA/block_A_colored_raw.csv":
        "e9778eb2d66366501e7016d0e3bfa3626aad6b0575322d94da995f583e6dbd7e",
    "results/rps2026/E9_blockA_replication/replication_raw.csv":
        "cb228581edc979e58dc8a152d2e8c484ff46d23b702141d5917f7e94d221121e",
    "results/rps2026/E9_blockA_replication/replication_summary.json":
        "769582050107b269361e35c2fddb8f5549a08e56f34d3736d3a8cc6f7df93bd7",
    "results/rps2026/E10_estimated_covariance/raw.csv":
        "9c17834614a67694986e026d0abaec69e3078b6d8747216c4bb98966383bcfae",
    "results/rps2026/E10_estimated_covariance/summary.json":
        "bd0e0336d03b4d3992199d5de71d97e6bbba4cde076ac90dc3fda586ce8c8737",
    "results/rps2026/E8_semantics/archived_fit_reproduction.csv":
        "a545925fc82e47a7f593731ed7b9f6f1df258d4662a64ed3704a4a48cca2a5f6",
}


def _projector(B: np.ndarray) -> np.ndarray:
    Q = np.linalg.qr(B)[0]
    return Q @ Q.T


def _spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    M = A @ A.T + n * np.eye(n)
    return 0.5 * (M + M.T)


# ------------------------------------------------------------- angle correction


def test_corrected_helper_matches_direct_column_by_column_whitening():
    C, T, rank = 4, 6, 3
    rng = np.random.default_rng(20260819)
    B = rng.standard_normal((C * T, rank))
    Sc, St = _spd(C, 1), _spd(T, 2)

    direct_cols = []
    for i in range(rank):
        trace = B[:, i].reshape(C, T, order="F")[None, :, :]
        direct_cols.append(whiten_separable(trace, Sc, St)[0].reshape(-1, order="F"))
    reference = np.linalg.qr(np.stack(direct_cols, axis=1))[0]

    corrected = av2.whiten_basis_columns(B, Sc, St)
    assert np.allclose(_projector(corrected), _projector(reference), atol=1e-10)


def test_corrected_helper_is_invariant_to_subspace_rotations():
    C, T, rank = 5, 7, 3
    rng = np.random.default_rng(7)
    B = rng.standard_normal((C * T, rank))
    R = np.linalg.qr(rng.standard_normal((rank, rank)))[0]
    Sc, St = _spd(C, 3), _spd(T, 4)
    P1 = _projector(av2.whiten_basis_columns(B, Sc, St))
    P2 = _projector(av2.whiten_basis_columns(B @ R, Sc, St))
    assert np.allclose(P1, P2, atol=1e-10)


@pytest.fixture(scope="module")
def smoke_ds():
    return make_dataset(910000, "colored", rc.smoke_config())


def test_nfpa_and_cw_pca_corrected_angles_agree_with_legacy(smoke_ds):
    ds = smoke_ds
    cw = fit_cw_pca(ds)
    nfpa = fit_nfpa(ds)
    for fit in (cw, nfpa):
        legacy = methods_mod.basis_whitened(fit, ds)
        v2 = av2.basis_whitened_v2(fit, ds)
        assert np.allclose(_projector(legacy), _projector(v2), atol=1e-10)
    legacy_angles = metrics_mod.subspace_angles_to(nfpa, cw, ds)
    v2_angles = av2.subspace_angles_to_v2(nfpa, cw, ds)
    assert np.allclose(legacy_angles, v2_angles, atol=1e-8)


def test_legacy_and_corrected_iso_mpca_bases_differ_on_the_exposing_fixture(smoke_ds):
    """The smoke config has C=6 != T=32, which exposes the interleaving error."""
    ds = smoke_ds
    iso = fit_iso_mpca(ds)
    assert iso.coord == "raw" and iso.U_c is not None  # the defective branch
    legacy = methods_mod.basis_whitened(iso, ds)
    v2 = av2.basis_whitened_v2(iso, ds)
    diff = float(np.abs(_projector(legacy) - _projector(v2)).max())
    assert diff > 1e-3, f"fixture failed to expose the interleaving defect (diff={diff})"

    cw = fit_cw_pca(ds)
    legacy_mean = float(np.mean(metrics_mod.subspace_angles_to(iso, cw, ds)))
    v2_mean = av2.angle_metrics_v2(iso, cw, ds)["angle_to_cw_pca_mean_deg_v2"]
    assert abs(legacy_mean - v2_mean) > 1e-3


def test_no_primary_reconstruction_metric_calls_the_legacy_helper():
    for fn in (metrics_mod.raw_mse, metrics_mod.normalized_chi2, methods_mod.reconstruct):
        assert "basis_whitened" not in inspect.getsource(fn)
    # The corrected module is the only sanctioned angle path for new outputs.
    assert "angle_metrics_v2" in inspect.getsource(e11.run_seed)


def test_archived_digests_are_unchanged():
    for rel, want in ARCHIVE_DIGESTS.items():
        path = rc.REPO_ROOT / rel
        assert path.exists(), f"archived file missing: {rel}"
        got = rc.sha256_of(path)
        assert got == want, f"ARCHIVE MODIFIED: {rel}\n  expected {want}\n  got      {got}"


# ------------------------------------------------- covariance scale correction


@pytest.fixture(scope="module")
def calibrated_estimate():
    cfg = rc.smoke_config()
    ds = make_dataset(910001, "colored", cfg)
    noise = generate_noise(np.random.default_rng(123), 96, cfg, ds.covariances)
    return cev2.estimate_calibrated_axis_covariances(noise), ds


def test_normalization_identity_on_a_deterministic_covariance(calibrated_estimate):
    est, _ = calibrated_estimate
    C = est.Sigma_c_hat.shape[0]
    T = est.Sigma_t_hat.shape[0]
    assert est.v_hat == pytest.approx(np.sqrt(est.v_c * est.v_t), rel=1e-12)
    product_diag_mean = float(
        np.trace(np.kron(est.Sigma_t_hat, est.Sigma_c_hat))
    ) / (C * T)
    assert product_diag_mean == pytest.approx(est.v_hat, rel=1e-10)


def test_corrected_dense_product_is_raw_product_over_v_hat(calibrated_estimate):
    est, _ = calibrated_estimate
    lhs = np.kron(est.Sigma_t_hat, est.Sigma_c_hat)
    rhs = np.kron(est.Sigma_t_regularized, est.Sigma_c_regularized) / est.v_hat
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=0)


def test_positive_definiteness_after_regularization_and_normalization(calibrated_estimate):
    est, _ = calibrated_estimate
    for M in (est.Sigma_c_hat, est.Sigma_t_hat):
        assert float(np.linalg.eigvalsh(0.5 * (M + M.T)).min()) > 0


def test_reconstruction_is_invariant_under_common_global_rescaling(smoke_ds):
    from dataclasses import replace
    from experiments.synthetic.nfpa_confirmatory.methods import reconstruct

    ds = smoke_ds
    a = 4.0  # exact in binary floating point
    scaled = replace(
        ds,
        covariances={
            "Sigma_c": a * ds.covariances["Sigma_c"],
            "Sigma_t": a * ds.covariances["Sigma_t"],
        },
    )
    fit_ref = fit_nfpa(ds)
    fit_scaled = fit_nfpa(scaled)
    X = ds.X_test[:16]
    Xhat_ref = reconstruct(fit_ref, ds, X)
    Xhat_scaled = reconstruct(fit_scaled, scaled, X)
    assert np.allclose(Xhat_ref, Xhat_scaled, rtol=1e-6, atol=1e-10)


def test_calibration_events_are_not_the_replication_unit():
    """The bootstrap unit is the seed; N_cal never inflates the sample size."""
    rows = []
    for seed in (0, 1, 2):
        rows.append({
            "seed": seed, "method": "NFPA",
            "delta_normalized_chi2_estimated_minus_oracle": 0.01 * (seed + 1),
            "delta_raw_mse_estimated_minus_oracle": 0.001,
            "test_normalized_chi2": 1.0, "test_raw_mse": 0.01,
            "v_c": 1.0, "v_t": 1.0, "v_hat": 1.0,
            "product_scale_ratio_vs_oracle": 1.0,
            "channel_shape_error": 0.01, "temporal_shape_error": 0.01,
            "kronecker_relative_frobenius_error": 0.02,
            "oracle_whitening_residual": 0.001,
        })
        rows.append({
            "seed": seed, "method": "Iso-MPCA",
            "test_normalized_chi2": 1.02, "test_raw_mse": 0.007,
        })
    out = e10b.summarize(rows, seeds=[0, 1, 2])
    deg = out["degradation.delta_normalized_chi2_estimated_minus_oracle"]
    assert deg["n_seeds"] == 3
    assert deg["resampling_unit"] == "seed"
    contrast = out["contrast_estimated_nfpa_minus_iso.test_normalized_chi2"]
    assert contrast["n_seeds"] == 3 and contrast["resampling_unit"] == "seed"


@pytest.mark.skipif(FAST, reason="full-config refit; run without NFPA_CONFIRMATORY_FAST=1")
def test_iso_mpca_control_reproduces_archived_primary_metrics():
    """Seed 0, full frozen config: the covariance-independent control must
    reproduce the archived E1 metrics to <=1e-9 relative."""
    from experiments.synthetic.nfpa_confirmatory.config import PRIMARY
    from experiments.synthetic.nfpa_confirmatory.metrics import normalized_chi2, raw_mse
    from experiments.synthetic.nfpa_confirmatory.methods import reconstruct

    archived = e10b.load_archived()[(0, "Iso-MPCA")]
    ds = make_dataset(0, "colored", PRIMARY)
    fit = fit_iso_mpca(ds)
    Xhat = reconstruct(fit, ds, ds.X_test)
    got_raw = raw_mse(ds.X_test, Xhat)
    got_chi2 = normalized_chi2(ds, ds.X_test, Xhat)
    assert abs(got_raw - archived["test_raw_mse"]) / abs(archived["test_raw_mse"]) <= 1e-9
    assert (
        abs(got_chi2 - archived["test_normalized_chi2"]) / abs(archived["test_normalized_chi2"])
        <= 1e-9
    )


# ----------------------------------------------------------- round 10 output safety


def test_round10_smoke_root_and_seeds_are_disjoint_from_round9():
    assert r10.SMOKE_ROOT != rc.SMOKE_ROOT
    assert not set(r10.SMOKE_SEEDS) & set(rc.SMOKE_SEEDS)
    assert not set(r10.SMOKE_SEEDS) & set(range(30))


def test_round10_output_refuses_every_archived_directory(tmp_path):
    for archived in r10.ARCHIVED_DIRS:
        with pytest.raises(r10.IntegrityError):
            r10.resolve_out_dir(archived, smoke=False)
        with pytest.raises(r10.IntegrityError):
            r10.resolve_out_dir(archived / "nested", smoke=False)
    with pytest.raises(r10.IntegrityError):
        r10.resolve_out_dir(rc.SMOKE_ROOT / "x", smoke=False)  # round 9 smoke root
    ok = r10.resolve_out_dir(r10.SMOKE_ROOT / "x", smoke=True)
    assert str(ok).startswith(str(r10.SMOKE_ROOT))


def test_round10_guard_rejects_all_reserved_smoke_seeds():
    with pytest.raises(r10.IntegrityError):
        r10.guard_seed_set([r10.SMOKE_SEEDS[0]], range(30), smoke=False)
    with pytest.raises(r10.IntegrityError):
        r10.guard_seed_set([rc.SMOKE_SEEDS[0]], range(30), smoke=False)
    with pytest.raises(r10.IntegrityError):
        r10.guard_seed_set(rc.SMOKE_SEEDS, range(30), smoke=True)  # r9 seeds, r10 smoke
    assert r10.guard_seed_set(r10.SMOKE_SEEDS, range(30), smoke=True) == sorted(r10.SMOKE_SEEDS)


def test_e11_and_e10b_smoke_runs_end_to_end(tmp_path, monkeypatch):
    """Both runners execute the full shard->raw->manifest path on the reduced
    config with reserved seeds, in an isolated smoke root."""
    monkeypatch.setattr(r10, "SMOKE_ROOT", tmp_path / "Round10_smoke")

    rc_e11 = e11.main(["--smoke", "--out", str(tmp_path / "Round10_smoke" / "E11")])
    assert rc_e11 == 0
    assert (tmp_path / "Round10_smoke" / "E11" / "raw.csv").exists()
    assert (tmp_path / "Round10_smoke" / "E11" / "manifest.json").exists()

    rc_e10b = e10b.main(["--smoke", "--out", str(tmp_path / "Round10_smoke" / "E10b")])
    assert rc_e10b == 0
    rows = rc.read_rows_csv(tmp_path / "Round10_smoke" / "E10b" / "raw.csv")
    nfpa_rows = [r for r in rows if r["method"] == "NFPA"]
    assert nfpa_rows
    # After calibration the product-scale ratio must sit near 1, not near
    # the squared-variance regime E10 lived in.
    for r in nfpa_rows:
        ratio = float(r["product_scale_ratio_vs_oracle"])
        assert 0.5 < ratio < 2.0, f"calibrated scale ratio {ratio} is not near 1"
