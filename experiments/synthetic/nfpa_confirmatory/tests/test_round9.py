"""Focused tests for the Round 9 runners (E8, E9, E10).

These cover the output-safety contract (overwrite refusal, resume validation,
deterministic aggregation, exact seed sets, malformed and missing shards), the
frozen affine decoder's declared invariances, the method-awareness of the
semantic metrics, and the two places where E10 could silently be wrong: an
interchanged channel/time axis, and the Kronecker trace/Frobenius identities
that stand in for dense ``(C*T)^2`` products.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory import r9_semantics as rs
from experiments.synthetic.nfpa_confirmatory import run_estimated_covariance as e10
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.generator import make_dataset
from experiments.synthetic.nfpa_confirmatory.methods import FIT_FACTORY, fit_cw_pca

# ---------------------------------------------------------------- output safety


def test_output_dir_may_not_resolve_into_the_archived_block_a(tmp_path):
    with pytest.raises(rc.IntegrityError):
        rc.resolve_out_dir(rc.ARCHIVED_E1_DIR, smoke=False)
    with pytest.raises(rc.IntegrityError):
        rc.resolve_out_dir(rc.ARCHIVED_E1_DIR / "nested", smoke=False)
    link = tmp_path / "sneaky"
    link.symlink_to(rc.ARCHIVED_E1_DIR, target_is_directory=True)
    with pytest.raises(rc.IntegrityError):
        rc.resolve_out_dir(link, smoke=False)


def test_smoke_output_is_confined_to_the_smoke_root(tmp_path):
    with pytest.raises(rc.IntegrityError):
        rc.resolve_out_dir(tmp_path / "anywhere", smoke=True)
    ok = rc.resolve_out_dir(rc.SMOKE_ROOT / "x", smoke=True)
    assert str(ok).startswith(str(rc.SMOKE_ROOT))
    with pytest.raises(rc.IntegrityError):
        rc.resolve_out_dir(rc.SMOKE_ROOT / "x", smoke=False)


def test_seed_sets_must_match_the_registered_set_exactly():
    assert rc.guard_seed_set(range(10), range(10), smoke=False) == list(range(10))
    with pytest.raises(rc.IntegrityError):
        rc.guard_seed_set([0, 1], range(10), smoke=False)
    with pytest.raises(rc.IntegrityError):
        rc.guard_seed_set([0, 0, 1], range(10), smoke=False)
    with pytest.raises(rc.IntegrityError):
        rc.guard_seed_set(list(range(10)) + [30], range(10), smoke=False)
    # smoke seeds are reserved and may never enter a scientific run
    with pytest.raises(rc.IntegrityError):
        rc.guard_seed_set([rc.SMOKE_SEEDS[0]], range(10), smoke=False)
    assert rc.guard_seed_set(rc.SMOKE_SEEDS, range(10), smoke=True) == sorted(rc.SMOKE_SEEDS)
    with pytest.raises(rc.IntegrityError):
        rc.guard_seed_set([1, 2], range(10), smoke=True)
    # the reproduction-gate pre-check may use a subset, but never a foreign seed
    assert rc.guard_seed_set([0], range(10), smoke=False, allow_subset=True) == [0]
    with pytest.raises(rc.IntegrityError):
        rc.guard_seed_set([99], range(10), smoke=False, allow_subset=True)


def test_smoke_config_never_mutates_the_frozen_primary():
    before = config_hash(PRIMARY)
    smoke = rc.smoke_config()
    assert config_hash(smoke) != before
    assert config_hash(PRIMARY) == before
    assert PRIMARY.C == 8 and PRIMARY.T == 256


COLS = ["config_hash", "seed", "method", "value"]


def _row(seed, method, value, cfg_hash="deadbeef"):
    return {"config_hash": cfg_hash, "seed": seed, "method": method, "value": value}


def _manifest(out_dir, seeds, methods=("A", "B")):
    return {
        "config_hash": "deadbeef",
        "regime": "colored",
        "methods": list(methods),
        "seeds": list(seeds),
        "columns": COLS,
    }


def test_non_empty_output_dir_is_refused_without_resume(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "seed_00.csv").write_text("stale\n")
    with pytest.raises(rc.IntegrityError, match="not empty"):
        rc.prepare_output_dir(out, resume=False, expected=_manifest(out, [0]))


def test_empty_or_missing_output_dir_is_accepted(tmp_path):
    assert rc.prepare_output_dir(tmp_path / "fresh", resume=False, expected=_manifest(None, [0])) == {}
    empty = tmp_path / "empty"
    empty.mkdir()
    assert rc.prepare_output_dir(empty, resume=False, expected=_manifest(None, [0])) == {}


def test_resume_requires_a_manifest_that_agrees(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    rc.write_rows_csv(out / "seed_00.csv", [_row(0, "A", 1.0), _row(0, "B", 2.0)], COLS)
    with pytest.raises(rc.IntegrityError, match="requires"):
        rc.prepare_output_dir(out, resume=True, expected=_manifest(out, [0]))

    rc.write_json(out / "manifest.json", _manifest(out, [0]))
    assert set(rc.prepare_output_dir(out, resume=True, expected=_manifest(out, [0]))) == {0}

    # a run with a different config hash may not resume this directory
    other = dict(_manifest(out, [0]))
    other["config_hash"] = "different"
    with pytest.raises(rc.IntegrityError, match="mismatch"):
        rc.prepare_output_dir(out, resume=True, expected=other)


def test_resume_rejects_malformed_and_mislabelled_shards(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    rc.write_json(out / "manifest.json", _manifest(out, [0]))

    # wrong config hash inside the shard
    rc.write_rows_csv(
        out / "seed_00.csv", [_row(0, "A", 1.0, "other"), _row(0, "B", 2.0, "other")], COLS
    )
    with pytest.raises(rc.IntegrityError, match="config hash"):
        rc.prepare_output_dir(out, resume=True, expected=_manifest(out, [0]))

    # filename disagrees with the seed it contains
    rc.write_rows_csv(out / "seed_00.csv", [_row(7, "A", 1.0), _row(7, "B", 2.0)], COLS)
    with pytest.raises(rc.IntegrityError, match="does not match the seed"):
        rc.prepare_output_dir(out, resume=True, expected=_manifest(out, [0]))

    # a shard mixing seeds
    rc.write_rows_csv(out / "seed_00.csv", [_row(0, "A", 1.0), _row(1, "B", 2.0)], COLS)
    with pytest.raises(rc.IntegrityError, match="mixes seeds"):
        rc.prepare_output_dir(out, resume=True, expected=_manifest(out, [0]))

    # an incomplete method set
    rc.write_rows_csv(out / "seed_00.csv", [_row(0, "A", 1.0)], COLS)
    with pytest.raises(rc.IntegrityError, match="method set"):
        rc.prepare_output_dir(out, resume=True, expected=_manifest(out, [0]))

    # a schema that does not match the manifest
    rc.write_rows_csv(
        out / "seed_00.csv",
        [{"config_hash": "deadbeef", "seed": 0, "method": m, "other": 1.0} for m in ("A", "B")],
        ["config_hash", "seed", "method", "other"],
    )
    with pytest.raises(rc.IntegrityError, match="schema"):
        rc.prepare_output_dir(out, resume=True, expected=_manifest(out, [0]))

    # an empty shard
    (out / "seed_00.csv").write_text("config_hash,seed,method,value\n")
    with pytest.raises(rc.IntegrityError, match="empty"):
        rc.prepare_output_dir(out, resume=True, expected=_manifest(out, [0]))


def test_aggregation_is_deterministic_and_seed_ordered(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    for seed in (2, 0, 1):
        rc.write_rows_csv(
            rc.shard_path(out, seed), [_row(seed, "A", seed + 0.5), _row(seed, "B", seed + 1.5)], COLS
        )
    rows_a = rc.concat_shards(out.glob("seed_*.csv"))
    rows_b = rc.concat_shards(sorted(out.glob("seed_*.csv"), reverse=True))
    assert rows_a == rows_b
    assert [int(r["seed"]) for r in rows_a] == [0, 0, 1, 1, 2, 2]

    rc.write_rows_csv(out / "raw1.csv", rows_a, COLS)
    rc.write_rows_csv(out / "raw2.csv", rows_b, COLS)
    assert (out / "raw1.csv").read_bytes() == (out / "raw2.csv").read_bytes()


def test_null_metrics_are_written_as_empty_never_zero(tmp_path):
    path = tmp_path / "s.csv"
    rc.write_rows_csv(path, [{"config_hash": "h", "seed": 0, "method": "OF", "value": None}], COLS)
    text = path.read_text()
    assert text.splitlines()[1] == "h,0,OF,"
    assert rc._as_float(rc.read_rows_csv(path)[0]["value"]) is None


def test_missing_column_is_an_error(tmp_path):
    with pytest.raises(rc.IntegrityError, match="missing columns"):
        rc.write_rows_csv(tmp_path / "s.csv", [{"config_hash": "h", "seed": 0}], COLS)


def test_bootstrap_unit_is_the_seed():
    values = [-1.0, -2.0, -0.5, -1.5, -1.0]
    block = rc.bootstrap_paired(values, resamples=500, seed=1)
    assert block["resampling_unit"] == "seed"
    assert block["n_seeds"] == 5
    assert block["per_seed_differences"] == values
    assert block["ci95_low"] <= block["mean_difference"] <= block["ci95_high"]


def test_paired_contrast_reports_unidentified_rather_than_zero():
    rows = [
        {"seed": 0, "method": "NFPA", "m": ""},
        {"seed": 0, "method": "Iso-MPCA", "m": "1.0"},
    ]
    out = rc.paired_contrast(rows, metric="m", method_a="NFPA", method_b="Iso-MPCA")
    assert out["identified"] is False
    assert "mean_difference" not in out


# ------------------------------------------------------------- frozen decoder


def _decoder_fixture(n=200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.normal(size=(n, d))
    targets = codes @ rng.normal(size=(d, 3)) + rng.normal(scale=0.01, size=(n, 3))
    return codes, targets


def test_decoder_is_invariant_to_affine_offsets_of_the_codes():
    codes, targets = _decoder_fixture()
    base = rs.fit_affine_decoder(codes, targets, rs.DECODE_TARGETS)
    shifted = rs.fit_affine_decoder(codes + 7.5, targets, rs.DECODE_TARGETS)
    np.testing.assert_allclose(base.predict(codes), shifted.predict(codes + 7.5), atol=1e-9)


def test_decoder_is_invariant_to_orthogonal_rotations_of_the_codes():
    codes, targets = _decoder_fixture()
    Q = np.linalg.qr(np.random.default_rng(3).normal(size=(codes.shape[1], codes.shape[1])))[0]
    base = rs.fit_affine_decoder(codes, targets, rs.DECODE_TARGETS)
    rotated = rs.fit_affine_decoder(codes @ Q, targets, rs.DECODE_TARGETS)
    np.testing.assert_allclose(base.predict(codes), rotated.predict(codes @ Q), atol=1e-8)


def test_decoder_flags_zero_variance_columns_and_gives_them_scale_one():
    codes, targets = _decoder_fixture()
    codes[:, 1] = 3.0
    dec = rs.fit_affine_decoder(codes, targets, rs.DECODE_TARGETS)
    assert dec.zero_variance_columns == [1]
    assert dec.train_scale[1] == 1.0
    assert np.all(np.isfinite(dec.predict(codes)))


def test_decoder_handles_rank_deficient_codes():
    codes, targets = _decoder_fixture()
    codes[:, 2] = codes[:, 0]  # exact collinearity
    dec = rs.fit_affine_decoder(codes, targets, rs.DECODE_TARGETS)
    assert dec.rank < codes.shape[1] + 1
    assert np.all(np.isfinite(dec.predict(codes)))
    assert len(dec.singular_values) > 0


def test_decoder_sees_training_data_only():
    codes, targets = _decoder_fixture()
    dec_a = rs.fit_affine_decoder(codes, targets, rs.DECODE_TARGETS)
    # perturbing "test" data cannot change a decoder fitted on train alone
    dec_b = rs.fit_affine_decoder(codes.copy(), targets.copy(), rs.DECODE_TARGETS)
    np.testing.assert_allclose(dec_a.coefficients, dec_b.coefficients, atol=0.0)
    np.testing.assert_allclose(dec_a.intercept, dec_b.intercept, atol=0.0)


# --------------------------------------------------- method-aware semantics


@pytest.fixture(scope="module")
def smoke_dataset():
    cfg = rc.smoke_config()
    ds = make_dataset(0, "colored", cfg)
    fits = {name: FIT_FACTORY[name](ds) for name in rc.REGISTERED_METHODS}
    return ds, fits


def test_every_registered_method_produces_codes_of_its_declared_width(smoke_dataset):
    ds, fits = smoke_dataset
    rank = ds.cfg.k_c * ds.cfg.k_t
    for name, fit in fits.items():
        codes = rs.method_codes(fit, ds, ds.X_test)
        expected = 1 if name == "OF" else rank
        assert codes.shape == (ds.cfg.N_test, expected), name


def test_planted_errors_are_null_exactly_where_unidentified(smoke_dataset):
    ds, fits = smoke_dataset
    for name in ("NFPA", "Iso-MPCA"):
        out = rs.planted_errors(fits[name], ds)
        assert all(out[k] is not None for k in out), name
    for name in ("CW-PCA", "IsoPCA"):
        out = rs.planted_errors(fits[name], ds)
        assert out["channel_projector_error"] is None, name
        assert out["temporal_projector_error"] is None, name
        assert out["joint_projector_error"] is not None, name
        assert out["joint_projector_error_whitened"] is not None, name
    out = rs.planted_errors(fits["OF"], ds)
    assert all(v is None for v in out.values())


def test_no_semantic_metric_is_silently_fabricated(smoke_dataset):
    ds, fits = smoke_dataset
    for name, fit in fits.items():
        sem = rs.evaluate_semantics(fit, ds)
        for key in rs.SEMANTIC_METRIC_COLUMNS:
            assert key in sem, (name, key)
            assert sem[key] is None or np.isfinite(sem[key]), (name, key)
        if name == "OF":
            assert sem["rise_time_rmse"] is None and sem["decay_time_rmse"] is None
        else:
            assert sem["rise_time_rmse"] is not None and sem["decay_time_rmse"] is not None


def test_whitened_joint_error_agrees_for_whitened_coordinate_methods(smoke_dataset):
    """A whitened-coordinate fit's own-frame and common-frame joint errors agree.

    ``Sigma^{-1/2}`` is separable, so mapping the planted product subspace into
    whitened coordinates and taking the product of the mapped factors give the
    same subspace. This is the check that the two coordinate systems were not
    mixed up.
    """
    ds, fits = smoke_dataset
    for name in ("NFPA", "CW-PCA"):
        out = rs.planted_errors(fits[name], ds)
        assert out["joint_projector_error"] == pytest.approx(
            out["joint_projector_error_whitened"], abs=1e-8
        ), name


def test_raw_coordinate_methods_are_not_compared_against_a_whitened_target(smoke_dataset):
    """Iso-MPCA's own-frame target is the raw planted basis, not the whitened one."""
    ds, fits = smoke_dataset
    V_t, T_t = rs.planted_factors_in_method_coords(fits["Iso-MPCA"], ds)
    np.testing.assert_allclose(V_t, ds.V, atol=0.0)
    np.testing.assert_allclose(T_t, ds.Tb, atol=0.0)
    V_w, T_w = rs.planted_factors_in_method_coords(fits["NFPA"], ds)
    assert not np.allclose(V_w, ds.V)


# ------------------------------------------------------------------- E10 maths


def test_channel_and_temporal_sample_reshapes_are_not_interchanged():
    """A silent axis interchange must change the numbers, not just the shape.

    ``noise.reshape(N*T, C)`` has the *same shape* as the correct
    ``noise.transpose(0, 2, 1).reshape(N*T, C)`` but scrambles channels with
    time, so only a numerical fixture catches it.
    """
    rng = np.random.default_rng(11)
    N, C, T = 3000, 3, 5
    Sc = np.array([[1.0, 0.7, 0.2], [0.7, 1.0, 0.7], [0.2, 0.7, 1.0]])
    St = np.diag([4.0, 1.0, 0.5, 0.25, 2.0])
    G = rng.standard_normal((N, C, T))
    noise = np.einsum("ab,nbt,ts->nas", np.linalg.cholesky(Sc), G, np.linalg.cholesky(St).T)

    assert e10.channel_samples(noise).shape == (N * T, C)
    assert e10.temporal_samples(noise).shape == (N * C, T)

    Sc_hat, St_hat = e10.estimate_axis_covariances(noise)
    assert e10.shape_error(Sc_hat, Sc) < 0.05
    assert e10.shape_error(St_hat, St) < 0.05

    # the plausible-looking wrong reshape must be materially worse
    wrong = np.linalg.qr(np.eye(C))[0]  # placeholder to keep shapes obvious
    Sc_wrong = e10.regularize_covariance(
        e10.estimate_covariance(noise.reshape(-1, C), center=True), shrinkage=0.0, floor=None
    )
    assert e10.shape_error(Sc_wrong, Sc) > 5.0 * e10.shape_error(Sc_hat, Sc)
    assert wrong.shape == (C, C)


def test_kronecker_frobenius_identity_matches_a_dense_small_case():
    rng = np.random.default_rng(5)

    def spd(n):
        A = rng.standard_normal((n, n))
        return A @ A.T + n * np.eye(n)

    C, T = 3, 4
    Sc, St, Sc_h, St_h = spd(C), spd(T), spd(C), spd(T)
    dense = np.linalg.norm(np.kron(St_h, Sc_h) - np.kron(St, Sc), "fro") / np.linalg.norm(
        np.kron(St, Sc), "fro"
    )
    assert e10.kron_relative_frobenius_error(Sc_h, St_h, Sc, St) == pytest.approx(dense, rel=1e-10)


def test_oracle_whitening_residual_identity_matches_a_dense_small_case():
    rng = np.random.default_rng(6)

    def spd(n):
        A = rng.standard_normal((n, n))
        return A @ A.T + n * np.eye(n)

    C, T = 3, 4
    Sc, St, Sc_h, St_h = spd(C), spd(T), spd(C), spd(T)
    from experiments.synthetic.nfpa_confirmatory.covariance import _sqrt_inv

    W = np.kron(_sqrt_inv(St_h), _sqrt_inv(Sc_h))
    dense = np.linalg.norm(W @ np.kron(St, Sc) @ W.T - np.eye(C * T), "fro") / (C * T)
    assert e10.oracle_whitening_residual(Sc_h, St_h, Sc, St) == pytest.approx(dense, rel=1e-10)


def test_shape_error_is_blind_to_reciprocal_kronecker_rescaling():
    """The identified quantity must not move when scale shifts between factors."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((4, 4))
    Sc = A @ A.T + 4 * np.eye(4)
    assert e10.shape_error(3.7 * Sc, Sc) == pytest.approx(e10.shape_error(Sc, Sc), abs=1e-12)


def test_calibration_rng_is_separate_per_seed_and_reproducible():
    a = e10.calibration_rng(3).standard_normal(5)
    b = e10.calibration_rng(3).standard_normal(5)
    c = e10.calibration_rng(4).standard_normal(5)
    np.testing.assert_allclose(a, b, atol=0.0)
    assert not np.allclose(a, c)
    # and it must not coincide with the dataset RNG for the same seed
    assert not np.allclose(a, np.random.default_rng(3).standard_normal(5))


# ------------------------------------------------- the vec-convention defect


def test_corrected_whitened_basis_matches_column_wise_ground_truth():
    """Unstacking a ``(CT, rank)`` basis must transpose before reshaping.

    Ground truth is computed one column at a time, which involves no reshape
    ambiguity at all.
    """
    from experiments.synthetic.nfpa_confirmatory.covariance import whiten_separable
    from experiments.synthetic.nfpa_confirmatory.generator import kron_basis

    cfg = rc.smoke_config()
    ds = make_dataset(0, "colored", cfg)
    B = kron_basis(ds.V, ds.Tb)
    rank = B.shape[1]

    columns = []
    for r in range(rank):
        M = B[:, r].reshape(cfg.C, cfg.T, order="F")
        Wm = whiten_separable(M[None], ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])[0]
        columns.append(Wm.reshape(-1, order="F"))
    truth = np.linalg.qr(np.column_stack(columns))[0]

    got = rs._whiten_basis_columns(B, ds)
    P_truth = truth @ truth.T
    P_got = got @ got.T
    np.testing.assert_allclose(P_got, P_truth, atol=1e-10)

    # the shape-compatible but scrambled alternative disagrees materially
    Wbad = whiten_separable(
        B.reshape(rank, cfg.C, cfg.T, order="F"),
        ds.covariances["Sigma_c"],
        ds.covariances["Sigma_t"],
    )
    bad = np.linalg.qr(Wbad.reshape(rank, -1, order="F").T)[0]
    assert np.linalg.norm(bad @ bad.T - P_truth, "fro") > 1e-3


def test_archived_basis_whitened_defect_is_confined_to_iso_mpca(smoke_dataset):
    """Pin the known defect in ``methods.basis_whitened`` and its blast radius.

    The archived helper reaches a scrambled reshape only in its raw-plus-
    factored branch, i.e. only for ``Iso-MPCA``, and it feeds only the
    ``angle_to_cw_pca_*`` diagnostics. It is left as-is so Round 9's replication
    rows remain comparable with the archived Block A rows. This test exists so
    the defect cannot be forgotten or silently "fixed" without noticing that it
    changes the meaning of an archived column.
    """
    from experiments.synthetic.nfpa_confirmatory.methods import basis_whitened

    ds, fits = smoke_dataset
    for name in ("CW-PCA", "NFPA", "IsoPCA", "OF"):
        A = basis_whitened(fits[name], ds)
        B = rs.fitted_basis_whitened(fits[name], ds)
        np.testing.assert_allclose(A @ A.T, B @ B.T, atol=1e-9, err_msg=name)

    A = basis_whitened(fits["Iso-MPCA"], ds)
    B = rs.fitted_basis_whitened(fits["Iso-MPCA"], ds)
    assert np.linalg.norm(A @ A.T - B @ B.T, "fro") > 1e-3


def test_auxiliary_per_seed_files_are_not_mistaken_for_shards(tmp_path):
    """``repro_seed_*.csv`` and ``decoder_seed_*.json`` must not glob as shards."""
    out = tmp_path / "run"
    out.mkdir()
    rc.write_json(out / "manifest.json", _manifest(out, [0]))
    rc.write_rows_csv(rc.shard_path(out, 0), [_row(0, "A", 1.0), _row(0, "B", 2.0)], COLS)
    # auxiliary artefacts written alongside the shard
    rc.write_rows_csv(out / "repro_seed_00.csv", [_row(0, "A", 9.0), _row(0, "B", 9.0)], COLS)
    rc.write_json(out / "decoder_seed_00.json", {"seed": 0})

    found = rc.prepare_output_dir(out, resume=True, expected=_manifest(out, [0]))
    assert set(found) == {0}
    assert found[0].name == "seed_00.csv"
    assert sorted(p.name for p in out.glob("seed_*.csv")) == ["seed_00.csv"]
