"""Known-truth sanity checks for the frozen T15 generator G1."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.mlst2026 import t15_covariance_regime as t15


def test_covariance_is_symmetric_positive_definite_and_unit_scaled():
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    np.testing.assert_allclose(S, S.T, atol=1e-12)
    assert np.linalg.eigvalsh(S).min() > 0
    assert np.trace(S) / t15.D == pytest.approx(1.0, rel=1e-12)


def test_every_misspecified_covariance_keeps_the_same_scale():
    """Misspecification must change the shape only, never the scale."""
    for e in t15.SLOPE_SUBARMS:
        S = t15.covariance_from_psd(t15.psd_vector(e, t15.KNEE_BINS))
        assert np.trace(S) / t15.D == pytest.approx(1.0, rel=1e-12)
    for mult in t15.CORRLENGTH_SUBARMS:
        S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT,
                                                   t15.KNEE_BINS * mult))
        assert np.trace(S) / t15.D == pytest.approx(1.0, rel=1e-12)


def test_circulant_eigenvalues_match_the_matrix_spectrum():
    psd = t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS)
    S = t15.covariance_from_psd(psd)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(S)),
                               np.sort(t15.circulant_eigenvalues(psd)), atol=1e-10)


def test_generated_noise_reproduces_the_declared_covariance():
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    Z = t15.sample_noise(np.random.default_rng(0), t15.matrix_sqrt(S), 200_000)
    emp = (Z.T @ Z) / Z.shape[0]
    assert t15.relative_operator_error(emp, S) < 0.05


def test_the_exponent_is_minus_one_not_the_s5_minus_two():
    """Guards the whole point of the mismatch resolution."""
    assert t15.TRUE_EXPONENT == -1.0
    f = np.fft.rfftfreq(t15.D, d=1.0)[1:]
    psd = t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS)[1:]
    hi = f > 8 * (t15.KNEE_BINS / t15.D)  # well above the knee
    slope = np.polyfit(np.log(f[hi]), np.log(psd[hi]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.05)


def test_signal_basis_is_orthonormal_and_not_a_sinusoid_pair():
    U = t15.frozen_signal_basis()
    np.testing.assert_allclose(U.T @ U, np.eye(t15.RANK), atol=1e-12)
    # a sinusoid pair would be concentrated in one rfft bin; a generic subspace
    # is spread across many
    p = np.abs(np.fft.rfft(U[:, 0])) ** 2
    assert p.max() / p.sum() < 0.25


#: Precision floor of the ``sqrt(1 - cos^2)`` principal-angle estimator that
#: ``t15.sin_theta`` implements.  Near coincidence a relative error ``delta`` in
#: the cosine becomes ``sqrt(2*delta)`` in the sine, so with ``delta`` at
#: double-precision epsilon nothing below ``sqrt(2*eps) ~ 2.1e-8`` is
#: resolvable; the measured value on an exactly coincident pair is 3.7e-8.
ARCCOS_FLOOR = 1e-7


def test_signal_basis_is_frozen_across_calls():
    np.testing.assert_array_equal(t15.frozen_signal_basis(),
                                  t15.frozen_signal_basis())


def test_oracle_arm_has_exactly_zero_covariance_error():
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    U = t15.frozen_signal_basis()
    r = t15.run_cell("oracle", 0, S, t15.matrix_sqrt(S), U)
    assert r["epsilon_relative_operator"] == pytest.approx(0.0, abs=1e-12)
    # The theorem's target is the shift away from the paired oracle fit, which
    # for the oracle arm is identically zero; the distance to the *planted*
    # subspace is nonzero even here, purely from the finite training sample.
    #
    # Tolerance: sin_theta is computed as sqrt(1 - cos^2), which near cos = 1
    # amplifies a relative error delta into sqrt(2*delta).  With delta at
    # double-precision epsilon the achievable floor is sqrt(2*eps) ~ 2.1e-8, and
    # the measured value on an exactly coincident pair is 3.7e-8.  The 1e-9 this
    # test used was below the estimator's own precision floor and could only pass
    # by luck of rounding; ARCCOS_FLOOR is that floor with a small margin.  The
    # estimator itself is unchanged, because the archived T15 results were
    # produced with it.
    assert r["sin_theta_vs_paired_oracle_fit"] == pytest.approx(0.0, abs=ARCCOS_FLOOR)
    assert r["bound_holds"] is True
    assert r["sin_theta_vs_planted"] > 0.0


def test_covariance_error_falls_as_calibration_traces_grow():
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    sq, U = t15.matrix_sqrt(S), t15.frozen_signal_basis()
    eps = [t15.run_cell(f"estimated-{m}", 0, S, sq, U)["epsilon_relative_operator"]
           for m in (64, 256, 2048)]
    assert eps[0] > eps[1] > eps[2]


def test_oracle_subspace_error_scales_as_one_over_sqrt_training_size():
    """Known truth: with the true covariance the subspace error is pure
    sampling error and must follow the n^{-1/2} rate."""
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    sq, U = t15.matrix_sqrt(S), t15.frozen_signal_basis()
    old = t15.N_TRAIN
    try:
        scaled = []
        for n in (2048, 8192, 32768):
            t15.N_TRAIN = n
            v = np.mean([t15.run_cell("oracle", s, S, sq, U)["sin_theta_vs_planted"]
                         for s in range(3)])
            scaled.append(v * np.sqrt(n))
    finally:
        t15.N_TRAIN = old
    assert max(scaled) / min(scaled) < 1.3


def test_theorem_bound_holds_on_every_arm():
    """The bound must never be violated by the quantity it constrains."""
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    sq, U = t15.matrix_sqrt(S), t15.frozen_signal_basis()
    for arm in t15.ARMS:
        for seed in (0, 7):
            r = t15.run_cell(arm, seed, S, sq, U)
            assert r["bound_holds"], (arm, seed, r["bound_slack"])


def test_misspecification_moves_the_fit_away_from_the_paired_oracle():
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    sq, U = t15.matrix_sqrt(S), t15.frozen_signal_basis()
    o = t15.run_cell("oracle", 2, S, sq, U)
    for arm in ("misspecified-slope", "misspecified-corrlength", "estimated-64"):
        r = t15.run_cell(arm, 2, S, sq, U)
        assert r["sin_theta_vs_paired_oracle_fit"] > o["sin_theta_vs_paired_oracle_fit"]


def test_subspace_error_vanishes_as_training_grows_with_the_oracle_covariance():
    """Known truth: with the true covariance and a large training sample the
    fitted subspace must converge to the planted one."""
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    sq, U = t15.matrix_sqrt(S), t15.frozen_signal_basis()
    old = t15.N_TRAIN
    try:
        t15.N_TRAIN = 200_000
        r = t15.run_cell("oracle", 1, S, sq, U)
    finally:
        t15.N_TRAIN = old
    assert r["sin_theta_vs_planted"] < 0.02


def test_operator_and_frobenius_errors_are_genuinely_different():
    """The whole point of the correction: the theorem's operator-norm error is
    not the old normalised Frobenius error."""
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    sq, U = t15.matrix_sqrt(S), t15.frozen_signal_basis()
    r = t15.run_cell("estimated-64", 0, S, sq, U)
    assert r["epsilon_relative_operator"] != pytest.approx(
        r["epsilon_relative_frobenius"], rel=0.05)


def test_arms_at_one_seed_share_the_same_data_so_comparisons_are_paired():
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    sq, U = t15.matrix_sqrt(S), t15.frozen_signal_basis()
    a = t15.run_cell("oracle", 3, S, sq, U)
    b = t15.run_cell("oracle", 3, S, sq, U)
    assert a["sin_theta_vs_planted"] == b["sin_theta_vs_planted"]
    # population quantities do not depend on the data draw
    c = t15.run_cell("estimated-512", 3, S, sq, U)
    assert c["lambda_1"] == pytest.approx(a["lambda_1"])
    assert c["delta_lambda"] == pytest.approx(a["delta_lambda"])


def test_arm_count_and_fit_budget_match_the_declared_eighty():
    assert len(t15.ARMS) == 8
    assert len(t15.SEEDS) == 10
    assert len(t15.ARMS) * len(t15.SEEDS) == t15.FIT_CAP == 80


def test_subarms_split_the_seeds_five_and_five():
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    sq = t15.matrix_sqrt(S)
    labels = {s: t15.covariance_for_arm("misspecified-slope", s,
                                        np.random.default_rng(0), S, sq)[1]["subarm"]
              for s in range(10)}
    assert len({labels[s] for s in range(5)}) == 1
    assert len({labels[s] for s in range(5, 10)}) == 1
    assert labels[0] != labels[9]


def test_sin_theta_is_zero_for_identical_subspaces_and_one_for_orthogonal():
    S = t15.covariance_from_psd(t15.psd_vector(t15.TRUE_EXPONENT, t15.KNEE_BINS))
    Sinv = np.linalg.inv(S)
    U = t15.frozen_signal_basis()
    # See ARCCOS_FLOOR above: sqrt(1 - cos^2) cannot resolve coincidence below
    # sqrt(2*eps) ~ 2.1e-8, so 1e-8 was below the formula's floor.
    assert t15.sin_theta(U, U, Sinv) == pytest.approx(0.0, abs=ARCCOS_FLOOR)
    assert t15.sin_theta(U, 2.5 * U, Sinv) == pytest.approx(0.0, abs=ARCCOS_FLOOR)
