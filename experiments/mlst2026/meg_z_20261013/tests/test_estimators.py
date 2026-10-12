"""Estimators: Ledoit-Wolf formula, flip-flop MLE (exactness, recovery, monotonicity), NKP, record layout."""
import numpy as np
import pytest

from experiments.mlst2026.meg_z_20261013 import common, estimators as E, fixture


def test_vec_layout_matches_t25_convention():
    C, T = 3, 4
    Z = np.arange(C * T, dtype=float).reshape(C, T)
    x = common.vec(Z)
    assert np.array_equal(x, Z.reshape(-1, order='F'))                     # index c + C t
    assert np.array_equal(common.unvec(x, C, T), Z)
    A, B = np.random.default_rng(0).normal(size=(C, C)), np.random.default_rng(1).normal(size=(T, T))
    assert np.allclose(common.vec(A @ Z @ B), np.kron(B.T, A) @ x)
    keep = [0, 2]
    idx = common.channel_indices(keep, C, T)
    assert np.array_equal(x[idx], common.vec(Z[keep]))


def test_rng_matches_t25_construction():
    from experiments.mlst2026.t25_production_20260930 import science as t25
    assert np.array_equal(common.rng('a', 1, 0.25).normal(size=5), t25.rng('a', 1, 0.25).normal(size=5))


def test_ledoit_wolf_matches_sklearn_and_streaming():
    X = np.random.default_rng(3).normal(size=(80, 12)) @ np.diag(np.linspace(0.5, 3, 12)) + 0.3
    cov, info = E.ledoit_wolf_full(X)
    sk = pytest.importorskip('sklearn.covariance')
    ref, shr = sk.ledoit_wolf(X)
    assert abs(info['shrinkage'] - shr) < 1e-12
    assert np.allclose(cov, ref, atol=1e-12)
    st = E.moment_stats_streamed(lambda: (X[i:i + 7] for i in range(0, len(X), 7)))
    st1 = E.moment_stats(X)
    assert st.n == st1.n and np.allclose(st.S, st1.S) and abs(st.sum_q4 - st1.sum_q4) < 1e-8 * st1.sum_q4


def test_flipflop_exact_on_kronecker_and_recovers_truth():
    C, T = 4, 6
    cov, St, Sc = fixture.fixture_covariance(C, T, 'kron')
    St2, Sc2, info = E.kronecker_flipflop_from_S(cov, C, T)
    assert info['status'] == 'converged' and info['monotone']
    assert np.allclose(np.kron(St2, Sc2), cov, atol=1e-9)
    assert abs(np.trace(Sc2) - C) < 1e-12
    assert abs(info['trace_whitened'] - C * T) < 1e-8                      # MLE stationarity: tr(Sigma^-1 S) = d
    X = np.random.default_rng(5).normal(size=(20000, C * T)) @ np.linalg.cholesky(cov).T
    est, _, _, _ = E.kronecker_flipflop(X, C, T)
    assert E.whitened_error(est, cov)['operator'] < 0.06


def test_flipflop_monotone_on_nonkronecker():
    C, T = 4, 6
    cov, _, _ = fixture.fixture_covariance(C, T, 'nonkron')
    _, _, info = E.kronecker_flipflop_from_S(cov, C, T)
    assert info['monotone'] and info['status'] == 'converged'


def test_nearest_kronecker():
    C, T = 4, 6
    cov, St, Sc = fixture.fixture_covariance(C, T, 'kron')
    Bt, Bc, rel = E.nearest_kronecker(cov, C, T)
    assert rel < 1e-12 and np.allclose(np.kron(Bt, Bc), cov)
    assert E.is_kronecker(cov, C, T)
    nk, _, _ = fixture.fixture_covariance(C, T, 'nonkron')
    assert E.nearest_kronecker(nk, C, T)[2] > 1e-3 and not E.is_kronecker(nk, C, T)


def test_reference_metric_kinds():
    C, T = 4, 6
    cov, _, _ = fixture.fixture_covariance(C, T, 'kron')
    X = np.random.default_rng(2).normal(size=(3000, C * T)) @ np.linalg.cholesky(cov).T
    st = E.moment_stats(X)
    k = E.reference_metric(st, C, T, 'kronecker_flipflop')
    lw = E.reference_metric(st, C, T, 'ledoit_wolf_full')
    assert E.is_kronecker(k['cov'], C, T) and not E.is_kronecker(lw['cov'], C, T)
    with pytest.raises(ValueError):
        E.reference_metric(st, C, T, 'oracle')


def test_ledoit_wolf_effective_sample_count():
    """F14: with n_eff < n (overlapping windows) the variance term grows and the shrinkage increases;
    n_eff = n is sklearn's formula exactly."""
    X = np.random.default_rng(3).normal(size=(400, 30)) @ np.diag(np.linspace(0.5, 3, 30))
    st = E.moment_stats(X)
    _, i1 = E.ledoit_wolf_from_stats(st)
    st.n_eff = st.n
    _, i2 = E.ledoit_wolf_from_stats(st)
    st.n_eff = 40
    _, i3 = E.ledoit_wolf_from_stats(st)
    assert i1['shrinkage'] == i2['shrinkage'] and i3['shrinkage'] > i1['shrinkage'] and i3['n_eff'] == 40
