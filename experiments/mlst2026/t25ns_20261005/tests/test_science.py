import numpy as np
import pytest
from numpy.testing import assert_allclose

from experiments.mlst2026.t25ns_20261005 import science as s
from experiments.mlst2026.t25ns_20261005.protocol import BASE, FIXTURE, BENCHMARK, T25_NAMESPACES, enumeration
from experiments.mlst2026.t25_production_20260930 import science as t25
from experiments.mlst2026.t25_production_20260930.protocol import FIXTURE as T25_FIXTURE, BASE as T25_BASE


@pytest.mark.parametrize('c', [BASE, FIXTURE])
@pytest.mark.parametrize('rho', [0.0, 0.25, 0.5, 0.9])
def test_sigma_is_spd_with_matched_trace_and_proved_bound(c, rho):
    sigma = s.covariance(c, rho)
    assert_allclose(sigma, sigma.T, atol=0)
    w = np.linalg.eigvalsh(sigma)
    sc = c['rho_c'] ** np.abs(np.arange(c['C'])[:, None] - np.arange(c['C']))
    st = c['rho_t'] ** np.abs(np.arange(c['T'])[:, None] - np.arange(c['T']))
    bound = (1 - rho) * np.linalg.eigvalsh(st)[0] * np.linalg.eigvalsh(sc)[0]
    assert w[0] > 0 and w[0] >= bound * (1 - 1e-12)
    assert_allclose(np.trace(sigma), np.trace(t25.covariance(c)), rtol=1e-14)
    assert_allclose(s.trace_scale(c, rho), 1.0, atol=1e-14)
    g = s.gain_vector(c)
    assert np.all(g > 0) and abs(g @ g - c['C']) < 1e-12
    assert np.linalg.eigvalsh(s.line_kernel(c))[0] > 0 and np.all(np.diag(s.line_kernel(c)) == 1)


def test_rho_zero_reproduces_t25_covariance_and_draws_bitwise():
    for c, tc in ((BASE, T25_BASE), (FIXTURE, T25_FIXTURE)):
        assert np.array_equal(s.covariance(c, 0.0), t25.covariance(tc))
    c = {**T25_FIXTURE, 'line': BASE['line'], 'gain': BASE['gain']}
    for seed, theta in ((900001, 0.3), (900002, 0.0)):
        mine, theirs = s.make_data(c, seed, 0.0, theta), t25.make_data(T25_FIXTURE, seed, theta)
        for a, b in zip(mine[1:], theirs[1:]):
            assert np.array_equal(a, b)


def test_shared_line_uses_time_major_kronecker_order():
    c = FIXTURE
    line = s.shared_line_part(c)
    g, S = s.gain_vector(c), s.line_kernel(c)
    for t in range(c['T']):
        for tp in range(c['T']):
            block = line[t * c['C']:(t + 1) * c['C'], tp * c['C']:(tp + 1) * c['C']]
            assert_allclose(block, S[t, tp] * np.outer(g, g), atol=1e-15)


def test_van_loan_pitsianis_on_planted_matrices():
    r = np.random.default_rng(5)
    C, T = 3, 4
    At, Cc = r.normal(size=(T, T)), r.normal(size=(C, C))
    Bt, Dc, rel = s.nearest_kronecker(np.kron(At, Cc), C, T)
    assert rel < 1e-12
    assert_allclose(np.kron(Bt, Dc), np.kron(At, Cc), atol=1e-12)
    # two Kronecker terms with orthonormal rearranged factors: rel = b / sqrt(a^2 + b^2)
    u = np.linalg.qr(r.normal(size=(T * T, 2)))[0]
    v = np.linalg.qr(r.normal(size=(C * C, 2)))[0]
    A = sum(w * np.kron(u[:, i].reshape(T, T), v[:, i].reshape(C, C)) for i, w in enumerate((3.0, 1.0)))
    assert_allclose(s.nearest_kronecker(A, C, T)[2], 1 / np.sqrt(10), rtol=1e-12)
    # index convention: Sigma_0 = St (x) Sc is recovered with St as the time factor
    Bt, Dc, rel = s.nearest_kronecker(s.covariance(BASE, 0.0), BASE['C'], BASE['T'])
    assert rel < 1e-12 and Bt.shape == (BASE['T'], BASE['T'])
    st = BASE['rho_t'] ** np.abs(np.arange(BASE['T'])[:, None] - np.arange(BASE['T']))
    assert_allclose(Bt / Bt[0, 0], st, atol=1e-10)
    assert s.nearest_kronecker(s.covariance(BASE, 0.5), BASE['C'], BASE['T'])[2] > 0.05


def test_moment_estimator_population_limits():
    lim0 = s.moment_limits(s.covariance(BASE, 0.0), BASE['C'], BASE['T'])
    assert_allclose(lim0['kron_unshrunk'], s.covariance(BASE, 0.0), atol=1e-12)
    lim5 = s.moment_limits(s.covariance(BASE, 0.5), BASE['C'], BASE['T'])
    assert np.linalg.norm(lim5['kron_unshrunk'] - s.covariance(BASE, 0.5)) / np.linalg.norm(s.covariance(BASE, 0.5)) > 0.1
    # the accepted T25 estimator converges to these limits (small-size consistency check)
    c = FIXTURE
    sigma = s.covariance(c, 0.5)
    L, _ = s.powers(sigma)
    pool = np.random.default_rng(11).normal(size=(400000, c['C'] * c['T'])) @ L
    est, _ = s.estimate(pool, c['C'], c['T'])
    lim = s.moment_limits(sigma, c['C'], c['T'])
    for k in ('full', 'kron'):
        assert np.linalg.norm(est[k] - lim[k]) / np.linalg.norm(lim[k]) < 0.01


def test_population_references_are_deterministic_and_ordered():
    refs = s.population_references(BASE)
    rows = {r['rho']: r for r in refs['covariance']}
    assert set(rows) == {0.0, 0.25, 0.5}
    assert rows[0.0]['nearest_kronecker_relative_distance'] < 1e-12
    assert rows[0.0]['kron_unshrunk_limit_whitened_error_op'] < 1e-10
    assert rows[0.5]['kron_unshrunk_limit_whitened_error_op'] > rows[0.25]['kron_unshrunk_limit_whitened_error_op'] > 0.5
    assert refs == s.population_references(BASE)
