"""TR09 first-order checks: chart derivatives, stationarity, special cases, moment formulas and the
first-order expectation identity (development namespace paper1-tr09-dev / test streams)."""
import numpy as np
import pytest

from experiments.mlst2026.p1nl_dev_20261006.generator import FIXTURE_GEOMETRY, SparseLatentModel, rng
from experiments.mlst2026.p1nl_dev_20261006.tr09_first_order import core, moments


@pytest.fixture(scope='module')
def fixture_point():
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=1.0)
    pt, dist = core.population_product_optimum(m, rng('tr09-test', 'pop'), n_random=6)
    return m, pt, dist


def fd_check(point, S, h=1e-4, idx=None):
    S4 = core.reshape4(S, point.C, point.T)
    f = lambda x: point.exact(S4, x)                                   # noqa: E731
    n = point.n
    E = np.eye(n)
    g = np.array([(f(h * e) - f(-h * e)) / (2 * h) for e in E])
    H = point.hessian(S4)
    idx = range(n) if idx is None else idx
    herr = 0.0
    for i in idx:
        for j in idx:
            a, b = E[i] * h, E[j] * h
            fd = (f(a + b) - f(a - b) - f(-a + b) + f(-a - b)) / (4 * h * h)
            herr = max(herr, abs(fd - H[i, j]))
    return np.max(np.abs(g - point.gradient(S4))), herr


def test_gradient_and_hessian_match_finite_differences():
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.4, p=1.0)
    g = rng('tr09-test', 'fd')
    A = g.normal(size=(m.d, m.d))
    S = A @ A.T / m.d + m.M                                           # generic symmetric matrix
    pt = core.ProductPoint(core.orth(g.normal(size=(m.C, m.kc))), core.orth(g.normal(size=(m.T, m.kt))))
    ge, he = fd_check(pt, S)
    assert ge < 1e-7 and he < 1e-5


def test_hessian_on_production_geometry_subset():
    m = SparseLatentModel(theta=0.25, p=1.0)
    g = rng('tr09-test', 'fd-prod')
    pt = core.ProductPoint(core.orth(g.normal(size=(m.C, m.kc))), core.orth(g.normal(size=(m.T, m.kt))))
    idx = list(range(0, 12, 3)) + list(range(12, pt.n, 17))          # both blocks and the cross block
    ge, he = fd_check(pt, m.M, idx=idx)
    assert ge < 1e-7 and he < 1e-5


def test_population_maximiser_is_strict_unique_and_stationary(fixture_point):
    m, pt, dist = fixture_point
    assert all(d['status'] == 'stationary' for d in dist)
    assert max(d['gap_to_best'] for d in dist) < 1e-10                # all starts reach one maximum
    fo = core.first_order(m, pt)
    assert fo['gradient_norm'] < 1e-10 and fo['H_min_eig'] > 0
    assert np.all(fo['w'] >= -1e-15)                                   # w_a >= 0 since H > 0


def test_theta_zero_has_zero_weights_and_zero_cost():
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.0, p=0.1)
    pt, _ = core.population_product_optimum(m, rng('tr09-test', 'pop0'), n_random=4)
    fo = core.first_order(m, pt)
    s = core.summary(m, pt, fo)
    assert abs(s['a']) < 1e-10 and max(abs(x) for x in s['w_sparse']) < 1e-12 and abs(s['K']) < 1e-10


def test_unrestricted_tangent_diagonals_vanish():
    """Unrestricted special case: the tangent directions v_j v_i^T + v_i v_j^T (i retained, j discarded)
    have zero diagonal in the eigenbasis, so no fourth-moment term enters at first order."""
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=0.1)
    V, k = m.V, m.k
    for i in range(k):
        for j in range(k, m.d):
            G = np.outer(V[:, j], V[:, i]) + np.outer(V[:, i], V[:, j])
            assert np.max(np.abs(np.diag(V.T @ G @ V))) < 1e-12


def test_sample_covariance_moment_formulas():
    lam = np.array([5.0, 2.0, 0.5])
    kappa = np.array([30.0, 3.0, 10.0])        # p = 0.1, Gaussian, p = 0.3
    p = 3.0 / kappa
    N, R = 40, 40000
    g = rng('tr09-test', 'moments')
    z = g.normal(size=(R, N, 3))
    z *= (g.random(size=(R, N, 3)) < p) / np.sqrt(p)
    y = z * np.sqrt(lam) + g.normal(size=(R, N, 3))
    S0 = np.einsum('rna,rnb->rab', y, y) / N
    yb = y.mean(1)
    Sc = S0 - np.einsum('ra,rb->rab', yb, yb)
    for est, formula in ((S0, moments.known_mean), (Sc, moments.centred)):
        V = formula(lam, kappa, N)
        emp = est.var(0)
        se = np.sqrt(2.0 / R) * emp * np.sqrt(1 + kappa.max())            # generous SE scale
        assert np.all(np.abs(emp - V) < 6 * se + 1e-12), (emp, V)
    # distinct entries uncorrelated
    c = np.cov(Sc[:, 0, 1], Sc[:, 0, 0])[0, 1] / np.sqrt(Sc[:, 0, 1].var() * Sc[:, 0, 0].var())
    assert abs(c) < 0.03
    # ... but not independent: E[E_01^2 E_00] = ell_1 (mu4_0 - ell_0^2) / N^2
    E = S0 - np.diag(lam + 1)
    emp3 = np.mean(E[:, 0, 1] ** 2 * E[:, 0, 0])
    want = moments.third_cross(lam, kappa, N, 0, 1)
    assert abs(emp3 - want) < 0.15 * want


def test_first_order_expectation_identity_and_kurtosis_weights():
    """E[q_F] = 1/2 tr(H^-1 Cov g) = (c_F + K(p)) / N up to O(1/N^2), checked by paired Monte Carlo
    across p at a geometry with non-negligible weights (fixture geometry, theta = 0.6)."""
    base = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.6, p=1.0)
    pt, _ = core.population_product_optimum(base, rng('tr09-test', 'pop6'), n_random=6)
    fo = core.first_order(base, pt)
    N, R = 400, 3000
    res = {}
    for p in (1.0, 0.1):
        m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.6, p=p)
        q = []
        for r in range(R):
            y, _, _ = m.sample_keyed(N, 'tr09-test', 'q', r)
            yc = y - y.mean(0)
            E4 = core.reshape4(yc.T @ yc / N - m.M, m.C, m.T)
            g = pt.gradient(E4)
            q.append(0.5 * g @ fo['Hinv'] @ g)
        q = np.array(q) * N
        res[p] = (q, fo['cF_gaussian'] + core.kurtosis_term(m, fo['w']))
    for p, (q, want) in res.items():
        assert abs(q.mean() - want) < 4 * q.std() / np.sqrt(R) + 0.02 * want, (p, q.mean(), want)
    d = res[0.1][0] - res[1.0][0]                                    # paired across p
    assert abs(d.mean() - (res[0.1][1] - res[1.0][1])) < 4 * d.std() / np.sqrt(R) + 0.02 * abs(res[0.1][1] - res[1.0][1])
    assert res[0.1][1] - res[1.0][1] > 0                              # K(p) >= 0 raises the product excess


def test_partial_second_order_kurtosis_coefficient_scales_with_excess_kurtosis():
    d = {p: core.unrestricted_kurtosis_second_order(SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=p))
         for p in (1.0, 0.3, 0.1)}
    assert d[1.0] == 0.0 and d[0.1] > 0
    assert d[0.1] / d[0.3] == pytest.approx((3 / 0.1 - 3) / (3 / 0.3 - 3))
