"""TR10/TR11 checks: SURE bias signs and constants for zero, identity, affine and a nonlinear estimator."""
import numpy as np
import pytest

from experiments.mlst2026.p1nl_dev_20261006 import theory_checks as tc
from experiments.mlst2026.p1nl_dev_20261006.generator import rng

D = 6


def setup(seed):
    g = rng('theory-test', seed)
    A = g.normal(size=(D, D))
    Sigma = A @ A.T / D + 0.5 * np.eye(D)
    B = g.normal(size=(D, D)) * 0.3
    Sigma_hat = Sigma + B @ B.T - 0.2 * np.diag(np.diag(Sigma))       # misspecified, still SPD
    assert np.linalg.eigvalsh(Sigma_hat)[0] > 0
    return g, Sigma, Sigma_hat


def sparse_t(n, g):                       # non-Gaussian signal (Bernoulli-Gaussian); only the noise must be Gaussian
    return g.normal(size=(n, D)) * (g.random(size=(n, D)) < 0.2) * 3.0


@pytest.mark.parametrize('case', ['zero', 'identity', 'affine', 'soft_threshold'])
def test_sure_bias_formula(case):
    g, Sigma, Sigma_hat = setup(case)
    if case == 'zero':
        fn, jt = (lambda y: np.zeros_like(y)), (lambda y, M: np.zeros(len(y)))
    elif case == 'identity':
        fn, jt = (lambda y: y), (lambda y, M: np.full(len(y), np.trace(M)))
    elif case == 'affine':
        Aff = g.normal(size=(D, D)) * 0.4
        b = g.normal(size=D)
        fn, jt = (lambda y: y @ Aff.T + b), (lambda y, M: np.full(len(y), np.trace(M @ Aff)))
    else:
        lam = 1.0
        fn = lambda y: np.sign(y) * np.maximum(np.abs(y) - lam, 0)                 # noqa: E731
        jt = lambda y, M: ((np.abs(y) > lam) * np.diag(M)).sum(1)                  # J = diag(1[|y| > lam])  # noqa: E731
    r = tc.sure_bias_check(fn, jt, Sigma, Sigma_hat, sparse_t, 400000, g)
    assert abs(r['mc_bias'] - r['formula']) < 5 * r['paired_se'] + 1e-9, r
    if case == 'zero':
        assert r['formula'] == pytest.approx(r['trace_Sigma_r'] - D)
    if case == 'identity':
        assert r['formula'] == pytest.approx(D - r['trace_Sigma_r'])


def test_sure_is_unbiased_with_the_true_covariance():
    g, Sigma, _ = setup('true')
    fn = lambda y: np.tanh(y)                                                       # noqa: E731
    jt = lambda y, M: ((1 - np.tanh(y) ** 2) * np.diag(M)).sum(1)                   # noqa: E731
    r = tc.sure_bias_check(fn, jt, Sigma, Sigma, sparse_t, 400000, g)
    assert abs(r['formula']) < 1e-9 and abs(r['mc_bias']) < 5 * r['mc_se']


def test_price_of_linearity_limits():
    rows = tc.price_of_linearity(ps=(1.0, 0.1), ks=(3,))
    g1, g01 = rows
    assert abs(g1['price_no_bottleneck']) < 1e-12               # Gaussian latents: Bayes = full Wiener
    assert g01['price_no_bottleneck'] > 0
    assert g1['gain_bound_bottleneck'] > 0                       # still positive at p = 1 (shrinkage + tail)
    assert g01['gain_bound_sparse_part'] + g01['gain_bound_tail_part'] == pytest.approx(g01['gain_bound_bottleneck'])
