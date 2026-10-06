"""TR09 countercase: latents independent along a non-eigen square root of K change the leading-order
unrestricted coefficient through off-diagonal fourth cumulants; along the eigenbasis they do not."""
import numpy as np

from experiments.mlst2026.p1nl_dev_20261006.generator import FIXTURE_GEOMETRY, SparseLatentModel, rng
from experiments.mlst2026.p1nl_dev_20261006.tr09_first_order import core


def test_eigenbasis_reduces_to_cU_and_rotation_breaks_invariance():
    m1 = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=1.0)
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=0.1)
    A = np.diag(np.sqrt(m.mu))
    cU = core.unrestricted_coefficient(m.ell, m.k)
    assert np.isclose(core.unrestricted_coefficient_general(A, m.kappa, m.ell, m.k), cU)
    Q = np.linalg.qr(rng('tr09-test', 'rot').normal(size=(m.d, m.d)))[0]
    assert np.isclose(core.unrestricted_coefficient_general(A @ Q, m1.kappa, m.ell, m.k), cU)   # Gaussian: no change
    assert core.unrestricted_coefficient_general(A @ Q, m.kappa, m.ell, m.k) > cU + 1e-6


def test_countercase_variance_formula_by_monte_carlo():
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=0.1)
    g = rng('tr09-test', 'rot-mc')
    Q = np.linalg.qr(g.normal(size=(m.d, m.d)))[0]
    A = np.diag(np.sqrt(m.mu)) @ Q
    sparse = np.arange(m.d) < m.n_sparse
    R, N = 20000, 30
    z = g.normal(size=(R, N, m.d))
    z[..., sparse] *= (g.random(size=(R, N, sparse.sum())) < m.p) / np.sqrt(m.p)
    c = z @ A.T + g.normal(size=(R, N, m.d))                    # eigen-coordinates
    E01 = (c[..., 0] * c[..., m.k]).mean(1)                     # known-mean off-diagonal entry (retained 0, discarded k)
    kap = m.kappa
    want = (m.ell[0] * m.ell[m.k] + np.sum((kap - 3) * A[0] ** 2 * A[m.k] ** 2)) / N
    assert abs(E01.var() - want) < 0.05 * want
