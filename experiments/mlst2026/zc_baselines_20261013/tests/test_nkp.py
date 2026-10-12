"""Nearest Kronecker product: index convention, exactness on Kronecker input, PSD factors."""
import numpy as np
import pytest

from experiments.mlst2026.zc_baselines_20261013 import nkp
from experiments.mlst2026.t25_production_20260930 import science as t25
from experiments.mlst2026.t25_production_20260930 import protocol as t25p


def _psd(n, rng, rank=None):
    G = rng.normal(size=(n, rank or n + 2))
    return G @ G.T / G.shape[1]


def test_rearrangement_index_convention():
    C, T = 3, 4
    S = np.arange((C * T) ** 2, dtype=float).reshape(C * T, C * T)
    R = nkp.rearrange(S, C, T)
    for c in range(C):
        for t in range(T):
            for c2 in range(C):
                for t2 in range(T):
                    assert R[t * T + t2, c * C + c2] == S[c + C * t, c2 + C * t2]


@pytest.mark.parametrize('C,T', [(3, 4), (8, 32), (5, 2)])
def test_exact_on_kronecker_input(C, T):
    rng = np.random.default_rng(1)
    At, Ac = _psd(T, rng), _psd(C, rng)
    S = np.kron(At, Ac)
    K, info = nkp.nkp_matrix(S, C, T)
    assert np.max(np.abs(K - S)) <= 1e-12 * np.max(np.abs(S))
    assert info['rank1_residual_rel'] < 1e-12
    Ac2, At2, _ = nkp.nearest_kronecker(S, C, T)
    # factors are recovered up to the scale split
    s = np.trace(Ac2) / np.trace(Ac)
    assert np.allclose(Ac2, s * Ac, rtol=1e-10, atol=1e-12) and np.allclose(At2, At / s, rtol=1e-10, atol=1e-12)


def test_exact_on_t25_noise_covariance():
    c = t25p.BASE
    S = t25.covariance(c)
    K, _ = nkp.nkp_matrix(S, c['C'], c['T'])
    assert np.max(np.abs(K - S)) <= 1e-12


def test_exact_on_rank_deficient_kronecker():
    rng = np.random.default_rng(2)
    C, T = 4, 6
    S = np.kron(_psd(T, rng, rank=2), _psd(C, rng, rank=1))
    K, _ = nkp.nkp_matrix(S, C, T)
    assert np.max(np.abs(K - S)) <= 1e-11 * np.max(np.abs(S))


@pytest.mark.parametrize('seed', range(5))
def test_factors_psd_and_symmetric_on_general_input(seed):
    rng = np.random.default_rng(10 + seed)
    C, T = 3, 5
    S = _psd(C * T, rng, rank=4)                 # PSD, far from Kronecker
    Ac, At, info = nkp.nearest_kronecker(S, C, T)
    for F in (Ac, At):
        assert np.allclose(F, F.T, atol=0)
        assert np.linalg.eigvalsh(F).min() >= -1e-12 * np.abs(F).max()
    K = np.kron(At, Ac)
    assert np.linalg.eigvalsh(K).min() >= -1e-10
    assert np.isclose(np.linalg.norm(Ac), np.linalg.norm(At))
    # Frobenius optimality among Kronecker products (perturbation never improves before the PSD step is active)
    if info['psd_clip_channel'] == 0 and info['psd_clip_time'] == 0:
        base = np.linalg.norm(S - K)
        for _ in range(5):
            Ec, Et = 1e-3 * rng.normal(size=(C, C)), 1e-3 * rng.normal(size=(T, T))
            assert np.linalg.norm(S - np.kron(At + Et, Ac + Ec)) >= base - 1e-12


def test_shape_check():
    with pytest.raises(ValueError):
        nkp.rearrange(np.eye(10), 3, 4)
