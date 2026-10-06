"""TR07 references, baselines and gate logic (no torch)."""
import numpy as np
import pytest

from experiments.mlst2026.p1nl_dev_20261006 import bayes
from experiments.mlst2026.p1nl_dev_20261006.generator import FIXTURE_GEOMETRY, SparseLatentModel, rng
from experiments.mlst2026.p1nl_dev_20261006.tr07_pilot import baselines, diagnostics, gate
from experiments.mlst2026.p1nl_dev_20261006.tr07_pilot.config import PILOT


@pytest.mark.parametrize('p', [0.1, 0.3, 1.0])
def test_factorised_posterior_matches_brute_force(p):
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=p, n_sparse=4)
    Y, _, _ = m.sample_keyed(64, 'tr07-test', 'bf', p)
    Y[:3] *= 6.0                                                   # include far-tail inputs
    a, b = bayes.posterior_mean(m, Y), bayes.brute_force_posterior_mean(m, Y)
    assert np.max(np.abs(a - b)) < 1e-9


def test_scalar_posterior_limits_and_stability():
    c = np.array([-200.0, -40.0, -1.0, 0.0, 1.0, 40.0, 200.0])
    assert np.allclose(bayes.scalar_posterior_mean(c, 3.0, 1.0), 0.75 * c)          # p = 1: Wiener
    assert np.all(bayes.scalar_posterior_mean(c, 0.0, 0.1) == 0)                    # zero signal
    pi = bayes.active_probability(c, 9.0, 1e-6)                                     # extreme but supported p
    assert np.all(np.isfinite(pi)) and pi[0] > 0.999 and pi[3] < 1e-5
    m = bayes.scalar_posterior_mean(c, 9.0, 0.1)
    assert np.all(np.isfinite(m)) and abs(m[-1] / c[-1] - (90 / 91)) < 1e-12


def test_quadrature_risk_limits_and_monte_carlo_agreement():
    assert bayes.sparse_mmse(4.0, 1.0)[0] == pytest.approx(0.8)
    assert bayes.sparse_mmse(0.0, 0.1)[0] == 0.0
    for lam, p in [(12.0, 0.1), (7.0, 0.3), (0.5, 0.1)]:
        r, info = bayes.sparse_mmse(lam, p)
        assert info['error_bound'] < 1e-10
        assert 0 < r < lam / (lam + 1)                    # Bayes beats Wiener strictly for p < 1
        g = rng('tr07-test', 'mmse', lam, p)
        n = 400000
        z = g.normal(size=n) * (g.random(n) < p) / np.sqrt(p)
        c = np.sqrt(lam) * z + g.normal(size=n)
        e = (bayes.scalar_posterior_mean(c, lam, p) - np.sqrt(lam) * z) ** 2
        assert abs(e.mean() - r) < 5 * e.std() / np.sqrt(n)


def test_linear_references_match_affine_risk_and_shrinkage_is_rank_k_optimal():
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=0.1, n_sparse=4)
    k = 2
    ref = bayes.linear_references(m.mu, k)
    U = m.V[:, :k]
    Ap = U @ U.T
    gains = m.mu[:k] / (m.mu[:k] + 1)
    As = (U * gains) @ U.T
    Aw = (m.V * (m.mu / (m.mu + 1))) @ m.V.T
    z = np.zeros(m.d)
    assert baselines.affine_risk(Ap, z, m.K) == pytest.approx(ref['projector'])
    assert baselines.affine_risk(As, z, m.K) == pytest.approx(ref['shrinkage'])
    assert baselines.affine_risk(Aw, z, m.K) == pytest.approx(ref['full_wiener'])
    assert np.allclose(diagnostics.affine_per_coordinate(As, z, m).sum(), ref['shrinkage'])
    g = rng('tr07-test', 'opt')
    for _ in range(50):                              # random rank-k perturbations never improve on shrinkage
        B = As + 1e-2 * g.normal(size=(m.d, k)) @ g.normal(size=(k, m.d))
        u, s, vt = np.linalg.svd(B)
        Bk = (u[:, :k] * s[:k]) @ vt[:k]
        assert baselines.affine_risk(Bk, z, m.K) >= ref['shrinkage'] - 1e-12
    r = bayes.references(m, k)
    assert r['bayes'] <= r['full_wiener'] <= r['shrinkage'] <= r['projector'] + 1e-12


def test_fitted_baselines_converge_and_handle_small_eigenvalues():
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=0.1, n_sparse=4)
    ref = bayes.linear_references(m.mu, 2)
    Y, _, _ = m.sample_keyed(400000, 'tr07-test', 'conv')
    Ap, bp, _ = baselines.fitted_projector(Y, 2)
    As, bs, _ = baselines.fitted_shrinkage(Y, 2)
    assert baselines.affine_risk(Ap, bp, m.K) == pytest.approx(ref['projector'], rel=2e-3)
    assert baselines.affine_risk(As, bs, m.K) == pytest.approx(ref['shrinkage'], rel=2e-3)
    Ysmall = rng('tr07-test', 'small').normal(size=(50, 6)) * 0.3      # every sample eigenvalue < 1
    A, b, info = baselines.fitted_shrinkage(Ysmall, 3)
    assert info['zero_gain_count'] == 3 and np.all(A == 0) and np.all(np.isfinite(b))


def test_literal_gap_rule_is_structurally_unattainable_on_the_t25_geometry():
    m = SparseLatentModel(theta=0.25, p=0.1)
    refs = bayes.references(m, 3)
    ceil = gate.ceiling_literal(refs, m)
    assert ceil < 0.25                                      # documented: about 0.240
    assert refs['gap_shrinkage_to_bayes'] == pytest.approx(107.0, abs=0.5)


def fake_cells(d_proj, d_shrink, R_ae=150.0, mcse=0.05, seeds=(1, 2, 3), sparse_num=20.0):
    return [{'seed': s, 'ok': True, 'R_AE': R_ae + i * 0.01, 'mcse': mcse, 'R_proj': R_ae - dp, 'R_shrink': R_ae - ds,
             'sparse_num': sparse_num, 'sparse_num_mcse': 0.05}
            for i, (s, dp, ds) in enumerate(zip(seeds, d_proj, d_shrink))]


def test_gate_logic():
    m = SparseLatentModel(theta=0.25, p=0.1)
    refs = bayes.references(m, 3)
    cfg = PILOT
    good = fake_cells([-20, -21, -19], [-20, -20.5, -19.5], R_ae=152.0)
    v = gate.evaluate(good, refs, m, cfg)
    assert v['G1']['projector']['holds'] and v['G1']['shrinkage']['holds']
    assert v['G2']['sparse_oracle']['holds'] and not v['G2']['literal']['holds']
    assert v['verdict'].startswith('PASS')
    spread = fake_cells([-20, -1, -0.5], [-20, -1, -0.5], R_ae=152.0)          # one seed barely better
    assert gate.evaluate(spread, refs, m, cfg)['verdict'].startswith('FAIL')
    worse = fake_cells([5, 5, 5], [5, 5, 5], R_ae=180.0)
    assert gate.evaluate(worse, refs, m, cfg)['verdict'].startswith('FAIL')
    assert gate.evaluate(good[:2], refs, m, cfg)['verdict'].startswith('FAIL')   # missing seed
    bad = [dict(c) for c in good]
    bad[1]['ok'] = False
    assert gate.evaluate(bad, refs, m, cfg)['verdict'].startswith('FAIL')
    tiny = {**cfg, 'gate': {**cfg['gate'], 'min_gap': 1e9}}
    v = gate.evaluate(good, refs, m, tiny)
    assert v['G2']['sparse_oracle']['status'] == 'undefined' and v['verdict'].startswith('FAIL')


def test_diagnostics_on_known_maps():
    m = SparseLatentModel(geometry=FIXTURE_GEOMETRY, theta=0.25, p=0.1, n_sparse=4)
    Y, S, Z = m.sample_keyed(20000, 'tr07-test', 'diag')
    k = 2
    U = m.V[:, :k]
    Shat = Y @ U @ U.T
    assert abs(diagnostics.leakage(Shat, k)) < 1e-12
    I = diagnostics.interference(lambda X: bayes.posterior_mean(m, X), m, [-1, -0.5, 0.5, 1])
    assert np.allclose(I - np.diag(np.diag(I)), 0, atol=1e-12)          # factorised Bayes: no interference
    rec = diagnostics.recovery(bayes.posterior_mean(m, Y), None, Z[:, :4], m)
    assert all(r > 0.5 for r in rec['output_r2'])
    ev = diagnostics.mc_risk(Shat, S, m)
    exact = baselines.affine_risk(U @ U.T, np.zeros(m.d), m.K)
    assert abs(ev['risk'] - exact) < 5 * ev['mcse']
