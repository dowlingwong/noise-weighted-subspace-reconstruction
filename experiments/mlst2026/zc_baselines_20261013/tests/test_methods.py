"""Method wrappers: fold determinism, CV determinism, lambda = 0 is U, N*-rule logic, T25 equivalence."""
import numpy as np
import pytest

from experiments.mlst2026.zc_baselines_20261013 import methods, design
from experiments.mlst2026.t25_production_20260930 import science as t25
from experiments.mlst2026.t25_production_20260930 import protocol as t25p

FIX = dict(t25p.FIXTURE, namespace='zc-test-methods')
RECIPE = {'max_iter': 200, 'gradient_tolerance': 1e-8}


@pytest.fixture(scope='module')
def cell():
    g, sigma, L, Li, pool, training = t25.make_data(FIX, 4242, 0.3)
    x = training[:80]
    W = np.linalg.inv(sigma)
    return g, sigma, x, W, (g.C, g.T, g.k_c, g.k_t)


def test_folds_deterministic_partition():
    a = methods.folds(1250, ('ns', 1, 0.25, 1250, 'cv-folds'), 5)
    b = methods.folds(1250, ('ns', 1, 0.25, 1250, 'cv-folds'), 5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    allidx = np.sort(np.concatenate(a))
    assert np.array_equal(allidx, np.arange(1250)) and all(len(f) == 250 for f in a)
    c = methods.folds(1250, ('ns', 2, 0.25, 1250, 'cv-folds'), 5)
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_fit_wrappers_match_t25_recipe(cell):
    g, sigma, x, W, dims = cell
    _, mu, S = t25.moments(x, len(x))
    U = methods.fit_U(x, W, dims)
    assert np.array_equal(U['B'], t25.unrestricted(W, S, g.k))
    key = ('zc-test-methods', 4242, 0.3, 80)
    F = methods.fit_F(x, W, dims, key, RECIPE)
    B, _, _ = t25.structured(W, S, t25.starts(S, g.C, g.T, g.k_c, g.k_t, key), RECIPE['max_iter'], RECIPE['gradient_tolerance'])
    assert np.array_equal(F['B'], B)
    A = t25.map_for(B, W)
    assert np.array_equal(F['A'], A) and np.array_equal(F['b'], (np.eye(g.d) - A) @ mu)


def test_cv_select_deterministic_and_refit(cell):
    g, sigma, x, W, dims = cell
    fi = methods.folds(len(x), ('zc-test', 'f'), 5)
    key = ('zc-test-methods', 4242, 0.3, 80)
    full = {'U': methods.fit_U(x, W, dims), 'F': methods.fit_F(x, W, dims, key, RECIPE)}
    a = methods.cv_select(x, W, dims, fi, RECIPE, key, full=full)
    b = methods.cv_select(x, W, dims, fi, RECIPE, key)               # refits the choice itself
    assert a['choice'] == b['choice'] and a['fold_scores'] == b['fold_scores']
    assert np.array_equal(a['fit']['A'], b['fit']['A'])
    assert a['fit'] is full[a['choice']]
    for c in ('U', 'F'):
        assert np.isclose(a['mean_scores'][c], np.mean(a['fold_scores'][c]))   # equal fold sizes
    assert a['choice'] == ('F' if a['mean_scores']['F'] < a['mean_scores']['U'] else 'U')


def test_heldout_residual_formula(cell):
    g, sigma, x, W, dims = cell
    U = methods.fit_U(x, W, dims)
    r = methods.heldout_residuals(U['A'], U['b'], x[:7], W)
    for i in range(7):
        e = x[i] - (U['A'] @ x[i] + U['b'])
        assert np.isclose(r[i], e @ W @ e)


def test_shr_lambda0_is_U_bitwise(cell):
    g, sigma, x, W, dims = cell
    fi = methods.folds(len(x), ('zc-test', 'f'), 5)
    shr = methods.fit_shr(x, W, dims, fi, [0.0, 0.5, 1.0], return_path=True)
    U = methods.fit_U(x, W, dims)
    p0 = shr['path'][0]
    assert np.array_equal(p0['B'], U['B']) and np.array_equal(p0['A'], U['A']) and np.array_equal(p0['b'], U['b'])
    only0 = methods.fit_shr(x, W, dims, fi, [0.0])
    assert only0['lambda_hat'] == 0.0 and np.array_equal(only0['fit']['A'], U['A'])
    # lambda_hat is the first minimiser of the mean fold score
    assert shr['index_hat'] == int(np.argmin(shr['mean_scores']))
    # held-out U score at lambda = 0 equals the CV U score on the same folds
    cv = methods.cv_select(x, W, dims, fi, RECIPE, ('k',), full={'U': U, 'F': U})
    assert np.isclose(shr['mean_scores'][0], cv['mean_scores']['U'], rtol=0, atol=1e-12)


def test_shr_lambda1_is_rank_k_projector_not_nfpa(cell):
    g, sigma, x, W, dims = cell
    fi = methods.folds(len(x), ('zc-test', 'f'), 5)
    p1 = methods.fit_shr(x, W, dims, fi, [1.0])['fit']
    A = p1['A']
    assert np.allclose(A @ A, A, atol=1e-10) and np.isclose(np.trace(A), g.k)


def test_nstar_rule_logic():
    table = design.build_protocol({'files': {}, 'in_accepted_t25_closure': []})['nstar']['table']
    assert methods.nstar_rule(0.0, 3200, table)['choice'] == 'F'                 # theta = 0 -> product
    assert all(methods.nstar_rule(0.12, N, table)['choice'] == 'F' for N in (400, 3200))
    assert [methods.nstar_rule(0.25, N, table)['choice'] for N in (400, 800, 1000, 1250, 1600, 3200)] == \
        ['F', 'F', 'F', 'U', 'U', 'U']
    assert [methods.nstar_rule(0.4, N, table)['choice'] for N in (400, 800)] == ['F', 'U']
    with pytest.raises(KeyError):
        methods.nstar_rule(0.3, 400, table)
