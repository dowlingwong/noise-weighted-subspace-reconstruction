import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from experiments.mlst2026.x3_nonlinear_metric_20261005 import science as s
from experiments.mlst2026.x3_nonlinear_metric_20261005.models import FlatAE, hutchinson_divergence, sure_terms
from experiments.mlst2026.x3_nonlinear_metric_20261005.protocol import BASE, FIXTURE, BENCHMARK, enumeration
from experiments.mlst2026.t25_production_20260930 import science as t25
from experiments.mlst2026.t25_production_20260930.protocol import FIXTURE as T25_FIXTURE
# Registered namespaces of T25 and X2, written out so that this test does not import the X2 package
T25_NAMESPACES = {'paper1-t25-v1/confirm': list(range(202609290100, 202609290124)),
                  'paper1-t25-v1/fixture': [900001, 900002],
                  'paper1-t25-v1/benchmark-20260930': [990001],
                  'paper1-t25-v1/solver-dev-20260930': [777]}
X2_NAMESPACES = {'paper1-t25ns-v1/confirm': list(range(202610050100, 202610050124)),
                 'paper1-t25ns-v1/fixture': [910001, 910002, 910003],
                 'paper1-t25ns-v1/benchmark-20261005': [991001]}

torch = s.import_torch(FIXTURE)


def test_architecture_matches_registration_and_shares_init():
    torch.manual_seed(7)
    lin = FlatAE(256, 6, BASE['hidden'], None)
    torch.manual_seed(7)
    silu = FlatAE(256, 6, BASE['hidden'], 'silu')
    dims = [m.out_features for m in lin.encoder if isinstance(m, torch.nn.Linear)] + \
           [m.out_features for m in lin.decoder if isinstance(m, torch.nn.Linear)]
    assert dims == [128, 32, 6, 32, 128, 256]
    assert [type(m).__name__ for m in silu.encoder] == ['Linear', 'SiLU', 'Linear', 'SiLU', 'Linear']
    assert [type(m).__name__ for m in silu.decoder] == ['Linear', 'SiLU', 'Linear', 'SiLU', 'Linear']
    a_vals, b_vals = list(lin.state_dict().values()), list(silu.state_dict().values())
    assert len(a_vals) == len(b_vals) == 12          # names differ (SiLU modules shift indices), values do not
    for a, b in zip(a_vals, b_vals):
        assert torch.equal(a, b)


def test_torch_version_is_enforced():
    with pytest.raises(ValueError, match='registered'):
        s.import_torch({**FIXTURE, 'torch_version': '0.0.0'})


def test_hutchinson_is_exact_for_diagonal_maps_and_sure_is_unbiased():
    r = np.random.default_rng(3)
    d = 5
    A = np.diag(r.uniform(0.1, 0.9, d))
    At = torch.from_numpy(A)
    x = torch.from_numpy(r.normal(size=(7, d)))
    div = hutchinson_divergence(lambda X: X @ At.T, x, 3, torch.Generator().manual_seed(1))
    assert_allclose(div.numpy(), np.trace(A), rtol=1e-14)
    # SURE vs exact whitened clean risk for a fixed linear map, white noise
    B = r.normal(size=(d, d)) * 0.3
    Bt = torch.from_numpy(B)
    n = 200000
    clean = r.normal(size=(n, d)) * np.array([3, 2, 1, .5, .2])
    noisy = clean + r.normal(size=(n, d))
    st = sure_terms(lambda X: X @ Bt.T, torch.from_numpy(noisy), torch.eye(d, dtype=torch.float64), 1,
                    torch.Generator().manual_seed(2))
    truth = float((((noisy @ B.T) - clean) ** 2).sum(1).mean())
    assert abs(float(st['sure'].mean()) - truth) < 0.02 * truth


def test_closed_forms_are_pca_and_cw_pca():
    c = FIXTURE
    g, sigma, L, Li, train, ec, en = s.make_data(c, 920001)
    x, mu, S = s.moments(train, 80)
    for W in (np.eye(g.d), Li @ Li):
        A, b = s.closed_form(c, S, mu, W, c['k'])
        assert_allclose(A @ A, A, atol=1e-10)
        assert_allclose(W @ A, A.T @ W, atol=1e-10)           # W-orthogonal projector
        assert np.linalg.matrix_rank(A) == c['k']
        B = t25.unrestricted(W, S, c['k'])
        assert_allclose(A, t25.map_for(B, W), atol=1e-12)
    A, b = s.closed_form(c, S, mu, np.eye(g.d), c['k'])
    U = np.linalg.eigh(S)[1][:, -c['k']:]
    assert_allclose(A, U @ U.T, atol=1e-10)
    assert_allclose(b, mu - U @ U.T @ mu, atol=1e-10)
    # independent CW-PCA: Sigma^{1/2} U U^T Sigma^{-1/2}, U the top-k eigenvectors of the whitened covariance
    A, b = s.closed_form(c, S, mu, Li @ Li, c['k'])
    Uw = np.linalg.eigh(Li @ S @ Li)[1][:, -c['k']:]
    want = L @ Uw @ Uw.T @ Li
    assert_allclose(A, want, atol=1e-10)
    assert_allclose(b, mu - want @ mu, atol=1e-10)


def test_training_draws_reuse_t25_streams_and_evaluation_is_separate():
    c = {**T25_FIXTURE, 'k': 4, 'evaluation': {'records': 8, 'hutchinson_probes': 4}, 'theta': 0.3}
    mine = s.make_data(c, 900001)
    theirs = t25.make_data(T25_FIXTURE, 900001, 0.3)
    assert np.array_equal(mine[4], theirs[5])                    # training records, bitwise
    assert np.array_equal(mine[1], theirs[1])                    # covariance
    assert not np.allclose(mine[5][:4], mine[4][:4])


def test_namespaces_and_seeds_are_disjoint():
    mine = {BASE['namespace']: BASE['seeds'], FIXTURE['namespace']: FIXTURE['seeds'], BENCHMARK['namespace']: BENCHMARK['seeds']}
    others = {**T25_NAMESPACES, **X2_NAMESPACES}
    assert not set(mine) & set(others)
    a = [x for v in mine.values() for x in v]
    b = [x for v in others.values() for x in v]
    assert len(a) == len(set(a)) and not set(a) & set(b)


def test_sure_and_divergence_with_correlated_sigma():
    """SURE under the registered Sigma^-1 metric with correlated noise, for affine maps (exact risk and
    trace) and for a trained SiLU map (Monte-Carlo truth on the same records)."""
    c = {**FIXTURE, 'evaluation': {'records': 200000, 'hutchinson_probes': 1}}
    g, sigma, L, Li, train, ec, en = s.make_data(c, 920002)
    assert np.abs(sigma - np.diag(np.diag(sigma))).max() > 0.05 * np.diag(sigma).max()    # genuinely correlated
    x, mu, S = s.moments(train, 64)
    Cs = L @ g.K(c['theta']) @ L
    for W in (np.eye(g.d), Li @ Li):
        A, b = s.closed_form(c, S, mu, W, c['k'])
        At, bt = s.tensor(torch, A), s.tensor(torch, b)
        ev = s.evaluate(c, torch, 920002, lambda X: X @ At.T + bt, ec, en, Li)
        exact = s.risks(A, b, Cs, sigma, Li @ Li)
        assert abs(ev['sure'] - exact['clean']) < 0.02 * exact['clean']
        assert abs(ev['divergence'] - np.trace(A)) < 0.03 * np.trace(A)
    model, _ = s.train_deep(c, torch, train[:64], train[64:80], Li, 'silu', 4242)
    model.eval()
    ev = s.evaluate(c, torch, 920002, model, ec, en, Li)
    assert abs(ev['sure'] - ev['clean_whitened']) < 0.02 * ev['clean_whitened']


def test_benchmark_path_computes_no_endpoint(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError('an endpoint was evaluated')
    monkeypatch.setattr(s, 'evaluate', refuse)
    monkeypatch.setattr(s, 'evaluate_deep', refuse)
    monkeypatch.setattr(s, 'risks', refuse)
    record, arrays = s.fit_cell(FIXTURE, 920003, 40, torch, evaluate_models=False)
    assert record['models'] == {} and len(record['training']) == 4
    assert {k for k in arrays if k.startswith('tied_linear__')} == {
        f'tied_linear__{loss}__{p}' for loss in FIXTURE['losses'] for p in 'Ab'}


def test_monte_carlo_risk_agrees_with_exact_affine_risk():
    c = {**FIXTURE, 'evaluation': {'records': 40000, 'hutchinson_probes': 1}}
    g, sigma, L, Li, train, ec, en = s.make_data(c, 920002)
    x, mu, S = s.moments(train, 80)
    A, b = s.closed_form(c, S, mu, Li @ Li, c['k'])
    At, bt = torch.from_numpy(A), torch.from_numpy(b)
    ev = s.evaluate(c, torch, 920002, lambda X: X @ At.T + bt, ec, en, Li)
    exact = s.risks(A, b, L @ g.K(c['theta']) @ L, sigma, Li @ Li)
    assert abs(ev['clean_whitened'] - exact['clean']) < 0.03 * exact['clean']
    assert abs(ev['clean_euclidean'] - exact['euclidean_clean']) < 0.03 * exact['euclidean_clean']
    assert abs(ev['divergence'] - np.trace(A)) < 0.05 * np.trace(A) + 0.1


def test_training_is_deterministic_and_follows_the_stopping_rule():
    from experiments.mlst2026.x3_nonlinear_metric_20261005.workflow import check_training_record
    c = FIXTURE
    g, sigma, L, Li, train, ec, en = s.make_data(c, 920003)
    runs = [s.train_deep(c, torch, train[:64], train[64:80], Li, 'silu', 12345) for _ in range(2)]
    assert runs[0][1] == runs[1][1]
    for (k, a), (_, b) in zip(runs[0][0].state_dict().items(), runs[1][0].state_dict().items()):
        assert torch.equal(a, b)
    check_training_record(c, runs[0][1])
    bad = dict(runs[0][1], best_epoch=runs[0][1]['best_epoch'] + 1)
    with pytest.raises(ValueError):
        check_training_record(c, bad)


def fake(c, w, sure_offset=0.0):
    recs = []
    for seed in c['seeds']:
        r0 = np.random.default_rng(seed)
        for n in c['N']:
            models = {f'{cls}__{loss}': {'clean_whitened': w(cls, loss, n) + 1e-3 * r0.normal(),
                                         'clean_euclidean': w(cls, loss, n), 'sure': w(cls, loss, n) + sure_offset}
                      for cls in c['classes'] for loss in c['losses']}
            recs.append({'seed': seed, 'N': n, 'models': models,
                         'training': {'x': {'restarts': [{'stopped': 'patience'}]}}})
    return recs


def test_family_and_prediction_rules():
    c = copy.deepcopy(BASE)
    good = lambda cls, loss, n: 300.0 - (40.0 if loss == 'sigma_inverse' else 0.0) + (2.0 if cls == 'deep_silu' else 0.0)  # noqa: E731
    red = s.reduce_records(c, fake(c, good))
    assert red['family_size'] == 54
    assert red['predictions']['P1']['verdict'] == 'HOLDS' and red['predictions']['P2']['verdict'] == 'HOLDS'
    bad = lambda cls, loss, n: 300.0 + (5.0 if loss == 'sigma_inverse' else 0.0) - (3.0 if cls == 'deep_silu' else 0.0)  # noqa: E731
    red = s.reduce_records(c, fake(c, bad))
    assert red['predictions']['P1']['verdict'] == 'FAILS' and red['predictions']['P2']['verdict'] == 'FAILS'
    flat = lambda cls, loss, n: 300.0  # noqa: E731
    assert s.reduce_records(c, fake(c, flat))['predictions']['P1']['verdict'] == 'UNDECIDED'
    e = enumeration(BASE)
    assert e['paired_cells'] == 72 and e['trainings'] == 1152


def test_p2_attribution_and_sure_agreement_rules():
    c = copy.deepcopy(BASE)
    # SiLU beats both linear classes, which tie: attributed to the nonlinearity
    nonlin = lambda cls, loss, n: 300.0 - (5.0 if cls == 'deep_silu' else 0.0)  # noqa: E731
    red = s.reduce_records(c, fake(c, nonlin))
    p2 = red['predictions']['P2']
    assert p2['verdict'] == 'FAILS' and {x['attribution'] for x in p2['locations']} == {'nonlinearity'}
    assert red['predictions']['SURE_agreement']['verdict'] == 'AGREES'
    # both deep classes beat the closed form and tie with each other: attributed to training
    train = lambda cls, loss, n: 300.0 - (5.0 if cls != 'tied_linear' else 0.0)  # noqa: E731
    p2 = s.reduce_records(c, fake(c, train))['predictions']['P2']
    assert p2['verdict'] == 'FAILS' and {x['attribution'] for x in p2['locations']} == {'training'}
    # SiLU worse than the linear classes: P2 holds, nothing to attribute
    worse = lambda cls, loss, n: 300.0 + (5.0 if cls == 'deep_silu' else 0.0)  # noqa: E731
    p2 = s.reduce_records(c, fake(c, worse))['predictions']['P2']
    assert p2['verdict'] == 'HOLDS' and {x['attribution'] for x in p2['locations']} == {None}
    # a systematic SURE bias is reported with its decided cells
    red = s.reduce_records(c, fake(c, lambda cls, loss, n: 300.0, sure_offset=1.0))
    sa = red['predictions']['SURE_agreement']
    assert sa['verdict'] == 'DISAGREES' and len(sa['decided']) == 18
    assert set(red['risk_summary']) == {f'{n}/{cls}__{loss}/{k}' for n in c['N'] for cls in c['classes']
                                        for loss in c['losses'] for k in ('clean_whitened', 'clean_euclidean', 'sure')}
