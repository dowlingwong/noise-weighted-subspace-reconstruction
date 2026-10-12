"""Learners (T25 recipe) and scores on a synthetic fixture subject."""
import numpy as np

from experiments.mlst2026.meg_z_20261013 import common, fit, fixture, population as P, signal as S
from experiments.mlst2026.meg_z_20261013.preprocess import load_subject
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol


def setup(tmp_path, theta=0.25):
    fixture.make_derived_subject(tmp_path, 'sub-91', kind='kron', seed=3)
    sd = load_subject(tmp_path / 'sub-91.h5', cache=False)
    proto = fixture.smoke_protocol(load_protocol())
    ref = P.reference_for(sd, proto)
    m, _ = S.build(sd.evoked, sd.sfreq, ref['cov'], theta, proto['signal'])
    return sd, proto, ref, m


def test_scores_identity_map_and_readout(tmp_path):
    sd, proto, ref, m = setup(tmp_path)
    W0 = fit.metric(ref['cov'])
    S_ev, al = m.draw(300, common.rng('e', 1))
    noise = sd.records('evaluation')[:300]
    ev = fit.Evaluation(S_ev, noise, al, m.templates(), m.variances, W0, m.second_moment())
    d = sd.d
    sc = ev.scores(np.eye(d), np.zeros(d))
    expect = float(np.mean(np.einsum('ij,ij->i', noise @ W0, noise)))
    assert abs(sc['clean_ref'] - expect) < 1e-9 * expect                       # identity map: clean risk = noise energy
    assert abs(sc['clean_ref_semi_analytic'] - expect) < 1e-8 * expect          # semi-analytic: same empirical noise moments
    ev0 = fit.Evaluation(S_ev, np.zeros_like(noise), al, m.templates(), m.variances, W0, m.second_moment())
    s0 = ev0.scores(np.eye(d), np.zeros(d))
    assert s0['readout_nmse_a11'] < 1e-20 and s0['readout_nmse_a21'] < 1e-20 and s0['clean_ref'] < 1e-20


def test_fit_pair_and_score_fit(tmp_path):
    sd, proto, ref, m = setup(tmp_path, theta=0.0)
    W0 = fit.metric(ref['cov'])
    S_tr, _ = m.draw(800, common.rng('t', 2))
    X = S_tr + sd.records('training')[:800]
    f = fit.fit_pair(X, 800, W0, sd.C, sd.T, 2, 3, proto['learners'], ('k', 1))
    assert set(f['maps']) == {'U', 'F'} and len(f['restart_maps']) == 4
    A, b = f['maps']['F']
    assert np.allclose(A @ A, A, atol=1e-8)                                    # oblique projector
    S_ev, al = m.draw(300, common.rng('e', 3))
    ev = fit.Evaluation(S_ev, sd.records('evaluation')[:300], al, m.templates(), m.variances, W0, m.second_moment())
    r = fit.score_fit(f, ev)
    assert abs(r['D'] - (r['F']['clean_ref'] - r['U']['clean_ref'])) < 1e-12
    assert r['D'] < 0                                                           # separable signal, N = 800, d = 24: product wins
    assert isinstance(r['optimization_limited'], bool) and len(r['restart_scores']) == 4
    f2 = fit.fit_pair(X, 800, W0, sd.C, sd.T, 2, 3, proto['learners'], ('k', 1))
    assert np.array_equal(f2['maps']['F'][0], A)                               # deterministic


def test_refined_optimization_flag_rule():
    """Delta review: refined flag (iteration_limit, or line_search_stopped with gradient > 1e-6, or spread >= |D|)
    beside the unchanged T25 flag."""
    r = lambda st, g: [{'status': st, 'gradient_norm': g}]                    # noqa: E731
    f = fit.optimization_flags(r('line_search_stopped', 2e-8), 0, 1e-7, 1.0)
    assert not f['optimization_limited'] and f['optimization_limited_t25_rule']
    assert fit.optimization_flags(r('line_search_stopped', 5e-6), 0, 1e-7, 1.0)['optimization_limited']
    assert fit.optimization_flags(r('iteration_limit', 1e-9), 0, 1e-7, 1.0)['optimization_limited']
    f = fit.optimization_flags(r('stationary', 1e-9), 0, 2.0, 1.0)
    assert f['optimization_limited'] and f['optimization_limited_t25_rule']
    f = fit.optimization_flags(r('stationary', 1e-9), 0, 1e-7, 1.0)
    assert not f['optimization_limited'] and not f['optimization_limited_t25_rule']
    assert fit.optimization_flags(r('line_search_stopped', 5e-6), 0, 1e-7, 1.0, {'line_search_gradient_max': 1e-5})['optimization_limited'] is False
    assert load_protocol()['learners']['flag_rule']['line_search_gradient_max'] == 1e-6
