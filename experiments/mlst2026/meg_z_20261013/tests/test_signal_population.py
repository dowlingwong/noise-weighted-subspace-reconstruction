"""Signal model (theta = 0 exactly separable, one latent scale per subject), the metric-corrected certificate
B_dagger, the general population path (= pinned Y19/TR09 path for a Kronecker reference), the theta_s rule."""
import json

import numpy as np
import pytest

from experiments.mlst2026.meg_z_20261013 import common, estimators as E, fixture, population as P, signal as S
from experiments.mlst2026.meg_z_20261013.preprocess import load_subject
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol
from experiments.mlst2026.y19_20261012 import predictions as y19pred


def smoke_proto(kind='ledoit_wolf_full'):
    p = fixture.smoke_protocol(load_protocol())
    p['reference_metric'] = dict(p['reference_metric'], kind=kind)
    return p


def test_protocol_defaults_after_review():
    p = load_protocol()
    pp = p['preprocess']
    assert (pp['decim'], pp['C'], pp['T']) == (12, 24, 20) and abs(1100 / 12 - 91.6667) < 1e-3
    assert p['reference_metric']['kind'] == 'ledoit_wolf_full' and p['calibration']['m_hi'] is None
    assert p['dataset']['license_in_snapshot'] == 'CC0'
    assert [a['name'] for a in p['arms']] == ['ref', 'lw_lo', 'lw_hi', 'kron_lo', 'kron_hi', 'iso']


def test_temporal_bases_and_rotation():
    B1, Cb = S.temporal_bases(20, 1100 / 12, [0.1, 0.17], 0.02)
    assert np.allclose(B1.T @ B1, np.eye(3)) and np.allclose(Cb.T @ Cb, np.eye(3)) and np.allclose(B1.T @ Cb, 0, atol=1e-12)
    assert np.array_equal(S.rotated_basis(0.0, B1, Cb), B1)
    for th in (0.12, 0.25, 0.4):
        B2 = S.rotated_basis(th, B1, Cb)
        assert np.allclose(B2.T @ B2, np.eye(3)) and np.allclose(np.linalg.svd(B1.T @ B2, compute_uv=False), np.cos(th))


@pytest.fixture(scope='module')
def subj(tmp_path_factory):
    d = tmp_path_factory.mktemp('pop')
    fixture.make_derived_subject(d, 'sub-91', kind='kron', seed=7)
    fixture.make_derived_subject(d, 'sub-92', kind='nonkron', seed=8)
    return load_subject(d / 'sub-91.h5', cache=False), load_subject(d / 'sub-92.h5', cache=False)


def test_signal_separable_at_theta0_and_fixed_scale(subj):
    sd, _ = subj
    proto = smoke_proto()
    ref = P.reference_for(sd, proto)
    vs = []
    for th in proto['signal']['theta']:
        m, info = S.build(sd.evoked, sd.sfreq, ref['cov'], th, proto['signal'])
        vs.append(m.variances)
        if th == 0:
            assert abs(info['trace_K'] - proto['signal']['trace_K_target']) < 1e-9 * proto['signal']['trace_K_target']
        Q = np.linalg.qr(m.templates())[0]
        sc_, st_ = S.partial_trace_spectra(Q @ Q.T, sd.C, sd.T)
        if th == 0:
            assert abs(sc_[:2].sum() - 6) < 1e-10 and abs(st_[:3].sum() - 6) < 1e-10
        else:
            assert st_[:3].sum() < 6 - 1e-6
    assert all(np.array_equal(v, vs[0]) for v in vs)                         # no per-theta rescaling
    m, _ = S.build(sd.evoked, sd.sfreq, ref['cov'], 0.25, proto['signal'])
    Xs, al = m.draw(40000, common.rng('t', 1))
    R = m.second_moment()
    assert np.linalg.norm(Xs.T @ Xs / len(Xs) - R) < 0.03 * np.linalg.norm(R)


@pytest.mark.parametrize('which', [0, 1])
def test_b_dagger_is_a_valid_certificate(subj, which):
    """B_dagger <= a_best (general path, LW reference) on Kronecker and non-Kronecker fixture noise; exactly 0 at
    theta = 0; strictly increasing over the fixed theta grid."""
    sd = subj[which]
    sp = P.SubjectPopulation(sd, smoke_proto(), 'test-ns')
    rows = [sp.row(th, 6, 'test') for th in (0.0, 0.12, 0.25, 0.4)]
    assert rows[0]['B_cert'] == 0.0 and abs(rows[0]['a_best']) < 1e-9
    for r in rows:
        assert r['B_cert'] <= r['a_best'] + 1e-12, (r['theta'], r['B_cert'], r['a_best'])
        assert 0 < r['B_dagger']['lambda_min'] <= 1.5
    assert all(b['B_cert'] > a['B_cert'] for a, b in zip(rows, rows[1:]))
    pz = P.pz1_subject(rows)
    assert pz['met'] and pz['zero_at_theta0'] and pz['a_best_zero_at_theta0']


def test_b_dagger_reduces_to_paper_certificate_for_kronecker_reference(subj):
    sd, _ = subj
    proto = smoke_proto('kronecker_flipflop')
    ref = P.reference_for(sd, proto)
    m, _ = S.build(sd.evoked, sd.sfreq, ref['cov'], 0.25, proto['signal'])
    bd = P.b_dagger(m.second_moment(), ref['cov'], ref['St'], ref['Sc'], 2, 3)
    assert abs(bd['lambda_min'] - 1) < 1e-9
    assert abs(bd['B_dagger'] - P.kron_certificate(m.second_moment(), ref['St'], ref['Sc'], 2, 3)['B_cert']) < 1e-12


def test_general_path_reproduces_pinned_path(subj):
    sd, _ = subj
    proto = smoke_proto('kronecker_flipflop')
    sp = P.SubjectPopulation(sd, proto, 'test-ns')
    r = sp.row(0.25, 8, 'x', crosscheck=True)
    for q in ('a_best', 'cU', 'cF'):
        assert r['crosscheck_pinned'][q]['abs_diff'] <= 1e-6 * max(1.0, abs(r[q])), (q, r['crosscheck_pinned'][q])
    assert abs(r['crosscheck_pinned']['B_cert']['pinned'] - r['crosscheck_pinned']['B_cert']['B_dagger']) < 1e-9


def test_general_gradient_and_hessian_finite_differences(subj):
    _, sd = subj
    proto = smoke_proto()
    lw = E.reference_metric(sd.ref, sd.C, sd.T, 'ledoit_wolf_full')['cov']
    m, _ = S.build(sd.evoked, sd.sfreq, lw, 0.25, proto['signal'])
    _, A = P.spd_powers(lw)
    M = A @ m.second_moment() @ A + np.eye(sd.d)
    cls = P.GeneralProductClass(A, M, sd.C, sd.T, 2, 3)
    g = np.random.default_rng(0)
    pt = cls.chart(g.normal(size=(sd.C, 2)), g.normal(size=(sd.T, 3)))
    grad, H = cls.gradient(pt), cls.hessian(pt)
    h = 1e-6
    fd = np.array([(cls.value(pt, h * e) - cls.value(pt, -h * e)) / (2 * h) for e in np.eye(cls.n)])
    assert np.allclose(grad, fd, atol=1e-6 * max(1, np.abs(grad).max()))
    fdh = np.array([(cls._grad(pt, h * e).real - cls._grad(pt, -h * e).real) / (2 * h) for e in np.eye(cls.n)])
    assert np.allclose(H, fdh, atol=1e-5 * max(1, np.abs(H).max()))


def test_naive_certificate_on_lw_whitened_K_is_not_a_certificate(subj):
    """Why B_dagger is needed: the partial-trace certificate on K = Sigma_LW^{-1/2} R_s Sigma_LW^{-1/2} is > 0 at
    theta = 0 although the fitted physical product class contains the signal subspace (a_best = 0)."""
    _, sd = subj
    proto = smoke_proto()
    lw = E.reference_metric(sd.ref, sd.C, sd.T, 'ledoit_wolf_full')['cov']
    m, _ = S.build(sd.evoked, sd.sfreq, lw, 0.0, proto['signal'])
    _, Li = P.spd_powers(lw)
    model = P.KroneckerPopulationModel(Li @ m.second_moment() @ Li, sd.C, sd.T, 2, 3, 0.0, None, None)
    assert y19pred.certificate(model)['B_cert'] > 1e-4
    assert P.SubjectPopulation(sd, proto, 'n').row(0.0, 4, 'y')['B_cert'] == 0.0


def test_theta_rule_root_search(subj):
    sd, _ = subj
    proto = smoke_proto()
    sp = P.SubjectPopulation(sd, proto, 'test-ns')
    rule = proto['theta_rule']
    r = P.theta_for_target(sp, 150.0, rule, 4, 'A')
    assert r['root'] and 0.005 < r['theta'] < 0.8
    assert abs(r['N_star'] - 150.0) < 0.05 * 150.0                         # final constants with more starts: close to target
    r_lo = P.theta_for_target(sp, 300.0, rule, 4, 'B')
    assert r_lo['theta'] < r['theta']                                       # larger N* <-> smaller theta (a increases in theta)
    none = P.theta_for_target(sp, 1.0, rule, 4, 'A')                        # unattainable target
    assert not none['root'] and none['reason']
    assert P.theta_targets(2150, 50) == {'P': 2150, 'n_min': 50, 'N_star_A': 430.0, 'N_star_B': 250.0}
    assert P.theta_targets(4000, 50)['N_star_B'] == 400.0


def test_m_grid_rule():
    cal = load_protocol()['calibration']
    assert P.choose_m_hi({'a': 1200, 'b': 1127}, cal)['m_hi'] == 1024
    assert P.choose_m_hi({'a': 1200, 'b': 1126}, cal)['m_hi'] == 768


def test_subject_predictions_eligibility(subj):
    sd, _ = subj
    proto = smoke_proto()
    targets = P.theta_targets(900, 20)
    out = P.subject_predictions(sd, proto, 'test-ns', targets, {'m_lo': 64, 'm_hi': 256})
    assert out['eligible'], out['exclusion_reasons']
    assert out['geometries']['A']['grid'][2] == P.round_half_up(out['geometries']['A']['N_star'])
    assert out['geometries']['control']['grid'] == [30, 60, 120, 240]
    bad = P.subject_predictions(sd, proto, 'test-ns', targets, {'m_lo': 64, 'm_hi': 10_000})
    assert not bad['eligible'] and any('calibration' in r for r in bad['exclusion_reasons'])


def test_b_dagger_raw_residue_and_relative_cut(subj):
    """Delta D5: the raw residue is stored and the zero cut is 1e-10 x max(1, tr K_G)."""
    sd, _ = subj
    sp = P.SubjectPopulation(sd, smoke_proto(), 'test-ns')
    m, _ = S.build(sd.evoked, sd.sfreq, sp.ref['cov'], 0.0, sp.sc)
    bd = P.b_dagger(m.second_moment(), sp.ref['cov'], sp.St, sp.Sc, 2, 3)
    assert bd['B_cert_KG'] == 0.0 and abs(bd['B_cert_KG_raw']) <= bd['zero_cut'] and bd['zero_cut'] >= 1e-10
    m, _ = S.build(sd.evoked, sd.sfreq, sp.ref['cov'], 0.25, sp.sc)
    bd = P.b_dagger(m.second_moment(), sp.ref['cov'], sp.St, sp.Sc, 2, 3)
    assert bd['B_cert_KG'] == bd['B_cert_KG_raw'] > bd['zero_cut']


def test_theta_search_treats_nonpositive_nstar_as_minus_infinity():
    """Delta D6: where a > 0 and B >= 0 (the product class never pays) N* counts as -infinity, so a bracket whose
    upper end has B >= 0 still yields a root."""
    class Stub:
        def row(self, th, nr, stream):
            a = 10.0 * th ** 2
            cu_cf = 50.0
            B = -cu_cf + a
            n = cu_cf / a - 1 if (a > 1e-9 and B < 0) else None
            return {'N_star_pred': n, 'a_best': a, 'B': B, 'restart_agreement': True}
    rule = {'theta_bracket': [0.005, 3.0], 'xtol': 1e-6, 'search_random_starts': 1}
    r = P.theta_for_target(Stub(), 100.0, rule, 1, 'A')
    assert r['root'] and abs(10 * r['theta'] ** 2 - 50.0 / 101.0) < 1e-4
    assert any(t['N_star_pred'] is None and t['B'] >= 0 for t in r['search'])


def test_population_cli_refuses_confirmation_subject(tmp_path):
    with pytest.raises(SystemExit, match='frozen protocol'):
        P.main(['--derived', str(tmp_path), '--subjects', 'sub-01,sub-09', '--out', str(tmp_path / 'p.json')])


def test_guard_refuses_variant_claiming_frozen(tmp_path):
    """Delta D3: a --protocol variant that claims to be frozen does not unlock confirmation subjects."""
    from experiments.mlst2026.meg_z_20261013.preprocess import refuse_unfrozen_confirmation
    p = load_protocol()
    p.update(frozen=True, status='FROZEN (fake variant)')
    f = tmp_path / 'variant.json'
    f.write_text(json.dumps(p))
    with pytest.raises(SystemExit, match='--protocol'):
        refuse_unfrozen_confirmation(['sub-05'], p, str(f))
    with pytest.raises(SystemExit, match='frozen protocol'):              # no kit protocol.json, dict claims frozen
        refuse_unfrozen_confirmation(['sub-05'], p, None)
    refuse_unfrozen_confirmation(['sub-01'], p, str(f))                    # dev subjects are unaffected
    with pytest.raises(SystemExit):
        P.main(['--protocol', str(f), '--derived', str(tmp_path), '--subjects', 'sub-05', '--out', str(tmp_path / 'p.json')])
