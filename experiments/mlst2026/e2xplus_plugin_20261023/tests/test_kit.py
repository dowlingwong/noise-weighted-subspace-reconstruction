"""E2X+ kit tests T1-T9 (protocol version 2, section 9 gate 2).  Run from the EXP root:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q experiments/mlst2026/e2xplus_plugin_20261023/tests
"""
from __future__ import annotations

import json
import glob
from pathlib import Path

import numpy as np
import pytest

from experiments.mlst2026.e2xplus_plugin_20261023 import data as D
from experiments.mlst2026.e2xplus_plugin_20261023 import plugin as PL
from experiments.mlst2026.e2xplus_plugin_20261023 import runner as R
from experiments.mlst2026.p1nl_dev_20261006.generator import SparseLatentModel
from experiments.mlst2026.successor_20260913.t20s import generator as t20s
from experiments.mlst2026.t25_production_20260930 import science as t25
from experiments.mlst2026.y19_20261012 import generator as gen
from experiments.mlst2026.y19_20261012 import predictions as Y19

PKG = Path(R.__file__).resolve().parent


def _pop_K(G, geo):
    if G == 'G1':
        return SparseLatentModel(theta=float(geo), p=1.0)
    return gen.PopulationModel(D.g2(), D.G2_PHI[geo])


# T1 ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('G,geo', [('G1', '0.12'), ('G1', '0.25'), ('G1', '0.4'), ('G2', 'A'), ('G2', 'B')])
def test_T1_plugin_at_truth_reproduces_frozen(G, geo):
    pop = R.population()[(G, geo)]
    m = _pop_K(G, geo)
    pm = PL.PluginModel(m.K, m.C, m.T, m.kc, m.kt)
    c = PL.plugin_constants(pm.K, m.C, m.T, m.kc, m.kt, t25.rng('e2xplus-test', G, geo))
    assert c['available']
    assert abs(c['N_star_plug'] - pop['N_star_pred']) / pop['N_star_pred'] < 1e-6
    assert abs(c['B_cert'] - pop['B_cert']) < 1e-9
    assert abs(c['cU'] - pop['cU']) < 1e-9 * pop['cU'] and abs(c['cF'] - pop['cF']) < 1e-9 * pop['cF']


# T2 ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('G,geo', [('G1', '0'), ('G2', 'control')])
def test_T2_separable_certificate_zero(G, geo):
    m = _pop_K(G, geo)
    assert PL.certificate_only(m.K, m.C, m.T, m.kc, m.kt)['B_cert'] < 1e-12


# T3 ---------------------------------------------------------------------------------------------------------
def test_T3_partial_trace_cap_2x2():
    # C = T = 2, k_c = k_t = 1, k = 1: a product vector has cap 1 (k - cap = 0); an entangled one has cap < ... -> k - cap > 0
    u = np.kron([1.0, 0.0], [0.0, 1.0])[:, None]          # column-major c + C t: kron(u_t, u_c)
    assert t20s.partial_trace_cap(u, 2, 2, 1, 1)['k_minus_cap'] == pytest.approx(0.0, abs=1e-12)
    v = (np.kron([1.0, 0.0], [1.0, 0.0]) + np.kron([0.0, 1.0], [0.0, 1.0]))[:, None] / np.sqrt(2)
    # partial traces are I/2 in both factors; the rank-1 caps are 1/2 each -> k - cap = 1 - 1/2 = 1/2
    assert t20s.partial_trace_cap(v, 2, 2, 1, 1)['k_minus_cap'] == pytest.approx(0.5, abs=1e-12)


# T4 ---------------------------------------------------------------------------------------------------------
def test_T4_plugin_never_reads_oracle_factors(monkeypatch):
    dd = D.draw('G1', '0.25', D.seed_of(0, R.FIXTURE_NAMESPACE), R.FIXTURE_NAMESPACE)
    Ws = PL.whiteners(dd['sigma'], dd['calibration'], dd['C'], dd['T'])
    Kh = PL.k_hat(dd['training'], 800, Ws['oracle'])

    def boom(self):
        raise AssertionError('oracle factor read')
    for cls in (t20s.Geometry, gen.G2Geometry):
        monkeypatch.setattr(cls, 'U_c0', property(boom), raising=False)
        monkeypatch.setattr(cls, 'U_t0', property(boom), raising=False)
    pm = PL.PluginModel(Kh, dd['C'], dd['T'], dd['kc'], dd['kt'])
    assert isinstance(pm.g, PL._Starts)
    c = PL.plugin_constants(Kh, dd['C'], dd['T'], dd['kc'], dd['kt'], t25.rng('e2xplus-test', 'T4'))
    assert 'a_best' in c


# T5 ---------------------------------------------------------------------------------------------------------
def test_T5_g2_calibration_stream():
    a = D.draw('G2', 'A', 123, R.FIXTURE_NAMESPACE)
    b = D.draw('G2', 'A', 123, R.FIXTURE_NAMESPACE)
    c = D.draw('G2', 'B', 123, R.FIXTURE_NAMESPACE)
    assert np.array_equal(a['calibration'], b['calibration']) and np.array_equal(a['training'], b['training'])
    assert np.array_equal(a['calibration'], c['calibration'])          # shared across phi for a seed (by design)
    for x in (a['training'][:1024], c['training'][:1024]):
        corr = np.corrcoef(x[:, 0], a['calibration'][:, 0])[0, 1]
        assert abs(corr) < 0.15


# T6 ---------------------------------------------------------------------------------------------------------
def test_T6_seeds_and_keys_distinct():
    ours = {D.seed_of(i) for i in range(R.N_Q1_SEEDS)}
    assert len(ours) == R.N_Q1_SEEDS
    assert not ours & {D.seed_of(i, R.FIXTURE_NAMESPACE) for i in range(10)}
    exp = PKG.parents[2]
    recorded = set()
    for f in [exp / 'experiments/mlst2026/t25_production_20260930/production.json',
              exp / 'experiments/mlst2026/zc_baselines_20261013/protocol.json',
              exp / 'experiments/mlst2026/y19_20261012/protocol.json']:
        if f.exists():
            txt = f.read_text()
            recorded |= {int(s) for s in __import__('re').findall(r'\b\d{9,13}\b', txt)}
    recorded |= set(Y19.seeds())
    assert not ours & recorded
    s0 = D.seed_of(0)
    keys = set()
    for G, geo in D.geometries():
        for N in (400, 2917):
            for arm in PL.ARMS:
                keys.add('/'.join(map(str, (D.NAMESPACE, G, geo, N, arm, s0, 'restarts'))))
    # the real data keys: t25.make_data (G1) and y19 generator.draw / the kit calibration line (G2)
    data_keys = {'/'.join(map(str, (f'{D.NAMESPACE}/G1', s0, 'calibration')))}
    data_keys |= {'/'.join(map(str, (f'{D.NAMESPACE}/G1', s0, float(t), part))) for t in D.G1_THETAS for part in ('signal', 'noise')}
    data_keys |= {'/'.join(map(str, (f'{D.NAMESPACE}/G2', s0, float(phi), part))) for phi in D.G2_PHI.values() for part in ('signal', 'noise')}
    data_keys.add('/'.join(map(str, (f'{D.NAMESPACE}/G2', s0, 'calibration'))))
    assert len(data_keys) == 1 + 8 + 6 + 1
    assert not keys & data_keys and len(keys) == len(D.geometries()) * 2 * len(PL.ARMS)


# T7 ---------------------------------------------------------------------------------------------------------
def test_T7_fixture_unit_has_all_fields(tmp_path):
    r = R.run_unit(('G1', '0.25', 0, 1, R.FIXTURE_NAMESPACE, str(tmp_path)))
    rec = json.loads(Path(r['written']).read_text())
    assert set(rec['q1']) == {f'{N}/{a}' for N in D.Q1_GRID for a in PL.ARMS}
    assert set(rec['q2']) == {f'{N}/{a}' for N in D.G1_Q2_GRID for a in PL.ARMS}
    for c in rec['q2'].values():
        for k in ('a_best', 'cU', 'cF', 'B_cert', 'N_star_plug', 'rule_choice', 'a_db', 'N_star_db', 'rule_choice_db', 'seconds'):
            assert k in c


def test_T7b_g2_unit_has_all_fields(tmp_path):
    r = R.run_unit(('G2', 'B', 0, 1, R.FIXTURE_NAMESPACE, str(tmp_path)))
    rec = json.loads(Path(r['written']).read_text())
    assert set(rec['q2']) == {f'{N}/{a}' for N in D.G2_Q2_GRID['B'] for a in PL.ARMS}
    assert all('rule_choice_db' in c for c in rec['q2'].values())


# T8 ---------------------------------------------------------------------------------------------------------
def test_T8_verify_kit_and_closure_pinned():
    assert R.verify_kit()['status'] == 'ok'
    D.draw('G2', 'A', 1, R.FIXTURE_NAMESPACE); D.draw('G1', '0.25', 1, R.FIXTURE_NAMESPACE)
    pinned = set(R.protocol()['imports_sha256'])
    assert set(R.import_closure()) <= pinned, sorted(set(R.import_closure()) - pinned)


# T9 ---------------------------------------------------------------------------------------------------------
def test_T9_decision_functions():
    base = {'available': True, 'N_star_plug': 1000.0, 'cause': None, 'a_best': 2.0, 'cU': 3000.0, 'cF': 200.0}
    assert PL.rule_choice(500, base) == 'F' and PL.rule_choice(1500, base) == 'U'
    assert PL.rule_choice(500, {'available': False, 'cause': 'a_zero'}) == 'F'
    assert PL.rule_choice(500, {'available': False, 'cause': 'B_nonnegative'}) == 'U'
    assert PL.rule_choice(500, {'available': False, 'cause': 'H_not_pos_def'}) is None
    assert PL.rule_choice(500, {'available': False, 'cause': 'nonpositive_gap'}) is None
    d = PL.debias(1000, base)                          # a_db = 2 - 2800/1000 = -0.8 -> restrict
    assert d['db_cause'] == 'a_db_nonpositive' and PL.rule_choice_db(1000, dict(base, **d)) == 'F'
    d = PL.debias(4000, base)                          # a_db = 2 - 0.7 = 1.3 -> N*_db = 2800/1.3 - 1
    assert d['N_star_db'] == pytest.approx(2800 / 1.3 - 1)
    assert PL.rule_choice_db(4000, dict(base, **d)) == 'U' and PL.rule_choice_db(1000, dict(base, a_best=5.0, **PL.debias(1000, dict(base, a_best=5.0)))) == 'F'
    d = PL.debias(1000, dict(base, cU=100.0, cF=200.0))
    assert d['db_cause'] == 'c_gap_nonpositive' and PL.rule_choice_db(1000, dict(base, **d)) == 'U'
    bad = {'available': False, 'cause': 'H_not_pos_def', 'a_best': 1.0, 'cU': 3000.0, 'cF': 200.0}
    assert PL.rule_choice_db(1000, dict(bad, **PL.debias(1000, bad))) is None


# T10 --------------------------------------------------------------------------------------------------------
def _synthetic(pop, n_det, frac, med, nan_db=False):
    """Minimal res dict for readings(): O2 N_det per (geometry, arm), O3 per (cell, arm, plug-in arm)."""
    res = {'O2': {}, 'O3': {}}
    for G, geo in D.geometries():
        if not pop[(G, geo)]['N_star_pred']:
            continue
        for arm in PL.ARMS:
            res['O2'][f'{G}/{geo}/{arm}'] = {'N_det_oracle_calibrated': n_det(G, geo, arm)}
            for N in D.q2_grid(G, geo):
                for p in ('naive', 'debiased'):
                    nst = pop[(G, geo)]['N_star_pred']
                    res['O3'][f'{G}/{geo}/{N}/{arm}/{p}'] = {
                        'frac_within_factor2_of_all': frac(G, geo, N, arm, p), 'frac_unavailable': 0.0,
                        'median_ratio': med(G, geo, N, arm, p),
                        'median_over_first_order': 1.1 if p == 'naive' else None}
    return res


def test_T10_readings_boundaries():
    pop = R.population()
    # all good with N_det = 800: holds at G1 0.25 (cell 1221) and G2 A (cell 2917), fails at G1 0.40 (cell 488)
    res = _synthetic(pop, lambda G, g, a: 800, lambda *k: 1.0, lambda *k: 1.0)
    rd = R.readings(res, pop)
    assert rd['outcome']['G1/0.25/debiased'] == 'A' and rd['outcome']['G2/A/debiased'] == 'A'
    assert rd['R-Q1']['G1/0.25/oracle']['holds'] and not rd['R-Q1']['G1/0.4/oracle']['holds']
    assert rd['outcome']['G1/0.4/debiased'] == 'D'
    res = _synthetic(pop, lambda G, g, a: 1600, lambda *k: 1.0, lambda *k: 1.0)      # 1600 > 1221 at G1 0.25
    assert R.readings(res, pop)['outcome']['G1/0.25/debiased'] == 'D'
    assert rd['R-Q2']['G2/A/oracle/debiased']['N_loc'] == 729 and rd['estimated_arms'] == 'held'
    assert rd['naive_first_order_prediction']['confirmed']
    assert '{' not in rd['main_text_sentence']
    # non-monotone location: good only at the smallest grid N -> N_loc None -> outcome B with defined blanks
    res = _synthetic(pop, lambda G, g, a: 400, lambda G, g, N, a, p: 1.0 if N == min(D.q2_grid(G, g)) else 0.5,
                     lambda G, g, N, a, p: float('inf'))
    rd = R.readings(res, pop)
    assert rd['R-Q2']['G1/0.25/oracle/debiased']['N_loc'] is None and rd['R-Q2']['G1/0.25/oracle/debiased']['non_monotone']
    assert rd['outcome']['G1/0.25/debiased'] == 'B' and 'more than 10' in rd['text']['G1/0.25/debiased']
    assert '50% of seeds' in rd['text']['G1/0.25/debiased'] and 'N = 1000' in rd['text']['G1/0.25/debiased']
    # degraded: Kronecker N_det None while oracle has a value
    res = _synthetic(pop, lambda G, g, a: None if a == 'kron256' else 400, lambda *k: 1.0, lambda *k: 1.0)
    assert R.readings(res, pop)['estimated_arms'] == 'degraded'
    # none/none is not later
    assert R.later(None, None) is False and R.later(None, 3) is True and R.later(3, None) is False
    # D and C
    res = _synthetic(pop, lambda G, g, a: None, lambda *k: 1.0, lambda *k: 1.0)
    assert R.readings(res, pop)['outcome']['G1/0.25/debiased'] == 'D'
    res = _synthetic(pop, lambda G, g, a: None, lambda *k: 0.0, lambda *k: 0.1)
    rd = R.readings(res, pop)
    assert rd['outcome']['G1/0.25/debiased'] == 'C' and rd['abstract_outcome_weakest'] == 'C'


def test_T11_summarize_with_error_records(tmp_path):
    """One nonfinite and one exception constants record pass through cmd_summarize (kit review R1)."""
    out = tmp_path / 'run'
    for G, geo in D.geometries():
        for idx in range(2):
            q1 = {f'{N}/{a}': {'B_cert': 0.1 * idx + (0.0 if geo in ('0', 'control') else 0.5), 'Delta_k': 1.0, 'n_clipped': 0,
                               'clipped_mass': 0.0, 'k_minus_cap': 0.1} for N in D.Q1_GRID for a in PL.ARMS}
            q2 = {}
            for N in D.q2_grid(G, geo):
                for a in PL.ARMS:
                    if idx == 0:
                        c = {'available': False, 'cause': 'nonfinite', 'n_clipped': 0, 'a_best': None, 'cU': 3000.0, 'cF': None, 'B': None}
                    else:
                        c = {'available': False, 'cause': 'exception', 'error': 'X', 'n_clipped': 0}
                    c['rule_choice'] = PL.rule_choice(N, c); c.update(PL.debias(N, c)); c['rule_choice_db'] = PL.rule_choice_db(N, c)
                    c['seconds'] = 0.0
                    q2[f'{N}/{a}'] = c
            rec = {'generator': G, 'geometry': geo, 'seed_index': idx, 'seed': idx, 'namespace': 'synthetic', 'in_q2': True,
                   'q1': q1, 'q2': q2, 'seconds': 0.0}
            path = out / 'records' / G / geo / f'seed{idx:03d}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec))
    rd = R.cmd_summarize(str(out))
    summ = json.loads((out / 'SUMMARY.json').read_text())
    o3 = summ['O3']['G1/0.25/400/oracle/debiased']
    assert o3['n_available'] == 0 and o3['n_a_db_nonpositive'] == 0
    assert summ['O5']['G1/0.25/400/oracle']['causes'] == {'exception': 1, 'nonfinite': 1, 'nonpositive_gap': 0, 'H_not_pos_def': 0}
    assert summ['O4']['G1/0.25/400/oracle/debiased']['no_decision'] == 2
    assert rd['outcome']['G1/0.25/debiased'] in 'ABDC'


def test_T10b_more_boundaries():
    pop = R.population()
    # unequal headline outcomes: G1 0.25 A, G2 A -> B (N_det 3200 > 2917? no: make G2 A location fail)
    res = _synthetic(pop, lambda G, g, a: 800,
                     lambda G, g, N, a, p: 0.0 if (G, g) == ('G2', 'A') else 1.0, lambda *k: 1.0)
    rd = R.readings(res, pop)
    assert rd['headline_debiased'] == {'G1/0.25': 'A', 'G2/A': 'B'}
    assert rd['main_text_geometry'] == 'G2/A' and rd['supplement_geometry'] == 'G1/0.25' and rd['abstract_outcome_weakest'] == 'B'
    # G2 B boundary: location good only from 1208 = N*_cell upward -> holds
    res = _synthetic(pop, lambda G, g, a: 800,
                     lambda G, g, N, a, p: 1.0 if (G, g) != ('G2', 'B') or N >= 1208 else 0.0, lambda *k: 1.0)
    rd = R.readings(res, pop)
    assert rd['R-Q2']['G2/B/oracle/debiased']['N_loc'] == 1208 and rd['R-Q2']['G2/B/oracle/debiased']['holds']
