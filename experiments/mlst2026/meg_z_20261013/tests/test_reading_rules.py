"""PZ1-PZ5 reading engine on fabricated subject-level inputs (geometries control / A / B; point-factor PZ2;
cell-averaged PZ4; isotropic-vs-reference PZ5); Bonferroni-t with df = n_subjects - 1."""
import numpy as np
from scipy.stats import t

from experiments.mlst2026.meg_z_20261013 import runner as R
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol

SUBS = ['s1', 's2', 's3', 's4']
NSTAR = {'A': 400.0, 'B': 250.0}
GRIDS = {'control': [100, 200, 400, 800, 1600], 'A': [100, 200, 400, 800, 1600], 'B': [62, 125, 250, 500, 1000]}


FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]


def pred(pz1=True, agree=True, agree_B=None):
    agree_B = agree if agree_B is None else agree_B
    geos = lambda: {'control': {'theta': 0.0, 'grid': GRIDS['control']},     # noqa: E731
                    'A': {'theta': 0.3, 'grid': GRIDS['A'], 'factors': FACTORS, 'N_star': NSTAR['A'], 'restart_agreement': agree,
                          'row': {'a_best': 0.05, 'B': -20.0, 'N_star_interval': [400.0, 1300.0]}},
                    'B': {'theta': 0.4, 'grid': GRIDS['B'], 'factors': FACTORS, 'N_star': NSTAR['B'], 'restart_agreement': agree_B,
                          'row': {'a_best': 0.08, 'B': -20.0, 'N_star_interval': [250.0, 800.0]}}}
    return {'subjects': {s: {'geometries': geos(), 'pz1': {'met': pz1}, 'rows': []} for s in SUBS}}


def sm_from(means: dict, jitter=0.01):
    """means[(arm, geometry)] = list over cells of D; per-subject R_U and R_F with arm-specific spreads."""
    sm = {}
    for (arm, g), vals in means.items():
        off = {'iso': 2.0}.get(arm, 0.0)
        jj = jitter * (1 + (sum(map(ord, arm + g)) % 5))
        for i, v in enumerate(vals):
            per = {}
            for j, s in enumerate(SUBS):
                d = v + jj * ((j * (j + 1)) % 4 - 1.5)
                ru = 1.0 + off + 0.01 * j * (1 + off) + 0.003 * (j % 2) * off
                per[s] = {'mean_D': d, 'mean_R_U': ru, 'mean_R_F': ru + d, 'resamples': 20, 'optimization_limited': 0,
                          'N': GRIDS[g][i]}
            sm[(arm, g, i)] = per
    return sm


def base_means(over=None):
    m = {('ref', 'control'): [-1, -0.5, -0.2, -0.1, -0.05], ('ref', 'A'): [-1, -0.5, 0.0, 0.5, 1.0],
         ('ref', 'B'): [-1, -0.5, 0.0, 0.5, 1.0], ('iso', 'control'): [0.5] * 5, ('iso', 'A'): [0.5] * 5, ('iso', 'B'): [0.5] * 5}
    for est in ('lw', 'kron'):
        for g in GRIDS:
            m[(f'{est}_lo', g)] = [x - 0.4 for x in m[('ref', g)]]
            m[(f'{est}_hi', g)] = [x - 0.1 for x in m[('ref', g)]]
    m.update(over or {})
    return m


def run(means, p=None, **kw):
    rr = load_protocol()['reading_rules']
    for k, v in kw.items():
        rr[k] = dict(rr[k], **v)
    p = p or pred()
    F = R.family_size(R.n_cells(p, SUBS), rr)
    sm = sm_from(means)
    return R.readings(R.cell_table(sm, F, 0.05), p, SUBS, rr, F, 0.05, sm), F


def test_interval_bonferroni_df():
    v = [10.0, 11.0, 12.0, 13.0]
    r = R.interval(v, 10, 0.05)
    half = t.ppf(1 - 0.05 / 20, 3) * np.std(v, ddof=1) / 2
    assert abs(r['half_width'] - half) < 1e-12 and r['decision'] == 'positive' and r['n'] == 4
    assert R.interval([1.0, 1.0, 1.0], 5, 0.05)['decision'] == 'undecided'
    assert R.interval([-3.0, -2.9, -3.1, -3.05], 5, 0.05)['decision'] == 'negative'


def test_family_size_is_cells_plus_six():
    rr = load_protocol()['reading_rules']
    assert R.family_size(R.n_cells(pred(), SUBS), rr) == 15 + 6
    rr['PZ4'] = dict(rr['PZ4'], aggregation='per_cell')
    assert R.family_size(R.n_cells(pred(), SUBS), rr) == 15 + 4 * 10 + 2


def test_all_met_and_point_factor_classes():
    rd, F = run(base_means())
    assert F == 21
    for k in ('PZ1', 'PZ2', 'PZ3', 'PZ4', 'PZ5'):
        assert rd[k]['reading'] == 'MET', (k, rd[k])
    pa = rd['PZ2']['per_geometry']['A']
    assert [c['class'] for c in pa['cells']] == ['below', 'below', 'inside', 'above', 'above']
    assert pa['pattern'] == '- - ? + +' and pa['B_dagger_interval_descriptive']['s1'] == [400.0, 1300.0]


def test_pz2_failures_and_restart_agreement():
    rd, _ = run(base_means({('ref', 'B'): [-1, 0.0, 0.0, 0.5, 1.0]}))
    assert rd['PZ2']['per_geometry']['B']['reading'] == 'NOT MET' and rd['PZ2']['reading'] == 'NOT MET'
    rd, _ = run(base_means(), p=pred(agree=False))
    assert rd['PZ2']['reading'] == 'NOT APPLICABLE'


def test_pz3_pz4_pz5_failures():
    rd, _ = run(base_means({('ref', 'control'): [-1, -0.5, 0.3, -0.1, -0.05]}))
    assert rd['PZ3']['reading'] == 'NOT MET'
    m = base_means()
    for g in GRIDS:
        m[('kron_hi', g)] = [x - 0.6 for x in m[('ref', g)]]
    rd, _ = run(m)
    assert rd['PZ4']['reading'] == 'NOT MET' and rd['PZ4']['estimators']['lw']['met'] and not rd['PZ4']['estimators']['kron']['met']
    sm_bad = base_means()
    rd, _ = run(sm_bad)
    assert rd['PZ5']['per_class']['U']['decision'] == 'positive'
    # isotropic arm no worse than the reference -> PZ5 NOT MET; the class contrast under iso is descriptive only
    sm = sm_from(base_means())
    for k, per in sm.items():
        if k[0] == 'iso':
            for j, v in enumerate(per.values()):
                v['mean_R_U'] = 1.0 + 0.01 * j - 0.002 * (j % 2)
    rr = load_protocol()['reading_rules']
    rdd = R.readings(R.cell_table(sm, 21, 0.05), pred(), SUBS, rr, 21, 0.05, sm)
    assert rdd['PZ5']['reading'] == 'NOT MET' and rdd['descriptive_iso_class_contrast']


def test_pz1_from_predictions():
    rd, _ = run(base_means(), p=pred(pz1=False))
    assert rd['PZ1']['reading'] == 'NOT MET'


def test_pz2_overall_requires_both_geometries():
    """Delta D2: one geometry not applicable -> NOT MET with the reason; none applicable -> NOT APPLICABLE."""
    rd, _ = run(base_means(), p=pred(agree=True, agree_B=False))
    assert rd['PZ2']['per_geometry']['A']['reading'] == 'MET'
    assert rd['PZ2']['per_geometry']['B']['reading'] == 'NOT APPLICABLE'
    assert rd['PZ2']['reading'] == 'NOT MET' and 'not applicable' in rd['PZ2']['reason']


def test_pz2_classification_with_non_integer_nstar():
    """Delta D1: 12 subjects with N* = 430 +- 3 (non-integer): cells are classified by their grid factor, so the
    1/2 and 2 cells stay 'below' / 'above' although round(f N*) differs from f N* by up to 0.5."""
    from experiments.mlst2026.meg_z_20261013.population import round_half_up
    subs = [f's{i}' for i in range(12)]
    g = np.random.default_rng(1)
    nstar = {s: 430 + g.uniform(-3, 3) for s in subs}
    grid = {s: [round_half_up(f * nstar[s]) for f in FACTORS] for s in subs}
    assert any(grid[s][1] > nstar[s] / 2 for s in subs) and any(grid[s][3] < 2 * nstar[s] for s in subs)
    geo = lambda s: {'control': {'theta': 0.0, 'grid': GRIDS['control']},                      # noqa: E731
                     **{lab: {'theta': 0.3, 'grid': grid[s], 'factors': FACTORS, 'N_star': nstar[s], 'restart_agreement': True,
                              'row': {'a_best': 0.05, 'B': -20.0, 'N_star_interval': [nstar[s], 3 * nstar[s]]}} for lab in ('A', 'B')}}
    p = {'subjects': {s: {'geometries': geo(s), 'pz1': {'met': True}, 'rows': []} for s in subs}}
    means = base_means()
    sm = {}
    for (arm, gg), vals in means.items():
        for i, v in enumerate(vals):
            sm[(arm, gg, i)] = {s: {'mean_D': v + 0.01 * ((j * 7) % 5 - 2), 'mean_R_U': 1.0 + 0.01 * j + (2.0 + 0.002 * j if arm == 'iso' else 0),
                                    'mean_R_F': 1.0 + v, 'resamples': 20, 'optimization_limited': 0,
                                    'N': (GRIDS['control'][i] if gg == 'control' else grid[s][i])} for j, s in enumerate(subs)}
    rr = load_protocol()['reading_rules']
    F = R.family_size(R.n_cells(p, subs), rr)
    rd = R.readings(R.cell_table(sm, F, 0.05), p, subs, rr, F, 0.05, sm)
    for lab in ('A', 'B'):
        assert [c['class'] for c in rd['PZ2']['per_geometry'][lab]['cells']] == ['below', 'below', 'inside', 'above', 'above']
    assert rd['PZ2']['reading'] == 'MET'
