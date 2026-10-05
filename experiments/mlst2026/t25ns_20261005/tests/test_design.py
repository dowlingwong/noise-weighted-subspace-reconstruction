import copy

import numpy as np
from numpy.testing import assert_allclose

from experiments.mlst2026.t25ns_20261005 import science as s
from experiments.mlst2026.t25ns_20261005.protocol import BASE, FIXTURE, BENCHMARK, T25_NAMESPACES, enumeration


def test_common_random_numbers_across_rho():
    c = FIXTURE
    out = {}
    for rho in c['rho']:
        g, sigma, L, Li, pool, x = s.make_data(c, c['seeds'][0], rho, c['theta'][0])
        out[rho] = (x @ Li, pool @ Li)
    a, b = out[c['rho'][0]], out[c['rho'][1]]
    assert_allclose(a[0], b[0], atol=1e-9)
    assert_allclose(a[1], b[1], atol=1e-9)
    # different raw data at different rho, and the calibration pool is shared across theta
    assert s.digest(s.make_data(c, c['seeds'][0], 0.25, 0.3)[5]) != s.digest(s.make_data(c, c['seeds'][0], 0.5, 0.3)[5])
    assert np.array_equal(s.make_data(c, 910001, 0.5, 0.3)[4], s.make_data(c, 910001, 0.5, 0.0)[4])


def test_seed_namespaces_are_disjoint_from_each_other_and_from_t25():
    mine = {BASE['namespace']: BASE['seeds'], FIXTURE['namespace']: FIXTURE['seeds'], BENCHMARK['namespace']: BENCHMARK['seeds']}
    allns = {**mine, **T25_NAMESPACES}
    assert len(allns) == len(mine) + len(T25_NAMESPACES)
    seeds = [x for v in allns.values() for x in v]
    assert len(seeds) == len(set(seeds))
    a = s.rng(BASE['namespace'], 1, 'calibration').normal(size=8)
    b = s.rng('paper1-t25-v1/confirm', 1, 'calibration').normal(size=8)
    assert not np.allclose(a, b)


def test_allocation():
    e = enumeration(BASE)
    assert e['paired_cells'] == 288 and e['total_charges'] == 9795 and e['unique_component_records'] == 331776
    assert enumeration(FIXTURE)['paired_cells'] == 12


def fake_records(c, risk):
    recs = []
    for seed in c['seeds']:
        r0 = np.random.default_rng(seed)
        for rho in c['rho']:
            for th in c['theta']:
                for n in c['N']:
                    methods = {}
                    for v in s.variants(c):
                        for cl in s.CLASSES:
                            methods[f'{v}__{cl}'] = {'clean': risk(v, cl, rho, n) + 1e-3 * r0.normal(),
                                                     'restart_risk_spread': 0.0,
                                                     'optimization': {'selected': 0, 'restarts': [{'status': 'stationary'}]}}
                        if v != 'oracle':
                            methods[v + '__domain'] = {'domain': 'out_of_domain', 'eta': 3.0, 'epsilon': 1.0}
                    recs.append({'seed': seed, 'rho': rho, 'theta': th, 'N': n, 'methods': methods})
    return recs


def floor_world(v, cl, rho, n):
    """Planted: Kronecker penalty grows with rho and with m (a floor); full penalty shrinks with m."""
    if v == 'oracle':
        return 10.0 + (cl == 'restricted') * -1.0
    arm, m = v.split('_')
    m = int(m)
    base = 10.0 + (cl == 'restricted') * -1.0
    if arm == 'kron':
        return base + 4 * rho + (0.1 if m == max(BASE['m']) else 0.0)
    return base + 5000.0 / m


def test_family_size_and_registered_prediction_rules():
    c = copy.deepcopy(BASE)
    red = s.reduce_records(c, fake_records(c, floor_world))
    assert red['family_size'] == 12 * 31 + 6 * 2 == 384
    assert red['predictions']['P1']['verdict'] == 'HOLDS'
    assert red['predictions']['P2']['verdict'] == 'HOLDS'
    assert red['domain_census'] == {'in_domain': 0, 'out_of_domain': 288 * 4, 'undefined_gap': 0}

    def no_floor(v, cl, rho, n):          # Kronecker penalty falls with m at high rho -> P1 FAILS
        x = floor_world(v, cl, rho, n)
        return x - (0.5 if v == f'kron_{max(BASE["m"])}' else 0.0)
    assert s.reduce_records(c, fake_records(c, no_floor))['predictions']['P1']['verdict'] == 'FAILS'

    def flat(v, cl, rho, n):              # no rho effect, no m effect -> undecided
        return 10.0
    red = s.reduce_records(c, fake_records(c, flat))
    assert red['predictions']['P1']['verdict'] == 'UNDECIDED'
    assert red['predictions']['P2']['verdict'] == 'UNDECIDED'


def test_unrestricted_only_contrasts_are_never_optimization_limited():
    ms = BASE['m']
    assert s.involved_metrics('metric/kron_1024/unrestricted', ms) == set()
    assert s.involved_metrics('metric_calibration/full/unrestricted/256-1024', ms) == set()
    assert s.involved_metrics('rho_change/kron_1024/unrestricted', ms) == set()
    assert s.involved_metrics('interaction/full_256', ms) == {'full_256', 'oracle'}
    assert s.involved_metrics('estimator/1024/restricted', ms) == {'full_1024', 'kron_1024'}
    values = s.contrast_values({f'{v}__{cl}': {'clean': 1.0} for v in s.variants(BASE) for cl in s.CLASSES}, ms)
    assert len(values) == 31 and all(s.involved_metrics(k, ms) is not None for k in values)


def test_extrapolated_floor_contrast_and_its_flags():
    ms = BASE['m']
    methods = {f'{v}__{cl}': {'clean': 0.0} for v in s.variants(BASE) for cl in s.CLASSES}
    for cl in s.CLASSES:
        methods[f'kron_256__{cl}']['clean'] = 3.0 + 400 / 256      # penalty F + c/m with F = 3, c = 400
        methods[f'kron_1024__{cl}']['clean'] = 3.0 + 400 / 1024
    v = s.contrast_values(methods, ms)
    assert abs(v['floor_extrapolated/kron/restricted'] - 3.0) < 1e-12
    assert s.involved_metrics('floor_extrapolated/kron/restricted', ms) == {'kron_256', 'kron_1024', 'oracle'}
    assert s.involved_metrics('floor_extrapolated/full/unrestricted', ms) == set()


def test_floor_wording_is_withheld_where_a_restricted_fit_is_flagged():
    c = copy.deepcopy(BASE)

    def world(v, cl, rho, n):          # P1 holds and the extrapolated floor is clearly positive
        return floor_world(v, cl, rho, n) + (2.0 if v.startswith('kron') else 0.0)
    recs = fake_records(c, world)
    target = (c['theta'][0], c['N'][0])
    for r in recs:
        if r['rho'] == max(c['rho']) and (r['theta'], r['N']) == target:
            r['methods']['kron_1024__restricted']['optimization']['restarts'][0]['status'] = 'iteration_limit'
    p1 = s.reduce_records(c, recs)['predictions']['P1']
    assert p1['verdict'] == 'HOLDS'
    flagged = [x for x in p1['locations'] if x['optimization_limited']]
    assert flagged and all(x['class'] == 'restricted' and not x['floor_wording_allowed'] for x in flagged)
    assert p1['counts']['floor_wording_allowed'] == 12 - len(flagged)


def test_population_limit_penalties_separate_shrinkage_from_separability():
    rows = s.population_penalties(FIXTURE)
    zero = [r for r in rows if r['rho'] == 0.0]
    assert zero and all(abs(r['penalty_kron_unshrunk_limit'][cl]) < 1e-9 for r in zero for cl in s.CLASSES)
    hi = [r for r in rows if r['rho'] == max(FIXTURE['rho'])]
    assert all(r['penalty_kron_unshrunk_limit']['unrestricted'] > 0 for r in hi)
    assert all(set(r) >= {'penalty_full_limit', 'penalty_kron_limit', 'oracle_risk'} for r in rows)
    # the oracle unrestricted population risk does not depend on rho (common whitened problem)
    o = {r['theta']: [] for r in rows}
    for r in rows:
        o[r['theta']].append(r['oracle_risk']['unrestricted'])
    assert all(max(v) - min(v) < 1e-9 for v in o.values())
