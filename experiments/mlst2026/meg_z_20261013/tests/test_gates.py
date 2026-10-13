"""Gates after the review: G1 (raw + spatial condition + full condition, automatic drop-one retest), G2 (band +
Marchenko-Pastur spread), G3 (effect size on positional means; significance on gap means, not gating), the
lag-one dependence statistic, G4, G5, E1 with the budget rule; confirmation guard of the CLI."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.mlst2026.meg_z_20261013 import estimators as E, fixture, gates as G, preprocess as Pp
from experiments.mlst2026.meg_z_20261013.preprocess import load_subject
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol

G1CFG = {'max_condition_spatial': 1e6, 'max_condition_full': 1e10}


def test_g1_three_checks_and_noisiest():
    C, T = 2, 3
    St = np.diag([1.0, 1e-3, 1e-6])                      # temporal factor ill-conditioned (filter-like): allowed
    Sc = np.eye(2)
    r = G.g1_spd(np.kron(St, Sc), C, T, G1CFG, True)
    assert r['pass'] and r['condition_temporal'] > 1e5 and r['condition_spatial'] < 2
    r = G.g1_spd(np.kron(np.eye(3), np.diag([1.0, 1e-7])), C, T, G1CFG, True)
    assert not r['pass'] and not r['checks']['spatial_condition']
    r = G.g1_spd(np.kron(np.diag([1.0, 1e-6, 1e-11]), Sc), C, T, G1CFG, True)
    assert not r['checks']['full_condition']
    assert not G.g1_spd(np.eye(6), C, T, G1CFG, False)['pass']          # processing history not empty
    cov = np.diag([1.0, 5.0, 1.0, 5.0, 1.0, 5.0])                       # index c + C t: channel 1 is noisy
    assert G.noisiest_channel(cov, C, T, ['a', 'b'])['channel'] == 'b'


def test_g2_band_and_spread():
    C, T = 4, 6
    cov, _, _ = fixture.fixture_covariance(C, T, 'kron')
    X = np.random.default_rng(0).normal(size=(4000, C * T)) @ np.linalg.cholesky(cov).T
    r = G.g2_stationarity(cov, X, [0.8, 1.25], 2.0)
    assert r['pass'] and abs(r['mean_whitened_eigenvalue'] - 1) < 0.03 and r['max_over_mp_upper'] < 1.2
    assert not G.g2_stationarity(cov, 1.3 * X, [0.8, 1.25], 2.0)['pass']
    Y = X.copy()                                                         # one direction 4x too strong: mean in band, spread not
    w, q = np.linalg.eigh(cov)
    Y = Y + 1.0 * (Y @ (np.linalg.inv(np.linalg.cholesky(cov)).T @ np.eye(C * T)[:, :1])) @ (np.linalg.cholesky(cov)[:, :1].T)
    r = G.g2_stationarity(cov, Y, [0.8, 1.25], 2.0)
    assert r['checks']['mean_in_band'] and not r['checks']['spread']


def test_g3_effect_size_and_gap_means(tmp_path):
    proto = load_protocol()
    fixture.make_derived_subject(tmp_path, 'sub-91', kind='kron', seed=1)
    fixture.make_derived_subject(tmp_path, 'sub-93', kind='kron', seed=3, locked_mean=0.6)
    gc = dict(proto['gates']['G3'])
    sd = load_subject(tmp_path / 'sub-91.h5', cache=False)
    W = np.linalg.inv(E.ledoit_wolf_from_stats(sd.ref)[0])
    ok = G.g3_locked_mean(sd, W, gc, 40.0)
    assert ok['pass'] and ok['max_fraction_of_trK'] < 0.01 and ok['positions']
    assert all(t['unit'] == 'gap means' for t in ok['significance_on_gap_means_reported_not_gating'])
    sdm = load_subject(tmp_path / 'sub-93.h5', cache=False)
    bad = G.g3_locked_mean(sdm, W, gc, 40.0)
    assert not bad['pass'] and bad['max_fraction_of_trK'] > 0.01
    # debiasing: under H0 the debiased energy is centred at 0
    X = np.random.default_rng(4).normal(size=(200, 24))
    e = [G.positional_energy(np.random.default_rng(i).normal(size=(200, 24)), np.eye(24))['energy_debiased'] for i in range(200)]
    assert abs(np.mean(e)) < 0.02 and G.positional_energy(X, np.eye(24))['energy_raw'] > 0.05


def test_lag_one_dependence():
    rng = np.random.default_rng(0)
    C, T, n = 2, 3, 2000
    z = rng.normal(size=(n, C * T))
    z[1:] = 0.6 * z[:-1] + 0.8 * z[1:]                                   # AR(1) across consecutive windows
    sd = SimpleNamespace(T=T, records=lambda part: z,
                         window_meta={'training': {'run': np.zeros(n, int), 'gap': np.arange(n) // 10, 'start': np.arange(n) * T}})
    r = G.lag_one_dependence(sd, np.eye(C * T))
    assert r['pairs'] == n - n // 10 and abs(r['trace_ratio'] - 0.48) < 0.05      # MA(1): 0.6 x 0.8 / (0.36 + 0.64)


def test_subject_gates_and_e1_with_budget_rule(tmp_path):
    proto = load_protocol()
    proto['signal'] = dict(proto['signal'], trace_K_target=40.0)
    fixture.make_derived_subject(tmp_path, 'sub-91', kind='kron', seed=1)
    fixture.make_derived_subject(tmp_path, 'sub-92', kind='nonkron', seed=2)
    fixture.make_derived_subject(tmp_path, 'sub-94', kind='kron', seed=4, heavy_tails=True)
    rk = G.subject_gates(load_subject(tmp_path / 'sub-91.h5', cache=False), proto)
    rn = G.subject_gates(load_subject(tmp_path / 'sub-92.h5', cache=False), proto)
    rt = G.subject_gates(load_subject(tmp_path / 'sub-94.h5', cache=False), proto)
    assert rk['failed'] == [] and rk['reference_kind'] == 'ledoit_wolf_full'
    assert rk['G4']['kron_vs_lw_whitened']['operator'] < rn['G4']['kron_vs_lw_whitened']['operator']
    assert abs(rk['G5']['b2_ratio'] - 1) < 0.05 and rt['G5']['b2_ratio'] > 1.1
    conf = proto['subjects']['confirmation']
    g = {s: {'excluded': i < 1, 'failed': ['G3'] if i < 1 else []} for i, s in enumerate(conf)}
    e = G.exclusion(g, proto, 'confirmation', extra_excluded={conf[1]: ['budget']})
    assert len(e['included']) == 10 and not e['stop'] and e['reasons'][conf[1]] == ['budget']
    e = G.exclusion(g, proto, 'confirmation', extra_excluded={conf[1]: ['budget'], conf[2]: ['no theta_s root']})
    assert len(e['included']) == 9 and e['stop']


def _raw_cfg():
    cfg = Pp.preprocess_config(load_protocol())
    cfg.update(C=4, T=6, decim=40, h_freq=10.0)
    cfg['timing_convention'] = {'name': 'synthetic-uncorrected-v1', 'bids_stimulus_sample_offset': 0,
                                'min_stimulus_plateau_samples': 2, 'onset_sample_tolerance_s': 0.00005}
    return cfg


def test_automatic_g1_retest(tmp_path):
    """A very noisy selected channel breaks the spatial condition; the gates CLI path drops it, re-preprocesses
    once and re-gates before E1 (F13)."""
    root = tmp_path / 'bids'
    fixture.make_bids_raw(root, 'sub-01', duration_s=50.0, noisy_channel='MEG0002')
    der = tmp_path / 'derived'
    cfg = _raw_cfg()
    s = Pp.process_subject(root, 'sub-01', cfg, der)
    Pp.update_derived_manifest(der, [s])
    assert 'MEG0002' in s['channels']
    proto = load_protocol()
    proto['preprocess'] = {k: v for k, v in cfg.items() if k != 'budget_requirements'}
    proto['gates'] = json.loads(json.dumps(proto['gates']))
    proto['gates']['G1']['max_condition_spatial'] = 50.0
    proto['signal'] = dict(proto['signal'], trace_K_target=40.0)
    r = G.gate_subject_with_retest(der, 'sub-01', proto, root)
    assert r['G1_retest']['status'] == 'done' and r['G1_retest']['dropped'] == 'MEG0002'
    assert not r['G1_retest']['first_attempt']['G1']['pass'] and r['G1']['pass']
    assert 'MEG0002' not in r['channels'] and r['C'] == 3
    man = json.loads((der / 'DERIVED_MANIFEST.json').read_text())
    assert man['subjects']['sub-01']['C'] == 3
    r2 = G.gate_subject_with_retest(der, 'sub-01', proto, root)                # never a second retest
    assert r2['G1_retest'] is None


def test_cli_refuses_confirmation_subject_before_freeze(tmp_path):
    with pytest.raises(SystemExit, match='frozen protocol'):
        G.main(['--derived', str(tmp_path), '--subjects', 'sub-01,sub-07', '--out', str(tmp_path / 'g.json')])
