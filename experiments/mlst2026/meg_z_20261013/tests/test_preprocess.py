"""Preprocessing: gap rule, run partition and leakage, raw (MNE RawArray) fixture end to end."""
import json

import numpy as np
import pytest

from experiments.mlst2026.meg_z_20261013 import common, fixture, preprocess as P
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol

RULE = {'after_onset_s': 1.2, 'before_next_s': 0.05, 'end_margin_s': 0.05, 'response_exclusion_s': [-0.05, 0.5]}


def _check_windows(w, n, fs, onsets, T, rule, resp=()):
    on = np.sort(onsets)
    for s, g in zip(w['starts'], w['gap']):
        t0, t1 = s / fs, (s + T - 1) / fs
        assert t0 >= on[g] + rule['after_onset_s'] - 1e-12
        hi = on[g + 1] - rule['before_next_s'] if g + 1 < len(on) else (n - 1) / fs - rule['end_margin_s']
        assert t1 <= hi + 1e-12
        assert not np.any(on[(on > on[g])] <= t1), 'window reaches into the next trial'
        for r in resp:
            ex = rule['response_exclusion_s']
            assert t1 < r + ex[0] or t0 > r + ex[1]


def test_gap_rule_nonoverlapping_and_reference_stride():
    fs, n, T = 100.0, 1000, 10
    onsets = np.array([1.0, 4.0, 7.0])
    resp = [2.3]
    w = P.noise_window_starts(n, fs, onsets, T, RULE, resp)
    _check_windows(w, n, fs, onsets, T, RULE, resp)
    st = np.sort(w['starts'])
    assert np.all(np.diff(st) >= T), 'training/calibration/evaluation windows must not overlap'
    assert w['dropped']['response'] >= 1
    ref = P.noise_window_starts(n, fs, onsets, T, RULE, resp, stride=1)
    _check_windows(ref, n, fs, onsets, T, RULE, resp)
    assert len(ref['starts']) > len(w['starts']) and np.any(np.diff(np.sort(ref['starts'])) < T)
    # explicit first window: onset 1.0 + 1.2 = 2.2 s -> sample 220, but the response span [2.25, 2.8] blocks it
    assert 220 not in set(w['starts']) and 280 not in set(w['starts'])
    # gap end: last window of gap 0 ends by 4.0 - 0.05 = 3.95 s
    g0 = w['starts'][w['gap'] == 0]
    assert (g0.max() + T - 1) / fs <= 3.95 + 1e-12
    bad = np.zeros(n, bool)
    bad[600:610] = True
    wb = P.noise_window_starts(n, fs, onsets, T, RULE, bad_mask=bad)
    assert not any(s <= 609 and s + T - 1 >= 600 for s in wb['starts']) and wb['dropped']['bad'] >= 1


def test_partition_must_be_disjoint():
    cfg = load_protocol()['preprocess']
    assert P.partition(cfg) == {'reference': [1, 2], 'training': [3, 4], 'calibration': [5], 'evaluation': [6]}
    bad = dict(cfg, partition={'reference': [1, 2], 'training': [2, 3], 'calibration': [5], 'evaluation': [6]})
    with pytest.raises(ValueError):
        P.partition(bad)


def raw_cfg():
    proto = load_protocol()
    cfg = P.preprocess_config(proto)
    cfg.update(C=4, T=6, decim=40, h_freq=10.0)
    cfg['budget_requirements'] = {'N_max': 50, 'm_max': 20, 'n_eval': 20}
    return cfg


@pytest.fixture(scope='module')
def bids(tmp_path_factory):
    root = tmp_path_factory.mktemp('bids')
    truth = fixture.make_bids_raw(root, 'sub-01', duration_s=50.0)
    return root, truth


@pytest.fixture(scope='module')
def processed(bids, tmp_path_factory):
    root, truth = bids
    out = tmp_path_factory.mktemp('derived')
    s = P.process_subject(root, 'sub-01', raw_cfg(), out)
    return out, s


def test_raw_fixture_end_to_end(bids, processed):
    root, truth = bids
    out, s = processed
    sd = P.load_subject(out / 'sub-01.h5', cache=False)
    cfg = raw_cfg()
    assert sd.C == 4 and sd.T == 6 and abs(sd.sfreq - 27.5) < 1e-12
    assert 'MEG0004' not in sd.channels and 'MEG0004' in s['bads_dropped']          # channels.tsv bad dropped
    assert all(c.startswith('MEG0') for c in sd.channels) and 'MEG9992' not in sd.channels   # magnetometers only
    assert all(v['ok'] for v in s['stim_checks'].values())                         # events.tsv agrees with STI101
    # trigger correction: the evoked peak (simulated at trigger + 34.5 ms + 100 ms) is at sample round(0.1 fs)
    gfp = np.sqrt((sd.evoked ** 2).mean(0))
    assert int(np.argmax(gfp)) == int(round(0.1 * sd.sfreq))
    # every window lies in its partition's runs and obeys the gap rule (corrected onsets)
    part = P.partition(cfg)
    for p in ('training', 'calibration', 'evaluation'):
        runs = sd.window_meta[p]['run']
        assert set(np.unique(runs)) <= set(part[p])
        for r in part[p]:
            st = sd.window_meta[p]['start'][runs == r]
            w = {'starts': st, 'gap': sd.window_meta[p]['gap'][runs == r]}
            n = int(np.ceil(50.0 * 1100 / 40))
            _check_windows(w, n, sd.sfreq, truth['onsets'][r] + 0.0345, sd.T, cfg['noise_windows'], truth['resp'][r])
    assert s['budget']['counts']['training'] > 0 and s['budget']['reference_overlapping'] > s['budget']['counts']['training'] / 2


def test_derived_file_is_deterministic(bids, processed, tmp_path):
    root, _ = bids
    _, s1 = processed
    s2 = P.process_subject(root, 'sub-01', raw_cfg(), tmp_path)
    assert s1['content_sha256'] == s2['content_sha256'] and s1['file_sha256'] == s2['file_sha256']


def test_partition_leakage(bids, processed, tmp_path):
    """Channel selection, templates and Sigma_ref statistics depend on runs 1-2 only; training, calibration
    and evaluation windows depend only on their own runs."""
    root, _ = bids
    _, base = processed
    sd0 = P.load_subject(processed[0] / 'sub-01.h5', cache=False)

    def perturbed(which):
        def loader(fif, ev, ch, cfg):
            r = P.load_run(fif, ev, ch, cfg)
            run = int(str(fif).split('run-')[1][:2])
            if run in which:
                g = np.random.default_rng(run)
                r['data'] = 7.0 * r['data'] + 1e-12 * g.normal(size=r['data'].shape)
            return r
        return loader

    P.process_subject(root, 'sub-01', raw_cfg(), tmp_path / 'a', runs_loader=perturbed({3, 4, 5, 6}))
    sa = P.load_subject(tmp_path / 'a' / 'sub-01.h5', cache=False)
    assert sa.channels == sd0.channels
    assert np.array_equal(sa.evoked, sd0.evoked)
    assert np.array_equal(sa.ref.S, sd0.ref.S) and np.array_equal(sa.ref.mean, sd0.ref.mean)
    for p in ('training', 'calibration', 'evaluation'):
        assert not np.array_equal(sa.windows[p], sd0.windows[p])
    P.process_subject(root, 'sub-01', raw_cfg(), tmp_path / 'b', runs_loader=perturbed({1}))
    sb = P.load_subject(tmp_path / 'b' / 'sub-01.h5', cache=False)
    assert not np.array_equal(sb.ref.S, sd0.ref.S)
    if sb.channels == sd0.channels:
        for p in ('training', 'calibration', 'evaluation'):
            assert np.array_equal(sb.windows[p], sd0.windows[p])
    P.process_subject(root, 'sub-01', raw_cfg(), tmp_path / 'c', runs_loader=perturbed({6}))
    sc = P.load_subject(tmp_path / 'c' / 'sub-01.h5', cache=False)
    assert np.array_equal(sc.windows['training'], sd0.windows['training'])
    assert np.array_equal(sc.windows['calibration'], sd0.windows['calibration'])
    assert not np.array_equal(sc.windows["evaluation"], sd0.windows["evaluation"])


def test_refuses_maxfiltered(bids, monkeypatch):
    import mne
    root, _ = bids
    real = mne.io.read_raw_fif

    def fake(*a, **k):
        raw = real(*a, **k)
        with raw.info._unlock():
            raw.info['proc_history'] = [{'max_info': {}}]
        return raw
    monkeypatch.setattr(mne.io, 'read_raw_fif', fake)
    from experiments.mlst2026.meg_z_20261013.data_manifest import run_stem
    stem = str(root / run_stem('sub-01', 1))
    with pytest.raises(P.NotRawError):
        P.load_run(stem + '_meg.fif', stem + '_events.tsv', stem + '_channels.tsv', raw_cfg())


def test_drop_selected_channel(bids, processed, tmp_path):
    root, _ = bids
    sd0 = P.load_subject(processed[0] / 'sub-01.h5', cache=False)
    s = P.process_subject(root, 'sub-01', raw_cfg(), tmp_path, drop_selected=[sd0.channels[1]])
    assert s['C'] == 3 and sd0.channels[1] not in s['channels']
    with pytest.raises(ValueError):
        P.process_subject(root, 'sub-01', raw_cfg(), tmp_path, drop_selected=['MEG0004'])


def test_window_meta_neff_and_raw_flag(processed):
    out, s = processed
    sd = P.load_subject(out / 'sub-01.h5', cache=False)
    for p in ('training', 'calibration', 'evaluation'):
        m = sd.window_meta[p]
        assert set(m) == {'run', 'start', 'gap', 'position', 'rposition'} and np.all(m['rposition'] >= 0)
    assert sd.attrs['proc_history_empty'] is True
    assert 0 < sd.ref.n_eff < sd.ref.n and s['budget']['reference_effective'] == sd.ref.n_eff     # F14


def test_rposition_counts_back_from_next_onset():
    w = P.noise_window_starts(1000, 100.0, [1.0, 4.0], 10, dict(RULE, response_exclusion_s=None))
    g0 = w['gap'] == 0
    assert np.array_equal(w['position'][g0] + w['rposition'][g0], np.full(g0.sum(), g0.sum() - 1))


def test_refuses_double_trigger_correction(tmp_path):
    """F8: an events.tsv that already contains the 34.5 ms delay disagrees with STI101 -> refused."""
    root = tmp_path / 'bids'
    fixture.make_bids_raw(root, 'sub-01', duration_s=30.0, events_shift_s=0.0345)
    with pytest.raises(P.TriggerCheckError):
        P.process_subject(root, 'sub-01', raw_cfg(), tmp_path / 'out')


def test_maxshield_is_a_stop_item(bids, monkeypatch):
    import mne
    root, _ = bids

    def fake(*a, **k):
        raise ValueError('This file contains raw Internal Active Shielding data. It may be distorted. ... allow_maxshield')
    monkeypatch.setattr(mne.io, 'read_raw_fif', fake)
    from experiments.mlst2026.meg_z_20261013.data_manifest import run_stem
    stem = str(root / run_stem('sub-01', 1))
    with pytest.raises(P.NotRawError, match='MaxShield'):
        P.load_run(stem + '_meg.fif', stem + '_events.tsv', stem + '_channels.tsv', raw_cfg())


def test_cli_refuses_confirmation_subject_before_freeze(tmp_path):
    with pytest.raises(SystemExit, match='frozen protocol'):
        P.main(['--bids-root', str(tmp_path), '--subject', 'sub-05', '--out', str(tmp_path / 'o')])
