"""Timing contract, pulse-safety, manifest binding, and six-run fail-fast tests."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mlst2026.meg_z_20261013 import timing, preprocess as P, fixture, common
from experiments.mlst2026.meg_z_20261013.protocol import load_protocol


def cfg(offset=37):
    c = P.preprocess_config(load_protocol())
    if offset == 0:
        c['timing_convention'] = {'name': 'synthetic-uncorrected-v1',
                                 'bids_stimulus_sample_offset': 0,
                                 'min_stimulus_plateau_samples': 2,
                                 'onset_sample_tolerance_s': 0.00005}
    return c


def files(tmp_path, signal=None, offset=37, first_samp=12100, rows=None, include_stim=True):
    import mne
    if signal is None:
        signal = np.zeros(500)
        signal[50:70] = 13
        signal[200:220] = 19
    names = ['STI101'] if include_stim else ['MEG0001']
    types = ['stim'] if include_stim else ['mag']
    raw = mne.io.RawArray(signal[None, :], mne.create_info(names, 1100, types),
                          first_samp=first_samp, verbose='error')
    fif, tsv = tmp_path / 'fixture_raw.fif', tmp_path / 'events.tsv'
    raw.save(fif, overwrite=True, verbose='error')
    if rows is None:
        rows = [{'onset': f'{(s + offset)/1100:.4f}', 'sample': str(s + offset), 'value': str(v)}
                for s, v in [(50, 13), (200, 19)]]
        rows.insert(1, {'onset': '0.123456', 'sample': '136', 'value': '256'})
    with tsv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['onset', 'sample', 'value'], delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    return fif, tsv


@pytest.mark.parametrize('offset', [0, 37])
def test_explicit_timing_first_sample_delay_and_responses(tmp_path, offset):
    checked = timing.preflight_run(*files(tmp_path, offset=offset), cfg(offset))
    ev = checked['events']
    assert np.array_equal(ev['stim_samples_raw_relative'], [50, 200])
    assert np.array_equal(ev['stim_samples_tsv'], np.array([50, 200]) + offset)
    assert np.array_equal(ev['stim_onsets_trigger'], np.array([50, 200])/1100)
    physical = ev['stim_onsets_trigger'] + cfg(offset)['trigger_delay_s']
    assert np.allclose(physical, np.array([50, 200])/1100 + 0.0345, rtol=0, atol=1e-15)
    assert np.array_equal(ev['resp_onsets'], [0.123456])
    assert checked['stim_check']['first_samp'] == 12100


@pytest.mark.parametrize('offset', [0, 36, 38, 75])
def test_snapshot_wrong_or_double_offset_stops(tmp_path, offset):
    with pytest.raises(timing.TriggerCheckError, match='offset'):
        timing.preflight_run(*files(tmp_path, offset=offset), cfg())


@pytest.mark.parametrize('bad', [
    {'sample': ''}, {'sample': 'nan'}, {'sample': '87.5'}, {'sample': '-1'},
    {'onset': 'nan'}, {'onset': 'inf'}, {'onset': '0.0792'}, {'value': '13.5'},
])
def test_invalid_tsv_refused(tmp_path, bad):
    rows = [{'onset': f'{87/1100:.4f}', 'sample': '87', 'value': '13'},
            {'onset': f'{237/1100:.4f}', 'sample': '237', 'value': '19'}]
    rows[0].update(bad)
    with pytest.raises(timing.TriggerCheckError):
        timing.preflight_run(*files(tmp_path, rows=rows), cfg())


@pytest.mark.parametrize('mode', ['swapped_codes', 'reordered', 'duplicate', 'missing'])
def test_order_codes_and_bijection_refused(tmp_path, mode):
    rows = [{'onset': f'{87/1100:.4f}', 'sample': '87', 'value': '13'},
            {'onset': f'{237/1100:.4f}', 'sample': '237', 'value': '19'}]
    if mode == 'swapped_codes':
        rows[0]['value'], rows[1]['value'] = rows[1]['value'], rows[0]['value']
    elif mode == 'reordered':
        rows.reverse()
    elif mode == 'duplicate':
        rows.append(dict(rows[-1]))
    else:
        rows.pop()
    with pytest.raises(timing.TriggerCheckError):
        timing.preflight_run(*files(tmp_path, rows=rows), cfg())


def test_missing_sti101_stops(tmp_path):
    with pytest.raises(timing.TriggerCheckError, match='missing STI101'):
        timing.preflight_run(*files(tmp_path, include_stim=False), cfg())


def test_f1_known_edges_and_exactly_two_sample_pulse():
    signal = np.zeros(100, int)
    signal[10:20] = 13; signal[20] = 5
    signal[40] = 18; signal[41:51] = 19
    signal[70:72] = 5
    stable, rejected = timing.stable_stimuli(signal, [5, 13, 18, 19])
    assert np.array_equal(stable, [[10, 13], [41, 19], [70, 5]])
    assert [r['classification'] for r in rejected] == ['falling_edge', 'rising_edge']
    assert all(r['admitted'] and r['length'] == 1 for r in rejected)


@pytest.mark.parametrize('pattern', [
    [0, 5, 0],                         # isolated omitted stimulus
    [0, 18, 5, 19, 19, 0],            # chained intermediates
    [0, 13, 13, 5, 19, 19, 0],        # between stable stimuli
    [0, 7, 13, 13, 0],                # bit pattern not subset
    [5, 0, 0],                         # truncated at boundary
])
def test_f1_unexplained_short_stimulus_refused(pattern):
    with pytest.raises(timing.TriggerCheckError, match='short stimulus'):
        timing.stable_stimuli(pattern, [5, 7, 13, 18, 19])


def test_overlapping_response_bit_does_not_change_low_stimulus(tmp_path):
    signal = np.zeros(500)
    signal[50:70] = 13; signal[60:65] += 256
    signal[200:220] = 19
    checked = timing.preflight_run(*files(tmp_path, signal=signal), cfg())
    assert np.array_equal(checked['events']['stim_values'], [13, 19])


def test_isolated_pulse_absent_from_tsv_cannot_hide(tmp_path):
    signal = np.zeros(500)
    signal[50:70] = 13; signal[200:220] = 19; signal[300] = 5
    with pytest.raises(timing.TriggerCheckError, match='short stimulus'):
        timing.preflight_run(*files(tmp_path, signal=signal), cfg())


def test_manifest_binding_rejects_missing_mutated_or_wrong_snapshot():
    c = cfg()
    manifest = {'dataset': 'ds000117', 'snapshot_tag': '2.0.0',
                'snapshot_doi': 'doi:10.18112/openneuro.ds000117.v2.0.0',
                'missing': [], 'second_source_mismatch': []}
    c['timing_convention']['data_manifest_content_sha256'] = common.sha_json(manifest)
    timing.validate_binding(c, manifest)
    for bad in [None, dict(manifest, snapshot_tag='1.1.0'), dict(manifest, missing=['x']),
                dict(manifest, files={}), dict(manifest, second_source_mismatch=['x'])]:
        with pytest.raises(timing.TriggerCheckError):
            timing.validate_binding(c, bad)
    with pytest.raises(timing.TriggerCheckError):
        timing.validate_binding(cfg(0), manifest)


def test_six_run_preflight_fails_before_first_magnetometer_loader(tmp_path):
    from experiments.mlst2026.meg_z_20261013.data_manifest import run_stem
    root = tmp_path / 'bids'
    fixture.make_bids_raw(root, 'sub-01', duration_s=30)
    tsv = Path(str(root / run_stem('sub-01', 6)) + '_events.tsv')
    lines = tsv.read_text().splitlines()
    row = lines[1].split('\t'); row[0] = 'nan'; lines[1] = '\t'.join(row)
    tsv.write_text('\n'.join(lines) + '\n')
    calls = []
    def forbidden_loader(*args):
        calls.append(args)
        raise AssertionError('magnetometer loader ran before all trigger checks')
    with pytest.raises(timing.TriggerCheckError, match='TSV event'):
        P.process_subject(root, 'sub-01', cfg(0), tmp_path/'derived', runs_loader=forbidden_loader)
    assert calls == [] and not (tmp_path/'derived').exists()


def test_derived_manifest_carries_binding_and_replaces_retest_diagnostics(tmp_path):
    cv = cfg()['timing_convention']
    summary = {'subject': 'sub-01', 'file': 'sub-01.h5', 'file_sha256': 'original-file',
               'content_sha256': 'original-content', 'bytes': 100, 'C': 24, 'T': 20,
               'config_sha256': 'original-config', 'timing_convention': cv,
               'stim_checks': {str(r): {'ok': True, 'rejected_edge_intermediates': []}
                               for r in range(1, 7)}}
    P.update_derived_manifest(tmp_path, [summary])
    man = json.loads((tmp_path/'DERIVED_MANIFEST.json').read_text())
    entry = man['subjects']['sub-01']
    assert entry['timing_convention']['data_manifest_content_sha256'] == \
        '66b7839b24e97d41977271197cf912af91d53d9b06e9259360331deaf9e85ac7'
    assert entry['stim_checks'] == summary['stim_checks']
    revised = dict(summary, C=23, file_sha256='retest-file', content_sha256='retest-content',
                   config_sha256='retest-config',
                   stim_checks={str(r): {'ok': True, 'retest_marker': True} for r in range(1, 7)})
    P.update_derived_manifest(tmp_path, [revised])
    updated = json.loads((tmp_path/'DERIVED_MANIFEST.json').read_text())['subjects']['sub-01']
    assert updated['C'] == 23 and updated['file_sha256'] == 'retest-file'
    assert updated['stim_checks'] == revised['stim_checks']
    assert updated['timing_convention'] == cv
