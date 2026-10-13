"""Fail-closed stimulus timing preflight; no magnetometer loading or filtering.

The snapshot-specific +37 sample TSV convention is metadata, not an estimated
display latency. Physical onset remains raw STI101 plus the protocol delay.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import common


class TriggerCheckError(RuntimeError):
    pass


class NotRawError(RuntimeError):
    pass


def validate_binding(cfg, manifest):
    convention = cfg.get('timing_convention')
    if not isinstance(convention, dict):
        raise TriggerCheckError('explicit timing_convention required: STOP')
    if (convention.get('min_stimulus_plateau_samples') != 2 or
            convention.get('onset_sample_tolerance_s') != 0.00005):
        raise TriggerCheckError('timing plateau/rounding rule differs from reviewed contract: STOP')
    if convention.get('name') == 'synthetic-uncorrected-v1':
        if convention.get('bids_stimulus_sample_offset') != 0 or manifest is not None:
            raise TriggerCheckError('synthetic offset-0 convention cannot accept dataset manifest: STOP')
        return
    if convention.get('name') != 'ds000117-2.0.0-stimulus-plus37-v1':
        raise TriggerCheckError('unknown timing convention: STOP')
    if (convention.get('dataset') != 'ds000117' or convention.get('snapshot') != '2.0.0' or
            convention.get('snapshot_doi') != 'doi:10.18112/openneuro.ds000117.v2.0.0' or
            convention.get('physical_onset_source') != 'raw_stable_sti101'):
        raise TriggerCheckError('declared dataset/onset convention differs from reviewed contract: STOP')
    expected = {'dataset': 'ds000117', 'snapshot_tag': '2.0.0',
                'snapshot_doi': 'doi:10.18112/openneuro.ds000117.v2.0.0'}
    if not manifest or any(manifest.get(k) != v for k, v in expected.items()):
        raise TriggerCheckError('timing convention dataset/snapshot/DOI binding failed: STOP')
    if (manifest.get('missing') or manifest.get('second_source_mismatch') or
            common.sha_json(manifest) != convention.get('data_manifest_content_sha256')):
        raise TriggerCheckError('verified DATA_MANIFEST content binding failed: STOP')
    if (convention.get('bids_stimulus_sample_offset') != 37 or
            convention.get('sfreq_raw') != 1100 or cfg.get('stim_mask', 255) != 255 or
            sorted(set(cfg['stimulus_codes']['face']) | set(cfg['stimulus_codes']['scrambled'])) !=
            [5, 6, 7, 13, 14, 15, 17, 18, 19]):
        raise TriggerCheckError('snapshot timing parameters differ from reviewed convention: STOP')


def open_raw(fif):
    import mne
    try:
        raw = mne.io.read_raw_fif(str(fif), preload=False, verbose='error')
    except Exception as exc:
        if 'maxshield' in str(exc).lower() or 'active shielding' in str(exc).lower():
            raise NotRawError(f'{fif}: MaxShield/IAS recording: STOP') from exc
        raise
    if raw.info.get('proc_history'):
        raise NotRawError(f'{fif}: proc_history present (SSS/MaxFilter applied): STOP')
    return raw


def stable_stimuli(values, codes, minimum=2):
    """Retain >=2-sample low-byte plateaus; admit only F1 one-sample edge shapes."""
    if minimum != 2:
        raise TriggerCheckError('reviewed minimum stimulus plateau is exactly two samples: STOP')
    values = np.asarray(values, int)
    edges = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
    starts, lengths, levels = edges[:-1], np.diff(edges), values[edges[:-1]]
    kept, rejected = [], []
    for i, (start, length, code) in enumerate(zip(starts, lengths, levels)):
        if code not in codes:
            continue
        if length >= minimum:
            kept.append((int(start), int(code)))
            continue
        prev = (int(levels[i-1]), int(lengths[i-1])) if i else (None, 0)
        nxt = (int(levels[i+1]), int(lengths[i+1])) if i + 1 < len(levels) else (None, 0)
        rising = prev[0] == 0 and nxt[0] in codes and nxt[1] >= minimum and (code & nxt[0]) == code
        falling = nxt[0] == 0 and prev[0] in codes and prev[1] >= minimum and (code & prev[0]) == code
        check = {'relative_sample': int(start), 'code': int(code), 'length': int(length),
                 'previous_code': prev[0], 'previous_length': prev[1],
                 'next_code': nxt[0], 'next_length': nxt[1],
                 'classification': 'rising_edge' if rising else 'falling_edge' if falling else 'unclassified',
                 'admitted': bool(rising or falling)}
        rejected.append(check)
        if not check['admitted']:
            raise TriggerCheckError(f'unexplained short stimulus plateau {check}: STOP')
    return np.asarray(kept, dtype=int).reshape(-1, 2), rejected


def preflight_run(fif, events_tsv, cfg):
    raw = open_raw(fif)
    try:
        sfreq = float(raw.info['sfreq'])
        cv = cfg['timing_convention']
        if cv.get('sfreq_raw', sfreq) != sfreq:
            raise TriggerCheckError(f'{fif}: raw sampling rate differs from timing convention: STOP')
        if 'STI101' not in raw.ch_names:
            raise TriggerCheckError(f'{fif}: missing STI101: STOP')
        with Path(events_tsv).open(newline='') as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        codes = sorted(set(cfg['stimulus_codes']['face']) | set(cfg['stimulus_codes']['scrambled']))
        stim_on, stim_samples, stim_values, resp, other = [], [], [], [], 0
        for row in rows:
            try:
                value_number = float(row['value'])
                if not np.isfinite(value_number) or value_number != int(value_number):
                    raise ValueError('noninteger/nonfinite event code')
                value = int(value_number)
                onset = float(row['onset'])
                if not np.isfinite(onset):
                    raise ValueError('nonfinite onset')
                if value in codes:
                    sample_number = float(row['sample'])
                    if (not np.isfinite(sample_number) or sample_number != int(sample_number) or sample_number < 0):
                        raise ValueError('invalid sample')
                    sample = int(sample_number)
                    if abs(onset - sample / sfreq) > float(cv['onset_sample_tolerance_s']) + 1e-10:
                        raise ValueError('onset/sample rounding mismatch')
                    stim_on.append(onset); stim_samples.append(sample); stim_values.append(value)
                elif value in cfg['response_codes']:
                    resp.append(onset)
                else:
                    other += 1
            except (ValueError, KeyError, OverflowError) as exc:
                raise TriggerCheckError(f'{events_tsv}: invalid TSV event {row}: {exc}: STOP') from exc
        signal = raw.get_data(picks=['STI101'])[0]
        if not np.all(np.isfinite(signal)) or not np.all(signal == np.round(signal)):
            raise TriggerCheckError('nonfinite/noninteger STI101 values: STOP')
        low = signal.astype(np.int64) & int(cfg.get('stim_mask', 255))
        stable, rejected = stable_stimuli(low, codes, int(cv['min_stimulus_plateau_samples']))
        expected_codes, samples = np.asarray(stim_values, int), np.asarray(stim_samples, int)
        if len(samples) == 0 or not np.array_equal(expected_codes, stable[:, 1]):
            raise TriggerCheckError(f'{fif}: ordered stimulus code/count mismatch: TSV {len(samples)}, raw {len(stable)}: STOP')
        offset = int(cv['bids_stimulus_sample_offset'])
        if not np.all(samples - stable[:, 0] == offset) or np.any(np.diff(samples) <= 0):
            raise TriggerCheckError(f'{fif}: exact stimulus sample offset/order mismatch: STOP')
        trigger = stable[:, 0] / sfreq
        lag = np.asarray(stim_on) - trigger
        check = {'ok': True, 'n_tsv': len(samples), 'n_stim_channel': len(stable),
                 'max_abs_lag_s': float(np.max(np.abs(lag))), 'signed_lag_s_minmax': [float(lag.min()), float(lag.max())],
                 'sample_offset_unique': np.unique(samples - stable[:, 0]).tolist(),
                 'first_samp': int(raw.first_samp), 'sfreq_raw': sfreq, 'timing_convention': cv,
                 'ordered_codes_equal': True, 'rejected_edge_intermediates': rejected,
                 'stimulus_pairs': [{'code': int(code), 'tsv_onset_s': float(onset),
                                     'tsv_sample': int(sample), 'raw_relative_sample': int(relative)}
                                    for code, onset, sample, relative in
                                    zip(expected_codes, stim_on, samples, stable[:, 0])],
                 'physical_onset_rule': 'raw_stable_relative_sample/sfreq + trigger_delay_s'}
        events = {'stim_onsets_trigger': trigger, 'stim_values': expected_codes,
                  'stim_onsets_tsv': np.asarray(stim_on), 'stim_samples_tsv': samples,
                  'stim_samples_raw_relative': stable[:, 0],
                  'is_face': np.isin(expected_codes, cfg['stimulus_codes']['face']),
                  'resp_onsets': np.sort(np.asarray(resp, float)), 'other_events': other}
        return {'events': events, 'stim_check': check}
    finally:
        raw.close()
