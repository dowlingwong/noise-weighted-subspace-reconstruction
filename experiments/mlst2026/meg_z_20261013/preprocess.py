"""D1 preprocessing of ds000117 raw MEG into per-subject derived HDF5 files (design §D.1-D.2, runbook §6.2).

Per subject, with the parameters of protocol['preprocess'] (dev defaults; frozen at D4):
 1. raw .fif (refuse SSS/MaxFiltered files: info['proc_history'] must be empty; refuse MaxShield/IAS recordings,
    which MNE cannot read without SSS: STOP item), magnetometers only, channels marked bad in *any* run's
    channels.tsv dropped (one channel set for all six runs; this lets run 3-6 *quality metadata* -- not signal --
    influence the channel set, which is accepted and stated);
 2. FIR band-pass l_freq-h_freq (zero phase), then decimation by ``decim`` (1100 Hz / 12 = 91.67 Hz since the
    review: decimation by 10 left the 45-55 Hz band empty and the temporal factor near-singular, F2);
 3. stimulus onsets = events.tsv trigger onsets + trigger_delay_s (34.5 ms projector delay).  The events.tsv
    stimulus onsets MUST agree with the STI101 trigger samples (same count, every lag <= 1.5 samples);
    otherwise the subject is refused (F8: guards against a double correction if a snapshot's events.tsv
    already contained the delay, or against missing events);
 4. channel subset: the C magnetometers with the largest RMS evoked field over the 0..T-1 sample window,
    from the reference runs (1-2) only (face trials, baseline-corrected);
 5. noise windows by the gap rule: windows of T samples lying entirely in [onset + after_onset_s,
    next onset - before_next_s] (last stimulus: end of data - end_margin_s), not intersecting a button-press
    exclusion span or a BAD annotation; stride T (non-overlapping) except in the reference pool (stride 1);
 6. run partition (no temporal leakage): reference (1-2) -> channel selection, templates, Sigma_ref moment
    statistics; training (3-4); calibration (5); evaluation (6);
 7. HDF5 per subject (windows float32; reference moments float64; templates; trials of every run for D8)
    with file SHA-256 and a layout-independent content SHA-256, and a window-budget report.

Commands (EXP root, PYTHONPATH=src):
    python -m experiments.mlst2026.meg_z_20261013.preprocess --bids-root RAW --subject sub-01 --out DERIVED
        [--protocol PATH] [--drop-selected MEG1234] [--data-manifest DATA_MANIFEST.json]
Confirmation subjects are refused unless the protocol is frozen (F7).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import common, estimators
from .common import vec


# ----------------------------------------------------------------------------------------------- BIDS text files
def read_tsv(path) -> list[dict]:
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def read_bads(channels_tsv) -> set[str]:
    return {r['name'] for r in read_tsv(channels_tsv) if str(r.get('status', 'good')).strip().lower() == 'bad'}


def read_events(events_tsv, cfg: dict) -> dict:
    """Stimulus and response events (trigger times, seconds from the first datapoint)."""
    face = set(cfg['stimulus_codes']['face'])
    scr = set(cfg['stimulus_codes']['scrambled'])
    resp = set(cfg['response_codes'])
    stim_on, stim_val, resp_on, other = [], [], [], []
    for r in read_tsv(events_tsv):
        try:
            v = int(float(r['value']))
        except (KeyError, ValueError):
            other.append(r.get('trial_type'))
            continue
        on = float(r['onset'])
        if v in face or v in scr:
            stim_on.append(on)
            stim_val.append(v)
        elif v in resp:
            resp_on.append(on)
        else:
            other.append(v)
    order = np.argsort(stim_on, kind='stable')
    stim_on = np.asarray(stim_on, float)[order]
    stim_val = np.asarray(stim_val, int)[order]
    return {'stim_onsets_trigger': stim_on, 'stim_values': stim_val,
            'is_face': np.isin(stim_val, sorted(face)), 'resp_onsets': np.sort(np.asarray(resp_on, float)),
            'other_events': len(other)}


# ----------------------------------------------------------------------------------------------- raw loading
class NotRawError(RuntimeError):
    pass


class TriggerCheckError(RuntimeError):
    pass


def refuse_unfrozen_confirmation(subjects, proto: dict, protocol_path=None) -> None:
    """F7 / delta D3: before the D4 freeze no confirmation subject is preprocessed, gated or analysed.  A
    confirmation subject is accepted only if the kit's own frozen ``protocol.json`` exists, no ``--protocol``
    file was given, and the loaded protocol is content-equal to it (the runner's test); a variant file that
    merely claims ``"frozen": true`` is refused."""
    from .protocol import is_frozen, load_protocol, PROTOCOL_FROZEN
    bad = sorted(set(subjects) & set(proto['subjects']['confirmation']))
    if not bad:
        return
    if protocol_path is not None:
        raise SystemExit(f'confirmation subjects {bad}: --protocol files are refused; only the frozen kit protocol.json counts')
    if not (PROTOCOL_FROZEN.is_file() and is_frozen(proto) and load_protocol(None) == proto):
        raise SystemExit(f'confirmation subjects {bad} are processed only under the frozen protocol (D5)')


def load_run(fif, events_tsv, channels_tsv, cfg: dict) -> dict:
    """One run: magnetometers (all of them; bads are reported, dropped later by the caller), filtered and
    decimated, plus events and a decimated BAD-annotation mask."""
    import mne
    try:
        raw = mne.io.read_raw_fif(str(fif), preload=False, verbose='error')
    except Exception as e:                                    # noqa: BLE001
        if 'maxshield' in str(e).lower() or 'active shielding' in str(e).lower():
            raise NotRawError(f'{fif}: MaxShield (IAS) recording; it needs SSS and cannot be used raw: STOP') from e
        raise
    if raw.info.get('proc_history'):
        raise NotRawError(f'{fif}: proc_history present (SSS/MaxFilter applied); raw data required')
    sfreq = float(raw.info['sfreq'])
    ev = read_events(events_tsv, cfg)
    stim_check = {'ok': False, 'reason': 'no STI101 channel'}
    if cfg.get('check_stim_channel', True) and 'STI101' in raw.ch_names:
        found = mne.find_events(raw, stim_channel='STI101', shortest_event=1, mask=cfg.get('stim_mask', 255),
                                mask_type='and', consecutive=True, verbose='error')
        codes = sorted(set(cfg['stimulus_codes']['face']) | set(cfg['stimulus_codes']['scrambled']))
        st = (found[np.isin(found[:, 2], codes), 0] - raw.first_samp) / sfreq
        lag = [float(np.min(np.abs(st - o))) if len(st) else float('inf') for o in ev['stim_onsets_trigger']]
        stim_check = {'n_tsv': int(len(ev['stim_onsets_trigger'])), 'n_stim_channel': int(len(st)),
                      'max_abs_lag_s': float(max(lag)) if lag else 0.0,
                      'ok': bool(len(st) == len(lag) and ((not lag) or max(lag) <= 1.5 / sfreq))}
    mags = [n for n, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'mag']
    bads = sorted((read_bads(channels_tsv) | set(raw.info['bads'])) & set(mags))
    raw.pick(mags)
    raw.load_data(verbose='error')
    badmask = np.isnan(raw.get_data(picks=[0], reject_by_annotation='NaN')[0])
    f = cfg['filter']
    raw.filter(cfg['l_freq'], cfg['h_freq'], picks='all', method=f.get('method', 'fir'), phase=f.get('phase', 'zero'),
               fir_design=f.get('fir_design', 'firwin'), n_jobs=1, verbose='error')
    decim = int(cfg['decim'])
    data = raw.get_data()[:, ::decim]
    n = data.shape[1]
    bm = np.zeros(n, bool)
    if badmask.any():
        idx = np.nonzero(badmask)[0] // decim
        bm[idx[idx < n]] = True
    return {'data': data, 'ch_names': mags, 'bads': bads, 'sfreq_raw': sfreq, 'sfreq': sfreq / decim,
            'n_raw': int(raw.n_times), 'events': ev, 'bad_mask': bm, 'stim_check': stim_check,
            'proc_history_empty': True}


# ----------------------------------------------------------------------------------------------- windows
def noise_window_starts(n: int, fs: float, onsets, T: int, rule: dict, resp_onsets=(), bad_mask=None,
                        stride: int | None = None) -> dict:
    """Start samples (decimated index, sample j at time j/fs) of noise windows by the gap rule.

    Window [s, s+T-1] is kept iff s/fs >= onset_i + after_onset_s and (s+T-1)/fs <= onset_{i+1} - before_next_s
    (for the last stimulus: (s+T-1)/fs <= (n-1)/fs - end_margin_s), it does not intersect any
    [r + exclusion[0], r + exclusion[1]] around a response r, and it contains no BAD sample.  Windows of one
    gap are tiled from its first admissible start with the given stride (default T: non-overlapping)."""
    stride = int(T if stride is None else stride)
    onsets = np.sort(np.asarray(onsets, float))
    resp = np.asarray(resp_onsets, float)
    ex = rule.get('response_exclusion_s')
    starts, gaps, pos, rpos = [], [], [], []
    dropped = {'response': 0, 'bad': 0}
    for i, on in enumerate(onsets):
        lo = on + rule['after_onset_s']
        hi = onsets[i + 1] - rule['before_next_s'] if i + 1 < len(onsets) else (n - 1) / fs - rule['end_margin_s']
        s0 = max(int(math.ceil(lo * fs - 1e-9)), 0)
        s_last = min(int(math.floor(hi * fs + 1e-9)) - (T - 1), n - T)
        slots = range(s0, s_last + 1, stride)
        for k, s in enumerate(slots):
            t0, t1 = s / fs, (s + T - 1) / fs
            if ex is not None and len(resp) and np.any((resp + ex[0] <= t1) & (resp + ex[1] >= t0)):
                dropped['response'] += 1
                continue
            if bad_mask is not None and bad_mask[s:s + T].any():
                dropped['bad'] += 1
                continue
            starts.append(s)
            gaps.append(i)
            pos.append(k)
            rpos.append(len(slots) - 1 - k)                      # tiling slot counted back from the next onset
            # (approximate within one window: slots are tiled from the preceding onset, so the last slot ends
            #  anywhere in the last T samples before next onset - before_next_s; delta review D8)
    return {'starts': np.asarray(starts, int), 'gap': np.asarray(gaps, int), 'position': np.asarray(pos, int),
            'rposition': np.asarray(rpos, int), 'dropped': dropped, 'n_gaps': int(len(onsets))}


def cut(data: np.ndarray, starts, T: int) -> np.ndarray:
    """(n_ch, n) continuous -> (n_win, n_ch, T) windows."""
    starts = np.asarray(starts, int)
    if len(starts) == 0:
        return np.zeros((0, data.shape[0], T))
    idx = starts[:, None] + np.arange(T)[None, :]
    return np.transpose(data[:, idx], (1, 0, 2))


def epochs(data: np.ndarray, fs: float, onsets, T: int, baseline_s, bad_mask=None):
    """Stimulus-locked windows of T samples from round(onset fs), baseline-corrected by the mean over
    [onset + baseline_s[0], onset + baseline_s[1]).  Returns (X (n, n_ch, T), kept boolean mask)."""
    starts = np.round(np.asarray(onsets, float) * fs).astype(int)
    b0, b1 = int(round(baseline_s[0] * fs)), int(round(baseline_s[1] * fs))
    n = data.shape[1]
    keep = (starts + min(b0, 0) >= 0) & (starts + T <= n)
    if bad_mask is not None:
        keep &= np.array([not bad_mask[max(s + min(b0, 0), 0):s + T].any() for s in starts], bool)
    X = cut(data, starts[keep], T)
    if b1 > b0:
        base = cut(data, starts[keep] + b0, b1 - b0).mean(-1, keepdims=True)
        X = X - base
    return X, keep


def select_channels(evoked: np.ndarray, C: int) -> dict:
    """Top-C channels by RMS over time of the evoked field (rows of ``evoked``); returned in channel order."""
    rms = np.sqrt(np.mean(evoked ** 2, axis=1))
    rank = np.argsort(-rms, kind='stable')
    if len(rank) < C:
        raise ValueError(f'only {len(rank)} good magnetometers for C = {C}')
    return {'selected': np.sort(rank[:C]), 'ranking': rank, 'rms': rms}


def partition(cfg: dict) -> dict:
    p = {k: [int(r) for r in v] for k, v in cfg['partition'].items()}
    allr = [r for v in p.values() for r in v]
    if len(allr) != len(set(allr)):
        raise ValueError(f'run partition not disjoint: {p}')
    if set(p) != {'reference', 'training', 'calibration', 'evaluation'}:
        raise ValueError(f'run partition must name reference/training/calibration/evaluation: {p}')
    return p


# ----------------------------------------------------------------------------------------------- HDF5 I/O
def write_h5(path, payload: dict, attrs: dict) -> dict:
    """Write nested dict payload (leaves: numpy arrays) deterministically; returns file and content SHA-256."""
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.h5')
    leaves = sorted(_flatten(payload).items())
    with h5py.File(tmp, 'w', libver='earliest') as f:
        for name, arr in leaves:
            f.create_dataset(name, data=np.asarray(arr), track_times=False)
        for k, v in sorted(attrs.items()):
            f.attrs[k] = json.dumps(v, sort_keys=True, default=common._default)
    content = content_sha256(leaves, attrs)
    tmp.replace(path)
    return {'file_sha256': common.sha_file(path), 'content_sha256': content, 'bytes': path.stat().st_size}


def _flatten(d: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in d.items():
        name = f'{prefix}/{k}' if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, name))
        else:
            out[name] = v
    return out


def content_sha256(leaves, attrs: dict) -> str:
    import hashlib
    h = hashlib.sha256()
    for name, arr in leaves:
        h.update(name.encode())
        h.update(common.sha_array(np.asarray(arr)).encode())
    h.update(json.dumps(attrs, sort_keys=True, default=common._default).encode())
    return h.hexdigest()


@dataclass
class SubjectData:
    subject: str
    C: int
    T: int
    sfreq: float
    channels: list
    windows: dict                      # partition -> (n, C, T) float64
    window_meta: dict                  # partition -> {'run', 'start', 'gap', 'position'}
    ref: estimators.MomentStats
    evoked: np.ndarray                 # (C, T) face evoked, reference runs
    trials: dict = field(default_factory=dict)   # run -> {'X': (n, C, T), 'face': (n,) bool, 'value': (n,)}
    attrs: dict = field(default_factory=dict)

    @property
    def d(self) -> int:
        return self.C * self.T

    def records(self, part: str) -> np.ndarray:
        return vec(self.windows[part])


_CACHE: dict = {}
META_KEYS = ('run', 'start', 'gap', 'position', 'rposition')


def load_subject(path, cache: bool = True) -> SubjectData:
    import h5py
    key = str(Path(path).resolve())
    if cache and key in _CACHE:
        return _CACHE[key]
    with h5py.File(path, 'r') as f:
        attrs = {k: json.loads(v) for k, v in f.attrs.items()}
        win, meta = {}, {}
        for part in ('training', 'calibration', 'evaluation'):
            g = f['windows'][part]
            win[part] = g['data'][()].astype(np.float64)
            meta[part] = {k: g[k][()] for k in META_KEYS if k in g}
        r = f['reference']
        ref = estimators.MomentStats(int(r['n'][()]), r['mean'][()], r['second_moment'][()], float(r['sum_q4'][()]),
                                     int(r['n_eff'][()]) if 'n_eff' in r else None)
        trials = {}
        if 'trials' in f:
            for run in f['trials']:
                g = f['trials'][run]
                trials[int(run.replace('run', ''))] = {'X': g['data'][()].astype(np.float64),
                                                       'face': g['face'][()].astype(bool), 'value': g['value'][()]}
        sd = SubjectData(attrs['subject'], int(attrs['C']), int(attrs['T']), float(attrs['sfreq']), attrs['channels'],
                         win, meta, ref, f['templates']['evoked_faces'][()], trials, attrs)
    if cache:
        _CACHE[key] = sd
    return sd


def subject_payload(windows: dict, window_meta: dict, ref: estimators.MomentStats, evoked: np.ndarray,
                    trials: dict, extra_templates: dict | None = None) -> dict:
    """The HDF5 layout shared by ``process_subject`` and the synthetic fixtures."""
    pay = {'windows': {}, 'reference': {'n': np.int64(ref.n), 'mean': ref.mean, 'second_moment': ref.S,
                                        'sum_q4': np.float64(ref.sum_q4), 'n_eff': np.int64(ref.n_eff or ref.n)},
           'templates': {'evoked_faces': np.asarray(evoked, float), **(extra_templates or {})}, 'trials': {}}
    for part, X in windows.items():
        m = window_meta[part]
        pay['windows'][part] = {'data': np.asarray(X, np.float32),
                                **{k: np.asarray(m[k], np.int64) for k in META_KEYS}}
    for run, t in trials.items():
        pay['trials'][f'run{int(run)}'] = {'data': np.asarray(t['X'], np.float32), 'face': np.asarray(t['face'], np.int8),
                                           'value': np.asarray(t['value'], np.int64)}
    return pay


def window_budget(counts: dict, n_ref: int, cfg: dict) -> dict:
    """Window counts against the requirements of the arm grid (protocol['budget_requirements'])."""
    req = cfg.get('budget_requirements', {})
    checks = {'training_ge_N_max': counts.get('training', 0) >= req.get('N_max', 0),
              'calibration_ge_m_max': counts.get('calibration', 0) >= req.get('m_max', 0),
              'evaluation_ge_n_eval': counts.get('evaluation', 0) >= req.get('n_eval', 0)}
    return {'counts': counts, 'reference_overlapping': int(n_ref), 'requirements': req, 'checks': checks,
            'sufficient': bool(all(checks.values()))}


def update_derived_manifest(out_dir, summaries: list) -> dict:
    out_dir = Path(out_dir)
    path = out_dir / 'DERIVED_MANIFEST.json'
    man = json.loads(path.read_text()) if path.is_file() else {'subjects': {}}
    for s in summaries:
        man['subjects'][s['subject']] = {k: s[k] for k in ('file', 'file_sha256', 'content_sha256', 'bytes', 'C', 'T', 'config_sha256')}
    common.atomic_json(path, man)
    return man


# ----------------------------------------------------------------------------------------------- subject pipeline
def process_subject(bids_root, subject: str, cfg: dict, out_dir, drop_selected=(), data_manifest: dict | None = None,
                    runs_loader=None) -> dict:
    """Preprocess one subject; writes OUT/<subject>.h5 and OUT/<subject>_summary.json.  ``runs_loader``
    (tests) replaces ``load_run`` and receives the same (fif, events.tsv, channels.tsv, cfg) arguments."""
    from .data_manifest import run_stem
    bids_root, out_dir = Path(bids_root), Path(out_dir)
    part = partition(cfg)
    loader = runs_loader or load_run
    C, T = int(cfg['C']), int(cfg['T'])
    rule = cfg['noise_windows']
    runs, sources = {}, {}
    for r in sorted({x for v in part.values() for x in v}):
        stem = bids_root / run_stem(subject, r)
        files = [Path(str(stem) + s) for s in ('_meg.fif', '_events.tsv', '_channels.tsv')]
        runs[r] = loader(*files, cfg)
        for p in files:
            rel = str(p.relative_to(bids_root))
            sources[rel] = {'bytes': p.stat().st_size if p.exists() else None}
            if data_manifest and rel in data_manifest.get('files', {}):
                sources[rel]['sha256_from_DATA_MANIFEST'] = data_manifest['files'][rel]['sha256']
                if sources[rel]['bytes'] != data_manifest['files'][rel]['size']:
                    raise ValueError(f'{rel}: size differs from DATA_MANIFEST')
    bad_checks = {r: runs[r]['stim_check'] for r in runs if not (runs[r]['stim_check'] or {}).get('ok')}
    if bad_checks and cfg.get('check_stim_channel', True):
        raise TriggerCheckError(f'{subject}: events.tsv stimulus onsets disagree with STI101 (or STI101 missing): '
                                f'{bad_checks}; possible double delay correction or missing events: STOP')
    names = runs[min(runs)]['ch_names']
    if any(runs[r]['ch_names'] != names for r in runs):
        raise ValueError('magnetometer lists differ between runs')
    bads = sorted(set().union(*[set(runs[r]['bads']) for r in runs]))
    good = [i for i, n in enumerate(names) if n not in bads]
    fs = runs[min(runs)]['sfreq']
    if any(abs(runs[r]['sfreq'] - fs) > 1e-9 for r in runs):
        raise ValueError('sampling rates differ between runs')
    delay = float(cfg['trigger_delay_s'])
    onsets = {r: runs[r]['events']['stim_onsets_trigger'] + delay for r in runs}
    resp = {r: runs[r]['events']['resp_onsets'] for r in runs}
    # --- channel selection and spatial-template data: reference runs ONLY
    sel_cond = cfg['selection'].get('conditions', 'faces')
    ev_list = []
    for r in part['reference']:
        mask = runs[r]['events']['is_face'] if sel_cond == 'faces' else np.ones(len(onsets[r]), bool)
        X, _ = epochs(runs[r]['data'][good], fs, onsets[r][mask], T, cfg['baseline_s'], runs[r]['bad_mask'])
        ev_list.append(X)
    ref_trials = np.concatenate(ev_list, 0)
    evoked_all = ref_trials.mean(0)                                  # (n_good, T)
    sel = select_channels(evoked_all, C)
    drop_selected = list(drop_selected)
    sel_names = [names[good[p]] for p in sel['selected']]
    unknown = [n for n in drop_selected if n not in sel_names]
    if unknown:
        raise ValueError(f'--drop-selected names channels that are not selected: {unknown}')
    chosen_pos = [p for p in sel['selected'] if names[good[p]] not in drop_selected]
    chosen = [good[p] for p in chosen_pos]
    C_eff = len(chosen)
    evoked = evoked_all[chosen_pos]
    # --- non-overlapping noise windows of the training / calibration / evaluation runs
    wins, meta, budget = {}, {}, {'per_run': {}}
    for pname in ('training', 'calibration', 'evaluation'):
        Xs, mm = [], {k: [] for k in META_KEYS}
        for r in part[pname]:
            w = noise_window_starts(runs[r]['data'].shape[1], fs, onsets[r], T, rule, resp[r], runs[r]['bad_mask'])
            Xs.append(cut(runs[r]['data'][chosen], w['starts'], T))
            for k, v in (('run', np.full(len(w['starts']), r)), ('start', w['starts']), ('gap', w['gap']),
                         ('position', w['position']), ('rposition', w['rposition'])):
                mm[k].append(v)
            budget['per_run'][r] = {'partition': pname, 'windows': int(len(w['starts'])), 'gaps': w['n_gaps'],
                                    'dropped_response': w['dropped']['response'], 'dropped_bad': w['dropped']['bad']}
        wins[pname] = np.concatenate(Xs, 0)
        meta[pname] = {k: np.concatenate(v) for k, v in mm.items()}
    # --- reference pool: overlapping windows (stride reference_stride) -> streamed moment statistics
    ref_w = {r: noise_window_starts(runs[r]['data'].shape[1], fs, onsets[r], T, rule, resp[r], runs[r]['bad_mask'],
                                    stride=int(cfg.get('reference_stride', 1))) for r in part['reference']}

    def chunks():
        for r in part['reference']:
            st = ref_w[r]['starts']
            for i in range(0, len(st), 4096):
                yield vec(cut(runs[r]['data'][chosen], st[i:i + 4096], T))
    ref = estimators.moment_stats_streamed(chunks)
    # F14: effective number of independent reference records = non-overlapping (stride-T) windows of the same gaps
    ref.n_eff = int(sum(len(noise_window_starts(runs[r]['data'].shape[1], fs, onsets[r], T, rule, resp[r],
                                                 runs[r]['bad_mask'])['starts']) for r in part['reference']))
    for r in part['reference']:
        budget['per_run'][r] = {'partition': 'reference', 'windows_overlapping': int(len(ref_w[r]['starts'])),
                                'gaps': ref_w[r]['n_gaps'], 'dropped_response': ref_w[r]['dropped']['response'],
                                'dropped_bad': ref_w[r]['dropped']['bad']}
    # --- stimulus-locked trials of every run (D8 secondary only; never noise, never predictions)
    trials = {}
    for r in sorted(runs):
        X, keep = epochs(runs[r]['data'][chosen], fs, onsets[r], T, cfg['baseline_s'], runs[r]['bad_mask'])
        trials[r] = {'X': X, 'face': runs[r]['events']['is_face'][keep], 'value': runs[r]['events']['stim_values'][keep]}
    budget.update(window_budget({p: len(wins[p]) for p in wins}, ref.n, cfg))
    budget['reference_effective'] = ref.n_eff
    top = sel['ranking'][:max(C + 8, 40)]
    attrs = {'subject': subject, 'C': C_eff, 'T': T, 'sfreq': fs, 'channels': [names[i] for i in chosen],
             'bads_dropped': bads, 'drop_selected': drop_selected, 'n_good_mags': len(good),
             'selection_ranking': [names[good[i]] for i in top],
             'selection_rms': {names[good[i]]: float(sel['rms'][i]) for i in top},
             'preprocess_config': cfg, 'config_sha256': common.sha_json(cfg), 'partition': part,
             'stim_checks': {str(r): runs[r]['stim_check'] for r in runs}, 'sources': sources,
             'n_trials_reference_selection': int(len(ref_trials)), 'created_by': 'meg_z_20261013.preprocess',
             'proc_history_empty': bool(all(runs[r].get('proc_history_empty') for r in runs))}
    pay = subject_payload(wins, meta, ref, evoked, trials)
    h = write_h5(out_dir / f'{subject}.h5', pay, attrs)
    summary = {'subject': subject, 'file': f'{subject}.h5', **h, 'C': C_eff, 'T': T, 'sfreq': fs, 'budget': budget,
               'channels': attrs['channels'], 'bads_dropped': bads, 'drop_selected': drop_selected,
               'stim_checks': attrs['stim_checks'], 'config_sha256': attrs['config_sha256'],
               'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    common.atomic_json(out_dir / f'{subject}_summary.json', summary)
    return summary


def main(argv=None):
    from .protocol import load_protocol
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids-root', required=True)
    ap.add_argument('--subject', required=True, action='append')
    ap.add_argument('--out', required=True)
    ap.add_argument('--protocol')
    ap.add_argument('--drop-selected', action='append', default=[])
    ap.add_argument('--data-manifest')
    a = ap.parse_args(argv)
    proto = load_protocol(a.protocol)
    refuse_unfrozen_confirmation(a.subject, proto, a.protocol)
    cfg = preprocess_config(proto)
    dm = json.loads(Path(a.data_manifest).read_text()) if a.data_manifest else None
    sums = []
    for s in a.subject:
        sums.append(process_subject(a.bids_root, s, cfg, a.out, a.drop_selected, dm))
        print(json.dumps({k: sums[-1][k] for k in ('subject', 'file_sha256', 'C', 'T', 'budget')}, indent=1))
    update_derived_manifest(a.out, sums)


def preprocess_config(proto: dict) -> dict:
    cfg = json.loads(json.dumps(proto['preprocess']))
    cfg['budget_requirements'] = proto['budget_requirements']
    return cfg


if __name__ == '__main__':
    main()
