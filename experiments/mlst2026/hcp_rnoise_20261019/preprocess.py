"""BTI metadata wrapper and array preprocessing. No import-time recording access."""
from pathlib import Path
import hashlib
import numpy as np
import mne
from .protocol import load, geometry, guard
from .windows import plan, cut


class Excluded(ValueError):
    """A fixed qualification rule failed; never repair or reselect."""


def selected_geometry(info=None):
    cfg = load()['geometry']; rows = geometry()['channels']
    ordered = sorted(rows, key=lambda r: (np.linalg.norm(np.array(r['loc'][:3])-cfg['target_device_m']),r['name']))
    names = sorted(r['name'] for r in ordered[:cfg['C']])
    if names != cfg['selected_channels']:
        raise RuntimeError('STOP pinned geometric selection differs')
    selected = [{r['name']:r for r in rows}[n] for n in names]
    if info is not None:
        actual = {c['ch_name']:c for c in info['chs']}
        for row in selected:
            if row['name'] not in actual or row['name'] in info.get('bads', []):
                raise Excluded('missing or bad fixed channel')
            ch = actual[row['name']]
            if (int(ch['coord_frame']) != cfg['coordinate_frame']
                    or not np.allclose(ch['loc'], row['loc'], atol=cfg['location_match_atol_m'], rtol=0)
                    or int(ch['coil_type']) != row['coil_type']):
                raise Excluded('fixed geometry differs; no reselection')
    return names, selected


def open_header(date, subject=None, reader=None):
    """Dev header only, guard before any recording path operation; preload=False."""
    u = guard(date, subject, header_only=True)
    cfg = load(); directory = Path(cfg['access']['data_root']) / u['subject_id']
    reader = mne.io.read_raw_bti if reader is None else reader
    opts = cfg['preprocessing']['reader'].copy()
    raw = reader(str(directory/'c,rfDC'), config_fname=str(directory/'config'), verbose='error', **opts)
    try:
        if raw.preload:
            raise Excluded('reader unexpectedly decoded samples')
        if raw.n_times != u['n_times'] or not np.isclose(raw.info['sfreq'], u['sfreq_hz'], rtol=0, atol=1e-9):
            raise Excluded('header differs from frozen inventory')
        selected_geometry(raw.info)
        if raw.info.get('proc_history') or raw.info.get('projs'):
            raise Excluded('processed input or projections are outside the candidate')
    except Exception:
        raw.close()
        raise
    return raw


def read_scan(date, subject=None):
    guard(date, subject, header_only=False)  # Candidate ALWAYS refuses before open_header.
    raise PermissionError('candidate has no real-data execution path')


def artifact_screen(data):
    a = np.asarray(data, float)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError('native channel x time array required')
    cfg = load()['gates']['artifact']
    finite = np.isfinite(a)
    result = {'nonfinite_count': int((~finite).sum()),
              'amplitude_count': int((np.abs(a) >= cfg['max_abs_tesla']).sum()),
              'step_count': int((np.abs(np.diff(a, axis=1)) >= cfg['max_step_tesla']).sum())}
    result['pass'] = not any(result.values())
    return result


def kernel():
    p = load()['preprocessing']; f = p['filter']
    h = mne.filter.create_filter(None, p['native_sfreq_hz'], **f, verbose='error')
    digest = hashlib.sha256(np.asarray(h, dtype='<f8').tobytes()).hexdigest()
    if digest != p['FIR_sha256_f64le'] or len(h) != p['FIR_taps']:
        raise RuntimeError('STOP FIR kernel differs from H1')
    half_support = (len(h)-1)/(2*p['native_sfreq_hz'])
    if not (half_support < p['edge_seconds'] and 2*half_support < load()['windows']['guard_seconds']):
        raise RuntimeError('STOP insufficient finite-filter margins')
    return h


def preprocess_array(data, sfreq):
    """Pure fabricated-array-tested kernel, same FIR as raw.filter then phase0 q20."""
    cfg = load(); p = cfg['preprocessing']; x = np.asarray(data,float)
    if x.ndim != 2 or x.shape[0] != cfg['geometry']['C'] or not np.isclose(sfreq,p['native_sfreq_hz'],rtol=0,atol=1e-9):
        raise Excluded('native array dimension/rate differs')
    artifact = artifact_screen(x)
    if not artifact['pass']:
        raise Excluded(f'artifact screen failed: {artifact}')
    kernel()
    y = mne.filter.filter_data(x, sfreq, **p['filter'], verbose='error')[:,::p['decimation_factor']]
    wp = plan(x.shape[1], sfreq)
    return cut(y, wp), {'artifact': artifact, 'plan': wp, 'n_decimated': y.shape[1]}
