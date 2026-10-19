"""Signed AD2: baseline-relative amplitude, raw steps and raw nonfinite count."""
import numpy as np
from .access import config


def screen(data):
    x = np.asarray(data, float)
    cfg = config()['gates']['artifact']
    if x.ndim != 2 or x.shape[0] != 12 or x.shape[1] < 2:
        raise ValueError('12 native channels x time required')
    rows = []
    for channel in x:
        finite = np.isfinite(channel)
        median = float(np.median(channel)) if finite.all() else None
        valid_pairs = finite[1:] & finite[:-1]
        steps = np.abs(channel[1:][valid_pairs] - channel[:-1][valid_pairs])
        rows.append({'median_tesla': median,
                     'amplitude_count': int(np.count_nonzero(np.abs(channel - median) >= cfg['max_abs_tesla'])) if median is not None else None,
                     'nonfinite_count': int(np.count_nonzero(~finite)),
                     'step_count': int(np.count_nonzero(steps >= cfg['max_step_tesla'])),
                     'step_unavailable_pairs': int(np.count_nonzero(~valid_pairs)),
                     'unavailable_reason': None if median is not None else 'raw nonfinite: median/amplitude unavailable'})
    result = {'channels': dict(zip(config()['geometry']['selected_channels'], rows)),
              'amplitude_count': sum(r['amplitude_count'] for r in rows) if all(r['amplitude_count'] is not None for r in rows) else None,
              'finite_channel_amplitude_count': sum(r['amplitude_count'] or 0 for r in rows),
              'step_count': sum(r['step_count'] for r in rows),
              'nonfinite_count': sum(r['nonfinite_count'] for r in rows),
              'step_unavailable_pairs': sum(r['step_unavailable_pairs'] for r in rows),
              'amplitude_baseline': cfg['amplitude_baseline']}
    result['pass'] = result['nonfinite_count'] == 0 and result['amplitude_count'] == 0 and result['step_count'] == 0
    return result
