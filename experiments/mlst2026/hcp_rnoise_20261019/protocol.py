"""Reviewed CONFIG identity, metadata split audit, and fail-closed real-data access."""
import hashlib
import json
from pathlib import Path
import numpy as np

PACKAGE = Path(__file__).resolve().parent
CONFIG_SHA = '8b90257f72f765aa006ee3d927329018284d4c04a69d983d5f8589a8a0bf35fa'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    p = PACKAGE / 'protocol.json'
    if sha(p) != CONFIG_SHA:
        raise RuntimeError('STOP protocol differs from reviewed MS CONFIG')
    return json.loads(p.read_text())


def split():
    p = PACKAGE / 'fixtures/SPLIT.json'
    if sha(p) != load()['inputs']['split_sha256']:
        raise RuntimeError('STOP frozen split identity differs')
    return json.loads(p.read_text())


def geometry():
    p = PACKAGE / 'fixtures/GEOMETRY.json'
    if sha(p) != load()['inputs']['geometry_sha256']:
        raise RuntimeError('STOP geometry identity differs')
    return json.loads(p.read_text())


def unit(date, subject=None):
    matches = [u for u in split()['units'] if u['measurement_date'] == date]
    if len(matches) != 1:
        raise PermissionError('unknown or excluded date')
    result = matches[0]
    if subject is not None and result['subject_id'] != subject:
        raise PermissionError('subject is not the frozen representative for this date')
    return result


def guard(date, subject=None, *, header_only=False):
    """Always check metadata before constructing or opening any recording path.

    This candidate allows dev headers only. A future author-adopted, independently
    reviewed kit must change its pinned protocol to enable real execution. A user
    supplied JSON, command-line flag, env var, or different path cannot unlock it.
    """
    u = unit(date, subject)
    cfg = load()
    if u['split'] == 'confirmation':
        if not (cfg['frozen'] is True and str(cfg['status']).startswith('FROZEN')
                and cfg.get('registration', {}).get('doi')
                and cfg['access']['confirmation_enabled'] is True):
            raise PermissionError('confirmation requires a frozen registered protocol; candidate refuses access')
        raise PermissionError('candidate has no confirmation execution authorization')
    if not header_only:
        raise PermissionError('H0–H3 candidate: real decoding disabled; author adoption and reviewed dev protocol required')
    return u


def split_check(inventory=None):
    s = split(); cfg = load()
    dates = [u['measurement_date'] for u in s['units']]
    if dates != sorted(set(dates)) or len(dates) != 72:
        raise RuntimeError('STOP unique chronological units required')
    draw = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(s['seed']))).choice(dates, 12, replace=False).tolist()
    if draw != s['draw_order_dev_dates'] or sorted(draw) != cfg['access']['dev_dates']:
        raise RuntimeError('STOP dev draw differs')
    if [d for d in dates if d not in draw] != cfg['access']['confirmation_pool_dates']:
        raise RuntimeError('STOP confirmation partition differs')
    for u in s['units']:
        if u['duration_seconds'] < 240 or u['split'] != ('dev' if u['measurement_date'] in draw else 'confirmation'):
            raise RuntimeError('STOP split unit rule differs')
    if inventory is not None:
        p = Path(inventory)
        if sha(p) != cfg['inputs']['inventory_sha256']:
            raise RuntimeError('STOP inventory identity differs')
        scans = json.loads(p.read_text())['scans']
        eligible = [v for v in scans if v['duration_seconds'] >= 240]
        expected = {d: min((v for v in eligible if v['measurement_date'] == d), key=lambda v: int(v['subject_id']))['subject_id']
                    for d in sorted({v['measurement_date'] for v in eligible})}
        if expected != {u['measurement_date']: u['subject_id'] for u in s['units']}:
            raise RuntimeError('STOP representative selection differs')
    return {'status': 'PASS', 'split_sha256': cfg['inputs']['split_sha256'], 'units': 72,
            'dev': 12, 'confirmation': 60, 'inventory_verified': inventory is not None, 'sample_access': False}
